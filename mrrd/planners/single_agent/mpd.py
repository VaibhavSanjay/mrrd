"""
Modified from https://github.com/jacarvalho/mpd-public
Modified by Vaibhav Sanjay (2026).
Originally authored by Yorai Shaoul (2024).

MIT License

Copyright (c) 2026 Vaibhav Sanjay
Copyright (c) 2024 Yorai Shaoul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import pickle
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from typing import Tuple, List
import math
import time

from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory, CostConstraint, CostMaxVelocity, CostCBF, CostProximityApproach
from mrrd.models import TemporalUnet, UNET_DIM_MULTS
from mrrd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mrrd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mrrd.trainer import get_dataset, get_model
from mrrd.utils.loading import load_params_from_yaml
from torch_robotics import environments
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.robots import *
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints, compute_human_cost, compute_max_jerk_metric
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from mrrd.planners.single_agent.common import PlannerOutput
from mrrd.planners.single_agent.single_agent_planner_base import SingleAgentPlanner
from mrrd.common.experiences import PathExperience, PathBatchExperience
from mrrd.common.constraints import MultiPointConstraint
from mrrd.common.pretty_print import *
from mrrd.common.trajectory_utils import smooth_trajs
from mrrd.humans import HumanTrajectories

import numpy as np
from mrrd.models import GoalModel, ContextModel

class MPD(SingleAgentPlanner):
    """
    A class that allows repeated calls to the same model with different inputs.
    This class keeps track of constraints and feeds them to the model only when needed.
    """

    def __init__(self,
                 model_id: str,
                 planner_alg: str,
                 start_state_pos: torch.tensor,
                 goal_state_pos: torch.tensor,
                 human_trajectories: HumanTrajectories,
                 use_guide_on_extra_objects_only: bool,
                 start_guide_steps_fraction: float,
                 n_guide_steps: int,
                 n_diffusion_steps_without_noise: int,
                 weight_grad_cost_collision: float,
                 weight_grad_cost_smoothness: float,
                 weight_grad_cost_constraints: float,
                 weight_grad_cost_soft_constraints: float,
                 weight_grad_cost_cbf: float,
                 weight_grad_cost_proximity_approach: float,
                 factor_num_interpolated_points_for_collision: float,
                 trajectory_duration: float,
                 device: str,
                 debug: bool,
                 seed: int,
                 results_dir: str,
                 trained_models_dir: str,
                 n_samples: int,
                 n_local_inference_noising_steps: int,
                 n_local_inference_denoising_steps: int,
                 env_id: str,
                 **kwargs
                 ):
        super().__init__()
        # The constraints are stored here. This is a list of ConstraintCost.
        self.constraints = []
        self.weight_grad_cost_collision = weight_grad_cost_collision if env_id != 'EnvEmptyNoWait2D' else 0.0
        self.weight_grad_cost_constraints = weight_grad_cost_constraints
        self.weight_grad_cost_soft_constraints = weight_grad_cost_soft_constraints
        self.weight_grad_cost_cbf = weight_grad_cost_cbf
        self.weight_grad_cost_proximity_approach = weight_grad_cost_proximity_approach

        ####################################
        fix_random_seed(seed)

        device = get_torch_device(device)
        tensor_args = {'device': device, 'dtype': torch.float32}
        ####################################
        print(f'####################################')
        print(f'Initializing Planner with Model -- {model_id}')
        print(f'Algorithm -- {planner_alg}')
        run_prior_only = False
        run_prior_then_guidance = False
        if planner_alg == 'mrrd':
            pass
        elif planner_alg == 'diffusion_prior_then_guide':
            run_prior_then_guidance = True
        elif planner_alg == 'diffusion_prior':
            run_prior_only = True
        else:
            raise NotImplementedError

        ####################################
        model_dir = os.path.join(trained_models_dir, model_id)
        results_dir = os.path.join(model_dir, 'results_inference', str(seed))
        os.makedirs(results_dir, exist_ok=True)

        args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

        ####################################
        # Load dataset with env, robot, task. The TrajectoryDataset type is used here.
        train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
            dataset_class='TrajectoryDataset',
            use_extra_objects=False,
            obstacle_cutoff_margin=0.6,
            **args,
            tensor_args=tensor_args
        )
        # Extract objects from the dataset.
        dataset = train_subset.dataset
        # Number of support points in the trajectory.
        n_support_points = dataset.n_support_points
        # The robot. Contains the dt, the joint limits, etc.
        robot = dataset.robot

        # Instantiate environment and planning task based on env_id
        env_class = getattr(environments, env_id)
        env = env_class(tensor_args=tensor_args)

        task = PlanningTask(
            env=env,
            robot=robot,
            tensor_args=tensor_args,
            obstacle_cutoff_margin=0.6,
            **args
        )
        dataset.env = env
        dataset.task = task
        dataset.planner_visualizer = PlanningVisualizer(task=task)

        dt = trajectory_duration / n_support_points  # time interval for finite differences

        # set robot's dt
        robot.dt = dt

        context = dict()
        # Pad goal with zeros
        goal_full = torch.cat((goal_state_pos, torch.zeros(goal_state_pos.shape[0], device=goal_state_pos.device)), dim=0)
        goal = dataset.normalize_trajectories(goal_full)
        context['goal'] = goal[:2]
        context['goal_time'] = torch.tensor([40], device=goal.device, dtype=torch.long)

        goal_model = GoalModel(in_dim=2, out_dim=16)

        context_model = ContextModel(
            env_model=None, 
            task_model=None,
            goal_model=goal_model
        )

        total_context_dim = context_model.out_dim

        diffusion_configs = dict(
            variance_schedule=args['variance_schedule'],
            n_diffusion_steps=args['n_diffusion_steps'],
            predict_epsilon=args['predict_epsilon'],
        )
        unet_configs = dict(
            state_dim=dataset.state_dim,
            n_support_points=dataset.n_support_points,
            unet_input_dim=args['unet_input_dim'],
            dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
            conditioning_embed_dim=total_context_dim,
            conditioning_type='concatenate'
        )
        diffusion_model = get_model(
            model_class=args['diffusion_model_class'],
            model=TemporalUnet(**unet_configs),
            tensor_args=tensor_args,
            context_model=context_model,
            **diffusion_configs,
            **unet_configs
        )
        diffusion_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args[
                'use_ema'] else 'model_current_state_dict.pth'),
                       map_location=tensor_args['device'], weights_only=True)
        )
        diffusion_model.eval()
        model = diffusion_model

        freeze_torch_model_params(model)
        model = torch.compile(model)
        model.warmup(horizon=n_support_points, device=device, context=context)

        ####################################
        # If the args specify a test start and goal, use those.
        print(f'Environment -- {env.name}')
        if start_state_pos is not None and goal_state_pos is not None:
            print(f'start_state_pos: {start_state_pos}')
            print(f'goal_state_pos: {goal_state_pos}')
        else:
            # Random initial and final positions
            n_tries = 100
            start_state_pos, goal_state_pos = None, None
            for _ in range(n_tries):
                q_free = task.random_coll_free_q(n_samples=2)
                start_state_pos = q_free[0]
                goal_state_pos = q_free[1]

                if torch.linalg.norm(start_state_pos - goal_state_pos) > dataset.threshold_start_goal_pos:
                    break

        if start_state_pos is None or goal_state_pos is None:
            raise ValueError(f"No collision free configuration was found\n"
                             f"start_state_pos: {start_state_pos}\n"
                             f"goal_state_pos:  {goal_state_pos}\n")

        start_state_pos = torch.tensor([0.0, 0.0], device='cuda')
        goal_state_pos = torch.tensor([1.0, 0.4], device='cuda')
        print(f'start_state_pos: {start_state_pos}')
        print(f'goal_state_pos: {goal_state_pos}')

        ####################################
        # Run motion planning inference

        ########
        # normalize start and goal positions
        #hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
        hard_conds = None
        # context = None

        ########
        # Set up the planning costs
        # Cost collisions
        cost_collision_l = []
        weights_grad_cost_l = []  # for guidance, the weights_cost_l are the gradient multipliers (after gradient clipping)
        if use_guide_on_extra_objects_only:
            collision_fields = task.get_collision_fields_extra_objects()
        else:
            collision_fields = task.get_collision_fields()

        for collision_field in collision_fields:
            cost_collision_l.append(
                CostCollision(
                    robot, n_support_points,
                    field=collision_field,
                    sigma_coll=1.0,
                    tensor_args=tensor_args
                )
            )
            weights_grad_cost_l.append(self.weight_grad_cost_collision)

        # Cost smoothness
        cost_smoothness_l = [
            CostGPTrajectory(
                robot, n_support_points, dt, sigma_gp=1.0,
                tensor_args=tensor_args
            )
        ]
        weights_grad_cost_l.append(weight_grad_cost_smoothness)

        ####### Cost composition
        cost_func_list = [
            *cost_collision_l,
            # *cost_smoothness_l,
            # *cost_cbf_l,
            # *cost_max_velocity_l,
        ]
        # A `*cost_constraints_l` will be added as "extra cost" and removed after each planning call.

        cost_composite = CostComposite(
            robot, n_support_points, cost_func_list,
            weights_cost_l=weights_grad_cost_l,
            tensor_args=tensor_args
        )

        ########
        # Guiding manager
        guide = GuideManagerTrajectoriesWithVelocity(
            dataset,
            cost_composite,
            clip_grad=True,
            interpolate_trajectories_for_collision=True,
            num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision),
            tensor_args=tensor_args,
        )

        t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)

        # Keep some variables in the class as members.
        self.start_state_pos = torch.clone(start_state_pos)
        self.goal_state_pos = torch.clone(goal_state_pos)
        # The number of trajectories to generate for each agent.
        self.B = params.B_per_agent
        self.robot_radius = params.robot_planar_disk_radius
        self.robot = robot
        self.context = context
        self.run_prior_only = run_prior_only
        self.run_prior_then_guidance = run_prior_then_guidance
        self.n_diffusion_steps_without_noise = n_diffusion_steps_without_noise
        self.hard_conds = hard_conds
        self.model = model
        self.n_support_points = n_support_points
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.guide = guide
        self.tensor_args = tensor_args
        # Batch-size. How many trajectories to generate at once.
        self.num_samples = n_samples
        # When doing local inference, how many steps to add noise for before denoising again.
        self.n_local_inference_noising_steps = n_local_inference_noising_steps  # n_local_inference_noising_steps
        self.n_local_inference_denoising_steps = n_local_inference_denoising_steps
        # Dataset.
        self.dataset = dataset
        # Environment.
        self.env = env
        # Task, e.g., planning task.
        self.task = task
        # Directories.
        self.results_dir = results_dir
        # Human trajectories.
        self.human_trajectories = human_trajectories

        # Cache of previous call data.
        self.recent_call_data = PlannerOutput()

        self.sample_fn_kwargs = dict(
                 guide=None if self.run_prior_then_guidance or self.run_prior_only else self.guide,
                 n_guide_steps=self.n_guide_steps,
                 t_start_guide=self.t_start_guide,
                 noise_std_extra_schedule_fn=lambda x: 0.5,
        )

    def plan_parallel(self, 
                      start_state_pos_l: List[torch.Tensor], 
                      goal_state_pos_l: List[torch.Tensor], 
                      constraints_l: List[List[MultiPointConstraint]] = None,
                      experience_l: List[PathBatchExperience] = None,
                      cur_time: int = 0,
                      urgencies: List[float] = None,
                      *args,
                      **kwargs) -> List[PlannerOutput]:
        """
        Plan in parallel for multiple robots.
        
        Args:
            start_state_pos_l: List of start states for each agent
            goal_state_pos_l: List of goal states for each agent  
            constraints_l: List of constraint lists, one for each agent (can be None)
            experience_l: List of PathBatchExperience objects, one for each agent (can be None)
        Returns:
            List of PlannerOutput objects, one for each agent. Each PlannerOutput holds a batch of trajectories for this agent as well as their associated costs.
        """
        print(f"======\nMPD plan_parallel called with {len(start_state_pos_l)} requests.\n======\n")
        num_agents = len(start_state_pos_l)
        if len(goal_state_pos_l) != num_agents:
            raise ValueError(f"Number of goal states ({len(goal_state_pos_l)}) must match number of start states ({num_agents})")
        
        if constraints_l is not None and len(constraints_l) != num_agents:
            raise ValueError(f"Number of constraint lists ({len(constraints_l)}) must match number of agents ({num_agents})")
        
        # Create batch of start/goal states for all agents
        # Each agent will get B trajectories, so we need B copies of each agent's start/goal
        batch_start_state_l = []
        batch_goal_state_l = []
        batch_constraint_l = []
        
        for agent_id in range(num_agents):
            # Add B copies of this agent's start/goal states
            for _ in range(self.B):
                batch_start_state_l.append(start_state_pos_l[agent_id])
                batch_goal_state_l.append(goal_state_pos_l[agent_id])
                
                # Add constraints for this agent if provided
                if constraints_l is not None and constraints_l[agent_id] is not None:
                    batch_constraint_l.append(constraints_l[agent_id])
                else:
                    batch_constraint_l.append([])
        
        # Stack all start and goal states into tensors
        batch_start_tensor = torch.stack(batch_start_state_l)  # Shape: [B*N, state_dim]
        batch_goal_tensor = torch.stack(batch_goal_state_l)    # Shape: [B*N, state_dim]

        self.hard_conds = self.dataset.get_multi_agent_hard_conditions(
            batch_start_tensor, batch_goal_tensor, normalize=True
        )
        goal = self.dataset.normalize_trajectories(batch_goal_tensor)
        goal = goal[:, :2]

        if urgencies is None:
            urgencies = [100.0] * num_agents
        goal_time_parts = []
        for urgency in urgencies:
            goal_time_agent = torch.linspace(urgency, 110, steps=self.B, device=goal.device, dtype=torch.long)
            goal_time_parts.append(goal_time_agent)
        goal_time = torch.cat(goal_time_parts, dim=0).to(goal.device)
        self.context = dict()
        self.context['goal'] = goal
        self.context['goal_time'] = goal_time

        human_trajs = None
        if self.human_trajectories is not None:
            human_trajs = self.human_trajectories.predict_trajectories(cur_time, self.n_support_points)

        if human_trajs is not None:
            cost_cbf_l = [
                CostCBF(
                    self.robot, self.n_support_points, human_trajs,
                    tensor_args=self.tensor_args
                )
            ]

            self.guide.add_extra_costs(cost_cbf_l, [self.weight_grad_cost_cbf]) #0.045

        if human_trajs is not None:
            cost_proximity_approach_l = [
                CostProximityApproach(
                    self.robot, self.n_support_points, human_trajs,
                    tensor_args=self.tensor_args
                )
            ]

            self.guide.add_extra_costs(cost_proximity_approach_l, [self.weight_grad_cost_proximity_approach]) #0.045

        with TimerCUDA() as timer_inference:
            if experience_l is None:
                trajs_normalized_iters, _, _ = self.run_constrained_multi_agent_inference(batch_constraint_l)
            else:
                trajs_normalized_iters, _, _ = self.run_constrained_multi_agent_local_inference(batch_constraint_l, experience_l)
            t_total = timer_inference.elapsed
        print(f"      MPD inference time: {t_total:.3f} seconds.")

        # Un-normalize trajectory samples from the models
        trajs_iters = self.dataset.unnormalize_trajectories(trajs_normalized_iters)
        trajs_final = trajs_iters[-1]  # Shape: [B*N, H, D]
        
        # Split trajectories back to individual agents
        # Each agent gets B trajectories
        agent_outputs = []
        for agent_id in range(num_agents):
            # Get trajectories for this agent (B trajectories)
            agent_start_idx = agent_id * self.B
            agent_end_idx = (agent_id + 1) * self.B
            agent_trajs_final = trajs_final[agent_start_idx:agent_end_idx]  # Shape: [B, H, D]
            agent_trajs_iters = [trajs_iter[agent_start_idx:agent_end_idx] for trajs_iter in trajs_iters]

            res_static = self.task.get_trajs_collision_and_free(agent_trajs_final, return_indices=True)
            res_human = self.human_trajectories.filter_trajs_against_humans(agent_trajs_final, human_trajs=human_trajs, human_radius=0.3, robot_radius=0.15, return_indices=True) if human_trajs is not None else None
            agent_trajs_final_coll, agent_trajs_final_coll_idxs, agent_trajs_final_free, agent_trajs_final_free_idxs = self.combine_collision_results(agent_trajs_final, res_static, res_human)
            
            # Compute best trajectory for this agent
            idx_best_traj = None
            idx_best_free_traj = None
            cost_best_free_traj = None
            cost_smoothness = None
            cost_path_length = None
            cost_all = None
            variance_waypoint_trajs_final_free = None
            
            if agent_trajs_final_free is not None:
                cost_smoothness = compute_smoothness(agent_trajs_final_free, self.robot)
                cost_path_length = compute_path_length(agent_trajs_final_free, self.robot)
                cost_linear_index = agent_trajs_final_free_idxs.reshape(-1).float() / self.B
                cost_jerk = compute_max_jerk_metric(agent_trajs_final_free, self.robot)
                
                # Compute best trajectory
                cost_all = cost_jerk + 2 * cost_linear_index # + cost_path_length + cost_smoothness
                idx_best_free_traj = torch.argmin(cost_all).item()
                idx_best_traj = agent_trajs_final_free_idxs[idx_best_free_traj]
                cost_best_free_traj = torch.min(cost_all).item()
                variance_waypoint_trajs_final_free = compute_variance_waypoints(agent_trajs_final_free, self.robot)
            else:
                print('No free trajectories found for agent', agent_id)
                idx_best_free_traj = self.num_samples - 1
                idx_best_traj = self.num_samples - 1
            
            # Create PlannerOutput for this agent
            agent_output = PlannerOutput()
            agent_output.trajs_iters = agent_trajs_iters
            agent_output.trajs_final = agent_trajs_final
            agent_output.trajs_final_coll = agent_trajs_final_coll
            agent_output.trajs_final_coll_idxs = agent_trajs_final_coll_idxs
            agent_output.trajs_final_free = agent_trajs_final_free
            agent_output.t_total = t_total
            agent_output.trajs_final_free_idxs = agent_trajs_final_free_idxs.flatten()
            agent_output.idx_best_traj = idx_best_traj
            agent_output.idx_best_free_traj = idx_best_free_traj
            agent_output.cost_best_free_traj = cost_best_free_traj
            agent_output.cost_smoothness = cost_smoothness
            agent_output.cost_path_length = cost_path_length
            agent_output.cost_all = cost_all
            agent_output.variance_waypoint_trajs_final_free = variance_waypoint_trajs_final_free
            agent_output.constraints_l = constraints_l[agent_id] if constraints_l is not None else None

            # Smooth the trajectories
            if agent_output.trajs_final is not None:
                agent_output.trajs_final = smooth_trajs(agent_output.trajs_final)
            
            agent_outputs.append(agent_output)
        print(f"======\nMPD returned {len(agent_outputs)} agent outputs.\n======\n")
        return agent_outputs

    def __call__(self, start_state_pos, goal_state_pos, constraints_l=None, experience: PathBatchExperience = None, cur_time=0, urgency=100, trajs_init=None, enforce_velocity_limit: bool=False,
                 *args,
                 **kwargs):
        """
        Call the model with the given parameters.
        :param n_samples: Number of trajectories to generate.
        :param start_state_pos: The start state of the robot.
        :param goal_state_pos: The goal state of the robot.
        :param constraints_l: A list of constraints.
        :param previous_path: The previous path of the robot. This would be used to guide the next path.
        """
        # Check that the requested start and goal states are similar to the ones stored.
        # if not torch.allclose(start_state_pos, self.start_state_pos):
        #     raise ValueError("The start state is different from the one stored in the planner.")
        # if not torch.allclose(goal_state_pos, self.goal_state_pos):
        #     raise ValueError("The goal state is different from the one stored in the planner.")

        # Process the constraints into cost components.
        if trajs_init is None:
            trajs_init = start_state_pos.unsqueeze(0)
            indices = np.arange(trajs_init.shape[0])
            self.hard_conds = self.dataset.get_hard_conditions_at_indices(trajs_init, indices, goal=None, normalize=True)
        else:
            indices = np.arange(trajs_init.shape[0])
            self.hard_conds = self.dataset.get_hard_conditions_at_indices(trajs_init, indices, goal=goal_state_pos if urgency <= 1 else None, normalize=True)
        
        self.context = dict()
        # Pad goal with zeros
        goal = self.dataset.normalize_trajectories(goal_state_pos)
        self.context['goal'] = goal[:2]
        self.context['goal_time'] = torch.linspace(urgency, 110, steps=self.num_samples, device=goal.device, dtype=torch.long)

        # for i in range(num_fixed_states_at_end):
        #     self.hard_conds[self.n_support_points - 1 - i] = goal_normalized

        if constraints_l is not None:
            print("Planning with " + str(len(constraints_l)) + " constraints.")
        else:
            print("Planning without constraints.")

        cost_constraints_l = []
        if constraints_l is not None:
            for c in constraints_l:
                cost_constraints_l.append(
                    CostConstraint(
                        self.robot,
                        self.n_support_points,
                        q_l=c.get_q_l(),
                        traj_range_l=c.get_t_range_l(),
                        radius_l=c.radius_l,
                        is_soft=c.is_soft,
                        tensor_args=self.tensor_args
                    )
                )

        human_trajs = None
        if self.human_trajectories is not None:
            human_trajs = self.human_trajectories.predict_trajectories(cur_time, self.n_support_points)

        if human_trajs is not None:
            cost_cbf_l = [
                CostCBF(
                    self.robot, self.n_support_points, human_trajs,
                    tensor_args=self.tensor_args
                )
            ]

            self.guide.add_extra_costs(cost_cbf_l, [self.weight_grad_cost_cbf]) #0.027

        if human_trajs is not None:
            cost_proximity_approach_l = [
                CostProximityApproach(
                    self.robot, self.n_support_points, human_trajs,
                    tensor_args=self.tensor_args
                )
            ]

            self.guide.add_extra_costs(cost_proximity_approach_l, [self.weight_grad_cost_proximity_approach]) #0.010

        # Carry out inference with the constraints. If there is no experience, inference from scratch.
        with TimerCUDA() as timer_inference:
            if experience is None:
                trajs_normalized_iters, _, _ = self.run_constrained_inference(
                    cost_constraints_l,
                    enforce_velocity_limit=enforce_velocity_limit)  # Shape [B (n_samples), H, D]
            # Otherwise, use the experience path as a seed for a local inference call.
            else:
                trajs_normalized_iters, _, _ = self.run_constrained_local_inference(cost_constraints_l, experience, enforce_velocity_limit=enforce_velocity_limit)
        t_total = timer_inference.elapsed

        # Un-normalize trajectory samples from the models.
        trajs_iters = self.dataset.unnormalize_trajectories(trajs_normalized_iters)

        trajs_final = trajs_iters[-1]
        res_static = self.task.get_trajs_collision_and_free(trajs_final, return_indices=True)
        res_human = self.human_trajectories.filter_trajs_against_humans(trajs_final, human_trajs=human_trajs, human_radius=0.3, robot_radius=self.robot_radius, return_indices=True) if human_trajs is not None else None
        trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs = self.combine_collision_results(trajs_final, res_static, res_human)

        # Compute best traj.
        idx_best_traj = None  # Index of the best trajectory in the list of all trajectories (trajs_final).
        idx_best_free_traj = None  # Index of the best trajectory in the list of all free trajs (trajs_final_free).
        cost_best_free_traj = None  # Cost of the best trajectory.
        cost_smoothness = None  # Cost of smoothness for all free trajectories.
        cost_path_length = None  # Cost of path length for all free trajectories.
        cost_all = None  # Cost of all factors for all free trajectories. This is a combination of some costs.
        variance_waypoint_trajs_final_free = None  # Variance of waypoints for all free trajectories.
        if trajs_final_free is not None:
            cost_smoothness = compute_smoothness(trajs_final_free, self.robot)

            cost_jerk = compute_max_jerk_metric(trajs_final_free, self.robot)

            cost_path_length = compute_path_length(trajs_final_free, self.robot)

            if human_trajs is not None:
                cost_human = compute_human_cost(trajs_final_free, human_trajs, self.robot, radius=0.35 + 0.3)
            else:
                cost_human = 0.0

            repeats = int(math.ceil(self.num_samples / goal_state_pos.shape[0]))
            cost_path_index = -2.0 * torch.repeat_interleave(torch.arange(goal_state_pos.shape[0], device=goal_state_pos.device), repeats=repeats)[:self.num_samples]
            cost_path_index = cost_path_index[trajs_final_free_idxs.reshape(-1)]

            cost_linear_index = trajs_final_free_idxs.reshape(-1).float() / self.num_samples

            # Compute best trajectory.
            # cost_all = cost_path_length + cost_smoothness
            cost_all = 2 * cost_jerk + cost_linear_index # + cost_human # + cost_path_index
            idx_best_free_traj = torch.argmin(cost_all).item()
            idx_best_traj = trajs_final_free_idxs[idx_best_free_traj]
            cost_best_free_traj = torch.min(cost_all).item()

            variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, self.robot)
        else:
            print('No free trajectories found for agent')
            idx_best_free_traj = torch.tensor([self.num_samples - 1])
            idx_best_traj = torch.tensor([self.num_samples - 1])
            trajs_final_free_idxs = torch.tensor([self.num_samples - 1])

        self.recent_call_data = PlannerOutput()
        self.recent_call_data.trajs_iters = trajs_iters
        self.recent_call_data.trajs_final = trajs_final
        self.recent_call_data.trajs_final_coll = trajs_final_coll
        self.recent_call_data.trajs_final_coll_idxs = trajs_final_coll_idxs
        self.recent_call_data.trajs_final_free = trajs_final_free  # Shape [B, H, D]
        self.recent_call_data.t_total = t_total
        self.recent_call_data.trajs_final_free_idxs = trajs_final_free_idxs  # Shape [B]
        self.recent_call_data.idx_best_traj = idx_best_traj
        self.recent_call_data.idx_best_free_traj = idx_best_free_traj
        self.recent_call_data.cost_best_free_traj = cost_best_free_traj
        self.recent_call_data.cost_smoothness = cost_smoothness
        self.recent_call_data.cost_path_length = cost_path_length
        self.recent_call_data.cost_all = cost_all
        self.recent_call_data.variance_waypoint_trajs_final_free = variance_waypoint_trajs_final_free
        self.recent_call_data.constraints_l = constraints_l

        # Smooth the trajectories in trajs_final.
        if self.recent_call_data.trajs_final is not None:
            print('Smoothing final trajectories.')
            self.recent_call_data.trajs_final = smooth_trajs(self.recent_call_data.trajs_final)

        return self.recent_call_data

    def run_constrained_inference(self, cost_constraints_l: List[CostConstraint], enforce_velocity_limit: bool=False):
        # Add these cost factors, alongside their weights as specified, to the `guide` object of the model.
        self.guide.add_extra_costs(cost_constraints_l,
                                   [self.weight_grad_cost_soft_constraints if c.is_soft else
                                    self.weight_grad_cost_constraints
                                    for c in cost_constraints_l])

        # Sample trajectories with the diffusion/cvae model
        with TimerCUDA() as timer_model_sampling:
            trajs_normalized_iters = self.model.run_inference(
                self.context, self.hard_conds,
                n_samples=self.num_samples, horizon=self.n_support_points,
                return_chain=True,
                sample_fn=ddpm_sample_fn,
                **self.sample_fn_kwargs,
                n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
                enforce_velocity_limit=enforce_velocity_limit,
                # ddim=True
            )
        t_model_sampling = timer_model_sampling.elapsed
        print(f't_model_sampling: {t_model_sampling:.3f} sec')

        ########
        # run extra guiding steps without diffusion
        t_post_diffusion_guide = 0.0
        if self.run_prior_then_guidance:
            n_post_diffusion_guide_steps = (self.t_start_guide +
                                            self.n_diffusion_steps_without_noise) * self.n_guide_steps
            print(CYAN + f'Running extra guiding steps without diffusion. Num steps:', n_post_diffusion_guide_steps,
                  RESET)
            with TimerCUDA() as timer_post_model_sample_guide:
                trajs = trajs_normalized_iters[-1]
                trajs_post_diff_l = []
                for i in range(n_post_diffusion_guide_steps):
                    trajs = guide_gradient_steps(
                        trajs,
                        hard_conds=self.hard_conds,
                        guide=self.guide,
                        n_guide_steps=1,
                        unnormalize_data=False
                    )
                    trajs_post_diff_l.append(trajs)

                chain = torch.stack(trajs_post_diff_l, dim=1)
                chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
                trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))
            t_post_diffusion_guide = timer_post_model_sample_guide.elapsed
            print(f't_post_diffusion_guide: {t_post_diffusion_guide:.3f} sec')

        # Remove the extra cost.
        self.guide.reset_extra_costs()

        return trajs_normalized_iters, t_model_sampling, t_post_diffusion_guide

    def run_constrained_multi_agent_inference(self, batch_constraints_l: List[List[MultiPointConstraint]]):
        """
        Run constrained inference with agent-specific hard conditions.
        :param batch_constraints_l: A list of lists of constraints, one for each query in the batch. Normally, each query will have two MultiPointConstraints, one soft and one hard.
        """

        # Create cost constraints dynamically based on the current batch size
        cost_constraints_l = []
        
        if batch_constraints_l:
            # Group constraints by type (soft vs hard) across all batch entries
            soft_constraints = []
            hard_constraints = []
            
            # Extract soft and hard constraints from each batch entry
            for batch_idx, constraints in enumerate(batch_constraints_l):
                if not constraints:
                    continue
                    
                batch_soft_q_l = []
                batch_soft_traj_range_l = []
                batch_soft_radius_l = []
                
                batch_hard_q_l = []
                batch_hard_traj_range_l = []
                batch_hard_radius_l = []
                
                for constraint in constraints:
                    q_l = constraint.get_q_l()
                    traj_range_l = constraint.get_t_range_l()
                    radius_l = constraint.radius_l
                    
                    if constraint.is_soft:
                        batch_soft_q_l.extend(q_l)
                        batch_soft_traj_range_l.extend(traj_range_l)
                        batch_soft_radius_l.extend(radius_l)
                    else:
                        batch_hard_q_l.extend(q_l)
                        batch_hard_traj_range_l.extend(traj_range_l)
                        batch_hard_radius_l.extend(radius_l)
                
                # Add to the global lists
                if batch_soft_q_l:
                    soft_constraints.append({
                        'batch_idx': batch_idx,
                        'q_l': batch_soft_q_l,
                        'traj_range_l': batch_soft_traj_range_l,
                        'radius_l': batch_soft_radius_l
                    })
                
                if batch_hard_q_l:
                    hard_constraints.append({
                        'batch_idx': batch_idx,
                        'q_l': batch_hard_q_l,
                        'traj_range_l': batch_hard_traj_range_l,
                        'radius_l': batch_hard_radius_l
                    })
            
            # Create CostConstraintBatched objects
            if soft_constraints:
                # Create a batched constraint for all soft constraints
                batch_q_l = [sc['q_l'] for sc in soft_constraints]
                batch_traj_range_l = [sc['traj_range_l'] for sc in soft_constraints]
                batch_radius_l = [sc['radius_l'] for sc in soft_constraints]
                
                cost_constraints_l.append(
                    CostConstraintBatched(
                        robot=self.robot,
                        n_support_points=self.n_support_points,
                        batch_q_l=batch_q_l,
                        batch_traj_range_l=batch_traj_range_l,
                        batch_radius_l=batch_radius_l,
                        is_soft=True,
                        tensor_args=self.tensor_args
                    )
                )
            
            if hard_constraints:
                # Create a batched constraint for all hard constraints
                batch_q_l = [hc['q_l'] for hc in hard_constraints]
                batch_traj_range_l = [hc['traj_range_l'] for hc in hard_constraints]
                batch_radius_l = [hc['radius_l'] for hc in hard_constraints]
                
                cost_constraints_l.append(
                    CostConstraintBatched(
                        robot=self.robot,
                        n_support_points=self.n_support_points,
                        batch_q_l=batch_q_l,
                        batch_traj_range_l=batch_traj_range_l,
                        batch_radius_l=batch_radius_l,
                        is_soft=False,
                        tensor_args=self.tensor_args
                    )
                )
        
        self.guide.add_extra_costs(cost_constraints_l,
                            [self.weight_grad_cost_soft_constraints if c.is_soft else
                            self.weight_grad_cost_constraints
                            for c in cost_constraints_l])

        # Sample trajectories with the diffusion/cvae model using multi-agent inference
        with TimerCUDA() as timer_model_sampling:
            # Update sample_fn_kwargs to use the efficient guide
            sample_fn_kwargs_updated = self.sample_fn_kwargs.copy()
            # sample_fn_kwargs_updated['guide'] = efficient_guide
            
            trajs_normalized_iters = self.model.run_multi_agent_inference(
                self.context, self.hard_conds,
                return_chain=True,
                sample_fn=ddpm_sample_fn,
                **sample_fn_kwargs_updated,
                n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
                # ddim=True
            )
        t_model_sampling = timer_model_sampling.elapsed
        print(f't_model_sampling: {t_model_sampling:.3f} sec')
        print(f"      MPD model sampling time: {t_model_sampling:.3f} seconds.")
        ########
        # run extra guiding steps without diffusion
        t_post_diffusion_guide = 0.0
        if self.run_prior_then_guidance:
            n_post_diffusion_guide_steps = (self.t_start_guide +
                                            self.n_diffusion_steps_without_noise) * self.n_guide_steps
            print(CYAN + f'Running extra guiding steps without diffusion. Num steps:', n_post_diffusion_guide_steps,
                  RESET)
            with TimerCUDA() as timer_post_model_sample_guide:
                trajs = trajs_normalized_iters[-1]
                trajs_post_diff_l = []
                for i in range(n_post_diffusion_guide_steps):
                    trajs = guide_gradient_steps(
                        trajs,
                        hard_conds=self.hard_conds,
                        guide=self.guide,  # Use the efficient guide here
                        n_guide_steps=1,
                        unnormalize_data=False
                    )
                    trajs_post_diff_l.append(trajs)

                chain = torch.stack(trajs_post_diff_l, dim=1)
                chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
                trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))
            t_post_diffusion_guide = timer_post_model_sample_guide.elapsed
            print(f't_post_diffusion_guide: {t_post_diffusion_guide:.3f} sec')

        # Remove the extra cost.
        self.guide.reset_extra_costs()

        return trajs_normalized_iters, t_model_sampling, t_post_diffusion_guide

    def run_constrained_local_inference(self, cost_constraints_l: List[CostConstraint],
                                        experience: PathBatchExperience, enforce_velocity_limit: bool=False):
        # Add these cost factors, alongside their weights as specified, to the `guide` object of the model.
        self.guide.add_extra_costs(cost_constraints_l,
                                   [self.weight_grad_cost_soft_constraints if c.is_soft else
                                    self.weight_grad_cost_constraints
                                    for c in cost_constraints_l])
        
        normalized_experience = self.dataset.normalize_trajectories(experience.path_b)

        # Sample trajectories with the diffusion/cvae model
        with TimerCUDA() as timer_model_sampling:
            trajs_normalized_iters = self.model.run_local_inference(
                normalized_experience,
                self.n_local_inference_noising_steps,
                self.n_local_inference_denoising_steps,
                self.context,
                self.hard_conds,
                n_samples=self.num_samples,
                horizon=self.n_support_points,
                return_chain=True,
                sample_fn=ddpm_sample_fn,
                **self.sample_fn_kwargs,
                n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
                enforce_velocity_limit=enforce_velocity_limit,
                # ddim=True
            )
        t_model_sampling = timer_model_sampling.elapsed
        print(f't_model_sampling: {t_model_sampling:.3f} sec')

        ########
        # Run extra guiding steps without diffusion. This is only done if variable `run_prior_then_guidance` is True.
        t_post_diffusion_guide = 0.0
        if self.run_prior_then_guidance:
            n_post_diffusion_guide_steps = (self.t_start_guide +
                                            self.n_diffusion_steps_without_noise) * self.n_guide_steps
            print(CYAN + f'Running extra guiding steps without diffusion. Num steps:', n_post_diffusion_guide_steps,
                  RESET)
            with TimerCUDA() as timer_post_model_sample_guide:
                trajs = trajs_normalized_iters[-1]
                trajs_post_diff_l = []
                for i in range(n_post_diffusion_guide_steps):
                    trajs = guide_gradient_steps(
                        trajs,
                        hard_conds=self.hard_conds,
                        guide=self.guide,
                        n_guide_steps=1,
                        unnormalize_data=False
                    )
                    trajs_post_diff_l.append(trajs)

                chain = torch.stack(trajs_post_diff_l, dim=1)
                chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
                trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))
            t_post_diffusion_guide = timer_post_model_sample_guide.elapsed
            print(f't_post_diffusion_guide: {t_post_diffusion_guide:.3f} sec')

        # Remove the extra cost.
        self.guide.reset_extra_costs()

        return trajs_normalized_iters, t_model_sampling, t_post_diffusion_guide

    def combine_collision_results(self, trajs, *collision_results):
        """
        Combines results from multiple collision/validity checks into a unified 4-tuple.
        Silently ignores None inputs and intersects all valid/free masks.
        """
        B = trajs.shape[0]
        is_free = torch.ones(B, dtype=torch.bool, device=trajs.device)

        for res in collision_results:
            if res is None:
                continue
            if isinstance(res, torch.Tensor) and res.dtype == torch.bool:
                is_free &= res
            elif isinstance(res, (tuple, list)) and len(res) >= 4:
                free_idxs = res[3]
                if free_idxs is None or free_idxs.nelement() == 0:
                    is_free.zero_()
                    break
                mask = torch.zeros(B, dtype=torch.bool, device=trajs.device)
                mask[free_idxs.reshape(-1)] = True
                is_free &= mask

        trajs_free = trajs[is_free] if is_free.any() else None
        trajs_coll = trajs[~is_free] if (~is_free).any() else None
        trajs_free_idxs = torch.argwhere(is_free)
        trajs_coll_idxs = torch.argwhere(~is_free)

        return trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs
