"""
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
# Standard imports.

import os
import pickle
from datetime import datetime
import time
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from typing import Tuple, List
import concurrent.futures

# Project imports.
from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory, CostConstraint
from mrrd.models import TemporalUnet, UNET_DIM_MULTS
from mrrd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mrrd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mrrd.trainer import get_dataset, get_model
from mrrd.utils.loading import load_params_from_yaml
from torch_robotics.robots import *
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints, \
    compute_average_acceleration, compute_average_acceleration_from_pos_vel, compute_path_length_from_pos, compute_human_cost, \
    compute_goal_reaching
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from mrrd.planners.multi_agent import CBS
from mrrd.planners.single_agent import MPD
from mrrd.common.constraints import MultiPointConstraint, VertexConstraint, EdgeConstraint
from mrrd.common.conflicts import VertexConflict, PointConflict, EdgeConflict
from mrrd.common.trajectory_utils import smooth_trajs, densify_trajs
from mrrd.common import compute_collision_intensity, is_multi_agent_start_goal_states_valid, global_pad_paths, \
    get_start_goal_pos_circle, get_state_pos_column, get_start_goal_pos_boundary, get_start_goal_pos_random_in_env, \
    get_start_goal_pos_random_boundary, get_start_goal_pos_random_circle, get_start_goal_pos_random_in_env_humans
from mrrd.common.pretty_print import *
from mrrd.config.mrrd_params import MRRDParams as params
from mrrd.common.experiments import MultiAgentPlanningSingleTrialConfig, MultiAgentPlanningSingleTrialResult, \
    get_result_dir_from_trial_config, TrialSuccessStatus
from torch_robotics.environments import *
from mrrd.humans import HumanTrajectories

allow_ops_in_compiled_graph()

TRAINED_MODELS_DIR = '../../data_trained_models/'
HUMANS_TRAJ_DIR = '../../data_human_trajectories/'
device = 'cuda'
device = get_torch_device(device)
tensor_args = {'device': device, 'dtype': torch.float32}


def run_multi_agent_trial(test_config: MultiAgentPlanningSingleTrialConfig):
    # ============================
    # Start time per agent.
    # ============================
    start_time_l = [i * test_config.stagger_start_time_dt for i in range(test_config.num_agents)]
    # Choose starts and goals.
    torch.random.manual_seed(test_config.random_seed)
    # torch.random.manual_seed(126)
    human_trajectories = HumanTrajectories(os.path.join(HUMANS_TRAJ_DIR, 'ucy_scenarios2.pt'), sample_idx=test_config.human_trajectory_index, max_humans=10)

    # ============================
    # Arguments for the high/low level planner.
    # ============================
    low_level_planner_model_args = {
        'planner_alg': 'mrrd',
        'use_guide_on_extra_objects_only': params.use_guide_on_extra_objects_only,
        'n_samples': params.n_samples,
        'n_local_inference_noising_steps': params.n_local_inference_noising_steps,
        'n_local_inference_denoising_steps': params.n_local_inference_denoising_steps,
        'start_guide_steps_fraction': params.start_guide_steps_fraction,
        'n_guide_steps': params.n_guide_steps,
        'n_diffusion_steps_without_noise': params.n_diffusion_steps_without_noise,
        'weight_grad_cost_collision': params.weight_grad_cost_collision,
        'weight_grad_cost_smoothness': params.weight_grad_cost_smoothness,
        'weight_grad_cost_constraints': params.weight_grad_cost_constraints,
        'weight_grad_cost_soft_constraints': params.weight_grad_cost_soft_constraints,
        'weight_grad_cost_cbf': params.weight_grad_cost_cbf,
        'weight_grad_cost_proximity_approach': params.weight_grad_cost_proximity_approach,
        'factor_num_interpolated_points_for_collision': params.factor_num_interpolated_points_for_collision,
        'trajectory_duration': params.trajectory_duration,
        'device': params.device,
        'debug': params.debug,
        'seed': params.seed,
        'results_dir': params.results_dir,
        'trained_models_dir': TRAINED_MODELS_DIR,
        'human_trajectories': human_trajectories,
        'env_id': test_config.env_id,
    }
    high_level_planner_model_args = {
        'is_xcbs': True if test_config.multi_agent_planner_class in ["XECBS", "XCBS", "BPCBS"] else False,
        'is_ecbs': True if test_config.multi_agent_planner_class in ["ECBS", "XECBS", "BPCBS"] else False,
        'start_time_l': start_time_l,
        'runtime_limit': test_config.runtime_limit,
        'conflict_type_to_constraint_types': {PointConflict: {MultiPointConstraint}},
        'use_storage': test_config.use_storage,
        'optimize_for': test_config.optimize_for,
        'num_ct_nodes_to_expand_in_parallel': test_config.num_ct_nodes_to_expand_in_parallel
    }

    # ============================
    # Create a results directory.
    # ============================
    results_dir = get_result_dir_from_trial_config(test_config, test_config.time_str, test_config.trial_number)
    os.makedirs(results_dir, exist_ok=True)
    num_agents = test_config.num_agents

    # ============================
    # Get planning problem.
    # ============================
    # Simplified to single tile since skeletons are removed
    start_l = [torch.tensor([-6.0, 0.0], device=device) for _ in range(num_agents)]
    goal_l = [torch.tensor([6.0, 0.0], device=device) for _ in range(num_agents)]
    global_model_ids = test_config.global_model_ids

    # ============================
    # Simplified single tile setup.
    # ============================
    # For single tile planning, we don't need complex transforms
    reference_model_id = global_model_ids[0][0]

    # ============================
    # Parse the single agent planner class name.
    # ============================
    if test_config.single_agent_planner_class == "MPD":
        low_level_planner_class = MPD

    else:
        raise ValueError(f'Unknown single agent planner class: {test_config.single_agent_planner_class}')

    # ============================
    # Create the single low level planner for all agents.
    # ============================
    planners_creation_start_time = time.time()
    # Create a single planner instance that will be used for all agents
    # The start/goal states will be provided dynamically in each call
    low_level_planner_model_args['start_state_pos'] = start_l[0]  # Will be provided dynamically
    low_level_planner_model_args['goal_state_pos'] = goal_l[0]   # Will be provided dynamically
    low_level_planner_model_args['model_id'] = reference_model_id
    low_level_planner = low_level_planner_class(**low_level_planner_model_args)
    print('Single planner creation time:', time.time() - planners_creation_start_time)
    print("\n\n\n\n")

    human_trajectories.filter_paths_by_obstacle_intersection(low_level_planner.task)
    torch.random.manual_seed(test_config.random_seed)
    test_config.start_state_pos_l, test_config.goal_state_pos_l = \
    get_start_goal_pos_random_in_env_humans(test_config.num_agents,
                                            type(low_level_planner.task.env),
                                            tensor_args,
                                            human_trajectories)

    start_l = test_config.start_state_pos_l
    goal_l = test_config.goal_state_pos_l

    # ============================
    # Run trial.
    # ============================
    exp_name = f'mrrd_single_trial'

    # ============================
    # Create the multi agent planner.
    # ============================
    if (test_config.multi_agent_planner_class in ["XECBS", "ECBS", "XCBS", "CBS"]):
        multi_agent_planner_class = CBS
    else:
        raise ValueError(f'Unknown multi agent planner class: {test_config.multi_agent_planner_class}')
    
    low_level_planner_l = []
    for i in range(num_agents):
        low_level_planner_l.append(low_level_planner)
    
    planner = multi_agent_planner_class(low_level_planner_l,
                                        start_l,
                                        goal_l,
                                        **high_level_planner_model_args)

    # ============================
    # Plan.
    # ============================
    startt = time.time()
    paths_l, num_ct_expansions, trial_success_status, num_collisions_in_solution, total_travel_time, rolling_horizon_times = \
        planner.run_continuous_plan(runtime_limit=test_config.runtime_limit)
    planning_time = time.time() - startt
    # Print planning times.
    print(GREEN, 'Planning times:', planning_time, RESET)

    # ============================
    # Gather stats.
    # ============================
    single_trial_result = MultiAgentPlanningSingleTrialResult()
    # The associated experiment config.
    single_trial_result.trial_config = test_config
    # The planning problem.
    single_trial_result.start_state_pos_l = [start_l[i].cpu().numpy().tolist() for i in range(num_agents)]
    single_trial_result.goal_state_pos_l = [goal_l[i].cpu().numpy().tolist() for i in range(num_agents)]
    single_trial_result.global_model_ids = global_model_ids

    # The agent paths. Each entry is of shape (H, 4).
    single_trial_result.agent_path_l = paths_l
    # Success.
    single_trial_result.success_status = trial_success_status
    # Number of collisions in the solution.
    single_trial_result.num_collisions_in_solution = num_collisions_in_solution
    # Planning time.
    single_trial_result.planning_time = planning_time
    # Total travel time.
    single_trial_result.total_travel_time = total_travel_time
    # Rolling horizon times.
    single_trial_result.rolling_horizon_times = rolling_horizon_times
    # Number of agent pairs in collision.
    if len(paths_l) > 0 and trial_success_status:
        # This handles paths of potentially different lengths.
        max_len = max([len(path) for path in paths_l])
        for t in range(max_len):
            for i in range(num_agents):
                t_i = min(t, len(paths_l[i]) - 1)
                pos_i = paths_l[i][t_i, :2]
                for j in range(i + 1, num_agents):
                    t_j = min(t, len(paths_l[j]) - 1)
                    pos_j = paths_l[j][t_j, :2]
                    if torch.norm(pos_i - pos_j) < 2.0 * params.robot_planar_disk_radius:
                        # The above should be reference_robot.radius.
                        print(RED, 'Collision in solution:', i, j, t, pos_i, pos_j, RESET)
                        single_trial_result.num_collisions_in_solution += 1
        if single_trial_result.num_collisions_in_solution > 0:
            single_trial_result.success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS

        # Number of human collisions in the solution.
        paths_l_tensor = torch.stack(paths_l, dim=0)
        trajs_coll, _ = human_trajectories.filter_trajs_against_humans(paths_l_tensor, 
                                                                       params.human_radius, 
                                                                       params.robot_planar_disk_radius)
        collision_fraction = 0 if trajs_coll is None else len(trajs_coll) / num_agents
        single_trial_result.fraction_agents_colliding_with_humans = collision_fraction

        # Obstacle collision rate.
        trajs_coll, _ = planner.reference_task.get_trajs_collision_and_free(paths_l_tensor)
        collision_fraction = 0 if trajs_coll is None else len(trajs_coll) / num_agents
        single_trial_result.fraction_agents_colliding_with_obstacles = collision_fraction

        # Human cost.
        human_avoidance_score = compute_human_cost(paths_l_tensor, human_trajectories.get_human_trajectories(), low_level_planner.robot, 1.0)
        single_trial_result.human_avoidance_score = human_avoidance_score.mean().item()

        # Goal reaching score.
        goal_reaching_score = compute_goal_reaching(paths_l_tensor, torch.stack(test_config.goal_state_pos_l, dim=0))
        single_trial_result.goal_reaching_score = goal_reaching_score.mean().item()

        # Percentage of rolling horizon times that exceeded the threshold.
        if len(rolling_horizon_times) > 0:
            single_trial_result.percentage_horizon_exceeded = sum([1 for t in rolling_horizon_times if t > 1.6]) / len(rolling_horizon_times)
        else:
            single_trial_result.percentage_horizon_exceeded = 0.0

    # If not successful, return here.
    if trial_success_status:
        single_trial_result.data_adherence = 0
        # CT nodes expanded.
        single_trial_result.num_ct_expansions = num_ct_expansions
        # Path length. Hack for experiments.
        single_trial_result.path_length_per_agent = 0.0
        for agent_id in range(num_agents):
            # agent_path_pos = low_level_planner.robot.get_position(paths_l[agent_id]).unsqueeze(0)
            agent_path_pos = paths_l[agent_id][:, :2].unsqueeze(0)
            single_trial_result.path_length_per_agent += compute_path_length_from_pos(agent_path_pos).item()
        single_trial_result.path_length_per_agent /= num_agents
        # Path smoothness.
        single_trial_result.mean_path_acceleration_per_agent = 0.0
        for agent_id in range(num_agents):
            agent_path_pos = paths_l[agent_id][:, :2].unsqueeze(0)
            agent_path_vel = paths_l[agent_id][:, 2:].unsqueeze(0)
            if agent_path_vel.shape[-1] == 0:
                agent_path_vel = low_level_planner.robot.get_velocity(paths_l[agent_id]).unsqueeze(0)

            single_trial_result.mean_path_acceleration_per_agent += (
                compute_average_acceleration_from_pos_vel(agent_path_pos, agent_path_vel).item())
        single_trial_result.mean_path_acceleration_per_agent /= num_agents
    # ============================
    # Save the results and config.
    # ============================
    print(GREEN, single_trial_result, RESET)
    results_dir_uri = f'file://{os.path.abspath(results_dir)}'
    print('Results dir:', results_dir_uri)
    single_trial_result.save(results_dir)
    test_config.save(results_dir)

    # ============================
    # Save trajectories as numpy arrays.
    # ============================
    if trial_success_status and len(paths_l) > 0:
        # Robot trajectories: each entry is (H_i, 4). Pad to equal length and stack.
        max_len = 200
        robot_trajs_np = np.stack([
            np.pad(p[:, :2].cpu().numpy(), ((0, max_len - p.shape[0]), (0, 0)), mode='edge')
            for p in paths_l
        ])  # shape: (num_agents, max_len, 2)
        np.save(os.path.join(results_dir, 'robot_trajectories.npy'), robot_trajs_np)

        # Human trajectories.
        human_trajs_np = human_trajectories.get_human_trajectories().cpu().numpy()
        np.save(os.path.join(results_dir, 'human_trajectories.npy'), human_trajs_np)

        print(f'Saved robot trajectories {robot_trajs_np.shape} and human trajectories {human_trajs_np.shape} to {results_dir}')

    # ============================
    # Render.
    # ============================
    if trial_success_status and len(paths_l) > 0:
        robot_trajs = [paths_l[i][:, :2] for i in range(len(paths_l))]
        pv = PlanningVisualizer(task=planner.reference_task)
        pv.render_recent_result_with_humans(robot_trajs,
                                           human_trajs=human_trajectories.get_human_trajectories(),
                                           start_positions=test_config.start_state_pos_l,
                                           goal_positions=test_config.goal_state_pos_l,
                                           animation_duration=20.0,
                                           video_filepath=os.path.join(results_dir, f'{exp_name}_with_humans.gif'))
        
        # pv.render_ego_centric_view(robot_trajs,
        #                            human_trajs=human_trajectories.get_human_trajectories(),
        #                            start_positions=test_config.start_state_pos_l,
        #                            goal_positions=test_config.goal_state_pos_l,
        #                            animation_duration=20.0,
        #                            video_filepath=os.path.join(results_dir, f'{exp_name}_ego.gif'))


if __name__ == '__main__':
    test_config_single_tile = MultiAgentPlanningSingleTrialConfig()
    test_config_single_tile.num_agents = 6
    test_config_single_tile.instance_name = "test"
    
    test_config_single_tile.multi_agent_planner_class = "XECBS"  # Or "ECBS" or "XCBS" or "CBS" or "PP".
    test_config_single_tile.single_agent_planner_class = "MPD"  # Or "MPD"
    test_config_single_tile.stagger_start_time_dt = 0
    test_config_single_tile.runtime_limit = 60 * 15  # 15 minutes.
    test_config_single_tile.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_config_single_tile.render_animation = True  # Change the `densify_trajs` call above to create nicer animations.

    # Environment to use for planning
    test_config_single_tile.env_id = "EnvEmptyNoWait2D"
    # test_config_single_tile.env_id = "EnvSimple2D"
    # test_config_single_tile.env_id = "EnvSpheres2DSparse"
    # test_config_single_tile.env_id = "EnvSpheres2D"

    # Name of the model to use
    test_config_single_tile.global_model_ids = [['HumanDiffusion']]

    # No need to touch these
    test_config_single_tile.use_storage = True
    test_config_single_tile.optimize_for = 'conflicts'  # 'cost' or 'conflicts'
    test_config_single_tile.num_ct_nodes_to_expand_in_parallel = 1  # Number of CT nodes to expand in parallel.

    # Decides the index of the ucy_scenarios file to use
    # There should be 200 indexes in total, but not all contain many agents
    test_config_single_tile.human_trajectory_index = 110

    # Random seed decides the start and goal locations of the agents
    test_config_single_tile.random_seed = 56

    run_multi_agent_trial(test_config_single_tile)
    print(GREEN, 'OK.', RESET)
