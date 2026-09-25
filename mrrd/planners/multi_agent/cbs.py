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
import time
from math import floor, ceil
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List, Dict, Type
from enum import Enum
import concurrent.futures
import uuid
import pickle
import math

# Project imports.
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer, create_fig_and_axes
from mrrd.common.conflicts import VertexConflict, Conflict, PointConflict, EdgeConflict
from mrrd.common.constraints import MultiPointConstraint, Constraint
from mrrd.common.conflict_conversion import convert_conflicts_to_constraints
from mrrd.common.experiences import PathExperience, PathBatchExperience
from mrrd.common.pretty_print import *
from mrrd.common import densify_trajs, smooth_trajs, is_multi_agent_start_goal_states_valid, global_pad_paths
from mrrd.config import MRRDParams as params
from mrrd.common.experiments import TrialSuccessStatus
from mrrd.planners.single_agent import MPD

"""
Some comments:
1. This assumes, as of now, a homogeneous team of robots (so that asking for one to compute
    the conflicts with others makes sense with a robot method).
"""


class CBSExperienceReuseStrategy(Enum):
    """
    Enum for replanning strategies in CBS.
    """
    NONE = 0
    XCBS = 1
    NOISE_AS_EXPERIENCE = 2


class SearchState:
    """
    Constraint Tree node for CBS.
    """

    def __init__(self, ix_best_path_in_batch_l, path_bl, constraints={}, level=0):
        self.path_bl = path_bl  # List of batch of paths. (list of n_agents) x B x H x q_dim.
        self.ix_best_path_in_batch_l = ix_best_path_in_batch_l  # List of indices of the best path in the batch. (n_agents,).
        self.conflict_l = []
        self.constraints = constraints  # Map of agent_id: List[Constraint].
        self.g = float('inf')  # Cost to reach this node.
        self.level = level  # Level in the CBS tree.

    def update_g_l2(self):
        """
        Update the cost to reach this node.
        """
        self.g = 0
        for i, ix_best_path_in_batch in enumerate(self.ix_best_path_in_batch_l):
            path = self.path_bl[i][ix_best_path_in_batch]
            path_cost = torch.norm(path[1:] - path[:-1], dim=-1).sum()
            self.g += path_cost

    def add_constraint(self, agent_id, constraint):
        """
        Add a constraint to the state.
        """
        if agent_id not in self.constraints:
            self.constraints[agent_id] = []
        print(RED + f'Adding constraint for agent {agent_id}. Before:', len(self.constraints[agent_id]), RESET)
        self.constraints[agent_id].append(constraint)
        print(GREEN + f'After:', len(self.constraints[agent_id]), RESET)

    def get_copy(self):
        """
        Create a copy of the state.
        """
        new_ix_best_path_in_batch_l = self.ix_best_path_in_batch_l.copy()
        new_path_bl = [path_b.clone() for path_b in self.path_bl]
        new_constraints = {k: [c.get_copy() for c in v] for k, v in self.constraints.items()}
        new_state = SearchState(new_ix_best_path_in_batch_l, new_path_bl, new_constraints, level=self.level)
        new_state.conflict_l = self.conflict_l
        new_state.g = self.g
        return new_state


class CBS:
    """
    Conflict-Based Search (CBS) algorithm.
    """
    def __init__(self, low_level_planner_l,
                 start_l: List[torch.Tensor],
                 goal_l: List[torch.Tensor],
                 start_time_l: List[int] = None,
                 is_xcbs=False,
                 is_ecbs=True,
                 conflict_type_to_constraint_types: Dict[Type[Conflict], Type[Constraint]] = None,
                 reference_robot=None,
                 reference_task=None,
                 use_storage=True,
                 optimize_for="conflicts",
                 **kwargs):
        # Some parameters:
        self.low_level_choose_path_from_batch_strategy = params.low_level_choose_path_from_batch_strategy
        # Set the low level planners.
        self.low_level_planner_l = low_level_planner_l
        # Whether to use experience in the low level planner.
        self.is_xcbs = is_xcbs
        self.experience_reuse_strategy = CBSExperienceReuseStrategy.XCBS
        # Which conflicts to find and what constraints to create from them.
        self.conflict_type_to_constraint_types = conflict_type_to_constraint_types
        # Whether to impose soft constraints from other agents' paths.
        self.is_ecbs = is_ecbs
        # Whether to use trajectory storage.
        self.use_storage = False
        # Whether to optimize for cost or conflicts.
        self.optimize_for = optimize_for
        self.num_agents = len(start_l)
        self.agent_color_l = plt.cm.get_cmap('tab20')(torch.linspace(0, 1, self.num_agents))

        start_l = [torch.cat([s, torch.zeros(2, device=s.device, dtype=s.dtype)]) if s.shape[-1] == 2 else s for s in start_l]
        goal_l = [torch.cat([g, torch.zeros(2, device=g.device, dtype=g.dtype)]) if g.shape[-1] == 2 else g for g in goal_l]

        self.start_state_pos_l = start_l
        self.goal_state_pos_l = goal_l
        if start_time_l is None:
            self.start_time_l = [0] * self.num_agents
        else:
            self.start_time_l = start_time_l
        # Keep a reference robot for collision checking in a group of robots.
        if reference_robot is None:
            print(CYAN + 'Using the first robot in the low level planner list as the reference robot.' + RESET)
            self.reference_robot = self.low_level_planner_l[0].robot
        else:
            self.reference_robot = reference_robot
        if reference_task is None:
            print(CYAN + 'Using the first task in the low level planner list as the reference task.' + RESET)
            self.reference_task = self.low_level_planner_l[0].task
        else:
            self.reference_task = reference_task
        self.tensor_args = self.low_level_planner_l[0].tensor_args
        self.results_dir = self.low_level_planner_l[0].results_dir
        # Check for collisions between robots, and between robots and obstacles, in their start and goal states.
        # if not is_multi_agent_start_goal_states_valid(self.reference_robot,
        #                                               self.reference_task,
        #                                               self.start_state_pos_l,
        #                                               self.goal_state_pos_l):
        #     print(RED + 'Start or goal states are invalid.')
        #     print(self.start_state_pos_l)
        #     print(self.goal_state_pos_l, RESET)
        #     raise ValueError('Start or goal states are invalid.')
        # Open list.
        self.open_l = []
        # Add trajectory storage for each agent
        self.trajectory_storage = {i: None for i in range(self.num_agents)}  # agent_id: tensor [N, T, D] or None
        self.storage_usage_by_level = {}  # level: count
        self.diffusion_usage_by_level = {}  # level: count

        # self.human_trajs = self.low_level_planner_l[0].human_trajs
        self.root = None

        self.execution_horizon = 16
        self.horizon = 32
        self.max_rounds = 4
        self.current_time = 0
        self.time_step = 0.1

    def trajectory_batch_satisfies_constraint(self, trajs, constraints_l):
        """
        Returns:
            satisfying_trajs: [K, T, D] tensor of trajectories that satisfy all hard constraints
            total_constraint_costs: [K] tensor, total constraint cost (hard+soft) for each satisfying traj
        """
        if not isinstance(constraints_l, list):
            constraints_l = [constraints_l]

        hard_constraints = [c for c in constraints_l if not getattr(c, 'is_soft', False)]
        soft_constraints = [c for c in constraints_l if getattr(c, 'is_soft', False)]
        device = trajs.device
        N, T, D = trajs.shape
        
        if len(hard_constraints) == 0:
            hard_mask = torch.ones(N, dtype=torch.bool, device=device)
        else:
            q_l = []
            t_range_l = []
            radius_l = []
            for c in hard_constraints:
                q_l.extend(list(c.q_l))
                t_range_l.extend(list(c.t_range_l))
                radius_l.extend(list(c.radius_l))
            q_tensor = torch.stack(q_l)  # (n, ws_dim)
            ws_dim = q_tensor.shape[-1]
            if trajs.shape[-1] != ws_dim:
                trajs_ws = self.reference_robot.get_position(trajs)  # (N, T, ws_dim)
            else:
                trajs_ws = trajs
            n_constraints = q_tensor.shape[0]
            mask = torch.zeros((n_constraints, T), dtype=torch.bool, device=trajs_ws.device)
            for i, t_range in enumerate(t_range_l):
                t0, t1 = t_range
                mask[i, t0:t1] = True
            trajs_ws_exp = trajs_ws.unsqueeze(0).expand(n_constraints, N, T, ws_dim)  # (n, N, T, ws_dim)
            q_exp = q_tensor.view(n_constraints, 1, 1, ws_dim)  # (n, 1, 1, ws_dim)
            dists = torch.norm(trajs_ws_exp - q_exp, dim=-1)  # (n, N, T)
            radius_tensor = torch.tensor(radius_l, device=trajs_ws.device, dtype=trajs_ws.dtype).view(n_constraints, 1, 1)
            mask_exp = mask.unsqueeze(1).expand(n_constraints, N, T)  # (n, N, T)
            violation = (dists < radius_tensor) & mask_exp  # (n, N, T)
            violates_any = torch.any(torch.any(violation, dim=0), dim=1)  # [N]
            hard_mask = ~violates_any

        satisfying_trajs = trajs[hard_mask]
        K = satisfying_trajs.shape[0]
        if K == 0:
            return satisfying_trajs, torch.empty(0, device=device)

        all_constraints = hard_constraints + soft_constraints
        if len(all_constraints) == 0:
            total_costs = torch.zeros(K, device=device)
        else:
            q_l = []
            t_range_l = []
            radius_l = []
            for c in all_constraints:
                q_l.extend(list(c.q_l))
                t_range_l.extend(list(c.t_range_l))
                radius_l.extend(list(c.radius_l))
            q_tensor = torch.stack(q_l)  # (n, ws_dim)
            ws_dim = q_tensor.shape[-1]
            if satisfying_trajs.shape[-1] != ws_dim:
                trajs_ws = self.reference_robot.get_position(satisfying_trajs)  # (K, T, ws_dim)
            else:
                trajs_ws = satisfying_trajs
            n_constraints = q_tensor.shape[0]
            mask = torch.zeros((n_constraints, T), dtype=torch.bool, device=trajs_ws.device)
            for i, t_range in enumerate(t_range_l):
                t0, t1 = t_range
                mask[i, t0:t1] = True
            trajs_ws_exp = trajs_ws.unsqueeze(0).expand(n_constraints, K, T, ws_dim)  # (n, K, T, ws_dim)
            q_exp = q_tensor.view(n_constraints, 1, 1, ws_dim)  # (n, 1, 1, ws_dim)
            dists = torch.norm(trajs_ws_exp - q_exp, dim=-1)  # (n, K, T)
            radius_tensor = torch.tensor(radius_l, device=trajs_ws.device, dtype=trajs_ws.dtype).view(n_constraints, 1, 1)
            mask_exp = mask.unsqueeze(1).expand(n_constraints, K, T)  # (n, K, T)
            # Only penalize inside the radius
            inside = (dists < radius_tensor) & mask_exp
            cost = torch.where(inside, radius_tensor - dists, torch.zeros_like(dists))  # (n, K, T)

            total_costs = cost.sum(dim=(0, 2))  # [K]
        return satisfying_trajs, total_costs

    def get_conflicts(self, state: SearchState, start=0, end=32) -> List[Conflict]:
        """
        Find conflicts between paths.
        """
        # Get a list of best paths from the search state. Each path is shape (H, q_dim).
        best_path_l = [state.path_bl[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch in
                       enumerate(state.ix_best_path_in_batch_l)]

        # Pad the paths to make them all the same length. Different agents may have different start times, accommodate that.
        best_path_l = global_pad_paths(best_path_l, self.start_time_l)

        # Get the positions of all robots at all times.
        paths_pos_l = [self.reference_robot.get_position(path[start:end]) for path in best_path_l]
        # Return empty list if there are no paths.
        if len(paths_pos_l) == 0:
            return []
        max_t = max([len(path) for path in paths_pos_l])  # This should be the same for all agents.
        # Find conflicts.
        conflicts = []
        densification_factor = 2 if EdgeConflict in self.conflict_type_to_constraint_types.keys() else 1
        paths_pos_l_dense = densify_trajs(paths_pos_l, densification_factor)  # n elements of shape (H * densification_factor, q_dim)

        # Check collisions for all time steps at once.
        paths_pos_b_dense = torch.stack(paths_pos_l_dense)  # n_robots, H, q_dim
        # Change to (H, n_robots, q_dim) for the robot method.
        paths_pos_b_dense = paths_pos_b_dense.permute(1, 0, 2)
        collisions_pairwise_b, collision_points_b = self.reference_robot.check_rr_collisions(paths_pos_b_dense)  # Shapes (H, n_robots, n_robots) bool, (H, n_robots, n_robots, ws_dim) float
        # Check all time steps that have collisions in then.
        collision_indices = torch.nonzero(
            collisions_pairwise_b.int())  # Shape: (num_collisions, 3), each row [t_dense, agent_id_a, agent_id_b]
        for collision_index in collision_indices:
            t_dense, agent_id_a, agent_id_b = collision_index
            t_global_from = floor(t_dense / densification_factor)
            t_global_to = ceil(t_dense / densification_factor)
            t_dense, t_global_from, t_global_to = int(t_dense), int(t_global_from), int(t_global_to)
            # We set the conflict to happen at the beginning of the interpolated time interval,
            # but use the midpoint between interpolated states.
            midpoint_state_pos = collision_points_b[t_dense, agent_id_a, agent_id_b, :]
            agent_id_a, agent_id_b = int(agent_id_a.item()), int(agent_id_b.item())
            # Check which conflict types are requested.
            # Create a vertex conflict if the collision time is integral.
            if VertexConflict in self.conflict_type_to_constraint_types and t_global_from == t_global_to:
                conflicts.append(
                    VertexConflict([agent_id_a, agent_id_b],
                                   [
                                       paths_pos_l[agent_id_a][t_global_from],  # Corresponds to a graph vertex.
                                       paths_pos_l[agent_id_b][t_global_from]   # Corresponds to a graph vertex.
                                   ],
                                   int(t_global_from)))

            # Create an edge conflict if the collision time is not integral.
            if EdgeConflict in self.conflict_type_to_constraint_types and t_global_from != t_global_to:
                conflicts.append(
                    EdgeConflict([agent_id_a, agent_id_b],
                                 q_from_l=[
                                     paths_pos_l[agent_id_a][t_global_from],
                                     paths_pos_l[agent_id_b][t_global_from]
                                 ],
                                 q_to_l=[
                                     paths_pos_l[agent_id_a][t_global_to],
                                     paths_pos_l[agent_id_b][t_global_to]
                                 ],
                                 t_from=t_global_from,
                                 t_to=t_global_to))

            # Create a point conflict.
            # NOTE(yorai): this is normally called without a densification factor (=1), so the time from/to is the same.
            if PointConflict in self.conflict_type_to_constraint_types:
                conflicts.append(
                    PointConflict([agent_id_a, agent_id_b],
                                  p_l=[
                                        paths_pos_l_dense[agent_id_a][t_dense],
                                        paths_pos_l_dense[agent_id_b][t_dense]
                                  ],
                                  q_l=[
                                       midpoint_state_pos,
                                       midpoint_state_pos
                                  ],
                                  t_from=int(t_global_from),
                                  t_to=int(t_global_to)))
        return conflicts

    def plan_rolling_horizon(self, runtime_limit=1000, prev_best_traj=None):
        """
        Plan with a rolling horizon approach.
        """
        """
        Plan a path from start to goal with constraints.
        """
        startt = time.time()
        # This success status will be updated by the search.
        success_status = TrialSuccessStatus.UNKNOWN
        # ======================
        # Create the root node.
        # ======================
        # Plan individual paths without constraints.
        root_creation_start_time = time.time()

        # Set human trajectories for the rolling horizon window.
        cur_t = self.current_time
        start = min(self.current_time, 200)
        end = self.current_time + self.horizon

        # Empty root node.
        self.root = SearchState([], [], level=0)
        self.open_l = []
        for i in range(len(self.low_level_planner_l)):
            # If ECBS, then pass the paths of other agents to this one in the form of constraints.
            soft_constraint_l = []
            if self.is_ecbs:
                soft_constraint_l = self.create_soft_constraints_from_other_agents_paths(self.root, agent_id=i)

            # Set experience, if allowed.
            agent_experience = None
            if self.is_xcbs and len(self.root.path_bl) == len(self.low_level_planner_l):
                if self.experience_reuse_strategy == CBSExperienceReuseStrategy.XCBS:
                    best_path = self.root.path_bl[i][self.root.ix_best_path_in_batch_l[i]]
                    print(best_path.shape)
                    agent_experience = PathBatchExperience(best_path.repeat(64, 1, 1))
                    # agent_experience = PathBatchExperience(self.root.path_bl[i])
                else:
                    raise ValueError(f'Invalid experience reuse strategy {self.experience_reuse_strategy}.')

            urgency = 30 if torch.norm(self.goal_state_pos_l[i] - self.start_state_pos_l[i]) < 1.5 else 40
            planner_output = self.low_level_planner_l[i](
                self.start_state_pos_l[i], 
                self.goal_state_pos_l[i],
                cur_time=cur_t,
                urgency=urgency,
                constraints_l=soft_constraint_l,
            )

            # Check for planning failure in root creation.
            if planner_output.trajs_final_free_idxs.shape[0] == 0:
                print("Failed to find valid paths in root CT node.")
                success_status = TrialSuccessStatus.FAIL_NO_SOLUTION
                state = self.root
                break

            # Update the root node.
            ix_best_traj = planner_output.idx_best_traj
            self.root.path_bl.append(planner_output.trajs_final)
            self.root.ix_best_path_in_batch_l.append(ix_best_traj)

            # Check for runtime limit reached in root creation.
            if time.time() - startt > runtime_limit:
                print('Runtime limit reached in root creation.')
                success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
                state = self.root
                break

        if success_status == TrialSuccessStatus.UNKNOWN:
            self.root.update_g_l2()
            conflict_l = self.get_conflicts(self.root)
            self.root.conflict_l = conflict_l
            # Create the open list.
            self.open_l.append(self.root)
            print(f'Root creation time: {time.time() - root_creation_start_time:.2f}s')

        # ======================
        # Start the search.
        # ======================
        num_ct_expansions = 0
        while success_status == TrialSuccessStatus.UNKNOWN:
            # If the open list is empty, return None.
            if not self.open_l:
                print('Open list is empty. NO SOLUTION.')
                success_status = TrialSuccessStatus.FAIL_NO_SOLUTION
                break

            # Sort the CT.
            state = None
            if len(self.open_l) == 1:
                pass  # Only one node, no need to sort.
            elif self.optimize_for == "cost":
                self.open_l.sort(key=lambda x: x.g)
            elif self.optimize_for == "conflicts":
                self.open_l.sort(key=lambda x: len(x.conflict_l))
            else:
                raise ValueError(f'Invalid optimize_for value: {self.optimize_for}.')
            
            # Get the first node from the open list.
            state = self.open_l.pop(0)
            #state.conflict_l = []
            # Check if the conflict set is empty.
            if not state.conflict_l:
                print('No conflicts found in CT node. GOAL.')
                success_status = TrialSuccessStatus.SUCCESS
                break

            # Expand the search tree.
            self.expand_rolling_horizon(state, prev_best_traj=None)
            num_ct_expansions += 1
            if time.time() - startt > runtime_limit:
                print('Runtime limit reached.')
                success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
                break
        
        # Return the best paths. Smoothed and padded to the same length accounting for start times.
        self.root = state
        best_path_l = [state.path_bl[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch in
                       enumerate(state.ix_best_path_in_batch_l)]
        path_index_l = state.ix_best_path_in_batch_l.copy()

        best_path_l = global_pad_paths(best_path_l, self.start_time_l)
        
        # Return the best paths, the number of CT expansions, success status, and number of collisions in the solution.
        return best_path_l, num_ct_expansions, success_status, len(state.conflict_l), path_index_l

    def plan_rolling_horizon_parallel(self, runtime_limit=1000, prev_best_traj=None):
        """
        Plan with a rolling horizon approach.
        """
        """
        Plan a path from start to goal with constraints.
        """
        startt = time.time()
        # This success status will be updated by the search.
        success_status = TrialSuccessStatus.UNKNOWN
        # ======================
        # Create the root node.
        # ======================
        # Plan individual paths without constraints.
        root_creation_start_time = time.time()

        # Set human trajectories for the rolling horizon window.
        cur_t = self.current_time
        start = min(self.current_time, 200)
        end = self.current_time + self.horizon

        # Empty root node.
        self.root = SearchState([], [], level=0)
        self.open_l = []
        urgencies = [30 if torch.norm(self.goal_state_pos_l[i] - self.start_state_pos_l[i]) < 1.5 else 40 for i in range(self.num_agents)]
        agent_outputs = self.low_level_planner_l[0].plan_parallel(
            self.start_state_pos_l, 
            self.goal_state_pos_l, 
            cur_time=cur_t,
            urgencies=urgencies,
        )

        for i in range(len(agent_outputs)):
            self.root.path_bl.append(agent_outputs[i].trajs_final)
            self.root.ix_best_path_in_batch_l.append(agent_outputs[i].idx_best_traj)

        # Check for runtime limit reached in root creation.
        if time.time() - startt > runtime_limit:
            print('Runtime limit reached in root creation.')
            success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
            state = self.root
            exit()

        if success_status == TrialSuccessStatus.UNKNOWN:
            self.root.update_g_l2()
            conflict_l = self.get_conflicts(self.root)
            self.root.conflict_l = conflict_l
            # Create the open list.
            self.open_l.append(self.root)
            print(f'Root creation time: {time.time() - root_creation_start_time:.2f}s')

        # ======================
        # Start the search.
        # ======================
        num_ct_expansions = 0
        while success_status == TrialSuccessStatus.UNKNOWN:
            # If the open list is empty, return None.
            if not self.open_l:
                print('Open list is empty. NO SOLUTION.')
                success_status = TrialSuccessStatus.FAIL_NO_SOLUTION
                break

            # Sort the CT.
            state = None
            if len(self.open_l) == 1:
                pass  # Only one node, no need to sort.
            elif self.optimize_for == "cost":
                self.open_l.sort(key=lambda x: x.g)
            elif self.optimize_for == "conflicts":
                self.open_l.sort(key=lambda x: len(x.conflict_l))
            else:
                raise ValueError(f'Invalid optimize_for value: {self.optimize_for}.')
            
            # Get the first node from the open list.
            state = self.open_l.pop(0)
            #state.conflict_l = []
            # Check if the conflict set is empty.
            if not state.conflict_l:
                print('No conflicts found in CT node. GOAL.')
                success_status = TrialSuccessStatus.SUCCESS
                break

            # Expand the search tree.
            self.expand_rolling_horizon(state, prev_best_traj=None)
            num_ct_expansions += 1
            if time.time() - startt > runtime_limit:
                print('Runtime limit reached.')
                success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
                break
        
        # Return the best paths. Smoothed and padded to the same length accounting for start times.
        self.root = state
        best_path_l = [state.path_bl[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch in
                       enumerate(state.ix_best_path_in_batch_l)]
        path_index_l = state.ix_best_path_in_batch_l.copy()

        best_path_l = global_pad_paths(best_path_l, self.start_time_l)
        
        endt = time.time()
        print(f'Rolling horizon CBS planning iteration {cur_t * self.execution_horizon} time: {endt - startt:.2f}s')
        print(f'Number of CT expansions: {num_ct_expansions}')

        # Return the best paths, the number of CT expansions, success status, and number of collisions in the solution.
        return best_path_l, num_ct_expansions, success_status, len(state.conflict_l), path_index_l

    def expand_rolling_horizon(self, state: SearchState, prev_best_traj=None):
        """
        Expand the search tree for rolling horizon planning.
        """
       # Choose a conflict to turn into constraints.
        conflict = state.conflict_l[0]
        constraints = convert_conflicts_to_constraints(conflict, self.conflict_type_to_constraint_types)
        cur_t = self.current_time
        # Create a CT node for each constraint tuple (agent_id, constraint).
        for agent_id, constraint in constraints:
            # Update the constraint time to account for agents starting after t=0.
            constraint.t_range_l = [(t_range[0] - self.start_time_l[agent_id],
                                     t_range[1] - self.start_time_l[agent_id])
                                    for t_range in constraint.t_range_l]
            # Clamp down to the maximum time of the paths.
            # constraint.t_range_l = [(max(0, min(t_range[0], len(state.path_bl[agent_id][0]) - 1)),
            #                          min(len(state.path_bl[agent_id][0]) - 1, t_range[1]))
            #                         for t_range in constraint.t_range_l]
            # constraint.t_range_l = [(max(cur_t, min(t_range[0], len(state.path_bl[agent_id][0]) - 1)),
            #                          min(cur_t + self.horizon, t_range[1]))
            #                         for t_range in constraint.t_range_l]
            # Create new state.
            new_state = state.get_copy()
            # Add the constraint.
            new_state.add_constraint(agent_id, constraint)  # In local time to agent.
            # Create a PlanningContext object for the agent which includes the current trajectories of all other agents.
            # Plan the path with the constraint. Add the previous path as experience as a seed if allowed.
            agent_constraint_l = new_state.constraints[agent_id].copy()

            # Set soft constraints from paths of other agents, if allowed.
            if self.is_ecbs:
                soft_constraint_l = self.create_soft_constraints_from_other_agents_paths(new_state, agent_id)
                agent_constraint_l.extend(soft_constraint_l)

            # Set experience, if allowed.
            agent_experience = None
            if self.is_xcbs:
                if self.experience_reuse_strategy == CBSExperienceReuseStrategy.XCBS:
                    agent_experience = PathBatchExperience(new_state.path_bl[agent_id])
                else:
                    raise ValueError(f'Invalid experience reuse strategy {self.experience_reuse_strategy}.')
            
            print('Using low level planner for agent', agent_id)
            planner_output = self.low_level_planner_l[agent_id](
                self.start_state_pos_l[agent_id], 
                self.goal_state_pos_l[agent_id],
                constraints_l=agent_constraint_l,
                cur_time=cur_t,
                urgency=100, 
                experience=agent_experience,
            )

            # Check if the planner found a valid path.
            if len(planner_output.trajs_final_free_idxs) == 0:
                print(RED + 'Failed to find valid path in CT node.' + RESET)
                continue  # Skip this node.

            new_state.path_bl[agent_id] = planner_output.trajs_final

            start_time = time.time()
            if self.low_level_choose_path_from_batch_strategy == 'least_cost':
                ix_best_traj = planner_output.idx_best_traj
                new_state.ix_best_path_in_batch_l[agent_id] = ix_best_traj
                # Find and set conflicts.
                conflict_l = self.get_conflicts(new_state)
                new_state.conflict_l = conflict_l


            elif self.low_level_choose_path_from_batch_strategy == 'least_collisions':
                # Batched conflict checking for efficiency.
                valid_idxs = planner_output.trajs_final_free_idxs
                min_conflict_count = float('inf')
                for ix_traj in valid_idxs:
                        conflict_count = self.count_conflicts_with_replacement(new_state, agent_id, new_state.path_bl[agent_id][ix_traj])
                        if conflict_count < min_conflict_count:
                            min_conflict_count = conflict_count
                            new_state.ix_best_path_in_batch_l[agent_id] = ix_traj
                            print(f'Found a path with fewer conflicts: {min_conflict_count}')
                
                # Find and set conflicts.
                conflict_l = self.get_conflicts(new_state)
                new_state.conflict_l = conflict_l
            else:
                raise ValueError('Invalid low level choose-path-from-batch strategy.')
            print(f"Choose path from batch took {time.time() - start_time:.4f} seconds.")

            # Update the cost to reach this node.
            new_state.update_g_l2()
            print("New state cost num conflicts:", new_state.g, len(new_state.conflict_l), "\n\n\n\n")

            # Check if conflicts in state contain the two agents in the conflict.
            resolved = True
            for conflict in new_state.conflict_l:
                if conflict.agent_ids[0] == agent_id or conflict.agent_ids[1] == agent_id:
                    resolved = False
                    break
            
            # Add the new state to the open list.
            self.open_l.append(new_state)

    def create_soft_constraints_from_other_agents_paths(self,
                                                        state: SearchState,
                                                        agent_id: int) -> List[MultiPointConstraint]:
        """
        Create soft constraints from the paths of other agents.
        """
        if len(state.path_bl) == 0:
            return []

        agent_constraint_l = []  # The output list of soft constraints.
        q_l = []
        t_range_l = []
        radius_l = []
        num_agents_in_state = len(state.path_bl)
        for agent_id_other in range(num_agents_in_state):
            if agent_id_other != agent_id:
                best_path_other_agent = \
                    state.path_bl[agent_id_other][state.ix_best_path_in_batch_l[agent_id_other]].squeeze(0)
                best_path_pos_other_agent = self.reference_robot.get_position(best_path_other_agent)
                for t_other_agent in range(0, len(best_path_other_agent), 1):
                    t_agent = t_other_agent + self.start_time_l[agent_id_other] - self.start_time_l[agent_id]
                    # The last timestep index for this agent is the length of its path - 1.
                    # If it does not have a path stored, then create constraints for all timesteps
                    # in the path of the other agent (starting from zero).
                    T_agent = len(state.path_bl[agent_id_other][0]) - 1
                    if agent_id >= len(state.path_bl):
                        T_agent = len(best_path_other_agent) - 1
                    else:
                        T_agent = len(state.path_bl[agent_id][0]) - 1

                    if 1 <= t_agent <= T_agent:
                        q_l.append(best_path_pos_other_agent[t_other_agent])
                        t_range_l.append((t_agent, t_agent + 1))
                        radius_l.append(params.vertex_constraint_radius)

        if len(q_l) > 0:
            soft_constraint = MultiPointConstraint(q_l=q_l, t_range_l=t_range_l)
            soft_constraint.radius_l = radius_l
            soft_constraint.is_soft = True
            agent_constraint_l.append(soft_constraint)
        return agent_constraint_l
    
    def run_continuous_plan(self, runtime_limit: float = 20.0, max_iters=11, tol=0.5):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        executed_segments = [[] for _ in range(len(self.start_state_pos_l))]
        reached_goal = [False] * len(self.start_state_pos_l)
        ct_expansions = 0
        n_conflicts = 0
        rolling_horizon_times = []
        
        # Configurations
        H = self.horizon          # Plan Horizon
        K = self.execution_horizon          # Execution Horizon (Steps to execute before replanning)
        cur_t = 0
        
        # "Warm Start" buffer: stores the remainder of the previous plan
        # to initialize the next plan.
        next_warm_start = None 

        self.current_time = 0
        it = 0
        while not all(reached_goal) and it < max_iters:
            start_time = time.time()

            print(f'Planner iteration {it}')

            best_path_l, num_ct_expansions, status, n_conflicts, path_index_l = \
                self.plan_rolling_horizon_parallel(runtime_limit=runtime_limit, prev_best_traj=None)
            ct_expansions += num_ct_expansions
            n_conflicts += n_conflicts

            if status != TrialSuccessStatus.SUCCESS:
                print(f'Planner failed to find solution at it={it}')
                break

            for i, best_traj in enumerate(best_path_l):
                steps_to_exec = min(K, best_traj.shape[0])
                
                exec_segment = best_traj[:steps_to_exec, :2].clone()
                executed_segments[i].append(exec_segment.cpu())
                
                current_start = best_traj[steps_to_exec]
                self.start_state_pos_l[i] = current_start.clone()

                dist_to_goal = torch.norm(self.start_state_pos_l[i][..., :2] - self.goal_state_pos_l[i][..., :2])
                if dist_to_goal < tol:
                    reached_goal[i] = True

            self.current_time += K
            it += 1
            rh_time = time.time() - start_time
            print(f'Continuous Planner iteration {it} took {rh_time} seconds with {num_ct_expansions} CT expansions')
            rolling_horizon_times.append(rh_time)

        # Reconstruct full path
        executed_trajs = []
        for i in range(len(executed_segments)):
            executed_trajs.append(torch.cat(executed_segments[i], dim=0).to(device))

        total_runtimes = it * K * self.time_step
        return executed_trajs, ct_expansions, status, n_conflicts, total_runtimes, rolling_horizon_times
