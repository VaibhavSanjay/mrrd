"""
MIT License

Copyright (c) 2026 Vaibhav Sanjay

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

import torch
import os
import einops

from torch_robotics.trajectory.utils import interpolate_traj_via_points

class HumanTrajectories:
    def __init__(self, file_path, t_start=0, t_end=-1, device='cuda', sample_idx=110, center_around_origin=True, max_humans=None):
        # Load the human trajectories.
        batch_neigh = torch.load(file_path, weights_only=True)
        
        # Take the specific sample index.
        # Format is [num_scenarios, 1, num_humans, time_steps, dims]
        state_neigh = batch_neigh[sample_idx]
        human_trajs = state_neigh[0]
        
        # Slice the trajectories from t_start to t_end.
        self.human_trajs = human_trajs[:, t_start:t_end, :2]
        self.human_trajs_vel = human_trajs[:, t_start:t_end, 2:4]
        
        self.human_trajs = self.human_trajs.to(device)
        self.human_trajs_vel = self.human_trajs_vel.to(device)
        self.t_start = t_start
        self.t_end = t_end
        self.device = device

        if center_around_origin:
            self.human_trajs = self.human_trajs - self.human_trajs.mean(dim=(0, 1), keepdim=True)

        if max_humans is not None:
            self.human_trajs = self.human_trajs[:max_humans, :, :]
            self.human_trajs_vel = self.human_trajs_vel[:max_humans, :, :]

    def get_human_trajectories(self, return_vel=False):
        if return_vel:
            return torch.cat([self.human_trajs, self.human_trajs_vel], dim=-1)
        return self.human_trajs
        
    def predict_trajectories(self, cur_t, horizon, return_vel=False):
        if return_vel:
            return torch.cat([self.human_trajs[:, cur_t:cur_t+horizon, :], self.human_trajs_vel[:, cur_t:cur_t+horizon, :]], dim=-1)
        return self.human_trajs[:, cur_t:cur_t+horizon, :]

    def filter_trajs_against_humans(self, trajs, human_radius: float, robot_radius: float, 
                                    human_trajs=None, robot=None, return_indices=False, num_interpolation=5):
        device = trajs.device
        if human_trajs is None:
            human_trajs = self.human_trajs

        # Robot position (B, H_interp, 2)
        trajs_interp = interpolate_traj_via_points(trajs, num_interpolation=num_interpolation)
        robot_obj = robot or getattr(self, 'robot', None)
        if robot_obj is not None and trajs_interp.shape[-1] != 2:
            try:
                robot_pos = robot_obj.get_position(trajs_interp)
            except Exception:
                robot_pos = trajs_interp[..., :2]
        else:
            robot_pos = trajs_interp[..., :2]

        # Human position (Nh, H_interp, 2)
        human_trajs_t = torch.as_tensor(human_trajs, device=device)
        human_interp = interpolate_traj_via_points(human_trajs_t, num_interpolation=num_interpolation)

        # Match time horizons if different
        H = min(robot_pos.shape[1], human_interp.shape[1])
        robot_pos = robot_pos[:, :H, :]
        human_interp = human_interp[:, :H, :]

        # Collision check: distance <= (human_radius + robot_radius)
        diff = robot_pos.unsqueeze(1) - human_interp.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1) # (B, Nh, H)
        threshold_sq = (float(human_radius) + float(robot_radius)) ** 2

        trajs_waypoints_collisions = (dist_sq <= threshold_sq).any(dim=1) # (B, H)
        is_colliding = trajs_waypoints_collisions.any(dim=-1)             # (B,)
        is_free_mask = ~is_colliding

        # Extract free and colliding trajectories
        trajs_free = trajs[is_free_mask] if is_free_mask.any() else None
        trajs_coll = trajs[is_colliding] if is_colliding.any() else None
        trajs_free_idxs = torch.argwhere(is_free_mask)
        trajs_coll_idxs = torch.argwhere(is_colliding)

        if return_indices:
            return trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs, trajs_waypoints_collisions
        return trajs_coll, trajs_free

    def filter_paths_by_obstacle_intersection(self, task):
        _, _, trajs_free, trajs_free_idxs, _ = task.get_trajs_collision_and_free(
            self.human_trajs, return_indices=True
        )
        if trajs_free is not None:
            free_idxs = trajs_free_idxs.squeeze()
            self.human_trajs = self.human_trajs[free_idxs]
            self.human_trajs_vel = self.human_trajs_vel[free_idxs]
        else:
            print("Warning: All human paths collide with obstacles. Keeping all paths.")