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
# General imports.
import numpy as np
import torch
from matplotlib import pyplot as plt
from typing import List
# Project imports.
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from mrrd.common.trajectory_utils import densify_trajs


class EnvSpheres2DDense(EnvBase):

    def __init__(self,
                 name='EnvSpheres2DDense',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs
                 ):

        obj_list = [
            # MultiSphereField(
            #     np.array([[13.92, 3.48]]),  # (n, 2) array of sphere centers.
            #     np.array([7.2]),  # (n, ) array of sphere radii.
            #     tensor_args=tensor_args
            # ),
            MultiSphereField(
                np.array([
                    # Original 15 spheres
                    [8, -3],
                    [4, 5],
                    [-3, 0],
                    [-8, 7],
                    [6, -8],
                    [-5, -6],
                    [9, 2],
                    [-2, 8],
                    [-9, -2],
                    [1, -9],
                    [7, 7],
                    [-6, 4],
                    [-3, -8],
                    [4, -4],
                    [-8, -8],
                    # 15 new spheres
                    [2, 2],
                    [-1, -4],
                    [5, -1],
                    [-7, 1],
                    [0, 6],
                    [3, -7],
                    [-4, 9],
                    [8, 5],
                    [-9, 5],
                    [6, 0],
                    [-2, -9],
                    [9, -7],
                    [-6, -3],
                    [1, 4],
                    [-5, 3],
                ]),  # (n, 2) array of sphere centers.
                np.array([
                    # Original 15 radii
                    0.5,
                    0.75,
                    0.75,
                    0.8,
                    0.6,
                    0.9,
                    0.7,
                    0.5,
                    1.0,
                    0.6,
                    0.8,
                    0.7,
                    0.9,
                    0.6,
                    0.8,
                    # 15 new radii
                    0.65,
                    0.8,
                    0.55,
                    0.9,
                    0.7,
                    0.6,
                    0.75,
                    0.85,
                    0.7,
                    0.5,
                    0.65,
                    0.8,
                    0.95,
                    0.6,
                    0.7,
                ]),  # (n, ) array of sphere radii.
                tensor_args=tensor_args
            ),
            # MultiBoxField(
            #     np.array([
            #         [0, 0],
            #         [0, 0.35 * 20],
            #         [0, -0.35 * 20]
            #     ]),
            #     np.array([
            #         [0.8 * 20, 0.1 * 20],
            #         [1.0 * 20, 0.1 * 20],
            #         [1.0 * 20, 0.1 * 20]
            #     ]),
            #     tensor_args=tensor_args
            # ),
        ]
        # np.array([
        #     [0, 0.0],
        #     [0., 0.875],
        #     [0., -0.875],
        #     [0.875, 0.0],
        #     [-0.875, 0.0]
        # ]),
        # np.array([
        super().__init__(
            name=name,
            limits=torch.tensor([[-10, -10], [10, 10]], **tensor_args),  # Environments limits.
            obj_fixed_list=[ObjectField(obj_list, 'simple2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def is_start_goal_valid_for_data_gen(self, robot, start_pos, goal_pos):
        """
        :param robot: Robot object.
        :param start_pos: Start position. (q_dim,), often (2, ).
        :param goal_pos: Goal position. (q_dim,), often (2, ).
        """
        if torch.linalg.norm(start_pos - goal_pos) > 0.6:
            return False
        # We set squares for starts and goals. Starts can only be at start squares and goals follow similarly.
        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            start_region_centers = torch.tensor([
                [0.8, 0.5],
                [-0.5, 0.8],
                [-0.8, -0.5],
                [0.5, -0.8]
            ], **self.tensor_args)
            goal_region_centers = torch.tensor([
                [0.8, -0.5],
                [0.5, 0.8],
                [-0.8, 0.5],
                [-0.5, -0.8]
            ], **self.tensor_args)
            start_region_radius = 0.15
            goal_region_radius = 0.15
            if torch.any(torch.norm(start_region_centers - start_pos, dim=-1) < start_region_radius).item() and \
               torch.any(torch.norm(goal_region_centers - goal_pos, dim=-1) < goal_region_radius).item():
                return True
            else:
                return False

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.05,
            n_pre_samples=50000,
            max_time=50
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params

        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params

        else:
            raise NotImplementedError

    def get_skill_pos_seq_l(self, robot=None, start_pos=None, goal_pos=None) -> List[torch.Tensor]:
        return None
        # from torch_robotics.robots import *
        # if isinstance(robot, RobotPlanarDisk):
        #     return [
        #         torch.tensor([[0.0, 0.0]]*35, **self.tensor_args),  # Pause at (0, 0) for a bit.
        #     ]
        # else:
        #     raise NotImplementedError

    def compute_traj_data_adherence(self, path: torch.Tensor,
                                    fraction_of_length=0.1) -> torch.Tensor:
        # The score is deviation of the path from a straight line. Cost in {0, 1}.
        # The score is 1 for each point on the path within a distance less than fraction_of_length * length from
        # the straight line. The computation is the average of the scores for all points in the path.
        start_state_pos = path[0][:2]
        goal_state_pos = path[-1][:2]
        length = torch.norm(goal_state_pos - start_state_pos)
        path = path[:, :2]
        path = torch.stack([path[:, 0], path[:, 1], torch.zeros_like(path[:, 0])], dim=1)
        start_state_pos = torch.stack([start_state_pos[0], start_state_pos[1], torch.zeros_like(start_state_pos[0])]).unsqueeze(0)
        goal_state_pos = torch.stack([goal_state_pos[0], goal_state_pos[1], torch.zeros_like(goal_state_pos[0])]).unsqueeze(0)
        deviation_from_line = torch.norm(torch.cross(goal_state_pos - start_state_pos, path - start_state_pos),
                                         dim=1) / length
        return (deviation_from_line < fraction_of_length).float().mean().item()

if __name__ == '__main__':
    env = EnvSpheres2DDense(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    # remove axis, labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig("env_spheres_2d_dense.png")

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
