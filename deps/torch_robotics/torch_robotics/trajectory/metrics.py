import numpy as np
import torch

from torch_robotics.torch_utils.torch_utils import to_numpy


def compute_path_length(trajs, robot):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = robot.get_position(trajs)
    return compute_path_length_from_pos(trajs_pos)


def compute_path_length_from_pos(trajs_pos):
    assert trajs_pos.ndim == 3
    path_length = torch.linalg.norm(torch.diff(trajs_pos, dim=-2), dim=-1).sum(-1)
    return path_length

def compute_variance_waypoints(trajs, robot):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = robot.get_position(trajs)
    parwise_distance_between_points_waypoints = torch.cdist(trajs_pos, trajs_pos, p=2)

    sum_var_waypoints = 0.
    for via_points in trajs_pos.permute(1, 0, 2):  # horizon, batch, position
        parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
        distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
        sum_var_waypoints += torch.var(distances)
    return sum_var_waypoints


def compute_smoothness(trajs, robot, trajs_vel=None):
    if trajs_vel is None:
        assert trajs.ndim == 3
        trajs_vel = robot.get_velocity(trajs)
    else:
        assert trajs_vel.ndim == 3
    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1)
    smoothness = smoothness.sum(-1)  # sum over trajectory horizon
    return smoothness


def compute_average_acceleration(trajs, robot, trajs_vel=None):
    if trajs_vel is None:
        assert trajs.ndim == 3
        trajs_vel = robot.get_velocity(trajs)
    else:
        assert trajs_vel.ndim == 3

    return compute_average_acceleration_from_pos_vel(trajs, trajs_vel)


def compute_average_acceleration_from_pos_vel(trajs_pos, trajs_vel):
    assert trajs_pos.ndim == 3
    assert trajs_vel.ndim == 3

    # Compute the change in velocity (acceleration).
    accelerations = torch.diff(trajs_vel, dim=-2)

    # Compute the norm of the accelerations (magnitude of acceleration)
    acceleration_magnitudes = torch.linalg.norm(accelerations, dim=-1)

    # Average the acceleration magnitudes over the trajectory horizon (time steps)
    average_acceleration = acceleration_magnitudes.mean(-1)

    return average_acceleration

def compute_human_cost(trajs, human_trajs, robot, radius: float = 0.15):
    if human_trajs is None:
        return torch.zeros(trajs.shape[0], device=trajs.device, dtype=trajs.dtype)

    device = trajs.device
    dtype = trajs.dtype
    B = trajs.shape[0]

    # robot positions [B, H_robot, 2]
    robot_pos = robot.get_position(trajs)

    # human_trajs -> tensor on device
    hp = torch.as_tensor(human_trajs, device=device, dtype=dtype)
    # Normalize to [B, N, H_human, 2]
    hp = hp.unsqueeze(0).repeat(B, 1, 1, 1)

    _, N_h, H_human, _ = hp.shape
    _, _, H_robot, _ = robot_pos.unsqueeze(1).shape

    # Initialize distances with large value for non-overlapping timesteps
    large_val = 1e6
    dists = torch.full((B, N_h, H_robot), large_val, device=device, dtype=dtype)

    # Compute distances for overlapping timesteps
    common_h = min(H_robot, H_human)
    if common_h > 0:
        rp = robot_pos.unsqueeze(1)  # [B,1,H_robot,2]
        dists[:, :, :common_h] = torch.norm(rp[:, :, :common_h, :] - hp[:, :, :common_h, :], dim=-1)

    # barrier h = dist^2 - r^2 (negative => violation)
    h = dists ** 2 - (radius ** 2)
    violations = torch.where(h < 0, h, torch.zeros_like(h))  # negative or zero

    # sum over humans and time, negate to produce positive cost
    cost_per_batch = - violations.sum(dim=(1, 2))
    return cost_per_batch

def compute_max_jerk_metric(trajs, robot):
    assert trajs.ndim == 3
    trajs_pos = robot.get_position(trajs)
    
    # Calculate 3rd derivative
    vel = torch.diff(trajs_pos, dim=-2)
    acc = torch.diff(vel, dim=-2)
    jerk = torch.diff(acc, dim=-2)
    
    # Get the magnitude of the jerk vectors
    jerk_magnitudes = torch.linalg.norm(jerk, dim=-1)
    
    # Return the maximum jerk encountered in each trajectory
    max_jerk = jerk_magnitudes.max(dim=-1).values
    
    return max_jerk


def compute_goal_reaching(trajs, goal_positions):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    last_points = trajs[:, -1, :]  # [B, D]
    if goal_positions.ndim == 1:
        goal_positions = goal_positions.unsqueeze(0).expand_as(last_points)
    distances = torch.linalg.norm(last_points - goal_positions, dim=-1)  # [B]
    return distances
