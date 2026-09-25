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
import torch
from scipy.signal import savgol_filter
# Project imports.
from mrrd.config.mrrd_params import MRRDParams as params

from typing import Literal

import torch
import torch.nn.functional as F


def _float_factorial(n: int) -> float:
    """Compute factorial as float."""
    val = 1.0
    for i in range(2, int(n) + 1):
        val *= float(i)
    return val


def savgol_coeffs(
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    pos: int = None,
    use: Literal["conv", "dot"] = "conv",
    *,
    device: torch.device = None,
) -> torch.Tensor:
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if pos is None:
        if rem == 0:
            pos = halflen - 0.5
        else:
            pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than window_length.")

    if use not in ["conv", "dot"]:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        return torch.zeros(window_length, device=device)

    # Form the design matrix A
    x = torch.arange(-pos, window_length - pos, dtype=torch.float64, device=device)

    if use == "conv":
        # Reverse so that result can be used in a convolution
        x = x.flip(0)

    order = torch.arange(polyorder + 1, dtype=torch.float64, device=device).reshape(
        -1, 1
    )
    A = x**order

    # y determines which order derivative is returned
    y = torch.zeros(polyorder + 1, dtype=torch.float64, device=device)
    # The coefficient assigned to y[deriv] scales the result
    y[deriv] = _float_factorial(deriv) / (delta**deriv)

    # Find the least-squares solution of A*c = y
    coeffs, _, _, _ = torch.linalg.lstsq(A, y, rcond=None)

    return coeffs.to(torch.float32)


def _polyder(p: torch.Tensor, m: int) -> torch.Tensor:
    """Differentiate polynomials represented with coefficients."""
    if m == 0:
        return p

    n = len(p)
    if n <= m:
        return torch.zeros_like(p[:1, ...])

    dp = p[:-m].clone()
    for k in range(m):
        rng = torch.arange(n - k - 1, m - k - 1, -1, dtype=dp.dtype, device=dp.device)
        dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
    return dp


def _polyfit(x: torch.Tensor, y: torch.Tensor, deg: int) -> torch.Tensor:
    """Polynomial fit using least squares."""
    # Build Vandermonde matrix (highest power first)
    vander = torch.stack([x ** (deg - i) for i in range(deg + 1)], dim=0)
    # Solve least squares
    coeffs, _, _, _ = torch.linalg.lstsq(vander.T, y, rcond=None)
    return coeffs


def _axis_slice(
    x: torch.Tensor, start: int, stop: int, axis: int
) -> torch.Tensor:
    """Get a slice along a specific axis."""
    indices = torch.arange(start, stop, device=x.device)
    return torch.index_select(x, axis, indices)


def _fit_edge(
    x: torch.Tensor,
    window_start: int,
    window_stop: int,
    interp_start: int,
    interp_stop: int,
    axis: int,
    polyorder: int,
    deriv: int,
    delta: float,
    y: torch.Tensor,
) -> None:
    """Fit polynomial to edge and interpolate in-place."""
    # Get the edge into a (window_length, -1) array
    x_edge = _axis_slice(x, window_start, window_stop, axis)
    if axis == 0 or axis == -x.ndim:
        xx_edge = x_edge
    else:
        xx_edge = x_edge.swapaxes(axis, 0)
    xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)

    # Fit the edges
    poly_coeffs = _polyfit(
        torch.arange(0, window_stop - window_start, dtype=x.dtype, device=x.device),
        xx_edge,
        polyorder,
    )

    if deriv > 0:
        poly_coeffs = _polyder(poly_coeffs, deriv)


def _fit_edges_polyfit(
    x: torch.Tensor,
    window_length: int,
    polyorder: int,
    deriv: int,
    delta: float,
    axis: int,
    y: torch.Tensor,
) -> None:
    """Fit edges using polynomial interpolation."""
    halflen = window_length // 2
    _fit_edge(x, 0, window_length, 0, halflen, axis, polyorder, deriv, delta, y)
    n = x.shape[axis]
    _fit_edge(
        x, n - window_length, n, n - halflen, n, axis, polyorder, deriv, delta, y
    )


def savgol(
    x: torch.Tensor,
    window_length: int,
    polyorder: int,
    deriv: int = 0,
    delta: float = 1.0,
    axis: int = -1,
    mode: Literal["mirror", "constant", "nearest", "interp", "reflect", "replicate"] = "interp",
    cval: float = 0.0,
) -> torch.Tensor:
    r"""Computes a Savitzky-Golay filter entirely on GPU using conv1d.

    Args:
        x (Tensor): Input tensor.
        window_length (int): Length of the filter window (must be a positive odd integer).
        polyorder (int): Order of the polynomial used to fit the samples.
        deriv (int, optional): Order of the derivative to compute. Default: 0
        delta (float, optional): Sample spacing. Default: 1.0
        axis (int, optional): Axis along which to apply the filter. Default: -1
        mode (str, optional): Padding mode. Default: 'replicate'
        cval (float, optional): Value to fill for 'constant' mode. Default: 0.0

    Returns:
        Tensor: The filtered tensor.
    """
    x = x.clone()
    if x.dtype not in [torch.float32, torch.float64]:
        x = x.to(torch.float32)

    coeffs = savgol_coeffs(
        window_length, polyorder, deriv=deriv, delta=delta, device=x.device
    )

    # Move the target axis to the last position for conv1d
    x_moved = x.movedim(axis, -1)
    original_shape = x_moved.shape
    seq_len = original_shape[-1]

    if window_length > seq_len:
        raise ValueError(
            f"window_length ({window_length}) must be <= the size of the "
            f"axis being filtered ({seq_len})."
        )

    # Reshape to (batch, 1, length) for conv1d
    x_reshaped = x_moved.reshape(-1, 1, seq_len)

    # Pad so that conv1d output length == input length.
    # We need: left_pad + right_pad = window_length - 1
    left_pad = (window_length - 1) // 2
    right_pad = window_length // 2
    mode_map = {"mirror": "reflect", "nearest": "replicate", "interp": "replicate",
                "constant": "constant"}
    pad_mode = mode_map.get(mode, mode)

    if pad_mode == "constant":
        x_padded = F.pad(x_reshaped, (left_pad, right_pad), mode=pad_mode, value=cval)
    else:
        x_padded = F.pad(x_reshaped, (left_pad, right_pad), mode=pad_mode)

    # Convolve: kernel shape (out_channels=1, in_channels=1, kernel_size)
    coeffs_kernel = coeffs.reshape(1, 1, -1)
    y_reshaped = F.conv1d(x_padded, coeffs_kernel)

    # Reshape back to original shape and restore axis order
    y = y_reshaped.reshape(original_shape)
    y = y.movedim(-1, axis)

    return y


def smooth_trajs(trajs, window_size=9, poly_order=2):
    """
    Smooth the trajectories.
    trajs: List of trajectories. Each is a tensor of shape (H, q_dim).
    """
    if isinstance(trajs, torch.Tensor):
        assert trajs.dim() == 3  # (B, H, q_dim)
        smoothed_trajs = savgol(trajs, window_size, poly_order, axis=1)
        return smoothed_trajs

    smoothed_trajs = []
    for traj in trajs:
        traj = traj.cpu().numpy()
        window_size_traj = min(window_size, traj.shape[0])
        if window_size_traj <= 2:
            smoothed_trajs.append(torch.tensor(traj).to(**params.tensor_args))
            continue
        smoothed_traj = savgol_filter(traj, window_size_traj, poly_order, axis=0)
        smoothed_traj = torch.tensor(smoothed_traj).to('cuda')
        smoothed_trajs.append(smoothed_traj)
    return smoothed_trajs


def densify_trajs(trajs, n_points_interp=10):
    """
    Densify the trajectories.
    :param trajs: List of trajectories. Each is a tensor of shape (H, n_dims).
    :param n_points_interp: Number of points to interpolate between each pair of points.
    """
    densified_trajs = []
    for traj in trajs:
        densified_traj = []
        for i in range(traj.shape[0] - 1):
            densified_traj.append(traj[i])
            for j in range(1, n_points_interp):
                densified_traj.append(traj[i] + j * (traj[i + 1] - traj[i]) / n_points_interp)
        densified_traj.append(traj[-1])
        densified_traj = torch.stack(densified_traj)
        densified_trajs.append(densified_traj)
    return densified_trajs


def are_points_closer_than_margin(points_batch, margin):
    """
    Check proximity between points. (Sometimes this is for Robot-robot collisions).
    Args:
        points_batch: (..., n_points, q_dim)
    Returns:
        collisions: (..., n_points, n_points), True if there is the pair of points is closer than the margin.
    """
    # Check collisions between robots.
    assert points_batch.dim() >= 2
    points_batch1 = points_batch.unsqueeze(-2)
    points_batch2 = points_batch.unsqueeze(-3)
    # Pairwise distances between robots.
    robot_dq = points_batch1 - points_batch2
    # (..., n_robots, n_robots)
    robot_dq_norm = torch.norm(robot_dq, dim=-1)
    # (..., n_robots, n_robots)
    collisions = robot_dq_norm < margin
    # Set the trace to be False.
    collisions = collisions & ~torch.eye(collisions.shape[-1], device=collisions.device, dtype=collisions.dtype)
    return collisions
