"""Depth-camera math for the Phase 3 student (PLAN.md §4.2, §5.4): pinhole geometry, the noise model and the
observation. Pure torch, so it runs in the CPU tests; the env (tasks/depth/depth_env.py) applies it to Isaac Lab's
RayCasterCamera output.

Depth images are (n, H, W) distances to the image plane in metres, row 0 at the top and column 0 on the left, as
Isaac Lab's pinhole pattern produces them.
"""

from __future__ import annotations

import math

import torch


def focal_length_px(size_px: int, aperture: float, focal_length: float) -> float:
    """Focal length in pixels along one image axis, from the pattern's aperture and focal length (same units)."""
    return focal_length * size_px / aperture


def horizontal_fov_deg(aperture: float, focal_length: float) -> float:
    return math.degrees(2.0 * math.atan(aperture / (2.0 * focal_length)))


def column_azimuths(width: int, aperture: float, focal_length: float) -> torch.Tensor:
    """(W,) horizontal angle of each pixel column's centre from the optical axis, in radians, positive to the left
    (Isaac Lab's pinhole pattern: pixel centres at +0.5, camera y axis to the left)."""
    f = focal_length_px(width, aperture, focal_length)
    u = torch.arange(width, dtype=torch.float64) + 0.5 - width / 2.0
    return torch.atan2(-u, torch.full_like(u, f))


def row_elevations(height: int, vertical_aperture: float, focal_length: float) -> torch.Tensor:
    """(H,) vertical angle of each pixel row's centre (in the image's centre column), radians, positive up."""
    f = focal_length_px(height, vertical_aperture, focal_length)
    v = torch.arange(height, dtype=torch.float64) + 0.5 - height / 2.0
    return torch.atan2(-v, torch.full_like(v, f))


def depth_edges(depth: torch.Tensor, threshold: float) -> torch.Tensor:
    """(n, H, W) bool: pixels whose depth differs from a 4-neighbour's by more than `threshold` (a depth edge)."""
    edge = torch.zeros_like(depth, dtype=torch.bool)
    dv = (depth[:, 1:, :] - depth[:, :-1, :]).abs() > threshold
    dh = (depth[:, :, 1:] - depth[:, :, :-1]).abs() > threshold
    edge[:, 1:, :] |= dv
    edge[:, :-1, :] |= dv
    edge[:, :, 1:] |= dh
    edge[:, :, :-1] |= dh
    return edge


def add_depth_noise(
    depth: torch.Tensor,
    max_range: float,
    gaussian_std: float,
    range_std: float,
    dropout_prob: float,
    edge_dropout_prob: float,
    edge_threshold: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A stereo-like depth sensor (PLAN.md §5.4), on clean depth (n, H, W):

    - Gaussian noise with std gaussian_std + range_std * depth^2 (stereo error grows with the square of range);
    - random dropout: each pixel reads max_range ("no return") with probability dropout_prob;
    - edge artifacts: pixels on a depth edge (depth_edges, edge_threshold) drop out with probability
      edge_dropout_prob, as stereo matching fails at occlusion boundaries;
    - the max-range cutoff: the result is clipped to [0, max_range], and max-range pixels stay at max_range.
    """
    noise = torch.randn(depth.shape, generator=generator, device=depth.device, dtype=depth.dtype)
    noisy = depth + noise * (gaussian_std + range_std * depth * depth)
    noisy = torch.where(depth >= max_range, depth, noisy)  # nothing within range stays nothing
    u = torch.rand(depth.shape, generator=generator, device=depth.device, dtype=depth.dtype)
    edges = depth_edges(depth, edge_threshold)
    drop = (u < dropout_prob) | (edges & (u < dropout_prob + edge_dropout_prob))
    noisy = torch.where(drop, torch.full_like(noisy, max_range), noisy)
    return noisy.clamp(0.0, max_range)


def depth_observation(depth: torch.Tensor, max_range: float) -> torch.Tensor:
    """(n, H * W) depth clipped to [0, max_range] and scaled to [0, 1] (1 = nothing within range), row-major."""
    return (depth.clamp(0.0, max_range) / max_range).flatten(1)
