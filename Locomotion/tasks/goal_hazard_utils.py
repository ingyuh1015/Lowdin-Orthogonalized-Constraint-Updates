from __future__ import annotations

from typing import Iterable

import numpy as np


def sample_entities(
    rng: np.random.Generator,
    *,
    count: int,
    extents: tuple[float, float, float, float],
    entity_radius: float,
    clearance: float = 0.0,
    avoid_positions: np.ndarray | None = None,
    avoid_radii: np.ndarray | None = None,
    max_tries: int = 2048,
) -> np.ndarray:
    """Sample non-overlapping XY positions inside extents."""
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    xmin, ymin, xmax, ymax = extents
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid extents: {extents}")

    avoid_positions = (
        np.asarray(avoid_positions, dtype=np.float64).reshape(-1, 2)
        if avoid_positions is not None
        else np.zeros((0, 2), dtype=np.float64)
    )
    avoid_radii = (
        np.asarray(avoid_radii, dtype=np.float64).reshape(-1)
        if avoid_radii is not None
        else np.zeros((0,), dtype=np.float64)
    )
    if len(avoid_positions) != len(avoid_radii):
        raise ValueError("avoid_positions and avoid_radii must have the same length")

    samples: list[np.ndarray] = []
    low = np.array([xmin + entity_radius, ymin + entity_radius], dtype=np.float64)
    high = np.array([xmax - entity_radius, ymax - entity_radius], dtype=np.float64)
    if np.any(low > high):
        raise ValueError("Entity radius is too large for the extents")

    for _ in range(max_tries):
        if len(samples) >= count:
            break
        candidate = rng.uniform(low=low, high=high)
        if _is_valid_candidate(
            candidate,
            entity_radius=entity_radius,
            clearance=clearance,
            sampled=samples,
            avoid_positions=avoid_positions,
            avoid_radii=avoid_radii,
        ):
            samples.append(candidate)

    if len(samples) != count:
        raise RuntimeError(f"Unable to sample {count} entities within {max_tries} tries")
    return np.asarray(samples, dtype=np.float64)


def pseudo_lidar(
    *,
    agent_xy: np.ndarray,
    agent_yaw: float,
    positions: np.ndarray,
    num_bins: int = 16,
    max_dist: float | None = 3.0,
    alias: bool = True,
    exp_gain: float = 1.0,
) -> np.ndarray:
    """Point-goal style pseudo lidar over XY positions."""
    obs = np.zeros(num_bins, dtype=np.float32)
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return obs

    cos_yaw = np.cos(agent_yaw)
    sin_yaw = np.sin(agent_yaw)
    rot = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]], dtype=np.float64)
    bin_size = (2.0 * np.pi) / num_bins

    for point in points:
        ego_xy = (point - agent_xy) @ rot.T
        z = complex(ego_xy[0], ego_xy[1])
        dist = np.abs(z)
        angle = np.angle(z) % (2.0 * np.pi)
        bin_idx = int(angle / bin_size)
        bin_angle = bin_idx * bin_size
        if max_dist is None:
            sensor = float(np.exp(-exp_gain * dist))
        else:
            sensor = max(0.0, float(max_dist - dist) / float(max_dist))
        obs[bin_idx] = max(obs[bin_idx], sensor)
        if alias:
            alias_value = (angle - bin_angle) / bin_size
            bin_plus = (bin_idx + 1) % num_bins
            bin_minus = (bin_idx - 1) % num_bins
            obs[bin_plus] = max(obs[bin_plus], alias_value * sensor)
            obs[bin_minus] = max(obs[bin_minus], (1.0 - alias_value) * sensor)
    return obs


def goal_progress_reward(
    *,
    previous_distance: float,
    current_distance: float,
    reached_goal: bool,
    progress_weight: float = 1.0,
    goal_bonus: float = 1.0,
) -> float:
    reward = progress_weight * (previous_distance - current_distance)
    if reached_goal:
        reward += goal_bonus
    return float(reward)


def hazard_contact_cost(
    contact_points: np.ndarray,
    hazard_positions: np.ndarray,
    hazard_radius: float,
    contact_margin: float = 0.0,
) -> float:
    """Indicator cost for floor contact points inside any hazard disk."""
    contacts = np.asarray(contact_points, dtype=np.float64).reshape(-1, 2)
    hazards = np.asarray(hazard_positions, dtype=np.float64).reshape(-1, 2)
    if len(contacts) == 0 or len(hazards) == 0:
        return 0.0
    threshold = hazard_radius + contact_margin
    for point in contacts:
        dists = np.linalg.norm(hazards - point, axis=1)
        if np.any(dists <= threshold):
            return 1.0
    return 0.0


def goal_reached(agent_xy: np.ndarray, goal_xy: np.ndarray, goal_radius: float) -> bool:
    return bool(np.linalg.norm(np.asarray(agent_xy) - np.asarray(goal_xy)) <= goal_radius)


def _is_valid_candidate(
    candidate: np.ndarray,
    *,
    entity_radius: float,
    clearance: float,
    sampled: Iterable[np.ndarray],
    avoid_positions: np.ndarray,
    avoid_radii: np.ndarray,
) -> bool:
    for other in sampled:
        if np.linalg.norm(candidate - other) < (2.0 * entity_radius + clearance):
            return False
    for other_pos, other_radius in zip(avoid_positions, avoid_radii):
        if np.linalg.norm(candidate - other_pos) < (entity_radius + other_radius + clearance):
            return False
    return True
