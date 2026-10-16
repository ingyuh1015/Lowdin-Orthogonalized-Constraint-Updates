from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from envs.env import MjxEnv

@dataclass(frozen=True)
class SigwallConfig:
    num: int = 4
    size: float = 0.1
    contact: bool = False
    keepout: float = 0.0
    lidar_observable: bool = False
    obj_type: str = "geom"
    collision_geom_count: int = 1
    walls: tuple = field(default_factory=tuple)

    def add(self, world: mujoco.MjsBody, mjcf: mujoco.MjSpec) -> None:
        half_thickness = 0.05
        half_perp = 3.5
        half_height = 0.3
        center_z = 0.25
        for i, (axis, coord) in enumerate(self.walls):
            if axis == "x":
                size = [half_thickness, half_perp, half_height]
                pos = [float(coord), 0.0, center_z]
            else:
                size = [half_perp, half_thickness, half_height]
                pos = [0.0, float(coord), center_z]
            world.add_geom(
                name=f"sigwall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size, pos=pos,
                rgba=[1.0, 0.5, 0.0, 0.1], contype=0, conaffinity=0,
            )

@dataclass(frozen=True)
class CircleConfig:
    num: int = 1
    size: float = 0.0
    contact: bool = False
    keepout: float = 0.0
    lidar_observable: bool = True
    obj_type: str = "geom"
    collision_geom_count: int = 1
    radius: float = 1.5

    def add(self, world: mujoco.MjsBody, mjcf: mujoco.MjSpec) -> None:
        world.add_geom(
            name="circle", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[self.radius, 0.01, 0], pos=[0, 0, 0.01],
            rgba=[0, 1, 0, 0.1], contype=0, conaffinity=0,
        )

    def get_positions(self, data: Any, info: dict[str, Any]) -> jnp.ndarray:
        return jnp.zeros((1, 2), dtype=jnp.float32)

def sample_point(rng: jax.Array, radius: float) -> jnp.ndarray:
    return jax.random.uniform(rng, (2,), minval=-radius, maxval=radius)

def sample_obstacles(
    rng: jax.Array,
    agent_pos: jnp.ndarray,
    goal: jnp.ndarray,
    count: int,
    spawn_radius: float,
    keepout: float = 0.2,
    agent_keepout: float = 0.0,
    goal_keepout: float = 0.0,
    max_attempts: int = 20,
) -> jnp.ndarray:
    safe_count = max(int(count), 1)
    dtype = jnp.float32
    placed_init = jnp.ones((safe_count, 2), dtype=dtype) * 100.0
    min_agent_dist_sq = jnp.asarray(keepout + agent_keepout, dtype=dtype) ** 2
    min_goal_dist_sq = jnp.asarray(keepout + goal_keepout, dtype=dtype) ** 2
    min_pair_dist_sq = jnp.asarray(keepout * 2.0, dtype=dtype) ** 2

    def sample_one(
        carry: tuple[jax.Array, jnp.ndarray], idx: jax.Array
    ) -> tuple[tuple[jax.Array, jnp.ndarray], jnp.ndarray]:
        key, placed = carry

        key, init_key = jax.random.split(key)
        init_pos = sample_point(init_key, spawn_radius)

        def attempt(
            sub_carry: tuple[jax.Array, jnp.ndarray], _unused: None
        ) -> tuple[tuple[jax.Array, jnp.ndarray], jax.Array]:
            key_in, best = sub_carry
            key_in, subkey = jax.random.split(key_in)
            cand = sample_point(subkey, spawn_radius)
            cand_agent_delta = cand - agent_pos
            cand_goal_delta = cand - goal
            placed_delta = placed - cand
            valid = (
                (jnp.sum(cand_agent_delta * cand_agent_delta) > min_agent_dist_sq)
                & (jnp.sum(cand_goal_delta * cand_goal_delta) > min_goal_dist_sq)
                & jnp.all(jnp.sum(placed_delta * placed_delta, axis=-1) > min_pair_dist_sq)
            )
            new_best = jnp.where(valid, cand, best)
            return (key_in, new_best), valid

        (key, final_pos), _ = jax.lax.scan(
            attempt,
            (key, init_pos),
            None,
            length=max_attempts,
        )
        placed = placed.at[idx].set(final_pos)
        return (key, placed), final_pos

    (_, _), result = jax.lax.scan(
        sample_one,
        (rng, placed_init),
        jnp.arange(safe_count),
    )
    return result[:count]

def place_freejoints(
    qpos: jnp.ndarray,
    positions: jnp.ndarray,
    jnt_qposadr: jnp.ndarray,
    z_height: float,
) -> jnp.ndarray:
    body_count = positions.shape[0]
    pose = jnp.concatenate(
        [
            positions,
            jnp.ones((body_count, 1), dtype=jnp.float32)
                * jnp.asarray(z_height, dtype=jnp.float32),
            jnp.tile(
                jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
                (body_count, 1),
            ),
        ],
        axis=1,
    )
    offsets = jnp.arange(7, dtype=jnp.int8)[None, :]
    indices = jnt_qposadr[:, None] + offsets
    return qpos.at[indices.reshape(-1)].set(pose.reshape(-1))

def compute_out_of_boundary(
    data: Any, info: dict[str, Any], *,
    boundaries: tuple,
) -> None:
    del data
    agent_pos = info["agent_pos"]
    result = jnp.asarray(False)
    for axis_idx, coord in boundaries:
        pos_on_axis = agent_pos[axis_idx]
        result = result | (jnp.sign(coord) * pos_on_axis > abs(coord))
    info["cost_out_of_boundary"] = result.astype(jnp.float32)

def compute_velocity_limit_high(data: Any, info: dict[str, Any], *, threshold: float) -> None:
    del data
    info["cost_velocity_limit_high"] = (
        info["agent_speed"] > threshold
    ).astype(jnp.float32)

def compute_velocity_limit_low(data: Any, info: dict[str, Any], *, threshold: float) -> None:
    del data
    info["cost_velocity_limit_low"] = (
        info["agent_speed"] < threshold
    ).astype(jnp.float32)

class NavEnv(MjxEnv):
    def __init__(self, *, lidar_num_bins: int, lidar_max_dist: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lidar_num_bins = lidar_num_bins
        self._lidar_max_dist = lidar_max_dist
        self._lidar_cfgs = tuple(
            cfg for cfg in self.objects.values() if cfg.lidar_observable
        )
        self._obs_size = self._compute_obs_size()

    def _lidar_group_count(self) -> int:
        return len(self._lidar_cfgs)

    def _compute_obs_size(self) -> int:
        return self.agent.agent_obs_size + self._lidar_group_count() * self._lidar_num_bins

    def lidar_position_groups(self, data: Any, info: dict) -> list[jnp.ndarray]:
        return [cfg.get_positions(data, info)[:, :2] for cfg in self._lidar_cfgs]

    def compute_obs(self, data: Any, info: dict) -> jnp.ndarray:
        groups = self.lidar_position_groups(data, info)
        lidar = self._compute_lidar_obs(
            agent_pos=info["agent_pos"],
            agent_theta=self.agent.compute_heading(data),
            position_groups=tuple(groups),
        )
        agent_obs = self.agent.compute_agent_obs(data).astype(self._dtype)
        agent_size = self.agent.agent_obs_size
        obs = jnp.zeros((self._obs_size,), dtype=self._dtype)
        obs = obs.at[:agent_size].set(agent_obs)
        obs = obs.at[agent_size:].set(lidar.astype(self._dtype))
        return obs

    def _compute_weighted_lidar(
        self,
        positions: jnp.ndarray,
        agent_pos: jnp.ndarray,
        agent_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        num_bins = self._lidar_num_bins
        max_dist = self._lidar_max_dist
        rel = positions - agent_pos
        dists = jnp.linalg.norm(rel, axis=-1)
        angles = jnp.arctan2(rel[:, 1], rel[:, 0]) - agent_theta
        angles = jnp.mod(angles, 2 * jnp.pi)
        bin_size = 2 * jnp.pi / num_bins
        clamped_dists = jnp.clip(dists, 0.0, max_dist)
        sensor = 1.0 - (clamped_dists / max_dist)
        bin_main = jnp.floor(angles / bin_size).astype(jnp.int32)
        alias_frac = (angles - bin_main.astype(self._dtype) * bin_size) / bin_size
        bin_plus = jnp.mod(bin_main + 1, num_bins)
        bin_minus = jnp.mod(bin_main - 1, num_bins)

        main_oh = jax.nn.one_hot(bin_main, num_bins, dtype=self._dtype)
        plus_oh = jax.nn.one_hot(bin_plus, num_bins, dtype=self._dtype)
        minus_oh = jax.nn.one_hot(bin_minus, num_bins, dtype=self._dtype)

        main_vals = main_oh * sensor[:, None]
        plus_vals = plus_oh * (alias_frac * sensor)[:, None]
        minus_vals = minus_oh * ((1.0 - alias_frac) * sensor)[:, None]
        return jnp.maximum(jnp.maximum(main_vals, plus_vals), minus_vals)

    def _compute_lidar_multi(
        self,
        *,
        agent_pos: jnp.ndarray,
        agent_theta: jnp.ndarray,
        position_groups: tuple[jnp.ndarray, ...],
    ) -> tuple[jnp.ndarray, ...]:
        dummy = jnp.array([[jnp.inf, jnp.inf]], dtype=self._dtype)
        padded_groups = [jnp.concatenate([g, dummy], axis=0) for g in position_groups]
        group_sizes = tuple(g.shape[0] for g in padded_groups)
        all_positions = jnp.concatenate(padded_groups, axis=0)

        all_weighted = self._compute_weighted_lidar(all_positions, agent_pos, agent_theta)

        offsets = tuple(itertools.accumulate((0,) + group_sizes[:-1]))
        return tuple(
            jnp.max(all_weighted[o : o + s], axis=0)
            for o, s in zip(offsets, group_sizes)
        )

    def _compute_lidar_obs(
        self,
        *,
        agent_pos: jnp.ndarray,
        agent_theta: jnp.ndarray,
        position_groups: tuple[jnp.ndarray, ...],
    ) -> jnp.ndarray:
        per_group = self._compute_lidar_multi(
            agent_pos=agent_pos,
            agent_theta=agent_theta,
            position_groups=position_groups,
        )
        return jnp.concatenate(per_group, axis=0)

class CircleEnv(NavEnv):
    def __init__(self, *, reward_factor: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._circle_radius = float(self.objects["circle"].radius)
        self._reward_factor = float(reward_factor)

    def reset_objects(self, data: Any, info: dict, rng: Any) -> Any:
        agent_pos = info["agent_pos"]
        agent_dist = jnp.linalg.norm(agent_pos)
        safe_dist = jnp.maximum(agent_dist, 1e-6)
        info["target_pos"] = agent_pos * (self._circle_radius / safe_dist)
        info["goal_distance"] = jnp.abs(agent_dist - self._circle_radius)
        return super().reset_objects(data, info, rng)

    def compute_reward(self, data: Any, info: dict) -> Any:
        dtype = jnp.float32
        pos = jnp.asarray(info["agent_pos"], dtype=dtype)
        vel = jnp.asarray(info["agent_vel_xy"], dtype=dtype)
        r = jnp.asarray(self._circle_radius, dtype=dtype)

        dist = jnp.linalg.norm(pos)
        safe_dist = jnp.maximum(dist, jnp.asarray(1e-6, dtype=dtype))
        sweep = -vel[0] * pos[1] + vel[1] * pos[0]
        radial_dev = jnp.abs(dist - r)

        info["target_pos"] = pos * (r / safe_dist)
        info["goal_distance"] = radial_dev
        info["reward"] = (sweep / (jnp.asarray(1.0, dtype=dtype) + radial_dev)
                          * jnp.asarray(self._reward_factor, dtype=dtype))
        return data

__all__ = [
    "SigwallConfig",
    "CircleConfig",
    "sample_point",
    "sample_obstacles",
    "place_freejoints",
    "compute_out_of_boundary",
    "compute_velocity_limit_high",
    "compute_velocity_limit_low",
    "NavEnv",
    "CircleEnv",
]
