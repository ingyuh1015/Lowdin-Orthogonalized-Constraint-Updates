from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Union

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx
if TYPE_CHECKING:
    from envs.agent import PointAgent

@dataclass(frozen=True)
class ConstraintConfig:

    name: str
    cost_limit: float = 25.0
    compute_fn: Any = None
    params: Mapping[str, Any] = field(default_factory=dict)
    cost_type: str = "binary"

    def compute(self, data: Any, info: dict[str, Any]) -> None:
        self.compute_fn(data, info, **self.params)

Observation = Union[jax.Array, Mapping[str, jax.Array]]

def _default_cost_vector() -> jax.Array:
    return jnp.zeros((0,), dtype=jnp.float32)

@struct.dataclass
class EnvState:

    data: mjx.Data
    obs: Observation
    reward: jax.Array
    cost_vector: jax.Array = struct.field(default_factory=_default_cost_vector)
    done: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0, dtype=jnp.float32))
    terminated: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0, dtype=jnp.float32))
    truncated: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0, dtype=jnp.float32))
    info: dict[str, Any] = struct.field(default_factory=dict)

def mjx_step(
    mjx_model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    objects: tuple[dict, ...] = (),
) -> mjx.Data:
    dtype = jnp.float32
    t_sin = jnp.sin(data.time)
    t_cos = jnp.cos(data.time)
    for obj in objects:
        time_xy = jnp.array([t_sin, t_cos], dtype=dtype) * jnp.asarray(obj["travel"], dtype=dtype)
        mocap_xy = obj["base_xy"] + time_xy[None, :]
        z = jnp.ones((mocap_xy.shape[0], 1), dtype=dtype) * jnp.asarray(obj["size"], dtype=dtype)
        mocap_pos = jnp.concatenate([mocap_xy, z], axis=1)
        data = data.replace(mocap_pos=data.mocap_pos.at[obj["body_mocapid"]].set(mocap_pos))
    data = data.replace(ctrl=jnp.asarray(action, dtype=data.ctrl.dtype))
    return mjx.step(mjx_model, data)

class MjxEnv:
    def __init__(
        self,
        *,
        mj_model: mujoco.MjModel,
        mjx_model: mjx.Model,
        agent: "PointAgent",
        objects: dict[str, Any],
        placements_extents: tuple,
        constraints: tuple[ConstraintConfig, ...],
        action_size: int,
        ctrl_dt: float,
        sim_dt: float,
        n_substeps: int,
        agent_geom_ids: jnp.ndarray,
    ) -> None:
        self._dtype = jnp.float32
        self._mj_model = mj_model
        self._mjx_model = mjx_model
        self.agent = agent
        self.objects = objects
        self.placements_extents = tuple(placements_extents)
        self.constraints = tuple(constraints)
        self._action_size_val = action_size
        self._num_constraints = len(self.constraints)
        self._ctrl_dt = float(ctrl_dt)
        self._sim_dt = float(sim_dt)
        self._n_substeps = int(n_substeps)
        self._zero_action = jnp.zeros((action_size,), dtype=self._dtype)
        self._agent_geom_ids = agent_geom_ids

        self._default_spawn_radius = (self.placements_extents[2] - self.placements_extents[0]) / 2.0
        self._agent_keepout = float(agent.keepout)
        goal_cfg = objects.get("goal")
        self._goal_keepout = float(goal_cfg.keepout) if goal_cfg is not None else 0.0

        self._placement_objects = tuple(
            (name, cfg) for name, cfg in objects.items()
            if name not in ("goal", "circle")
        )

        self._mocap_descriptors = tuple(
            {
                "name": name,
                "body_mocapid": cfg.model_info.body_mocapid,
                "travel": float(cfg.travel),
                "size": float(cfg.size),
                "base_xy_key": ("gremlin_base_xy" if name == "gremlins"
                                else f"{name}_base_xy"),
            }
            for name, cfg in objects.items()
            if cfg.obj_type == "mocap" and cfg.num > 0
        )

        self._obs_size = int(self.agent.agent_obs_size)

    def _compute_obs_size(self) -> int:
        return int(self.agent.agent_obs_size)

    def compute_step_info(self, data, info: dict, action) -> None:
        del action
        self.agent.compute_state_info(data, info, ctrl_dt=self._ctrl_dt)

    def compute_reward(self, data, info: dict):
        info["reward"] = jnp.zeros((), dtype=self._dtype)
        return data

    def compute_obs(self, data, info: dict) -> jnp.ndarray:
        return self.agent.compute_agent_obs(data).astype(self._dtype)

    def compute_cost_vector(self, data, info: dict) -> jnp.ndarray:
        cost_vector = jnp.zeros((self._num_constraints,), dtype=self._dtype)
        for i, c in enumerate(self.constraints):
            c.compute(data, info)
            cost_vector = cost_vector.at[i].set(info[f"cost_{c.name}"])
        return cost_vector

    def mjx_step(self, data, action, info: dict):
        objects = tuple(
            {**d, "base_xy": info[d["base_xy_key"]]}
            for d in self._mocap_descriptors
        )

        def substep(d, _):
            return mjx_step(self._mjx_model, d, action, objects), None

        return jax.lax.scan(substep, data, None, length=self._n_substeps)[0]

    @property
    def dt(self) -> float:
        return self._ctrl_dt

    @property
    def sim_dt(self) -> float:
        return self._sim_dt

    @property
    def n_substeps(self) -> int:
        return self._n_substeps

    @property
    def action_size(self) -> int:
        return self._action_size_val

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def unwrapped(self) -> "MjxEnv":
        return self

    def reset_objects(self, data, info: dict, rng) -> Any:
        from envs.tasks import sample_obstacles, place_freejoints
        needs_forward = False
        target_ref = info.get("target_pos", info["agent_pos"])
        for name, cfg in self._placement_objects:
            obj_rng = jax.random.fold_in(
                rng, int.from_bytes(name.encode(), "little") % (2**31)
            )
            positions = sample_obstacles(
                obj_rng, info["agent_pos"], target_ref,
                cfg.num, self._default_spawn_radius, float(cfg.keepout),
                agent_keepout=self._agent_keepout, goal_keepout=self._goal_keepout,
            )
            info[f"{name}_positions"] = positions
            if cfg.obj_type in ("freejoint", "mocap"):
                data = data.replace(qpos=place_freejoints(
                    data.qpos, positions, cfg.model_info.jnt_qposadr, float(cfg.size),
                ))
                needs_forward = True
            elif cfg.obj_type == "static_mocap":
                dtype = jnp.float32
                mocap_xy = jnp.asarray(positions, dtype=dtype)
                center_z = getattr(cfg, "height", 0.0) / 2.0 if getattr(cfg, "height", 0.0) > 0 else float(cfg.size)
                z = jnp.ones((mocap_xy.shape[0], 1), dtype=dtype) * jnp.asarray(center_z, dtype=dtype)
                mocap_pose = jnp.concatenate([mocap_xy, z], axis=1)
                data = data.replace(
                    mocap_pos=data.mocap_pos.at[cfg.model_info.body_mocapid].set(mocap_pose)
                )
                needs_forward = True
        if needs_forward:
            data = mjx.forward(self._mjx_model, data)
        return data

    def reset(self, rng: jax.Array) -> EnvState:
        data = mjx.make_data(self._mjx_model)
        data = mjx.forward(self._mjx_model, data)

        rng, sub_rng = jax.random.split(rng)
        rng, agent_rng = jax.random.split(rng)
        ext = self.placements_extents
        xy = jax.random.uniform(
            agent_rng, (2,),
            minval=jnp.asarray([ext[0], ext[1]], dtype=self._dtype),
            maxval=jnp.asarray([ext[2], ext[3]], dtype=self._dtype),
        )
        data = data.replace(qpos=data.qpos.at[0:2].set(xy))
        rng, theta_rng = jax.random.split(rng)
        data = self.agent.randomize_init_qpos(data, theta_rng)
        data = mjx.forward(self._mjx_model, data)

        info: dict[str, Any] = {
            "step_count": jnp.asarray(0, dtype=jnp.int32),
            "agent_pos": data.qpos[:2].astype(self._dtype),
        }
        data = self.reset_objects(data, info, sub_rng)
        info["rng"] = rng

        self.compute_step_info(data, info, self._zero_action)
        data = self.compute_reward(data, info)
        cost_vector = self.compute_cost_vector(data, info)
        obs = self.compute_obs(data, info)

        zero = jnp.zeros((), dtype=self._dtype)
        return EnvState(
            data=data, obs=obs, reward=info["reward"], cost_vector=cost_vector,
            done=zero, terminated=zero, truncated=zero, info=info,
        )

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        rng = state.info.get("rng", jax.random.PRNGKey(0))
        rng, _ = jax.random.split(rng)

        info = dict(state.info)
        data = self.mjx_step(state.data, action, info)
        self.compute_step_info(data, info, action)
        data = self.compute_reward(data, info)
        cost_vector = self.compute_cost_vector(data, info)
        obs = self.compute_obs(data, info)

        info["rng"] = rng
        terminated = jnp.asarray(info.get("terminated", 0.0), dtype=self._dtype)

        return EnvState(
            data=data, obs=obs, reward=info["reward"], cost_vector=cost_vector,
            done=terminated, terminated=terminated,
            truncated=jnp.zeros((), dtype=self._dtype),
            info=info,
        )

__all__ = ["ConstraintConfig", "EnvState", "Observation", "MjxEnv", "mjx_step"]
