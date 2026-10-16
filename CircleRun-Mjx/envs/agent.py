from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
ASSETS_AGENT_PATH = Path(__file__).resolve().parent / "assets" / "agent"

@dataclass(frozen=True)
class PointAgent:
    name: str = "Point"
    action_size: int = 2
    agent_obs_size: int = 12
    n_substeps: int = 10
    action_scale: float = 1.0
    keepout: float = 0.4
    geom_size: float = 0.1
    collision_geom_count: int = 1
    max_speed: float = 1.5
    randomize_heading: bool = True
    xml_path: str = str(ASSETS_AGENT_PATH / "point" / "point.xml")

    def compute_agent_obs(self, data: Any) -> jnp.ndarray:
        return jnp.asarray(data.sensordata, dtype=jnp.float32)

    def compute_heading(self, data: Any) -> jnp.ndarray:
        return jnp.asarray(data.qpos[2], dtype=jnp.float32)

    def randomize_init_qpos(self, data: Any, rng: Any) -> Any:
        if not self.randomize_heading:
            return data
        theta = jax.random.uniform(rng, (), minval=-jnp.pi, maxval=jnp.pi,
                                   dtype=jnp.float32)
        return data.replace(qpos=data.qpos.at[2].set(theta))

    def compute_state_info(
        self, data: Any, info: dict[str, Any], *, ctrl_dt: float,
    ) -> None:
        dtype = jnp.float32
        agent_position = jnp.asarray(data.qpos[:2], dtype=dtype)
        delta = agent_position - info["agent_pos"]
        move_dist = jnp.linalg.norm(delta)
        info["agent_pos"] = agent_position
        info["agent_vel_xy"] = data.qvel[:2].astype(dtype)
        info["move_dist"] = move_dist
        info["agent_speed"] = move_dist / jnp.maximum(
            jnp.asarray(ctrl_dt, dtype=dtype), jnp.asarray(1e-6, dtype=dtype),
        )

__all__ = ["ASSETS_AGENT_PATH", "PointAgent"]
