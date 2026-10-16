from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import os
import pickle
import platform as py_platform
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml
from flax import struct
from mujoco import mjx

from flashbax.buffers import trajectory_buffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as RolloutBufferState

from algos.base import OnPolicyBase, BasePolicyState
from config.config import (
    COST_FIELD_DEFAULTS,
    apply_task_preset,
    available_task_presets,
    available_training_algorithms,
    build_algo_config,
    build_constraint_spec,
    get_algo_class,
    get_object_config_class,
    get_task_env_class,
    load_task_config,
)
from envs.agent import PointAgent
from envs.env import ConstraintConfig, EnvState
from envs.tasks import (
    SigwallConfig, CircleConfig,
    compute_out_of_boundary,
    compute_velocity_limit_high,
    compute_velocity_limit_low,
)
from envs.wrappers import Wrapper, EpisodeWrapper, VmapWrapper
from estimation import batch_critic_values, compute_targets
from train.types import (
    CompiledTrainingPlan,
    ExecutionResult,
    FullRunResult,
    Metrics,
    RunContext,
    RuntimeState,
)

_logger = logging.getLogger(__name__)

_start: float | None = None

def mark_start() -> None:
    global _start
    if _start is None:
        _start = time.monotonic()

def elapsed() -> float:
    return time.monotonic() - (_start if _start is not None else time.monotonic())

_VALID_BACKENDS = {"cpu", "gpu"}
_VALID_MATMUL_PRECISIONS = {"default", "high", "highest"}

def _canonical_backend(value: str) -> str:
    candidate = value.strip().lower()
    if candidate.startswith(("gpu", "cuda", "rocm")):
        return "gpu"
    return "cpu"

def _env_backend() -> str | None:
    platforms = os.environ.get("JAX_PLATFORMS", "").strip()
    if not platforms:
        return None
    return _canonical_backend(platforms.split(",", maxsplit=1)[0])

def init_backend(backend: str = "gpu") -> str:
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR", ""):
        cache_dir = str(Path(__file__).resolve().parents[1] / ".cache" / "jax_compilation_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

    requested = backend.strip().lower()
    if requested not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {_VALID_BACKENDS}.")

    jax_platform = "cuda" if requested == "gpu" else requested
    if jax_platform == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    else:
        os.environ["JAX_PLATFORMS"] = jax_platform
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return requested

def get_backend() -> str:
    env_backend = _env_backend()
    if env_backend is not None:
        return env_backend
    return _canonical_backend(jax.default_backend())

_PLATFORM_XLA_FLAGS: dict[str, str] = {
    "gpu": "--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true",
    "cpu": "--xla_cpu_multi_thread_eigen=true",
}

def init_xla_flags(profile: str) -> None:
    selected = profile.strip().lower()
    platform_flags = _PLATFORM_XLA_FLAGS.get(selected, "")
    if platform_flags:
        existing = os.environ.get("XLA_FLAGS", "").strip()
        if not existing:
            os.environ["XLA_FLAGS"] = platform_flags
    raw_precision = (os.environ.get("MJX_SAFETY_MATMUL_PRECISION") or "").strip().lower()
    if raw_precision and raw_precision in _VALID_MATMUL_PRECISIONS:
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = raw_precision

def bootstrap_backend(backend: str) -> str:
    active_backend = str(init_backend(backend))
    init_xla_flags(active_backend)
    return active_backend

def load_bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"Expected boolean-like value, got: {value!r}")

def load_choice_value(value: str, *, choices: Sequence[str]) -> str:
    normalized = str(value).strip().lower()
    choice_map = {choice.lower(): choice for choice in choices}
    if normalized in choice_map:
        return choice_map[normalized]
    raise argparse.ArgumentTypeError(
        f"invalid choice: {value!r} (choose from {', '.join(repr(choice) for choice in choices)})"
    )

def load_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    training_algorithms = available_training_algorithms()
    parser = argparse.ArgumentParser(description="Unified MJX training entrypoint")
    parser.add_argument(
        "--algo",
        type=lambda value: load_choice_value(value, choices=training_algorithms),
        choices=training_algorithms,
        required=True,
        help=(
            "Training algorithm. Pick an explicit trainable algo such as PPO, "
            "PPOLag, SAC, or CPO. Matching is case-insensitive."
        ),
    )
    parser.add_argument("--task", default="SafetyPointGoal1")
    parser.add_argument(
        "--task-preset",
        dest="task_preset",
        choices=available_task_presets(),
        default=None,
        help="Task preset (config/yaml/task_presets/) appending velocity/etc. constraints to the loaded task config.",
    )
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--logging", choices=["true", "false"], default=None)
    parser.add_argument(
        "--scale", type=int, default=None,
        help="Scale factor (>=2): num_envs, rollout_steps_per_update, update_batch_size all multiplied by N.",
    )
    parser.add_argument(
        "--redun", type=int, default=None,
        help="Constraint redundancy (>=1, default 1): replicate every task constraint N times.",
    )
    parser.add_argument("--lagrange-lr", type=float, default=None, help="Override lagrange_lr.")
    parser.add_argument("--lagrange-multiplier-init", type=float, default=None, help="Override lagrange_multiplier_init.")
    parser.add_argument("--update-epochs", type=int, default=None, help="Override update_epochs.")
    parser.add_argument("--update-batch-size", type=int, default=None, help="Override update_batch_size.")
    return parser.parse_args(argv)

def _field_default(field_info: Any) -> Any:
    if field_info.default is not MISSING:
        return deepcopy(field_info.default)
    if field_info.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return field_info.default_factory()
    raise ValueError(f"Field {field_info.name} has no default")

@dataclass(frozen=True, init=False)
class AlgoTrainConfig:
    algo_name: str = ""
    task_name: str = ""
    seed: int = 0
    backend: str | None = None
    use_logging: bool = True
    num_envs: int = 0
    obs_size: int = 0
    action_size: int = 0
    rollout_steps_per_update: int = 0
    update_batch_size: int = 0
    total_steps: int = 0
    total_updates: int = 0
    log_interval_updates: int = 0
    lagrange_lr: float | None = None
    lagrange_multiplier_init: float | None = None
    constraint_spec: Any = None
    task_config: Mapping[str, Any] | None = None
    raw_cost_limits: tuple[Any, ...] = field(default_factory=tuple)
    derived_cost_limits: tuple[Any, ...] = field(default_factory=tuple)
    cost_limits: Any = field(default_factory=tuple)
    discounted: bool = False
    num_constraints: int = 0
    update_epochs: int = 0
    compile_identity_base: tuple[Any, ...] = field(default_factory=tuple)
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)
    _explicit_keys: set[str] = field(default_factory=set, repr=False)

    def __init__(self, **kwargs: Any) -> None:
        explicit_keys = set(kwargs)
        extras: dict[str, Any] = {}
        dataclass_fields = {field_info.name: field_info for field_info in fields(type(self))}
        for name, field_info in dataclass_fields.items():
            if name.startswith("_"):
                continue
            value = kwargs.pop(name, _field_default(field_info))
            object.__setattr__(self, name, value)
        for key, value in kwargs.items():
            extras[str(key)] = deepcopy(value)
        object.__setattr__(self, "_extras", extras)
        object.__setattr__(self, "_explicit_keys", explicit_keys | set(extras))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AlgoTrainConfig":
        normalized: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, list) and key in {"raw_cost_limits", "derived_cost_limits", "compile_identity_base"}:
                normalized[key] = tuple(value)
            else:
                normalized[key] = value
        return cls(**normalized)

    def to_dict(self, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in self._explicit_keys:
            if key in exclude:
                continue
            if key in self._extras:
                result[key] = deepcopy(self._extras[key])
            elif hasattr(self, key):
                result[key] = deepcopy(getattr(self, key))
        return result

    def copy(self) -> "AlgoTrainConfig":
        return type(self).from_dict(self.to_dict())

    def with_updates(self, updates: Mapping[str, Any]) -> "AlgoTrainConfig":
        merged = self.to_dict()
        for key, value in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return type(self).from_dict(merged)

    def get(self, key: str, default: Any = None) -> Any:
        if key in {field_info.name for field_info in fields(type(self)) if not field_info.name.startswith("_")}:
            return getattr(self, key)
        return self._extras.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name in self._extras:
            return self._extras[name]
        raise AttributeError(name)

    def __contains__(self, key: object) -> bool:
        field_names = {field_info.name for field_info in fields(type(self)) if not field_info.name.startswith("_")}
        return isinstance(key, str) and (key in field_names or key in self._extras)

    def __repr__(self) -> str:
        return f"AlgoTrainConfig({self.to_dict()!r})"

def get_collection_shape(cfg: AlgoTrainConfig, scale: int = 1) -> tuple[int, int, int]:
    rollout_steps_per_update = int(cfg.rollout_steps_per_update) * scale
    if rollout_steps_per_update <= 0:
        raise ValueError(f"rollout_steps_per_update must be positive, got {rollout_steps_per_update}")
    episode_length = int(cfg.episode_length)
    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    if rollout_steps_per_update % episode_length != 0:
        raise ValueError(
            f"rollout_steps_per_update ({rollout_steps_per_update}) "
            f"must be divisible by episode_length ({episode_length})"
        )
    return rollout_steps_per_update, rollout_steps_per_update // episode_length, episode_length

def build_algo_train_config(args: Any, *, seed: int) -> AlgoTrainConfig:
    task_name = str(args.task)
    task_config = load_task_config(task_name)
    task_preset = getattr(args, "task_preset", None)
    if task_preset:
        task_config = apply_task_preset(task_config, task_preset, PointAgent().max_speed)
    constraints = task_config.get("constraints", [])
    redun = max(1, int(getattr(args, "redun", None) or 1))
    if constraints and redun > 1:
        constraints = [dict(item) for item in constraints for _ in range(redun)]
        task_config = dict(task_config)
        task_config["constraints"] = constraints
    if constraints:
        names = [item["name"] for item in constraints]
        limits = [float(item["cost_limit"]) for item in constraints]
        constraint_spec = build_constraint_spec(names, limits)
    else:
        constraint_spec = None

    algo_name = str(args.algo).lower()
    cfg = build_algo_config(algo_name)

    base_updates: dict[str, Any] = {
        "algo_name": algo_name,
        "task_name": task_name,
        "seed": int(seed),
        "use_logging": load_bool_value(getattr(args, "logging", None), default=True),
        "constraint_spec": constraint_spec,
        "task_config": task_config,
    }
    arg_overrides: tuple[tuple[str, str, Any], ...] = (
        ("num_envs", "num_envs", int),
        ("backend", "backend", str),
        ("total_steps", "total_steps", int),
        ("lagrange_lr", "lagrange_lr", float),
        ("lagrange_multiplier_init", "lagrange_multiplier_init", float),
        ("update_epochs", "update_epochs", int),
        ("update_batch_size", "update_batch_size", int),
    )
    for arg_name, cfg_key, coerce in arg_overrides:
        value = getattr(args, arg_name, None)
        if value is not None:
            base_updates[cfg_key] = coerce(value)
    base_updates["episode_length"] = int(task_config["episode_length"])
    cfg = cfg.with_updates(base_updates)

    if getattr(args, "num_envs", None) is not None:
        cfg = cfg.with_updates({
            "rollout_steps_per_update": int(cfg.num_envs) * int(cfg.episode_length),
        })

    scale_value = getattr(args, "scale", None)
    scale = int(scale_value) if scale_value is not None and int(scale_value) >= 2 else 1
    total_steps = int(getattr(cfg, "total_steps", 0)) * scale
    log_interval_updates = max(1, int(cfg.log_interval))
    rollout_steps_per_update, num_envs, rollout_horizon = get_collection_shape(cfg, scale)
    raw_cost_limits = constraint_spec.cost_limits if constraint_spec is not None else ()
    discounted = getattr(cfg, "discounted", COST_FIELD_DEFAULTS["discounted"])
    episode_length = int(cfg.episode_length)
    gamma = float(cfg.gamma)
    if discounted and abs(gamma - 1.0) >= 1e-9:
        cost_scale = ((1.0 - gamma**episode_length) / (1.0 - gamma)) / episode_length
    else:
        cost_scale = 1.0
    derived_cost_limits = tuple(float(limit) * cost_scale for limit in raw_cost_limits)
    num_constraints = len(raw_cost_limits) if getattr(cfg, "use_cost", False) else 0

    user_zeta = getattr(cfg, "con_zeta", None)
    if user_zeta is None and derived_cost_limits:
        con_zeta_value = float(min(derived_cost_limits))
    elif user_zeta is None:
        con_zeta_value = 0.0
    else:
        con_zeta_value = float(user_zeta)

    common_updates: dict[str, Any] = {
        "raw_cost_limits": tuple(raw_cost_limits),
        "derived_cost_limits": tuple(derived_cost_limits),
        "cost_limits": jnp.asarray(derived_cost_limits, dtype=jnp.float32).reshape(-1)[:num_constraints],
        "discounted": discounted,
        "num_constraints": num_constraints,
        "num_envs": num_envs,
        "rollout_horizon": rollout_horizon,
        "rollout_steps_per_update": rollout_steps_per_update,
        "update_batch_size": int(cfg.update_batch_size) * scale,
        "total_steps": total_steps,
        "total_updates": max(1, total_steps // rollout_steps_per_update),
        "log_interval_updates": log_interval_updates,
        "con_zeta": con_zeta_value,
    }

    compile_identity_base = (
        algo_name, task_name, "on_policy",
        ",".join(str(n) for n in getattr(constraint_spec, "constraint_names", ())),
        ",".join(str(v) for v in raw_cost_limits),
    )
    return cfg.with_updates({
        **common_updates,
        "compile_identity_base": compile_identity_base,
    })

def build_algo_train_config_from_saved_config(saved_config: dict[str, Any]) -> AlgoTrainConfig:
    schema_version = int(saved_config.get("saved_config_schema_version", 0))
    if schema_version < 2:
        raise ValueError(
            "Unsupported saved config schema. Only schema version 2+ "
            "checkpoints are supported."
        )
    config_values = dict(saved_config)
    for key in ("saved_config_schema_version", "constraint_names", "constraint_cost_limits"):
        config_values.pop(key, None)
    if "derived_cost_limits" not in config_values:
        config_values["derived_cost_limits"] = list(config_values.get("raw_cost_limits", ()))
    for field_name, default_value in COST_FIELD_DEFAULTS.items():
        if field_name not in config_values:
            config_values[field_name] = default_value
    config_values.pop("cost_limit_spec", None)
    config_values.setdefault("algo_name", str(config_values.get("algo", "unknown")))
    return AlgoTrainConfig.from_dict(config_values)

@dataclass(frozen=True)
class TaskEnvConfig:
    task_name: str
    task_config: Mapping[str, Any]
    episode_length: int
    num_envs: int
    gamma: float

def build_task_env_config(
    *,
    args: Any,
    algo_train_config: Any,
) -> TaskEnvConfig:
    task_name = str(args.task)
    task_config = getattr(algo_train_config, "task_config", None)
    if not task_config:
        task_config = load_task_config(task_name)
    episode_length = int(task_config["episode_length"])
    return TaskEnvConfig(
        task_name=task_name,
        task_config=dict(task_config),
        episode_length=episode_length,
        num_envs=int(algo_train_config.num_envs),
        gamma=float(algo_train_config.gamma),
    )

def build_task_env_config_from_saved_config(
    saved_config: dict[str, Any],
    *,
    algo_train_config: Any,
) -> TaskEnvConfig:
    task_name = str(algo_train_config.task_name)
    task_config = saved_config.get("task_config")
    if not task_config:
        task_config = load_task_config(task_name)
    return TaskEnvConfig(
        task_name=task_name,
        task_config=dict(task_config),
        episode_length=int(algo_train_config.episode_length),
        num_envs=int(algo_train_config.num_envs),
        gamma=float(algo_train_config.gamma),
    )

def _name2id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(model, objtype, name))

def _build_mj_model(
    mjcf: mujoco.MjSpec, objects: dict[str, Any], agent: PointAgent,
) -> tuple[mujoco.MjModel, mjx.Model]:
    n_g = int(agent.collision_geom_count)
    n_o = sum(
        cfg.num * int(cfg.collision_geom_count)
        for cfg in objects.values() if cfg.contact
    )
    budget = n_g * (n_o + 1) + n_o
    mjcf.nconmax = budget
    mjcf.njmax = budget
    mj_model = mjcf.compile()
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON  # pyright: ignore[reportAttributeAccessIssue]
    iters = int(os.environ.get("MJX_SOLVER_ITER", "1"))
    ls_iters = int(os.environ.get("MJX_SOLVER_LS", "4"))
    if os.environ.get("MJX_EULERDAMP", "off").lower() not in ("on", "1", "true"):
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP  # pyright: ignore[reportAttributeAccessIssue]
    mj_model.opt.iterations = iters
    mj_model.opt.ls_iterations = ls_iters
    return mj_model, mjx.put_model(mj_model)

def _build_constraint(
    constraint_config: Mapping[str, Any],
    objects: dict[str, Any],
    mj_model: mujoco.MjModel,
    agent_gids: jnp.ndarray,
) -> ConstraintConfig:
    name = str(constraint_config["name"])
    cost_limit = float(constraint_config["cost_limit"])
    cost_type = str(constraint_config.get("type", "binary"))
    if name == "out_of_boundary":
        boundaries = tuple(
            (0 if w[0] == "x" else 1, w[1])
            for w in objects["sigwalls"].walls
        )
        return ConstraintConfig(
            name=name,
            cost_limit=cost_limit,
            compute_fn=compute_out_of_boundary,
            params={"boundaries": boundaries},
            cost_type=cost_type,
        )
    if name in ("velocity_limit_high", "velocity_limit_low"):
        return ConstraintConfig(
            name=name, cost_limit=cost_limit,
            compute_fn={
                "velocity_limit_high": compute_velocity_limit_high,
                "velocity_limit_low": compute_velocity_limit_low,
            }[name],
            params={"threshold": float(constraint_config["threshold"])},
            cost_type=cost_type,
        )
    raise ValueError(f"Unsupported constraint name: {name}")

def _build_object_config(name: str, object_config: Mapping[str, Any], num: int) -> Any:
    cls = get_object_config_class(name)
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {"num": num}
    for key, value in object_config.items():
        if key in field_names and key != "num":
            if key == "walls" and isinstance(value, list):
                kwargs[key] = tuple((str(w["axis"]), float(w["coord"])) for w in value)
            else:
                kwargs[key] = value
    return cls(**kwargs)

def _agent_geom_ids(mj_model: mujoco.MjModel) -> jnp.ndarray:
    root_bid = _name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "agent")
    body_ids: list[int] = []
    for bid in range(int(mj_model.nbody)):
        current = bid
        while current > 0 and current != root_bid:
            current = int(mj_model.body_parentid[current])
        if current == root_bid:
            body_ids.append(bid)
    return jnp.asarray(
        [gid for gid in range(int(mj_model.ngeom))
         if int(mj_model.geom_bodyid[gid]) in body_ids],
        dtype=jnp.int32,
    )

def _object_geom_ids(mj_model: mujoco.MjModel, cfg: Any) -> jnp.ndarray:
    return jnp.asarray(
        [_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, cfg.geom_id(i))
         for i in range(cfg.num)],
        dtype=jnp.int32,
    )

def build_env(
    *,
    task_env_config: TaskEnvConfig,
    episode_length: int,
    batch_size: int,
    gamma: float,
) -> Wrapper:
    raw = dict(task_env_config.task_config)

    objects: dict[str, Any] = {}
    for object_config in raw["objects"]:
        name = str(object_config["name"])
        num = int(object_config["num"])
        objects[name] = _build_object_config(name, object_config, num)

    task_type = str(raw["task_type"])
    task_specific = dict(raw.get("task_specific", {}))
    placements_extents = tuple(raw["placements_extents"])

    agent = PointAgent()
    agent_overrides = dict(raw.get("agent", {}))
    if agent_overrides:
        known = {f.name for f in dataclasses.fields(agent)}
        unknown = set(agent_overrides) - known
        if unknown:
            raise ValueError(f"Unknown agent YAML keys: {sorted(unknown)}")
        agent = dataclasses.replace(agent, **agent_overrides)
    mjcf = mujoco.MjSpec.from_file(agent.xml_path)  # pyright: ignore[reportAttributeAccessIssue]
    for obj_cfg in objects.values():
        obj_cfg.add(mjcf.worldbody, mjcf)
    mj_model, mjx_model = _build_mj_model(mjcf, objects, agent)

    for name, cfg in list(objects.items()):
        objects[name] = cfg.resolve(mj_model)
    agent_gids = _agent_geom_ids(mj_model)
    constraints = tuple(sorted(
        (_build_constraint(item, objects, mj_model, agent_gids)
         for item in raw.get("constraints", [])),
        key=lambda item: item.name,
    ))

    sim_dt = float(mj_model.opt.timestep)
    n_substeps = int(agent.n_substeps)

    env_cls = get_task_env_class(task_type)
    raw_env = env_cls(
        mj_model=mj_model,
        mjx_model=mjx_model,
        agent=agent,
        objects=objects,
        placements_extents=placements_extents,
        constraints=constraints,
        action_size=int(agent.action_size),
        ctrl_dt=sim_dt * n_substeps,
        sim_dt=sim_dt,
        n_substeps=n_substeps,
        agent_geom_ids=agent_gids,
        **task_specific,
    )
    env = VmapWrapper(raw_env, batch_size=batch_size)
    env = EpisodeWrapper(env, episode_length, gamma=gamma)
    return env

@struct.dataclass
class BatchState:
    obs: Any
    actions: Any
    rewards: Any
    costs: Any
    dones: Any
    terminated: Any
    next_obs: Any
    discounted_rewards: Any
    discounted_costs: Any

    @classmethod
    def build_zero(
        cls,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        nc: int,
    ) -> "BatchState":
        dtype = jnp.float32
        return cls(
            obs=jnp.zeros((num_envs, obs_dim), dtype=dtype),
            actions=jnp.zeros((num_envs, act_dim), dtype=dtype),
            rewards=jnp.zeros((num_envs,), dtype=dtype),
            costs=jnp.zeros((num_envs, nc), dtype=dtype),
            dones=jnp.zeros((num_envs,), dtype=dtype),
            terminated=jnp.zeros((num_envs,), dtype=dtype),
            next_obs=jnp.zeros((num_envs, obs_dim), dtype=dtype),
            discounted_rewards=jnp.zeros((num_envs,), dtype=dtype),
            discounted_costs=jnp.zeros((num_envs, nc), dtype=dtype),
        )

    @classmethod
    def from_step(
        cls, *, obs: Any, action: Any, next_env: Any, next_obs: Any,
    ) -> "BatchState":
        info = next_env.info
        return cls(
            obs=obs,
            actions=action,
            rewards=next_env.reward,
            costs=info["step_cost"],
            dones=next_env.done.astype(jnp.float32),
            terminated=next_env.terminated.astype(jnp.float32),
            next_obs=next_obs,
            discounted_rewards=info["discounted_rewards"],
            discounted_costs=info["discounted_costs"],
        )

@struct.dataclass
class OnBatchState(BatchState):
    logp: Any = None
    steps: Any = None
    policy_mean: Any = None
    policy_log_std: Any = None
    returns: Any = None
    advantages: Any = None
    values: Any = None
    next_values: Any = None
    bootstrap: Any = None
    cost_signal: Any = None
    cost_limit: Any = None

    @classmethod
    def build_zero(
        cls,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        nc: int,
    ) -> "OnBatchState":
        from dataclasses import fields as dc_fields
        core = BatchState.build_zero(num_envs, obs_dim, act_dim, nc)
        dtype = jnp.float32
        return cls(
            **{f.name: getattr(core, f.name) for f in dc_fields(BatchState)},
            logp=jnp.zeros((num_envs,), dtype=dtype),
            steps=jnp.zeros((num_envs,), dtype=dtype),
            policy_mean=jnp.zeros((num_envs, act_dim), dtype=dtype),
            policy_log_std=jnp.zeros((num_envs, act_dim), dtype=dtype),
        )

    @classmethod
    def from_step(
        cls,
        obs: Any,
        action: Any,
        next_env: Any,
        zero_steps: Any,
        steps: Any,
        sample_stats: Any,
    ) -> "OnBatchState":
        logp, act_mean, act_lstd = sample_stats
        return super().from_step(
            obs=obs,
            action=action,
            next_env=next_env,
            next_obs=next_env.info.get("terminal_obs", next_env.obs),
        ).replace(
            logp=logp,
            steps=steps.astype(jnp.float32),
            policy_mean=act_mean,
            policy_log_std=act_lstd,
        )

class RolloutBuffer:
    def __init__(
        self,
        num_envs: int,
        sample_length: int,
        zero_batch_state: BatchState,
    ) -> None:
        self.num_envs = num_envs
        self.length = sample_length
        self.zero_batch_state = zero_batch_state
        self._prototype = jax.tree.map(lambda x: x[0], zero_batch_state)
        self._init = partial(
            trajectory_buffer.init,
            add_batch_size=num_envs,
            max_length_time_axis=sample_length,
        )
        self._add = trajectory_buffer.add

    def init(self) -> RolloutBufferState:
        return self._init(self._prototype)

    def add(self, state: RolloutBufferState, batch_state: BatchState) -> RolloutBufferState:
        return self._add(
            state,
            jax.tree.map(lambda x: x[:, None], batch_state),
        )

    def sample(self, state: RolloutBufferState) -> BatchState:
        return jax.tree.map(lambda x: x.swapaxes(0, 1), state.experience)

def rollout_env(
    runtime_state: RuntimeState,
    *,
    buf: RolloutBuffer,
    num_steps: int,
    env_step_fn: Callable,
    sample_action_fn: Callable,
    actor_obs_fn: Callable,
    build_batch_state_fn: Callable,
) -> RuntimeState:
    policy_state = runtime_state.policy_state
    zero_steps = jnp.zeros((buf.num_envs,), dtype=jnp.int32)

    def _step(carry, _):
        env_st, rng_, b_st = carry
        rng_, rng_act = jax.random.split(rng_)
        obs = env_st.obs
        steps = env_st.info["steps"]
        actor_obs = actor_obs_fn(policy_state, obs)
        action, sample_stats = sample_action_fn(
            policy_state.actor_params, actor_obs, rng_act,
            getattr(policy_state, "obs_rms", None),
        )
        next_env = env_step_fn(env_st, action)
        batch_state = build_batch_state_fn(obs, action, next_env, zero_steps, steps, sample_stats)
        return (next_env, rng_, buf.add(b_st, batch_state)), None

    (final_env, final_rng, final_buf), _ = jax.lax.scan(
        _step,
        (runtime_state.env_state, policy_state.rng, runtime_state.rollout_buf_state),
        None, length=num_steps,
    )
    return runtime_state.replace(
        policy_state=policy_state.replace(rng=final_rng),
        env_state=final_env,
        rollout_buf_state=final_buf,
    )

def compute_episode_metrics(
    *,
    batch: Any,
    env_state: EnvState,
    cfg: Any,
    compute_cost_signal_fn: Callable,
) -> dict[str, jax.Array]:
    num_groups = cfg.rollout_steps_per_update / cfg.episode_length
    cost_signal = compute_cost_signal_fn(batch.discounted_costs, batch.costs, num_groups)
    done_f = jnp.any(batch.dones > 0, axis=0).astype(jnp.float32)
    safe_count = jnp.maximum(jnp.sum(done_f), 1.0)
    terminal = env_state.info.get("terminal_episode_metrics", {})
    cast_fn = lambda x: jnp.asarray(x, dtype=jnp.float32)
    return {
        "Ep/reward": jnp.sum(batch.rewards) / num_groups,
        "Ep/d_reward": jnp.sum(batch.discounted_rewards) / num_groups,
        "Ep/r_cost": jnp.sum(batch.costs, axis=(0, 1)) / num_groups,
        "Ep/d_cost": jnp.sum(batch.discounted_costs, axis=(0, 1)) / num_groups,
        "cost_signal": cost_signal,
        "cost_limit": cfg.cost_limits,
        "Epd/reward": jnp.sum(cast_fn(terminal["sum_reward"]) * done_f) / safe_count,
        "Epd/d_reward": jnp.sum(cast_fn(terminal["sum_d_reward"]) * done_f) / safe_count,
        "Epd/r_cost": jnp.sum(cast_fn(terminal["sum_r_costs"]) * done_f[..., None], axis=0) / safe_count,
        "Epd/d_cost": jnp.sum(cast_fn(terminal["sum_d_costs"]) * done_f[..., None], axis=0) / safe_count,
    }

_REWARD_COL_ORDER = (
    "Ep/reward", "Ep/d_reward",
    "Epd/reward", "Epd/d_reward",
    "reward_critic_loss", "reward_value_mean",
    "reward_grad_norm", "actor_grad_norm",
)

_COST_COL_STEMS = (
    "Ep/r_cost", "Ep/d_cost",
    "Epd/r_cost", "Epd/d_cost",
    "cost_critic_loss", "cost_value_mean",
    "cost_grad_norm",
)

def _fmt(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    return str(v)

def format_epoch_log(metrics: dict[str, Any], episode_only: bool = True) -> list[str]:
    cost_pairs: dict[str, tuple[float, float]] = {}
    for k in sorted(metrics):
        if k.startswith("cost_signal"):
            suffix = k[len("cost_signal"):]
            cost_pairs[suffix] = (
                float(metrics.get(k, 0.0)),
                float(metrics.get(f"cost_limit{suffix}", 0.0)),
            )

    reward_col: list[tuple[str, str]] = []
    for key in _REWARD_COL_ORDER:
        if key in metrics:
            reward_col.append((key, _fmt(metrics[key])))

    cost_cols: list[list[tuple[str, str]]] = []
    for suffix in sorted(cost_pairs):
        sig, lim = cost_pairs[suffix]
        col: list[tuple[str, str]] = [(f"cost/limit{suffix}", f"{sig:.1f}/{lim:.1f}")]
        for stem in _COST_COL_STEMS:
            key = f"{stem}{suffix}"
            if key in metrics:
                col.append((key, _fmt(metrics[key])))
        cost_cols.append(col)

    columns = [c for c in [reward_col, *cost_cols] if c]
    if not columns:
        return []

    col_kw = [max(len(k) for k, _ in c) for c in columns]
    col_vw = [max(len(v) for _, v in c) for c in columns]
    num_rows = max(len(c) for c in columns)

    lines: list[str] = []
    for i in range(num_rows):
        cells: list[str] = []
        for j, col in enumerate(columns):
            if i < len(col):
                k, v = col[i]
                cells.append(f"{k:<{col_kw[j]}} {v:>{col_vw[j]}}")
            else:
                cells.append(" " * (col_kw[j] + 1 + col_vw[j]))
        lines.append(" | ".join(cells))
    return lines

def merge_raw_metrics(
    *,
    episode_metrics: dict[str, Any],
    update_metrics: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(episode_metrics)
    for key, value in update_metrics.items():
        if key == "state":
            continue
        merged[key] = value
    return merged

def build_host_emit_dispatcher(
    *,
    log_paths: dict[int, str] | None = None,
    constraint_names: tuple[str, ...] | None = None,
):
    handles: dict[int, Any] = {}
    if isinstance(log_paths, dict):
        for idx, path in log_paths.items():
            handles[int(idx)] = open(path, "a", encoding="utf-8")

    n_names = len(constraint_names) if constraint_names else 0

    def _emit(args) -> None:
        raw_scalars, update_index, seed_idx = args
        scalars: dict[str, float] = {}
        for k, v in raw_scalars.items():
            arr = jnp.asarray(v)
            if arr.ndim == 0:
                scalars[k] = float(arr)
            elif arr.ndim == 1 and arr.shape[0] > 0:
                use_names = n_names > 0 and arr.shape[0] == n_names
                for i in range(arr.shape[0]):
                    suffix = constraint_names[i] if use_names else i
                    scalars[f"{k}_{suffix}"] = float(arr[i])
        step_num = int(update_index) + 1 if update_index is not None else 0
        log_lines = format_epoch_log(scalars)
        prefix = f"#{step_num} "
        indent = " " * len(prefix)
        if log_lines:
            body = ("\n" + indent).join([prefix + log_lines[0]] + log_lines[1:])
            width = max(len(prefix + log_lines[0]), *(len(indent + ln) for ln in log_lines[1:]))
            sep = "-" * width
            block = f"{sep}\n{body}\n{sep}"
        else:
            block = prefix
        print(block, flush=True)
        handle = handles.get(0)
        if handle is not None:
            handle.write(block + "\n")
            handle.flush()
    return _emit

class CallbackLoggingHook:
    def __init__(self, host_emit) -> None:
        self.host_emit = host_emit

    def emit(self, raw_scalars: dict[str, Any], debug: Any | None = None, seed_idx: Any | None = None) -> None:
        jax.debug.callback(self.host_emit, (raw_scalars, debug, seed_idx), ordered=False)

def build_logging_hook(
    *,
    log_paths: dict[int, str] | None = None,
    constraint_names: tuple[str, ...] | None = None,
) -> CallbackLoggingHook:
    dispatcher = build_host_emit_dispatcher(
        log_paths=log_paths,
        constraint_names=constraint_names,
    )
    return CallbackLoggingHook(host_emit=dispatcher)

def _to_yaml_safe(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    return obj

_CONFIG_EXCLUDE_KEYS = frozenset({
    "constraint_spec", "compile_identity_base",
    "task_config", "cost_limits",
})

def _build_saved_config(algo_train_config: Any) -> dict[str, Any]:
    raw = algo_train_config.to_dict(exclude=_CONFIG_EXCLUDE_KEYS)
    cs = getattr(algo_train_config, "constraint_spec", None)
    if cs is not None and hasattr(cs, "constraint_names"):
        raw["constraint_names"] = list(getattr(cs, "constraint_names", ()))
        raw["constraint_cost_limits"] = [
            float(x) for x in getattr(cs, "cost_limits", ())
        ]
    return {"saved_config_schema_version": 2, **raw}

def build_run_context(
    algo_train_config: Any,
    *,
    seed: int,
    run_id: str | None = None,
    preset: str | None = None,
) -> RunContext:
    if run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        hex4 = uuid.uuid4().hex[:4]
        run_id = f"{ts}_{hex4}"
    algo_name = str(algo_train_config.algo_name)
    task_name = str(algo_train_config.task_name)
    policy_type = str(algo_train_config.policy_type)
    experiment_name = str(
        Path(task_name) / algo_name / f"{run_id}_s{seed}"
    )
    _constraint_spec = getattr(algo_train_config, "constraint_spec", None)
    resolved_constraint = ",".join(
        str(n) for n in getattr(_constraint_spec, "constraint_names", ())
    ) or "none"
    out_dir = Path("results").resolve() / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    progress_csv = out_dir / "progress.csv"
    meta_path = out_dir / "metadata.yaml"
    if not meta_path.exists():
        meta_path = save_metadata(
            out_dir,
            algo=algo_name,
            task=task_name,
            seed=int(seed),
            constraint=resolved_constraint,
            extra_meta={
                "env_id": task_name,
                "policy_type": policy_type,
            },
        )
    config_path = out_dir / "config.yaml"
    if not config_path.exists():
        saved_config = _build_saved_config(algo_train_config)
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                _to_yaml_safe(saved_config), f,
                default_flow_style=False, allow_unicode=True,
            )
    return RunContext(
        experiment_name=experiment_name,
        out_dir=out_dir,
        progress_csv=progress_csv,
        meta_path=meta_path,
    )

def save_run_metadata(
    run_context: RunContext,
    execution_result: Any,
    algo_train_config: Any = None,
) -> None:
    meta_data = cast(dict[str, Any], load_run_metadata(run_context.meta_path))
    meta_data["ended_at"] = datetime.now(timezone.utc).isoformat()
    meta_data["exit_status"] = 0 if execution_result is not None else 1
    if execution_result is not None:
        meta_data["static_compile_fingerprint"] = execution_result.compile_fingerprint
        meta_data["prepare_time_seconds"] = float(execution_result.prepare_time_seconds)
        meta_data["compile_time_seconds"] = float(execution_result.compile_time_seconds)
        meta_data["training_time_seconds"] = float(execution_result.training_time_seconds)
        meta_data["total_time_seconds"] = float(execution_result.total_time_seconds)
    if algo_train_config is not None:
        meta_data["total_updates"] = int(algo_train_config.total_updates)
    run_context.meta_path.write_text(
        yaml.safe_dump(
            _to_yaml_safe(meta_data),
            default_flow_style=False, allow_unicode=True,
        ),
        encoding="utf-8",
    )

def load_run_metadata(path: Path) -> dict[str, Any]:
    loaded_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded_data is None:
        return {}
    if not isinstance(loaded_data, dict):
        raise ValueError("metadata YAML must decode to a mapping")
    return dict(loaded_data)

def _get_git_info() -> tuple[str, bool]:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "unknown"

    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        dirty = bool(status)
    except Exception:
        dirty = False

    return commit, dirty

def _get_key_packages() -> dict[str, str]:
    packages = ["jax", "jaxlib", "mujoco", "dm-tree", "optax", "flax"]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not installed"
    return versions

def save_metadata(
    exp_dir: Path,
    algo: str,
    task: str,
    seed: int,
    constraint: str,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    git_commit, git_dirty = _get_git_info()
    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "command": " ".join(sys.argv),
        "hostname": py_platform.node(),
        "python_version": py_platform.python_version(),
        "key_packages": _get_key_packages(),
        "notes": "",
        "algo": algo,
        "task": task,
        "seed": seed,
        "constraint": constraint,
        "exit_status": None,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = Path(exp_dir) / "metadata.yaml"
    with meta_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            _to_yaml_safe(meta), f,
            default_flow_style=False, allow_unicode=True,
        )
    return meta_path

class Pipeline:
    def __init__(
        self,
        *,
        policy: OnPolicyBase,
        env: Wrapper,
        rollout_buf: RolloutBuffer,
    ) -> None:
        self.policy = policy
        self.env = env
        self.rollout_buf = rollout_buf
        self.sample_action_fn = policy.actor.sample_action
        if policy.cfg.discounted:
            self.compute_cost_signal = lambda d, r, n: jnp.sum(d, axis=(0, 1)) / n
        else:
            self.compute_cost_signal = lambda d, r, n: jnp.sum(r, axis=(0, 1)) / n

    def build_init_runtime_state(self) -> Callable[[jax.Array], RuntimeState]:
        policy = self.policy

        def _init(rng: jax.Array) -> RuntimeState:
            key_init, key_env, key_next = jax.random.split(rng, 3)
            policy_state = policy.init_state(key_init).replace(rng=key_next)
            return RuntimeState(
                policy_state=policy_state,
                env_state=self.env.reset(key_env),
                rollout_buf_state=self.rollout_buf.init(),
            )

        return _init

    def prepare_runtime_state(self, runtime_state: RuntimeState) -> RuntimeState:
        return runtime_state

def _compute_returns_fn(policy: OnPolicyBase) -> Callable:
    cfg = policy.cfg
    num_constraints = cfg.num_constraints
    finite = getattr(cfg, "horizon", "infinite") == "finite"

    def _run(batch):
        returns, _ = compute_targets(
            batch, num_constraints,
            reward_gamma=cfg.gamma, reward_lambda=cfg.gae_lambda,
            cost_lambda=cfg.cost_gae_lambda,
            finite=finite,
        )
        return batch.replace(returns=returns)

    return _run

def _compute_advantages_fn(policy: OnPolicyBase) -> Callable:
    cfg = policy.cfg
    num_constraints = cfg.num_constraints
    finite = getattr(cfg, "horizon", "infinite") == "finite"
    adv_standardize = getattr(cfg, "adv_standardize", True)

    def _run(batch):
        _, advantages = compute_targets(
            batch, num_constraints,
            reward_gamma=cfg.gamma, reward_lambda=cfg.gae_lambda,
            cost_lambda=cfg.cost_gae_lambda,
            finite=finite,
        )
        if adv_standardize:
            reward_slot = advantages[..., 0]
            reward_slot = (reward_slot - jnp.mean(reward_slot)) / (jnp.std(reward_slot) + 1e-8)
            advantages = advantages.at[..., 0].set(reward_slot)
            if num_constraints > 0:
                cost_mean = jnp.mean(advantages[..., 1:], axis=(0, 1), keepdims=True)
                advantages = advantages.at[..., 1:].add(-cost_mean)
        return batch.replace(advantages=advantages)

    return _run

class OnPolicyPipeline(Pipeline):
    def __init__(self, *, policy: OnPolicyBase, env: Wrapper) -> None:
        super().__init__(
            policy=policy,
            env=env,
            rollout_buf=RolloutBuffer(
                num_envs=policy.cfg.num_envs,
                sample_length=policy.cfg.rollout_horizon,
                zero_batch_state=OnBatchState.build_zero(
                    policy.cfg.num_envs,
                    policy.cfg.obs_size,
                    policy.cfg.action_size,
                    policy.cfg.num_constraints,
                ),
            ),
        )
        self.compute_returns = _compute_returns_fn(policy)
        self.compute_advantages = _compute_advantages_fn(policy)
        if policy.cfg.obs_normalize:
            self.actor_obs = lambda policy_state, obs: policy_state.obs_rms.normalize(obs)
        else:
            self.actor_obs = lambda policy_state, obs: obs
        self._rollout_env = partial(
            rollout_env,
            buf=self.rollout_buf,
            num_steps=policy.cfg.rollout_horizon,
            env_step_fn=self.env.step,
            sample_action_fn=self.sample_action_fn,
            actor_obs_fn=self.actor_obs,
            build_batch_state_fn=OnBatchState.from_step,
        )
        self._compute_episode_metrics = partial(
            compute_episode_metrics,
            cfg=policy.cfg,
            compute_cost_signal_fn=self.compute_cost_signal,
        )

    def prepare_batch_for_critic(
        self,
        runtime_state: RuntimeState,
        *,
        collected_rollout: OnBatchState,
        cost_signal: jax.Array,
        cost_limit: jax.Array,
    ) -> OnBatchState:
        policy = self.policy
        policy_state = runtime_state.policy_state
        normalized_obs = self.actor_obs(policy_state, collected_rollout.obs)
        next_obs = self.actor_obs(policy_state, collected_rollout.next_obs)
        values, next_values = batch_critic_values(
            normalized_obs,
            next_obs,
            policy.critic.apply,
            policy_state.critic_params,
            policy.cfg.num_constraints,
        )
        rollout = collected_rollout.replace(
            obs=normalized_obs,
            next_obs=next_obs,
            values=values,
            next_values=next_values,
            bootstrap=next_values[-1],
            cost_signal=cost_signal,
            cost_limit=cost_limit,
        )
        return self.compute_returns(rollout)

    def prepare_actor_advantages(
        self,
        runtime_state: RuntimeState,
        rollout: OnBatchState,
    ) -> OnBatchState:
        policy = self.policy
        policy_state = runtime_state.policy_state
        values, next_values = batch_critic_values(
            rollout.obs,
            rollout.next_obs,
            policy.critic.apply,
            policy_state.critic_params,
            policy.cfg.num_constraints,
        )
        rollout = rollout.replace(
            values=values,
            next_values=next_values,
            bootstrap=next_values[-1],
        )
        return self.compute_advantages(rollout)

    def _flatten_for_update(self, rollout: OnBatchState) -> OnBatchState:
        cfg = self.policy.cfg
        nc = cfg.num_constraints
        kwargs = dict(
            obs=rollout.obs.reshape((-1, cfg.obs_size)),
            actions=rollout.actions.reshape((-1, cfg.action_size)),
            logp=rollout.logp.reshape((-1,)),
            policy_mean=rollout.policy_mean.reshape((-1, cfg.action_size)),
            policy_log_std=rollout.policy_log_std.reshape((-1, cfg.action_size)),
        )
        if rollout.returns is not None:
            kwargs["returns"] = rollout.returns.reshape((-1, 1 + nc))
        if rollout.advantages is not None:
            kwargs["advantages"] = rollout.advantages.reshape((-1, 1 + nc))
        return rollout.replace(**kwargs)

    def one_update(
        self,
        runtime_state: RuntimeState,
        update_index: jax.Array,
        *,
        logging_hook: Any | None = None,
    ) -> tuple[RuntimeState, Metrics]:
        policy = self.policy
        runtime_state = self._rollout_env(runtime_state)
        collected_rollout = self.rollout_buf.sample(runtime_state.rollout_buf_state)
        episode_metrics = self._compute_episode_metrics(
            batch=collected_rollout,
            env_state=runtime_state.env_state,
        )

        rollout_TN = self.prepare_batch_for_critic(
            runtime_state,
            collected_rollout=collected_rollout,
            cost_signal=episode_metrics["cost_signal"],
            cost_limit=episode_metrics["cost_limit"],
        )
        batch_for_critic = self._flatten_for_update(rollout_TN)

        policy_state, critic_metrics = policy.update_critics(
            runtime_state.policy_state, batch_for_critic,
        )
        runtime_state = runtime_state.replace(policy_state=policy_state)

        rollout_TN = self.prepare_actor_advantages(runtime_state, rollout_TN)
        batch_for_actor = self._flatten_for_update(rollout_TN)

        policy_state, actor_metrics = policy.update_actor_only(
            policy_state, batch_for_actor,
        )
        policy_state, lag_metrics = policy.update_lagrange(
            policy_state, batch_for_actor,
        )
        policy_state = policy._update_obs_stats(
            batch_for_actor.obs[..., :policy.cfg.obs_size], policy_state,
        )

        merged = merge_raw_metrics(
            episode_metrics=episode_metrics,
            update_metrics={**critic_metrics, **actor_metrics, **lag_metrics},
        )
        if logging_hook is not None:
            logging_hook.emit(merged, debug=update_index + 1, seed_idx=None)
        return runtime_state.replace(policy_state=policy_state), merged

_COMPILE_FINGERPRINT_KEYS = (
    "discounted",
    "advantage_estimator", "num_constraints", "num_envs", "episode_length",
    "rollout_horizon", "update_epochs",
    "raw_cost_limits", "target_kl", "total_updates", "rollout_steps_per_update",
)

def build_compile_fingerprint(
    cfg: AlgoTrainConfig,
    *,
    compile_identity_base: tuple[str, ...] = (),
    env_signature: Mapping[str, Any] | None = None,
) -> str:
    cfg_dict = cfg.to_dict()
    compile_key_data: dict[str, Any] = {
        key: (list(cfg_dict[key]) if isinstance(cfg_dict[key], (list, tuple)) else cfg_dict[key])
        for key in _COMPILE_FINGERPRINT_KEYS if key in cfg_dict
    }
    compile_key_data["compile_identity_base"] = list(compile_identity_base)
    compile_key_data["env_signature"] = (
        {str(key): env_signature[key] for key in sorted(env_signature)}
        if env_signature is not None else {}
    )
    return hashlib.sha1(json.dumps(compile_key_data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def build_compiled_plan(
    *,
    policy: OnPolicyBase,
    pipeline: Pipeline,
    progress_path: str,
    use_logging: bool = True,
    log_paths: dict[int, str] | None = None,
) -> CompiledTrainingPlan:
    logging_hook = None
    if use_logging:
        env = getattr(policy, "env", None)
        env_unwrapped = getattr(env, "unwrapped", env) if env is not None else None
        constraint_names = tuple(
            c.name for c in getattr(env_unwrapped, "constraints", ())
        ) or None
        if log_paths is not None:
            logging_hook = build_logging_hook(
                log_paths=log_paths,
                constraint_names=constraint_names,
            )
        elif progress_path:
            log_path = Path(progress_path).parent / "logs" / "train.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logging_hook = build_logging_hook(
                log_paths={0: str(log_path)},
                constraint_names=constraint_names,
            )

    one_update = partial(pipeline.one_update, logging_hook=logging_hook)
    init_runtime_state = pipeline.build_init_runtime_state()

    cfg = policy.cfg

    def _run_full_jit(runtime_state: RuntimeState) -> tuple[RuntimeState, FullRunResult]:
        next_state, all_metrics = jax.lax.scan(
            one_update, runtime_state, jnp.arange(cfg.total_updates, dtype=jnp.int32),
        )
        return next_state, FullRunResult(
            last_metrics=jax.tree.map(lambda x: x[-1], all_metrics),
            all_metrics=all_metrics,
        )

    return CompiledTrainingPlan(
        init_runtime_state=init_runtime_state,
        prepare_runtime_state=pipeline.prepare_runtime_state,
        one_update=one_update,
        run_full_jit=jax.jit(_run_full_jit),
        compile_fingerprint=build_compile_fingerprint(
            cfg, compile_identity_base=cfg.compile_identity_base,
            env_signature={
                "obs_size": policy.cfg.obs_size,
                "action_size": policy.cfg.action_size,
            },
        ),
    )

_CSV_COLUMN_ORDER = (
    "time_s",
    "Ep/reward", "Ep/d_reward", "Ep/r_cost", "Ep/d_cost",
    "Epd/reward", "Epd/d_reward", "Epd/r_cost", "Epd/d_cost",
    "cost_signal", "cost_limit",
    "actor_loss", "reward_critic_loss", "cost_critic_loss",
    "reward_value_mean", "cost_value_mean",
    "reward_grad_norm", "cost_grad_norm", "actor_grad_norm",
    "entropy", "ratio_mean",
    "lagrange", "lagrange_delta",
)

def save_model(path: str | Path, state: dict[str, Any]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    np_leaves = [np.asarray(leaf) for leaf in leaves]
    np.savez(path / "params.npz", *np_leaves)
    with open(path / "treedef.pkl", "wb") as f:
        pickle.dump(treedef, f)
    return path

def get_num_devices() -> int:
    return len(jax.devices())

def _order_metric_keys(cols: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    remaining = set(cols)
    for prefix in _CSV_COLUMN_ORDER:
        for k in sorted(remaining):
            if k == prefix or k.startswith(prefix + "_"):
                ordered.append(k)
        remaining -= set(ordered)
    ordered.extend(sorted(remaining))
    return ordered

def _save_progress_csv(
    progress_path: str,
    all_metrics: Any,
    train_seconds: float | None = None,
) -> None:
    if not progress_path or not all_metrics:
        return

    fields_data = {k: np.asarray(v) for k, v in all_metrics.items()}
    n = None
    cols: dict[str, np.ndarray] = {}
    for key, arr in fields_data.items():
        if arr.ndim == 1:
            cols[key] = arr
            n = len(arr) if n is None else n
        elif arr.ndim == 2:
            for j in range(arr.shape[1]):
                cols[f"{key}_{j}"] = arr[:, j]
            n = arr.shape[0] if n is None else n

    if not cols or n is None:
        return

    if train_seconds is not None and n > 0:
        cols["time_s"] = np.linspace(
            train_seconds / n, train_seconds, n, dtype=np.float64,
        )

    out_path = Path(progress_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = _order_metric_keys(cols)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        for i in range(n):
            writer.writerow({k: float(v[i]) for k, v in cols.items()})

    print(f"[progress] Saved {n} updates to {out_path}", flush=True)

def _save_final_model(out_dir: Path, policy_state: Any) -> None:
    state: dict[str, Any] = {"actor_params": policy_state.actor_params}
    if hasattr(policy_state, "critic_params"):
        state["critic_params"] = policy_state.critic_params
    else:
        state["reward_critic_params"] = policy_state.reward_critic_params
        state["cost_critic_params"] = policy_state.cost_critic_params
    obs_rms = getattr(policy_state, "obs_rms", None)
    if obs_rms is not None:
        state["obs_rms"] = obs_rms
    save_model(out_dir / "model", state)

def execute_training_run(
    compiled_plan: Any,
    *,
    seed: int,
    out_dir: Path,
    compile_timing: bool = True,
) -> ExecutionResult:
    base_rng = jax.random.PRNGKey(seed)
    runtime_state = compiled_plan.prepare_runtime_state(compiled_plan.init_runtime_state(base_rng))
    jax.block_until_ready(runtime_state)
    t_prepare_done = elapsed()
    _logger.info("[timing] prepare: %.1fs", t_prepare_done)
    print(f"[timing] prepare: {t_prepare_done:.1f}s", flush=True)

    run_full_jit = compiled_plan.run_full_jit

    if compile_timing:
        lowered = run_full_jit.lower(runtime_state)
        compiled_fn = lowered.compile()
        t_compile_done = elapsed()
        _logger.info("[timing] compile: %.1fs", t_compile_done - t_prepare_done)
        print(f"[timing] compile: {t_compile_done - t_prepare_done:.1f}s", flush=True)

        runtime_state, summary = compiled_fn(runtime_state)
    else:
        t_compile_done = t_prepare_done
        runtime_state, summary = run_full_jit(runtime_state)

    for v in (summary.last_metrics or {}).values():
        arr = jnp.asarray(v).ravel()
        if arr.size > 0:
            float(arr[0])
            break
    t_train_done = elapsed()
    _logger.info("[timing] train: %.1fs", t_train_done - t_compile_done)
    print(f"[timing] train: {t_train_done - t_compile_done:.1f}s", flush=True)

    _save_progress_csv(
        str(out_dir / "progress.csv"), summary.all_metrics,
        train_seconds=t_train_done - t_compile_done,
    )

    final_policy_state = runtime_state.policy_state
    _save_final_model(out_dir, final_policy_state)
    return ExecutionResult(
        compile_fingerprint=str(compiled_plan.compile_fingerprint),
        prepare_time_seconds=t_prepare_done,
        compile_time_seconds=t_compile_done - t_prepare_done,
        training_time_seconds=t_train_done - t_compile_done,
        total_time_seconds=elapsed(),
    )

__all__ = [
    "mark_start",
    "elapsed",
    "bootstrap_backend",
    "get_backend",
    "load_bool_value",
    "load_choice_value",
    "load_train_args",
    "AlgoTrainConfig",
    "get_collection_shape",
    "build_algo_train_config",
    "build_algo_train_config_from_saved_config",
    "TaskEnvConfig",
    "build_task_env_config",
    "build_task_env_config_from_saved_config",
    "build_env",
    "BatchState",
    "OnBatchState",
    "RolloutBuffer",
    "RolloutBufferState",
    "rollout_env",
    "compute_episode_metrics",
    "CallbackLoggingHook",
    "merge_raw_metrics",
    "build_logging_hook",
    "format_epoch_log",
    "build_run_context",
    "save_run_metadata",
    "load_run_metadata",
    "save_metadata",
    "Pipeline",
    "OnPolicyPipeline",
    "build_compile_fingerprint",
    "build_compiled_plan",
    "save_model",
    "get_num_devices",
    "execute_training_run",
]
