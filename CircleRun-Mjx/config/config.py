from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import optax
import yaml

@dataclass(frozen=True)
class ConstraintSpec:
    constraint_names: tuple[str, ...]
    cost_limits: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.constraint_names) != len(self.cost_limits):
            raise ValueError(
                f"length mismatch: names={len(self.constraint_names)}, "
                f"limits={len(self.cost_limits)}"
            )

    @property
    def width(self) -> int:
        return len(self.constraint_names)

def build_constraint_spec(
    names: Sequence[str],
    limits: Sequence[float],
) -> ConstraintSpec:
    sorted_pairs = sorted(zip(names, limits), key=lambda x: x[0])
    if sorted_pairs:
        sorted_names, sorted_limits = zip(*sorted_pairs)
    else:
        sorted_names, sorted_limits = (), ()
    return ConstraintSpec(
        constraint_names=tuple(sorted_names),
        cost_limits=tuple(float(x) for x in sorted_limits),
    )

ALGO_MAP: dict[str, str] = {
    "crpo":    "algos.crpo.CRPO",
    "locu":    "algos.locu.LOCU",
    "onsdac":  "algos.on_sdac.onSDAC",
    "ppolag":  "algos.ppo_lag.PPOLagrange",
    "trpolag": "algos.trpo_lag.TRPOLagrange",
}

def get_algo_class(key: str) -> type:
    k = str(key).lower()
    if k not in ALGO_MAP:
        raise ValueError(
            f"Unknown algorithm key: {key!r}. Available: {sorted(ALGO_MAP)}"
        )
    path = ALGO_MAP[k]
    module_path, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)

NETWORK_REGISTRY: dict[str, type] = {}
_networks_imported = False

def register_network(key: str):
    def _decorator(cls):
        if key in NETWORK_REGISTRY:
            existing = NETWORK_REGISTRY[key].__name__
            raise ValueError(
                f"Network key '{key}' is already registered by {existing}"
            )
        NETWORK_REGISTRY[key] = cls
        return cls
    return _decorator

def build_network(name: str, **kwargs: Any) -> Any:
    _ensure_networks_imported()
    if name not in NETWORK_REGISTRY:
        raise ValueError(
            f"Unknown network: {name!r}. Registered: {sorted(NETWORK_REGISTRY)}"
        )
    cls = NETWORK_REGISTRY[name]
    valid_params = set(inspect.signature(cls).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)

def _ensure_networks_imported():
    global _networks_imported
    if _networks_imported:
        return
    from networks import GaussianLearningActor, ValueCritic
    NETWORK_REGISTRY["gaussian"] = GaussianLearningActor
    NETWORK_REGISTRY["value_critic"] = ValueCritic
    _networks_imported = True

OPTIMIZER_REGISTRY: dict[str, Callable[..., optax.GradientTransformation]] = {}

def register_optimizer(key: str):
    def _decorator(fn):
        OPTIMIZER_REGISTRY[key] = fn
        return fn
    return _decorator

def build_optimizer(
    name: str,
    *,
    lr: float,
    max_grad_norm: float | None = 40.0,
    **kwargs: Any,
) -> optax.GradientTransformation:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {name!r}. Registered: {sorted(OPTIMIZER_REGISTRY)}"
        )
    return OPTIMIZER_REGISTRY[name](lr=lr, max_grad_norm=max_grad_norm, **kwargs)

def _maybe_clip(tx: optax.GradientTransformation, max_grad_norm: float | None) -> optax.GradientTransformation:
    if max_grad_norm is None:
        return tx
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)

@register_optimizer("adam")
def _build_adam(
    lr: float, max_grad_norm: float | None = 40.0, **kwargs: Any,
) -> optax.GradientTransformation:
    return _maybe_clip(optax.adam(lr), max_grad_norm)

TASK_MAP: dict[str, str] = {
    "circle": "envs.tasks.navigation.circle.CircleEnv",
}

OBJECT_MAP: dict[str, str] = {
    "sigwalls": "envs.tasks.navigation.objects.SigwallConfig",
    "circle":   "envs.tasks.navigation.objects.CircleConfig",
}

def _resolve(path: str) -> type:
    module_path, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)

def get_task_env_class(key: str) -> type:
    k = str(key).lower()
    if k not in TASK_MAP:
        raise ValueError(f"Unknown task type: {key!r}. Available: {sorted(TASK_MAP)}")
    return _resolve(TASK_MAP[k])

def get_object_config_class(key: str) -> type:
    k = str(key)
    if k not in OBJECT_MAP:
        raise ValueError(f"Unknown object kind: {key!r}. Available: {sorted(OBJECT_MAP)}")
    return _resolve(OBJECT_MAP[k])

_CONFIG_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _CONFIG_ROOT / "yaml"
_ALGO_CONFIG_DIR = _DATA_ROOT / "algorithms"
_TASKS_ROOT = _DATA_ROOT / "tasks"
_TASK_PRESETS_ROOT = _DATA_ROOT / "task_presets"
_ALGO_BLOCK_KEYS = frozenset({
    "train", "network", "estimator", "algo", "cost",
    "lagrange", "pid", "logging", "meta",
})

def _mapping_to_str_object_dict(data: Mapping[object, object]) -> dict[str, object]:
    converted: dict[str, object] = {}
    for key, value in data.items():
        converted[str(key)] = value
    return converted

def _load_yaml_file(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file_handle:
        raw = cast(object, yaml.safe_load(file_handle))
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return _mapping_to_str_object_dict(cast(Mapping[object, object], raw))

def _find_algo_yaml(algo_name: str) -> Path:
    direct = _ALGO_CONFIG_DIR / f"{algo_name}.yaml"
    if direct.exists():
        return direct

    lower = algo_name.lower()
    for p in _ALGO_CONFIG_DIR.glob("*.yaml"):
        if p.stem.lower() == lower:
            return p

    raise FileNotFoundError(
        f"No algorithm YAML found for '{algo_name}' in {_ALGO_CONFIG_DIR}"
    )

def _load_algo_yaml_raw(algo_name: str) -> dict[str, object]:
    algo_path = _find_algo_yaml(algo_name)
    return _load_yaml_file(algo_path)

def _validate_algo_yaml_layout(
    algo_name: str,
    raw: Mapping[str, object],
) -> None:
    invalid_keys = sorted(key for key in raw if key not in _ALGO_BLOCK_KEYS)
    if invalid_keys:
        raise ValueError(
            f"Algorithm YAML '{algo_name}' uses unsupported top-level keys "
            f"{invalid_keys}. Expected only {sorted(_ALGO_BLOCK_KEYS)}."
        )

    non_mapping_blocks = sorted(
        key for key, value in raw.items() if not isinstance(value, Mapping)
    )
    if non_mapping_blocks:
        raise ValueError(
            f"Algorithm YAML '{algo_name}' must use mapping blocks for "
            f"{non_mapping_blocks}."
        )

def available_training_algorithms() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in _ALGO_CONFIG_DIR.glob("*.yaml")))

def load_algo_config_data(algo_name: str) -> dict[str, object]:
    raw = _load_algo_yaml_raw(algo_name)
    _validate_algo_yaml_layout(algo_name, raw)

    flat: dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(cast(dict[str, object], value))
        else:
            flat[key] = value

    advantage = flat.pop("advantage", None)
    if isinstance(advantage, dict):
        if "type" in advantage:
            flat.setdefault("advantage_estimator", advantage["type"])
        if "reward_lambda" in advantage:
            flat.setdefault("gae_lambda", advantage["reward_lambda"])
        if "cost_lambda" in advantage:
            flat.setdefault("cost_gae_lambda", advantage["cost_lambda"])

    return flat

def _ensure_cost_critic_lr(config_data: dict[str, Any]) -> None:
    if config_data.get("cost_critic_lr") is None and "critic_lr" in config_data:
        config_data["cost_critic_lr"] = config_data["critic_lr"]

def build_algo_config(algo_name: str, **overrides: Any) -> "AlgoTrainConfig":
    from train.setup.algo_train_config import AlgoTrainConfig
    config_data = dict(load_algo_config_data(algo_name))
    config_data.update(overrides)
    config_data.setdefault("algo_name", str(algo_name))
    _ensure_cost_critic_lr(config_data)
    return AlgoTrainConfig.from_dict(config_data)

def find_task_yaml(task_name: str) -> Path | None:
    stem = str(task_name).split("-v")[0].lower()
    matches = [p for p in _TASKS_ROOT.rglob("*.yaml") if p.stem.lower() == stem]
    if not matches:
        return None
    return matches[0]

def load_task_config(task_name: str) -> dict[str, Any]:
    path = find_task_yaml(task_name)
    if path is None:
        return {
            "episode_length": 1000,
            "env_cfgs": {},
            "constraints": [],
            "joystick_config": {},
            "push_config": {},
        }

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    result: dict[str, Any] = {}
    env_cfgs = raw.get("env_cfgs") or {}
    result["episode_length"] = int(env_cfgs.get("episode_length", 1000))
    result["env_cfgs"] = env_cfgs
    result["constraints"] = raw.get("constraints") or []
    result["joystick_config"] = raw.get("joystick_config") or {}
    result["push_config"] = raw.get("push_config") or {}

    for key, value in raw.items():
        if key in {"env_cfgs", "constraints", "joystick_config", "push_config"}:
            continue
        result[key] = value

    return result

COST_FIELD_DEFAULTS: dict[str, Any] = {
    "discounted": False,
}

def available_task_presets() -> list[str]:
    return sorted(p.stem for p in _TASK_PRESETS_ROOT.glob("*.yaml"))

def load_task_preset(name: str) -> dict[str, Any]:
    path = _TASK_PRESETS_ROOT / f"{name}.yaml"
    if not path.exists():
        raise ValueError(
            f"Unknown task preset {name!r}. Available under "
            f"{_TASK_PRESETS_ROOT}: {available_task_presets()}"
        )
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Task preset {name!r} did not parse to a mapping")
    return cast(dict[str, Any], raw)

def apply_task_preset(
    task_config: dict[str, Any],
    preset_name: str | None,
    agent_max_speed: float,
) -> dict[str, Any]:
    if preset_name is None:
        return task_config
    preset = load_task_preset(preset_name)
    add = preset.get("add_constraints") or []
    new_constraints = list(task_config.get("constraints") or [])
    for entry in add:
        spec = dict(entry)
        factor = float(spec.pop("threshold_factor"))
        spec["threshold"] = float(agent_max_speed) * factor
        new_constraints.append(spec)
    out = dict(task_config)
    out["constraints"] = new_constraints
    return out

__all__ = [
    "ConstraintSpec",
    "build_constraint_spec",
    "ALGO_MAP",
    "get_algo_class",
    "NETWORK_REGISTRY",
    "register_network",
    "build_network",
    "OPTIMIZER_REGISTRY",
    "register_optimizer",
    "build_optimizer",
    "TASK_MAP",
    "OBJECT_MAP",
    "get_task_env_class",
    "get_object_config_class",
    "available_training_algorithms",
    "load_algo_config_data",
    "build_algo_config",
    "find_task_yaml",
    "load_task_config",
    "COST_FIELD_DEFAULTS",
    "available_task_presets",
    "load_task_preset",
    "apply_task_preset",
]
