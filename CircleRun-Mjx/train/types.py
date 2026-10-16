from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
from flax import struct
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as RolloutBufferState

from algos.base import BasePolicyState
from envs.env import EnvState

Metrics = dict[str, jax.Array]

@struct.dataclass
class RuntimeState:
    policy_state: BasePolicyState
    env_state: EnvState
    rollout_buf_state: RolloutBufferState | None

@struct.dataclass
class FullRunResult:
    last_metrics: Metrics
    all_metrics: Metrics

@dataclass(frozen=True)
class CompiledTrainingPlan:
    init_runtime_state: Callable[[jax.Array], RuntimeState] | None = None
    prepare_runtime_state: Callable[[RuntimeState], RuntimeState] | None = None
    one_update: Callable[[RuntimeState, jax.Array], tuple[RuntimeState, Metrics]] | None = None
    run_full_jit: Callable[[RuntimeState], tuple[RuntimeState, FullRunResult]] | None = None
    compile_fingerprint: str = ""

@dataclass(frozen=True)
class ExecutionResult:
    compile_fingerprint: str
    prepare_time_seconds: float
    compile_time_seconds: float
    training_time_seconds: float
    total_time_seconds: float

@dataclass(frozen=True)
class RunContext:
    experiment_name: str
    out_dir: Path
    progress_csv: Path
    meta_path: Path

__all__ = [
    "Metrics",
    "RuntimeState",
    "FullRunResult",
    "CompiledTrainingPlan",
    "ExecutionResult",
    "RunContext",
]
