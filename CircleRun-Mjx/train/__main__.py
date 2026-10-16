#!/usr/bin/env python
from __future__ import annotations

import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

_logger = logging.getLogger(__name__)

def main() -> None:
    from train.train import mark_start, elapsed, bootstrap_backend
    mark_start()

    from train.train import load_train_args, get_backend
    args = load_train_args()
    bootstrap_backend_name = bootstrap_backend(args.backend)

    from train.train import (
        build_run_context,
        save_run_metadata,
        build_algo_train_config,
        build_task_env_config,
        build_env,
        OnPolicyPipeline,
        build_compiled_plan,
        execute_training_run,
        get_num_devices,
    )
    from config.config import get_algo_class
    seed = int(args.seed)
    algo_train_config = build_algo_train_config(args, seed=seed)
    task_env_config = build_task_env_config(
        args=args,
        algo_train_config=algo_train_config,
    )
    available_devices = get_num_devices()
    env = build_env(
        task_env_config=task_env_config,
        episode_length=task_env_config.episode_length,
        batch_size=task_env_config.num_envs,
        gamma=task_env_config.gamma,
    )
    algo_train_config = algo_train_config.with_updates({
        "obs_size": int(env.observation_size),
        "action_size": int(env.action_size),
    })
    policy = get_algo_class(str(algo_train_config.algo_name))(
        config=algo_train_config, compute_dtype=jnp.float32,
    )
    pipeline = OnPolicyPipeline(policy=policy, env=env)

    backend = getattr(algo_train_config, "backend", None) or get_backend()
    _logger.info(
        "[platform] requested_backend=%s bootstrap_backend=%s "
        "active_backend=%s available_devices=%s",
        args.backend, bootstrap_backend_name, backend, available_devices,
    )

    _preset = getattr(args, "preset", None)

    last_result = None
    run_context = build_run_context(algo_train_config, seed=seed, preset=_preset)
    compiled_plan = build_compiled_plan(
        policy=policy, pipeline=pipeline,
        progress_path=str(run_context.progress_csv), use_logging=algo_train_config.use_logging,
    )
    try:
        result = execute_training_run(
            compiled_plan, seed=seed, out_dir=run_context.out_dir,
            compile_timing=algo_train_config.use_logging,
        )
        save_run_metadata(run_context, result, algo_train_config)
        last_result = result
    except Exception:
        save_run_metadata(run_context, None, algo_train_config)
        raise

    total = elapsed()
    if last_result is not None:
        finalize = max(0.0, total - last_result.total_time_seconds)
        print(
            f"[timing] total: {total:.1f}s "
            f"(prepare: {last_result.prepare_time_seconds:.1f}s, "
            f"compile: {last_result.compile_time_seconds:.1f}s, "
            f"train: {last_result.training_time_seconds:.1f}s, "
            f"finalize: {finalize:.1f}s)",
            flush=True,
        )
    else:
        print(f"[timing] total: {total:.1f}s", flush=True)

__all__ = ["main"]

if __name__ == "__main__":
    main()
