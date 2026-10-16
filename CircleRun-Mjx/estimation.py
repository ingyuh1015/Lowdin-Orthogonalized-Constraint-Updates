from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_mask: jnp.ndarray,
    bootstrap: jnp.ndarray,
    gamma: float,
    lam: float,
    *,
    next_values: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    deltas = rewards + gamma * bootstrap_mask * next_values - values
    discounts = gamma * lam * bootstrap_mask

    def scan_fn(gae, step):
        delta, discount = step
        gae = delta + discount * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(bootstrap),
        (deltas, discounts),
        reverse=True,
        unroll=16,
    )
    return advantages + values, advantages

def compute_targets(
    batch, num_constraints, *,
    reward_gamma, reward_lambda, cost_lambda,
    finite=False,
):
    if finite:
        bootstrap_mask = 1.0 - jnp.asarray(batch.dones)
    else:
        bootstrap_mask = 1.0 - batch.terminated

    signals = jnp.concatenate([batch.rewards[..., None], batch.costs], axis=-1)
    signals_t = jnp.moveaxis(signals, -1, 0)
    values_t = jnp.moveaxis(batch.values, -1, 0)
    next_values_t = jnp.moveaxis(batch.next_values, -1, 0)
    bootstrap_t = jnp.moveaxis(batch.bootstrap, -1, 0)
    lambdas = jnp.concatenate([
        jnp.asarray([reward_lambda], dtype=signals.dtype),
        jnp.full((num_constraints,), cost_lambda, dtype=signals.dtype),
    ])

    def _gae_slot(r, v, nv, b, lam):
        return compute_gae(r, v, bootstrap_mask, b, reward_gamma, lam, next_values=nv)

    returns_t, advs_t = jax.vmap(_gae_slot)(
        signals_t, values_t, next_values_t, bootstrap_t, lambdas,
    )
    returns = jnp.moveaxis(returns_t, 0, -1)
    advs = jnp.moveaxis(advs_t, 0, -1)
    return returns, advs

def batch_critic_values(
    obs: jnp.ndarray,
    next_obs: jnp.ndarray,
    critic_apply: Callable,
    critic_params: Any,
    num_constraints: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    T, B = obs.shape[:2]
    stacked_obs = jnp.concatenate([obs, next_obs], axis=0)
    flat = stacked_obs.reshape(-1, stacked_obs.shape[-1])

    all_values = jax.vmap(lambda p: critic_apply(p, flat))(critic_params)
    all_2d = jnp.moveaxis(all_values, 0, -1).reshape(2 * T, B, 1 + num_constraints)
    return all_2d[:T], all_2d[T:]

__all__ = [
    "compute_gae",
    "compute_targets",
    "batch_critic_values",
]
