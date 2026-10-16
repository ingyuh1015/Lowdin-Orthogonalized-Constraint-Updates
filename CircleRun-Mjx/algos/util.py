# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingParameterType=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from algos.constants import EPS
from networks import flatten_pytree, gaussian_kl, safe_nan_zero

def compute_kl_fn(actor_apply_fn, params, obs):
    flat_params, unravel_fn = flatten_pytree(params)
    old_mean, old_log_std = actor_apply_fn(params, obs)
    old_mean = jax.lax.stop_gradient(old_mean)
    old_log_std = jax.lax.stop_gradient(old_log_std)

    def kl_fn(flat_actor_params):
        actor_params = unravel_fn(flat_actor_params)
        mean_new, log_std_new = actor_apply_fn(actor_params, obs)
        kl = jnp.mean(gaussian_kl(old_mean, old_log_std, mean_new, log_std_new))
        return safe_nan_zero(kl)

    return kl_fn, flat_params

def fvp(
    kl_fn: Callable[[jax.Array], jax.Array],
    params: jax.Array,
    v: jax.Array,
    damping: float | jax.Array = 0.1,
) -> jax.Array:
    v = safe_nan_zero(v)
    h = safe_nan_zero(jax.jvp(jax.grad(kl_fn), (params,), (v,))[1])
    return h + damping * v

def natural_gradient(
    fvp_fn: Callable[[jax.Array], jax.Array],
    grad_flat: jax.Array,
    cg_max_iters: int = 15,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    x = safe_nan_zero(jax.scipy.sparse.linalg.cg(fvp_fn, grad_flat, maxiter=max(1, cg_max_iters))[0])

    fvp_x = fvp_fn(x)
    xHx = jnp.dot(x, fvp_x)

    metrics: dict[str, jax.Array] = {"xHx": xHx}
    return x, metrics

def trpo_line_search(
    direction: jax.Array,
    objective_fn: Callable[[object], jax.Array] | None,
    kl_fn: Callable[[jax.Array], jax.Array],
    old_loss: jax.Array,
    old_flat: jax.Array,
    unravel_fn: Callable[[jax.Array], object],
    max_iters: int = 10,
    coeff: float = 0.5,
    target_kl: float = 0.01,
    *,
    evaluate_fn: Callable[[jax.Array], Any] | None = None,
    acceptance_fn: Callable[[Any], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array]:
    if evaluate_fn is None:
        if objective_fn is None:
            raise ValueError("objective_fn is required when evaluate_fn is not provided")

        def evaluate_fn(new_flat_params):
            new_loss = objective_fn(unravel_fn(new_flat_params))
            kl = kl_fn(new_flat_params)
            return new_loss, kl

    if acceptance_fn is None:
        def acceptance_fn(eval_state):
            new_loss, kl = eval_state
            return (new_loss <= old_loss) & (kl <= target_kl)

    initial_eval = evaluate_fn(old_flat)

    def cond_fn(val):
        iter_, _params, _eval, accepted = val
        return (~accepted) & (iter_ < max_iters)

    def body_fn(val):
        iter_, _params, _eval, _accepted = val
        step_size = coeff**iter_
        new_params = old_flat + step_size * direction
        eval_state = evaluate_fn(new_params)
        accepted = acceptance_fn(eval_state)
        return (iter_ + 1, new_params, eval_state, accepted)

    init = (
        jnp.int8(0),
        old_flat,
        initial_eval,
        jnp.bool_(False),
    )
    iters, new_params, _eval, accepted = jax.lax.while_loop(cond_fn, body_fn, init)
    new_params = jax.lax.cond(
        accepted,
        lambda: new_params,
        lambda: old_flat,
    )
    acceptance_step = jnp.where(accepted, iters, jnp.int8(0))
    return new_params, acceptance_step

def prepare_trust_region(
    *,
    actor_apply_fn: Callable,
    actor_params: Any,
    batch: Any,
    cfg: Any,
) -> tuple[Callable, Callable, jax.Array, Callable]:
    kl_fn, flat_params = compute_kl_fn(actor_apply_fn, actor_params, batch.obs)
    def fvp_fn(v):
        return fvp(kl_fn, flat_params, v, damping=cfg.cg_damping)
    _, unravel_fn = flatten_pytree(actor_params)
    return kl_fn, fvp_fn, flat_params, unravel_fn

_PREPARE_RETURN_DOC = """Unified prepare-helper return tuple (order fixed):

    reward_grad      : (d,)        flat, reward-ASCENDING direction
    cost_grads       : (d, nc)     flat, cost-ASCENDING direction per constraint
    delta            : (nc,)       batch.cost_limit - batch.cost_signal
                                   (negative components = active violation)
    fvp_fn           : Callable    F·v + damping·v
    kl_fn            : Callable    scalar KL (flat params → scalar)
    flat_params      : (d,)        current actor params, flattened
    unravel_fn       : Callable    flat → pytree inverse of flatten
    old_reward       : scalar      current reward_scalar_fn(actor_params)
    reward_scalar_fn : Callable    (params) -> scalar; minimize = reward ASCEND
    cost_scalars_fn  : Callable    (params) -> (nc,); minimize = cost DESCEND;
                                   values live on the cost (positive = bad) scale
"""

def prepare_dual(
    *,
    actor_loss_fn: Callable,
    actor_apply_fn: Callable,
    actor_params: Any,
    batch: Any,
    cfg: Any,
) -> tuple[Any, ...]:
    f"""Build dual-gradient pieces for any constrained actor update.

    {_PREPARE_RETURN_DOC}"""
    nc = int(cfg.num_constraints)
    kl_fn, fvp_fn, flat_params, unravel_fn = prepare_trust_region(
        actor_apply_fn=actor_apply_fn, actor_params=actor_params,
        batch=batch, cfg=cfg,
    )

    def reward_scalar_fn(params):
        return actor_loss_fn(params, batch, adv=batch.advantages[..., 0])[0]

    def cost_scalars_fn(params):
        return jax.vmap(
            lambda i: -actor_loss_fn(params, batch, adv=batch.advantages[..., 1 + i])[0],
        )(jnp.arange(nc))

    old_reward, reward_grad_tree = jax.value_and_grad(reward_scalar_fn)(actor_params)
    reward_grad = -flatten_pytree(reward_grad_tree)[0]

    cost_grad_cols = jax.vmap(
        lambda i: flatten_pytree(
            jax.grad(lambda p: cost_scalars_fn(p)[i])(actor_params)
        )[0],
    )(jnp.arange(nc))
    cost_grads = cost_grad_cols.T

    delta = batch.cost_limit[:nc] - batch.cost_signal[:nc]

    return (
        reward_grad, cost_grads, delta, fvp_fn, kl_fn,
        flat_params, unravel_fn, old_reward, reward_scalar_fn, cost_scalars_fn,
    )

def ng_linesearch_update(
    *,
    actor_loss_fn: Callable,
    actor_apply_fn: Callable,
    actor_params: Any,
    batch: Any,
    cfg: Any,
) -> tuple[Any, dict[str, Any]]:
    kl_fn, fvp_fn, flat_params, unravel_fn = prepare_trust_region(
        actor_apply_fn=actor_apply_fn, actor_params=actor_params, batch=batch, cfg=cfg,
    )

    def scalar_loss(params):
        return actor_loss_fn(params, batch)[0]

    tkl = jnp.asarray(cfg.target_kl, dtype=jnp.float32)
    old_loss, grad = jax.value_and_grad(scalar_loss)(actor_params)
    flat_grad, _ = flatten_pytree(grad)
    x, ng_metrics = natural_gradient(fvp_fn, -flat_grad, cg_max_iters=cfg.cg_max_iters)

    alpha = jnp.sqrt(2.0 * tkl / (jnp.maximum(ng_metrics["xHx"], 0.0) + EPS))
    full_step = alpha * x

    new_flat, acceptance_step = trpo_line_search(
        direction=full_step,
        objective_fn=scalar_loss,
        kl_fn=kl_fn,
        old_loss=old_loss,
        old_flat=flat_params,
        unravel_fn=unravel_fn,
        max_iters=cfg.line_search_max_iter,
        coeff=cfg.line_search_coeff,
        target_kl=tkl,
    )
    new_params = unravel_fn(new_flat)
    return new_params, {
        "actor_loss": scalar_loss(new_params),
        "alpha": alpha,
        "acceptance_step": acceptance_step,
    }

def _shuffle_batch(rng: jax.Array, batch: Any) -> Any:
    batch_size = batch.obs.shape[0]
    perm = jax.random.permutation(rng, batch_size)
    return jax.tree_util.tree_map(
        lambda x: x[perm] if x.ndim > 0 and x.shape[0] == batch_size else x,
        batch,
    )

def _get_minibatch(batch: Any, start: int, size: int) -> Any:
    batch_size = batch.obs.shape[0]
    return jax.tree_util.tree_map(
        lambda x: jax.lax.dynamic_slice_in_dim(x, start, size) if x.ndim > 0 and x.shape[0] == batch_size else x,
        batch,
    )

__all__ = [
    "compute_kl_fn", "fvp", "natural_gradient", "trpo_line_search",
    "prepare_trust_region", "prepare_dual",
    "ng_linesearch_update",
    "_shuffle_batch", "_get_minibatch",
]
