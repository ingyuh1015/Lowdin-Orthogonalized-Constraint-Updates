# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

from algos.base import EPS, OnPolicyBase
from algos.util import prepare_dual, natural_gradient


def _solve_nonneg_qp_osqp(S, c, iters: int = 200):
    n = c.shape[0]
    A = jnp.eye(n, dtype=S.dtype)
    l = jnp.zeros(n, dtype=S.dtype)
    u = jnp.full(n, jnp.inf, dtype=S.dtype)
    solver = BoxOSQP(maxiter=iters, tol=1e-6, jit=True)
    sol = solver.run(params_obj=(S, -c), params_eq=A, params_ineq=(l, u))
    return jnp.maximum(sol.params.primal[0], 0.0)


def _solve_full_dual(S, r, c, g_H_inv_g, max_kl, iters: int = 200):
    nu = jnp.sqrt(jnp.maximum(g_H_inv_g, EPS) / (2.0 * max_kl))
    lam = jnp.zeros_like(c)

    def outer(carry, _):
        lam, nu = carry
        Sn = S / nu
        qn = -((r / nu) + c)
        n = c.shape[0]
        solver = BoxOSQP(maxiter=iters, tol=1e-6, jit=True)
        sol = solver.run(
            params_obj=(Sn, qn),
            params_eq=jnp.eye(n, dtype=S.dtype),
            params_ineq=(jnp.zeros(n, dtype=S.dtype), jnp.full(n, jnp.inf, dtype=S.dtype)),
        )
        lam = jnp.maximum(sol.params.primal[0], 0.0)
        num = g_H_inv_g - 2.0 * jnp.dot(r, lam) + jnp.dot(lam, S @ lam)
        nu = jnp.sqrt(jnp.maximum(num, EPS) / (2.0 * max_kl))
        return (lam, nu), None

    (lam, nu), _ = jax.lax.scan(outer, (lam, nu), None, length=8)
    return lam, nu


def gradient_integration_update(
    *,
    actor_loss_fn: Callable,
    actor_apply_fn: Callable,
    actor_params: Any,
    batch: Any,
    cfg: Any,
) -> tuple[Any, dict[str, Any]]:
    (reward_grad, cost_grads, delta, fvp_fn, _kl_fn,
     flat_params, unravel_fn, _old_reward, _reward_scalar_fn, _cost_scalars_fn) = prepare_dual(
        actor_loss_fn=actor_loss_fn, actor_apply_fn=actor_apply_fn,
        actor_params=actor_params, batch=batch, cfg=cfg,
    )

    max_kl = jnp.asarray(cfg.target_kl, dtype=reward_grad.dtype)
    zeta = jnp.asarray(getattr(cfg, "con_zeta", 0.0) or 0.0, dtype=reward_grad.dtype)
    cg_iters = cfg.cg_max_iters

    H_inv_g, reward_info = natural_gradient(fvp_fn, reward_grad, cg_max_iters=cg_iters)
    g_H_inv_g = jnp.maximum(reward_info["xHx"], EPS)

    H_inv_B = jax.vmap(
        lambda v: natural_gradient(fvp_fn, v, cg_max_iters=cg_iters)[0],
        in_axes=1, out_axes=1,
    )(cost_grads)
    approx_B = jax.vmap(fvp_fn, in_axes=1, out_axes=1)(H_inv_B)

    S = approx_B.T @ H_inv_B
    S = 0.5 * (S + S.T)
    eigvals = jnp.linalg.eigvalsh(S)
    min_eig = jnp.minimum(jnp.min(eigvals), 0.0)
    nc = S.shape[0]
    S = S + jnp.eye(nc, dtype=S.dtype) * (-min_eig + EPS)

    approx_g = fvp_fn(H_inv_g)
    r = approx_g @ H_inv_B

    b_H_inv_b = jnp.einsum("ij,ij->j", approx_B, H_inv_B)
    max_c = jnp.sqrt(jnp.maximum(2.0 * max_kl * b_H_inv_b, 0.0))
    raw_c = -delta
    c_base = jnp.minimum(max_c, raw_c)

    qp_iters = int(getattr(cfg, "sdac_qp_iters", 200))
    lam_scaling = _solve_nonneg_qp_osqp(S, c_base, iters=qp_iters)
    approx_kl_scaling = 0.5 * jnp.dot(lam_scaling, S @ lam_scaling)
    cost_only = (approx_kl_scaling / max_kl - 1.0) > -1e-3

    def cost_only_branch(_):
        c_inflated = jnp.minimum(max_c, raw_c + zeta)
        lam = _solve_nonneg_qp_osqp(S, c_inflated, iters=qp_iters)
        approx_kl = 0.5 * jnp.dot(lam, S @ lam)
        scaling = jnp.sqrt(max_kl / jnp.maximum(approx_kl, EPS))
        delta_theta = -scaling * (H_inv_B @ lam)
        return delta_theta, lam

    def feasible_branch(_):
        lam, nu = _solve_full_dual(
            S, r, c_base, g_H_inv_g, max_kl, iters=qp_iters,
        )
        delta_theta = (H_inv_g - H_inv_B @ lam) / (nu + EPS)
        return delta_theta, lam

    delta_theta, lambdas = jax.lax.cond(
        cost_only, cost_only_branch, feasible_branch, operand=None,
    )

    new_params = unravel_fn(flat_params + delta_theta)
    safety_mode = jnp.any(raw_c > 0.0)
    return new_params, {
        "Misc/Violation": jnp.max(raw_c).astype(jnp.float32),
        "Misc/LambdaMax": jnp.max(lambdas).astype(jnp.float32),
        "Misc/ActiveConstraints": jnp.sum((raw_c > 0.0).astype(jnp.float32)),
        "Misc/CostOnlyStep": cost_only.astype(jnp.float32),
        "Misc/SafetyMode": safety_mode.astype(jnp.float32),
        "Misc/ApproxKLScaling": approx_kl_scaling.astype(jnp.float32),
        "reward_grad_norm": jnp.linalg.norm(reward_grad).astype(jnp.float32),
        "cost_grad_norm": jnp.linalg.norm(cost_grads, axis=0).astype(jnp.float32),
    }


class onSDAC(OnPolicyBase):

    def update_actor(self, state, batch):
        new_params, actor_info = gradient_integration_update(
            actor_loss_fn=self.actor_loss, actor_apply_fn=self.actor.apply,
            actor_params=state.actor_params, batch=batch, cfg=self.cfg,
        )
        return state.replace(actor_params=new_params), actor_info


__all__ = ["onSDAC"]
