
from __future__ import annotations

from typing import Any, Callable

import jax
from jaxopt import OSQP
import jax.numpy as jnp
from jax.scipy import sparse

from algos.base import EPS, OnPolicyBase
from algos.util import prepare_dual, natural_gradient
from networks import safe_nan_zero

_REGULARIZATION_SCALES = (1.0, 1e2, 1e4)

def _safe_norm(x, eps: float = EPS):
    return jnp.sqrt(jnp.maximum(jnp.dot(x, x), eps))

def _safe_solve(system, rhs, eps: float):
    n = system.shape[0]
    eye = jnp.eye(n, dtype=system.dtype)
    scales = jnp.asarray(_REGULARIZATION_SCALES, dtype=system.dtype)

    def body_fn(carry, scale):
        solved, solution = carry
        candidate = jnp.linalg.solve(system + (eps * scale) * eye, rhs)
        candidate = safe_nan_zero(candidate)
        finite = jnp.all(jnp.isfinite(candidate))
        use = (~solved) & finite
        solution = jnp.where(use, candidate, solution)
        solved = solved | finite
        return (solved, solution), None

    init_solution = jnp.zeros_like(rhs)
    (solved, solution), _ = jax.lax.scan(
        body_fn, (jnp.bool_(False), init_solution), scales,
    )
    return solution, solved

def compute_gram(reward_nat_grad, cost_nat_grads, fvp_fn):
    nat_grads = jnp.concatenate([reward_nat_grad[:, None], cost_nat_grads], axis=1)
    fisher_nat_grads = jax.vmap(fvp_fn, in_axes=1, out_axes=1)(nat_grads)
    gram = nat_grads.T @ fisher_nat_grads
    gram = 0.5 * (gram + gram.T)
    return gram

def _component_representatives(adjacency, candidates, score):
    n = candidates.shape[0]
    reach = adjacency & candidates[:, None] & candidates[None, :]
    reach = reach | jnp.diag(candidates)
    reach = jax.lax.fori_loop(
        0, n, lambda k, r: r | (r[:, k, None] & r[k, None, :]), reach,
    )
    scores = jnp.where(reach, score[None, :], jnp.inf)
    reps = jnp.argmin(scores, axis=1)
    selected = jnp.zeros((n,), dtype=jnp.int32).at[reps].add(candidates.astype(jnp.int32))
    return selected > 0


def _joint_lowdin_basis(reward_nat_grad, cost_nat_grads, gram, delta,
                        trust_radius, similarity_threshold, eps=EPS):
    norms = jnp.sqrt(jnp.maximum(jnp.diag(gram), 0.0))
    cost_norms = norms[1:]
    violated = delta < 0.0
    eps_nz = jnp.where(jnp.any(violated), 1e-12, 1e-6)
    retained = ((cost_norms ** 2 > eps_nz) & (delta < cost_norms * trust_radius)) | violated
    correlation = gram / (norms[:, None] * norms[None, :] + eps)
    candidates = retained & (jnp.abs(correlation[0, 1:]) < 0.99)
    margin = delta / (cost_norms + eps)
    positive = correlation[1:, 1:] >= similarity_threshold
    hopeless = delta < -trust_radius * cost_norms
    score = jnp.where(hopeless, -1e6 - cost_norms, margin)
    representatives = _component_representatives(positive, candidates, score)
    negative = correlation[1:, 1:] <= -0.99
    reach = negative & representatives[:, None] & representatives[None, :]
    reach = reach | jnp.diag(representatives)
    n = delta.shape[0]
    reach = jax.lax.fori_loop(0, n, lambda k, r: r | (r[:, k, None] & r[k, None, :]), reach)
    group_violated = jnp.any(reach & violated[None, :], axis=1)
    group_hopeless = jnp.any(reach & hopeless[None, :], axis=1)
    scores = jnp.where(group_violated[:, None], margin[None, :], -correlation[0, 1:][None, :])
    scores = jnp.where(group_hopeless[:, None], -cost_norms[None, :], scores)
    chosen = jnp.argmin(jnp.where(reach, scores, jnp.inf), axis=1)
    basis_costs = jnp.zeros((n,), dtype=jnp.int32).at[chosen].add(representatives.astype(jnp.int32)) > 0
    basis_mask = jnp.concatenate([jnp.ones((1,), dtype=jnp.bool_), basis_costs])
    mask = basis_mask.astype(gram.dtype)
    masked_gram = gram * mask[:, None] * mask[None, :] + jnp.diag(1.0 - mask)
    vals, vecs = jnp.linalg.eigh(0.5 * (masked_gram + masked_gram.T))
    inv_sqrt = vecs @ jnp.diag(jnp.maximum(vals, eps) ** -0.5) @ vecs.T
    natural_grads = jnp.concatenate([reward_nat_grad[:, None], cost_nat_grads], axis=1)
    E = (natural_grads * mask[None, :]) @ inv_sqrt
    E = E * mask[None, :]
    violated_any = jnp.any(retained & violated)
    remove_reward = violated_any & jnp.any(basis_costs)
    E = E.at[:, 0].set(jnp.where(remove_reward, 0.0, E[:, 0]))
    return E, retained, basis_mask, remove_reward


def _compute_pareto_objective(C_basis, delta, violated_mask, eps: float = EPS):
    work_dtype = jnp.float64 if jax.config.x64_enabled else C_basis.dtype
    C = C_basis.astype(work_dtype) * violated_mask[None, :]
    delta = delta.astype(work_dtype)
    norms = jnp.linalg.norm(C, axis=0) + eps
    target = jnp.where(violated_mask, -delta / norms, 0.0)
    n = delta.shape[0]
    eye = jnp.eye(n, dtype=C.dtype)
    G = jnp.concatenate([-(C.T @ C), -eye, eye], axis=0)
    h = jnp.concatenate([
        jnp.where(violated_mask, -1e-9, 0.0),
        jnp.zeros(n, dtype=C.dtype),
        jnp.where(violated_mask, jnp.inf, 0.0),
    ])
    result = OSQP(tol=1e-8, maxiter=4000, rho_start=jnp.asarray(0.1, dtype=C.dtype)).run(
        params_obj=(eye, -target), params_ineq=(G, h),
    )
    weights = result.params.primal
    recovery = C @ weights
    norm = jnp.linalg.norm(recovery)
    idx = jnp.argmin(jnp.where(violated_mask, delta / norms, jnp.inf))
    fallback = -C[:, idx] / (norms[idx] + eps)
    return jnp.where((norm > 1e-12) & jnp.all(jnp.isfinite(recovery)),
                     -recovery / (norm + eps), fallback).astype(C_basis.dtype)

def _is_feasible(w, C_basis, delta, trust_radius, tol=1e-6):
    vals = C_basis.T @ w
    active = jnp.isfinite(delta)
    linear_ok = jnp.all(jnp.where(active, vals <= delta + tol, True))
    tr_ok = jnp.dot(w, w) <= (trust_radius * trust_radius + tol)
    return linear_ok & tr_ok

def _solve_trust_kkt(active_f, C_T, constraint_gram, objective, delta, trust_radius, eps):
    system = (active_f[:, None] * active_f[None, :]) * constraint_gram
    system = system + jnp.diag((1.0 - active_f) + eps * active_f)

    obj_proj = C_T @ objective
    rhs = jnp.stack([active_f * delta, active_f * obj_proj], axis=1)

    sol, solved = _safe_solve(system, rhs, eps=eps)
    w_base = C_T.T @ (active_f * sol[:, 0])
    obj_perp = objective - C_T.T @ (active_f * sol[:, 1])

    qa = jnp.dot(w_base, w_base) - trust_radius * trust_radius
    qb = 2.0 * jnp.dot(obj_perp, w_base)
    qc = jnp.dot(obj_perp, obj_perp)
    disc = qb * qb - 4.0 * qa * qc
    disc = jnp.where((disc < 0.0) & (disc > -1e-10), 0.0, disc)

    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0) + eps)
    beta = jnp.where(
        jnp.abs(qa) < eps,
        -qc / (qb + eps),
        jnp.where(qa > 0.0, (-qb + sqrt_disc) / (2.0 * qa + eps), (-qb - sqrt_disc) / (2.0 * qa + eps)),
    )

    dual_rhs = active_f * (obj_proj - beta * delta)
    dual_vars, solved_dual = _safe_solve(system, dual_rhs, eps=eps)
    dual_vars = active_f * dual_vars

    candidate = safe_nan_zero((objective - C_T.T @ dual_vars) / (beta + eps))
    ok = solved & solved_dual & (disc >= 0.0) & (beta > 0.0) & jnp.all(jnp.isfinite(candidate))

    return candidate, dual_vars, ok

def _solve_active_set(
    start_mask, objective, C_basis, delta,
    trust_radius, max_iters, eps,
):
    C_T = C_basis.T
    constraint_gram = C_T @ C_T.T
    drop_tol = 100.0 * eps

    init = (
        jnp.int32(0), start_mask, jnp.zeros_like(objective),
        jnp.bool_(False), jnp.bool_(False),
    )

    def cond_fn(carry):
        it, _mask, _w, success, failed = carry
        return (it < max_iters) & (~success) & (~failed)

    def body_fn(carry):
        it, mask, w_prev, _success, _failed = carry
        active_f = mask.astype(objective.dtype)

        def no_active():
            return it + 1, mask, w_prev, jnp.bool_(False), jnp.bool_(True)

        def do_solve():
            candidate, dual_vars, kkt_ok = _solve_trust_kkt(
                active_f, C_T, constraint_gram, objective, delta, trust_radius, eps,
            )

            dual_masked = jnp.where(mask, dual_vars, jnp.inf)
            drop_idx = jnp.argmin(dual_masked)
            should_drop = jnp.min(dual_masked) < -drop_tol

            violation = C_basis.T @ candidate - delta
            add_idx = jnp.argmax(violation)
            should_add = (jnp.max(violation) > 1e-6) & (~mask[add_idx])

            do_drop = kkt_ok & should_drop
            do_add = kkt_ok & (~should_drop) & should_add
            success = kkt_ok & (~should_drop) & (~should_add)

            new_mask = jax.lax.cond(
                do_drop,
                lambda: mask.at[drop_idx].set(False),
                lambda: jax.lax.cond(
                    do_add,
                    lambda: mask.at[add_idx].set(True),
                    lambda: mask,
                ),
            )
            new_w = jnp.where(success, candidate, w_prev)
            return it + 1, new_mask, new_w, success, ~kkt_ok

        return jax.lax.cond(jnp.sum(active_f) <= 0.5, no_active, do_solve)

    _, _mask, w, success, _failed = jax.lax.while_loop(cond_fn, body_fn, init)
    return w, success

def _solve_qcqp(
    objective, C_basis, delta, trust_radius,
    active_tolerance=1e-6, max_active_steps=8, max_drop_steps=10, eps=EPS,
):
    num_constraints = delta.shape[0]
    zero = jnp.zeros_like(objective)
    obj_norm = _safe_norm(objective, eps)

    def zero_solution():
        return zero

    def nonzero_solution():
        w_unconstrained = trust_radius * objective / (obj_norm + eps)

        constraint_vals = C_basis.T @ w_unconstrained
        is_unconstrained_feasible = jnp.all(constraint_vals <= delta - EPS)

        def unconstrained_ok():
            return w_unconstrained

        def constrained_branch():
            is_violated = constraint_vals > delta + EPS
            constraint_norms = jnp.sqrt(jnp.sum(C_basis ** 2, axis=0)) + eps
            severity = jnp.where(is_violated, delta / constraint_norms, jnp.inf)
            order = jnp.argsort(severity)

            init = (jnp.int32(0), zero, jnp.bool_(False))

            def cond_fn(carry):
                it, _w, found = carry
                return (it < num_constraints) & (~found)

            def body_fn(carry):
                it, current_w, found = carry
                idx = order[it]
                valid_start = jnp.isfinite(severity[idx])

                def try_candidate():
                    start_mask = jnp.zeros((num_constraints,), dtype=jnp.bool_).at[idx].set(True)
                    candidate_w, solved = _solve_active_set(
                        start_mask=start_mask, objective=objective,
                        C_basis=C_basis, delta=delta,
                        trust_radius=trust_radius, max_iters=max_active_steps + max_drop_steps,
                        eps=eps,
                    )
                    feasible = solved & _is_feasible(
                        candidate_w, C_basis, delta, trust_radius,
                    )
                    next_w = jnp.where(feasible, candidate_w, current_w)
                    return next_w, feasible

                next_w, next_found = jax.lax.cond(
                    valid_start, try_candidate,
                    lambda: (current_w, jnp.bool_(False)),
                )
                return it + 1, next_w, found | next_found

            _, best_w, found = jax.lax.while_loop(cond_fn, body_fn, init)

            def fallback():
                dots = C_basis.T @ w_unconstrained
                pos = dots > eps
                alphas = jnp.where(pos, delta / (dots + eps), jnp.inf)
                alpha = jnp.minimum(1.0, jnp.min(alphas))
                return jax.lax.cond(
                    (alpha > 0.0) & jnp.isfinite(alpha),
                    lambda: alpha * w_unconstrained,
                    lambda: zero,
                )

            return jax.lax.cond(found, lambda: best_w, fallback)

        return jax.lax.cond(is_unconstrained_feasible, unconstrained_ok, constrained_branch)

    return jax.lax.cond(obj_norm <= eps, zero_solution, nonzero_solution)

def solve_locu_step(
    reward_nat_grad: jnp.ndarray,
    cost_nat_grads: jnp.ndarray,
    reward_grad: jnp.ndarray,
    cost_grads: jnp.ndarray,
    gram: jnp.ndarray,
    delta: jnp.ndarray,
    trust_radius: float | jnp.ndarray,
    screening_threshold: float,
    active_tolerance: float,
    max_active_steps: int,
    max_drop_steps: int,
    eps: float = EPS,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    num_constraints = delta.shape[0]
    if num_constraints == 0:
        fisher_norm = jnp.sqrt(jnp.maximum(gram[0, 0], eps))
        direction = trust_radius * reward_nat_grad / (fisher_norm + eps)
        return direction, {
            "retained_count": jnp.int32(0), "basis_count": jnp.int32(1),
            "violated_any": jnp.bool_(False),
            "direction_norm": jnp.linalg.norm(direction), "w_norm": jnp.asarray(trust_radius),
        }

    E, retained, basis_mask, removed_reward = _joint_lowdin_basis(
        reward_nat_grad, cost_nat_grads, gram, delta, trust_radius,
        screening_threshold, eps,
    )
    C_basis = E.T @ cost_grads
    violated_mask = retained & (delta < 0.0)
    violated_any = jnp.any(violated_mask)
    objective = jax.lax.cond(
        violated_any,
        lambda: _compute_pareto_objective(C_basis, delta, violated_mask, eps),
        lambda: E.T @ reward_grad,
    )
    enforced = retained & (~violated_mask)
    C_enforced = C_basis * enforced[None, :]
    effective_delta = jnp.where(enforced, delta, jnp.inf)
    w = _solve_qcqp(
        objective, C_enforced, effective_delta, trust_radius,
        active_tolerance, max_active_steps, max_drop_steps, eps,
    )
    direction = E @ w
    return direction, {
        "retained_count": jnp.sum(retained.astype(jnp.int32)),
        "basis_count": jnp.sum(basis_mask.astype(jnp.int32)) - removed_reward.astype(jnp.int32),
        "violated_any": violated_any,
        "direction_norm": jnp.linalg.norm(direction), "w_norm": jnp.linalg.norm(w),
    }


def locu_update(
    *,
    actor_loss_fn: Callable,
    actor_apply_fn: Callable,
    actor_params: Any,
    batch: Any,
    cfg: Any,
) -> tuple[Any, dict[str, Any]]:
    (reward_grad, cost_grads, delta, fvp_fn, _kl_fn,
     flat_params, unravel_fn, _, _reward_scalar_fn, _cost_scalars_fn) = prepare_dual(
        actor_loss_fn=actor_loss_fn, actor_apply_fn=actor_apply_fn,
        actor_params=actor_params, batch=batch, cfg=cfg,
    )
    cg_iters = cfg.cg_max_iters

    reward_nat_grad, _ = natural_gradient(fvp_fn, reward_grad, cg_max_iters=cg_iters)
    cost_nat_grads = jax.vmap(
        lambda vec: natural_gradient(fvp_fn, vec, cg_max_iters=cg_iters)[0],
        in_axes=1,
        out_axes=1,
    )(cost_grads)
    gram = compute_gram(reward_nat_grad, cost_nat_grads, fvp_fn)

    trust_radius = jnp.sqrt(jnp.maximum(2.0 * cfg.trust_region_coef, 1e-12))

    direction, solver_info = solve_locu_step(
        reward_nat_grad=reward_nat_grad,
        cost_nat_grads=cost_nat_grads,
        reward_grad=reward_grad,
        cost_grads=cost_grads,
        gram=gram,
        delta=delta,
        trust_radius=trust_radius,
        screening_threshold=cfg.constraint_screening_threshold,
        active_tolerance=cfg.active_set_tolerance,
        max_active_steps=cfg.locu_max_active_steps,
        max_drop_steps=cfg.locu_max_drop_steps,
    )

    new_params = unravel_fn(flat_params + direction)

    return new_params, {
        "locu_retained": solver_info["retained_count"],
        "locu_basis": solver_info["basis_count"],
        "locu_w_norm": solver_info["w_norm"],
        "locu_direction_norm": solver_info["direction_norm"],
        "reward_grad_norm": jnp.linalg.norm(reward_grad).astype(jnp.float32),
        "cost_grad_norm": jnp.linalg.norm(cost_grads, axis=0).astype(jnp.float32),
    }

class LOCU(OnPolicyBase):

    def update_actor(self, state, batch):
        new_params, actor_info = locu_update(
            actor_loss_fn=self.actor_loss, actor_apply_fn=self.actor.apply,
            actor_params=state.actor_params, batch=batch, cfg=self.cfg,
        )
        return state.replace(actor_params=new_params), actor_info

__all__ = ["LOCU"]
