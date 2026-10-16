# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportIncompatibleMethodOverride=false
from __future__ import annotations

import jax
import jax.numpy as jnp

from algos.base import OnPolicyBase
from algos.util import ng_linesearch_update
class CRPO(OnPolicyBase):

    def update_actor(self, state, batch):
        actor_loss = self.actor_loss

        if self.cfg.num_constraints == 0:
            actor_loss_fn = self.reward_loss_fn
            constraint_metrics = {}
        else:
            violations = batch.cost_signal - batch.cost_limit
            active_idx = jnp.argmax(violations)
            safety_mode = jnp.any(violations > 0.0)

            def actor_loss_fn(params, mb):
                def cost_branch():
                    loss, aux = actor_loss(params, mb, adv=mb.advantages[..., 1 + active_idx])
                    return -loss, aux
                def reward_branch():
                    return actor_loss(params, mb, adv=mb.advantages[..., 0])
                return jax.lax.cond(safety_mode, cost_branch, reward_branch)

            constraint_metrics = {
                "crpo_safety_mode": safety_mode.astype(jnp.float32),
                "crpo_active_constraint": active_idx.astype(jnp.float32),
                "crpo_max_violation": jax.lax.cond(
                    safety_mode, lambda: violations[active_idx], lambda: jnp.float32(0.0),
                ),
            }

        new_params, metrics = ng_linesearch_update(
            actor_loss_fn=actor_loss_fn,
            actor_apply_fn=self.actor.apply,
            actor_params=state.actor_params,
            batch=batch, cfg=self.cfg,
        )
        return state.replace(actor_params=new_params), {**metrics, **constraint_metrics}

__all__ = ["CRPO"]
