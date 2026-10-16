# pyright: reportMissingImports=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false
from __future__ import annotations

from algos.base import OnPolicyBase, OnPolicyLagrangeMixin
from algos.util import ng_linesearch_update

class TRPOLagrange(OnPolicyLagrangeMixin, OnPolicyBase):

    def update_actor(self, state, batch):
        new_params, metrics = ng_linesearch_update(
            actor_loss_fn=self.reward_loss_fn,
            actor_apply_fn=self.actor.apply,
            actor_params=state.actor_params,
            batch=batch, cfg=self.cfg,
        )
        return state.replace(actor_params=new_params), metrics

__all__ = ["TRPOLagrange"]
