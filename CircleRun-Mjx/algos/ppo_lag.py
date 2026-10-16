# pyright: reportMissingImports=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportIncompatibleMethodOverride=false
from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

from algos.base import OnPolicyBase, OnPolicyLagrangeMixin
from algos.util import _shuffle_batch, _get_minibatch
from networks import gaussian_kl

def gradient_update(
    *,
    actor_params: Any,
    actor_opt_state: Any,
    batch: Any,
    loss_fn: Callable[[Any, Any], tuple[Any, dict[str, Any]]],
    opt_update: Callable[..., tuple[Any, Any]],
) -> tuple[Any, Any, dict[str, Any]]:
    (loss_val, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True,
    )(actor_params, batch)

    updates, new_opt_state = opt_update(grads, actor_opt_state, actor_params)
    new_params = optax.apply_updates(actor_params, updates)

    metrics = dict(metrics)
    metrics["actor_loss"] = loss_val
    metrics["actor_grad_norm"] = optax.global_norm(grads)
    return new_params, new_opt_state, metrics

def train_minibatch_actor_early_stop(
    *,
    state: Any,
    batch: Any,
    loss_fn: Callable[[Any, Any], tuple[Any, dict[str, Any]]],
    opt_update: Callable[..., tuple[Any, Any]],
    kl_fn: Callable[[Any, Any], Any],
    target_kl: float,
    update_epochs: int,
    minibatch_size: int,
    rng: jax.Array,
) -> tuple[Any, dict[str, Any]]:
    batch_size = batch.obs.shape[0]
    num_minibatches = batch_size // minibatch_size
    tkl = jnp.asarray(target_kl, dtype=jnp.float32)

    def _one_epoch(carry, epoch_rng):
        st, stopped = carry
        shuffled = _shuffle_batch(epoch_rng, batch)

        def _one_minibatch(carry, mb_idx):
            st, stopped = carry
            mb = _get_minibatch(shuffled, mb_idx * minibatch_size, minibatch_size)
            new_params, new_opt, actor_metrics = gradient_update(
                actor_params=st.actor_params,
                actor_opt_state=st.actor_opt,
                batch=mb,
                loss_fn=loss_fn,
                opt_update=opt_update,
            )
            st_new = st.replace(actor_params=new_params, actor_opt=new_opt)
            st_out = jax.tree_util.tree_map(
                lambda new, old: jnp.where(stopped, old, new), st_new, st
            )
            return (st_out, stopped), actor_metrics

        (st, stopped), metrics = jax.lax.scan(
            _one_minibatch, (st, stopped), jnp.arange(num_minibatches)
        )

        kl = kl_fn(st.actor_params, batch)
        stopped = stopped | (kl > tkl)

        return (st, stopped), metrics

    epoch_rngs = jax.random.split(rng, update_epochs)
    (state, _), all_metrics = jax.lax.scan(
        _one_epoch, (state, jnp.array(False)), epoch_rngs
    )
    last_metrics = jax.tree_util.tree_map(lambda x: x[-1, -1], all_metrics)
    return state, last_metrics

def _compute_kl(new_params, batch, *, actor_apply_fn):
    new_mean, new_log_std = actor_apply_fn(new_params, batch.obs)
    return jnp.mean(
        gaussian_kl(batch.policy_mean, batch.policy_log_std, new_mean, new_log_std)
    )

class PPOLagrange(OnPolicyLagrangeMixin, OnPolicyBase):

    def update_actor(self, state, batch):
        new_params, new_opt, metrics = gradient_update(
            actor_params=state.actor_params, actor_opt_state=state.actor_opt,
            batch=batch, loss_fn=self.reward_loss_fn, opt_update=self.actor_opt.update,
        )
        return state.replace(actor_params=new_params, actor_opt=new_opt), metrics

    def update_actor_only(self, state, batch):
        lagrange = jnp.maximum(state.lagrange, 0.0)
        from algos.base import inject_lagrange_combined_adv
        batch = inject_lagrange_combined_adv(batch, lagrange)
        kl_fn = partial(_compute_kl, actor_apply_fn=self.actor.apply)
        return train_minibatch_actor_early_stop(
            state=state, batch=batch,
            loss_fn=self.reward_loss_fn,
            opt_update=self.actor_opt.update,
            kl_fn=kl_fn, target_kl=self.cfg.target_kl,
            update_epochs=self.cfg.update_epochs,
            minibatch_size=self.cfg.update_batch_size,
            rng=state.rng,
        )

__all__ = ["PPOLagrange"]
