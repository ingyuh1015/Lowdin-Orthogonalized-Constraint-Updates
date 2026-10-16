# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportIncompatibleMethodOverride=false
from __future__ import annotations

from functools import partial
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp
import optax
from flax import struct

from config.config import build_network, build_optimizer
from networks import RunningMeanStd
from algos.constants import EPS

from algos.util import _shuffle_batch, _get_minibatch

def lagrange_update(
    state: Any,
    *,
    cost_signal: Any,
    cost_limits: jnp.ndarray,
    opt_update: Callable,
) -> tuple[Any, dict[str, Any]]:
    delta = cost_signal - cost_limits
    grad = -delta  # Optax minimizes, so this update adds lr * delta to lambda.

    updates, new_opt = opt_update(grad, state.lagrange_opt)
    new_lagrange = jnp.maximum(
        optax.apply_updates(state.lagrange, updates),
        0.0,
    )

    return state.replace(
        lagrange=new_lagrange, lagrange_opt=new_opt,
    ), {
        "lagrange": new_lagrange,
        "lagrange_delta": delta,
    }

def train_minibatch_critics(
    *,
    state: Any,
    batch: Any,
    update_critics_fn: Callable[[Any, Any], tuple[Any, dict[str, Any]]],
    update_epochs: int,
    minibatch_size: int,
    rng: jax.Array,
) -> tuple[Any, dict[str, Any]]:
    batch_size = batch.obs.shape[0]
    num_minibatches = batch_size // minibatch_size

    def _one_epoch(carry, epoch_rng):
        st = carry
        shuffled = _shuffle_batch(epoch_rng, batch)

        def _one_minibatch(carry, mb_idx):
            st = carry
            mb = _get_minibatch(shuffled, mb_idx * minibatch_size, minibatch_size)
            st, critic_metrics = update_critics_fn(st, mb)
            return st, critic_metrics

        st, metrics = jax.lax.scan(_one_minibatch, st, jnp.arange(num_minibatches))
        return st, metrics

    epoch_rngs = jax.random.split(rng, update_epochs)
    state, all_metrics = jax.lax.scan(_one_epoch, state, epoch_rngs)
    last_metrics = jax.tree_util.tree_map(lambda x: x[-1, -1], all_metrics)
    return state, last_metrics

@struct.dataclass
class BasePolicyState:
    actor_params: Any
    actor_opt: optax.OptState
    rng: jax.Array

@struct.dataclass
class OnPolicyState(BasePolicyState):
    critic_params: Any
    critic_opt: Any
    obs_rms: RunningMeanStd

@struct.dataclass
class LagrangeOnPolicyState(OnPolicyState):
    lagrange: Any = None
    lagrange_opt: Any = None

def _critic_update(
    params: Any,
    opt_state: Any,
    loss_fn: Callable[[Any], jnp.ndarray],
    opt_update: Callable[..., tuple[Any, Any]],
) -> tuple[Any, Any, jnp.ndarray]:
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt = opt_update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt, loss

class OnPolicyBase:

    policy_type: ClassVar[str] = "on_policy"
    action_distribution: ClassVar[str] = "gaussian"

    def __init__(self, *, config: Any, compute_dtype: Any | None = None) -> None:
        self.cfg = config
        hs = tuple(self.cfg.hidden_sizes)
        dtype = compute_dtype or jnp.float32

        self.actor = build_network(
            "gaussian",
            action_dim=self.cfg.action_size, hidden_sizes=hs,
            activation=self.cfg.activation, weight_init=self.cfg.weight_init,
            layer_norm=self.cfg.layer_norm,
            compute_dtype=dtype,
        )
        self.critic = build_network(
            "value_critic", hidden_sizes=hs, activation=self.cfg.activation,
            weight_init=self.cfg.weight_init,
            layer_norm=self.cfg.layer_norm, compute_dtype=dtype,
        )
        self.actor_opt = build_optimizer(
            "adam", lr=self.cfg.actor_lr, max_grad_norm=self.cfg.max_grad_norm,
        )
        self.critic_opt = build_optimizer(
            "adam", lr=self.cfg.critic_lr, max_grad_norm=self.cfg.max_grad_norm,
        )

        if self.cfg.critic_target_transform != "none":
            raise ValueError(
                f"Unknown critic_target_transform: {self.cfg.critic_target_transform!r}. Only 'none' is supported."
            )

        self.actor_loss = partial(
            self.actor.loss,
            entropy_coef=self.cfg.entropy_coef,
            clip_ratio=getattr(self.cfg, "clip_ratio", float("inf")),
        )
        self.reward_loss_fn = lambda p, b: self.actor_loss(p, b, adv=b.advantages[..., 0])

        if self.cfg.obs_normalize:
            obs_size = self.cfg.obs_size
            self._update_obs_stats = lambda obs, state: state.replace(
                obs_rms=state.obs_rms.update(obs.reshape((-1, obs_size))),
            )
        else:
            self._update_obs_stats = lambda obs, state: state

    def init_state(self, rng: jax.Array) -> OnPolicyState:
        k_a, k_c = jax.random.split(rng, 2)
        obs = jnp.zeros((self.cfg.obs_size,))
        rms = RunningMeanStd.create(self.cfg.obs_size)
        a_p = self.actor.init(k_a, obs, rms)
        critic_keys = jax.random.split(k_c, 1 + self.cfg.num_constraints)
        critic_params = jax.vmap(lambda k: self.critic.init(k, obs, rms))(critic_keys)
        critic_opt = jax.vmap(self.critic_opt.init)(critic_params)
        return OnPolicyState(
            actor_params=a_p, critic_params=critic_params,
            actor_opt=self.actor_opt.init(a_p), critic_opt=critic_opt,
            rng=rng, obs_rms=rms,
        )

    def update_lagrange(self, state: Any, batch: Any) -> tuple[Any, dict[str, Any]]:
        del batch
        return state, {}

    def update_actor(self, state: Any, batch: Any) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    def update_actor_only(self, state, batch) -> tuple[Any, dict[str, Any]]:
        return self.update_actor(state, batch)

    def critic_step(self, state, batch):
        targets = jnp.moveaxis(batch.returns, -1, 0)
        critic_loss = self.critic.loss
        opt_update = self.critic_opt.update
        use_norm = self.cfg.use_critic_norm
        norm_coef = self.cfg.critic_norm_coef

        def per_slot_update(params, opt, target):
            loss_fn = lambda p: critic_loss(
                p, batch, target=target,
                use_critic_norm=use_norm,
                critic_norm_coef=norm_coef,
            )
            return _critic_update(
                params=params, opt_state=opt,
                loss_fn=loss_fn, opt_update=opt_update,
            )

        new_p, new_o, losses = jax.vmap(per_slot_update)(
            state.critic_params, state.critic_opt, targets,
        )
        value_means = jax.vmap(
            lambda p: jnp.mean(self.critic.apply(p, batch.obs))
        )(new_p)
        return state.replace(critic_params=new_p, critic_opt=new_o), {
            "reward_critic_loss": losses[0],
            "cost_critic_loss": losses[1:],
            "reward_value_mean": value_means[0],
            "cost_value_mean": value_means[1:],
        }

    def update_critics(self, state, batch):
        return train_minibatch_critics(
            state=state, batch=batch,
            update_critics_fn=self.critic_step,
            update_epochs=self.cfg.update_epochs,
            minibatch_size=self.cfg.update_batch_size,
            rng=state.rng,
        )

def inject_lagrange_combined_adv(batch, effective_lagrange):
    reward_adv = batch.advantages[..., 0]
    cost_advs = batch.advantages[..., 1:]
    combined = (reward_adv - jnp.sum(effective_lagrange * cost_advs, axis=-1)) / (
        1.0 + jnp.sum(effective_lagrange)
    )
    combined = (combined - jnp.mean(combined)) / (jnp.std(combined) + EPS)
    return batch.replace(advantages=batch.advantages.at[..., 0].set(combined))

class OnPolicyLagrangeMixin:

    dual_updater: ClassVar[str] = "naive"

    def __init__(self, *, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.lagrange_opt = optax.adam(self.cfg.lagrange_lr)

    def init_state(self, rng):
        state = super().init_state(rng)
        lagrange = (
            jnp.zeros((self.cfg.num_constraints,), dtype=jnp.float32)
            + self.cfg.lagrange_multiplier_init
        )
        state_dict = {k: v for k, v in state.__dict__.items()}
        state_dict.update(
            lagrange=lagrange,
            lagrange_opt=self.lagrange_opt.init(lagrange),
        )
        return LagrangeOnPolicyState(**state_dict)

    def update_actor_only(self, state, batch):
        lagrange = jnp.maximum(state.lagrange, 0.0)
        batch = inject_lagrange_combined_adv(batch, lagrange)
        return super().update_actor_only(state, batch)

    def update_lagrange(self, state, batch):
        return lagrange_update(
            state,
            cost_signal=batch.cost_signal,
            cost_limits=batch.cost_limit,
            opt_update=self.lagrange_opt.update,
        )

__all__ = [
    "BasePolicyState",
    "OnPolicyState",
    "LagrangeOnPolicyState",
    "EPS",
    "OnPolicyBase",
    "OnPolicyLagrangeMixin",
    "inject_lagrange_combined_adv",
    "lagrange_update",
    "train_minibatch_critics",
]
