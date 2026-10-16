# pyright: reportArgumentType=false, reportReturnType=false, reportCallIssue=false
from __future__ import annotations

from typing import Callable, Sequence, cast

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen.initializers import lecun_normal, variance_scaling

def safe_nan_zero(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def flatten_pytree(tree: object) -> tuple[jax.Array, Callable[[jax.Array], object]]:
    return jax.flatten_util.ravel_pytree(tree)

def mse(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((x - y) ** 2)

def _maybe_clip_log_std(log_std: jnp.ndarray, clip_log_std: bool) -> jnp.ndarray:
    return jnp.where(clip_log_std, jnp.clip(log_std, -20.0, 2.0), log_std)

def _gaussian_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    action: jnp.ndarray,
    clip_log_std: bool = False,
) -> jnp.ndarray:
    log_std = _maybe_clip_log_std(log_std, clip_log_std)
    var = jnp.exp(2.0 * log_std)
    log_prob = -0.5 * (
        ((action - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    )
    return jnp.sum(log_prob, axis=-1)

def gaussian_log_prob_tanh(
    pre_tanh: jnp.ndarray,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    clip_log_std: bool = False,
) -> jnp.ndarray:
    logp = _gaussian_log_prob(mean, log_std, pre_tanh, clip_log_std)
    squash = jnp.sum(
        jnp.log(jnp.clip(1.0 - jnp.tanh(pre_tanh) ** 2, 1e-6, 1.0)), axis=-1
    )
    return logp - squash

def tanh_action_to_pre_tanh(action: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    clipped = jnp.clip(action, -1.0 + eps, 1.0 - eps)
    return jnp.arctanh(clipped)

def gaussian_log_prob_from_tanh_action(
    action: jnp.ndarray,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    clip_log_std: bool = False,
) -> jnp.ndarray:
    pre_tanh = tanh_action_to_pre_tanh(action)
    return gaussian_log_prob_tanh(pre_tanh, mean, log_std, clip_log_std)

def gaussian_entropy(log_std: jnp.ndarray, clip_log_std: bool = False) -> jnp.ndarray:
    log_std = _maybe_clip_log_std(log_std, clip_log_std)
    return jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1)

def gaussian_sample(
    key: jax.Array,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    clip_log_std: bool = False,
) -> jnp.ndarray:
    log_std = _maybe_clip_log_std(log_std, clip_log_std)
    std = jnp.exp(log_std)
    return mean + std * jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)

def gaussian_kl(
    mean_old: jnp.ndarray,
    log_std_old: jnp.ndarray,
    mean_new: jnp.ndarray,
    log_std_new: jnp.ndarray,
    clip_log_std: bool = False,
) -> jnp.ndarray:
    log_std_old = _maybe_clip_log_std(log_std_old, clip_log_std)
    log_std_new = _maybe_clip_log_std(log_std_new, clip_log_std)
    var_old = jnp.exp(2.0 * log_std_old)
    var_new = jnp.exp(2.0 * log_std_new)
    diff = mean_old - mean_new
    kl_per_dim = (
        (log_std_new - log_std_old) + (var_old + diff**2) / (2.0 * var_new + 1e-8) - 0.5
    )
    return jnp.sum(kl_per_dim, axis=-1)

def get_activation(name: str) -> Callable[..., jnp.ndarray]:
    activations = {"swish": nn.swish, "tanh": jnp.tanh, "relu": nn.relu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Options: {list(activations.keys())}")
    return activations[name]

def get_kernel_init(name: str):
    if name == "lecun_normal":
        return lecun_normal()
    elif name == "kaiming_uniform":
        return variance_scaling(scale=1 / 3, mode="fan_in", distribution="uniform")
    elif name == "kaiming_standard":
        return variance_scaling(scale=2.0, mode="fan_in", distribution="uniform")
    elif name == "orthogonal":
        from flax.linen.initializers import orthogonal
        return orthogonal()
    else:
        raise ValueError(f"Unknown weight_init: {name}.")

def _pytorch_bias_init(fan_in: int):
    return nn.initializers.uniform(scale=1.0 / (float(fan_in) ** 0.5))

def build_mlp(
    x: jnp.ndarray,
    hidden_sizes: Sequence[int],
    *,
    activation: Callable = nn.swish,
    kernel_init=nn.initializers.lecun_normal(),
    layer_norm: bool = False,
    compute_dtype: jnp.dtype = jnp.float32,
    param_dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    for i, size in enumerate(hidden_sizes):
        fan_in = x.shape[-1]
        x = nn.Dense(
            size, name=f"fc_{i}", kernel_init=kernel_init,
            bias_init=_pytorch_bias_init(int(fan_in)),
            dtype=compute_dtype, param_dtype=param_dtype,
        )(x)
        if layer_norm:
            x = nn.LayerNorm(dtype=compute_dtype, param_dtype=param_dtype, name=f"ln_{i}")(x)
        x = activation(x)
    return x

@struct.dataclass
class RunningMeanStd:
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, obs_dim: int) -> "RunningMeanStd":
        return cls(
            mean=jnp.zeros((obs_dim,), dtype=jnp.float32),
            var=jnp.ones((obs_dim,), dtype=jnp.float32),
            count=jnp.array(1e-6, dtype=jnp.float32),
        )

    def update(self, obs_batch: jax.Array) -> "RunningMeanStd":
        obs_batch = jnp.atleast_2d(obs_batch)
        batch_mean = jnp.mean(obs_batch, axis=0)
        batch_var = jnp.var(obs_batch, axis=0)
        batch_size = jnp.array(obs_batch.shape[0], dtype=jnp.float32)
        new_count = self.count + batch_size
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_size / new_count
        new_var = (
            self.var * self.count
            + batch_var * batch_size
            + jnp.square(delta) * self.count * batch_size / new_count
        ) / new_count
        return RunningMeanStd(
            mean=new_mean,
            var=jnp.maximum(new_var, 1e-12),
            count=new_count,
        )

    def normalize(self, obs: jax.Array) -> jax.Array:
        normalized = (obs - self.mean) / jnp.sqrt(self.var + 1e-8)
        return jnp.clip(normalized, -10.0, 10.0)

class GaussianLearningActor(nn.Module):
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    log_std_init: float = 0.0
    activation: str = "swish"
    weight_init: str = "lecun_normal"
    layer_norm: bool = False
    compute_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        obs_rms: RunningMeanStd | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del obs_rms
        x = obs.astype(self.compute_dtype)
        x = build_mlp(
            x, self.hidden_sizes,
            activation=get_activation(self.activation),
            kernel_init=get_kernel_init(self.weight_init),
            layer_norm=self.layer_norm,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )
        mean = nn.Dense(
            self.action_dim, name="mean",
            kernel_init=get_kernel_init(self.weight_init),
            dtype=self.compute_dtype, param_dtype=self.param_dtype,
        )(x)
        log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.full(shape, self.log_std_init),
            (self.action_dim,),
        )
        return mean.astype(jnp.float32), jnp.broadcast_to(log_std, mean.shape).astype(jnp.float32)

    def sample_action(self, params, obs, rng, obs_rms=None):
        mean, log_std = cast(tuple, self.apply(params, obs, obs_rms))
        pre_tanh = gaussian_sample(rng, mean, log_std)
        action = jnp.tanh(pre_tanh)
        logp = gaussian_log_prob_tanh(pre_tanh, mean, log_std)
        return action, (logp, mean, log_std)

    def loss(self, params, batch, *, adv, entropy_coef=0.0, clip_ratio=float('inf')):
        mean, log_std = cast(tuple, self.apply(params, batch.obs))
        logp_new = gaussian_log_prob_from_tanh_action(batch.actions, mean, log_std)
        ratio = jnp.exp(logp_new - batch.logp)
        clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        objective = jnp.minimum(ratio * adv, clipped)
        entropy = jnp.mean(gaussian_entropy(log_std))
        policy_loss = -jnp.mean(objective) - jnp.asarray(entropy_coef, dtype=jnp.float32) * entropy
        return policy_loss, {"entropy": entropy, "ratio_mean": jnp.mean(ratio)}

class ValueCritic(nn.Module):
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "swish"
    weight_init: str = "lecun_normal"
    layer_norm: bool = False
    compute_dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        obs_rms: RunningMeanStd | None = None,
    ) -> jnp.ndarray:
        del obs_rms
        x = obs.astype(self.compute_dtype)
        x = build_mlp(
            x, self.hidden_sizes,
            activation=get_activation(self.activation),
            kernel_init=get_kernel_init(self.weight_init),
            layer_norm=self.layer_norm,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )
        raw = nn.Dense(
            1, name="value",
            kernel_init=get_kernel_init(self.weight_init),
            dtype=jnp.float32, param_dtype=self.param_dtype,
        )(x)
        return jnp.squeeze(raw, axis=-1).astype(jnp.float32)

    def loss(
        self, params, batch, *, target, target_transform,
        use_critic_norm=False, critic_norm_coef=0.001,
    ):
        value = self.apply(params, batch.obs)
        mse_loss = mse(target_transform(target), value)
        l2 = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y ** 2), params, initializer=0.0)
        return mse_loss + critic_norm_coef * l2 * use_critic_norm

__all__ = [
    # math_utils
    "flatten_pytree", "mse", "safe_nan_zero",
    # gaussian_math
    "tanh_action_to_pre_tanh", "gaussian_log_prob_from_tanh_action",
    "gaussian_log_prob_tanh", "gaussian_entropy", "gaussian_sample", "gaussian_kl",
    # common
    "RunningMeanStd", "build_mlp", "get_activation", "get_kernel_init",
    # models
    "GaussianLearningActor", "ValueCritic",
]
