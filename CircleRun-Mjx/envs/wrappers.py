from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from envs.env import EnvState

class Wrapper:

    def __init__(self, env):
        self.env = env

    def reset(self, rng: jax.Array) -> EnvState:
        return self.env.reset(rng)

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def mj_model(self):
        return self.env.mj_model

    @property
    def mjx_model(self):
        return self.env.mjx_model

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

class EpisodeWrapper(Wrapper):

    def __init__(
        self,
        env: Any,
        episode_length: int,
        *,
        gamma: float = 1.0,
    ):
        super().__init__(env)
        self.episode_length = int(episode_length)
        self.gamma = float(gamma)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    def reset(self, rng: jax.Array) -> EnvState:
        rng_env, rng_autoreset = jax.random.split(rng)
        state = self.env.reset(rng_env)
        zeros = jnp.zeros_like(jnp.asarray(state.done, dtype=jnp.float32))
        state.info["steps"] = jnp.zeros_like(jnp.asarray(state.done, dtype=jnp.int32))
        state.info["truncation"] = zeros
        state.info["episode_done"] = zeros
        state.info["move_dist"] = zeros
        step_cost = jnp.asarray(state.cost_vector, dtype=jnp.float32)
        state.info["step_cost"] = step_cost
        cost_zeros = jnp.zeros_like(step_cost)
        state.info["discounted_rewards"] = zeros
        state.info["discounted_costs"] = cost_zeros
        episode_metrics = {
            "sum_reward": zeros,
            "sum_d_reward": zeros,
            "length": zeros,
            "sum_r_costs": cost_zeros,
            "sum_d_costs": cost_zeros,
        }
        state.info["episode_metrics"] = episode_metrics
        state.info["terminal_episode_metrics"] = episode_metrics
        num_envs = state.done.shape[0]
        state.info["rng"] = jax.random.split(rng_autoreset, num_envs)
        state.info["terminal_obs"] = state.obs
        return state.replace(
            terminated=jnp.zeros_like(state.done),
            truncated=jnp.zeros_like(state.done),
            done=jnp.zeros_like(state.done),
        )

    def _auto_reset_done_envs(
        self,
        *,
        state: EnvState,
        done: jax.Array,
        done_mask: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        rng_reset: jax.Array,
        rng_next: jax.Array,
    ) -> EnvState:
        def _reset_branch(args):
            current_state, reset_keys = args
            reset_state = self.env.reset(reset_keys)

            def _select(reset_val, current_val):
                mask = done_mask.reshape(
                    done_mask.shape + (1,) * (reset_val.ndim - done_mask.ndim)
                )
                return jnp.where(mask, reset_val, current_val)

            aligned_info = {
                key: reset_state.info[key] if key in reset_state.info
                else jax.tree.map(jnp.zeros_like, current_state.info[key])
                for key in current_state.info
            }
            reset_state = reset_state.replace(info=aligned_info)
            return jax.tree.map(_select, reset_state, current_state)

        def _no_reset_branch(args):
            current_state, _ = args
            return current_state

        auto_reset_state = jax.lax.cond(
            jnp.any(done_mask),
            _reset_branch,
            _no_reset_branch,
            (state, rng_reset),
        )

        auto_reset_state.info["terminal_obs"] = state.info["terminal_obs"]
        auto_reset_state.info["terminal_episode_metrics"] = state.info["terminal_episode_metrics"]
        auto_reset_state.info["rng"] = rng_next

        auto_reset_state.info["steps"] = jnp.where(
            done_mask,
            jnp.zeros_like(state.info["steps"]),
            state.info["steps"],
        )
        auto_reset_state.info["episode_done"] = jnp.where(
            done_mask,
            jnp.zeros_like(done),
            done,
        )

        return auto_reset_state.replace(
            done=done,
            terminated=terminated,
            truncated=truncated,
        )

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        per_env_rng = state.info["rng"]
        rng_pairs = jax.vmap(jax.random.split)(per_env_rng)
        rng_next = rng_pairs[:, 0]
        rng_reset = rng_pairs[:, 1]

        nstate = self.env.step(state, action)
        merged_info = dict(state.info)
        merged_info.update(nstate.info)
        state = nstate.replace(info=merged_info)
        prev_steps = state.info["steps"]
        steps = prev_steps + 1

        base_term = jnp.asarray(state.terminated, dtype=jnp.float32)
        base_trunc = jnp.asarray(state.truncated, dtype=jnp.float32)

        zero = jnp.zeros_like(base_term)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        time_trunc = jnp.where(
            steps >= episode_length,
            1.0 - jnp.maximum(base_term, base_trunc),
            zero,
        )
        terminated = base_term
        truncated = jnp.maximum(base_trunc, time_trunc)
        done = jnp.maximum(terminated, truncated)
        state.info["truncation"] = jnp.where(steps >= episode_length, time_trunc, zero)
        state.info["steps"] = steps

        prev_done = state.info["episode_done"]
        alive = 1 - prev_done
        gamma_power = jnp.power(jnp.asarray(self.gamma, dtype=jnp.float32), prev_steps)
        step_cost = jnp.asarray(state.cost_vector, dtype=jnp.float32)
        state.info["step_cost"] = step_cost
        state.info["discounted_rewards"] = gamma_power * state.reward
        state.info["discounted_costs"] = gamma_power[..., None] * step_cost

        em = dict(state.info["episode_metrics"])
        em["sum_reward"] = (em["sum_reward"] + state.reward) * alive
        em["sum_d_reward"] = (em["sum_d_reward"] + state.info["discounted_rewards"]) * alive
        em["length"] = (em["length"] + 1) * alive
        em["sum_r_costs"] = (em["sum_r_costs"] + step_cost) * alive[..., None]
        em["sum_d_costs"] = (em["sum_d_costs"] + state.info["discounted_costs"]) * alive[..., None]
        state.info["episode_metrics"] = em
        state.info["episode_done"] = done

        state = getattr(state, "replace")(
            done=done,
            terminated=terminated,
            truncated=truncated,
        )

        state.info["terminal_obs"] = state.obs
        done_mask = done.astype(bool)
        def _where_terminal(new_val, old_val):
            mask = done_mask.reshape(done_mask.shape + (1,) * (new_val.ndim - done_mask.ndim))
            return jnp.where(mask, new_val, old_val)
        state.info["terminal_episode_metrics"] = jax.tree.map(
            _where_terminal,
            state.info["episode_metrics"],
            state.info["terminal_episode_metrics"],
        )

        return self._auto_reset_done_envs(
            state=state,
            done=done,
            done_mask=done_mask,
            terminated=terminated,
            truncated=truncated,
            rng_reset=rng_reset,
            rng_next=rng_next,
        )

class VmapWrapper(Wrapper):

    def __init__(self, env: Any, batch_size: int):
        super().__init__(env)
        self.batch_size = int(batch_size)

    def _ensure_batched(self, value):
        def _broadcast(leaf):
            shape = getattr(leaf, "shape", None)
            if shape == ():
                return jax.numpy.broadcast_to(leaf, (self.batch_size,))
            return leaf

        return jax.tree_util.tree_map(_broadcast, value)

    def reset(self, rng: jax.Array) -> EnvState:
        if rng.ndim == 1:
            keys = jax.random.split(rng, self.batch_size)
        else:
            keys = rng
        state = jax.vmap(self.env.reset)(keys)
        return self._ensure_batched(state)

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        state = self._ensure_batched(state)
        if action.ndim == 1:
            action = jax.numpy.broadcast_to(action, (self.batch_size, action.shape[0]))
        return jax.vmap(self.env.step)(state, action)

__all__ = ["Wrapper", "EpisodeWrapper", "VmapWrapper"]
