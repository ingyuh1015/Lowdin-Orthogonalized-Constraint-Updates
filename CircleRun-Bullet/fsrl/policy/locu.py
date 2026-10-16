import json
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from itertools import combinations

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from torch import nn
from torch.distributions import kl_divergence
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger


class LOCU(BasePolicy):

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        optim: List[torch.optim.Optimizer],
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        target_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        damping_coeff: float = 0.1,
        max_backtracks: int = 10,
        optim_critic_iters: int = 20,
        l2_reg: float = 0.001,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
        cost_limit: Union[List, float] = np.inf,
        eps_dup: float = 1e-2,
        eps_dup_neg: float = 0.01,
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        use_relative_weights: bool = True,
        neutralize_reward: bool = True,
        use_screening: bool = True,
        use_lowdin: bool = True,
    ) -> None:
        super().__init__(
            actor, critics, dist_fn, logger, gamma, max_batchsize, reward_normalization,
            deterministic_eval, action_scaling, action_bound_method, observation_space,
            action_space, lr_scheduler
        )
        self.optim = optim
        self.cost_limit = cost_limit
        self._lambda = gae_lambda
        self._norm_adv = advantage_normalization
        self._max_backtracks = max_backtracks
        self._optim_critic_iters = optim_critic_iters
        self._l2_reg = l2_reg
        self._kl_limit = target_kl
        self._backtrack_coeff = backtrack_coeff
        self._damping_coeff = damping_coeff
        self._eps_dup = eps_dup
        self._eps_dup_neg = eps_dup_neg
        self.use_relative_weights = use_relative_weights
        self._neutralize_reward = neutralize_reward
        self._use_screening = use_screening
        self._use_lowdin = use_lowdin
        self.critics_num = 1 + len(cost_limit)

    def pre_update_fn(self, stats_train: Dict, **kwarg) -> Any:
        m_cost = stats_train.get("m_cost", [])
        if isinstance(m_cost, torch.Tensor):
            m_cost = m_cost.detach().cpu().numpy()
        m_cost = list(m_cost)
        target_len = max(0, self.critics_num - 1)
        if len(m_cost) >= target_len:
            m_cost = m_cost[:target_len]
        else:
            m_cost = m_cost + [0.0] * (target_len - len(m_cost))
        self._ave_cost_return = m_cost

    def update_cost_limit(self, cost_limit: float) -> None:
        self.cost_limit = (
            [cost_limit] * (self.critics_num - 1)
            if np.isscalar(cost_limit)
            else cost_limit
        )

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_gae_returns(batch, buffer, indices, self._lambda)
        if self._norm_adv:
            adv = batch.advs[..., 0]
            mean, std = adv.mean(), adv.std()
            batch.advs[..., 0] = (adv - mean) / (std + 1e-8)
        batch.act = to_torch_as(batch.act, batch.values[..., 0])
        old_log_prob, old_mean, old_std = [], [], []
        with torch.no_grad():
            for minibatch in batch.split(
                self._max_batchsize, shuffle=False, merge_last=True
            ):
                res = self.forward(minibatch)
                old_log_prob.append(res.dist.log_prob(minibatch.act))
                old_mean.append(res.logits[0, ...])
                old_std.append(res.logits[1, ...])
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.mean_old = torch.cat(old_mean, dim=0)
        batch.std_old = torch.cat(old_std, dim=0)
        return batch

    def critics_loss(self, minibatch: Batch) -> Tuple[torch.Tensor, dict]:
        stats = {}
        total_loss = 0.0
        for i in range(self.critics_num):
            critic = self.critics[i]
            optim = self.optim[i]
            value = critic(minibatch.obs).flatten()
            ret = minibatch.rets[..., i]
            vf_loss = (ret - value).pow(2).mean()
            for p in critic.parameters():
                vf_loss += self._l2_reg * p.pow(2).sum()
            optim.zero_grad()
            vf_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            optim.step()
            stats[f"loss/vf{i}"] = vf_loss.item()
            total_loss += vf_loss.item()
        stats["loss/vf_total"] = total_loss
        return torch.tensor(total_loss), stats

    def _get_objective(
        self, logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(torch.exp(logp - logp_old) * adv)

    def _get_cost_surrogate(
        self, logp: torch.Tensor, logp_old: torch.Tensor, cadv: torch.Tensor, i
    ) -> torch.Tensor:
        return self._ave_cost_return[i - 1] + torch.mean(
            torch.exp(logp - logp_old) * cadv
        )

    def _MVP(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        kl_v = torch.dot(flat_kl_grad, v.detach())
        hvp = torch.autograd.grad(kl_v, self.actor.parameters(), retain_graph=True)
        flat_hvp = torch.cat([g.view(-1) for g in hvp])
        return flat_hvp + self._damping_coeff * v

    def _MVP1(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        kl_v = torch.dot(flat_kl_grad, v.detach())
        hvp = torch.autograd.grad(kl_v, self.actor.parameters(), retain_graph=True)
        return torch.cat([g.view(-1) for g in hvp])

    def _conjugate_gradients(
        self,
        g: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        nsteps: int = 20,
        residual_tol: float = 1e-8,
    ) -> torch.Tensor:
        x = torch.zeros_like(g)
        r, p = g.clone(), g.clone()
        rs_old = torch.dot(r, r)
        for _ in range(nsteps):
            z = self._MVP(p, flat_kl_grad)
            alpha = rs_old / torch.dot(p, z)
            x += alpha * p
            r -= alpha * z
            rs_new = torch.dot(r, r)
            if rs_new < residual_tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def _get_flat_grad(
        self,
        y: torch.Tensor,
        model: nn.Module,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> torch.Tensor:
        retain_graph = True if create_graph else retain_graph
        grads = torch.autograd.grad(
            y, model.parameters(), retain_graph=retain_graph, create_graph=create_graph
        )
        return torch.cat([grad.view(-1) for grad in grads])

    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        return torch.cat([p.view(-1) for p in model.parameters()])

    def _set_from_flat_params(self, model: nn.Module, new_params: torch.Tensor) -> None:
        n = 0
        for param in model.parameters():
            numel = param.numel()
            param.data = new_params[n : n + numel].view(param.size())
            n += numel



    def policy_loss(self, minibatch: Batch) -> Tuple[torch.Tensor, dict]:
        self.actor.train()
        eps = 1e-8
        R = float((2 * self._kl_limit) ** 0.5)

        if not isinstance(self.cost_limit, list):
            full_limits = [self.cost_limit] * (self.critics_num - 1)
        else:
            full_limits = self.cost_limit

        dist = self.forward(minibatch).dist
        logp = dist.log_prob(minibatch.act)
        assert self.dist_fn is not None
        dist_old = self.dist_fn(*(minibatch.mean_old, minibatch.std_old))
        kl = kl_divergence(dist_old, dist).mean()
        flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)

        objective = self._get_objective(
            logp, minibatch.logp_old, minibatch.advs[..., 0]
        )
        g_tensor = self._get_flat_grad(objective, self.actor, retain_graph=True)
        H_inv_g_tensor = self._conjugate_gradients(g_tensor, flat_kl_grad)

        g_cs, tilde_gcs, c_scalars = [], [], []
        safety_mode = False
        limits_tensor = torch.as_tensor(
            full_limits, device=g_tensor.device, dtype=g_tensor.dtype
        )
        for i in range(1, self.critics_num):
            con_val = self._get_cost_surrogate(
                logp, minibatch.logp_old, minibatch.advs[..., i], i
            )
            b_tensor = self._get_flat_grad(con_val, self.actor, retain_graph=True)
            g_cs.append(b_tensor)
            tilde_gcs.append(self._conjugate_gradients(b_tensor, flat_kl_grad))
            c_scalars.append(limits_tensor[i - 1] - con_val)
            if bool((con_val > limits_tensor[i - 1]).item()):
                safety_mode = True

        Delta = (
            torch.stack(c_scalars)
            if c_scalars
            else torch.zeros(0, device=g_tensor.device)
        )

        E, a_vec, c_mat, retained_ids, basis_ids = \
            self._lowdin_basis_and_linear_vectors_multi(
                H_inv_g_tensor, tilde_gcs, g_tensor, g_cs, Delta, R,
                flat_kl_grad, full_limits, eps=eps,
            )

        violated_any = bool((Delta[torch.as_tensor(retained_ids, device=Delta.device)] < 0).any().item()) \
            if retained_ids else False
        Delta_ret = (
            Delta[torch.as_tensor(retained_ids, device=Delta.device)]
            if retained_ids
            else torch.zeros(0, device=Delta.device, dtype=Delta.dtype)
        )

        w_star = self._solve_qcqp_multi(
            a_vec, c_mat, Delta_ret, R, eps=eps,
            limits=Delta_ret, recovery_mode=violated_any,
        )
        if w_star is None:
            w_star = torch.zeros_like(a_vec)
        d_star = (E @ w_star).view(-1)

        init_theta = self._get_flat_params(self.actor).detach().clone()
        with torch.no_grad():
            self._set_from_flat_params(self.actor, init_theta + d_star)

        with torch.no_grad():
            dist = self.forward(minibatch).dist
            new_kl = kl_divergence(dist_old, dist).mean().item()


        return objective, {
            "loss/kl": new_kl,
            "reduced": int(E.shape[1]),
            "Update/safety_mode": int(safety_mode),
            "Update/ave_cost_0": self._ave_cost_return[0]
            if self._ave_cost_return
            else 0.0,
            "Update/w_norm": float(torch.norm(w_star).item()),
        }


    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        for epoch in range(repeat):
            self.logger.epoch = epoch
            for minibatch in batch.split(batch_size, merge_last=True):
                for _ in range(self._optim_critic_iters):
                    loss_vf, stats_critic = self.critics_loss(minibatch)
                loss_actor, stats_actor = self.policy_loss(minibatch)
                self.logger.store(**stats_actor)
                self.gradient_steps += 1
                self.logger.store(**stats_critic)
        self.logger.store(gradient_steps=self.gradient_steps, tab="update")

    def _compute_violation_recovery(
        self,
        c_mat: torch.Tensor,
        retained_ids: List[int],
        Delta: torch.Tensor,
        full_limits,
        eps: float,
    ) -> Tuple[torch.Tensor, List[int]]:
        device = c_mat.device
        dtype = c_mat.dtype
        ridx_tensor = torch.as_tensor(retained_ids, device=device)
        Delta_red = Delta[ridx_tensor]
        violated_mask_red = Delta_red < 0
        v_idx = violated_mask_red.nonzero().flatten().tolist()

        if not v_idx:
            return torch.zeros((c_mat.shape[0],), device=device, dtype=dtype), v_idx

        C = c_mat[:, v_idx]
        v_deltas = Delta_red[v_idx]

        v_norms = torch.norm(C, dim=0) + eps
        w_target = (-v_deltas) / v_norms

        P = csc_matrix(np.eye(len(v_idx), dtype=np.float64))
        q = -w_target.detach().cpu().numpy().astype(np.float64)
        lb = np.zeros(len(v_idx), dtype=np.float64)

        C_np = C.detach().cpu().numpy().astype(np.float64)
        G_mat = -(C_np.T @ C_np)
        h_vec = -1e-9 * np.ones(len(v_idx), dtype=np.float64)

        violated_pareto = solve_qp(
            P=P,
            q=q,
            G=G_mat,
            h=h_vec,
            lb=lb,
            solver="osqp",
            eps_abs=1e-8,
            eps_rel=1e-8,
        )

        def _fallback_single() -> torch.Tensor:
            v_mu = v_deltas / (v_norms + eps)  
            j = int(torch.argmin(v_mu).item())
            col = C[:, j]
            return -col / (torch.norm(col) + eps)

        if violated_pareto is not None:
            p_weights = torch.tensor(violated_pareto, device=device, dtype=dtype)
            r_vec = C @ p_weights
            norm_r = torch.norm(r_vec)
            if float(norm_r.item()) <= 1e-12:
                a_vec = _fallback_single()
            else:
                a_vec = -r_vec / (norm_r + eps)
        else:
            r_vec = C.sum(dim=1)
            norm_r = torch.norm(r_vec)
            if float(norm_r.item()) <= 1e-12:
                a_vec = _fallback_single()
            else:
                a_vec = -r_vec / (norm_r + eps)

        return a_vec, v_idx

    def _pick_rep(self, comp_ids, Delta, nu, mu, R, corr_r=None):
        
        if len(comp_ids) == 1:
            return comp_ids[0].item()
        if (Delta[comp_ids] < 0).any() or corr_r is None:
            group_nus = nu[comp_ids]
            if (Delta[comp_ids] < -(R * group_nus)).any():
                return comp_ids[int(torch.argmax(group_nus).item())].item()
            return comp_ids[int(torch.argmin(mu[comp_ids]).item())].item()
        return comp_ids[int(torch.argmax(corr_r[comp_ids]).item())].item()

    def _lowdin_basis_and_linear_vectors_multi(
        self,
        tilde_gr,
        tilde_gcs,
        g_r,
        g_cs,
        Delta,
        R,
        flat_kl_grad,
        full_limits,
        eps=1e-8,
        eps_nz=1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
        device = tilde_gr.device
        Delta = torch.as_tensor(Delta, device=device, dtype=tilde_gr.dtype)
        eps_dup = self._eps_dup
        eps_dup_neg = self._eps_dup_neg
        K = len(tilde_gcs)

        if K == 0:
            fnorm = torch.sqrt(
                torch.clamp(tilde_gr @ self._MVP1(tilde_gr, flat_kl_grad), min=eps)
            )
            E = tilde_gr.unsqueeze(1) / fnorm
            return (E, E.T @ g_r, E.new_zeros((1, 0)), [], [])

        S_full = torch.stack([tilde_gr] + tilde_gcs, dim=1)
        FS_full = torch.stack(
            [self._MVP1(S_full[:, j], flat_kl_grad) for j in range(K + 1)], dim=1
        )
        O_full = 0.5 * (S_full.T @ FS_full + (S_full.T @ FS_full).T)

        diag_full = torch.diag(O_full)
        nu_r = torch.sqrt(torch.clamp(diag_full[0], min=0.0))
        fnorm_sq = diag_full[1:]
        nu = torch.sqrt(torch.clamp(fnorm_sq, min=0.0))
        mu = Delta / (nu + eps)

        denom_cc = nu[:, None] * nu[None, :] + eps
        corr_cc = O_full[1:, 1:] / denom_cc
        denom_r = nu_r * nu + eps
        corr_r = O_full[0, 1:] / denom_r

        violated_full_mask = Delta < 0
        violated_any_full = bool(violated_full_mask.any().item())
        eps_nz_eff = eps_nz if not violated_any_full else min(eps_nz, 1e-12)
        fnorm_mask = fnorm_sq > eps_nz_eff
        margin_mask = Delta < (nu * R)

        if not violated_any_full:
            keep_mask = fnorm_mask & margin_mask
        else:
            
            keep_mask = (fnorm_mask & margin_mask) | violated_full_mask
        kept_ids = keep_mask.nonzero(as_tuple=False).flatten().tolist()

        if not kept_ids:
            nu_r = torch.sqrt(torch.clamp(O_full[0, 0], min=eps))
            E = S_full[:, 0:1] / nu_r
            return (
                E,
                E.T @ g_r,
                torch.zeros((E.shape[1], 0), device=device, dtype=E.dtype),
                [],
                [],
            )

        retained_ids = kept_ids
        ridx = torch.as_tensor(retained_ids, device=device, dtype=torch.long)
        violated_any = bool((Delta[ridx] < 0).any().item())

        corr_r_ret = corr_r[ridx]
        near_reward_mask = (corr_r_ret >= (1.0 - eps_dup_neg)) | (
            corr_r_ret <= (-1.0 + eps_dup_neg)
        )

        cand_mask = ~near_reward_mask
        cand_ids = ridx[cand_mask]

        basis_ids: List[int] = []
        if cand_ids.numel() > 0:
            O_sub_indices = cand_ids + 1
            O_sub = O_full[O_sub_indices][:, O_sub_indices]
            diag_sub = torch.sqrt(torch.diag(O_sub))
            outer_sub = torch.outer(diag_sub, diag_sub) + eps
            corr_sub_robust = O_sub / outer_sub

            adj_pos = corr_sub_robust >= (1.0 - eps_dup)
            adj_pos.fill_diagonal_(False)

            adj_pos_np = adj_pos.cpu().numpy().astype(np.uint8)
            n_pos_comps, pos_labels = connected_components(
                csr_matrix(adj_pos_np), directed=False
            )

            pos_reps = []
            for i in range(n_pos_comps):
                comp_ids = cand_ids[
                    torch.as_tensor(np.where(pos_labels == i)[0], device=device)
                ]
                pos_reps.append(self._pick_rep(comp_ids, Delta, nu, mu, R))

            if pos_reps:
                ridx_pos = torch.as_tensor(pos_reps, device=device, dtype=torch.long)
                corr_pos = corr_cc[ridx_pos][:, ridx_pos]

                adj_neg = corr_pos <= (-1.0 + eps_dup_neg)
                adj_neg.fill_diagonal_(False)
                adj_neg_np = adj_neg.cpu().numpy().astype(np.uint8)
                n_final_comps, final_labels = connected_components(
                    csr_matrix(adj_neg_np), directed=False
                )

                for i in range(n_final_comps):
                    comp_rep_ids = ridx_pos[
                        torch.as_tensor(np.where(final_labels == i)[0], device=device)
                    ]
                    basis_ids.append(
                        self._pick_rep(comp_rep_ids, Delta, nu, mu, R, corr_r)
                    )

        basis_ids = sorted(list(set(basis_ids)))
        basis_cols = [0] + [bid + 1 for bid in basis_ids]

        S_B = S_full[:, basis_cols]

        O_B = 0.5 * (
            O_full[basis_cols][:, basis_cols] + O_full[basis_cols][:, basis_cols].T
        )
        eigvals, eigvecs = torch.linalg.eigh(O_B)
        eigvals = torch.clamp(eigvals, min=eps)
        O_inv_sqrt = eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T
        E = S_B @ O_inv_sqrt

        if violated_any and E.shape[1] > 1:
            E = E[:, 1:]

        g_cs_stacked = (
            torch.stack([g_cs[k] for k in retained_ids], dim=1)
            if retained_ids
            else torch.zeros((g_r.shape[0], 0), device=device, dtype=g_r.dtype)
        )
        c_mat = E.T @ g_cs_stacked

        if violated_any:
            recovery_dir, v_idx = self._compute_violation_recovery(
                c_mat, retained_ids, Delta, full_limits, eps
            )
            a_vec = recovery_dir
        else:
            a_vec = E.T @ g_r

        return E, a_vec, c_mat, retained_ids, basis_ids

    def _solve_qcqp_multi(
        self, a_vec, c_mat, Delta, R, eps=1e-8, limits=None, recovery_mode=False
    ):
        tol = 1e-8
        if recovery_mode:
            safe_mask = Delta >= 0
            if safe_mask.any():
                c_mat = c_mat[:, safe_mask]
                Delta = Delta[safe_mask]
            else:
                a_norm = torch.norm(a_vec)
                return R * a_vec / (a_norm + eps)

        a_norm = torch.norm(a_vec)
        w_uc = R * a_vec / (a_norm + eps)
        vals_uc = c_mat.T @ w_uc
        if torch.all(vals_uc <= Delta - tol):
            return w_uc

        
        violated = (vals_uc > Delta + tol).nonzero(as_tuple=False).flatten()
        if violated.numel() == 0:
            return w_uc
        c_norms = torch.norm(c_mat[:, violated], dim=0) + eps
        order = torch.argsort(Delta[violated] / c_norms)
        candidates = violated[order].tolist()

        
        for idx in candidates:
            w = self._solve_active_set_expansion([idx], a_vec, c_mat, R, Delta, eps)
            if w is not None:
                return w

        
        dots = c_mat.T @ w_uc
        pos = dots > eps
        if torch.any(pos):
            alpha = float((Delta[pos] / (dots[pos] + eps)).min().item())
            if 0.0 < alpha <= 1.0:
                return alpha * w_uc
        return torch.zeros_like(a_vec)

    def _solve_active_set(
        self, S, a_vec, c_mat, R, Delta, eps=1e-6, max_drop_steps: int = 10
    ):
        
        device = a_vec.device
        drop_tol = 1e2 * eps

        for _ in range(max_drop_steps):
            p = len(S)
            if p == 0:
                
                return None

            A_S = c_mat[:, S].T
            b_S = Delta[torch.as_tensor(S, device=Delta.device)]
            M = A_S @ A_S.T
            eye = torch.eye(p, device=device)
            A_S_a = A_S @ a_vec

            sol = None
            for scale in (1.0, 1e2, 1e4):
                try:
                    L = torch.linalg.cholesky(M + (eps * scale) * eye)
                    sol = torch.cholesky_solve(torch.stack([b_S, A_S_a], dim=1), L)
                    break
                except RuntimeError:
                    continue
            if sol is None:
                return None

            M_inv_b, M_inv_A_S_a = sol[:, 0], sol[:, 1]
            w0 = A_S.T @ M_inv_b
            a_perp = a_vec - A_S.T @ M_inv_A_S_a

            w0_norm2 = torch.dot(w0, w0)
            A = w0_norm2 - R * R
            B = 2 * torch.dot(a_perp, w0)
            C = torch.dot(a_perp, a_perp)
            disc = B * B - 4 * A * C
            if disc < 0 and disc > -1e-10:
                disc = disc * 0.0
            if disc < 0:
                return None

            sqrt_disc = torch.sqrt(disc + eps)
            if abs(A) < eps:
                if abs(B) < eps:
                    return None
                beta = -C / B
            elif A > 0:
                beta = (-B + sqrt_disc) / (2.0 * A)
            else:
                beta = (-B - sqrt_disc) / (2.0 * A)
            if beta <= 0:
                return None

            
            try:
                lambda_S = torch.linalg.solve(M + eps * eye, A_S_a - beta * b_S)
            except RuntimeError:
                return None

            min_lam = float(lambda_S.min().item())
            if min_lam < -drop_tol:
                
                j = int(torch.argmin(lambda_S).item())
                S.pop(j)
                continue

            return (a_vec - A_S.T @ lambda_S) / (beta + eps)

        
        return None

    def _solve_active_set_expansion(
        self, S_start, a_vec, c_mat, R, Delta, eps=1e-6, max_steps=5
    ):
        S = list(S_start)
        for _ in range(max_steps):
            w = self._solve_active_set(S, a_vec, c_mat, R, Delta, eps)
            if w is None:
                return None
            vals = c_mat.T @ w
            max_viol_val, max_viol_idx = (vals - Delta).max(dim=0)
            if max_viol_val <= 1e-6:
                return w
            new_c = max_viol_idx.item()
            if new_c in S:
                return None
            S.append(new_c)
        return None
