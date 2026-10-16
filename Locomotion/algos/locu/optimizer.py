from algos.common.optimizer_base import TROptimizer, flatGrad

import torch
import numpy as np
from typing import List, Tuple
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

EPS = 1e-8


class LOCUOptimizer(TROptimizer):
    def __init__(
        self,
        device,
        actor,
        damping_coeff,
        num_conjugate,
        line_decay,
        max_kl,
        con_thresholds,
        con_zeta,
        con_entropy,
        eps_dup: float = 0.01,
        eps_dup_neg: float = 0.01,
    ) -> None:
        super().__init__(device, actor, damping_coeff, num_conjugate, line_decay, max_kl)
        self.con_thresholds = torch.as_tensor(con_thresholds, device=device, dtype=torch.float32)
        self.con_zeta = con_zeta
        self.con_entropy = con_entropy
        self.n_cons = len(con_thresholds)
        self._eps_dup = eps_dup
        self._eps_dup_neg = eps_dup_neg

    def _MVP1(self, v: torch.Tensor, flat_kl_grad=None) -> torch.Tensor:
        kl_v = torch.dot(self._flat_grad_kl, v.detach())
        return flatGrad(kl_v, self.actor.parameters(), retain_graph=True)

    def step(self, get_obj_con_kl, mu_kl: float = 0.0):
        max_kl = self._getMaxKL(mu_kl)
        eps = EPS
        radius = float((2.0 * max_kl) ** 0.5)

        objective, constraints, kl = get_obj_con_kl()
        self._computeKLGrad(kl)

        g_tensor = flatGrad(objective, self.actor.parameters(), retain_graph=True)
        H_inv_g_tensor = self._conjugateGradient(g_tensor)

        b_tensors = []
        H_inv_b_tensors = []
        deltas = []
        safety_mode = False
        limits_tensor = self.con_thresholds.to(g_tensor.dtype)
        for con_idx in range(self.n_cons):
            con_val = constraints[con_idx]
            b_tensor = flatGrad(con_val, self.actor.parameters(), retain_graph=True)
            H_inv_b_tensor = self._conjugateGradient(b_tensor)
            delta = limits_tensor[con_idx] - con_val
            if bool((con_val > limits_tensor[con_idx]).item()):
                safety_mode = True
            b_tensors.append(b_tensor)
            H_inv_b_tensors.append(H_inv_b_tensor)
            deltas.append(delta)

        Delta = torch.stack(deltas) if deltas else torch.zeros(0, device=self.device, dtype=g_tensor.dtype)

        E, a_vec, c_mat, retained_ids, basis_ids = self._lowdin_basis_and_linear_vectors_multi(
            H_inv_g_tensor, H_inv_b_tensors, g_tensor, b_tensors,
            Delta, radius, self._flat_grad_kl, limits_tensor, eps=eps,
        )
        retained_idx = torch.as_tensor(retained_ids, device=Delta.device, dtype=torch.long)
        Delta_ret = Delta[retained_idx]
        violated_any = bool((Delta_ret < 0).any().item())
        w = self._solve_qcqp_multi(
            a_vec, c_mat, Delta_ret, radius, eps=eps,
            limits=Delta_ret, recovery_mode=violated_any,
        )
        if w is None:
            w = torch.zeros_like(a_vec)
        step_dir = (E @ w).view(-1)

        n_retained = len(retained_ids)
        n_inactive = self.n_cons - n_retained
        n_duplicate = n_retained - len(basis_ids)
        n_zero_grad = sum(int(torch.norm(v).item() <= eps) for v in H_inv_b_tensors)
        n_reduced = self.n_cons - len(basis_ids)

        with torch.no_grad():
            init_theta = torch.cat([p.view(-1) for p in self.actor.parameters()]).detach().clone()
            self._applyParams(init_theta + step_dir)

        objective, constraints, kl = get_obj_con_kl()
        reduce_info = {
            'n_reduced': n_reduced,
            'n_inactive': n_inactive,
            'n_duplicate': n_duplicate,
            'n_zero_grad': n_zero_grad,
            'n_retained': n_retained,
        }
        return objective, constraints, kl, safety_mode, reduce_info

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


