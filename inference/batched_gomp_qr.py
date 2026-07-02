"""Batched generalised OMP (gOMP) with structure-aware QR refit.

Same as `inference/batched_structured_omp_qr.py` except each iteration
picks the top-$L$ correlated columns and adds all $L$ to the support
simultaneously (Wang, Kim & Shim, IEEE TSP 2012). For $L{=}1$ it
reduces to the original batched OMP.

Within each block of $L$ picks, modified Gram-Schmidt is applied
sequentially so that each new column is orthogonalised against both
the previous active set and the picks already orthogonalised earlier
in the block. The block's $L$ picks all come from the same correlation
vector, computed at the start of the block.

On Expander dictionaries with column degree $d$, the columns picked
by top-$L$ are nearly-disjoint when $Ld \\lesssim m$, so the block is
already well-conditioned without an explicit disjoint-support
constraint. Beyond that combinatorial threshold, expect rel-err
degradation as picked columns must share rows by pigeonhole.
"""
from __future__ import annotations

import torch


def batched_gomp_qr(values: torch.Tensor, rows: torch.Tensor,
                    m: int, n: int, d: int,
                    Y_centered: torch.Tensor, k: int, L: int = 1,
                    eps: float = 1e-10
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run gOMP with block size L on a batch of samples.

    Args:
        values, rows: flat $(d{\\cdot}n,)$ storage as in `inference/structured_omp.py`.
        m, n, d:   decoder shape and column degree.
        Y_centered: (B, m) measurements.
        k:         target sparsity.
        L:         block size; total iterations = ceil(k / L).
        eps:       numerical floor for orthogonalised column norm.

    Returns:
        x_hat:   (B, n) recovered coefficients.
        support: (B, k) selected feature indices in pick order.
    """
    B = Y_centered.shape[0]
    device = Y_centered.device
    dtype = Y_centered.dtype

    values_2d = values.view(n, d).to(dtype=dtype)
    rows_2d = rows.view(n, d).long()

    Q = torch.zeros(B, m, k, device=device, dtype=dtype)
    R = torch.zeros(B, k, k, device=device, dtype=dtype)
    z = torch.zeros(B, k, device=device, dtype=dtype)
    support = torch.full((B, k), 0, device=device, dtype=torch.long)

    residual = Y_centered.clone()
    n_picked = 0

    while n_picked < k:
        block_L = min(L, k - n_picked)

        # One correlation per block; all block picks come from this vector.
        gathered = residual[:, rows_2d]                      # (B, n, d)
        corrs = (values_2d * gathered).sum(-1)               # (B, n)
        if n_picked > 0:
            corrs.scatter_(1, support[:, :n_picked], float("-inf"))
        _, topk_idx = corrs.topk(block_L, dim=-1)            # (B, block_L)
        support[:, n_picked:n_picked + block_L] = topk_idx

        # Sequential MGS within the block. Each new column is orthogonalised
        # against both the prior active set and the picks already
        # orthogonalised earlier in this block.
        for ell in range(block_L):
            t = n_picked + ell
            j = topk_idx[:, ell]
            sel_rows = rows_2d[j]
            sel_vals = values_2d[j]
            w_col = torch.zeros(B, m, device=device, dtype=dtype)
            w_col.scatter_(1, sel_rows, sel_vals)

            if t > 0:
                Q_t = Q[:, :, :t]
                r_new = torch.einsum("bmt,bm->bt", Q_t, w_col)
                q_perp = w_col - torch.einsum("bmt,bt->bm", Q_t, r_new)
                R[:, :t, t] = r_new
            else:
                q_perp = w_col

            r_diag = torch.linalg.norm(q_perp, dim=-1)
            q_new = q_perp / r_diag.clamp(min=eps).unsqueeze(-1)
            Q[:, :, t] = q_new
            R[:, t, t] = r_diag

            z_t = (q_new * Y_centered).sum(-1)
            z[:, t] = z_t
            residual = residual - z_t.unsqueeze(-1) * q_new

        n_picked += block_L

    # solve_triangular has no bfloat16 CUDA kernel; cast to fp32 for the
    # k x k solve and back.
    x_S = torch.linalg.solve_triangular(
        R.float(), z.unsqueeze(-1).float(), upper=True
    ).squeeze(-1).to(dtype)
    x_hat = torch.zeros(B, n, device=device, dtype=dtype)
    x_hat.scatter_(1, support, x_S)
    return x_hat, support
