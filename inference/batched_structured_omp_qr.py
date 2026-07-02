"""Batched structure-aware OMP with incremental QR refit (PyTorch).

Same algorithm as `inference/structured_omp_qr.py` but processes B
samples in parallel. The k=64 OMP iterations remain sequential
(per-iteration correlation depends on the previous iteration's
residual), but every per-iteration step is a batched op:

  * correlation:    $(B, n, d)$ gather + reduce.
  * argmax:         per-sample $\\arg\\max$ over $n$.
  * residual update: per-sample scatter.
  * MGS update of $\\mathbf{Q}, \\mathbf{R}$: batched matvecs via einsum.
  * final back-substitution: $\\mathtt{torch.linalg.solve\\_triangular}$ on
    the batch dimension.

Device-agnostic: runs on CPU or GPU. Numerically identical to the
serial CPU version to within float-precision tolerance (verified by
the test harness in `experiments/batched_omp_throughput.py`).
"""
from __future__ import annotations

import torch


def batched_structured_omp_qr(values: torch.Tensor, rows: torch.Tensor,
                              m: int, n: int, d: int,
                              Y_centered: torch.Tensor, k: int,
                              eps: float = 1e-10
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run structure-aware OMP with incremental QR on a batch of samples.

    Args:
        values:     (d*n,) flat decoder values.
        rows:       (d*n,) int row indices.
        m, n, d:    decoder shape and column degree.
        Y_centered: (B, m) measurements with $\\mathbf{b}_{\\mathrm{dec}}$ subtracted.
        k:          target sparsity.
        eps:        numerical floor for orthogonalised column norm.

    Returns:
        x_hat:   (B, n) recovered coefficients.
        support: (B, k) ordered selected feature indices per sample.
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

    for t in range(k):
        # Batched structured correlation.
        gathered = residual[:, rows_2d]                      # (B, n, d)
        corrs = (values_2d * gathered).sum(-1)               # (B, n)
        if t > 0:
            corrs.scatter_(1, support[:, :t], float("-inf"))
        j = corrs.argmax(-1)                                 # (B,)
        support[:, t] = j

        # Build the new sparse column $\mathbf{w}_j$ per sample as a
        # length-m dense vector with d nonzeros.
        sel_rows = rows_2d[j]                                # (B, d)
        sel_vals = values_2d[j]                              # (B, d)
        w_col = torch.zeros(B, m, device=device, dtype=dtype)
        w_col.scatter_(1, sel_rows, sel_vals)

        if t > 0:
            # MGS: project off the existing Q[:, :, :t].
            Q_t = Q[:, :, :t]                                # (B, m, t)
            r_new = torch.einsum("bmt,bm->bt", Q_t, w_col)   # (B, t)
            q_perp = w_col - torch.einsum("bmt,bt->bm", Q_t, r_new)
            R[:, :t, t] = r_new
        else:
            q_perp = w_col

        r_diag = torch.linalg.norm(q_perp, dim=-1)           # (B,)
        q_new = q_perp / r_diag.clamp(min=eps).unsqueeze(-1)
        Q[:, :, t] = q_new
        R[:, t, t] = r_diag

        # z_t = q_new^T y = q_new^T residual (q_new ⊥ span(Q[:, :, :t])).
        z_t = (q_new * Y_centered).sum(-1)                   # (B,)
        z[:, t] = z_t
        residual = residual - z_t.unsqueeze(-1) * q_new

    # Solve R x = z per sample. R is upper triangular.
    # solve_triangular has no bfloat16 CUDA kernel as of PyTorch 2.4, and
    # the active-set system is small (k x k), so cast to fp32 for the solve
    # and back to the storage dtype afterwards. No accuracy cost.
    x_S = torch.linalg.solve_triangular(
        R.float(), z.unsqueeze(-1).float(), upper=True
    ).squeeze(-1).to(dtype)                                  # (B, k)
    x_hat = torch.zeros(B, n, device=device, dtype=dtype)
    x_hat.scatter_(1, support, x_S)
    return x_hat, support
