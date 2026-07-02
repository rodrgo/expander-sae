"""One-shot batched OMP via lstsq refit, no iteration.

Pick the top-$k$ columns by $|\\mathbf{W}^\\top \\mathbf{y}|$ in a single
correlation pass, build the active-set matrix $\\mathbf{W}_S$ explicitly,
and refit $\\mathbf{x}_S$ with one batched least-squares solve.

Algorithmically equivalent to gOMP at $L{=}k$ (single block), but
implemented as ~3 kernel launches instead of $k$ sequential
modified-Gram-Schmidt iterations. Same picks, same rel-err — only the
refit kernel differs.

In the regime $k \\le m/d$ (e.g. Expander $d{=}7$ with $k{=}64$, where
$m/d \\approx 73$) the picked columns are nearly row-disjoint by
pigeonhole, so $\\mathbf{W}_S^\\top \\mathbf{W}_S$ is well-conditioned and
the lstsq is stable. Outside that regime conditioning is not guaranteed
and a multi-block variant is needed.
"""
from __future__ import annotations

import torch


def batched_oneshot_lstsq(values: torch.Tensor, rows: torch.Tensor,
                          m: int, n: int, d: int,
                          Y_centered: torch.Tensor, k: int
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-block top-$k$ pick + batched lstsq refit.

    Args:
        values, rows: flat $(d \\cdot n,)$ storage as in inference/structured_omp.py.
        m, n, d:   decoder shape and column degree.
        Y_centered: (B, m) measurements.
        k:         target sparsity.

    Returns:
        x_hat:   (B, n) recovered coefficients.
        support: (B, k) selected feature indices, sorted by descending |corr|.
    """
    B = Y_centered.shape[0]
    device = Y_centered.device
    dtype = Y_centered.dtype

    values_2d = values.view(n, d).to(dtype=dtype)
    rows_2d = rows.view(n, d).long()

    # 1. Signed correlations $c_j = \\langle \\mathbf{W}_j, \\mathbf{y}
    # \\rangle$ via the structured encoder-forward Triton kernel. We keep
    # the signs because they are exactly the right-hand side $\\mathbf{W}_S^\\top
    # \\mathbf{y}$ that the Cholesky solve needs --- gathered at the
    # picked support, no bmm required.
    if device.type == "cuda" and d <= 64:
        from kernels.triton.encoder_fwd_v3 import encoder_forward_v3
        values_fp32 = values.view(-1).to(torch.float32).contiguous()
        rows_int32 = rows.view(-1).to(torch.int32).contiguous()
        Y_fp32 = Y_centered.to(torch.float32).contiguous()
        b_zero = torch.zeros(n, device=device, dtype=torch.float32)
        signed_corrs = encoder_forward_v3(Y_fp32, values_fp32, rows_int32,
                                          b_zero, n=n, d=d)          # (B, n) fp32
    else:
        gathered = Y_centered[:, rows_2d]                          # (B, n, d)
        signed_corrs = (values_2d * gathered).sum(-1)              # (B, n)

    # 2. Top-k support by raw signed correlation, matching the trained
    # encoder's TopK convention (k largest pre-activations, non-negative
    # output). This makes the OMP support strictly comparable to the
    # encoder's pick rule.
    _, support = signed_corrs.topk(k, dim=-1)                      # (B, k)

    # 3. Right-hand side $\\mathbf{W}_S^\\top \\mathbf{y}$ is just the
    # signed correlations gathered at the picked support --- one GPU
    # gather, no bmm. Force fp32 since cholesky_solve below requires it.
    rhs = torch.gather(signed_corrs, 1, support).float()           # (B, k)

    # 4. Gram matrix $\\mathbf{A} = \\mathbf{W}_S^\\top \\mathbf{W}_S$ via
    # the structured-Gram Triton kernel: exploits the $d$-regular column
    # sparsity to skip the $(B, m, k)$ dense W_S materialisation and the
    # bmm. Only viable at small $d$ ($d{\\le}8$ tested); beyond that the
    # unrolled $d^2$ inner loop is prohibitive and dense bmm wins.
    if device.type == "cuda" and d <= 8:
        from kernels.triton.structured_gram_v2 import structured_gram_v2
        if not values_fp32.is_contiguous():
            values_fp32 = values_fp32.contiguous()
        values_2d_fp32 = values_fp32.view(n, d)
        rows_2d_int32 = rows_int32.view(n, d)
        support_int32 = support.to(torch.int32)
        A = structured_gram_v2(values_2d_fp32, rows_2d_int32,
                               support_int32, d=d)                  # (B, k, k) fp32
    else:
        # Fallback: build dense W_S and bmm (the prior implementation).
        sel_rows = rows_2d[support].transpose(1, 2).contiguous()
        sel_vals = values_2d[support].transpose(1, 2).contiguous()
        W_S = torch.zeros(B, m, k, device=device, dtype=dtype)
        W_S.scatter_(1, sel_rows, sel_vals)
        A = torch.bmm(W_S.transpose(1, 2), W_S).float()

    # 5. Refit via Cholesky on normal equations.
    L = torch.linalg.cholesky(A)
    x_S = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1).to(dtype)

    # 5. Scatter into output.
    x_hat = torch.zeros(B, n, device=device, dtype=dtype)
    x_hat.scatter_(1, support, x_S)
    return x_hat, support
