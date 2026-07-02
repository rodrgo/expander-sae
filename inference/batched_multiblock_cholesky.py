"""Multi-block batched OMP via per-block Cholesky refit.

At each outer iteration: pick the top-$L = \\lfloor m/d \\rfloor$ columns
by current correlation, append to the running support, and refit
*all* active coefficients via a Cholesky-on-normal-equations solve.
The residual is then the orthogonal projection of $\\mathbf{y}$ onto the
span of the inactive columns, ready for the next pick.

Outer iterations: $\\lceil k / L \\rceil$. When $k \\le m/d$ this collapses
to a single block, recovering ``inference/batched_oneshot_lstsq.py``.

Per-block work is dominated by two cuBLAS bmms ($\\mathbf{W}_S^\\top
\\mathbf{W}_S$ and the residual update) plus a small fp32 Cholesky on
the $(t L) \\times (t L)$ Gram matrix at iteration $t$. All steps run as
fused batched kernels --- no per-pick MGS loop --- so per-iteration
launch overhead is independent of $L$.
"""
from __future__ import annotations

import torch


def batched_multiblock_cholesky(values: torch.Tensor, rows: torch.Tensor,
                                m: int, n: int, d: int,
                                Y_centered: torch.Tensor, k: int,
                                L: int | None = None
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-block top-$L$ pick + per-block Cholesky refit.

    Args:
        values, rows: flat $(d \\cdot n,)$ storage.
        m, n, d:   decoder shape and column degree.
        Y_centered: (B, m) measurements.
        k:         target sparsity.
        L:         block size; defaults to $\\lfloor m/d \\rfloor$ (capped at 1).

    Returns:
        x_hat:   (B, n) recovered coefficients.
        support: (B, k) selected feature indices in pick order.
    """
    if L is None:
        L = max(1, m // d)

    B = Y_centered.shape[0]
    device = Y_centered.device
    dtype = Y_centered.dtype

    values_2d = values.view(n, d).to(dtype=dtype)
    rows_2d = rows.view(n, d).long()

    support = torch.full((B, k), 0, device=device, dtype=torch.long)
    n_picked = 0
    residual = Y_centered.clone()
    x_S = torch.zeros(B, k, device=device, dtype=dtype)

    # Pre-cast for Triton kernels (used on CUDA at small $d$). The
    # encoder kernel and structured-Gram kernel both need fp32 inputs;
    # values and rows don't change between blocks so we cast once.
    use_triton_corr = (device.type == "cuda") and (d <= 64)
    use_triton_gram = (device.type == "cuda") and (d <= 8)
    if use_triton_corr:
        from kernels.triton.encoder_fwd_v3 import encoder_forward_v3
        values_fp32_flat = values.view(-1).to(torch.float32).contiguous()
        rows_int32_flat = rows.view(-1).to(torch.int32).contiguous()
        values_2d_fp32 = values_fp32_flat.view(n, d)
        rows_2d_int32 = rows_int32_flat.view(n, d)
        b_zero = torch.zeros(n, device=device, dtype=torch.float32)
        # Initial signed correlations against $\\mathbf{y}$ --- reused
        # across all blocks via a gather to build $\\mathbf{W}_S^\\top
        # \\mathbf{y}$ without a bmm.
        Y_fp32 = Y_centered.to(torch.float32).contiguous()
        initial_signed = encoder_forward_v3(
            Y_fp32, values_fp32_flat, rows_int32_flat, b_zero,
            n=n, d=d,
        )                                                   # (B, n) fp32
    if use_triton_gram:
        from kernels.triton.structured_gram_v2 import structured_gram_v2

    while n_picked < k:
        block_L = min(L, k - n_picked)

        # 1. Top-block_L picks against current residual. We use raw
        # signed correlations (matching the trained encoder's TopK
        # convention --- k largest signed pre-activations). On the
        # first iteration we already have the initial signed correlations
        # against y (== residual at iter 0), so we reuse them.
        if use_triton_corr:
            if n_picked == 0:
                corrs = initial_signed
            else:
                corrs = encoder_forward_v3(
                    residual.to(torch.float32).contiguous(),
                    values_fp32_flat, rows_int32_flat, b_zero,
                    n=n, d=d,
                )
        else:
            gathered = residual[:, rows_2d]
            corrs = (values_2d * gathered).sum(-1)
        if n_picked > 0:
            # ``corrs`` is a fresh tensor from encoder_forward_v3 here
            # (n_picked > 0 only on iters >= 2), safe to scatter in-place.
            corrs.scatter_(1, support[:, :n_picked], float("-inf"))
        _, new_pick = corrs.topk(block_L, dim=-1)
        support[:, n_picked:n_picked + block_L] = new_pick
        n_picked += block_L

        active = support[:, :n_picked]                    # (B, n_picked)

        # 2. Build A and rhs. On CUDA at small d we use the structured-
        # Gram Triton kernel (no W_S materialisation) and gather rhs
        # directly from the cached initial signed correlations.
        if use_triton_gram and use_triton_corr:
            active_int32 = active.to(torch.int32)
            A = structured_gram_v2(values_2d_fp32, rows_2d_int32,
                                   active_int32, d=d)      # (B, n_picked, n_picked)
            rhs = torch.gather(initial_signed, 1, active)  # (B, n_picked) fp32
            W_S = None  # not built; residual update needs fallback
        else:
            sel_rows = rows_2d[active].transpose(1, 2).contiguous()
            sel_vals = values_2d[active].transpose(1, 2).contiguous()
            W_S = torch.zeros(B, m, n_picked, device=device, dtype=dtype)
            W_S.scatter_(1, sel_rows, sel_vals)
            W_S_T = W_S.transpose(1, 2)
            A = torch.bmm(W_S_T, W_S).float()
            rhs = torch.bmm(
                W_S_T, Y_centered.unsqueeze(-1)
            ).squeeze(-1).float()

        # 3. Cholesky on normal equations.
        L_chol = torch.linalg.cholesky(A)
        x_active = torch.cholesky_solve(
            rhs.unsqueeze(-1), L_chol
        ).squeeze(-1).to(dtype)
        x_S[:, :n_picked] = x_active

        # 4. Residual update --- skip on the final block.
        if n_picked < k:
            if W_S is None:
                # Build a temporary W_S only for the residual bmm. Cheaper
                # than the structured atomic-add path at d=7 because the
                # bmm is bandwidth-bound and the W_S build is lightweight.
                sel_rows = rows_2d[active].transpose(1, 2).contiguous()
                sel_vals = values_2d[active].transpose(1, 2).contiguous()
                W_S = torch.zeros(B, m, n_picked, device=device, dtype=dtype)
                W_S.scatter_(1, sel_rows, sel_vals)
            residual = Y_centered - torch.bmm(
                W_S, x_active.unsqueeze(-1)
            ).squeeze(-1)

    x_hat = torch.zeros(B, n, device=device, dtype=dtype)
    x_hat.scatter_(1, support, x_S)
    return x_hat, support
