"""Fused correlate + top-$k$ Triton kernel for OMP support discovery.

For each batch row, compute the structured correlation $|\\mathbf{W}^\\top
\\mathbf{r}|$ via the $(\\texttt{values}, \\texttt{rows})$ flat storage and
return the top-$k$ indices --- all in a single CUDA launch with the
intermediate $(B, n)$ correlation tensor staying in registers.

Eliminates the HBM round-trip on $\\texttt{corrs}$ and the kernel-launch
overhead between the structured-correlation kernel
(``encoder_fwd_v3``) and PyTorch's ``topk``.

One program per batch. Per-program work:
  * Read $\\mathbf{r}$ (size $m$) and the full flat $(\\texttt{values},
    \\texttt{rows})$ table (size $n d$). Both modest at our scales.
  * Compute corrs (a register-resident $(n,)$ vector).
  * Run $k$ sequential argmax-then-mask passes on corrs.

Specialised for small $d$ (the kernel keeps a $(N_{\\text{padded}},)$
register tile of corrs --- $n{=}4096$ at fp32 fits comfortably).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_corr_topk_kernel(
    residual_ptr,             # (B, M)   fp32
    values_ptr,               # (N*D,)   fp32
    rows_ptr,                 # (N*D,)   int32
    support_ptr,              # (B, K)   int32
    B, M, N, D,
    K: tl.constexpr,
    N_PADDED: tl.constexpr,
    D_PADDED: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    n_off = tl.arange(0, N_PADDED)
    n_mask = n_off < N
    d_off = tl.arange(0, D_PADDED)
    d_mask = d_off < D

    # Load (N_PADDED, D_PADDED) values + rows tile for this batch.
    flat = n_off[:, None] * D + d_off[None, :]
    flat_mask = n_mask[:, None] & d_mask[None, :]
    v = tl.load(values_ptr + flat, mask=flat_mask, other=0.0)
    r = tl.load(rows_ptr + flat, mask=flat_mask, other=0).to(tl.int32)

    # Gather residual at row indices: (N_PADDED, D_PADDED) of fp32.
    res_addr = pid * M + r
    res_at = tl.load(residual_ptr + res_addr, mask=flat_mask, other=0.0)

    # Reduce over D axis to get per-feature inner product, then abs.
    corrs = tl.sum(v * res_at, axis=1)                # (N_PADDED,)
    corrs = tl.where(n_mask, tl.abs(corrs), float("-inf"))

    # Sequential top-k: argmax + mask, K times.
    INVALID_IDX = N_PADDED + 1
    for i in tl.static_range(0, K):
        max_val = tl.max(corrs, axis=0)
        is_max = corrs == max_val
        masked_idx = tl.where(is_max, n_off, INVALID_IDX)
        idx = tl.min(masked_idx, axis=0)

        tl.store(support_ptr + pid * K + i, idx.to(tl.int32))
        corrs = tl.where(n_off == idx, float("-inf"), corrs)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def fused_corr_topk(residual: torch.Tensor,
                    values: torch.Tensor,
                    rows: torch.Tensor,
                    n: int, d: int, k: int) -> torch.Tensor:
    """Run the fused gather+correlate+abs+topk kernel.

    Args:
        residual: (B, m) fp32 contiguous.
        values:   (n*d,) fp32 contiguous.
        rows:     (n*d,) int32 contiguous.
        n, d, k:  decoder size, column degree, sparsity.

    Returns:
        support: (B, k) int32. Top-k indices per batch in descending
            magnitude order with first-occurrence tie-breaking.
    """
    assert residual.is_cuda and values.is_cuda and rows.is_cuda
    assert residual.dtype == torch.float32
    assert values.dtype == torch.float32
    assert rows.dtype == torch.int32
    B, m = residual.shape
    n_padded = _next_pow2(n)
    d_padded = max(_next_pow2(d), 8)

    support = torch.empty(B, k, dtype=torch.int32, device=residual.device)
    grid = (B,)
    _fused_corr_topk_kernel[grid](
        residual, values, rows, support,
        B, m, n, d,
        K=k, N_PADDED=n_padded, D_PADDED=d_padded,
        num_warps=4, num_stages=2,
    )
    return support
