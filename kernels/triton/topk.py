"""Top-$k$ Triton kernel for OMP support discovery.

For each batch row of an $(B, n)$ correlation tensor, return the indices
of the top-$k$ entries in descending magnitude order.

We do $k$ sequential argmax-then-mask passes inside a single kernel:
each pass is one $O(n)$ register-reduction. Total $O(B \\cdot k \\cdot n)$
work, but it all happens in registers within a single program per batch
--- no per-pass kernel launch, no shared-memory shuffles, no auxiliary
buffers. PyTorch's ``topk`` for the same shape goes through a
quickselect-style routine that has more launch overhead than is
warranted at $k{=}64$.

The "first-occurrence argmax" handles ties by picking the smallest
index, matching numpy/PyTorch convention.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_kernel(
    corrs_ptr,                # (B, N) fp32
    support_ptr,              # (B, K) int32 (output)
    B, N,
    K: tl.constexpr,
    N_PADDED: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    n_off = tl.arange(0, N_PADDED)
    n_mask = n_off < N

    c = tl.load(
        corrs_ptr + pid * N + n_off,
        mask=n_mask,
        other=float("-inf"),
    )

    INVALID_IDX = N_PADDED + 1
    for i in tl.static_range(0, K):
        # First-occurrence argmax: smallest index where c equals its max.
        max_val = tl.max(c, axis=0)
        is_max = c == max_val
        masked_idx = tl.where(is_max, n_off, INVALID_IDX)
        idx = tl.min(masked_idx, axis=0)

        tl.store(support_ptr + pid * K + i, idx.to(tl.int32))

        # Mask out the picked position so the next iteration sees the next-best.
        c = tl.where(n_off == idx, float("-inf"), c)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def topk(corrs: torch.Tensor, k: int) -> torch.Tensor:
    """Return the indices of the top-$k$ entries of each row of ``corrs``.

    Args:
        corrs: (B, n) fp32 contiguous.
        k:     number of top elements to return per row.

    Returns:
        support: (B, k) int32. Indices in descending magnitude order, with
            first-occurrence tie-breaking.
    """
    assert corrs.is_cuda and corrs.dtype == torch.float32
    B, n = corrs.shape
    n_padded = _next_pow2(n)

    support = torch.empty(B, k, dtype=torch.int32, device=corrs.device)
    grid = (B,)
    _topk_kernel[grid](
        corrs, support,
        B, n, K=k, N_PADDED=n_padded,
        num_warps=4, num_stages=2,
    )
    return support
