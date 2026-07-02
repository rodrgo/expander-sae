"""Tiled encoder forward Triton kernel (v3).

Same math as v1 (`encoder_fwd.py`) but tiled over both the output-feature
axis (BLOCK_J) and the batch axis (BLOCK_B), so each program block does
BLOCK_B * BLOCK_J * D FMAs instead of BLOCK_J * D. Includes
`@triton.autotune` over BLOCK_B, BLOCK_J, num_warps, num_stages.

This file is the simple "fused-multiply-add reduction" variant. A
Tensor-Core variant using `tl.dot` is added later (Step 5 of the
optimisation plan) if this one doesn't hit the wall-clock target.

Shape contract identical to v1's ``encoder_forward``:

    pre[b, j] = sum_i values[j, i] * tilde_h[b, rows[j, i]] + b_enc[j]

Inputs / outputs are the same:
    tilde_h : (B, m)  fp32, contiguous
    values  : (n*d,)  fp32, contiguous   (column-major: values[j, i])
    rows    : (n*d,)  int32, contiguous
    b_enc   : (n,)    fp32
    out     : (B, n)  fp32 (allocated by the wrapper)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------
# Autotune config sweep. Keys are (M, N, D) so each (m, n, d) shape
# compiles its own optimum once and caches afterwards.
# ------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_B": bb, "BLOCK_J": bj}, num_warps=w, num_stages=s)
    for bb in (8, 16, 32)
    for bj in (32, 64, 128)
    for w in (4, 8)
    for s in (2, 3)
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "D"])
@triton.jit
def _encoder_fwd_v3_kernel(
    tilde_h_ptr,              # (B, m)   fp32
    values_ptr,               # (n*d,)   fp32
    rows_ptr,                 # (n*d,)   int32
    b_enc_ptr,                # (n,)     fp32
    out_ptr,                  # (B, n)   fp32
    B, M, N,
    stride_h_b, stride_h_m,
    stride_o_b, stride_o_n,
    D: tl.constexpr,
    D_PADDED: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)

    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)            # (BLOCK_B,)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)            # (BLOCK_J,)
    b_mask = b_offsets < B
    j_mask = j_offsets < N

    # Load (BLOCK_J, D_PADDED) values and rows tiles. D_PADDED is a
    # power-of-2 ≥ D so masked loads zero out the padding.
    i_offsets = tl.arange(0, D_PADDED)                             # (D_PADDED,)
    i_mask = i_offsets < D                                          # (D_PADDED,)
    flat = j_offsets[:, None] * D + i_offsets[None, :]             # (BLOCK_J, D_PADDED)
    flat_mask = j_mask[:, None] & i_mask[None, :]
    v_tile = tl.load(values_ptr + flat, mask=flat_mask, other=0.0)
    r_tile = tl.load(rows_ptr + flat, mask=flat_mask, other=0)     # int32

    # Gather (BLOCK_B, BLOCK_J, D_PADDED) of tilde_h via row indices.
    # tilde_h_at[b, j, i] = tilde_h[b, rows[j, i]]
    h_indices = (
        b_offsets[:, None, None] * stride_h_b
        + r_tile[None, :, :] * stride_h_m
    )                                                              # (BLOCK_B, BLOCK_J, D_PADDED)
    full_mask = b_mask[:, None, None] & flat_mask[None, :, :]
    h_at = tl.load(tilde_h_ptr + h_indices, mask=full_mask, other=0.0)

    # Reduce over D axis: pre[b, j] = sum_i values[j, i] * tilde_h[b, rows[j, i]]
    pre = tl.sum(h_at * v_tile[None, :, :], axis=2)                # (BLOCK_B, BLOCK_J)

    # Add bias.
    bias = tl.load(b_enc_ptr + j_offsets, mask=j_mask, other=0.0)
    pre = pre + bias[None, :]

    # Store.
    out_offsets = (
        b_offsets[:, None] * stride_o_b
        + j_offsets[None, :] * stride_o_n
    )
    tl.store(out_ptr + out_offsets, pre,
             mask=b_mask[:, None] & j_mask[None, :])


# ------------------------------------------------------------------
# Smallest power-of-2 ≥ D (compile-time constant, picked at launch time).
# ------------------------------------------------------------------
def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def encoder_forward_v3(tilde_h: torch.Tensor,
                       values: torch.Tensor,
                       rows: torch.Tensor,
                       b_enc: torch.Tensor,
                       n: int, d: int) -> torch.Tensor:
    """Run the v3 (tiled) encoder forward kernel.

    Returns preact of shape (B, n) fp32.
    """
    if tilde_h.dim() == 1:
        tilde_h = tilde_h.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    assert tilde_h.is_cuda and values.is_cuda and rows.is_cuda and b_enc.is_cuda
    assert tilde_h.dtype == torch.float32
    assert values.dtype == torch.float32
    assert rows.dtype == torch.int32
    assert b_enc.dtype == torch.float32
    assert values.numel() == d * n
    assert rows.numel() == d * n
    B, M = tilde_h.shape

    out = torch.empty(B, n, dtype=torch.float32, device=tilde_h.device)
    d_padded = max(_next_pow2(d), 8)  # Triton needs power-of-2 for tile shapes

    grid = lambda meta: (
        triton.cdiv(B, meta["BLOCK_B"]),
        triton.cdiv(n, meta["BLOCK_J"]),
    )
    _encoder_fwd_v3_kernel[grid](
        tilde_h, values, rows, b_enc, out,
        B, M, n,
        tilde_h.stride(0), tilde_h.stride(1),
        out.stride(0), out.stride(1),
        D=d, D_PADDED=d_padded,
    )
    return out.squeeze(0) if squeeze else out
