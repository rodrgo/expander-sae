"""Structured Gram matrix Triton kernel (v2): loop-accumulator design.

Same math as ``structured_gram.py`` --- compute
$\\mathbf{A} = \\mathbf{W}_S^\\top \\mathbf{W}_S$ exploiting the $d$-regular
column structure --- but with a fundamentally different register-pressure
profile.

The v1 kernel materialised a $(\\text{BLOCK\\_K1}, \\text{BLOCK\\_K2},
D_{\\text{padded}}, D_{\\text{padded}})$ 4-D register tensor for the
all-pairs comparison, which forced tiny $\\text{BLOCK\\_K1}{=}8$ tiles
(to fit registers) and an explosion of programs ($\\sim\\!65$k at our
scale). Each program did only $\\sim\\!4$k ops --- launch-overhead
bound.

v2 instead nests two static-range loops over $(p, q) \\in [0, d)^2$ and
accumulates into just a $(\\text{BLOCK\\_K1}, \\text{BLOCK\\_K2})$
register tile. This lets us push $\\text{BLOCK\\_K1}{=}\\text{BLOCK\\_K2}{=}32$,
shrinking the program grid by $\\sim\\!16\\times$ and amortising launch
overhead. Same total FLOPs, dramatically better hardware utilisation.

Specialised for small $d$ (compile-time-unrolled $d^2$ loop). Works
through $d \\le 16$; beyond that the unrolled loop body becomes
prohibitive.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _structured_gram_v2_kernel(
    values_2d_ptr,            # (n, d) fp32
    rows_2d_ptr,              # (n, d) int32
    support_ptr,              # (B, k) int32
    A_ptr,                    # (B, k, k) fp32
    B, K, N, D,
    stride_a_b, stride_a_t1, stride_a_t2,
    D_CONST: tl.constexpr,
    BLOCK_K1: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t1 = tl.program_id(1)
    pid_t2 = tl.program_id(2)

    t1_offs = pid_t1 * BLOCK_K1 + tl.arange(0, BLOCK_K1)
    t2_offs = pid_t2 * BLOCK_K2 + tl.arange(0, BLOCK_K2)
    t1_mask = t1_offs < K
    t2_mask = t2_offs < K

    s1 = tl.load(support_ptr + pid_b * K + t1_offs,
                 mask=t1_mask, other=0)
    s2 = tl.load(support_ptr + pid_b * K + t2_offs,
                 mask=t2_mask, other=0)

    A_tile = tl.zeros((BLOCK_K1, BLOCK_K2), dtype=tl.float32)

    # Nested compile-time-unrolled (p, q) loops. Each iteration loads
    # only $(BLOCK\\_K1,)$ and $(BLOCK\\_K2,)$ slices of v/r, computes a
    # 2-D match mask, and accumulates one rank-1 contribution into
    # A_tile. No 4-D tensor ever materialised.
    for p in tl.static_range(D_CONST):
        addr1_p = s1 * D + p
        r1_p = tl.load(rows_2d_ptr + addr1_p,
                       mask=t1_mask, other=-1).to(tl.int32)
        v1_p = tl.load(values_2d_ptr + addr1_p,
                       mask=t1_mask, other=0.0)
        for q in tl.static_range(D_CONST):
            addr2_q = s2 * D + q
            r2_q = tl.load(rows_2d_ptr + addr2_q,
                           mask=t2_mask, other=-2).to(tl.int32)
            v2_q = tl.load(values_2d_ptr + addr2_q,
                           mask=t2_mask, other=0.0)

            match = (r1_p[:, None] == r2_q[None, :]).to(tl.float32)
            A_tile += match * v1_p[:, None] * v2_q[None, :]

    a_addr = (
        pid_b * stride_a_b
        + t1_offs[:, None] * stride_a_t1
        + t2_offs[None, :] * stride_a_t2
    )
    tl.store(A_ptr + a_addr, A_tile,
             mask=t1_mask[:, None] & t2_mask[None, :])


def structured_gram_v2(values_2d: torch.Tensor,
                       rows_2d: torch.Tensor,
                       support: torch.Tensor,
                       d: int,
                       block_k1: int = 32,
                       block_k2: int = 32) -> torch.Tensor:
    """Compute the Gram matrix $A = W_S^T W_S$ via the loop-accumulator path.

    Args:
        values_2d: (n, d) fp32 contiguous.
        rows_2d:   (n, d) int32 contiguous.
        support:   (B, k) int32 (or int64 -> cast).
        d:         column degree, compile-time constant.
        block_k1, block_k2: tile sizes (default 32, 32).

    Returns:
        A: (B, k, k) fp32.
    """
    assert values_2d.is_cuda and rows_2d.is_cuda and support.is_cuda
    assert values_2d.dtype == torch.float32
    assert rows_2d.dtype == torch.int32
    n, d_actual = values_2d.shape
    assert d_actual == d
    if support.dtype != torch.int32:
        support = support.to(torch.int32)
    B, K = support.shape

    A = torch.empty(B, K, K, dtype=torch.float32, device=values_2d.device)
    grid = (B, triton.cdiv(K, block_k1), triton.cdiv(K, block_k2))
    _structured_gram_v2_kernel[grid](
        values_2d, rows_2d, support, A,
        B, K, n, d,
        A.stride(0), A.stride(1), A.stride(2),
        D_CONST=d,
        BLOCK_K1=block_k1, BLOCK_K2=block_k2,
        num_warps=4, num_stages=2,
    )
    return A
