"""Clustered-sparse SAE control.

Same tied-support architecture as `ExpanderSAE`, but with a
*block-structured* mask: the m rows are partitioned into
G = m // d disjoint blocks of size d, and every column picks one block as
its entire support. Columns assigned to the same block therefore have
identical supports.

This is a random-sparse-mask baseline designed to match Expander's
`(m, n, d, k)` and `d·n` learned decoder values exactly — while
deliberately breaking the expansion property (multiple columns can share
a full set of d neighbors instead of having near-disjoint supports).
If an Expander SAE outperforms this baseline, that's evidence the win
comes from expansion, not just from decoder-column sparsity.

Leftover rows (when m is not divisible by d) are ignored.
"""
import numpy as np
import torch
import torch.nn as nn


def _sample_clustered_mask(m: int, n: int, d: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Block-structured mask: m rows → G disjoint blocks of size d;
    each of the n columns gets assigned one block uniformly."""
    G = m // d
    if G < 1:
        raise ValueError(f"clustered_sparse requires m >= d; got m={m}, d={d}")
    blocks = [list(range(g * d, (g + 1) * d)) for g in range(G)]
    group_idx = rng.integers(0, G, size=n)
    M = np.zeros((m, n), dtype=np.float32)
    for j in range(n):
        rows = blocks[int(group_idx[j])]
        M[rows, j] = 1.0
    return M


class ClusteredSparseSAE(nn.Module):
    """Block-structured sparse tied-support SAE."""

    def __init__(self, m: int, n: int, d: int, k: int, seed: int = 0):
        super().__init__()
        assert 1 <= d <= m, f"d={d} out of range for m={m}"
        self.m, self.n, self.d, self.k = m, n, d, k
        self.arch = "clustered_sparse"

        mask_rng = np.random.default_rng(seed + 20_000)
        mask_np = _sample_clustered_mask(m, n, d, mask_rng)
        self.register_buffer("mask", torch.from_numpy(mask_np))

        torch.manual_seed(seed)
        self.decoder_vals = nn.Parameter(torch.randn(m, n) / float(np.sqrt(d)))

        self.b_dec = nn.Parameter(torch.zeros(m))
        self.b_enc = nn.Parameter(torch.zeros(n))

    @property
    def W_dec(self) -> torch.Tensor:
        W = self.mask * self.decoder_vals
        norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return W / norms

    @property
    def W_enc(self) -> torch.Tensor:
        return self.W_dec.T

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        pre = (y - self.b_dec) @ self.W_enc.T + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec.T + self.b_dec

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(y)
        y_hat = self.decode(h)
        return y_hat, h

    def loss(self, y: torch.Tensor) -> torch.Tensor:
        y_hat, _ = self.forward(y)
        return (y - y_hat).pow(2).sum(dim=-1).mean()

    def resample_feature(self, j: int, residual: torch.Tensor) -> None:
        """Re-aim dead feature j at `residual`, restricted to its mask support."""
        with torch.no_grad():
            support = self.mask[:, j].bool()
            r_masked = residual.clone()
            r_masked[~support] = 0.0
            norm = r_masked.norm().clamp(min=1e-8)
            self.decoder_vals.data[:, j] = r_masked / norm / float(np.sqrt(self.d))
            self.b_enc.data[j] = 0.0
