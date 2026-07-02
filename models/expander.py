"""Tied-support Expander SAE.

W_dec = colnorm(M * V) where M is a frozen random d-regular bipartite mask.
W_enc = W_dec.T (tied support — encoder reads from the same d neurons that
the decoder writes to).

Unique parameters: d * n (decoder values; encoder is derived).
"""
import numpy as np
import torch
import torch.nn as nn


def _sample_mask(m: int, n: int, d: int, rng: np.random.Generator,
                 max_resamples: int = 32) -> np.ndarray:
    """Uniform d-regular bipartite mask: d ones per column, support drawn
    without replacement per column.

    Post-condition: every row has at least one nonzero. Zero rows are
    vanishingly rare when $dn/m$ is large (e.g. $\\approx 56$ at $d{=}7$,
    $m{=}2048$, $n{=}16384$) but possible at smaller scales. We first
    try patching them post-hoc by redirecting one pick from a
    non-conflicting column onto the empty row (preserving
    column-d-regularity); if that fails we re-roll the whole mask, up
    to ``max_resamples`` attempts. ``RuntimeError`` if no zero-row-free
    mask is found --- only happens when $dn$ is too close to $m$
    (i.e. the assignment problem is genuinely infeasible).
    """
    for _attempt in range(max_resamples):
        M = np.zeros((m, n), dtype=np.float32)
        for j in range(n):
            rows = rng.choice(m, size=d, replace=False)
            M[rows, j] = 1.0

        zero_rows = np.where(M.sum(axis=1) == 0)[0]
        for r in zero_rows:
            for _ in range(64):
                j = int(rng.integers(0, n))
                col_rows = np.flatnonzero(M[:, j] > 0)
                if len(col_rows) >= 2 and r not in col_rows:
                    drop = int(rng.choice(col_rows))
                    M[drop, j] = 0.0
                    M[r, j] = 1.0
                    break

        if (M.sum(axis=1) > 0).all():
            return M

    raise RuntimeError(
        f"Could not sample zero-row-free d-regular mask at "
        f"m={m}, n={n}, d={d} after {max_resamples} attempts. "
        f"Increase d, n, or check that m * d >= n / d (loose feasibility)."
    )


class ExpanderSAE(nn.Module):
    def __init__(self, m: int, n: int, d: int, k: int, seed: int = 0):
        super().__init__()
        assert 1 <= d <= m, f"d={d} out of range for m={m}"
        self.m, self.n, self.d, self.k = m, n, d, k
        self.arch = "expander_tied"

        # Mask RNG is offset from training seed so mask and init are decorrelated
        # but both reproducible from the same seed input.
        mask_rng = np.random.default_rng(seed + 10_000)
        mask_np = _sample_mask(m, n, d, mask_rng)
        self.register_buffer("mask", torch.from_numpy(mask_np))

        # Reproducibility fix: seed torch before any randn() so decoder_vals init
        # is identical across sessions for the same seed.
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
        return self.W_dec.T  # (n, m)

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        pre = (y - self.b_dec) @ self.W_enc.T + self.b_enc  # (B, n)
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

    # ---- dead-feature resampling hook ----
    def resample_feature(self, j: int, residual: torch.Tensor) -> None:
        """Re-aim dead feature j at `residual`, restricted to its mask support."""
        with torch.no_grad():
            support = self.mask[:, j].bool()
            r_masked = residual.clone()
            r_masked[~support] = 0.0
            norm = r_masked.norm().clamp(min=1e-8)
            self.decoder_vals.data[:, j] = r_masked / norm / float(np.sqrt(self.d))
            self.b_enc.data[j] = 0.0
