"""Pruned-retuned dense SAE baseline.

Recipe:
  1. Load a trained `dense_tied` SAE at matching (m, n, k, seed).
  2. For each decoder column j, keep the d rows with largest |w_j|.
  3. Freeze this mask.
  4. Initialize a sparse tied-support decoder from the retained weights,
     column-normalized.
  5. Fine-tune only the retained sparse decoder values (same optimizer as
     Expander, 5000 steps by default).

This tests whether a mask extracted from an *already-trained* dense SAE
carries better support geometry than a random d-regular expander mask.

At inference time, if the source dense checkpoint is missing we fall
back to zero placeholders — the caller is expected to `load_state_dict`
from a saved pruned_retuned_dense checkpoint, which will overwrite
mask/decoder_vals/biases from the stored state.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Enable late import of db.model_path without import cycles.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _extract_topd_mask(W_dec: np.ndarray, d: int) -> np.ndarray:
    """For each column of W_dec (shape m, n), keep the d rows with largest
    absolute value. Returns (m, n) float32 mask."""
    m, n = W_dec.shape
    abs_W = np.abs(W_dec)
    # argpartition: the last d entries after -abs sort are the top-d abs values
    top_rows = np.argpartition(-abs_W, d - 1, axis=0)[:d]  # (d, n)
    mask = np.zeros((m, n), dtype=np.float32)
    mask[top_rows, np.arange(n)[None, :].repeat(d, axis=0)] = 1.0
    return mask


def _default_source_path(m: int, n: int, k: int, seed: int) -> str:
    """Look up the dense_tied encoder's model_path from the DB first (the
    existing Modal-trained checkpoints have `_encoder` in the filename);
    fall back to `db.model_path(...)` formatting if the DB entry is missing."""
    try:
        from db import load_db, get_entry
        e = get_entry(load_db(), "dense_tied", m, n, m, k, seed, "encoder")
        if e and e.get("model_path"):
            return e["model_path"]
    except Exception:
        pass
    from db import model_path
    return model_path("dense_tied", m, n, m, k, seed)


class PrunedRetunedDenseSAE(nn.Module):
    def __init__(self, m: int, n: int, d: int, k: int, seed: int = 0,
                 source_dense_path: str | None = None):
        super().__init__()
        assert 1 <= d <= m, f"d={d} out of range for m={m}"
        self.m, self.n, self.d, self.k = m, n, d, k
        self.arch = "pruned_retuned_dense"

        src = source_dense_path or _default_source_path(m, n, k, seed)
        if os.path.exists(src):
            sd = torch.load(src, map_location="cpu", weights_only=True)
            W_dense = sd["W_dec_param"].numpy()  # (m, n)
            b_dec_src = sd["b_dec"].clone()
            b_enc_src = sd["b_enc"].clone()

            mask_np = _extract_topd_mask(W_dense, d)
            init_np = (W_dense * mask_np).astype(np.float32)

            self.register_buffer("mask", torch.from_numpy(mask_np))
            self.decoder_vals = nn.Parameter(torch.from_numpy(init_np))
            self.b_dec = nn.Parameter(b_dec_src)
            self.b_enc = nn.Parameter(b_enc_src)
        else:
            # Placeholder — caller is expected to load_state_dict.
            self.register_buffer("mask", torch.zeros(m, n))
            self.decoder_vals = nn.Parameter(torch.zeros(m, n))
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
