"""Structure-aware Orthogonal Matching Pursuit for $d$-regular decoders.

Same algorithm as `inference/omp.py` but exploits the flat $(d{\\cdot}n,)$
storage so the correlation step $\\mathbf{W}^\\top \\mathbf{r}$ costs
$O(dn)$ instead of $O(mn)$, and the residual update costs
$O(d|\\mathrm{support}|)$ instead of $O(mn)$. The active-set lstsq refit
is unchanged.

The result is numerically identical to vanilla OMP on the dense
$\\mathbf{W}_{\\mathrm{dec}}$ recovered from $(values, rows)$.
"""
from __future__ import annotations

import numpy as np


def _flat_storage_from_dense(W_dec: np.ndarray, mask: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract column-major flat storage from a dense decoder + binary mask.

    Returns:
        values: (d*n,) float32  -- W_dec[rows[jd:(j+1)d], j] in column order.
        rows:   (d*n,) int64    -- row indices of nonzeros per column.
        d:      int             -- column degree (max nonzeros per column).
    """
    m, n = W_dec.shape
    mask_b = mask.astype(bool)
    d = int(mask_b.sum(axis=0).max())
    values = np.zeros(d * n, dtype=W_dec.dtype)
    rows = np.zeros(d * n, dtype=np.int64)
    for j in range(n):
        col_rows = np.flatnonzero(mask_b[:, j])
        # Pad with arbitrary row 0 if a column is sub-degree (shouldn't
        # happen for d-regular masks, but guards against degenerate cases).
        if col_rows.shape[0] < d:
            pad = np.zeros(d - col_rows.shape[0], dtype=col_rows.dtype)
            col_rows = np.concatenate([col_rows, pad])
        rows[j * d:(j + 1) * d] = col_rows[:d]
        values[j * d:(j + 1) * d] = W_dec[col_rows[:d], j]
    return values, rows, d


def structured_omp(values: np.ndarray, rows: np.ndarray,
                   m: int, n: int, d: int,
                   y_centered: np.ndarray, k: int
                   ) -> tuple[np.ndarray, list[int]]:
    """OMP exploiting d-regular decoder structure.

    Args:
        values:     (d*n,) flat decoder values, column-major.
        rows:       (d*n,) int row indices of nonzeros, column-major.
        m, n, d:    matrix dimensions and column degree.
        y_centered: (m,) measurement with $\\mathbf{b}_{\\mathrm{dec}}$ subtracted.
        k:          target sparsity.

    Returns:
        x_hat:   (n,) recovered coefficients.
        support: ordered list of selected feature indices.
    """
    values_2d = values.reshape(n, d)
    rows_2d = rows.reshape(n, d)

    r = y_centered.copy()
    support: list[int] = []
    x_hat = np.zeros(n, dtype=values.dtype)

    for _ in range(k):
        # Structured correlation: O(dn) gather + segment-sum.
        gathered = r[rows_2d]                    # (n, d)
        # Raw signed correlations; matches the trained encoder's TopK
        # convention (k largest pre-activations).
        corrs = (values_2d * gathered).sum(axis=1)           # (n,)
        if support:
            corrs[support] = -np.inf
        j = int(np.argmax(corrs))
        support.append(j)

        # Materialise the (m, |support|) submatrix on demand for lstsq.
        s_size = len(support)
        W_S = np.zeros((m, s_size), dtype=values.dtype)
        for s_idx, j_s in enumerate(support):
            W_S[rows_2d[j_s], s_idx] = values_2d[j_s]
        x_S, *_ = np.linalg.lstsq(W_S, y_centered, rcond=None)
        x_hat = np.zeros(n, dtype=values.dtype)
        x_hat[support] = x_S

        # Structured residual update: r = y - sum_j x[j] * W[:, j].
        r = y_centered.copy()
        for s_idx, j_s in enumerate(support):
            r[rows_2d[j_s]] -= values_2d[j_s] * x_S[s_idx]

    return x_hat, support
