"""Compressive Sampling Matching Pursuit.

Per iteration:
  1. Proxy:   pick 2k columns most correlated with the current residual.
  2. Merge:   union with the current support.
  3. Solve:   least-squares on the merged support.
  4. Prune:   keep top-k.
"""
import numpy as np


def cosamp(W: np.ndarray, y_centered: np.ndarray, k: int,
           max_iter: int = 50, tol: float = 1e-6) -> tuple[np.ndarray, list[int]]:
    m, n = W.shape
    x = np.zeros(n)
    support: set[int] = set()

    for _ in range(max_iter):
        r = y_centered - W @ x

        corrs = np.abs(W.T @ r)
        proxy = set(int(i) for i in np.argsort(corrs)[-2 * k:].tolist())

        merged = sorted(support | proxy)
        if not merged:
            break

        W_merged = W[:, merged]
        x_merged, *_ = np.linalg.lstsq(W_merged, y_centered, rcond=None)

        x_full = np.zeros(n)
        x_full[merged] = x_merged
        topk_idx = np.argsort(np.abs(x_full))[-k:]

        x_new = np.zeros(n)
        x_new[topk_idx] = x_full[topk_idx]
        new_support = set(int(i) for i in topk_idx.tolist())

        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12) < tol:
            x = x_new
            support = new_support
            break
        x = x_new
        support = new_support

    return x, sorted(support)
