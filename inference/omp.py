"""Orthogonal Matching Pursuit.

Greedily selects the column most correlated with the current residual and
re-solves least-squares on the active set each iteration.
"""
import numpy as np


def omp(W: np.ndarray, y_centered: np.ndarray, k: int) -> tuple[np.ndarray, list[int]]:
    """
    Args:
        W: (m, n) decoder (unit-norm columns).
        y_centered: (m,) measurement with b_dec subtracted.
        k: target sparsity.

    Returns:
        x_hat: (n,) recovered coefficients.
        support: ordered list of selected feature indices.
    """
    m, n = W.shape
    r = y_centered.copy()
    support: list[int] = []
    x_hat = np.zeros(n)

    for _ in range(k):
        # Raw signed correlations; matches the trained encoder's TopK
        # convention (k largest pre-activations).
        corrs = W.T @ r
        if support:
            corrs[support] = -np.inf  # don't reselect
        j = int(np.argmax(corrs))
        support.append(j)

        W_S = W[:, support]
        x_S, *_ = np.linalg.lstsq(W_S, y_centered, rcond=None)

        x_hat = np.zeros(n)
        x_hat[support] = x_S
        r = y_centered - W @ x_hat

    return x_hat, support
