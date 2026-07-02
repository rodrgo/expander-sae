"""Normalised Iterative Hard Thresholding.

x_{t+1} = H_k(x_t + mu_t * W^T r_t),
where mu_t = ||W^T r||^2 / ||W W^T r||^2 is the normalised step size
(steepest descent on the quadratic).
"""
import numpy as np


def niht(W: np.ndarray, y_centered: np.ndarray, k: int,
         max_iter: int = 50, tol: float = 1e-6) -> tuple[np.ndarray, list[int]]:
    m, n = W.shape
    x = np.zeros(n)

    for _ in range(max_iter):
        r = y_centered - W @ x
        grad = W.T @ r

        Wgrad = W @ grad
        denom = float(np.dot(Wgrad, Wgrad))
        num = float(np.dot(grad, grad))
        mu = num / (denom + 1e-12) if denom > 1e-12 else 1.0

        x_new = x + mu * grad
        topk_idx = np.argsort(np.abs(x_new))[-k:]
        x_thresh = np.zeros(n)
        x_thresh[topk_idx] = x_new[topk_idx]

        if np.linalg.norm(x_thresh - x) / (np.linalg.norm(x) + 1e-12) < tol:
            x = x_thresh
            break
        x = x_thresh

    support = sorted(int(i) for i in np.where(np.abs(x) > 1e-10)[0])
    return x, support
