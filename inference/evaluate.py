"""Evaluation drivers: trained-encoder forward pass and iterative CS algorithms.

Returns per-sample numpy arrays of reconstruction errors and sparse support
matrices — all the raw data the tables and figures need.
"""
import time
import numpy as np
import scipy.sparse
import torch

from .omp import omp
from .niht import niht
from .cosamp import cosamp


METHODS = {
    "omp": omp,
    "niht": niht,
    "cosamp": cosamp,
}


def evaluate_encoder(model, test_acts_tensor: torch.Tensor
                     ) -> tuple[np.ndarray, scipy.sparse.csr_matrix]:
    """Run the trained encoder on the full test set.

    Returns:
        per_sample_relerr: (N,) float array.
        supports: (N, n) boolean CSR matrix.
    """
    model = model.eval()
    with torch.no_grad():
        y_hat, h = model(test_acts_tensor)
        per_err = (torch.norm(test_acts_tensor - y_hat, dim=-1) /
                   torch.norm(test_acts_tensor, dim=-1).clamp(min=1e-12))
        supports = (h.abs() > 1e-10).cpu().numpy()
    return per_err.cpu().numpy(), scipy.sparse.csr_matrix(supports)


def evaluate_iterative(model, test_acts_np: np.ndarray, k: int,
                       method_name: str, n_samples: int
                       ) -> tuple[np.ndarray, scipy.sparse.csr_matrix, float]:
    """Run one iterative CS algorithm on `n_samples` test activations.

    Returns:
        per_sample_relerr: (n_samples,).
        supports: (n_samples, n) boolean CSR matrix.
        ms_per_sample: mean wall-clock per sample.
    """
    if method_name not in METHODS:
        raise ValueError(f"Unknown method {method_name}; choose from {list(METHODS)}")
    method = METHODS[method_name]

    W_np = model.W_dec.detach().cpu().numpy()
    b_np = model.b_dec.detach().cpu().numpy()
    n = W_np.shape[1]

    errs = np.zeros(n_samples, dtype=np.float32)
    supports = np.zeros((n_samples, n), dtype=bool)

    test = test_acts_np[:n_samples]
    t0 = time.time()
    for i in range(n_samples):
        y_centered = test[i] - b_np
        x_hat, supp = method(W_np, y_centered, k)
        recon = W_np @ x_hat + b_np
        errs[i] = float(np.linalg.norm(test[i] - recon) /
                        (np.linalg.norm(test[i]) + 1e-12))
        if supp:
            supports[i, supp] = True
    elapsed = time.time() - t0

    ms_per_sample = (elapsed / n_samples) * 1000.0
    return errs, scipy.sparse.csr_matrix(supports), ms_per_sample
