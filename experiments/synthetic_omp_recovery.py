"""Synthetic OMP recovery on d-regular Expander sensing matrices.

Classical CS sanity study: fix m, n; sweep (d, k_synth, mask_seed, sigma);
for each config draw N_TRIALS k-sparse Gaussian signals x, measure
h = Wx + sigma·noise, run OMP for k_synth iterations, record exact
support recovery + code error + reconstruction error.

Writes to `results/synthetic_db.json` (disjoint from `benchmark_db.json`).

CLI:
    python experiments/synthetic_omp_recovery.py
    python experiments/synthetic_omp_recovery.py --d 7 30 --sigma 0 0.01
    python experiments/synthetic_omp_recovery.py --n-trials 50 --mask-seeds 0
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import load_db, save_db, upsert_safe, make_id

SYNTH_DB_PATH = "results/synthetic_db.json"
LOG_PATH = "results/synthetic_omp_recovery.log"
RAW_DIR = Path("results/raw_synthetic")


def _log(msg: str) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().isoformat()}  {msg}\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Decoder generation
# ---------------------------------------------------------------------------
def _sample_expander_mask(m: int, n: int, d: int,
                          rng: np.random.Generator) -> np.ndarray:
    """d-regular bipartite mask: d ones per column, support drawn without
    replacement per column. Same recipe as models/expander.py."""
    M = np.zeros((m, n), dtype=np.float32)
    for j in range(n):
        rows = rng.choice(m, size=d, replace=False)
        M[rows, j] = 1.0
    return M


def _build_decoder(mask: np.ndarray, d: int,
                   rng: np.random.Generator) -> np.ndarray:
    """W_{ij} = ±1/√d on the mask support, 0 elsewhere."""
    W = mask.astype(np.float32) / np.sqrt(d)
    signs = rng.choice([-1.0, 1.0], size=W.shape).astype(np.float32)
    W = W * signs * mask.astype(np.float32)
    return W


# ---------------------------------------------------------------------------
# Batched OMP (all trials advance together; only lstsq is per-trial)
# ---------------------------------------------------------------------------
def batched_omp(W: np.ndarray, Y: np.ndarray, k: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Batched OMP on T trials simultaneously.

    Args:
        W: (m, n) decoder.
        Y: (m, T) measurements, one column per trial.
        k: target sparsity per trial.

    Returns:
        supports: (T, k) int array — support indices per trial, in selection order.
        X_hat:    (n, T) float array — recovered coefficients per trial.
    """
    m, n = W.shape
    T = Y.shape[1]
    R = Y.copy()
    supports = np.full((T, k), -1, dtype=np.int32)
    X_hat = np.zeros((n, T), dtype=np.float32)

    for step in range(k):
        corrs = np.abs(W.T @ R)  # (n, T) — batched BLAS
        # Mask already-selected
        for t in range(T):
            prev = supports[t, :step]
            corrs[prev, t] = -1.0
        js = corrs.argmax(axis=0)  # (T,)
        supports[:, step] = js

        # Per-trial lstsq refit on current support
        for t in range(T):
            S = supports[t, :step + 1]
            W_S = W[:, S]
            x_S, *_ = np.linalg.lstsq(W_S, Y[:, t], rcond=None)
            X_hat[:, t] = 0.0
            X_hat[S, t] = x_S
            R[:, t] = Y[:, t] - W_S @ x_S
    return supports, X_hat


# ---------------------------------------------------------------------------
# One config
# ---------------------------------------------------------------------------
def _run_one(m: int, n: int, d: int, k_synth: int, mask_seed: int,
             sigma: float, n_trials: int) -> dict:
    """Returns dict of summary metrics + raw arrays paths."""
    rng = np.random.default_rng(mask_seed + 50_000)
    mask = _sample_expander_mask(m, n, d, rng)
    W = _build_decoder(mask, d, rng)  # columns have unit norm by construction

    trial_rng = np.random.default_rng(mask_seed * 100 + int(k_synth) * 7 + int(sigma * 1000))

    # Sample supports and coefficients for all trials at once.
    supports_true = np.stack([trial_rng.choice(n, size=k_synth, replace=False)
                              for _ in range(n_trials)], axis=0)  # (T, k)
    X_true = np.zeros((n, n_trials), dtype=np.float32)
    for t in range(n_trials):
        X_true[supports_true[t], t] = trial_rng.standard_normal(k_synth).astype(np.float32)

    H = W @ X_true  # (m, T)
    if sigma > 0:
        noise_norm = np.linalg.norm(H, axis=0) / np.sqrt(m)  # (T,)
        Z = trial_rng.standard_normal(H.shape).astype(np.float32)
        H = H + sigma * noise_norm[None, :] * Z

    supports_hat, X_hat = batched_omp(W, H, k_synth)

    # Exact support recovery: sort and compare
    true_sorted = np.sort(supports_true, axis=1)
    hat_sorted = np.sort(supports_hat, axis=1)
    exact = (true_sorted == hat_sorted).all(axis=1)
    recovery_rate = float(exact.mean())

    code_err = np.linalg.norm(X_hat - X_true, axis=0) / np.linalg.norm(X_true, axis=0).clip(min=1e-12)
    H_hat = W @ X_hat
    recon_err = np.linalg.norm(H_hat - (W @ X_true), axis=0) / np.linalg.norm(W @ X_true, axis=0).clip(min=1e-12)

    # Cheap geometry quantities on this mask (seed 0 semantics):
    col_norms = np.linalg.norm(W, axis=0, keepdims=True).clip(min=1e-12)
    W_n = W / col_norms
    col_inf = np.abs(W_n).max(axis=0)
    beta = np.sqrt(d) * col_inf
    beta_max = float(beta.max())
    # mu_1 over k_synth selected top columns
    G = np.abs(W_n.T @ W_n)
    np.fill_diagonal(G, 0.0)
    if k_synth < n:
        topk = np.partition(G, -k_synth, axis=1)[:, -k_synth:]
        mu1_k = float(topk.sum(axis=1).max())
    else:
        mu1_k = float(G.sum(axis=1).max())
    s = k_synth + 1
    epsilon_count = float(max(0.0, 1.0 - m / (d * s)))
    R_est_count = beta_max ** 2 * epsilon_count * (2 * k_synth + 1)

    # Persist raw per-trial arrays.
    arch_id = make_id("synthetic_expander", m, n, d, k_synth, mask_seed,
                      f"omp_sigma{sigma:.3f}")
    raw_dir = RAW_DIR / arch_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "exact.npy", exact.astype(bool))
    np.save(raw_dir / "code_err.npy", code_err.astype(np.float32))
    np.save(raw_dir / "recon_err.npy", recon_err.astype(np.float32))

    return {
        "id": arch_id,
        "architecture": "synthetic_expander",
        "m": m, "n": n, "d": d, "k_synth": k_synth,
        "mask_seed": mask_seed, "sigma": sigma,
        "n_trials": n_trials,
        "metrics": {
            "support_recovery_rate": recovery_rate,
            "code_err_mean":   float(code_err.mean()),
            "code_err_std":    float(code_err.std()),
            "code_err_sem":    float(code_err.std() / np.sqrt(n_trials)),
            "recon_err_mean":  float(recon_err.mean()),
            "recon_err_std":   float(recon_err.std()),
            "recon_err_sem":   float(recon_err.std() / np.sqrt(n_trials)),
        },
        "geometry": {
            "beta_max": beta_max,
            "mu1_k": mu1_k,
            "epsilon_count": epsilon_count,
            "R_count": R_est_count,
        },
        "raw_data_paths": {
            "exact":     str(raw_dir / "exact.npy"),
            "code_err":  str(raw_dir / "code_err.npy"),
            "recon_err": str(raw_dir / "recon_err.npy"),
        },
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "sweep_type": "synthetic_omp_recovery",
        },
    }


# ---------------------------------------------------------------------------
# Sweep CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--d", type=int, nargs="+", default=[7, 30, 50, 100, 200])
    p.add_argument("--k-synth", type=int, nargs="+",
                   default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--sigma", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    p.add_argument("--mask-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    configs = [(d, k, s, sig) for d in args.d for k in args.k_synth
               for s in args.mask_seeds for sig in args.sigma]
    _log(f"Synthetic OMP sweep: {len(configs)} configs × {args.n_trials} trials")

    db = load_db(path=SYNTH_DB_PATH)
    existing_ids = {e["id"] for e in db}

    for i, (d, k, mask_seed, sigma) in enumerate(configs, 1):
        cfg_id = make_id("synthetic_expander", args.m, args.n, d, k, mask_seed,
                         f"omp_sigma{sigma:.3f}")
        if not args.force and cfg_id in existing_ids:
            continue
        t0 = time.time()
        entry = _run_one(args.m, args.n, d, k, mask_seed, sigma, args.n_trials)
        upsert_safe(entry, path=SYNTH_DB_PATH)
        mt = entry["metrics"]
        _log(f"[{i}/{len(configs)}] d={d:3d} k={k:3d} seed={mask_seed} sigma={sigma:.2f}  "
             f"rec={mt['support_recovery_rate']:.3f} code_err={mt['code_err_mean']:.3f} "
             f"recon_err={mt['recon_err_mean']:.3f}  ({time.time()-t0:.1f}s)")

    _log("Synthetic OMP sweep complete.")


if __name__ == "__main__":
    main()
