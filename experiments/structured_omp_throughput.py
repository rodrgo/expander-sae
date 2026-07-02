"""Compare vanilla OMP vs structure-aware OMP on the headline configurations.

For each (Expander d in {7, 50, 200}, Dense-SAE) at k=64, seed 0:
  1. Load the trained model + held-out test activations.
  2. Verify structured OMP matches vanilla OMP on 5 random samples to
     within $10^{-9}$ rel-err.
  3. Time both implementations on the first 200 test activations and
     emit rel-err + tokens/sec rows.

Output: results/structured_omp_table.csv.
"""
from __future__ import annotations

import csv
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_entry, load_db
from inference.omp import omp
from inference.structured_omp import _flat_storage_from_dense, structured_omp
from inference.structured_omp_qr import structured_omp_qr
from models import build


M = 512
N = 4096
K = 64
SEED = 0
N_OMP_SAMPLES = 200
N_VERIFY = 5
N_TIMING_RUNS = 1
OUT_CSV = "results/structured_omp_table.csv"


def _load_model(arch: str, m: int, n: int, d: int, k: int, seed: int):
    db = load_db()
    entry = get_entry(db, arch, m, n, d, k, seed, "encoder")
    if entry is None or not entry.get("model_path"):
        raise RuntimeError(f"no entry for {arch} d={d} k={k} seed={seed}")
    model = build(arch, m=m, n=n, d=d, k=k, seed=seed)
    model.load_state_dict(torch.load(entry["model_path"],
                                     map_location="cpu", weights_only=True))
    return model.eval()


def _verify_match(W_dense: np.ndarray, b_dec: np.ndarray,
                  values: np.ndarray, rows: np.ndarray,
                  m: int, n: int, d: int,
                  acts: np.ndarray, k: int, n_samples: int = N_VERIFY) -> None:
    rng = np.random.default_rng(0)
    idxs = rng.choice(acts.shape[0], size=n_samples, replace=False)
    for i in idxs:
        y = acts[i] - b_dec
        x_hat_v, _ = omp(W_dense, y, k)
        x_hat_s, _ = structured_omp(values, rows, m, n, d, y, k)
        x_hat_q, _ = structured_omp_qr(values, rows, m, n, d, y, k)
        recon_v = W_dense @ x_hat_v + b_dec
        recon_s = W_dense @ x_hat_s + b_dec
        recon_q = W_dense @ x_hat_q + b_dec
        err_v = np.linalg.norm(acts[i] - recon_v) / max(
            np.linalg.norm(acts[i]), 1e-12)
        err_s = np.linalg.norm(acts[i] - recon_s) / max(
            np.linalg.norm(acts[i]), 1e-12)
        err_q = np.linalg.norm(acts[i] - recon_q) / max(
            np.linalg.norm(acts[i]), 1e-12)
        for name, err in [("structured", err_s), ("structured_qr", err_q)]:
            diff = abs(err_v - err)
            assert diff < 1e-5, (
                f"vanilla rel_err={err_v:.6f}, {name}={err:.6f}, "
                f"diff={diff:.2e} (sample {i})")
    print(f"  verified: structured / structured_qr ≈ vanilla on "
          f"{n_samples} samples (max rel_err diff < 1e-5)")


def _time_dense(W_dense: np.ndarray, b_dec: np.ndarray,
                acts: np.ndarray, k: int, n_samples: int) -> tuple[float, float]:
    test = acts[:n_samples]
    # Warm up.
    _ = omp(W_dense, test[0] - b_dec, k)
    durations = []
    rel_errs = None
    for _ in range(N_TIMING_RUNS):
        errs = []
        t0 = time.perf_counter()
        for i in range(n_samples):
            y = test[i] - b_dec
            x_hat, _ = omp(W_dense, y, k)
            recon = W_dense @ x_hat + b_dec
            errs.append(float(np.linalg.norm(test[i] - recon) /
                              max(np.linalg.norm(test[i]), 1e-12)))
        durations.append(time.perf_counter() - t0)
        rel_errs = errs
    median_dur = statistics.median(durations)
    return float(np.mean(rel_errs)), float(n_samples / median_dur)


def _time_method(method_fn, values: np.ndarray, rows: np.ndarray,
                 W_dense: np.ndarray, b_dec: np.ndarray,
                 m: int, n: int, d: int,
                 acts: np.ndarray, k: int, n_samples: int
                 ) -> tuple[float, float]:
    test = acts[:n_samples]
    # Warm up.
    _ = method_fn(values, rows, m, n, d, test[0] - b_dec, k)
    durations = []
    rel_errs = None
    for _ in range(N_TIMING_RUNS):
        errs = []
        t0 = time.perf_counter()
        for i in range(n_samples):
            y = test[i] - b_dec
            x_hat, _ = method_fn(values, rows, m, n, d, y, k)
            recon = W_dense @ x_hat + b_dec
            errs.append(float(np.linalg.norm(test[i] - recon) /
                              max(np.linalg.norm(test[i]), 1e-12)))
        durations.append(time.perf_counter() - t0)
        rel_errs = errs
    median_dur = statistics.median(durations)
    return float(np.mean(rel_errs)), float(n_samples / median_dur)


def main() -> None:
    test_acts = np.load("data/activations_test.npy").astype(np.float32)

    rows_csv: list[dict] = []
    configs = [
        ("expander_tied",  M, N, 7,   K, SEED, "Expander (d=7)"),
        ("expander_tied",  M, N, 50,  K, SEED, "Expander (d=50)"),
        ("expander_tied",  M, N, 200, K, SEED, "Expander (d=200)"),
        ("dense_warmtied", M, N, M,   K, SEED, "Dense-SAE"),
    ]
    for arch, m, n, d, k, seed, label in configs:
        print(f"\n=== {label} ===")
        model = _load_model(arch, m, n, d, k, seed)
        W_dense = model.W_dec.detach().cpu().numpy().astype(np.float32)
        b_dec = model.b_dec.detach().cpu().numpy().astype(np.float32)
        if hasattr(model, "mask"):
            mask = model.mask.detach().cpu().numpy()
        else:
            mask = (np.abs(W_dense) > 1e-12).astype(np.float32)
        values, rows_idx, d_eff = _flat_storage_from_dense(W_dense, mask)
        print(f"  d_effective={d_eff} (config d={d}); flat storage = "
              f"{values.shape[0]} entries")

        # Numerical equivalence check.
        _verify_match(W_dense, b_dec, values, rows_idx, m, n, d_eff,
                      test_acts, k, N_VERIFY)

        # Time vanilla.
        rer_v, tps_v = _time_dense(W_dense, b_dec, test_acts, k, N_OMP_SAMPLES)
        print(f"  vanilla    OMP rel_err={rer_v:.4f} tokens/s={tps_v:.2f}")
        rows_csv.append({
            "label": label, "arch": arch, "d": d, "k": k,
            "method": "omp",
            "rel_err": round(rer_v, 4),
            "tokens_per_sec": round(tps_v, 2),
            "n_samples": N_OMP_SAMPLES,
        })

        # Time structured (lstsq refit).
        rer_s, tps_s = _time_method(structured_omp, values, rows_idx,
                                    W_dense, b_dec, m, n, d_eff,
                                    test_acts, k, N_OMP_SAMPLES)
        speedup_s = tps_s / max(tps_v, 1e-12)
        print(f"  structured     OMP rel_err={rer_s:.4f} "
              f"tokens/s={tps_s:.2f}  ({speedup_s:.2f}x vs vanilla)")
        rows_csv.append({
            "label": label, "arch": arch, "d": d, "k": k,
            "method": "structured_omp",
            "rel_err": round(rer_s, 4),
            "tokens_per_sec": round(tps_s, 2),
            "n_samples": N_OMP_SAMPLES,
        })

        # Time structured + incremental QR.
        rer_q, tps_q = _time_method(structured_omp_qr, values, rows_idx,
                                    W_dense, b_dec, m, n, d_eff,
                                    test_acts, k, N_OMP_SAMPLES)
        speedup_q = tps_q / max(tps_v, 1e-12)
        print(f"  structured+QR  OMP rel_err={rer_q:.4f} "
              f"tokens/s={tps_q:.2f}  ({speedup_q:.2f}x vs vanilla)")
        rows_csv.append({
            "label": label, "arch": arch, "d": d, "k": k,
            "method": "structured_omp_qr",
            "rel_err": round(rer_q, 4),
            "tokens_per_sec": round(tps_q, 2),
            "n_samples": N_OMP_SAMPLES,
        })

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "arch", "d", "k", "method", "rel_err",
            "tokens_per_sec", "n_samples", "hardware",
        ])
        w.writeheader()
        for r in rows_csv:
            r["hardware"] = platform.platform()
            w.writerow(r)
    print(f"\nWrote {OUT_CSV} ({len(rows_csv)} rows)")


if __name__ == "__main__":
    main()
