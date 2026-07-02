"""Generalised OMP (gOMP) block-size sweep on Modal A10G.

For each (Expander d in {7, 50, 200}, Dense-SAE) at k=64, seed 0, runs
gOMP at $L \\in \\{1, 2, 4, 8, 16, 32\\}$ on a held-out subset, capped at the
combinatorial safety threshold $L_{\\max} \\approx m/d$ per architecture
(d=7 → all L; d=50 → L \\le 4; d=200 → L \\le 2; Dense-SAE → only L=1).

Reports rel-err and tokens/s per (arch, L). Output:
results/gomp_sweep.csv.

Run: ``venv/bin/modal run experiments/gomp_sweep.py::main``
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


APP_NAME = "mech-expander-gomp-sweep"
VOLUME_NAME = "mech-expander-cache"
VOL_MOUNT = "/cache"

M = 512
N = 4096
K = 64
SEED = 0
N_SAMPLES = 1024
N_TIMING_RUNS = 3
BATCH = 1024            # bf16 + B=1024 (best from option C)

OUT_PATH = "results/gomp_sweep.csv"


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "numpy", "scipy",
        "transformers==4.44.0", "datasets", "accelerate", "tqdm", "zstandard",
    )
    .add_local_python_source("config", "db", "models", "inference")
    .add_local_dir(
        "results/models",
        "/cache_local/models")
    .add_local_dir(
        "data",
        "/cache_local/data")
    .add_local_file(
        "results/benchmark_db.json",
        "/cache_local/benchmark_db.json")
)


@app.function(image=image, gpu="A10G", timeout=1800)
def time_all() -> list[dict]:
    import numpy as np
    import torch

    from inference.batched_gomp_qr import batched_gomp_qr
    from inference.structured_omp import _flat_storage_from_dense
    from models import build

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")

    db_path = "/cache_local/benchmark_db.json"
    with open(db_path) as f:
        db = json.load(f)

    def find(arch, m, n, d, k, seed):
        for e in db:
            if (e.get("architecture") == arch and e.get("m") == m
                    and e.get("n") == n and e.get("d") == d
                    and e.get("k") == k and e.get("seed") == seed
                    and e.get("inference_method") == "encoder"):
                return e
        return None

    test_acts = np.load("/cache_local/data/activations_test.npy"
                        ).astype(np.float32)[:N_SAMPLES]
    print(f"loaded {test_acts.shape[0]} test activations of dim "
          f"{test_acts.shape[1]}")

    rows_csv: list[dict] = []
    # (arch, d, label, [L values to sweep])
    configs = [
        ("expander_tied",  7,   "Expander (d=7)",   [1, 2, 4, 8, 16, 32]),
        ("expander_tied",  50,  "Expander (d=50)",  [1, 2, 4]),
        ("expander_tied",  200, "Expander (d=200)", [1, 2]),
        ("dense_warmtied", M,   "Dense-SAE",        [1]),
    ]

    for arch, d, label, L_values in configs:
        print(f"\n=== {label} (m/d={M/d:.0f}, sweep L in {L_values}) ===")
        e = find(arch, M, N, d, K, SEED)
        if e is None:
            print(f"  no DB entry for {label}")
            continue
        repo_path = e.get("model_path")
        local_path = repo_path.replace("results/models/", "/cache_local/models/")
        model = build(arch, m=M, n=N, d=d, k=K, seed=SEED)
        sd = torch.load(local_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model = model.eval()

        W_dense = model.W_dec.detach().cpu().numpy().astype(np.float32)
        b_dec = model.b_dec.detach().cpu().numpy().astype(np.float32)
        mask = (model.mask.detach().cpu().numpy() if hasattr(model, "mask")
                else (np.abs(W_dense) > 1e-12).astype(np.float32))
        values_np, rows_np, d_eff = _flat_storage_from_dense(W_dense, mask)
        print(f"  d_effective={d_eff} flat_size={values_np.shape[0]}")

        rows_t = torch.from_numpy(rows_np).long().to(device)
        values_bf16 = torch.from_numpy(values_np).to(device).to(torch.bfloat16)
        Y_bf16 = torch.from_numpy(test_acts - b_dec).to(device).to(torch.bfloat16)
        truth_fp32 = torch.from_numpy(test_acts).to(device)
        b_fp32 = torch.from_numpy(b_dec).to(device)
        W_T_fp32 = torch.from_numpy(W_dense.T).to(device)

        # Pick batch size that fits.
        B = BATCH if d != M else 256

        for L in L_values:
            # Warm up.
            try:
                _ = batched_gomp_qr(values_bf16, rows_t, M, N, d_eff,
                                    Y_bf16[:B], K, L=L)
                if device == "cuda":
                    torch.cuda.synchronize()
            except Exception as exc:
                print(f"  L={L:>3}  FAILED warmup: {exc}")
                continue

            n_full_batches = N_SAMPLES // B
            tokens = n_full_batches * B
            durations = []
            for _ in range(N_TIMING_RUNS):
                t0 = time.perf_counter()
                for i in range(n_full_batches):
                    sl = slice(i * B, (i + 1) * B)
                    _ = batched_gomp_qr(values_bf16, rows_t, M, N, d_eff,
                                        Y_bf16[sl], K, L=L)
                if device == "cuda":
                    torch.cuda.synchronize()
                durations.append(time.perf_counter() - t0)

            # rel_err sanity (fp32 reconstruction).
            x_hat, _ = batched_gomp_qr(values_bf16, rows_t, M, N, d_eff,
                                       Y_bf16[:B], K, L=L)
            recon = (x_hat.float() @ W_T_fp32) + b_fp32
            rel = (torch.norm(truth_fp32[:B] - recon, dim=-1) /
                   torch.norm(truth_fp32[:B], dim=-1).clamp(min=1e-12))
            mean_err = float(rel.mean())

            median_dur = sorted(durations)[len(durations) // 2]
            tps = tokens / median_dur
            print(f"  L={L:>3}  rel_err≈{mean_err:.4f}  "
                  f"tokens/s={tps:>10.1f}  ({tokens} samples in "
                  f"{median_dur:.2f}s)")
            rows_csv.append({
                "label": label, "arch": arch, "d": d, "k": K,
                "L": L, "batch": B,
                "rel_err": round(mean_err, 4),
                "tokens_per_sec": round(tps, 2),
                "n_samples": tokens,
                "device": device,
            })

    return rows_csv


@app.local_entrypoint()
def main():
    results = time_all.remote()
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "arch", "d", "k", "L", "batch",
            "rel_err", "tokens_per_sec", "n_samples", "device",
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nWrote {OUT_PATH} ({len(results)} rows)")
