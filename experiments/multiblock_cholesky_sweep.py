"""Multi-block Cholesky throughput on Modal A10G.

Per-architecture, runs ``batched_multiblock_cholesky`` at $L = m/d$ on
the held-out test split. Algorithmically equivalent to gOMP $L{=}m/d$
(verified to fp32 noise) but with a fused Cholesky refit per block in
place of the sequential MGS picks.

Run: ``venv/bin/modal run experiments/multiblock_cholesky_sweep.py::main``
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


APP_NAME = "mech-expander-multiblock-cholesky"

M = 512
N = 4096
K = 64
SEED = 0
N_SAMPLES = 1024
N_TIMING_RUNS = 3
BATCH = 1024

OUT_PATH = "results/multiblock_cholesky_sweep.csv"


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "numpy", "scipy", "triton",
        "transformers==4.44.0", "datasets", "accelerate", "tqdm", "zstandard",
    )
    .add_local_python_source("config", "db", "models", "inference",
                             "kernels")
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

    from inference.batched_multiblock_cholesky import batched_multiblock_cholesky
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

    rows_csv: list[dict] = []
    configs = [
        ("expander_tied",  7,   "Expander (d=7)"),
        ("expander_tied",  50,  "Expander (d=50)"),
        ("expander_tied",  200, "Expander (d=200)"),
        ("dense_warmtied", M,   "Dense-SAE"),
    ]

    for arch, d, label in configs:
        L_max = max(1, M // d)
        n_blocks = -(-K // L_max)
        print(f"\n=== {label} (m/d={M/d:.1f}, L_max={L_max}, "
              f"n_blocks={n_blocks}) ===")
        e = find(arch, M, N, d, K, SEED)
        if e is None:
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

        rows_t = torch.from_numpy(rows_np).long().to(device)
        values_bf16 = torch.from_numpy(values_np).to(device).to(torch.bfloat16)
        Y_bf16 = torch.from_numpy(test_acts - b_dec).to(device).to(torch.bfloat16)
        truth_fp32 = torch.from_numpy(test_acts).to(device)
        b_fp32 = torch.from_numpy(b_dec).to(device)
        W_T_fp32 = torch.from_numpy(W_dense.T).to(device)

        B = BATCH if d != M else 256

        try:
            _ = batched_multiblock_cholesky(values_bf16, rows_t, M, N, d_eff,
                                            Y_bf16[:B], K)
            if device == "cuda":
                torch.cuda.synchronize()
        except Exception as exc:
            print(f"  FAILED warmup: {exc}")
            continue

        n_full_batches = N_SAMPLES // B
        tokens = n_full_batches * B
        durations = []
        for _ in range(N_TIMING_RUNS):
            t0 = time.perf_counter()
            for i in range(n_full_batches):
                sl = slice(i * B, (i + 1) * B)
                _ = batched_multiblock_cholesky(values_bf16, rows_t, M, N,
                                                d_eff, Y_bf16[sl], K)
            if device == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - t0)

        x_hat, _ = batched_multiblock_cholesky(values_bf16, rows_t, M, N,
                                               d_eff, Y_bf16[:B], K)
        recon = (x_hat.float() @ W_T_fp32) + b_fp32
        rel = (torch.norm(truth_fp32[:B] - recon, dim=-1) /
               torch.norm(truth_fp32[:B], dim=-1).clamp(min=1e-12))
        mean_err = float(rel.mean())

        median_dur = sorted(durations)[len(durations) // 2]
        tps = tokens / median_dur
        print(f"  multi-block chol  rel_err≈{mean_err:.4f}  "
              f"tokens/s={tps:>10.1f}")
        rows_csv.append({
            "label": label, "arch": arch, "d": d, "k": K,
            "L_max": L_max, "n_blocks": n_blocks, "batch": B,
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
            "label", "arch", "d", "k", "L_max", "n_blocks", "batch",
            "rel_err", "tokens_per_sec", "n_samples", "device",
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nWrote {OUT_PATH} ({len(results)} rows)")
