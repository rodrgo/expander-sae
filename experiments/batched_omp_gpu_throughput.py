"""Batched structure-aware OMP+QR throughput on Modal A10G.

For each (Expander d in {7, 50, 200}, Dense-SAE) at k=64, seed 0, runs
the batched-PyTorch implementation on the held-out test split at batch
sizes B in {1, 32, 256} and reports tokens/s for each.

Output: results/batched_omp_gpu_table.csv (mirrored from the volume).

Run: ``venv/bin/modal run experiments/batched_omp_gpu_throughput.py::main``
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


APP_NAME = "mech-expander-batched-omp"
VOLUME_NAME = "mech-expander-cache"
VOL_MOUNT = "/cache"

M = 512
N = 4096
K = 64
SEED = 0
N_OMP_SAMPLES = 1024              # held-out subset for timing
BATCH_SIZES = [256, 1024]
DTYPES = [("fp32", "float32"), ("bf16", "bfloat16")]
N_TIMING_RUNS = 3

OUT_PATH = "results/batched_omp_gpu_table.csv"


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

    from inference.batched_structured_omp_qr import batched_structured_omp_qr
    from inference.batched_structured_omp_qr_sparse import (
        batched_structured_omp_qr_sparse,
    )
    from inference.structured_omp import _flat_storage_from_dense
    from models import build

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")

    # Load DB to find checkpoint paths.
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

    # Test activations.
    test_acts = np.load("/cache_local/data/activations_test.npy"
                        ).astype(np.float32)
    test_acts = test_acts[:N_OMP_SAMPLES]
    print(f"loaded {test_acts.shape[0]} test activations of dim "
          f"{test_acts.shape[1]}")

    rows_csv: list[dict] = []
    configs = [
        ("expander_tied",  M, N, 7,   K, SEED, "Expander (d=7)"),
        ("expander_tied",  M, N, 50,  K, SEED, "Expander (d=50)"),
        ("expander_tied",  M, N, 200, K, SEED, "Expander (d=200)"),
        ("dense_warmtied", M, N, M,   K, SEED, "Dense-SAE"),
    ]

    for arch, m, n, d, k, seed, label in configs:
        print(f"\n=== {label} ===")
        e = find(arch, m, n, d, k, seed)
        if e is None:
            print(f"  no DB entry for {label}; skipping")
            continue

        # Resolve local checkpoint path inside the image.
        repo_path = e.get("model_path")           # e.g. results/models/expander_tied_...
        local_path = repo_path.replace("results/models/", "/cache_local/models/")
        model = build(arch, m=m, n=n, d=d, k=k, seed=seed)
        sd = torch.load(local_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model = model.eval()

        W_dense = model.W_dec.detach().cpu().numpy().astype(np.float32)
        b_dec = model.b_dec.detach().cpu().numpy().astype(np.float32)
        if hasattr(model, "mask"):
            mask = model.mask.detach().cpu().numpy()
        else:
            mask = (np.abs(W_dense) > 1e-12).astype(np.float32)
        values_np, rows_np, d_eff = _flat_storage_from_dense(W_dense, mask)
        print(f"  d_effective={d_eff} flat_size={values_np.shape[0]}")

        # CPU numpy → GPU fp32 base tensors. We cast per-dtype below.
        rows_t = torch.from_numpy(rows_np).long().to(device)
        values_fp32 = torch.from_numpy(values_np).to(device)               # (d*n,) fp32
        Y_fp32 = torch.from_numpy(test_acts - b_dec).to(device)            # (N, m) fp32
        b_fp32 = torch.from_numpy(b_dec).to(device)
        truth = torch.from_numpy(test_acts).to(device)
        W_T_fp32 = torch.from_numpy(W_dense.T).to(device)

        # ---- Encoder forward on GPU (apples-to-apples comparison row) ----
        gpu_model = model.to(device).eval()
        for enc_dtype_name, enc_dtype_str in DTYPES:
            enc_dtype = getattr(torch, enc_dtype_str)
            try:
                gpu_model_d = gpu_model.to(enc_dtype) if enc_dtype_str != "float32" else gpu_model
                # Pick the largest batch that fits.
                for enc_B in [1024, 256]:
                    Y_enc = torch.from_numpy(test_acts).to(device).to(enc_dtype)
                    try:
                        # Warm up.
                        with torch.no_grad():
                            _ = gpu_model_d(Y_enc[:enc_B])
                        if device == "cuda":
                            torch.cuda.synchronize()
                        durations = []
                        for _ in range(N_TIMING_RUNS):
                            t0 = time.perf_counter()
                            for i in range(N_OMP_SAMPLES // enc_B):
                                with torch.no_grad():
                                    _ = gpu_model_d(Y_enc[i * enc_B:(i + 1) * enc_B])
                            if device == "cuda":
                                torch.cuda.synchronize()
                            durations.append(time.perf_counter() - t0)
                        median_dur = sorted(durations)[len(durations) // 2]
                        tokens = (N_OMP_SAMPLES // enc_B) * enc_B
                        tps = tokens / median_dur
                        # rel_err sanity, in fp32.
                        with torch.no_grad():
                            yh, _ = gpu_model_d(Y_enc[:enc_B])
                        rel = (torch.norm(truth[:enc_B].to(enc_dtype) - yh, dim=-1) /
                               torch.norm(truth[:enc_B].to(enc_dtype),
                                          dim=-1).clamp(min=1e-6))
                        mean_err = float(rel.float().mean())
                        print(f"  encoder GPU dtype={enc_dtype_name:>4} "
                              f"B={enc_B:>4}  rel_err≈{mean_err:.4f}  "
                              f"tokens/s={tps:>10.1f}")
                        rows_csv.append({
                            "label": label, "arch": arch, "d": d, "k": k,
                            "variant": f"encoder_{enc_dtype_name}",
                            "batch": enc_B,
                            "rel_err": round(mean_err, 4),
                            "tokens_per_sec": round(tps, 2),
                            "n_samples": tokens,
                            "device": device,
                        })
                        break  # success at this batch size
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        continue
            except Exception as exc:
                print(f"  encoder GPU dtype={enc_dtype_name} FAILED: {exc}")
        # Restore model on CPU and clear GPU model state for the OMP path.
        gpu_model = gpu_model.to("cpu")
        torch.cuda.empty_cache()

        # We only sweep the dense-QR variant here (sparse-QR was a wash on
        # GPU; see the previous run). Engineering knobs: dtype × batch size.
        fn = batched_structured_omp_qr
        for dtype_name, dtype_str in DTYPES:
            torch_dtype = getattr(torch, dtype_str)
            values_t = values_fp32.to(torch_dtype)
            Y_all = Y_fp32.to(torch_dtype)
            b_t = b_fp32.to(torch_dtype)
            W_T_t = W_T_fp32.to(torch_dtype)

            for B in BATCH_SIZES:
                n_full_batches = N_OMP_SAMPLES // B
                tokens = n_full_batches * B

                try:
                    # Warm up: compile + cache first.
                    _ = fn(values_t, rows_t, m, n, d_eff, Y_all[:B], k)
                    if device == "cuda":
                        torch.cuda.synchronize()
                except Exception as exc:
                    print(f"  dtype={dtype_name:>4}  B={B:>4}  FAILED: {exc}")
                    continue

                durations = []
                for _ in range(N_TIMING_RUNS):
                    t0 = time.perf_counter()
                    for i in range(n_full_batches):
                        sl = slice(i * B, (i + 1) * B)
                        _ = fn(values_t, rows_t, m, n, d_eff, Y_all[sl], k)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    durations.append(time.perf_counter() - t0)

                # rel_err sanity in fp32 (cast x_hat back).
                x_hat, _ = fn(values_t, rows_t, m, n, d_eff, Y_all[:B], k)
                recon = (x_hat.float() @ W_T_fp32) + b_fp32
                rel = (torch.norm(truth[:B] - recon, dim=-1) /
                       torch.norm(truth[:B], dim=-1).clamp(min=1e-12))
                mean_err = float(rel.mean())

                median_dur = sorted(durations)[len(durations) // 2]
                tps = tokens / median_dur
                print(f"  dtype={dtype_name:>4}  B={B:>4}  "
                      f"rel_err≈{mean_err:.4f}  tokens/s={tps:>10.1f}  "
                      f"({tokens} samples in {median_dur:.2f}s)")
                rows_csv.append({
                    "label": label, "arch": arch, "d": d, "k": k,
                    "variant": dtype_name,
                    "batch": B,
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
            "label", "arch", "d", "k", "variant", "batch",
            "rel_err", "tokens_per_sec", "n_samples", "device",
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nWrote {OUT_PATH} ({len(results)} rows)")
