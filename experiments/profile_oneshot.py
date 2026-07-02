"""Profile the per-op cost of single-shot Cholesky on Modal A10G.

Runs ``batched_oneshot_lstsq`` (Cholesky variant) at $d{=}7$ with
B=1024 and reports CUDA-event-timed breakdowns of every op in the
function so we can see where the ~2.8 ms per batch goes.

Run: ``venv/bin/modal run experiments/profile_oneshot.py::main``
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


APP_NAME = "mech-expander-profile-oneshot"

M = 512
N = 4096
K = 64
SEED = 0
N_SAMPLES = 1024
N_RUNS = 20


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


def _time_op(name: str, fn, n_runs: int) -> dict:
    """Run ``fn`` n_runs times and return CUDA-event median ms."""
    import torch
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    # Warm.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for i in range(n_runs):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(starts[i].elapsed_time(ends[i]) for i in range(n_runs))
    median = times_ms[n_runs // 2]
    return {"name": name, "median_ms": median,
            "p25_ms": times_ms[n_runs // 4],
            "p75_ms": times_ms[3 * n_runs // 4]}


@app.function(image=image, gpu="A10G", timeout=600)
def run() -> list[dict]:
    import json
    import numpy as np
    import torch

    from inference.batched_oneshot_lstsq import batched_oneshot_lstsq
    from inference.structured_omp import _flat_storage_from_dense
    from models import build

    device = "cuda"
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

    arch, d = "expander_tied", 7
    e = find(arch, M, N, d, K, SEED)
    repo_path = e.get("model_path")
    local_path = repo_path.replace("results/models/", "/cache_local/models/")
    model = build(arch, m=M, n=N, d=d, k=K, seed=SEED)
    sd = torch.load(local_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.eval()

    W_dense = model.W_dec.detach().cpu().numpy().astype(np.float32)
    b_dec = model.b_dec.detach().cpu().numpy().astype(np.float32)
    mask = model.mask.detach().cpu().numpy()
    values_np, rows_np, d_eff = _flat_storage_from_dense(W_dense, mask)

    rows_t = torch.from_numpy(rows_np).long().to(device)
    values_bf16 = torch.from_numpy(values_np).to(device).to(torch.bfloat16)
    Y_bf16 = torch.from_numpy(test_acts - b_dec).to(device).to(torch.bfloat16)

    B = 1024
    Y = Y_bf16[:B]
    n = N
    m = M
    k = K
    dtype = torch.bfloat16
    values_2d = values_bf16.view(n, d_eff).to(dtype=dtype)
    rows_2d = rows_t.view(n, d_eff).long()

    print(f"\n=== Profiling oneshot Cholesky at d={d}, B={B}, K={k} ===\n")

    rows = []

    # Time end-to-end first.
    rows.append(_time_op("FULL  batched_oneshot_lstsq",
                         lambda: batched_oneshot_lstsq(values_bf16, rows_t,
                                                       m, n, d_eff, Y, k),
                         N_RUNS))

    # Step 1: Triton-fused correlation = abs(encoder_forward(Y, values, rows)).
    from kernels.triton.encoder_fwd_v3 import encoder_forward_v3
    values_fp32_flat = values_bf16.view(-1).to(torch.float32).contiguous()
    rows_int32_flat = rows_t.view(-1).to(torch.int32).contiguous()
    b_zero = torch.zeros(n, device=device, dtype=torch.float32)
    def step1_triton():
        Y_fp32 = Y.to(torch.float32).contiguous()
        return encoder_forward_v3(Y_fp32, values_fp32_flat, rows_int32_flat,
                                  b_zero, n=n, d=d_eff).abs()
    rows.append(_time_op("step1 Triton correlation",
                         step1_triton, N_RUNS))

    # Step 2: topk
    corrs = step1_triton()
    def step2():
        return corrs.topk(k, dim=-1)
    rows.append(_time_op("step2 topk", step2, N_RUNS))

    _, support = corrs.topk(k, dim=-1)

    # Step 3: gather sel_rows / sel_vals
    def step3():
        return rows_2d[support], values_2d[support]
    rows.append(_time_op("step3 gather (sel_rows, sel_vals)",
                         step3, N_RUNS))

    sel_rows = rows_2d[support]
    sel_vals = values_2d[support]

    # Step 4: scatter to build W_S_T
    def step4():
        W_S_T = torch.zeros(B, k, m, device=device, dtype=dtype)
        W_S_T.scatter_(2, sel_rows, sel_vals)
        return W_S_T
    rows.append(_time_op("step4 scatter to build W_S_T",
                         step4, N_RUNS))

    W_S_T = step4()

    # Step 5: transpose + contiguous to get W_S = (B, m, k)
    def step5():
        return W_S_T.transpose(1, 2).contiguous()
    rows.append(_time_op("step5 W_S = transpose+contiguous",
                         step5, N_RUNS))

    W_S = W_S_T.transpose(1, 2).contiguous()

    # Step 6: bmm A = W_S_T @ W_S
    def step6():
        return torch.bmm(W_S_T, W_S)
    rows.append(_time_op("step6 bmm A = W_S_T @ W_S",
                         step6, N_RUNS))

    A = torch.bmm(W_S_T, W_S)

    # Step 7: bmm rhs = W_S_T @ Y
    def step7():
        return torch.bmm(W_S_T, Y.unsqueeze(-1))
    rows.append(_time_op("step7 bmm rhs = W_S_T @ Y",
                         step7, N_RUNS))

    rhs = torch.bmm(W_S_T, Y.unsqueeze(-1))

    # Step 8: cast A to fp32
    def step8():
        return A.float()
    rows.append(_time_op("step8 A.float() cast",
                         step8, N_RUNS))

    # Step 9: cholesky on (B, k, k) fp32
    A_fp32 = A.float()
    def step9():
        return torch.linalg.cholesky(A_fp32)
    rows.append(_time_op("step9 cholesky on (B, k, k) fp32",
                         step9, N_RUNS))

    L_chol = torch.linalg.cholesky(A_fp32)

    # Step 10: cholesky_solve
    rhs_fp32 = rhs.float()
    def step10():
        return torch.cholesky_solve(rhs_fp32, L_chol).squeeze(-1).to(dtype)
    rows.append(_time_op("step10 cholesky_solve + cast back",
                         step10, N_RUNS))

    x_S = torch.cholesky_solve(rhs_fp32, L_chol).squeeze(-1).to(dtype)

    # Step 11: scatter into x_hat
    def step11():
        x_hat = torch.zeros(B, n, device=device, dtype=dtype)
        x_hat.scatter_(1, support, x_S)
        return x_hat
    rows.append(_time_op("step11 final scatter to x_hat",
                         step11, N_RUNS))

    return rows


@app.local_entrypoint()
def main():
    rows = run.remote()
    full_ms = rows[0]["median_ms"]
    sum_steps = sum(r["median_ms"] for r in rows[1:])
    print("\n=== Per-op breakdown ===\n")
    print(f"  {'op':<45} {'ms':>8} {'%':>6}")
    for r in rows:
        pct = 100.0 * r["median_ms"] / full_ms
        marker = "**" if r["median_ms"] / full_ms > 0.10 else "  "
        print(f"  {marker} {r['name']:<43} {r['median_ms']:>7.3f}  {pct:>5.1f}%")
    print(f"\n  sum of step1..11 = {sum_steps:.3f} ms (vs FULL "
          f"{full_ms:.3f} ms; ratio {sum_steps / full_ms:.2f})")
    print(f"  per-token = {full_ms * 1000 / 1024:.2f} us/tok  "
          f"(throughput = {1024 / (full_ms / 1000):.0f} tok/s)")
