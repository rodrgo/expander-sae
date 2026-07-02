"""Run OMP / NIHT / CoSaMP on every trained model.

For each `encoder` entry in the DB:
  - Load the .pt model
  - For each requested method, run evaluate_iterative
  - Save per-sample relerr + boolean support matrix as raw numpy
  - Upsert one DB entry per (config, method)

Runs a ProcessPool so the 231 x 3 = 693 jobs go through in minutes, not hours.
Each worker pins OMP_NUM_THREADS=1 so BLAS doesn't oversubscribe the pool.

Data-efficiency entries (ids with `_b{budget}` suffix) are skipped by default —
fig5 only uses encoder rel_err and running iterative methods on them would
triple the job count for no headline-figure benefit. Pass
--include-data-efficiency to evaluate those too.

CLI:
    python experiments/inference_sweep.py
    python experiments/inference_sweep.py --methods omp niht
    python experiments/inference_sweep.py --arch expander_tied --d 30
    python experiments/inference_sweep.py --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    N_TEST_ITERATIVE, N_TEST_ITERATIVE_SMALL_N, SMALL_N_THRESHOLD,
)
from db import (
    load_db, entry_exists, raw_path, upsert_safe, new_entry_skeleton,
)
from inference import evaluate_iterative
from models import build

LOG_PATH = "results/inference_sweep.log"


def _log(msg: str) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().isoformat()}  {msg}\n")
    print(msg, flush=True)


def _n_samples_for(n: int) -> int:
    return N_TEST_ITERATIVE_SMALL_N if n <= SMALL_N_THRESHOLD else N_TEST_ITERATIVE


def _worker(job: dict) -> dict | None:
    """Train-time work done in the worker process. Saves raw files directly
    (each job writes different paths so no race) and returns the DB entry."""
    # Prevent BLAS from over-threading inside a worker when we've already
    # parallelised at the job level.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import numpy as np
    import scipy.sparse
    import torch
    from inference import evaluate_iterative
    from models import build
    from db import raw_path, new_entry_skeleton

    entry = job["entry"]
    method = job["method"]
    test_acts = np.load(job["test_acts_path"])

    arch = entry["architecture"]
    m, n, d, k, seed = entry["m"], entry["n"], entry["d"], entry["k"], entry["seed"]
    budget = entry["training"].get("data_budget")

    model = build(arch, m=m, n=n, d=d, k=k, seed=seed)
    model.load_state_dict(
        torch.load(entry["model_path"], map_location="cpu", weights_only=True))
    model.eval()

    n_samples = min(_n_samples_for(n), len(test_acts))
    errs, supports, ms_per_sample = evaluate_iterative(
        model, test_acts, k=k, method_name=method, n_samples=n_samples)

    raw_rel = raw_path(arch, m, n, d, k, seed, method, "relerr.npy",
                       data_budget=budget)
    raw_supp = raw_path(arch, m, n, d, k, seed, method, "supports.npz",
                        data_budget=budget)
    Path(raw_rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_rel, errs.astype(np.float32))
    scipy.sparse.save_npz(raw_supp, supports)

    new = new_entry_skeleton(
        arch, m, n, d, k, seed, method,
        train_info=entry.get("training", {}),
        data_budget=budget if budget is not None else 0,
    )
    new["metrics"].update({
        "rel_err_mean": float(errs.mean()),
        "rel_err_std": float(errs.std()),
        "rel_err_median": float(np.median(errs)),
        "n_test_samples": int(len(errs)),
    })
    new["practical"].update({
        "unique_params": entry["practical"].get("unique_params"),
        "decoder_params": entry["practical"].get("decoder_params"),
        "encoder_params": entry["practical"].get("encoder_params"),
        "inference_ms_per_sample": float(ms_per_sample),
    })
    new["raw_data_paths"] = {
        "per_sample_relerr": raw_rel,
        "per_sample_supports": raw_supp,
    }
    new["model_path"] = entry["model_path"]
    return new


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["omp", "niht", "cosamp"],
                   choices=["omp", "niht", "cosamp"])
    p.add_argument("--arch", type=str, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--include-data-efficiency", action="store_true",
                   help="Also evaluate iterative methods on the data-efficiency "
                        "grid (entries with `_b{budget}` in their id).")
    args = p.parse_args()

    db = load_db()
    encoder_entries = [e for e in db if e["inference_method"] == "encoder"]

    def keep(e):
        if args.arch and e["architecture"] != args.arch:
            return False
        if args.d is not None and e["d"] != args.d:
            return False
        if args.k is not None and e["k"] != args.k:
            return False
        if args.seed is not None and e["seed"] != args.seed:
            return False
        if not args.include_data_efficiency and "_b" in e["id"]:
            return False
        return True

    encoder_entries = [e for e in encoder_entries if keep(e)]
    test_acts_path = "data/activations_test.npy"

    # Build full job list, skip rows that already have a DB entry.
    jobs = []
    for e in encoder_entries:
        for method in args.methods:
            if entry_exists(db, e["architecture"], e["m"], e["n"], e["d"],
                            e["k"], e["seed"], method,
                            data_budget=e["training"].get("data_budget")):
                continue
            jobs.append({"entry": e, "method": method,
                         "test_acts_path": test_acts_path})

    _log(f"Inference sweep: {len(jobs)} jobs across {args.workers} workers.")
    if not jobs:
        return

    done, new = 0, 0
    total = len(jobs)
    t0 = datetime.now()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result is None:
                continue
            # Race-safe upsert: reloads DB under lock so a concurrent
            # ce_loss_sweep write isn't clobbered.
            upsert_safe(result)
            new += 1
            mt = result["metrics"]
            mpp = result["practical"]["inference_ms_per_sample"]
            if done % 10 == 0 or done == total:
                elapsed = (datetime.now() - t0).total_seconds()
                rate = done / max(elapsed, 1e-3)
                _log(f"[{done}/{total}] last={result['id']} "
                     f"rel_err={mt['rel_err_mean']:.4f} "
                     f"ms/sample={mpp:.1f} "
                     f"rate={rate*60:.1f}/min")

    _log(f"Inference sweep finished: {new} new entries.")


if __name__ == "__main__":
    main()
