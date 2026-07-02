"""Train every (arch, m, n, d, k, seed, data_budget) in the sweep grid.

Two ways to run:

1. Local (CPU or Mac MPS, small / smoke scale):
       python experiments/training_sweep.py --smoke
       python experiments/training_sweep.py --arch expander_tied --d 30 --k 64

2. Modal (GPU, full sweep):
       modal run experiments/training_sweep.py::sweep

In both modes the driver iterates the config list, skips rows already present
in `results/benchmark_db.json`, trains, saves the checkpoint to
`results/models/`, writes the per-sample rel-err to `results/raw/`, and
upserts one `encoder` entry into the DB. All writes are idempotent — a crash
mid-sweep is safe to rerun.

CLI flags for local mode:
    --arch {expander_tied,dense_tied,dense_indep}
    --d, --k, --seed
    --data-efficiency   (restrict to the data-efficiency grid)
    --smoke             (single tiny expander config for pipeline testing)
    --limit N           (stop after training N new configs; useful for resuming
                         under Modal's 10-GPU concurrency cap)
    --device cpu|mps|cuda
"""
from __future__ import annotations

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from config import (
    TRAIN_DEFAULTS, all_training_configs, data_efficiency_configs,
)
from db import (
    load_db, save_db, entry_exists, model_path, raw_path,
    new_entry_skeleton, upsert, upsert_safe,
)
from models import build, train_sae

LOG_PATH = "results/training_sweep.log"

APP_NAME = "mech-expander-training"
VOLUME_NAME = "mech-expander-cache"
VOL_MOUNT = "/cache"

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "numpy", "scipy",
        "transformers==4.44.0", "datasets", "accelerate", "tqdm", "zstandard",
    )
    .add_local_python_source("config", "db", "models", "inference")
)
_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Core (pure-Python) training routine — no I/O
# ---------------------------------------------------------------------------
def _param_counts(model) -> tuple[int, int, int]:
    arch = model.arch
    m, n = model.m, model.n
    if arch in ("expander_tied", "clustered_sparse", "pruned_retuned_dense"):
        return model.d * n, model.d * n, model.d * n
    if arch == "dense_tied":
        return m * n, m * n, m * n
    if arch in ("dense_warmtied", "dense_randinit", "dense_indep"):
        return 2 * m * n, m * n, m * n
    raise ValueError(arch)


def _log(msg: str) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().isoformat()}  {msg}\n")
    print(msg, flush=True)


def train_single(arch: str, m: int, n: int, d: int, k: int, seed: int,
                 data_budget: int, train_acts: np.ndarray, test_acts: np.ndarray,
                 device: str = "cpu", gpu_name: str | None = None,
                 overrides: dict | None = None
                 ) -> tuple[dict, dict, np.ndarray]:
    """Train one config. Returns (entry_skeleton, state_dict, per_sample_relerr)."""
    hp = dict(TRAIN_DEFAULTS)
    if overrides:
        hp.update(overrides)
    hp["data_budget"] = data_budget

    model = build(arch, m=m, n=n, d=d, k=k, seed=seed)
    acts = train_acts[:data_budget]

    model, info = train_sae(
        model, acts,
        steps=hp["steps"], batch_size=hp["batch_size"],
        lr_max=hp["lr_max"], lr_min=hp["lr_min"],
        grad_clip=hp["grad_clip"],
        resample_interval=hp["resample_interval"],
        device=device,
    )

    state_dict = {k_: v.detach().cpu() for k_, v in model.state_dict().items()}

    test_tensor = torch.from_numpy(test_acts).float()
    model.eval()
    with torch.no_grad():
        y_hat, h = model(test_tensor)
        per_err = (torch.norm(test_tensor - y_hat, dim=-1) /
                   torch.norm(test_tensor, dim=-1).clamp(min=1e-12)).cpu().numpy()
        expl_var = 1.0 - float(
            ((test_tensor - y_hat).pow(2).sum() /
             (test_tensor - test_tensor.mean(dim=0)).pow(2).sum())
        )
        dead_frac = float((h.abs().sum(dim=0) == 0).float().mean())
        l0_mean = float((h.abs() > 1e-10).float().sum(dim=-1).mean())

    uniq, dec_p, enc_p = _param_counts(model)

    entry = new_entry_skeleton(
        arch, m, n, d, k, seed, "encoder",
        train_info={**hp, "wall_clock_s": info["wall_clock_s"], "gpu": gpu_name},
        data_budget=data_budget,
    )
    entry["metrics"].update({
        "rel_err_mean": float(per_err.mean()),
        "rel_err_std": float(per_err.std()),
        "rel_err_median": float(np.median(per_err)),
        "explained_var": expl_var,
        "dead_frac": dead_frac,
        "l0_mean": l0_mean,
        "n_test_samples": int(len(per_err)),
    })
    entry["practical"].update({
        "unique_params": uniq,
        "decoder_params": dec_p,
        "encoder_params": enc_p,
    })
    return entry, state_dict, per_err.astype(np.float32)


# ---------------------------------------------------------------------------
# Shared sweep filter
# ---------------------------------------------------------------------------
def build_sweep(arch_filter: str | None, d_filter: int | None,
                k_filter: int | None, seed_filter: int | None,
                data_efficiency: bool) -> list[tuple]:
    configs = (list(data_efficiency_configs()) if data_efficiency
               else list(all_training_configs()))

    def keep(c):
        a, _m, _n, d, k, seed, _b = c
        if arch_filter and a != arch_filter:
            return False
        if d_filter is not None and d != d_filter:
            return False
        if k_filter is not None and k != k_filter:
            return False
        if seed_filter is not None and seed != seed_filter:
            return False
        return True

    return [c for c in configs if keep(c)]


# ---------------------------------------------------------------------------
# Modal app (always created at module level — required by Modal)
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)


@app.function(image=_image, gpu="A10G", timeout=1200,
              volumes={VOL_MOUNT: _volume}, max_containers=10)
def train_modal(arch: str, m: int, n: int, d: int, k: int, seed: int,
                data_budget: int) -> dict:
    """Train one config on a Modal GPU. Saves checkpoint + per-sample errs to
    the volume; returns the entry dict (without file contents)."""
    import os
    import numpy as np
    import torch

    from models import build  # noqa: F401  (ensured on path by image_with_source)

    # Load activations from the shared volume.
    train_p = f"{VOL_MOUNT}/activations_train.npy"
    test_p = f"{VOL_MOUNT}/activations_test.npy"
    train = np.load(train_p)
    test = np.load(test_p)

    entry, state_dict, per_err = train_single(
        arch, m, n, d, k, seed, data_budget,
        train_acts=train, test_acts=test,
        device="cuda", gpu_name="A10G",
    )

    eid = entry["id"]
    ckpt_path = f"{VOL_MOUNT}/models/{eid}.pt"
    raw_path_remote = f"{VOL_MOUNT}/raw/{eid}_relerr.npy"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_path_remote), exist_ok=True)
    torch.save(state_dict, ckpt_path)
    np.save(raw_path_remote, per_err)
    _volume.commit()

    entry["model_path"] = f"results/models/{eid}.pt"
    entry["raw_data_paths"] = {"per_sample_relerr": f"results/raw/{eid}_relerr.npy"}
    entry["meta"]["host"] = "modal-a10g"
    entry["_volume_paths"] = {
        "checkpoint": ckpt_path,
        "per_sample_relerr": raw_path_remote,
    }
    return entry


@app.local_entrypoint()
def sweep(arch: str = "", d: int = 0, k: int = 0, seed: int = -1,
          data_efficiency: bool = False, limit: int = 0):
    """Dispatch the sweep on Modal GPUs, pull checkpoints + raw files back."""
    arch_f = arch or None
    d_f = d or None
    k_f = k or None
    seed_f = seed if seed >= 0 else None

    configs = build_sweep(arch_f, d_f, k_f, seed_f, data_efficiency)
    db = load_db()
    # entry_exists now includes data_budget so data-efficiency runs are distinct
    # from the main sweep even when (arch, m, n, d, k, seed) match.
    pending = [c for c in configs
               if not entry_exists(db, c[0], c[1], c[2], c[3], c[4], c[5],
                                   "encoder", data_budget=c[6])]
    if limit > 0:
        pending = pending[:limit]

    _log(f"Modal sweep: {len(pending)} new configs of {len(configs)} total.")
    if not pending:
        print("All configs already completed.")
        return

    args_list = [tuple(c) for c in pending]

    for entry in train_modal.starmap(args_list):
        vpaths = entry.pop("_volume_paths", {})
        for remote, local in [
            (vpaths.get("checkpoint"), entry["model_path"]),
            (vpaths.get("per_sample_relerr"),
             entry["raw_data_paths"]["per_sample_relerr"]),
        ]:
            if remote is None:
                continue
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            with open(local, "wb") as f:
                for chunk in _volume.read_file(remote.removeprefix(VOL_MOUNT + "/")):
                    f.write(chunk)

        db = upsert(db, entry)
        save_db(db)
        mt = entry["metrics"]
        _log(f"{entry['id']} rel_err={mt['rel_err_mean']:.4f} "
             f"dead={mt['dead_frac']:.2%}")

    _log(f"Modal sweep finished: {len(pending)} entries written.")


# ---------------------------------------------------------------------------
# Local driver (no Modal)
# ---------------------------------------------------------------------------
def _save_local_artifacts(entry: dict, state_dict: dict, per_err: np.ndarray) -> dict:
    arch, m, n = entry["architecture"], entry["m"], entry["n"]
    d, k, seed = entry["d"], entry["k"], entry["seed"]
    budget = entry["training"]["data_budget"]

    mpath = model_path(arch, m, n, d, k, seed, data_budget=budget)
    Path(mpath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, mpath)

    rpath = raw_path(arch, m, n, d, k, seed, "encoder", "relerr.npy",
                     data_budget=budget)
    Path(rpath).parent.mkdir(parents=True, exist_ok=True)
    np.save(rpath, per_err)

    entry["model_path"] = mpath
    entry["raw_data_paths"] = {"per_sample_relerr": rpath}
    entry["meta"]["host"] = socket.gethostname()
    return entry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default=None,
                   choices=[None, "expander_tied", "dense_tied", "dense_warmtied",
                            "dense_randinit", "dense_indep",
                            "clustered_sparse", "pruned_retuned_dense"])
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data-efficiency", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "mps", "cuda"])
    args = p.parse_args()

    if args.smoke:
        configs = [("expander_tied", 64, 256, 4, 8, 0, 2000)]
    else:
        configs = build_sweep(args.arch, args.d, args.k, args.seed,
                              args.data_efficiency)

    train_acts = np.load("data/activations_train.npy")
    test_acts = np.load("data/activations_test.npy")

    total, done, new = len(configs), 0, 0
    _log(f"Local sweep: {total} configs.")

    for arch, m, n, d, k, seed, budget in configs:
        done += 1
        # Re-read DB on each iteration so concurrent writers' entries are
        # visible and we don't race (upsert_safe below is the write-side
        # concurrency guard).
        db_fresh = load_db()
        if entry_exists(db_fresh, arch, m, n, d, k, seed, "encoder",
                        data_budget=budget):
            continue
        if args.limit is not None and new >= args.limit:
            break
        entry, state_dict, per_err = train_single(
            arch, m, n, d, k, seed, budget,
            train_acts=train_acts, test_acts=test_acts,
            device=args.device, gpu_name=None,
        )
        entry = _save_local_artifacts(entry, state_dict, per_err)
        upsert_safe(entry)
        new += 1
        tl = entry["training"]["wall_clock_s"]
        mt = entry["metrics"]
        _log(f"[{done}/{total}] {entry['id']} "
             f"wall={tl:.1f}s rel_err={mt['rel_err_mean']:.4f} "
             f"dead={mt['dead_frac']:.2%}")

    _log(f"Sweep finished: {new} new entries, {done} configs considered.")


if __name__ == "__main__":
    main()
