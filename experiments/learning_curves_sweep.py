"""Re-run the DATA_EFF_* configs with per-step test-rel_err logging.

Produces one `<id>_learning_curve.npy` raw file per (arch, n, d, k, seed) at
results/raw/. Updates the corresponding DB entry's
`training.learning_curve_path` field in-place; the existing model checkpoint
and per-sample rel_err are left untouched (training is deterministic given
the same seed, so re-running is safe but not needed to overwrite artifacts).

Scope:
  - expander_tied,  n=4096, d in DATA_EFF_D ({7, 30, 100}), k=DATA_EFF_K (=64)
  - dense_tied,     n=4096, d=512,                          k=DATA_EFF_K (=64)
  - dense_warmtied, n=4096, d=512,                          k=DATA_EFF_K (=64)
  All seeds in SEEDS.

CLI:
    python experiments/learning_curves_sweep.py
    python experiments/learning_curves_sweep.py --eval-every 100
    python experiments/learning_curves_sweep.py --arch expander_tied --d 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_EFF_D, DATA_EFF_K, M, N, SEEDS, TRAIN_DEFAULTS,
)
from db import (
    load_db, save_db, get_entry, raw_path, upsert,
)
from models import build, train_sae

LOG_PATH = "results/learning_curves_sweep.log"


def _log(msg: str) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().isoformat()}  {msg}\n")
    print(msg, flush=True)


def _configs() -> list[tuple[str, int, int, int, int, int]]:
    """(arch, m, n, d, k, seed) tuples for the learning-curve sweep."""
    out = []
    for d in DATA_EFF_D:
        for seed in SEEDS:
            out.append(("expander_tied", M, N, d, DATA_EFF_K, seed))
    for arch in ("dense_tied", "dense_warmtied"):
        for seed in SEEDS:
            out.append((arch, M, N, M, DATA_EFF_K, seed))
    return out


def _run_one(arch: str, m: int, n: int, d: int, k: int, seed: int,
             train_acts: np.ndarray, test_acts: np.ndarray,
             eval_every: int) -> np.ndarray:
    """Retrain one config with curve logging. Returns the curve array."""
    hp = dict(TRAIN_DEFAULTS)
    acts = train_acts[:hp["data_budget"]]

    model = build(arch, m=m, n=n, d=d, k=k, seed=seed)
    _, info = train_sae(
        model, acts, test_acts=test_acts, eval_every=eval_every,
        steps=hp["steps"], batch_size=hp["batch_size"],
        lr_max=hp["lr_max"], lr_min=hp["lr_min"],
        grad_clip=hp["grad_clip"],
        resample_interval=hp["resample_interval"],
        device="cpu",
    )
    return info["learning_curve"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-every", type=int, default=100,
                   help="evaluate test rel_err every N training steps")
    p.add_argument("--arch", type=str, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    configs = _configs()
    if args.arch:
        configs = [c for c in configs if c[0] == args.arch]
    if args.d is not None:
        configs = [c for c in configs if c[3] == args.d]
    if args.seed is not None:
        configs = [c for c in configs if c[5] == args.seed]

    train_acts = np.load("data/activations_train.npy")
    test_acts = np.load("data/activations_test.npy")

    _log(f"Learning-curves sweep: {len(configs)} configs, eval_every={args.eval_every}.")

    new = 0
    for i, (arch, m, n, d, k, seed) in enumerate(configs, 1):
        curve_p = raw_path(arch, m, n, d, k, seed, "encoder", "learning_curve.npy")
        if Path(curve_p).exists():
            _log(f"[{i}/{len(configs)}] skip {Path(curve_p).name} (exists)")
            continue

        curve = _run_one(arch, m, n, d, k, seed,
                         train_acts, test_acts, args.eval_every)
        Path(curve_p).parent.mkdir(parents=True, exist_ok=True)
        np.save(curve_p, curve)

        # Update DB entry in-place.
        db = load_db()
        entry = get_entry(db, arch, m, n, d, k, seed, "encoder")
        if entry is not None:
            entry["training"]["learning_curve_path"] = curve_p
            db = upsert(db, entry)
            save_db(db)
            db_note = "db-updated"
        else:
            db_note = "no-db-entry"

        new += 1
        final = float(curve[-1, 1])
        _log(f"[{i}/{len(configs)}] {arch} d={d} k={k} seed={seed} "
             f"final_rel_err={final:.4f} ({db_note})")

    _log(f"Finished: {new} new curves, {len(configs) - new} skipped.")


if __name__ == "__main__":
    main()
