"""Database read/write/query helpers for benchmark_db.json.

Each entry is one fully specified evaluation identified by
make_id(arch, m, n, d, k, seed, method). Writes are atomic (write to tmp,
then rename) so a crash mid-write never corrupts the DB.

`upsert_safe(entry)` is the concurrency-safe variant: takes a file lock,
re-reads the DB from disk, upserts, and writes back. Use it when multiple
writers (e.g. parallel inference workers, or inference + CE-loss running at
once) might race. `upsert(db, entry)` is the in-memory single-writer variant.
"""
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

DB_PATH = "results/benchmark_db.json"
LOCK_PATH = "results/benchmark_db.lock"


def load_db(path: str = DB_PATH) -> list[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_db(db: list[dict], path: str = DB_PATH) -> None:
    """Atomic write: tmp file + rename."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=Path(path).parent, prefix=".benchmark_db.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(db, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


DEFAULT_DATA_BUDGET = 200_000

# TopK convention used by the trained encoder. ``unsigned`` means
# ``pre.topk(k)`` (the published-literature standard, Gao et al. 2024) and
# is the default for backward-compatibility with existing entries.
# ``signed`` means ``pre.abs().topk(k)``-style picks (allows negative
# coefficients in the encoder output). The default value is omitted from
# generated IDs so existing entries remain matched without migration; only
# non-default values get a disambiguating suffix.
DEFAULT_TOPK_MODE = "unsigned"


def make_id(arch: str, m: int, n: int, d: int, k: int, seed: int,
            method: str, data_budget: int | None = None,
            topk_mode: str = DEFAULT_TOPK_MODE) -> str:
    """Canonical id. ``_b{budget}`` suffix only when budget !=
    DEFAULT_DATA_BUDGET, and ``_topk{mode}`` suffix only when topk_mode !=
    DEFAULT_TOPK_MODE, so main-sweep ids stay as
    ``{arch}_m{m}_n{n}_d{d}_k{k}_seed{seed}_{method}`` and only
    non-default variants get a disambiguator."""
    base = f"{arch}_m{m}_n{n}_d{d}_k{k}_seed{seed}_{method}"
    if data_budget is not None and data_budget != DEFAULT_DATA_BUDGET:
        base = f"{base}_b{data_budget}"
    if topk_mode != DEFAULT_TOPK_MODE:
        base = f"{base}_topk{topk_mode}"
    return base


def entry_exists(db: list[dict], arch: str, m: int, n: int, d: int, k: int,
                 seed: int, method: str, data_budget: int | None = None,
                 topk_mode: str = DEFAULT_TOPK_MODE) -> bool:
    target = make_id(arch, m, n, d, k, seed, method, data_budget, topk_mode)
    return any(e["id"] == target for e in db)


def get_entry(db: list[dict], arch: str, m: int, n: int, d: int, k: int,
              seed: int, method: str,
              data_budget: int | None = None,
              topk_mode: str = DEFAULT_TOPK_MODE) -> Optional[dict]:
    target = make_id(arch, m, n, d, k, seed, method, data_budget, topk_mode)
    for e in db:
        if e["id"] == target:
            return e
    return None


def upsert(db: list[dict], entry: dict) -> list[dict]:
    """Replace existing entry with same id, or append. In-memory only."""
    target = entry["id"]
    for i, e in enumerate(db):
        if e["id"] == target:
            db[i] = entry
            return db
    db.append(entry)
    return db


@contextmanager
def _db_lock(lock_path: str = LOCK_PATH, timeout: float = 30.0):
    """File-lock context. All DB-mutating sections should run inside this."""
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        # Blocking exclusive lock. flock returns when acquired.
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _lock_path_for(db_path: str) -> str:
    """Sibling .lock file for a given DB path. Preserves legacy
    `benchmark_db.lock` naming when the DB ends in `.json`."""
    p = Path(db_path)
    if p.suffix == ".json":
        return str(p.with_suffix(".lock"))
    return str(p) + ".lock"


def upsert_safe(entry: dict, path: str = DB_PATH) -> None:
    """Concurrency-safe upsert: lock + re-read + merge + atomic write.

    Use this when multiple processes (inference workers, concurrent sweeps)
    might mutate the DB at the same time. The lock is on a sibling .lock file
    derived from the DB path, so distinct DBs don't contend on one lock.
    """
    with _db_lock(_lock_path_for(path)):
        db = load_db(path)
        db = upsert(db, entry)
        save_db(db, path)


def query(db: list[dict], **filters: Any) -> list[dict]:
    """Filter entries by top-level equality. List values expand to set-membership."""
    out = db
    for key, val in filters.items():
        if isinstance(val, list):
            out = [e for e in out if e.get(key) in val]
        else:
            out = [e for e in out if e.get(key) == val]
    return out


def model_path(arch: str, m: int, n: int, d: int, k: int, seed: int,
               root: str = "results/models",
               data_budget: int | None = None,
               topk_mode: str = DEFAULT_TOPK_MODE) -> str:
    suffix = ""
    if data_budget is not None and data_budget != DEFAULT_DATA_BUDGET:
        suffix = f"_b{data_budget}"
    if topk_mode != DEFAULT_TOPK_MODE:
        suffix = f"{suffix}_topk{topk_mode}"
    return f"{root}/{arch}_m{m}_n{n}_d{d}_k{k}_seed{seed}{suffix}.pt"


def raw_path(arch: str, m: int, n: int, d: int, k: int, seed: int, method: str,
             suffix: str, root: str = "results/raw",
             data_budget: int | None = None,
             topk_mode: str = DEFAULT_TOPK_MODE) -> str:
    eid = make_id(arch, m, n, d, k, seed, method, data_budget, topk_mode)
    return f"{root}/{eid}_{suffix}"


def new_entry_skeleton(arch: str, m: int, n: int, d: int, k: int, seed: int,
                       method: str, train_info: dict,
                       data_budget: int,
                       topk_mode: str = DEFAULT_TOPK_MODE) -> dict:
    """Empty schema with nulls everywhere — fill in as experiments complete."""
    return {
        "id": make_id(arch, m, n, d, k, seed, method, data_budget, topk_mode),
        "architecture": arch,
        "m": m, "n": n, "d": d, "k": k, "seed": seed,
        "inference_method": method,
        "topk_mode": topk_mode,
        "training": {
            "steps": train_info.get("steps"),
            "data_budget": data_budget,
            "wall_clock_s": train_info.get("wall_clock_s"),
            "gpu": train_info.get("gpu"),
            "lr_max": train_info.get("lr_max"),
            "lr_min": train_info.get("lr_min"),
            "batch_size": train_info.get("batch_size"),
            "grad_clip": train_info.get("grad_clip"),
            "resample_interval": train_info.get("resample_interval"),
        },
        "metrics": {
            "rel_err_mean": None,
            "rel_err_std": None,
            "rel_err_median": None,
            "explained_var": None,
            "dead_frac": None,
            "l0_mean": None,
            "n_test_samples": None,
            "ce_recovered": None,
            "ce_clean": None,
            "ce_zero": None,
            "ce_reconstructed": None,
            "n_ce_sequences": None,
        },
        "practical": {
            "unique_params": None,
            "decoder_params": None,
            "encoder_params": None,
            "storage_decoder_kb": None,
            "inference_ms_per_sample": None,
        },
        "features": {},
        "raw_data_paths": {},
        "model_path": None,
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "host": None,
            "notes": "",
        },
    }
