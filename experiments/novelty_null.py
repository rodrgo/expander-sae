"""Firing-rate-matched novelty null for Expander SAE features.

For each Expander feature j with firing count c_j, samples N random firing
sets of size c_j from the held-out activation set, computes the best
Jaccard against the dense reference fire matrix, and emits the per-feature
null mean and the per-(d, k, decile) novelty-vs-null comparison.

Outputs:
  results/novelty_null.json   — summary of observed vs null novelty fractions
                                 stratified by firing-rate decile, per (d, k).
  results/figures/novelty_null.pdf  — figure used in the paper.

This script reuses ``_encode_activations`` and ``_best_jaccard_per_feature``
from experiments/feature_analysis.py for parity with the production novelty
metric.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import M, N, N_TOKENS_FEATURE
from db import get_entry, load_db
from experiments.feature_analysis import (
    _best_jaccard_per_feature, _encode_activations, _load_model,
)


def _null_jaccard_by_count(counts: np.ndarray, fire_ref: np.ndarray,
                           n_null: int, n_bins: int, rng) -> np.ndarray:
    """Per-feature firing-rate-matched null mean of best-Jaccard.

    Strategy: bin features by firing-count into ``n_bins`` quantile bins.
    For each bin, sample ``n_null`` random firing sets at the bin's
    representative count (median) and compute the mean of best-Jaccard
    against ``fire_ref``. Each feature is assigned its bin's mean.
    Reduces compute from O(unique_counts * n_null * matmul) to
    O(n_bins * n_null * matmul).
    """
    T, n_ref = fire_ref.shape
    cnt_ref = fire_ref.sum(axis=0).astype(np.int64)
    fa_ref = fire_ref.astype(np.float32)

    counts = counts.astype(np.int64)
    out = np.zeros(counts.shape[0], dtype=np.float32)
    alive = counts > 0
    if alive.sum() == 0:
        return out

    # Build quantile bins on alive firing counts.
    alive_idx = np.flatnonzero(alive)
    rates = counts[alive_idx]
    edges = np.unique(np.quantile(rates, np.linspace(0, 1, n_bins + 1)))
    edges[-1] = edges[-1] + 1  # make right edge inclusive
    bin_assign = np.digitize(rates, edges) - 1  # 0..n_bins-1
    bin_assign = np.clip(bin_assign, 0, len(edges) - 2)

    for b in range(len(edges) - 1):
        members = alive_idx[bin_assign == b]
        if members.size == 0:
            continue
        c_rep = int(np.median(counts[members]))
        if c_rep <= 0:
            continue
        sets = np.zeros((n_null, T), dtype=np.float32)
        for r in range(n_null):
            idx = rng.choice(T, size=c_rep, replace=False)
            sets[r, idx] = 1.0
        inter = (sets @ fa_ref).astype(np.int64)             # (n_null, n_ref)
        union = c_rep + cnt_ref[None, :] - inter
        union = np.maximum(union, 1)
        jac = inter / union
        best = jac.max(axis=1)
        out[members] = float(best.mean())
    return out


def analyse_pair(d: int, k: int, seed: int, db: list,
                 acts: np.ndarray, n_null: int = 20, n_bins: int = 40,
                 rng_seed: int = 0,
                 target_arch: str = "expander_tied",
                 ref_arch: str = "dense_tied",
                 ref_d: int | None = None,
                 ref_seed: int | None = None) -> dict | None:
    rng = np.random.default_rng(rng_seed)
    T = len(acts)

    tgt = get_entry(db, target_arch, M, N, d, k, seed, "encoder")
    ref = get_entry(db, ref_arch, M, N,
                    ref_d if ref_d is not None else M, k,
                    ref_seed if ref_seed is not None else seed,
                    "encoder")
    if tgt is None or ref is None:
        return None

    t0 = time.time()
    tgt_model = _load_model(tgt)
    ref_model = _load_model(ref)
    tgt_fire = _encode_activations(tgt_model, acts)  # (T, n)
    ref_fire = _encode_activations(ref_model, acts)
    t_encode = time.time() - t0

    counts = tgt_fire.sum(axis=0).astype(np.int64)
    firing_rate = counts.astype(np.float32) / T

    t0 = time.time()
    obs = _best_jaccard_per_feature(tgt_fire, ref_fire)  # (n,)
    t_obs = time.time() - t0

    t0 = time.time()
    null_means = _null_jaccard_by_count(counts, ref_fire, n_null, n_bins, rng)
    t_null = time.time() - t0

    # Decile stratification by firing-rate (drop dead features).
    alive = counts > 0
    if alive.sum() == 0:
        deciles = []
    else:
        rates = firing_rate[alive]
        edges = np.quantile(rates, np.linspace(0, 1, 11))
        deciles = []
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            mask = alive & (firing_rate >= lo) & (firing_rate <= hi)
            if mask.sum() == 0:
                continue
            deciles.append({
                "decile": i + 1,
                "rate_lo": float(lo),
                "rate_hi": float(hi),
                "n_features": int(mask.sum()),
                "obs_novel_frac": float((obs[mask] < 0.1).mean()),
                "null_novel_frac": float((null_means[mask] < 0.1).mean()),
                "obs_mean":  float(obs[mask].mean()),
                "null_mean": float(null_means[mask].mean()),
            })

    return {
        "d": d, "k": k, "seed": seed,
        "T": int(T),
        "n_total": int(len(counts)),
        "n_alive": int(alive.sum()),
        "obs_novel_frac": float((obs < 0.1).mean()),
        "null_novel_frac": float((null_means < 0.1).mean()),
        "deciles": deciles,
        "_timings": {"encode": round(t_encode, 1),
                     "obs_jacc": round(t_obs, 1),
                     "null_jacc": round(t_null, 1)},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ds", type=int, nargs="+", default=[7, 50, 200])
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-null", type=int, default=20)
    p.add_argument("--n-bins", type=int, default=40)
    p.add_argument("--target-arch", type=str, default="expander_tied")
    p.add_argument("--ref-arch", type=str, default="dense_tied")
    p.add_argument("--ref-d", type=int, default=None,
                   help="Reference d (defaults to M for dense baselines).")
    p.add_argument("--ref-seed", type=int, default=None,
                   help="Reference seed (defaults to target seed).")
    p.add_argument("--out-json", type=str, default="results/novelty_null.json")
    args = p.parse_args()

    payload = np.load("data/tokens_and_acts.npz")
    acts = payload["activations"][:N_TOKENS_FEATURE]

    db = load_db()
    out: list[dict] = []
    for d in args.ds:
        print(f"==> d={d}, k={args.k}, seed={args.seed}")
        rec = analyse_pair(d, args.k, args.seed, db, acts,
                           n_null=args.n_null, n_bins=args.n_bins,
                           rng_seed=args.seed,
                           target_arch=args.target_arch,
                           ref_arch=args.ref_arch,
                           ref_d=args.ref_d,
                           ref_seed=args.ref_seed)
        if rec is None:
            print(f"   skip: no entry for d={d}, k={args.k}, seed={args.seed}")
            continue
        out.append(rec)
        print(f"   obs_novel={rec['obs_novel_frac']:.3f} "
              f"null_novel={rec['null_novel_frac']:.3f} "
              f"({rec['_timings']})")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
