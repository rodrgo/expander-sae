"""Active-support collision diagnostic.

For each trained sparse SAE (Expander / Clustered-sparse / Pruned dense)
at the headline configuration, encode the held-out activations and read
off the empirical TopK support $S \subseteq [n]$ on each sample. Compute:

  * coverage $|\\Gamma(S)|$ = number of distinct neurons hit by the
    union of the $|S|$ selected columns' supports.
  * empirical deficit $1 - |\\Gamma(S)| / (d \\cdot |S|)$ -- how much
    of the worst-case neighbourhood the active support actually covers.
  * duplicate-edge count $\\sum_i \\max(0, \\deg_i(S) - 1)$ -- excess
    incidence on already-covered neurons (collisions).

Reports per-(arch, d, seed) median / p95 / max over the held-out set.
This is the data-dependent counterpart to the worst-case expansion
diagnostics in Appendix~\\ref{app:bench_geometry}: it tests the same
mechanism on the supports the SAE actually uses.

Output: results/active_support_collisions.json.
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
from db import load_db
from experiments.feature_analysis import _encode_activations, _load_model


SUPPORTED_ARCHS = ["expander_tied", "clustered_sparse", "pruned_retuned_dense"]


def _topk_supports(model, acts: np.ndarray, k: int,
                   batch_size: int = 1024) -> np.ndarray:
    """Return per-sample top-k support indices, shape (T, k) int32."""
    model = model.eval()
    T = len(acts)
    out = np.zeros((T, k), dtype=np.int32)
    tensor = torch.from_numpy(acts).float()
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch = tensor[i:i + batch_size]
            _, h = model(batch)
            # h is (B, n) signed dense; the active support is the indices
            # of the k entries with largest |h|.
            absh = h.abs()
            _, idx = absh.topk(k, dim=-1)
            out[i:i + batch_size] = idx.numpy().astype(np.int32)
    return out


def _collision_stats(model, acts: np.ndarray) -> dict:
    """Compute per-sample coverage + duplicate count using the model's mask."""
    if not hasattr(model, "mask"):
        # Pruned/clustered may also expose ``mask`` (binary). If not, the
        # support is read off the decoder weight magnitudes per column.
        W = model.W_dec.detach().cpu().numpy()
        mask_b = (np.abs(W) > 1e-12)
    else:
        mask_b = model.mask.detach().cpu().numpy().astype(bool)
    d = int(mask_b.sum(axis=0).max())
    k = model.k
    T = len(acts)
    supports = _topk_supports(model, acts, k)              # (T, k)
    coverage = np.zeros(T, dtype=np.int32)
    duplicates = np.zeros(T, dtype=np.int32)
    for t in range(T):
        S = supports[t]
        sub = mask_b[:, S]                                  # (m, k) bool
        edge_count = int(sub.sum())                         # = d*k for d-regular
        cov = int(sub.any(axis=1).sum())
        coverage[t] = cov
        duplicates[t] = edge_count - cov
    deficit = 1.0 - coverage.astype(np.float64) / (d * k)
    return {
        "d_effective": d,
        "k": k,
        "T": T,
        "coverage_mean":   float(coverage.mean()),
        "coverage_median": float(np.median(coverage)),
        "deficit_mean":    float(deficit.mean()),
        "deficit_median":  float(np.median(deficit)),
        "deficit_p95":     float(np.quantile(deficit, 0.95)),
        "deficit_max":     float(deficit.max()),
        "duplicates_mean":   float(duplicates.mean()),
        "duplicates_median": float(np.median(duplicates)),
        "duplicates_p95":    float(np.quantile(duplicates, 0.95)),
        "duplicates_max":    float(duplicates.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ds", type=int, nargs="+", default=[7, 50, 200])
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--archs", type=str, nargs="+", default=SUPPORTED_ARCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out-json", type=str,
                   default="results/active_support_collisions.json")
    p.add_argument("--n-acts", type=int, default=5000,
                   help="Number of held-out activations to score over.")
    args = p.parse_args()

    payload = np.load("data/tokens_and_acts.npz")
    acts = payload["activations"][:args.n_acts]
    print(f"loaded {len(acts)} held-out activations of dim {acts.shape[1]}")

    db = load_db()
    out: list[dict] = []
    for arch in args.archs:
        for d in args.ds:
            for seed in args.seeds:
                # Find matching encoder entry.
                cand = [e for e in db
                        if e.get("architecture") == arch
                        and e.get("m") == M and e.get("n") == N
                        and e.get("d") == d and e.get("k") == args.k
                        and e.get("seed") == seed
                        and e.get("inference_method") == "encoder"]
                if not cand:
                    print(f"  skip {arch} d={d} k={args.k} seed={seed} "
                          f"(no DB entry)")
                    continue
                entry = cand[0]
                if not entry.get("model_path"):
                    print(f"  skip {arch} d={d} k={args.k} seed={seed} "
                          f"(no model_path)")
                    continue
                t0 = time.time()
                try:
                    model = _load_model(entry)
                except Exception as exc:  # pragma: no cover -- environment-dep.
                    print(f"  skip {arch} d={d} k={args.k} seed={seed} "
                          f"(load failed: {exc})")
                    continue
                stats = _collision_stats(model, acts)
                stats.update({"architecture": arch, "d": d,
                              "k": args.k, "seed": seed,
                              "n_acts": int(len(acts)),
                              "_secs": round(time.time() - t0, 1)})
                out.append(stats)
                print(f"  {arch:<22} d={d:>3} seed={seed} "
                      f"deficit median={stats['deficit_median']:.3f} "
                      f"p95={stats['deficit_p95']:.3f} "
                      f"dup median={stats['duplicates_median']:.0f} "
                      f"({stats['_secs']}s)")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json} ({len(out)} entries)")


if __name__ == "__main__":
    main()
