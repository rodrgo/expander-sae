"""Geometry diagnostics for trained decoders.

For each encoder DB entry, compute:

  * beta_j = sqrt(d) * ||w_j||_∞   → median / p95 / max across columns
  * mu_1(k; W) = max_j Σ_{ℓ ∈ T_k(j)} |⟨w_j, w_ℓ⟩|   (normalized columns,
    exclude self, top-k by absolute inner product)
  * ε_count(s) = max(0, 1 − m/(d·s))   evaluated at both s = k+1 and s = 2k
  * ε_greedy(s) = empirical stress-test max over (random subsets, greedy
                  collision-max restarts, top-overlap neighborhoods),
                  evaluated at both s = k+1 and s = 2k
  * R_count   = β_max^2 · ε_count(k+1) · (2k+1)
  * R_OMP     = β_max^2 · ε_greedy(k+1) · (2k+1)   (sufficient OMP condition)
  * R_id      = 2·β_max^2 · ε_greedy(2k)            (Theorem 1 identifiability ratio)

Pure NumPy, no training. Results go into `entry["geometry"]` on the encoder
entry; the mask pattern is read from `model.mask` for sparse archs or is
treated as all-ones for dense.

CLI:
    python experiments/geometry_diagnostics.py
    python experiments/geometry_diagnostics.py --arch expander_tied
    python experiments/geometry_diagnostics.py --n-greedy 100 --n-random 5000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import load_db, get_entry, upsert_safe
from models import build

LOG_PATH = "results/geometry_diagnostics.log"


def _log(msg: str) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().isoformat()}  {msg}\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def _load_decoder(entry: dict) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Return (W_dec, mask, d_effective) for an encoder entry.
    W_dec shape: (m, n), column-normalized. mask shape: (m, n) bool or None."""
    model = build(entry["architecture"], m=entry["m"], n=entry["n"],
                  d=entry["d"], k=entry["k"], seed=entry["seed"])
    model.load_state_dict(torch.load(entry["model_path"], map_location="cpu",
                                     weights_only=True))
    W = model.W_dec.detach().cpu().numpy().astype(np.float32)
    mask_np = None
    d_eff = entry["d"]
    if hasattr(model, "mask") and isinstance(model.mask, torch.Tensor):
        mask_np = (model.mask.detach().cpu().numpy() > 0)
        # use column support size as effective d (may differ from entry["d"]
        # if the mask construction didn't hit d exactly)
        d_eff = int(mask_np.sum(axis=0).max())
    return W, mask_np, d_eff


def _beta_stats(W: np.ndarray, d: int) -> dict:
    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    W_n = W / col_norms.clip(min=1e-12)
    col_inf = np.abs(W_n).max(axis=0)
    beta = np.sqrt(d) * col_inf
    return {
        "beta_median": float(np.median(beta)),
        "beta_p95":    float(np.percentile(beta, 95)),
        "beta_max":    float(beta.max()),
    }


def _mu_1(W: np.ndarray, k: int) -> float:
    """Normalized-column cumulative coherence."""
    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    W_n = W / col_norms.clip(min=1e-12)
    G = np.abs(W_n.T @ W_n)
    np.fill_diagonal(G, 0.0)
    # For each row, sum top-k entries.
    # np.partition with kth=-k places the k largest at the tail.
    if k >= G.shape[1]:
        return float(G.sum(axis=1).max())
    topk = np.partition(G, -k, axis=1)[:, -k:]
    sums = topk.sum(axis=1)
    return float(sums.max())


def _epsilon_count(m: int, d: int, s: int) -> float:
    return float(max(0.0, 1.0 - m / (d * s)))


def _epsilon_greedy(mask: np.ndarray, d: int, s: int,
                    n_random: int = 1000, n_greedy: int = 30,
                    seed: int = 0) -> dict:
    """Empirical stress test of expansion deficit at subset size `s`.
    Returns dict with the max per-method + the overall max."""
    rng = np.random.default_rng(seed)
    m, n = mask.shape
    if s > n:
        # Not enough columns to form a size-s subset.
        return {"epsilon_random": 0.0, "epsilon_greedy_restart": 0.0,
                "epsilon_top_overlap": 0.0, "epsilon_greedy": 0.0}
    mask_b = mask.astype(bool)

    def deficit(S_idx: np.ndarray | list[int]) -> float:
        covered = mask_b[:, list(S_idx)].any(axis=1).sum()
        return 1.0 - covered / (d * s)

    # Method 1: random subsets of size s
    eps_random = 0.0
    for _ in range(n_random):
        S = rng.choice(n, size=s, replace=False)
        eps_random = max(eps_random, deficit(S))

    # Method 2: greedy collision-maximizing restarts
    eps_greedy_restart = 0.0
    for _ in range(n_greedy):
        S = [int(rng.integers(n))]
        N_S = mask_b[:, S[0]].copy()
        for _step in range(s - 1):
            # for each candidate column c, count new rows it would add
            # new_rows[c] = sum over rows of (mask_b[:, c] & ~N_S)
            # vectorized: (mask_b & ~N_S[:, None]).sum(axis=0)
            new_rows = (mask_b & ~N_S[:, None]).sum(axis=0)
            new_rows[S] = m + 1  # forbid reuse
            j = int(new_rows.argmin())
            S.append(j)
            N_S |= mask_b[:, j]
        eps_greedy_restart = max(eps_greedy_restart, deficit(S))

    # Method 3: top-overlap neighborhoods
    # C[j, ℓ] = |supp(j) ∩ supp(ℓ)|
    M_f = mask_b.astype(np.int32)
    C = M_f.T @ M_f
    np.fill_diagonal(C, -1)
    eps_top_overlap = 0.0
    if s - 1 < n:
        for j in range(n):
            top = np.argpartition(-C[j], s - 2)[:s - 1]
            S = [j, *top.tolist()]
            eps_top_overlap = max(eps_top_overlap, deficit(S))

    eps_greedy = max(eps_random, eps_greedy_restart, eps_top_overlap)
    return {
        "epsilon_random":         float(eps_random),
        "epsilon_greedy_restart": float(eps_greedy_restart),
        "epsilon_top_overlap":    float(eps_top_overlap),
        "epsilon_greedy":         float(eps_greedy),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
SUPPORTED_ARCHS = {"expander_tied", "dense_tied", "dense_warmtied",
                   "dense_randinit", "clustered_sparse", "pruned_retuned_dense"}


def _mask_type_for(arch: str, has_mask: bool) -> str:
    if arch == "expander_tied":
        return "expander"
    if arch == "clustered_sparse":
        return "clustered"
    if arch == "pruned_retuned_dense":
        return "pruned_retuned"
    return "dense"


def analyse_one(entry: dict, n_random: int, n_greedy: int) -> dict:
    t0 = time.time()
    W, mask, d_eff = _load_decoder(entry)
    k = entry["k"]
    m, n = W.shape

    beta = _beta_stats(W, d_eff)
    t_beta = time.time() - t0

    t0 = time.time()
    mu1 = _mu_1(W, k)
    t_mu1 = time.time() - t0

    s_omp = k + 1
    s_id = 2 * k
    eps_count_omp = _epsilon_count(m, d_eff, s_omp)
    eps_count_id = _epsilon_count(m, d_eff, s_id)

    t0 = time.time()
    if mask is None:
        # Dense: every column covers all m rows. Closed form.
        mask_dense = np.ones((m, n), dtype=bool)
        eps_omp = _epsilon_greedy(mask_dense, d_eff, s_omp,
                                  n_random=min(n_random, 100),
                                  n_greedy=min(n_greedy, 5),
                                  seed=entry["seed"])
        eps_id = _epsilon_greedy(mask_dense, d_eff, s_id,
                                 n_random=min(n_random, 100),
                                 n_greedy=min(n_greedy, 5),
                                 seed=entry["seed"])
    else:
        eps_omp = _epsilon_greedy(mask, d_eff, s_omp,
                                  n_random=n_random, n_greedy=n_greedy,
                                  seed=entry["seed"])
        eps_id = _epsilon_greedy(mask, d_eff, s_id,
                                 n_random=n_random, n_greedy=n_greedy,
                                 seed=entry["seed"])
    t_eps = time.time() - t0

    R_count = beta["beta_max"] ** 2 * eps_count_omp * (2 * k + 1)
    R_OMP = beta["beta_max"] ** 2 * eps_omp["epsilon_greedy"] * (2 * k + 1)
    R_id = 2.0 * beta["beta_max"] ** 2 * eps_id["epsilon_greedy"]

    geom = {
        "mask_type": _mask_type_for(entry["architecture"], mask is not None),
        "d_effective": int(d_eff),
        **beta,
        "mu1_k": float(mu1),
        # OMP / coherence quantities (s = k+1)
        "epsilon_count": eps_count_omp,
        **eps_omp,
        "R_count": float(R_count),
        "R_OMP":   float(R_OMP),
        "R_est":   float(R_OMP),  # legacy alias
        "condition_active_count":  bool(R_count < 1),
        "condition_active_greedy": bool(R_OMP < 1),
        # Identifiability quantities (s = 2k)
        "epsilon_count_s2k":   eps_count_id,
        "epsilon_greedy_s2k":  float(eps_id["epsilon_greedy"]),
        "R_id":                float(R_id),
        "condition_active_id": bool(R_id < 1),
        "_timings": {"beta_s": round(t_beta, 2),
                     "mu1_s": round(t_mu1, 2),
                     "eps_s": round(t_eps, 2)},
    }
    return geom


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default=None, choices=[None, *SUPPORTED_ARCHS])
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, nargs="*")
    p.add_argument("--n", dest="n_only", type=int, default=None, nargs="*")
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--n-greedy", type=int, default=30)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    db = load_db()
    enc = [e for e in db if e["inference_method"] == "encoder"
           and "_b" not in e["id"]
           and e["architecture"] in SUPPORTED_ARCHS]

    n_filter = args.n_only if args.n_only else [4096]
    seed_filter = set(args.seed) if args.seed else None

    def keep(e):
        if args.arch and e["architecture"] != args.arch: return False
        if args.d is not None and e["d"] != args.d: return False
        if args.k is not None and e["k"] != args.k: return False
        if seed_filter is not None and e["seed"] not in seed_filter: return False
        if e["n"] not in n_filter: return False
        return True

    targets = [e for e in enc if keep(e)]
    _log(f"Geometry diagnostics: {len(targets)} targets "
         f"(n_random={args.n_random}, n_greedy={args.n_greedy})")

    done = 0
    for i, e in enumerate(targets, 1):
        if not args.force and e.get("geometry") and e["geometry"].get("R_est") is not None:
            continue
        if not e.get("model_path") or not os.path.exists(e["model_path"]):
            _log(f"[{i}/{len(targets)}] skip {e['id']} (missing model)")
            continue

        t_total = time.time()
        geom = analyse_one(e, args.n_random, args.n_greedy)
        e["geometry"] = geom
        upsert_safe(e)
        done += 1
        t = geom.pop("_timings", {})
        _log(f"[{i}/{len(targets)}] {e['architecture']} d={e['d']} k={e['k']} seed={e['seed']}  "
             f"β_max={geom['beta_max']:.2f} μ_1={geom['mu1_k']:.3f} "
             f"ε_c={geom['epsilon_count']:.3f} ε_g={geom['epsilon_greedy']:.3f} "
             f"R_est={geom['R_est']:.2f}  ({time.time()-t_total:.1f}s)")

    _log(f"Geometry done: {done} new/updated entries.")


if __name__ == "__main__":
    main()
