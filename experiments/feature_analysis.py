"""Feature-level analysis per (arch, m, n, d, k) trained SAE.

Metrics computed per target model (seed=0), against a reference model
(default: dense_warmtied = Standard-SAE) at the same k:

  * Jaccard-activation novelty — per-feature best-Jaccard vs reference,
    fraction below thresholds in JACCARD_THRESHOLDS
  * Decoder-cosine novelty — per-feature max |cos| between decoder columns
    and any reference decoder column, fraction below thresholds
  * Firing rate — per-feature mean activation rate over 128k tokens
  * Target-token entropy — Shannon entropy (base 2) of the token-ID
    distribution each feature fires on

All per-feature arrays are saved as raw npy files; summary stats go into
the `features` dict of the encoder DB entry.

CPU only. ~2 min per target config.

CLI:
    python experiments/feature_analysis.py
    python experiments/feature_analysis.py --arch expander_tied --d 50
    python experiments/feature_analysis.py --k 64
    python experiments/feature_analysis.py --reference-arch dense_tied
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_EFF_K, JACCARD_THRESHOLDS, K_VALUES, M, N, N_TOKENS_FEATURE,
)
from db import get_entry, load_db, raw_path, save_db, upsert
from models import build

# Decoder-cosine novelty thresholds (analog of JACCARD_THRESHOLDS).
COSINE_THRESHOLDS = [0.1, 0.3]
# Expander fan-ins to analyse when --d is not specified.
EXPANDER_DS = [7, 30, 50, 100, 200]


def _encode_activations(model, acts: np.ndarray, batch_size: int = 1024
                        ) -> np.ndarray:
    """Return per-token feature-firing boolean matrix (T, n)."""
    model = model.eval()
    T = len(acts)
    out = np.zeros((T, model.n), dtype=bool)
    tensor = torch.from_numpy(acts).float()
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch = tensor[i:i + batch_size]
            _, h = model(batch)
            out[i:i + batch_size] = (h.abs() > 1e-10).numpy()
    return out


def _best_jaccard_per_feature(fire_a: np.ndarray, fire_b: np.ndarray) -> np.ndarray:
    counts_a = fire_a.sum(axis=0).astype(np.int64)
    counts_b = fire_b.sum(axis=0).astype(np.int64)
    fa = fire_a.astype(np.float32)
    fb = fire_b.astype(np.float32)
    inter = (fa.T @ fb).astype(np.int64)
    union = counts_a[:, None] + counts_b[None, :] - inter
    union = np.maximum(union, 1)
    return (inter / union).max(axis=1)


def _best_decoder_cos_per_feature(W_tgt: np.ndarray, W_ref: np.ndarray
                                  ) -> np.ndarray:
    """For each column of W_tgt, max |cos| against any column of W_ref.
    W shapes (m, n). Returns (n_tgt,)."""
    W_t = W_tgt / (np.linalg.norm(W_tgt, axis=0, keepdims=True) + 1e-12)
    W_r = W_ref / (np.linalg.norm(W_ref, axis=0, keepdims=True) + 1e-12)
    return np.abs(W_t.T @ W_r).max(axis=1)


def _per_feature_token_entropy(fire: np.ndarray, tokens: np.ndarray
                               ) -> np.ndarray:
    """Shannon entropy (bits) of each feature's firing-token distribution."""
    T, n = fire.shape
    vocab = int(tokens.max()) + 1
    out = np.zeros(n, dtype=np.float32)
    for j in range(n):
        mask = fire[:, j]
        if not mask.any():
            out[j] = 0.0
            continue
        cnt = np.bincount(tokens[mask], minlength=vocab)
        p = cnt[cnt > 0]
        p = p / p.sum()
        out[j] = float(-np.sum(p * np.log2(p)))
    return out


def _split_half_reliability(fire: np.ndarray) -> np.ndarray:
    N_samples = fire.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N_samples)
    half = N_samples // 2
    a, b = fire[perm[:half]], fire[perm[half:half * 2]]
    rates_a = a.mean(axis=0)
    rates_b = b.mean(axis=0)
    return np.minimum(rates_a, rates_b) / np.maximum(rates_a, rates_b).clip(min=1e-12)


def _load_model(entry: dict):
    arch = entry["architecture"]
    model = build(arch, m=entry["m"], n=entry["n"], d=entry["d"],
                  k=entry["k"], seed=entry["seed"])
    model.load_state_dict(torch.load(entry["model_path"], map_location="cpu",
                                     weights_only=True))
    return model


def _reference_bundle(db: list, arch: str, k: int, seed: int,
                      activations: np.ndarray) -> dict | None:
    """Fire matrix + decoder + entry for the reference model at given k."""
    entry = get_entry(db, arch, M, N, M, k, seed, "encoder")
    if entry is None:
        return None
    model = _load_model(entry)
    fire = _encode_activations(model, activations)
    W_dec = model.W_dec.detach().cpu().numpy()
    return {"entry": entry, "fire": fire, "W_dec": W_dec}


def analyse_one(tgt_entry: dict, reference: dict,
                tokens: np.ndarray, activations: np.ndarray) -> dict:
    arch = tgt_entry["architecture"]
    m, n, d, k, seed = (tgt_entry["m"], tgt_entry["n"], tgt_entry["d"],
                        tgt_entry["k"], tgt_entry["seed"])

    t0 = time.time()
    model = _load_model(tgt_entry)
    tgt_fire = _encode_activations(model, activations)
    W_tgt = model.W_dec.detach().cpu().numpy()
    t_encode = time.time() - t0

    t0 = time.time()
    best_jacc = _best_jaccard_per_feature(tgt_fire, reference["fire"])
    t_jacc = time.time() - t0

    t0 = time.time()
    best_cos = _best_decoder_cos_per_feature(W_tgt, reference["W_dec"])
    t_cos = time.time() - t0

    firing_rate = tgt_fire.mean(axis=0).astype(np.float32)

    t0 = time.time()
    tok_entropy = _per_feature_token_entropy(tgt_fire, tokens)
    t_ent = time.time() - t0

    rel = _split_half_reliability(tgt_fire)

    jac_novel = {f"jaccard_novel_frac_{int(t*10):02d}": float((best_jacc < t).mean())
                 for t in JACCARD_THRESHOLDS}
    cos_novel = {f"decoder_cos_novel_frac_{int(t*10):02d}": float((best_cos < t).mean())
                 for t in COSINE_THRESHOLDS}
    novel_mask = best_jacc < JACCARD_THRESHOLDS[0]
    shared_mask = best_jacc >= JACCARD_THRESHOLDS[-1]

    # Raw per-feature arrays.
    paths = {}
    for suffix, arr in [
        ("feature_jaccard.npy", best_jacc.astype(np.float32)),
        ("decoder_cos.npy",     best_cos.astype(np.float32)),
        ("firing_rate.npy",     firing_rate),
        ("token_entropy.npy",   tok_entropy),
    ]:
        p = raw_path(arch, m, n, d, k, seed, "encoder", suffix)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        paths[suffix.replace(".npy", "_path")] = p

    active = tok_entropy > 0
    token_entropy_median = (float(np.median(tok_entropy[active]))
                            if active.any() else 0.0)

    feats = {
        **jac_novel,
        **cos_novel,
        "firing_rate_median":   float(np.median(firing_rate)),
        "firing_rate_mean":     float(firing_rate.mean()),
        "dead_frac_test":       float((firing_rate == 0).mean()),
        "token_entropy_median": token_entropy_median,
        "token_entropy_mean":   float(tok_entropy.mean()),
        "split_half_novel":     float(rel[novel_mask].mean()) if novel_mask.any() else 0.0,
        "split_half_shared":    float(rel[shared_mask].mean()) if shared_mask.any() else 0.0,
        "reference_model":      reference["entry"]["id"],
        "n_tokens":             int(len(activations)),
        **paths,
    }
    feats["_timings"] = {
        "encode_s": round(t_encode, 1),
        "jaccard_s": round(t_jacc, 1),
        "cos_s": round(t_cos, 1),
        "token_entropy_s": round(t_ent, 1),
    }
    return feats


def _build_targets(args) -> list[tuple[str, int, int]]:
    """(arch, d, k) targets to analyse."""
    ks = [args.k] if args.k is not None else list(K_VALUES)
    targets = []
    baseline_d_default = [7, 50, 200]
    if args.arch in (None, "expander_tied"):
        ds = [args.d] if args.d is not None else EXPANDER_DS
        for d in ds:
            for k in ks:
                targets.append(("expander_tied", d, k))
    if args.arch in (None, "dense_tied"):
        for k in ks:
            targets.append(("dense_tied", M, k))
    if args.arch == "clustered_sparse":
        ds = [args.d] if args.d is not None else baseline_d_default
        for d in ds:
            for k in ks:
                targets.append(("clustered_sparse", d, k))
    if args.arch == "pruned_retuned_dense":
        ds = [args.d] if args.d is not None else baseline_d_default
        for d in ds:
            for k in ks:
                targets.append(("pruned_retuned_dense", d, k))
    return targets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-arch", type=str, default="dense_warmtied",
                   choices=["dense_tied", "dense_warmtied"])
    p.add_argument("--arch", type=str, default=None,
                   choices=[None, "expander_tied", "dense_tied",
                            "clustered_sparse", "pruned_retuned_dense"])
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    payload = np.load("data/tokens_and_acts.npz")
    tokens_all = payload["tokens"]
    activations_all = payload["activations"]

    acts = activations_all[:N_TOKENS_FEATURE]
    tokens = tokens_all.flatten() if tokens_all.ndim > 1 else tokens_all
    tokens = tokens[:N_TOKENS_FEATURE]

    targets = _build_targets(args)
    print(f"Feature analysis: {len(targets)} targets "
          f"(reference={args.reference_arch}, seed={args.seed})")

    ref_cache: dict[int, dict] = {}
    for i, (arch, d, k) in enumerate(targets, 1):
        if arch == args.reference_arch:
            print(f"[{i}/{len(targets)}] skip {arch} d={d} k={k} (== reference)")
            continue

        db = load_db()
        tgt_entry = get_entry(db, arch, M, N, d, k, args.seed, "encoder")
        if tgt_entry is None:
            print(f"[{i}/{len(targets)}] no-entry {arch} d={d} k={k}")
            continue

        if k not in ref_cache:
            t0 = time.time()
            ref = _reference_bundle(db, args.reference_arch, k, args.seed, acts)
            if ref is None:
                print(f"[{i}/{len(targets)}] no-reference {args.reference_arch} k={k}")
                continue
            ref_cache[k] = ref
            print(f"  reference {args.reference_arch} k={k} encoded in {time.time()-t0:.1f}s")

        feats = analyse_one(tgt_entry, ref_cache[k], tokens, acts)
        tgt_entry["features"] = feats
        db = upsert(db, tgt_entry)
        save_db(db)
        t = feats.pop("_timings", {})
        print(f"[{i}/{len(targets)}] {arch} d={d} k={k}  "
              f"jac<0.1={feats['jaccard_novel_frac_01']:.3f} "
              f"cos<0.3={feats['decoder_cos_novel_frac_03']:.3f} "
              f"dead={feats['dead_frac_test']:.3f} "
              f"tok_H={feats['token_entropy_median']:.2f} "
              f"({t.get('encode_s','?')}+{t.get('jaccard_s','?')}+{t.get('token_entropy_s','?')}s)")

    print("Feature analysis done.")


if __name__ == "__main__":
    main()
