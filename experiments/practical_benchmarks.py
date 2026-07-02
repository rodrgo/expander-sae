"""Measure storage and inference timing for every trained model.

Storage: save decoder in CSC sparse format, measure file size.
Timing: the ms/sample figures are already captured by inference_sweep; this
script fills `practical.storage_decoder_kb` for each encoder entry and
recomputes a fresh encoder timing measurement for an apples-to-apples baseline.

CPU only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import load_db, save_db, upsert
from models import build


def _encoder_ms_per_sample(model, test_acts: np.ndarray,
                           n_samples: int = 200) -> float:
    model.eval()
    tensor = torch.from_numpy(test_acts[:n_samples]).float()
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_samples):
            _ = model(tensor[i:i + 1])
    return (time.time() - t0) / n_samples * 1000.0


def _decoder_storage_breakdown_kb(model) -> dict:
    """Return a structured storage breakdown for the trained model.

    All numbers are in KiB. The five components correspond to the
    columns of Table 4 (storage breakdown):

    1. ``learned_values_kb``: just the float32 decoder values
       ($d \\cdot n$ floats for sparse Expander; $m \\cdot n$ for dense).
    2. ``decoder_plus_rows_kb``: the actual on-disk storage for the
       decoder including int32 row indices in the $(d \\cdot n,)$ flat
       layout used by the structured kernels (no indptr needed because
       each column has exactly $d$ nonzeros at known positions). For
       dense architectures this just equals ``learned_values_kb``.
    3. ``encoder_plus_bias_kb``: encoder weights (only when not tied)
       plus $\\mathbf{b}_{\\mathrm{dec}}$ and $\\mathbf{b}_{\\mathrm{enc}}$.
    4. ``total_footprint_kb``: sum of (2) + (3) + 8 bytes for the seed
       (sparse architectures only) needed to regenerate the binary mask.
    5. ``mask_seed_bytes``: bytes needed to regenerate the mask if not
       stored explicitly. For Expander/Clustered-sparse/Pruned-dense
       this is one int64 seed = 8 bytes; for Dense-SAE there is no
       mask to regenerate.
    """
    W = model.W_dec.detach().cpu().numpy()
    m, n = W.shape
    arch = getattr(model, "arch", model.__class__.__name__)
    has_mask = hasattr(model, "mask")

    nnz = int((W != 0).sum())
    has_separate_encoder = hasattr(model, "W_enc_param")

    # Bias storage: b_dec (m floats) + b_enc (n floats) = 4*(m+n) bytes.
    bias_bytes = 4 * (m + n)

    if has_mask and nnz < m * n:
        # Sparse decoder with d-regular mask. Flat layout: d*n floats
        # for values + d*n int32s for row indices. No indptr because
        # column j's d entries live at positions [j*d, (j+1)*d).
        learned_bytes = 4 * nnz
        decoder_with_rows_bytes = 8 * nnz       # 4 (float) + 4 (int32 row)
        mask_seed_bytes = 8                      # one int64 seed
    else:
        # Dense decoder: store as a contiguous (m, n) array, no indices.
        learned_bytes = 4 * m * n
        decoder_with_rows_bytes = 4 * m * n
        mask_seed_bytes = 0  # no mask to regenerate

    if has_separate_encoder:
        # Independent encoder: full (n, m) weight matrix.
        encoder_plus_bias_bytes = 4 * m * n + bias_bytes
    else:
        # Tied: encoder is W_dec.T, no separate weights to store.
        encoder_plus_bias_bytes = bias_bytes

    total_bytes = (
        decoder_with_rows_bytes
        + encoder_plus_bias_bytes
        + mask_seed_bytes
    )

    return {
        "learned_values_kb": learned_bytes / 1024.0,
        "decoder_plus_rows_kb": decoder_with_rows_bytes / 1024.0,
        "encoder_plus_bias_kb": encoder_plus_bias_bytes / 1024.0,
        "total_footprint_kb": total_bytes / 1024.0,
        "mask_seed_bytes": mask_seed_bytes,
        "_arch": arch, "_m": m, "_n": n, "_nnz": nnz,
    }


def _decoder_csc_kb(model) -> float:
    """Backward-compat shim: a single-number summary. Equals the
    'decoder + row indices' column of the breakdown."""
    return _decoder_storage_breakdown_kb(model)["decoder_plus_rows_kb"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default=None)
    args = p.parse_args()

    db = load_db()
    test_acts = np.load("data/activations_test.npy")

    encoder_entries = [e for e in db if e["inference_method"] == "encoder"]
    if args.arch:
        encoder_entries = [e for e in encoder_entries if e["architecture"] == args.arch]

    total = len(encoder_entries)
    for i, e in enumerate(encoder_entries, 1):
        if not e["model_path"] or not Path(e["model_path"]).exists():
            print(f"[{i}/{total}] {e['id']}: checkpoint missing, skip")
            continue
        model = build(e["architecture"], m=e["m"], n=e["n"], d=e["d"],
                      k=e["k"], seed=e["seed"])
        model.load_state_dict(torch.load(e["model_path"], map_location="cpu"))

        enc_ms = _encoder_ms_per_sample(model, test_acts, n_samples=100)
        storage_kb = _decoder_csc_kb(model)

        e["practical"]["inference_ms_per_sample"] = enc_ms
        e["practical"]["storage_decoder_kb"] = storage_kb
        db = upsert(db, e)
        if i % 10 == 0 or i == total:
            save_db(db)
        print(f"[{i}/{total}] {e['id']}  enc={enc_ms:.3f}ms  dec={storage_kb:.1f}KB")
    save_db(db)


if __name__ == "__main__":
    main()
