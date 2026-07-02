"""CE-loss recovered for trained SAEs.

Splices each SAE's reconstruction into the layer-3 residual stream of
Pythia-70M and measures next-token CE over `N_CE_SEQUENCES` Pile sequences.

Formula (E1 from plan_chatgpt.md):

    ce_recovered = (ce_zero - ce_recon) / (ce_zero - ce_clean)

where
  ce_clean = CE of the unmodified LM,
  ce_zero  = CE with the layer-3 activation zeroed,
  ce_recon = CE with the layer-3 activation replaced by the SAE reconstruction.

Entry points:
  - `modal run experiments/ce_loss_sweep.py::main` (GPU, original pipeline).
  - `python experiments/ce_loss_sweep.py --help`  (CPU, local).

Both paths compute ce_clean and ce_zero once per run (they don't depend on
the SAE) and cache them; only ce_recon is computed per-SAE.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CE_CRITICAL_CONFIGS, CE_SEQ_LENGTH, N_CE_SEQUENCES
from db import (
    load_db, entry_exists, get_entry, make_id, upsert_safe,
    new_entry_skeleton, DEFAULT_DATA_BUDGET,
)

APP_NAME = "mech-expander-ce-loss"
VOLUME_NAME = "mech-expander-cache"
VOL_MOUNT = "/cache"
LAYER = 3
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
LOG_PATH = "results/ce_loss_sweep.log"


def _log(msg: str) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().isoformat()}  {msg}\n")
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# CE computation primitives (shared by Modal and local paths)
# ---------------------------------------------------------------------------
def _unwrap_and_rewrap(out):
    """Handle both transformers-4.x (tuple output) and transformers-5.x
    (bare tensor output) from GPTNeoX decoder layers.

    Returns (hidden_tensor, rewrap_fn) where rewrap_fn takes a replacement
    tensor and returns the same shape the model expects."""
    import torch
    if isinstance(out, tuple):
        def rewrap(new_h):
            return (new_h,) + out[1:]
        return out[0], rewrap
    return out, lambda new_h: new_h


def _zero_hook(_mod, _inp, out):
    import torch
    h, rewrap = _unwrap_and_rewrap(out)
    return rewrap(torch.zeros_like(h))


def _recon_hook(sae):
    """Factory: forward hook replacing activations with sae(h) reconstructions.
    sae must already be on the correct device."""
    import torch
    def hook(_mod, _inp, out):
        h, rewrap = _unwrap_and_rewrap(out)
        B, S, H = h.shape
        with torch.no_grad():
            recon, _ = sae(h.reshape(-1, H))
        return rewrap(recon.reshape(B, S, H).to(h.dtype))
    return hook


def _ce_weighted(lm, ids_list, device, hook_module=None, hook_fn=None
                 ) -> tuple[float, int]:
    """Token-weighted mean CE over `ids_list`, optionally with a hook.
    Returns (ce_mean, n_tokens)."""
    import torch
    tot, n_tok = 0.0, 0
    handle = None
    if hook_module is not None and hook_fn is not None:
        handle = hook_module.register_forward_hook(hook_fn)
    try:
        for ids in ids_list:
            ids = ids.to(device)
            with torch.no_grad():
                loss = float(lm(input_ids=ids, labels=ids).loss)
            nt = ids.shape[1]
            tot += loss * nt
            n_tok += nt
    finally:
        if handle is not None:
            handle.remove()
    return tot / max(n_tok, 1), n_tok


def _tokenize_pile(tokenizer, n_sequences: int, max_length: int):
    """Fetch first `n_sequences` usable Pile texts, tokenize, return a list
    of (1, L) int tensors."""
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    ids_list = []
    for item in ds:
        text = (item.get("text") or "").strip()
        if not text or len(text) < 50:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=False)["input_ids"]
        if ids.shape[1] < 4:
            continue
        ids_list.append(ids)
        if len(ids_list) >= n_sequences:
            break
    return ids_list


# ---------------------------------------------------------------------------
# Modal (GPU) path
# ---------------------------------------------------------------------------
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "numpy", "scipy",
        "transformers==4.44.0", "datasets", "accelerate", "tqdm", "zstandard",
    )
    .add_local_python_source("config", "db", "models", "inference")
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=1800,
              volumes={VOL_MOUNT: volume}, max_containers=10)
def ce_one(arch: str, m: int, n: int, d: int, k: int, seed: int,
           volume_ckpt_path: str, n_sequences: int = N_CE_SEQUENCES,
           max_length: int = CE_SEQ_LENGTH,
           ce_clean: float | None = None,
           ce_zero: float | None = None) -> dict:
    """Compute CE triple for one SAE (GPU). Optionally receives cached
    (ce_clean, ce_zero) to avoid recomputation per worker."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models import build

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = build(arch, m=m, n=n, d=d, k=k, seed=seed)
    sae.load_state_dict(torch.load(volume_ckpt_path, map_location="cpu"))
    sae = sae.to(device).eval()

    lm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    ids_list = _tokenize_pile(tok, n_sequences, max_length)
    target = lm.gpt_neox.layers[LAYER]

    if ce_clean is None:
        ce_clean, _ = _ce_weighted(lm, ids_list, device)
    if ce_zero is None:
        ce_zero, _ = _ce_weighted(lm, ids_list, device,
                                  hook_module=target, hook_fn=_zero_hook)
    ce_recon, n_tok = _ce_weighted(lm, ids_list, device,
                                   hook_module=target, hook_fn=_recon_hook(sae))
    denom = ce_zero - ce_clean
    ce_recovered = (ce_zero - ce_recon) / denom if abs(denom) > 1e-8 else float("nan")

    return {
        "arch": arch, "m": m, "n": n, "d": d, "k": k, "seed": seed,
        "ce_clean": ce_clean,
        "ce_zero": ce_zero,
        "ce_reconstructed": ce_recon,
        "ce_recovered": ce_recovered,
        "n_tokens": n_tok,
        "n_sequences": n_sequences,
    }


def _is_critical(arch: str, n: int, d: int) -> bool:
    return any(arch == a and n == nn and d == dd
               for (a, nn, dd) in CE_CRITICAL_CONFIGS)


@app.local_entrypoint()
def main(priority: int = 0):
    """priority: 0 = all encoder entries at seed 0 + critical seed 1.
                 1 = critical configs only (seed 0 + 1)."""
    db = load_db()
    enc = [e for e in db
           if e["inference_method"] == "encoder" and "_b" not in e["id"]]

    jobs = []
    for e in enc:
        arch, m, n, d, k, seed = (e["architecture"], e["m"], e["n"],
                                  e["d"], e["k"], e["seed"])
        is_crit = _is_critical(arch, n, d)
        if priority == 1 and not is_crit:
            continue
        allowed_seeds = {0}
        if is_crit:
            allowed_seeds.add(1)
        if seed not in allowed_seeds:
            continue
        budget = e["training"].get("data_budget")
        if entry_exists(db, arch, m, n, d, k, seed, "ce_encoder",
                        data_budget=budget):
            continue
        eid = make_id(arch, m, n, d, k, seed, "encoder", data_budget=budget)
        volume_ckpt = f"{VOL_MOUNT}/models/{eid}.pt"
        jobs.append((arch, m, n, d, k, seed, volume_ckpt))

    print(f"Dispatching {len(jobs)} CE evaluations...")
    if not jobs:
        return

    for r in ce_one.starmap(jobs):
        _persist(db, r)

    print(f"Done: {len(jobs)} CE entries written.")


# ---------------------------------------------------------------------------
# DB write helper (shared)
# ---------------------------------------------------------------------------
def _persist(db, r: dict) -> None:
    """Upsert ce_encoder entry + mirror onto encoder row. Expects r to carry
    arch/m/n/d/k/seed + the four CE fields."""
    arch, m, n, d, k, seed = r["arch"], r["m"], r["n"], r["d"], r["k"], r["seed"]
    base = get_entry(db, arch, m, n, d, k, seed, "encoder") or {}
    budget = base.get("training", {}).get("data_budget", DEFAULT_DATA_BUDGET)
    new = new_entry_skeleton(
        arch, m, n, d, k, seed, "ce_encoder",
        train_info=base.get("training", {}),
        data_budget=budget,
    )
    new["metrics"].update({
        "ce_clean": r["ce_clean"],
        "ce_zero": r["ce_zero"],
        "ce_reconstructed": r["ce_reconstructed"],
        "ce_recovered": r["ce_recovered"],
        "n_ce_sequences": r["n_sequences"],
    })
    new["model_path"] = base.get("model_path")
    upsert_safe(new)

    # Mirror onto encoder entry so plots can read ce_* directly.
    if base:
        base["metrics"]["ce_clean"] = r["ce_clean"]
        base["metrics"]["ce_zero"] = r["ce_zero"]
        base["metrics"]["ce_reconstructed"] = r["ce_reconstructed"]
        base["metrics"]["ce_recovered"] = r["ce_recovered"]
        base["metrics"]["n_ce_sequences"] = r["n_sequences"]
        upsert_safe(base)


# ---------------------------------------------------------------------------
# Local (CPU) entrypoint
# ---------------------------------------------------------------------------
def main_local() -> None:
    """Run CE sweep locally on CPU. Caches ce_clean and ce_zero once per
    run since they don't depend on the SAE."""
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, nargs="*",
                   help="if omitted: all seeds in DB; otherwise explicit list")
    p.add_argument("--n", dest="n_only", type=int, default=None, nargs="*",
                   help="filter to entries with n in this list (default: 4096)")
    p.add_argument("--n-sequences", type=int, default=N_CE_SEQUENCES)
    p.add_argument("--max-length", type=int, default=CE_SEQ_LENGTH)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if ce_recovered already populated with v2 formula.")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cpu"
    _log(f"Loading {MODEL_NAME} on {device}")
    lm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    target = lm.gpt_neox.layers[LAYER]

    _log(f"Tokenizing {args.n_sequences} Pile sequences (max_length={args.max_length})")
    ids_list = _tokenize_pile(tok, args.n_sequences, args.max_length)
    total_tok = sum(x.shape[1] for x in ids_list)
    _log(f"  {len(ids_list)} sequences, {total_tok} tokens")

    _log("Computing ce_clean (no hook)...")
    ce_clean, _ = _ce_weighted(lm, ids_list, device)
    _log(f"  ce_clean = {ce_clean:.4f}")

    _log("Computing ce_zero (layer-3 activation zeroed)...")
    ce_zero, _ = _ce_weighted(lm, ids_list, device,
                              hook_module=target, hook_fn=_zero_hook)
    _log(f"  ce_zero  = {ce_zero:.4f}")

    from models import build  # after torch is loaded
    db = load_db()
    enc = [e for e in db
           if e["inference_method"] == "encoder" and "_b" not in e["id"]]

    n_filter = args.n_only if args.n_only else [4096]
    seed_filter = set(args.seed) if args.seed else None

    def keep(e):
        if args.arch and e["architecture"] != args.arch:
            return False
        if args.d is not None and e["d"] != args.d:
            return False
        if args.k is not None and e["k"] != args.k:
            return False
        if seed_filter is not None and e["seed"] not in seed_filter:
            return False
        if e["n"] not in n_filter:
            return False
        return True

    candidates = [e for e in enc if keep(e)]
    _log(f"{len(candidates)} candidate encoder entries")

    processed = 0
    for i, e in enumerate(candidates, 1):
        arch, m, n, d, k, seed = (e["architecture"], e["m"], e["n"],
                                  e["d"], e["k"], e["seed"])
        if not args.force and e["metrics"].get("ce_zero") is not None:
            continue
        if not e.get("model_path") or not os.path.exists(e["model_path"]):
            _log(f"[{i}/{len(candidates)}] skip {e['id']} (missing model)")
            continue

        sae = build(arch, m=m, n=n, d=d, k=k, seed=seed)
        sae.load_state_dict(torch.load(e["model_path"], map_location="cpu",
                                       weights_only=True))
        sae = sae.to(device).eval()

        ce_recon, _ = _ce_weighted(lm, ids_list, device,
                                   hook_module=target, hook_fn=_recon_hook(sae))
        denom = ce_zero - ce_clean
        ce_recovered = (ce_zero - ce_recon) / denom if abs(denom) > 1e-8 else float("nan")

        r = {
            "arch": arch, "m": m, "n": n, "d": d, "k": k, "seed": seed,
            "ce_clean": ce_clean, "ce_zero": ce_zero,
            "ce_reconstructed": ce_recon, "ce_recovered": ce_recovered,
            "n_sequences": args.n_sequences,
        }
        _persist(db, r)
        processed += 1
        _log(f"[{i}/{len(candidates)}] {e['id']}  "
             f"recon={ce_recon:.3f}  recovered={ce_recovered:.4f}")

    _log(f"CE sweep local complete: {processed} entries updated.")


if __name__ == "__main__":
    main_local()
