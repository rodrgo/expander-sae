"""Qwen2.5-3B cross-architecture replication of the headline Expander-SAE result.

Mirrors ``experiments/scaling_pythia160m.py`` but targets a much larger,
non-Pythia model (Qwen2.5-3B, ~3B params, m=2048, RoPE + GQA + SwiGLU).
Trains Expander-SAE at $d \\in \\{7, 30, 102\\}$ (corresponding to
$d/m \\in \\{0.34\\%, 1.5\\%, 5\\%\\}$) plus a Dense-SAE baseline plus
matched-parameter reduced-$n$ Dense-SAEs at the residual stream of two
layers (12 and 24 of 36, i.e. third and two-thirds of depth), three
seeds. Produces ``results/qwen2_5_3b_replication.json``.

Constraints / choices specific to this script:
- LM is loaded in **bf16** for both activation extraction and CE-loss
  evaluation (~6 GB resident on A10G, vs ~12 GB at fp32).
- Hook attaches to ``lm.model.layers[layer]`` (HF Llama-style
  ``Qwen2ForCausalLM``); confirmed by inspection.
- Reduced-n dense baselines: $n' = d \\cdot N / m$. At $d{=}7$ this
  gives $n'{=}56$ which falls below the NSP gate ($n' < 2k = 128$);
  we skip that cell. Reported matched-dense baselines: $n' \\in
  \\{240, 816\\}$ for $d \\in \\{30, 102\\}$.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


APP_NAME = "mech-expander-qwen25-3b"
VOLUME_NAME = "mech-expander-cache"
VOL_MOUNT = "/cache"
MODEL_NAME = "Qwen/Qwen2.5-3B"

LAYERS = [12, 24]
EXPANDER_DS = [7, 30, 102]
DENSE_ARCH = "dense_warmtied"
M = 2048
N = 16384
K = 64
SEEDS = [0, 1, 2]

# Matched-parameter reduced-n dense baselines: n' = d * N / m.
# At d=7: n' = 56 < 2k = 128 (NSP gate), so skipped.
MATCHED_DENSE_NS = {
    30:  240,   # 30 * 16384 / 2048
    102: 816,   # 102 * 16384 / 2048
}

N_TRAIN = 200_000
N_TEST = 5_000
N_TOTAL = N_TRAIN + N_TEST
CE_SEQ_LENGTH = 128
CE_N_SEQUENCES = 100

OUT_PATH = "results/qwen2_5_3b_replication.json"


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "scipy",
        "transformers==4.44.0",
        "datasets",
        "accelerate",
        "tqdm",
        "zstandard",
    )
    .add_local_python_source("config", "db", "models", "inference")
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=14400,
              volumes={VOL_MOUNT: volume})
def run_layer(layer: int) -> list:
    import json
    import os
    import time
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    from models import build, train_sae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"qwen25_3b_layer{layer}"

    # ---- Stage 1: activation extraction (fp16-cached on volume) ----
    pool_cache = f"{VOL_MOUNT}/{tag}_pool_{N_TOTAL}.npy"
    train_cache = f"{VOL_MOUNT}/{tag}_train.npy"
    test_cache = f"{VOL_MOUNT}/{tag}_test.npy"

    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print(f"[{tag}] cache hit, reusing train/test split")
    else:
        if os.path.exists(pool_cache):
            print(f"[{tag}] pool cache hit, splitting")
            pool = np.load(pool_cache).astype(np.float32)
        else:
            print(f"[{tag}] extracting pool ({N_TOTAL} tokens, bf16 LM)")
            t0 = time.time()
            lm = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            captured: list[np.ndarray] = []
            target = lm.model.layers[layer]  # HF Llama-style attribute path

            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # Cast bf16→fp32 before numpy; keep activations in fp32 for SAE training.
                captured.append(h.detach().float().cpu().numpy().reshape(
                    -1, h.shape[-1]))

            handle = target.register_forward_hook(hook)
            ds = load_dataset("monology/pile-uncopyrighted",
                              split="train", streaming=True)
            acts_list, total = [], 0
            for item in ds:
                if total >= N_TOTAL:
                    break
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                ids = tok(text, return_tensors="pt", truncation=True,
                          max_length=CE_SEQ_LENGTH,
                          padding=False)["input_ids"].to(device)
                if ids.shape[1] < 4:
                    continue
                captured.clear()
                with torch.no_grad():
                    lm(ids)
                if captured:
                    acts_list.append(captured[0])
                    total += captured[0].shape[0]
            handle.remove()
            del lm
            torch.cuda.empty_cache()
            pool = np.concatenate(acts_list, axis=0)[:N_TOTAL]
            np.save(pool_cache, pool.astype(np.float16))
            print(f"[{tag}] extracted {pool.shape} in "
                  f"{time.time()-t0:.1f}s (cached as fp16)")
        np.save(train_cache, pool[:N_TRAIN].astype(np.float32))
        np.save(test_cache, pool[N_TRAIN:N_TRAIN + N_TEST].astype(np.float32))
        volume.commit()
        del pool

    train_acts = np.load(train_cache)
    test_acts = np.load(test_cache)
    print(f"[{tag}] train={train_acts.shape} test={test_acts.shape}")
    print(f"[{tag}] act stats: mean={train_acts.mean():.3f} "
          f"std={train_acts.std():.3f} "
          f"abs_max={np.abs(train_acts).max():.3f}")

    # ---- Stage 2: train all configs ----
    configs = []
    # Expander-SAE.
    for d in EXPANDER_DS:
        for seed in SEEDS:
            configs.append(("expander_tied", M, N, d, K, seed))
    # Dense-SAE at full n.
    for seed in SEEDS:
        configs.append((DENSE_ARCH, M, N, M, K, seed))
    # Matched-parameter reduced-n Dense-SAE baselines.
    for d, n_matched in MATCHED_DENSE_NS.items():
        for seed in SEEDS:
            # Use n_matched for n; m and d=m collapse the dense SAE to
            # the standard "dense at small n" parameter-matched control.
            configs.append((DENSE_ARCH, M, n_matched, M, K, seed))

    trained: list[dict] = []
    for arch, m, n, d, k, seed in configs:
        t0 = time.time()
        ckpt_path = (f"{VOL_MOUNT}/scaling/{tag}_"
                     f"{arch}_n{n}_d{d}_k{k}_seed{seed}.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        # Skip already-trained checkpoints (resume after crashes /
        # heartbeat disconnects). Re-evaluate rel-err on test split so
        # the per-record metrics are populated. If the cached rel_err
        # is degenerate (>1.0, i.e. worse than predicting zero), the
        # training was unstable for that seed --- delete and retrain.
        cache_loaded = False
        if os.path.exists(ckpt_path):
            try:
                model = build(arch, m=m, n=n, d=d, k=k, seed=seed)
                model.load_state_dict(torch.load(
                    ckpt_path, map_location="cpu", weights_only=True))
                model = model.to(device).eval()
                test_t = torch.from_numpy(test_acts).float().to(device)
                with torch.no_grad():
                    y_hat, h = model(test_t)
                    per_err = (torch.norm(test_t - y_hat, dim=-1) /
                               torch.norm(test_t, dim=-1).clamp(min=1e-12))
                    rel_err_cached = float(per_err.mean())
                if rel_err_cached <= 1.0:
                    rel_err = rel_err_cached
                    rel_err_sem = float(per_err.std() / np.sqrt(per_err.numel()))
                    dead_frac = float((h.abs().sum(dim=0) == 0).float().mean())
                    print(f"[{tag}] cached {arch} n={n} d={d} seed={seed} "
                          f"rel_err={rel_err:.4f} (skipped train, eval only)")
                    cache_loaded = True
                else:
                    print(f"[{tag}] cached {arch} n={n} d={d} seed={seed} "
                          f"rel_err={rel_err_cached:.4f} > 1.0, retraining")
                    del model
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[{tag}] cache load failed for {arch} n={n} d={d} "
                      f"seed={seed}: {exc}; retraining")
        if not cache_loaded:
            model = build(arch, m=m, n=n, d=d, k=k, seed=seed)
            model, info = train_sae(
                model, train_acts,
                steps=5000, batch_size=256,
                lr_max=3e-4, lr_min=1e-5, grad_clip=1.0,
                resample_interval=1000,
                device=device,
            )
            model = model.to(device).eval()
            test_t = torch.from_numpy(test_acts).float().to(device)
            with torch.no_grad():
                y_hat, h = model(test_t)
                per_err = (torch.norm(test_t - y_hat, dim=-1) /
                           torch.norm(test_t, dim=-1).clamp(min=1e-12))
                rel_err = float(per_err.mean())
                rel_err_sem = float(per_err.std() / np.sqrt(per_err.numel()))
                dead_frac = float((h.abs().sum(dim=0) == 0).float().mean())
            torch.save({k_: v.detach().cpu()
                        for k_, v in model.state_dict().items()},
                       ckpt_path)
            print(f"[{tag}] trained {arch} n={n} d={d} seed={seed} "
                  f"rel_err={rel_err:.4f} ({time.time()-t0:.1f}s)")

        trained.append({
            "model": "qwen2.5-3b", "layer": layer,
            "arch": arch, "m": m, "n": n,
            "d": d, "k": k, "seed": seed,
            "rel_err_mean": rel_err,
            "rel_err_sem": rel_err_sem,
            "dead_frac": dead_frac,
            "ckpt": ckpt_path,
            "train_secs": round(time.time() - t0, 1),
        })
        del model
        torch.cuda.empty_cache()

    volume.commit()

    # ---- Stage 3: CE-loss recovered ----
    print(f"[{tag}] computing CE-recovered ({CE_N_SEQUENCES} seqs)")
    lm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("monology/pile-uncopyrighted",
                      split="train", streaming=True)
    ids_list = []
    for item in ds:
        text = (item.get("text") or "").strip()
        if not text or len(text) < 50:
            continue
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=CE_SEQ_LENGTH, padding=False)["input_ids"]
        if ids.shape[1] < 4:
            continue
        ids_list.append(ids)
        if len(ids_list) >= CE_N_SEQUENCES:
            break

    target = lm.model.layers[layer]

    def _unwrap(out):
        if isinstance(out, tuple):
            return out[0], lambda new_h: (new_h,) + out[1:]
        return out, lambda new_h: new_h

    def _ce_with_hook(hook_fn=None) -> float:
        tot, n_tok = 0.0, 0
        handle = (target.register_forward_hook(hook_fn)
                  if hook_fn is not None else None)
        try:
            for ids in ids_list:
                ids = ids.to(device)
                with torch.no_grad():
                    loss = float(lm(input_ids=ids, labels=ids).loss)
                tot += loss * ids.shape[1]
                n_tok += ids.shape[1]
        finally:
            if handle is not None:
                handle.remove()
        return tot / max(n_tok, 1)

    def _zero_hook(_mod, _inp, out):
        h, rewrap = _unwrap(out)
        return rewrap(torch.zeros_like(h))

    ce_clean = _ce_with_hook(None)
    ce_zero = _ce_with_hook(_zero_hook)
    print(f"[{tag}] ce_clean={ce_clean:.4f} ce_zero={ce_zero:.4f}")

    for rec in trained:
        sae = build(rec["arch"], m=rec["m"], n=rec["n"],
                    d=rec["d"], k=rec["k"], seed=rec["seed"])
        sae.load_state_dict(torch.load(rec["ckpt"], map_location="cpu"))
        sae = sae.to(device).eval()

        def _recon_hook(_mod, _inp, out):
            h, rewrap = _unwrap(out)
            B, S, H = h.shape
            with torch.no_grad():
                # SAE expects fp32, LM activations are bf16; cast in/out.
                recon, _ = sae(h.float().reshape(-1, H))
            return rewrap(recon.reshape(B, S, H).to(h.dtype))

        ce_recon = _ce_with_hook(_recon_hook)
        denom = ce_zero - ce_clean
        ce_recovered = ((ce_zero - ce_recon) / denom
                        if abs(denom) > 1e-8 else float("nan"))
        rec["ce_clean"] = ce_clean
        rec["ce_zero"] = ce_zero
        rec["ce_reconstructed"] = ce_recon
        rec["ce_recovered"] = ce_recovered
        print(f"[{tag}] {rec['arch']} n={rec['n']} d={rec['d']} "
              f"seed={rec['seed']} CE_rec={ce_recovered:.3f}")
        del sae
        torch.cuda.empty_cache()
        rec.pop("ckpt", None)

    # Per-layer JSON: written to volume so it survives heartbeat
    # disconnects. Aggregated into the final results JSON in main().
    layer_json = f"{VOL_MOUNT}/scaling/{tag}_results.json"
    with open(layer_json, "w") as f:
        json.dump(trained, f, indent=2)
    volume.commit()
    print(f"[{tag}] wrote {layer_json}")

    return trained


@app.local_entrypoint()
def main():
    all_results: list = []
    for layer in LAYERS:
        layer_results = run_layer.remote(layer)
        all_results.extend(layer_results)
        print(f"[main] layer {layer}: {len(layer_results)} configs")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {OUT_PATH} ({len(all_results)} entries)")
