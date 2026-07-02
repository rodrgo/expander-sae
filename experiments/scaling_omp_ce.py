"""OMP-decoded CE-loss recovered for matched-parameter cells of the
Qwen2.5-3B and Llama-3.2-1B sweeps.

The trained-encoder CE-recovered already exists in
``results/qwen2_5_3b_replication.json`` and
``results/llama32_1b_replication.json``. This script reloads the same
trained checkpoints from the Modal volume, replaces the encoder forward
with a one-shot OMP solver (top-$k$ pick by raw $\\mathbf{W}^\\top\\mathbf{r}$,
Cholesky refit on the picked support), and recomputes CE-loss recovered.

The point is to disentangle two effects that the trained-encoder
comparison conflates:
  - *encoder amortisation gap*: the trained encoder's TopK gives an
    approximate sparse code; OMP's batched Cholesky refit gives the
    exact least-squares coefficients on the same picked support.
  - *decoder quality*: the parameter-matched dense decoder has full
    connectivity at small $n$; the Expander has $d$-regular sparsity
    at full $n$.

We re-evaluate both Expander and matched-Dense cells under OMP, so the
comparison is symmetric (no architecture is given the encoder boost).
The full-Dense $n{=}m \\cdot 8$ row is included for context.

Output: ``results/{tag}_omp_ce.json`` with one record per
(model, layer, arch, n, d, k, seed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


APP_NAME = "mech-expander-omp-ce"
VOLUME_NAME = "mech-expander-cache"
VOL_MOUNT = "/cache"

# Configurations. Each entry is a self-contained eval target. Cells are
# the union of Expander {d=7, 30, 102} and matched-Dense {n'=240, 816}
# plus full-Dense (n=16384) for context, three seeds each.
CONFIGS = {
    "llama32_1b_layer6": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "layer": 6,
        "needs_hf_token": True,
    },
    "llama32_1b_layer12": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "layer": 12,
        "needs_hf_token": True,
    },
    "qwen25_3b_layer12": {
        "model_name": "Qwen/Qwen2.5-3B",
        "layer": 12,
        "needs_hf_token": False,
    },
    "qwen25_3b_layer24": {
        "model_name": "Qwen/Qwen2.5-3B",
        "layer": 24,
        "needs_hf_token": False,
    },
}

M = 2048
N = 16384
K = 64
SEEDS = [0, 1, 2]
EXPANDER_DS = [7, 30, 102]
MATCHED_DENSE_NS = {30: 240, 102: 816}
DENSE_ARCH = "dense_warmtied"

CE_SEQ_LENGTH = 128
CE_N_SEQUENCES = 100


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
hf_secret = modal.Secret.from_name("huggingface")


@app.function(image=image, gpu="A10G", timeout=10800,
              volumes={VOL_MOUNT: volume},
              secrets=[hf_secret])
def run_one_layer(tag: str, iterative: bool = False) -> list:
    import os, time
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from models import build

    cfg = CONFIGS[tag]
    model_name = cfg["model_name"]
    layer = cfg["layer"]
    hf_kwargs = {"token": os.environ["HF_TOKEN"]} if cfg["needs_hf_token"] else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{tag}] running OMP-CE eval on {model_name} layer {layer}")

    # ---- Build target cell list ----
    cells: list[tuple[str, int, int, int]] = []  # (arch, n, d, seed)
    for d in EXPANDER_DS:
        for seed in SEEDS:
            cells.append(("expander_tied", N, d, seed))
    for seed in SEEDS:
        cells.append((DENSE_ARCH, N, M, seed))  # full-Dense
    for d, n_matched in MATCHED_DENSE_NS.items():
        for seed in SEEDS:
            cells.append((DENSE_ARCH, n_matched, M, seed))

    # ---- Locate trained checkpoints (already on volume) ----
    for arch, n, d, seed in cells:
        ckpt = f"{VOL_MOUNT}/scaling/{tag}_{arch}_n{n}_d{d}_k{K}_seed{seed}.pt"
        if not os.path.exists(ckpt):
            raise RuntimeError(f"missing checkpoint: {ckpt}")
    print(f"[{tag}] all {len(cells)} checkpoints present on volume")

    # ---- Load LM and tokenize CE eval set ----
    print(f"[{tag}] loading LM in bf16")
    lm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, **hf_kwargs).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
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

    # ---- One-shot OMP (gOMP at L=k; one Cholesky refit) ----
    def oneshot_omp_decode(W_dec, b_dec, y, k):
        B, m = y.shape
        y_centered = y - b_dec.unsqueeze(0)
        corrs = y_centered @ W_dec                              # (B, n)
        _, support = corrs.topk(k, dim=-1)                      # (B, k)
        W_S = W_dec.unsqueeze(0).expand(B, -1, -1).gather(
            2, support.unsqueeze(1).expand(-1, m, -1))          # (B, m, k)
        rhs = torch.bmm(W_S.transpose(1, 2), y_centered.unsqueeze(-1)
                        ).squeeze(-1)                           # (B, k)
        Gram = torch.bmm(W_S.transpose(1, 2), W_S)
        Gram = Gram + 1e-6 * torch.eye(k, device=Gram.device, dtype=Gram.dtype)
        L = torch.linalg.cholesky(Gram)
        x_S = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        recon_centered = torch.bmm(W_S, x_S.unsqueeze(-1)).squeeze(-1)
        return recon_centered + b_dec.unsqueeze(0)

    # ---- Iterative OMP (k sequential picks, refit-on-active-set each iter) ----
    def iterative_omp_decode(W_dec, b_dec, y, k):
        """Standard iterative OMP. At each of the k iterations: compute
        $\\mathbf{W}^\\top \\mathbf{r}$ for the current residual, mask
        already-picked columns, take argmax, append to support, refit
        coefficients via Cholesky on the (i+1, i+1) Gram leading block,
        update residual. Final reconstruction uses the full-k refit."""
        B, m = y.shape
        n = W_dec.shape[1]
        device = y.device
        dtype = y.dtype

        y_centered = y - b_dec.unsqueeze(0)                     # (B, m)
        r = y_centered.clone()
        support = torch.full((B, k), -1, dtype=torch.long, device=device)
        picked = torch.zeros(B, n, dtype=torch.bool, device=device)
        # Pre-allocate active-set columns buffer; we fill column i at iter i.
        W_S = torch.empty(B, m, k, dtype=dtype, device=device)
        Gram = torch.zeros(B, k, k, dtype=dtype, device=device)
        rhs = torch.empty(B, k, dtype=dtype, device=device)
        ridge = 1e-6

        for i in range(k):
            corrs = r @ W_dec                                   # (B, n)
            corrs.masked_fill_(picked, float("-inf"))
            j = corrs.argmax(dim=-1)                            # (B,)
            support[:, i] = j
            picked.scatter_(1, j.unsqueeze(-1), True)

            # Pull new column for each sample: W_dec[:, j[b]] -> (B, m)
            new_col = W_dec.index_select(1, j).t().contiguous()
            # Wait: W_dec is (m, n); W_dec[:, j[b]] has shape (m,).
            # index_select(1, j) gives (m, B); .t() → (B, m). OK.
            W_S[:, :, i] = new_col

            # Update Gram leading (i+1, i+1) block: only new row/col
            # need to be filled.
            # Gram[:, :i+1, i] = W_S[:, :, :i+1]^T @ new_col
            # Gram[:, i, :i+1] = same (symmetry)
            new_inner = torch.bmm(W_S[:, :, :i+1].transpose(1, 2),
                                  new_col.unsqueeze(-1)).squeeze(-1)  # (B, i+1)
            Gram[:, :i+1, i] = new_inner
            Gram[:, i, :i+1] = new_inner

            # rhs[:, i] = new_col^T @ y_centered  (per sample)
            rhs[:, i] = (new_col * y_centered).sum(dim=-1)

            # Refit on leading (i+1) active columns.
            G_block = Gram[:, :i+1, :i+1] + \
                      ridge * torch.eye(i+1, device=device, dtype=dtype).unsqueeze(0)
            L = torch.linalg.cholesky(G_block)
            x_S = torch.cholesky_solve(rhs[:, :i+1].unsqueeze(-1), L).squeeze(-1)

            # New residual: y_centered - W_S[:i+1] @ x_S
            r = y_centered - torch.bmm(W_S[:, :, :i+1],
                                       x_S.unsqueeze(-1)).squeeze(-1)

        # Final reconstruction (final x_S already computed at i=k-1).
        recon_centered = torch.bmm(W_S, x_S.unsqueeze(-1)).squeeze(-1)
        return recon_centered + b_dec.unsqueeze(0)

    omp_decode = (iterative_omp_decode if iterative else oneshot_omp_decode)
    print(f"[{tag}] OMP variant: {'iterative (L=1)' if iterative else 'one-shot (L=k)'}")

    # ---- Per-cell CE-loss recovered with OMP forward ----
    results: list[dict] = []
    for arch, n, d, seed in cells:
        ckpt = f"{VOL_MOUNT}/scaling/{tag}_{arch}_n{n}_d{d}_k{K}_seed{seed}.pt"
        sae = build(arch, m=M, n=n, d=d, k=K, seed=seed)
        sae.load_state_dict(torch.load(ckpt, map_location="cpu",
                                       weights_only=True))
        sae = sae.to(device).eval()

        # Pre-compute fp32 W_dec, b_dec for OMP.
        with torch.no_grad():
            W_dec = sae.W_dec.detach().to(torch.float32).contiguous()  # (m, n)
            b_dec = sae.b_dec.detach().to(torch.float32).contiguous()  # (m,)

        def _omp_hook(_mod, _inp, out):
            h, rewrap = _unwrap(out)
            B, S, H = h.shape
            with torch.no_grad():
                y_flat = h.reshape(-1, H).to(torch.float32)
                recon = omp_decode(W_dec, b_dec, y_flat, K)
            return rewrap(recon.reshape(B, S, H).to(h.dtype))

        ce_recon = _ce_with_hook(_omp_hook)
        denom = ce_zero - ce_clean
        ce_recovered = (ce_zero - ce_recon) / denom if abs(denom) > 1e-8 else float("nan")
        rec = {
            "model": model_name, "layer": layer,
            "arch": arch, "m": M, "n": n, "d": d, "k": K, "seed": seed,
            "ce_clean": ce_clean, "ce_zero": ce_zero,
            "ce_reconstructed_omp": ce_recon,
            "ce_recovered_omp": ce_recovered,
        }
        results.append(rec)
        print(f"[{tag}] {arch:<14} n={n:>5} d={d:<5} seed={seed} "
              f"CE_rec_OMP={ce_recovered:.3f}")
        del sae, W_dec, b_dec
        torch.cuda.empty_cache()

    suffix = "iter_omp_ce" if iterative else "omp_ce"
    out_path = f"{VOL_MOUNT}/scaling/{tag}_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"[{tag}] wrote {out_path}")
    return results


@app.local_entrypoint()
def main(tag: str = "llama32_1b_layer12", iterative: bool = False):
    """Run one (model, layer) tag at a time. Avoids the multi-layer
    detached-cancellation issue we hit with the original sweep."""
    print(f"main: dispatching {tag} (iterative={iterative})")
    results = run_one_layer.remote(tag, iterative)
    suffix = "iter_omp_ce" if iterative else "omp_ce"
    out_local = f"results/{tag}_{suffix}.json"
    Path(out_local).parent.mkdir(parents=True, exist_ok=True)
    with open(out_local, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_local} ({len(results)} entries)")
