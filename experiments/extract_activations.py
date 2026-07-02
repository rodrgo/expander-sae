"""Extract Pythia-70M layer-3 residual-stream activations + token IDs.

Runs on Modal GPU. Caches outputs on a Modal volume so repeat runs are free.

After extraction, use `modal volume get mech-expander-cache <path> <local>` to
pull the arrays down. The local entrypoint does this for you via `main()` and
writes:

  data/activations_train.npy    — (200_000, 512) float32
  data/activations_test.npy     — (5_000,   512) float32
  data/tokens_and_acts.npz      — token_ids (1000, 128) int64
                                   activations (128_000, 512) float32

Usage (full Modal pipeline):
    modal run experiments/extract_activations.py::main

Usage (local smoke-test, no Modal): see experiments/make_synthetic_data.py.
"""
import modal

APP_NAME = "mech-expander-extract"
VOLUME_NAME = "mech-expander-cache"
VOL_MOUNT = "/cache"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "transformers==4.44.0",
        "datasets",
        "accelerate",
        "tqdm",
        "zstandard",
    )
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

N_TRAIN = 200_000
N_TEST = 5_000
N_TOTAL = N_TRAIN + N_TEST
N_TOKENS_FEATURE = 128_000
FEATURE_SEQUENCES = 1000
FEATURE_SEQ_LEN = 128
LAYER = 3
MODEL_NAME = "EleutherAI/pythia-70m-deduped"


@app.function(image=image, gpu="A10G", timeout=1800, volumes={VOL_MOUNT: volume})
def extract_pool(n_total: int = N_TOTAL, layer: int = LAYER) -> dict:
    """Extract n_total residual-stream activations. Cached on volume."""
    import os
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    cache = f"{VOL_MOUNT}/activations_{n_total}_layer{layer}.npy"
    if os.path.exists(cache):
        acts = np.load(cache)
        return {"shape": list(acts.shape), "path": cache}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    captured: list[np.ndarray] = []
    target = lm.gpt_neox.layers[layer]

    def hook(_mod, _inp, out):
        h = out[0]
        captured.append(h.detach().float().cpu().numpy().reshape(-1, h.shape[-1]))

    handle = target.register_forward_hook(hook)
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    acts_list, total = [], 0
    for item in ds:
        if total >= n_total:
            break
        text = (item.get("text") or "").strip()
        if not text:
            continue
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=128, padding=False)["input_ids"].to(device)
        if ids.shape[1] < 4:
            continue
        captured.clear()
        with torch.no_grad():
            lm(ids)
        if captured:
            acts_list.append(captured[0])
            total += captured[0].shape[0]
            if len(acts_list) % 100 == 0:
                print(f"  collected {total} / {n_total} tokens")
    handle.remove()

    acts = np.concatenate(acts_list, axis=0)[:n_total]
    np.save(cache, acts)
    volume.commit()
    return {"shape": list(acts.shape), "path": cache}


@app.function(image=image, gpu="A10G", timeout=1800, volumes={VOL_MOUNT: volume})
def extract_feature_batch(n_sequences: int = FEATURE_SEQUENCES,
                          seq_len: int = FEATURE_SEQ_LEN,
                          layer: int = LAYER) -> dict:
    """Extract paired (tokens, activations) for feature analysis."""
    import os
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    cache = f"{VOL_MOUNT}/tokens_and_acts_{n_sequences}x{seq_len}_layer{layer}.npz"
    if os.path.exists(cache):
        data = np.load(cache)
        return {"path": cache, "tokens": list(data["tokens"].shape),
                "activations": list(data["activations"].shape)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    captured: list[np.ndarray] = []
    target = lm.gpt_neox.layers[layer]

    def hook(_mod, _inp, out):
        captured.append(out[0].detach().float().cpu().numpy())

    handle = target.register_forward_hook(hook)
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    token_rows: list[np.ndarray] = []
    act_rows: list[np.ndarray] = []
    for item in ds:
        if len(token_rows) >= n_sequences:
            break
        text = (item.get("text") or "").strip()
        if not text:
            continue
        ids_t = tok(text, return_tensors="pt", truncation=True,
                    max_length=seq_len, padding=False)["input_ids"]
        if ids_t.shape[1] < seq_len:
            continue  # only keep full-length sequences for a regular tensor
        ids = ids_t.to(device)
        captured.clear()
        with torch.no_grad():
            lm(ids)
        if not captured:
            continue
        h = captured[0][0]  # (seq_len, hidden)
        token_rows.append(ids_t[0].cpu().numpy().astype(np.int64))
        act_rows.append(h.astype(np.float32))

    tokens = np.stack(token_rows, axis=0)  # (n_sequences, seq_len)
    activations = np.concatenate(act_rows, axis=0)  # (n_sequences*seq_len, hidden)
    handle.remove()

    np.savez(cache, tokens=tokens, activations=activations)
    volume.commit()
    return {"path": cache, "tokens": list(tokens.shape),
            "activations": list(activations.shape)}


@app.function(image=image, timeout=3600, volumes={VOL_MOUNT: volume})
def split_pool(src: str, n_train: int = N_TRAIN, n_test: int = N_TEST) -> dict:
    """Split the pool file into train/test halves on the volume.

    Cheaper + simpler than returning a 400MB blob over the wire — the user
    downloads train/test separately with `modal volume get`.
    """
    import numpy as np
    pool = np.load(src)
    assert pool.shape[0] >= n_train + n_test, f"pool too small: {pool.shape[0]}"
    train_path = f"{VOL_MOUNT}/activations_train.npy"
    test_path = f"{VOL_MOUNT}/activations_test.npy"
    np.save(train_path, pool[:n_train])
    np.save(test_path, pool[n_train:n_train + n_test])
    volume.commit()
    return {"train": train_path, "test": test_path,
            "train_shape": [n_train, pool.shape[1]],
            "test_shape": [n_test, pool.shape[1]]}


@app.local_entrypoint()
def main():
    """Extract on Modal, then print the `modal volume get` commands to run.

    The actual download is left to the user so we never pull 400MB through
    a Modal function return value.
    """
    info = extract_pool.remote(n_total=N_TOTAL, layer=LAYER)
    print(f"Pool: {info['shape']} at {info['path']}")

    split = split_pool.remote(info["path"], n_train=N_TRAIN, n_test=N_TEST)
    print(f"Split: train={split['train_shape']} at {split['train']} | "
          f"test={split['test_shape']} at {split['test']}")

    feat = extract_feature_batch.remote(
        n_sequences=FEATURE_SEQUENCES, seq_len=FEATURE_SEQ_LEN, layer=LAYER)
    print(f"Features: tokens={feat['tokens']} acts={feat['activations']} "
          f"at {feat['path']}")

    print()
    print("To download locally:")
    for remote, local in [
        (split["train"], "data/activations_train.npy"),
        (split["test"], "data/activations_test.npy"),
        (feat["path"], "data/tokens_and_acts.npz"),
    ]:
        remote_rel = remote[len(VOL_MOUNT) + 1:]
        print(f"  modal volume get mech-expander-cache {remote_rel} {local}")
