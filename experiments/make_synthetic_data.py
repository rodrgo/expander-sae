"""Generate small synthetic activation files for local smoke-testing.

Produces data/activations_train.npy, data/activations_test.npy, and
data/tokens_and_acts.npz with the real schema but at tiny sizes.
"""
import argparse
from pathlib import Path

import numpy as np


def main(m: int = 64, n_train: int = 2000, n_test: int = 200,
         n_feat_sequences: int = 50, feat_seq_len: int = 32,
         vocab_size: int = 256, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    Path("data").mkdir(exist_ok=True)

    train = rng.standard_normal((n_train, m)).astype(np.float32)
    test = rng.standard_normal((n_test, m)).astype(np.float32)
    np.save("data/activations_train.npy", train)
    np.save("data/activations_test.npy", test)

    tokens = rng.integers(0, vocab_size, size=(n_feat_sequences, feat_seq_len),
                          dtype=np.int64)
    activations = rng.standard_normal(
        (n_feat_sequences * feat_seq_len, m)).astype(np.float32)
    np.savez("data/tokens_and_acts.npz", tokens=tokens, activations=activations)

    print(f"Wrote synthetic data: train={train.shape} test={test.shape} "
          f"tokens={tokens.shape} feat_acts={activations.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(m=args.m, n_train=args.n_train, n_test=args.n_test, seed=args.seed)
