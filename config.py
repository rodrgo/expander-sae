"""Sweep grids and training defaults.

Single source of truth for hyperparameters. Imported by every experiment
script and by the results-generation scripts.
"""
from typing import Iterator

# ---------------------------------------------------------------------------
# Model sizes
# ---------------------------------------------------------------------------
M = 512   # Pythia-70M hidden dim
N = 4096  # Dictionary size (standard 8x expansion)

# ---------------------------------------------------------------------------
# Sweep grids
# ---------------------------------------------------------------------------

# Primary Expander sweep: expander_tied at n=4096.
EXPANDER_D = [7, 30, 50, 100, 200]
K_VALUES = [16, 32, 64, 128]
SEEDS = [0, 1, 2]

# Matched-parameter dense with tied encoder.
# n_matched = d * N / M (same unique-param budget as expander_tied at n=N).
MATCHED_TIED = {
    7:   {"n": 56,   "k_values": [16, 28]},
    30:  {"n": 240,  "k_values": [16, 32, 64, 120]},
    50:  {"n": 400,  "k_values": [16, 32, 64, 128]},
    100: {"n": 800,  "k_values": [16, 32, 64, 128]},
    200: {"n": 1600, "k_values": [16, 32, 64, 128]},
}

# Matched-parameter dense with independent encoder + random init.
# n_matched = d * N / (2*M) (expander has d*n unique params;
# dense_randinit and dense_warmtied both have 2*m*n).
MATCHED_RANDINIT = {
    7:   {"n": 28,   "k_values": [14]},
    30:  {"n": 120,  "k_values": [16, 32, 60]},
    50:  {"n": 200,  "k_values": [16, 32, 64, 100]},
    100: {"n": 400,  "k_values": [16, 32, 64, 128]},
    200: {"n": 800,  "k_values": [16, 32, 64, 128]},
}

# Same matched-width grid for dense_warmtied (same 2mn param formula).
MATCHED_WARMTIED = dict(MATCHED_RANDINIT)

# Back-compat alias — old code refers to this name.
MATCHED_INDEP = MATCHED_RANDINIT

# Data-efficiency sweep: fixed k, varying data budget.
DATA_BUDGETS = [5000, 10000, 20000, 50000, 100000, 200000]
DATA_EFF_D = [7, 30, 100]
DATA_EFF_K = 64

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
# NOTE: `data_budget` = pool size available for with-replacement sampling
# during training. Samples-seen = batch_size * steps (1.28M at defaults).
TRAIN_DEFAULTS = {
    "steps": 5000,
    "data_budget": 200000,
    "batch_size": 256,
    "lr_max": 3e-4,
    "lr_min": 1e-5,
    "grad_clip": 1.0,
    "resample_interval": 1000,
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
N_TEST_ENCODER = 5000        # Full test set for encoder (instant)
N_TEST_ITERATIVE = 200       # Subset for OMP/NIHT/CoSaMP (slower)
N_TEST_ITERATIVE_SMALL_N = 5000  # Full test set when n <= SMALL_N_THRESHOLD
SMALL_N_THRESHOLD = 800

# CE-loss evaluation.
N_CE_SEQUENCES = 100
CE_SEQ_LENGTH = 128

# Pareto-critical configs get a second CE seed (seed=1 in addition to seed=0).
# Everything else is single-seed for CE only (training is still 3-seed).
CE_CRITICAL_CONFIGS = [
    ("expander_tied",     4096, 30),
    ("expander_tied",     4096, 100),
    ("dense_tied",        4096, 512),
    ("dense_warmtied",    4096, 512),
    ("dense_randinit",    4096, 512),
]

# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------
N_TOKENS_FEATURE = 128000
JACCARD_THRESHOLDS = [0.1, 0.3]


# ---------------------------------------------------------------------------
# Null-space gate (NSP)
# ---------------------------------------------------------------------------
def is_recovery_feasible(n: int, k: int) -> bool:
    """Null-space property requires n > 2k for k-sparse uniqueness.

    When n <= 2k every k-sparse vector is at most n-sparse, so 2k-sparseness
    is vacuous and the nullspace condition for unique k-sparse recovery
    degenerates. Dense baselines at small matched-n (e.g. n=28, k=14) hit
    this boundary and are excluded from headline comparisons.
    """
    return n > 2 * k


# ---------------------------------------------------------------------------
# Sweep enumeration
# ---------------------------------------------------------------------------
def expander_configs() -> Iterator[tuple]:
    """(arch, m, n, d, k, seed, data_budget) for expander_tied at n=4096."""
    for d in EXPANDER_D:
        for k in K_VALUES:
            if not is_recovery_feasible(N, k):
                continue
            for seed in SEEDS:
                yield ("expander_tied", M, N, d, k, seed, TRAIN_DEFAULTS["data_budget"])


def dense_tied_matched_configs() -> Iterator[tuple]:
    for d, spec in MATCHED_TIED.items():
        n = spec["n"]
        for k in spec["k_values"]:
            if not is_recovery_feasible(n, k):
                continue
            for seed in SEEDS:
                yield ("dense_tied", M, n, M, k, seed, TRAIN_DEFAULTS["data_budget"])


def dense_randinit_matched_configs() -> Iterator[tuple]:
    for d, spec in MATCHED_RANDINIT.items():
        n = spec["n"]
        for k in spec["k_values"]:
            if not is_recovery_feasible(n, k):
                continue
            for seed in SEEDS:
                yield ("dense_randinit", M, n, M, k, seed, TRAIN_DEFAULTS["data_budget"])


def dense_warmtied_matched_configs() -> Iterator[tuple]:
    for d, spec in MATCHED_WARMTIED.items():
        n = spec["n"]
        for k in spec["k_values"]:
            if not is_recovery_feasible(n, k):
                continue
            for seed in SEEDS:
                yield ("dense_warmtied", M, n, M, k, seed, TRAIN_DEFAULTS["data_budget"])


# Back-compat alias — older code paths still call this.
dense_indep_matched_configs = dense_randinit_matched_configs


def dense_full_configs() -> Iterator[tuple]:
    """Full-size dense baselines at n=4096 for each k."""
    for k in K_VALUES:
        if not is_recovery_feasible(N, k):
            continue
        for seed in SEEDS:
            yield ("dense_tied", M, N, M, k, seed, TRAIN_DEFAULTS["data_budget"])
            yield ("dense_warmtied", M, N, M, k, seed, TRAIN_DEFAULTS["data_budget"])
            yield ("dense_randinit", M, N, M, k, seed, TRAIN_DEFAULTS["data_budget"])


def data_efficiency_configs() -> Iterator[tuple]:
    """(arch, m, n, d, k, seed, data_budget) for the data-efficiency sweep."""
    for d in DATA_EFF_D:
        for budget in DATA_BUDGETS:
            for seed in SEEDS:
                yield ("expander_tied", M, N, d, DATA_EFF_K, seed, budget)
    # Dense baseline data-efficiency (at n=4096)
    for budget in DATA_BUDGETS:
        for seed in SEEDS:
            yield ("dense_tied", M, N, M, DATA_EFF_K, seed, budget)


BASELINE_D = [7, 50, 200]
PRUNED_RETUNED_K_VALUES = [16, 64, 128]


def clustered_sparse_configs() -> Iterator[tuple]:
    """Block-structured-mask baseline at n=4096."""
    for d in BASELINE_D:
        for k in K_VALUES:
            if not is_recovery_feasible(N, k):
                continue
            for seed in SEEDS:
                yield ("clustered_sparse", M, N, d, k, seed,
                       TRAIN_DEFAULTS["data_budget"])


def pruned_retuned_dense_configs() -> Iterator[tuple]:
    """Prune-then-retune baseline. Fine-tunes only the retained decoder
    values extracted from a trained dense_tied SAE at the matching seed.
    Requires dense_tied_m{m}_n{n}_d{m}_k{k}_seed{seed}.pt to exist."""
    for d in BASELINE_D:
        for k in PRUNED_RETUNED_K_VALUES:
            if not is_recovery_feasible(N, k):
                continue
            for seed in SEEDS:
                yield ("pruned_retuned_dense", M, N, d, k, seed,
                       TRAIN_DEFAULTS["data_budget"])


def all_training_configs() -> list[tuple]:
    """All unique (arch, m, n, d, k, seed, data_budget) tuples to train."""
    configs = list(expander_configs())
    configs += list(dense_tied_matched_configs())
    configs += list(dense_randinit_matched_configs())
    configs += list(dense_warmtied_matched_configs())
    configs += list(dense_full_configs())
    configs += list(data_efficiency_configs())
    configs += list(clustered_sparse_configs())
    configs += list(pruned_retuned_dense_configs())
    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique
