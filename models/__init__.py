from .expander import ExpanderSAE
from .dense import (
    DenseTiedSAE, DenseWarmTiedSAE, DenseRandinitSAE, DenseIndepSAE,
)
from .clustered_sparse import ClusteredSparseSAE
from .pruned_retuned_dense import PrunedRetunedDenseSAE
from .training import train_sae

__all__ = [
    "ExpanderSAE",
    "DenseTiedSAE", "DenseWarmTiedSAE", "DenseRandinitSAE", "DenseIndepSAE",
    "ClusteredSparseSAE", "PrunedRetunedDenseSAE",
    "train_sae", "build",
]


def build(arch: str, m: int, n: int, d: int, k: int, seed: int = 0):
    """Factory: (arch_name, sizes) -> instantiated module."""
    if arch == "expander_tied":
        return ExpanderSAE(m=m, n=n, d=d, k=k, seed=seed)
    if arch == "dense_tied":
        return DenseTiedSAE(m=m, n=n, k=k, seed=seed)
    if arch == "dense_warmtied":
        return DenseWarmTiedSAE(m=m, n=n, k=k, seed=seed)
    if arch in ("dense_randinit", "dense_indep"):
        # Accept the legacy name so existing DB entries / checkpoints still load.
        return DenseRandinitSAE(m=m, n=n, k=k, seed=seed)
    if arch == "clustered_sparse":
        return ClusteredSparseSAE(m=m, n=n, d=d, k=k, seed=seed)
    if arch == "pruned_retuned_dense":
        return PrunedRetunedDenseSAE(m=m, n=n, d=d, k=k, seed=seed)
    raise ValueError(f"Unknown architecture: {arch}")
