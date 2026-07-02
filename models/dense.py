"""Dense SAE baselines: three encoder strategies.

`dense_tied`    — strict tie: W_enc = W_dec.T at every step (no independent encoder
                  params). Parameter-minimal dense baseline.
`dense_warmtied`— independent W_enc params but W_enc initialised to W_dec.T at
                  construction; both drift during training. This matches the
                  real-world SAE-Lens / Anthropic recipe.
`dense_randinit`— fully independent W_enc with random init (no warm start).
                  The strict straw-man baseline: tests whether the warm-tied
                  init does real work.

All three share the same TopK decoder and dead-feature resampling interface.
"""
import numpy as np
import torch
import torch.nn as nn


class DenseTiedSAE(nn.Module):
    """W_enc = W_dec.T. Unique params: m * n."""

    def __init__(self, m: int, n: int, k: int, seed: int = 0):
        super().__init__()
        self.m, self.n, self.k = m, n, k
        self.d = m  # for schema compatibility
        self.arch = "dense_tied"

        torch.manual_seed(seed)
        self.W_dec_param = nn.Parameter(torch.randn(m, n) / float(np.sqrt(m)))
        self.b_dec = nn.Parameter(torch.zeros(m))
        self.b_enc = nn.Parameter(torch.zeros(n))

    @property
    def W_dec(self) -> torch.Tensor:
        norms = self.W_dec_param.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return self.W_dec_param / norms

    @property
    def W_enc(self) -> torch.Tensor:
        return self.W_dec.T

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        pre = (y - self.b_dec) @ self.W_enc.T + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec.T + self.b_dec

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(y)
        return self.decode(h), h

    def loss(self, y: torch.Tensor) -> torch.Tensor:
        y_hat, _ = self.forward(y)
        return (y - y_hat).pow(2).sum(dim=-1).mean()

    def resample_feature(self, j: int, residual: torch.Tensor) -> None:
        with torch.no_grad():
            norm = residual.norm().clamp(min=1e-8)
            self.W_dec_param.data[:, j] = residual / norm
            self.b_enc.data[j] = 0.0


class _DenseIndepBase(nn.Module):
    """Shared forward / training hooks for the two independent-encoder variants.

    Subclasses set `self.W_enc_param` and `self.W_dec_param` in __init__.
    """

    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.m, self.n, self.k = m, n, k
        self.d = m

    @property
    def W_dec(self) -> torch.Tensor:
        norms = self.W_dec_param.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return self.W_dec_param / norms

    @property
    def W_enc(self) -> torch.Tensor:
        return self.W_enc_param

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        pre = (y - self.b_dec) @ self.W_enc.T + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, idx, vals)
        return out

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W_dec.T + self.b_dec

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(y)
        return self.decode(h), h

    def loss(self, y: torch.Tensor) -> torch.Tensor:
        y_hat, _ = self.forward(y)
        return (y - y_hat).pow(2).sum(dim=-1).mean()

    def resample_feature(self, j: int, residual: torch.Tensor) -> None:
        with torch.no_grad():
            norm = residual.norm().clamp(min=1e-8)
            self.W_dec_param.data[:, j] = residual / norm
            self.W_enc_param.data[j] = self.W_dec[:, j].detach()
            self.b_enc.data[j] = 0.0


class DenseWarmTiedSAE(_DenseIndepBase):
    """Independent W_enc + W_dec parameters, warm-tied at init.

    At step 0: W_enc = W_dec.T (so the forward pass is identical to
    `DenseTiedSAE` at that moment). During training both drift independently.
    This is the standard SAE-Lens recipe.

    Unique params: 2 * m * n.
    """

    def __init__(self, m: int, n: int, k: int, seed: int = 0):
        super().__init__(m, n, k)
        self.arch = "dense_warmtied"

        torch.manual_seed(seed)
        W_dec_init = torch.randn(m, n) / float(np.sqrt(m))
        self.W_dec_param = nn.Parameter(W_dec_init)

        dec_norms = W_dec_init.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_enc_init = (W_dec_init / dec_norms).T.contiguous()
        self.W_enc_param = nn.Parameter(W_enc_init)

        self.b_dec = nn.Parameter(torch.zeros(m))
        self.b_enc = nn.Parameter(torch.zeros(n))


class DenseRandinitSAE(_DenseIndepBase):
    """Independent W_enc + W_dec with fully random init. No warm-tie.

    Tests whether the warm-tied init does real work. Historically called
    `dense_indep` in this codebase.

    Unique params: 2 * m * n.
    """

    def __init__(self, m: int, n: int, k: int, seed: int = 0):
        super().__init__(m, n, k)
        self.arch = "dense_randinit"

        torch.manual_seed(seed)
        self.W_dec_param = nn.Parameter(torch.randn(m, n) / float(np.sqrt(m)))
        self.W_enc_param = nn.Parameter(torch.randn(n, m) / float(np.sqrt(m)))
        self.b_dec = nn.Parameter(torch.zeros(m))
        self.b_enc = nn.Parameter(torch.zeros(n))


# Legacy alias — old code / DB entries refer to DenseIndepSAE.
DenseIndepSAE = DenseRandinitSAE
