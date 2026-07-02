"""Training loop shared by all SAE architectures.

The models expose `.loss(batch)` and `.resample_feature(j, residual)`; this
module handles the optimiser, scheduler, batching, and dead-feature
resampling.
"""
import time
import numpy as np
import torch


def train_sae(model, train_acts: np.ndarray, steps: int = 5000,
              batch_size: int = 256, lr_max: float = 3e-4, lr_min: float = 1e-5,
              grad_clip: float = 1.0, resample_interval: int = 1000,
              test_acts: np.ndarray | None = None, eval_every: int = 0,
              device: str = "cpu") -> tuple:
    """Train any SAE with a .loss(y) method. Returns (model, info).

    When `test_acts` and `eval_every > 0` are set, records test-set rel_err
    at step 0 (pre-training) and every `eval_every` steps thereafter. The
    resulting learning curve is returned as info["learning_curve"], an
    (T, 2) float32 array of (step, rel_err_mean) rows.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_max, betas=(0.9, 0.999))

    def lr_at(step: int) -> float:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * step / steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_at(s) / lr_max)

    train_tensor = torch.from_numpy(train_acts).float().to(device)
    n_train = len(train_tensor)
    feature_counts = torch.zeros(model.n, device=device)

    log_curve = test_acts is not None and eval_every > 0
    test_tensor = torch.from_numpy(test_acts).float().to(device) if log_curve else None
    curve: list[tuple[int, float]] = []

    def _eval_test() -> float:
        model.eval()
        with torch.no_grad():
            y_hat, _ = model(test_tensor)
            err = (torch.norm(test_tensor - y_hat, dim=-1) /
                   torch.norm(test_tensor, dim=-1).clamp(min=1e-12))
            out = float(err.mean())
        model.train()
        return out

    if log_curve:
        curve.append((0, _eval_test()))

    t0 = time.time()
    last_loss = float("nan")
    for step in range(steps):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        batch = train_tensor[idx]

        loss = model.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        last_loss = loss.item()

        with torch.no_grad():
            _, h = model(batch)
            feature_counts += (h.abs() > 1e-6).float().sum(dim=0)

        if (step + 1) % resample_interval == 0:
            _resample_dead(model, batch, feature_counts, device)
            feature_counts.zero_()

        if log_curve and (step + 1) % eval_every == 0:
            curve.append((step + 1, _eval_test()))

    wall = time.time() - t0

    # Final metrics on a fresh batch from training data.
    with torch.no_grad():
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        final_batch = train_tensor[idx]
        _, h = model(final_batch)
        dead_frac = (h.abs().sum(dim=0) == 0).float().mean().item()

    model = model.cpu()
    info = {
        "wall_clock_s": wall,
        "final_loss": last_loss,
        "dead_frac_train": dead_frac,
    }
    if curve:
        info["learning_curve"] = np.array(curve, dtype=np.float32)
    return model, info


def _resample_dead(model, batch: torch.Tensor, feature_counts: torch.Tensor,
                   device: str, min_count: int = 5) -> None:
    """Re-aim dead features at the highest-residual samples in the batch.

    A feature is "dead" if it fired < min_count times since the last resample.
    Each dead feature j is pointed at the residual (y - y_hat) of a sample
    chosen with probability proportional to its squared error.
    """
    with torch.no_grad():
        dead = feature_counts < min_count
        dead_idx = torch.where(dead)[0]
        n_dead = int(dead_idx.numel())
        if n_dead == 0 or n_dead > 0.8 * model.n:
            return

        y_hat, _ = model(batch)
        residuals_sq = (batch - y_hat).pow(2).sum(dim=-1) + 1e-12

        for j in dead_idx.tolist():
            sample_idx = int(torch.multinomial(residuals_sq, 1).item())
            r = batch[sample_idx] - y_hat[sample_idx]
            model.resample_feature(j, r)
