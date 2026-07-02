"""Generate paper figures from benchmark_db.json.

Reads only — never writes to the DB. Each figure function is self-contained
and gated by data availability, so partial runs of the pipeline still produce
the figures they can.

Usage:
    python results/plot_all.py              # all figures
    python results/plot_all.py --fig 4      # just the Pareto
    python results/plot_all.py --fig 4 5    # several
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_EFF_D, DATA_EFF_K, M, N, is_recovery_feasible

DB_PATH = "results/benchmark_db.json"
FIG_DIR = Path("results/figures/")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.figsize": (5.5, 3.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLORS_D = {7: "#9467bd", 30: "#d62728", 50: "#ff7f0e",
            100: "#2ca02c", 200: "#8c564b", 512: "#1f77b4"}
MARKER_ARCH = {"expander_tied": "*", "dense_tied": "o",
               "dense_warmtied": "D", "dense_randinit": "s"}
LABEL_ARCH = {
    "expander_tied":  "Expander-SAE",
    "dense_tied":     "Expander-SAE",
    "dense_warmtied": "Dense-SAE",
    "dense_randinit": "Dense (rand-init)",
    "dense_indep":    "Dense (rand-init)",  # legacy alias
}

# Plotting colour for the Dense-SAE (dense_warmtied) reference line:
# always black across all comparisons that include Dense-SAE.
DENSE_SAE_COLOR = "black"


def _arch_label(arch: str, d: int | None = None) -> str:
    """Legend label. Expander-SAE family always carries '(d=…)'; tied dense
    is rendered as 'Expander-SAE (d=m)' to keep the label scheme consistent."""
    base = LABEL_ARCH.get(arch, arch)
    if arch == "expander_tied" and d is not None:
        return f"{base} (d={d})"
    if arch == "dense_tied":
        return "Expander-SAE (d=m)"
    return base


def _arch_color(arch: str, d: int | None = None) -> str:
    """Colour for the (arch, d) legend entry. Dense-SAE is always black;
    Expander-SAE family uses COLORS_D[d]."""
    if arch == "dense_warmtied":
        return DENSE_SAE_COLOR
    return COLORS_D.get(d if d is not None else 512, "gray")


def _legend_sort_key(arch: str, d: int | None) -> tuple:
    """Order legend entries as: Expander-SAE (d=7, 50, 200), then
    Expander-SAE (d=m), then Dense-SAE."""
    if arch == "expander_tied":
        return (0, d if d is not None else 0)
    if arch == "dense_tied":
        return (0, float("inf"))  # sits after expander_tied entries
    if arch == "dense_warmtied":
        return (1, 0)
    return (2, 0)


def _load_db() -> list[dict]:
    with open(DB_PATH) as f:
        return json.load(f)


def _aggregate_seeds(entries: list[dict], metric: str):
    """Group by (arch, m, n, d, k) and return mean/std/count over seeds."""
    buckets = defaultdict(list)
    for e in entries:
        key = (e["architecture"], e["m"], e["n"], e["d"], e["k"])
        v = e["metrics"].get(metric)
        if v is None:
            continue
        buckets[key].append(v)
    out = {}
    for key, vals in buckets.items():
        arr = np.array(vals, dtype=float)
        out[key] = {"mean": float(arr.mean()), "std": float(arr.std()),
                    "min": float(arr.min()), "max": float(arr.max()),
                    "n": len(arr)}
    return out


def _encoder_entries(db: list[dict]) -> list[dict]:
    return [e for e in db if e["inference_method"] == "encoder"]


def _inference_entries(db: list[dict], method: str) -> list[dict]:
    return [e for e in db if e["inference_method"] == method]


# ---------------------------------------------------------------------------
def relerr_vs_k(db: list[dict]) -> None:
    """1x3: per d, x=k, y=rel_err for each inference algorithm.
    Fixed: arch=expander_tied, m=512, n=4096. d in {7, 50, 200}."""
    methods = ["encoder", "omp", "niht", "cosamp"]
    ds_shown = [7, 50, 200]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)

    # Determine common y-range across panels for fairness.
    all_means = []
    for d in ds_shown:
        for method in methods:
            entries = [e for e in db
                       if e["inference_method"] == method
                       and e["architecture"] == "expander_tied"
                       and e["n"] == 4096
                       and e["d"] == d]
            if not entries:
                continue
            agg = _aggregate_seeds(entries, "rel_err_mean")
            for stats in agg.values():
                all_means.append(stats["mean"])
    y_top = max(all_means) * 1.05 if all_means else 1.0
    y_bot = max(min(all_means) * 0.95, 0.0) if all_means else 0.0

    for col_i, d in enumerate(ds_shown):
        ax = axes[col_i]
        for method in methods:
            entries = [e for e in db
                       if e["inference_method"] == method
                       and e["architecture"] == "expander_tied"
                       and e["n"] == 4096
                       and e["d"] == d]
            if not entries:
                continue
            agg = _aggregate_seeds(entries, "rel_err_mean")
            rows = sorted((key[4], stats) for key, stats in agg.items())
            if not rows:
                continue
            ks = [r[0] for r in rows]
            mu = [r[1]["mean"] for r in rows]
            lo = [r[1]["min"] for r in rows]
            hi = [r[1]["max"] for r in rows]
            ax.errorbar(ks, mu, yerr=[np.array(mu) - np.array(lo),
                                      np.array(hi) - np.array(mu)],
                        marker="o", label=method, capsize=3)
        ax.set_title(f"d={d}")
        ax.set_xlabel("k")
        ax.set_ylabel("rel. err" if col_i == 0 else "")
        ax.set_ylim(y_bot, y_top)
        if col_i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Inference procedures on frozen Expander SAE decoders ($m{=}512$, $n{=}4096$)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "relerr_vs_k.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "relerr_vs_k.pdf",
                        paper_dir / "relerr_vs_k.pdf")


_BENCH_CE_ARCHS = ("expander_tied", "dense_tied", "dense_warmtied")


def ce_vs_k(db: list[dict]) -> None:
    """CE-recovered vs k, one line per (arch, d). Fixed: m=512, n=4096.
    Includes Expander-SAE at d in {7, 50, 200}, Expander-SAE-m (dense_tied),
    and Dense-SAE (dense_warmtied). Clustered and pruned controls are excluded."""
    entries = _encoder_entries(db)
    entries = [e for e in entries
               if e["metrics"].get("ce_recovered") is not None
               and e["n"] == 4096
               and e["architecture"] in _BENCH_CE_ARCHS
               and e["d"] not in (30, 100)]
    if not entries:
        print("ce_vs_k: no CE data yet")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    agg = _aggregate_seeds(entries, "ce_recovered")
    by_arch_d = defaultdict(list)
    for (arch, m, n, d, k), s in agg.items():
        by_arch_d[(arch, d)].append((k, s))
    for (arch, d) in sorted(by_arch_d, key=lambda kd: _legend_sort_key(*kd)):
        rows = sorted(by_arch_d[(arch, d)], key=lambda r: r[0])
        ks = [r[0] for r in rows]
        mu = [r[1]["mean"] for r in rows]
        ax.plot(ks, mu, marker=MARKER_ARCH.get(arch, "o"),
                label=_arch_label(arch, d),
                color=_arch_color(arch, d))
    ax.set_xlabel("k")
    ax.set_ylabel("CE-loss recovered")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("CE recovered vs k (m=512, n=4096)")
    ax.legend(fontsize=8, ncol=1, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ce_vs_k.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "ce_vs_k.pdf",
                        paper_dir / "ce_vs_k.pdf")


def relerr_vs_k_encoder_only(db: list[dict]) -> None:
    """Trained-encoder relative reconstruction error vs k for the same
    architectures as ``ce_vs_k`` (Expander-SAE at d in {7, 50, 200},
    Expander-SAE-m, Dense-SAE). No inference-method overlay."""
    entries = [e for e in db
               if e["inference_method"] == "encoder"
               and e.get("metrics", {}).get("rel_err_mean") is not None
               and e["n"] == 4096
               and e["architecture"] in _BENCH_CE_ARCHS
               and e["d"] not in (30, 100)]
    if not entries:
        print("relerr_vs_k_encoder_only: no rel_err data")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    agg = _aggregate_seeds(entries, "rel_err_mean")
    by_arch_d = defaultdict(list)
    for (arch, m, n, d, k), s in agg.items():
        by_arch_d[(arch, d)].append((k, s))
    for (arch, d) in sorted(by_arch_d, key=lambda kd: _legend_sort_key(*kd)):
        rows = sorted(by_arch_d[(arch, d)], key=lambda r: r[0])
        ks = [r[0] for r in rows]
        mu = [r[1]["mean"] for r in rows]
        ax.plot(ks, mu, marker=MARKER_ARCH.get(arch, "o"),
                label=_arch_label(arch, d),
                color=_arch_color(arch, d))
    ax.set_xlabel("k")
    ax.set_ylabel("relative reconstruction error")
    ax.set_title("Reconstruction error vs k (m=512, n=4096)")
    ax.legend(fontsize=8, ncol=1, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "relerr_vs_k_encoder_only.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "relerr_vs_k_encoder_only.pdf",
                        paper_dir / "relerr_vs_k_encoder_only.pdf")


def dead_frac_vs_k(db: list[dict]) -> None:
    """Dead-feature fraction vs k. Fixed: m=512, n=4096.
    Same architecture set and label scheme as ``ce_vs_k``: Expander-SAE at
    d in {7, 50, 200}, Expander-SAE (d=m), and Dense-SAE."""
    entries = _encoder_entries(db)
    entries = [e for e in entries
               if e.get("metrics", {}).get("dead_frac") is not None
               and e["n"] == 4096
               and e["architecture"] in _BENCH_CE_ARCHS
               and e["d"] not in (30, 100)]
    if not entries:
        print("dead_frac_vs_k: no data")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    agg = _aggregate_seeds(entries, "dead_frac")
    by_arch_d = defaultdict(list)
    for (arch, m, n, d, k), s in agg.items():
        by_arch_d[(arch, d)].append((k, s))
    for (arch, d) in sorted(by_arch_d, key=lambda kd: _legend_sort_key(*kd)):
        rows = sorted(by_arch_d[(arch, d)], key=lambda r: r[0])
        ks = [r[0] for r in rows]
        mu = [r[1]["mean"] for r in rows]
        ax.plot(ks, mu, marker=MARKER_ARCH.get(arch, "o"),
                label=_arch_label(arch, d),
                color=_arch_color(arch, d))
    ax.set_xlabel("k")
    ax.set_ylabel("dead feature fraction")
    ax.set_yscale("log")
    ax.set_title("Dead-feature fraction vs k (m=512, n=4096)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dead_frac_vs_k.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "dead_frac_vs_k.pdf",
                        paper_dir / "dead_frac_vs_k.pdf")


def ce_vs_uniq_params(db: list[dict]) -> None:
    """1x3: Expander vs each Dense variant. x=unique params (log), y=CE-recovered.
    Fixed: m=512.

    Error bars: min/max across seeds.
    NSP gate: excludes any (arch, n, k) where n < 2k — at that boundary the
    null-space property degenerates and the "baseline" isn't sparse.
    Per-point annotation shows the varying axes (d,k for Expander; n,k for dense).
    """
    ce_entries = [e for e in _encoder_entries(db)
                  if e["metrics"].get("ce_recovered") is not None]
    if not ce_entries:
        print("ce_vs_uniq_params: CE data not populated; run ce_loss_sweep first")
        return

    ce_entries = [e for e in ce_entries if is_recovery_feasible(e["n"], e["k"])]

    by_arch_cfg = defaultdict(list)
    for e in ce_entries:
        key = (e["architecture"], e["m"], e["n"], e["d"], e["k"])
        by_arch_cfg[key].append(e)

    def _plot_one(ax, arch: str) -> None:
        """Plot markers for a single architecture onto ax; label only the first one."""
        label_used = False
        for (a, m, n, d, k), group in by_arch_cfg.items():
            if a != arch:
                continue
            ces = [g["metrics"]["ce_recovered"] for g in group]
            uniq = group[0]["practical"].get("unique_params")
            if uniq is None or not ces:
                continue
            mu = float(np.mean(ces))
            lo, hi = float(np.min(ces)), float(np.max(ces))
            ax.errorbar([uniq], [mu], yerr=[[mu - lo], [hi - mu]],
                        marker=MARKER_ARCH.get(arch, "o"),
                        markersize=11 if arch == "expander_tied" else 7,
                        color=COLORS_D.get(d if arch == "expander_tied" else 512, "gray"),
                        capsize=3, linestyle="",
                        label=None if label_used else LABEL_ARCH.get(arch, arch))
            label_used = True
            annot = f"d={d},k={k}" if arch == "expander_tied" else f"n={n},k={k}"
            ax.annotate(annot, (uniq, mu), xytext=(5, 3),
                        textcoords="offset points", fontsize=6, alpha=0.7)

    _EXP = LABEL_ARCH["expander_tied"]

    dense_pairs = ["dense_tied", "dense_warmtied", "dense_randinit"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, dense_arch in zip(axes, dense_pairs):
        _plot_one(ax, "expander_tied")
        _plot_one(ax, dense_arch)
        ax.set_xscale("log")
        ax.set_xlabel("unique parameters (log)")
        ax.set_ylabel("CE-loss recovered")
        ax.set_title(f"{_EXP} vs {LABEL_ARCH[dense_arch]}")
        ax.legend(fontsize=9)
        ax.text(0.02, 0.02, "NSP gate: n > 2k",
                transform=ax.transAxes, fontsize=7, alpha=0.6)

    fig.suptitle("m=512")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ce_vs_uniq_params.pdf")
    plt.close(fig)


def learning_curves(db: list[dict]) -> None:
    """Test rel_err vs training step for each (arch, d). Fixed: m=512, n=4096, k=64.

    Reads curves from `training.learning_curve_path` on each entry (populated by
    experiments/learning_curves_sweep.py). One line per (arch, d), shaded band
    across seeds (min/max)."""
    wanted = [e for e in _encoder_entries(db)
              if e["k"] == DATA_EFF_K and e["n"] == 4096
              and "_b" not in e["id"]
              and ((e["architecture"] == "expander_tied" and e["d"] in DATA_EFF_D)
                   or e["architecture"] in ("dense_tied", "dense_warmtied"))]
    curves_by_key: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for e in wanted:
        path = e.get("training", {}).get("learning_curve_path")
        if not path or not Path(path).exists():
            continue
        curves_by_key[(e["architecture"], e["d"])].append(np.load(path))

    if not curves_by_key:
        print("learning_curves: no curves found; run experiments/learning_curves_sweep.py")
        return

    fig, ax = plt.subplots()
    for (arch, d), curves in sorted(curves_by_key.items()):
        # Align on the step axis (assumed identical across seeds).
        steps = curves[0][:, 0]
        stack = np.stack([c[:, 1] for c in curves], axis=0)
        mu = stack.mean(axis=0)
        lo = stack.min(axis=0)
        hi = stack.max(axis=0)
        if arch == "dense_warmtied":
            color = "black"
            linestyle = "--"
        else:
            color = COLORS_D.get(d, "gray")
            linestyle = "-"
        label = _arch_label(arch, d)
        ax.plot(steps, mu, color=color, linestyle=linestyle, label=label)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.18)

    ax.set_xlabel("training step")
    ax.set_ylabel("test rel. err")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Learning curves — m=512, n=4096, k=64")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curves.pdf")
    plt.close(fig)


def novelty_vs_d(db: list[dict]) -> None:
    """Jaccard-novelty fraction across d. Fixed: arch=expander_tied, m=512, n=4096, k=64."""
    entries = [e for e in _encoder_entries(db)
               if e["architecture"] == "expander_tied" and e["k"] == DATA_EFF_K
               and e["features"].get("jaccard_novel_frac_01") is not None]
    if not entries:
        print("novelty_vs_d: feature analysis not yet run")
        return
    fig, ax = plt.subplots()
    rows = sorted((e["d"], e["features"]["jaccard_novel_frac_01"]) for e in entries)
    ds = [r[0] for r in rows]
    fracs = [r[1] for r in rows]
    ax.bar([str(d) for d in ds], fracs, color=[COLORS_D.get(d, "gray") for d in ds])
    ax.set_xlabel("d")
    ax.set_ylabel("fraction of features with Jaccard<0.1 vs Standard-SAE")
    ax.set_title("Feature novelty — Expander-SAE-d, m=512, n=4096, k=64")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "novelty_vs_d.pdf")
    plt.close(fig)


def algorithm_comparison(db: list[dict]) -> None:
    """rel_err bars per d × inference method. Fixed: arch=expander_tied, m=512, n=4096, k=64."""
    methods = ["encoder", "omp", "niht", "cosamp"]
    entries = [e for e in db
               if e["architecture"] == "expander_tied"
               and e["n"] == 4096 and e["k"] == DATA_EFF_K
               and e["inference_method"] in methods]
    if not entries:
        print("algorithm_comparison: no data")
        return
    fig, ax = plt.subplots()
    by_d_method = defaultdict(list)
    for e in entries:
        by_d_method[(e["d"], e["inference_method"])].append(e["metrics"]["rel_err_mean"])

    ds = sorted({d for d, _ in by_d_method})
    x = np.arange(len(ds))
    width = 0.2
    for i, method in enumerate(methods):
        mu = [np.mean(by_d_method.get((d, method), [np.nan])) for d in ds]
        lo = [np.min(by_d_method.get((d, method), [np.nan])) for d in ds]
        hi = [np.max(by_d_method.get((d, method), [np.nan])) for d in ds]
        ax.bar(x + i * width, mu, width, label=method,
               yerr=[np.array(mu) - np.array(lo), np.array(hi) - np.array(mu)])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("d")
    ax.set_ylabel("rel. err")
    ax.set_title("Algorithm comparison — Expander-SAE-d, m=512, n=4096, k=64")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "algorithm_comparison.pdf")
    plt.close(fig)


def matched_params_comparison(db: list[dict]) -> None:
    r"""Budget-matched 2-way comparison: Expander-SAE vs Dense-SAE at the
    same learned-parameter budget for each $d \in \{7, 30, 50, 100, 200\}$
    at $k{=}64$. Dense-SAE uses the matched width $n_{\mathrm{indep}} =
    dN/(2M)$ since it has an independent encoder; configurations failing
    the null-space gate $n > 2k$ are not trained and shown as a blank
    slot with an ``n<2k'' annotation."""
    arch_order = ["expander_tied", "dense_warmtied"]
    arch_label = {
        "expander_tied":  "Expander-SAE",
        "dense_warmtied": "Dense-SAE",
    }
    arch_color = {
        "expander_tied":  "#1f77b4",
        "dense_warmtied": "black",
    }
    # d-ordered budgets to display (cap at d=200 / 819,200 per spec).
    expander_d_budget = [
        (7,   28672),
        (30,  122880),
        (50,  204800),
        (100, 409600),
        (200, 819200),
    ]
    budgets = [b for _, b in expander_d_budget]

    entries = [e for e in _encoder_entries(db)
               if e["metrics"].get("ce_recovered") is not None
               and is_recovery_feasible(e["n"], e["k"])
               and e["k"] == DATA_EFF_K
               and e["architecture"] in arch_order]
    if not entries:
        print("matched_params_comparison: no CE data for k=64")
        return

    by_uniq_arch_n = defaultdict(list)
    for e in entries:
        uniq = e["practical"].get("unique_params")
        if uniq is None or uniq not in budgets:
            continue
        by_uniq_arch_n[(uniq, e["architecture"], e["n"])].append(
            e["metrics"]["ce_recovered"])

    n_arch = len(arch_order)
    width = 0.36
    xpos = list(range(len(budgets)))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, arch in enumerate(arch_order):
        xs, mu, lo, hi, ns = [], [], [], [], []
        offset = (i - (n_arch - 1) / 2) * width
        for j, u in enumerate(budgets):
            cells = [(n, vs) for (uu, a, n), vs in by_uniq_arch_n.items()
                     if uu == u and a == arch]
            if not cells:
                continue
            all_vs = [v for _, vs in cells for v in vs]
            xs.append(xpos[j] + offset)
            mu.append(float(np.mean(all_vs)))
            lo.append(float(np.min(all_vs)))
            hi.append(float(np.max(all_vs)))
            ns.append(cells[0][0])
        if not xs:
            continue
        ax.bar(xs, mu, width, label=arch_label[arch],
               color=arch_color[arch], edgecolor="black", linewidth=0.6,
               yerr=[np.array(mu) - np.array(lo),
                     np.array(hi) - np.array(mu)],
               capsize=3)
        for x, m_val, n in zip(xs, mu, ns):
            ax.text(x, m_val - 0.04, f"$n{{=}}{n}$",
                    ha="center", va="top",
                    fontsize=8, color="white" if arch == "dense_warmtied" else "white",
                    fontweight="bold")

    # Mark NSP-excluded Dense-SAE positions (the matched-$n$ dense width
    # would be d*N/(2*M), with k=DATA_EFF_K).
    dense_offset = (1 - (n_arch - 1) / 2) * width
    for j, (d_exp, u) in enumerate(expander_d_budget):
        n_dense = d_exp * N // (2 * M)
        if not is_recovery_feasible(n_dense, DATA_EFF_K):
            ax.text(xpos[j] + dense_offset, 0.04,
                    f"$n{{=}}{n_dense}$\n($n{{<}}2k$)",
                    ha="center", va="bottom", fontsize=7,
                    color=arch_color["dense_warmtied"], alpha=0.6)

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{u:,}\n($d{{=}}{d_exp}$)"
                        for d_exp, u in expander_d_budget])
    ax.set_xlabel("learned-parameter budget")
    ax.set_ylabel("CE-loss recovered")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Budget-matched comparison ($m{=}512$, $k{=}64$)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "matched_params_comparison.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "matched_params_comparison.pdf",
                        paper_dir / "matched_params_comparison.pdf")


def novelty_vs_k(db: list[dict]) -> None:
    """Jaccard-novelty fraction vs k, for Expander-SAE-d {7,50,200} and
    Expander-SAE-m. Reference: Standard-SAE (dense_warmtied). Fixed:
    m=512, n=4096, seed=0."""
    ds_shown = [7, 50, 200]
    wanted = [e for e in _encoder_entries(db)
              if e["n"] == 4096 and "_b" not in e["id"] and e["seed"] == 0
              and (e.get("features") or {}).get("jaccard_novel_frac_01") is not None
              and ((e["architecture"] == "expander_tied" and e["d"] in ds_shown)
                   or e["architecture"] == "dense_tied")]
    if not wanted:
        print("novelty_vs_k: no feature data; run experiments/feature_analysis.py")
        return

    by_key: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for e in wanted:
        by_key[(e["architecture"], e["d"])].append(
            (e["k"], e["features"]["jaccard_novel_frac_01"]))

    fig, ax = plt.subplots()
    for (arch, d), rows in sorted(by_key.items()):
        rows.sort()
        ks = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        if arch == "dense_tied":
            color = COLORS_D[512]
            linestyle = "--"
        else:
            color = COLORS_D.get(d, "gray")
            linestyle = "-"
        ax.plot(ks, vals, marker="o", linestyle=linestyle, color=color,
                label=_arch_label(arch, d))

    ax.set_xlabel("k")
    ax.set_ylabel("fraction with Jaccard<0.1 vs Standard-SAE")
    ax.set_title("Feature novelty — m=512, n=4096, seed=0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "novelty_vs_k.pdf")
    plt.close(fig)


def decoder_cos_novelty_vs_k(db: list[dict]) -> None:
    """Decoder-cosine novelty (max |cos|<0.3) vs k. Same config as novelty_vs_k."""
    ds_shown = [7, 50, 200]
    wanted = [e for e in _encoder_entries(db)
              if e["n"] == 4096 and "_b" not in e["id"] and e["seed"] == 0
              and (e.get("features") or {}).get("decoder_cos_novel_frac_03") is not None
              and ((e["architecture"] == "expander_tied" and e["d"] in ds_shown)
                   or e["architecture"] == "dense_tied")]
    if not wanted:
        print("decoder_cos_novelty_vs_k: no feature data")
        return

    by_key: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for e in wanted:
        by_key[(e["architecture"], e["d"])].append(
            (e["k"], e["features"]["decoder_cos_novel_frac_03"]))

    fig, ax = plt.subplots()
    for (arch, d), rows in sorted(by_key.items()):
        rows.sort()
        ks = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        if arch == "dense_tied":
            color = COLORS_D[512]
            linestyle = "--"
        else:
            color = COLORS_D.get(d, "gray")
            linestyle = "-"
        ax.plot(ks, vals, marker="o", linestyle=linestyle, color=color,
                label=_arch_label(arch, d))

    ax.set_xlabel("k")
    ax.set_ylabel("fraction with max |cos|<0.3 vs Standard-SAE")
    ax.set_title("Decoder-cosine novelty — m=512, n=4096, seed=0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decoder_cos_novelty_vs_k.pdf")
    plt.close(fig)


def token_entropy_vs_k(db: list[dict]) -> None:
    """Median per-feature target-token entropy (bits) vs k."""
    ds_shown = [7, 50, 200]
    wanted = [e for e in _encoder_entries(db)
              if e["n"] == 4096 and "_b" not in e["id"] and e["seed"] == 0
              and (e.get("features") or {}).get("token_entropy_median") is not None
              and ((e["architecture"] == "expander_tied" and e["d"] in ds_shown)
                   or e["architecture"] == "dense_tied")]
    if not wanted:
        print("token_entropy_vs_k: no feature data")
        return

    by_key: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for e in wanted:
        by_key[(e["architecture"], e["d"])].append(
            (e["k"], e["features"]["token_entropy_median"]))

    fig, ax = plt.subplots()
    for (arch, d), rows in sorted(by_key.items()):
        rows.sort()
        ks = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        if arch == "dense_tied":
            color = COLORS_D[512]
            linestyle = "--"
        else:
            color = COLORS_D.get(d, "gray")
            linestyle = "-"
        ax.plot(ks, vals, marker="o", linestyle=linestyle, color=color,
                label=_arch_label(arch, d))

    ax.set_xlabel("k")
    ax.set_ylabel("median target-token entropy (bits)")
    ax.set_title("Feature monosemanticity proxy — m=512, n=4096, seed=0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "token_entropy_vs_k.pdf")
    plt.close(fig)


def firing_rate_vs_k(db: list[dict]) -> None:
    """Test-time dead-feature fraction vs k, from feature_analysis."""
    ds_shown = [7, 50, 200]
    wanted = [e for e in _encoder_entries(db)
              if e["n"] == 4096 and "_b" not in e["id"] and e["seed"] == 0
              and (e.get("features") or {}).get("dead_frac_test") is not None
              and ((e["architecture"] == "expander_tied" and e["d"] in ds_shown)
                   or e["architecture"] == "dense_tied")]
    if not wanted:
        print("firing_rate_vs_k: no feature data")
        return

    by_key: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for e in wanted:
        by_key[(e["architecture"], e["d"])].append(
            (e["k"], e["features"]["dead_frac_test"]))

    fig, ax = plt.subplots()
    for (arch, d), rows in sorted(by_key.items()):
        rows.sort()
        ks = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        if arch == "dense_tied":
            color = COLORS_D[512]
            linestyle = "--"
        else:
            color = COLORS_D.get(d, "gray")
            linestyle = "-"
        ax.plot(ks, vals, marker="o", linestyle=linestyle, color=color,
                label=_arch_label(arch, d))

    ax.set_xlabel("k")
    ax.set_ylabel("dead feature fraction (128k test tokens)")
    ax.set_yscale("log")
    ax.set_title("Test-time dead features — m=512, n=4096, seed=0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "firing_rate_vs_k.pdf")
    plt.close(fig)


def geometry_diagnostics(db: list[dict]) -> None:
    """1x2 line plot of the certificate ratios over the trained Expander grid.
      Left:  R_id  = 2β²·ε(2k)         (Theorem 1 identifiability ratio).
      Right: R_OMP = β²·ε(k+1)·(2k+1)  (sufficient OMP recovery condition).
    X-axis: k (log). Y-axis: ratio (log). One line per d. Horizontal dashed
    line at y=1 marks the certificate threshold."""
    entries = [e for e in _encoder_entries(db)
               if e["n"] == 4096 and "_b" not in e["id"] and e["seed"] == 0
               and (e.get("geometry") or {}).get("R_est") is not None
               and e["architecture"] == "expander_tied"]
    if not entries:
        print("geometry_diagnostics: no data; run experiments/geometry_diagnostics.py")
        return

    R_OMP_key = "R_OMP" if any(e["geometry"].get("R_OMP") is not None
                                for e in entries) else "R_est"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    panels = [
        (axes[0], "R_id",
         r"$R_{\mathrm{id}} = 2\beta_{\max}^2\,\varepsilon_{\mathrm{greedy}}(2k)$"),
        (axes[1], R_OMP_key,
         r"$R_{\mathrm{OMP}} = \beta_{\max}^2\,\varepsilon_{\mathrm{greedy}}(k{+}1)\,(2k{+}1)$"),
    ]

    for ax, key, ylabel in panels:
        by_d_k: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for e in entries:
            v = e["geometry"].get(key)
            if v is None:
                continue
            by_d_k[e["d"]].append((e["k"], float(v)))
        for d in sorted(by_d_k):
            rows = sorted(by_d_k[d])
            ks = [r[0] for r in rows]
            vals = [r[1] for r in rows]
            ax.plot(ks, vals, marker="o", label=f"d={d}",
                    color=COLORS_D.get(d, "gray"), linewidth=1.5)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle("Geometry diagnostics (m=512, n=4096, seed=0; dashed line = certificate threshold)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_diagnostics.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "geometry_diagnostics.pdf",
                        paper_dir / "geometry_diagnostics.pdf")


LABEL_ARCH["clustered_sparse"] = "Clustered-sparse SAE"
LABEL_ARCH["pruned_retuned_dense"] = "Pruned-retuned dense"


def _baseline_compare(db, target_arch: str, include_eps: bool,
                      k_grid: list[int] | None = None) -> tuple | None:
    """Shared layout for Expander-vs-baseline comparison figures."""
    ds_shown = [7, 50, 200]
    exp = [e for e in _encoder_entries(db)
           if e["architecture"] == "expander_tied" and e["n"] == 4096
           and e["d"] in ds_shown and "_b" not in e["id"]]
    base = [e for e in _encoder_entries(db)
            if e["architecture"] == target_arch and e["n"] == 4096
            and e["d"] in ds_shown and "_b" not in e["id"]]
    if not base:
        print(f"_baseline_compare: no {target_arch} entries yet")
        return None
    return exp, base, ds_shown


def _plot_two_or_three_panel(exp, base, ds_shown, target_arch: str,
                             include_eps: bool, outname: str) -> None:
    # Aggregate by (arch, d, k).
    def agg(rows, metric_getter):
        out = defaultdict(list)
        for e in rows:
            v = metric_getter(e)
            if v is None:
                continue
            out[(e["architecture"], e["d"], e["k"])].append(v)
        return {key: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))))
                for key, v in out.items()}

    n_panels = 3 if include_eps else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.0))

    # Panel A: rel_err
    ax = axes[0]
    rel = agg(exp + base, lambda e: e["metrics"].get("rel_err_mean"))
    for d in ds_shown:
        for arch, ls in [("expander_tied", "-"), (target_arch, "--")]:
            rows = sorted((k, rel[(arch, d, k)])
                          for k in sorted({key[2] for key in rel if key[0] == arch and key[1] == d}))
            if not rows: continue
            ks = [r[0] for r in rows]
            mu = [r[1][0] for r in rows]
            sem = [r[1][1] for r in rows]
            ax.errorbar(ks, mu, yerr=sem, marker="o", linestyle=ls,
                        color=COLORS_D.get(d, "gray"),
                        label=f"{_arch_label(arch, d)}" if arch == "expander_tied" else LABEL_ARCH[target_arch] + f" d={d}")
    ax.set_xlabel("k"); ax.set_ylabel("rel. err")
    ax.set_title("Reconstruction error")
    ax.legend(fontsize=7)

    # Panel B: CE recovered
    ax = axes[1]
    ce = agg(exp + base, lambda e: e["metrics"].get("ce_recovered"))
    for d in ds_shown:
        for arch, ls in [("expander_tied", "-"), (target_arch, "--")]:
            rows = sorted((k, ce[(arch, d, k)])
                          for k in sorted({key[2] for key in ce if key[0] == arch and key[1] == d}))
            if not rows: continue
            ks = [r[0] for r in rows]
            mu = [r[1][0] for r in rows]
            sem = [r[1][1] for r in rows]
            ax.errorbar(ks, mu, yerr=sem, marker="o", linestyle=ls,
                        color=COLORS_D.get(d, "gray"),
                        label=f"{_arch_label(arch, d)}" if arch == "expander_tied" else LABEL_ARCH[target_arch] + f" d={d}")
    ax.set_xlabel("k"); ax.set_ylabel("CE-loss recovered")
    ax.set_title("CE recovered")
    ax.legend(fontsize=7)

    if include_eps:
        ax = axes[2]
        eps = agg(exp + base, lambda e: (e.get("geometry") or {}).get("epsilon_greedy"))
        for d in ds_shown:
            for arch, ls in [("expander_tied", "-"), (target_arch, "--")]:
                rows = sorted((k, eps[(arch, d, k)])
                              for k in sorted({key[2] for key in eps if key[0] == arch and key[1] == d}))
                if not rows: continue
                ks = [r[0] for r in rows]
                mu = [r[1][0] for r in rows]
                sem = [r[1][1] for r in rows]
                ax.errorbar(ks, mu, yerr=sem, marker="o", linestyle=ls,
                            color=COLORS_D.get(d, "gray"),
                            label=f"{_arch_label(arch, d)}" if arch == "expander_tied" else LABEL_ARCH[target_arch] + f" d={d}")
        ax.set_xlabel("k"); ax.set_ylabel(r"$\varepsilon_{\mathrm{greedy}}$")
        ax.set_title("Expansion deficit")
        ax.legend(fontsize=7)

    fig.suptitle(f"Expander-SAE-d vs {LABEL_ARCH[target_arch]} (m=512, n=4096, seeds mean ± SEM)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / outname)
    plt.close(fig)


def expander_vs_clustered_sparse(db: list[dict]) -> None:
    res = _baseline_compare(db, "clustered_sparse", include_eps=True)
    if res is None: return
    exp, base, ds = res
    _plot_two_or_three_panel(exp, base, ds, "clustered_sparse",
                             include_eps=True,
                             outname="expander_vs_clustered_sparse.pdf")


def expander_vs_pruned_retuned_dense(db: list[dict]) -> None:
    """Restyled comparison: one fixed colour per architecture, dotted lines
    for Pruned dense, marker-only distinguishing across d."""
    res = _baseline_compare(db, "pruned_retuned_dense", include_eps=False)
    if res is None:
        return
    exp, base, ds_shown = res

    def agg(rows, metric_getter):
        out = defaultdict(list)
        for e in rows:
            v = metric_getter(e)
            if v is None:
                continue
            out[(e["architecture"], e["d"], e["k"])].append(v)
        return {key: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))))
                for key, v in out.items()}

    style = {
        "expander_tied":        {"color": "#1f77b4", "linestyle": "-",
                                 "label_prefix": "Expander-SAE"},
        "pruned_retuned_dense": {"color": "#d62728", "linestyle": ":",
                                 "label_prefix": "Pruned dense"},
    }
    d_marker = {7: "o", 50: "s", 200: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # Panel A: rel_err
    ax = axes[0]
    rel = agg(exp + base, lambda e: e["metrics"].get("rel_err_mean"))
    for arch, st in style.items():
        for d in ds_shown:
            rows = sorted((k, rel[(arch, d, k)])
                          for k in sorted({key[2] for key in rel
                                           if key[0] == arch and key[1] == d}))
            if not rows:
                continue
            ks = [r[0] for r in rows]
            mu = [r[1][0] for r in rows]
            sem = [r[1][1] for r in rows]
            ax.errorbar(ks, mu, yerr=sem,
                        marker=d_marker.get(d, "o"),
                        linestyle=st["linestyle"], color=st["color"],
                        label=f"{st['label_prefix']} (d={d})")
    ax.set_xlabel("k")
    ax.set_ylabel("rel. err")
    ax.set_title("Reconstruction error")
    ax.legend(fontsize=7, ncol=2)

    # Panel B: CE recovered
    ax = axes[1]
    ce = agg(exp + base, lambda e: e["metrics"].get("ce_recovered"))
    for arch, st in style.items():
        for d in ds_shown:
            rows = sorted((k, ce[(arch, d, k)])
                          for k in sorted({key[2] for key in ce
                                           if key[0] == arch and key[1] == d}))
            if not rows:
                continue
            ks = [r[0] for r in rows]
            mu = [r[1][0] for r in rows]
            sem = [r[1][1] for r in rows]
            ax.errorbar(ks, mu, yerr=sem,
                        marker=d_marker.get(d, "o"),
                        linestyle=st["linestyle"], color=st["color"],
                        label=f"{st['label_prefix']} (d={d})")
    ax.set_xlabel("k")
    ax.set_ylabel("CE-loss recovered")
    ax.set_title("CE recovered")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Expander-SAE vs Pruned dense (m=512, n=4096, seeds mean ± SEM)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "expander_vs_pruned_retuned_dense.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "expander_vs_pruned_retuned_dense.pdf",
                        paper_dir / "expander_vs_pruned_retuned_dense.pdf")


def cross_layer_table(_db: list[dict]) -> None:
    """Print a compact LaTeX-ready table of cross-layer replication results.
    Reads results/cross_layer_replication.json (produced by
    experiments/cross_layer_replication.py) and writes the same content as
    a small bar plot for quick inspection. The table itself is rendered in
    the paper directly from a hand-edited tabular -- this function just
    sanity-checks the numbers and emits a one-figure cross-layer comparison.
    """
    import matplotlib.pyplot as plt
    p = Path("results/cross_layer_replication.json")
    if not p.exists():
        print("cross_layer_table: results/cross_layer_replication.json missing")
        return
    data = json.loads(p.read_text())
    if not data:
        print("cross_layer_table: empty payload")
        return

    # Aggregate over seeds: per (layer, arch, d) → mean rel_err, mean ce_rec.
    agg = defaultdict(list)
    for rec in data:
        agg[(rec["layer"], rec["arch"], rec["d"])].append(rec)

    rows = []
    for key in sorted(agg):
        layer, arch, d = key
        recs = agg[key]
        rer = float(np.mean([r["rel_err_mean"] for r in recs]))
        cer = float(np.mean([r.get("ce_recovered", float("nan")) for r in recs]))
        rows.append((layer, arch, d, rer, cer, len(recs)))

    print(f"{'layer':>5}  {'arch':>16}  {'d':>4}  {'rel_err':>8}  "
          f"{'CE_rec':>8}  {'#seeds':>6}")
    for layer, arch, d, rer, cer, ns in rows:
        print(f"{layer:>5}  {arch:>16}  {d:>4}  {rer:>8.4f}  {cer:>8.4f}  "
              f"{ns:>6}")

    # Two-panel figure: rel-err and CE-recovered vs layer, one bar per arch/d.
    layers = sorted({r[0] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bar_groups = [
        ("Expander-SAE (d=7)",   "expander_tied",  7,    "#9467bd"),
        ("Expander-SAE (d=50)",  "expander_tied",  50,   "#ff7f0e"),
        ("Expander-SAE (d=200)", "expander_tied",  200,  "#8c564b"),
        ("Dense-SAE",            "dense_warmtied", 512,  "black"),
    ]
    n_bars = len(bar_groups)
    width = 0.85 / n_bars
    for ax_i, (metric, ylabel) in enumerate([
        ("rel_err", "rel. err"),
        ("ce_rec",  "CE-loss recovered"),
    ]):
        ax = axes[ax_i]
        xpos = list(range(len(layers)))
        for i, (lab, arch, d, color) in enumerate(bar_groups):
            offset = (i - (n_bars - 1) / 2) * width
            xs, ys = [], []
            for j, layer in enumerate(layers):
                match = [r for r in rows
                         if r[0] == layer and r[1] == arch and r[2] == d]
                if not match:
                    continue
                xs.append(xpos[j] + offset)
                ys.append(match[0][3] if metric == "rel_err" else match[0][4])
            if xs:
                ax.bar(xs, ys, width, label=lab if ax_i == 0 else None,
                       color=color, edgecolor="black", linewidth=0.4)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"layer {layer}" for layer in layers])
        ax.set_ylabel(ylabel)
        if metric == "ce_rec":
            ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Cross-layer replication on Pythia-70M ($k{=}64$, three seeds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_layer_replication.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "cross_layer_replication.pdf",
                        paper_dir / "cross_layer_replication.pdf")


def active_support_collisions(_db: list[dict]) -> None:
    """Active-support collision diagnostic. Reads
    results/active_support_collisions.json produced by
    experiments/active_support_collisions.py and produces a 2-panel figure:
    (left) per-sample empirical deficit by architecture and d;
    (right) duplicate-edge count by architecture and d (log scale)."""
    import matplotlib.pyplot as plt
    p = Path("results/active_support_collisions.json")
    if not p.exists():
        print("active_support_collisions: results/active_support_collisions.json "
              "missing (run experiments/active_support_collisions.py)")
        return
    data = json.loads(p.read_text())
    if not data:
        print("active_support_collisions: empty payload")
        return

    arch_label = {
        "expander_tied":        "Expander-SAE",
        "clustered_sparse":     "Clustered-sparse",
        "pruned_retuned_dense": "Pruned dense",
    }
    arch_color = {
        "expander_tied":        "#1f77b4",
        "clustered_sparse":     "#d62728",
        "pruned_retuned_dense": "#2ca02c",
    }
    arch_order = ["expander_tied", "clustered_sparse", "pruned_retuned_dense"]
    ds_shown = sorted({rec["d"] for rec in data})

    by_arch_d: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in data:
        by_arch_d[(rec["architecture"], rec["d"])].append(rec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n_arch = len(arch_order)
    width = 0.85 / n_arch
    xpos = list(range(len(ds_shown)))
    metrics = [
        (axes[0], "deficit_median", "deficit_p95",
         "active-support deficit  $1 - |\\Gamma(S)|/(d\\,k)$",
         False),
        (axes[1], "duplicates_median", "duplicates_p95",
         "duplicate-edge count  $\\sum_i \\max(0, \\deg_i(S) - 1)$",
         True),
    ]
    for ax, key_med, key_p95, ylabel, log_y in metrics:
        for i, arch in enumerate(arch_order):
            xs, mu, lo, hi = [], [], [], []
            offset = (i - (n_arch - 1) / 2) * width
            for j, d in enumerate(ds_shown):
                seeds = by_arch_d.get((arch, d), [])
                if not seeds:
                    continue
                meds = [s[key_med] for s in seeds]
                p95s = [s[key_p95] for s in seeds]
                xs.append(xpos[j] + offset)
                mu.append(float(np.mean(meds)))
                lo.append(float(np.min(meds)))
                hi.append(float(np.max(p95s)))
            if not xs:
                continue
            mu_arr = np.array(mu)
            lo_arr = np.minimum(np.array(lo), mu_arr)
            hi_arr = np.maximum(np.array(hi), mu_arr)
            ax.bar(xs, mu_arr, width, label=arch_label[arch],
                   color=arch_color[arch], edgecolor="black", linewidth=0.4,
                   yerr=[mu_arr - lo_arr, hi_arr - mu_arr],
                   capsize=2)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"d={d}" for d in ds_shown])
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Active-support collisions on held-out activations "
                 "($k{=}64$, mean over seeds; bar: median, error: min--p95)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "active_support_collisions.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "active_support_collisions.pdf",
                        paper_dir / "active_support_collisions.pdf")


def novelty_null(_db: list[dict]) -> None:
    """Firing-rate-matched novelty null. Reads results/novelty_null.json
    produced by experiments/novelty_null.py and renders a 2-panel figure:
      Left:  per-d novelty fraction observed vs firing-rate-matched null,
             stratified by firing-rate decile.
      Right: per-d aggregate novelty fractions (overall observed vs null).
    """
    import matplotlib.pyplot as plt
    p = Path("results/novelty_null.json")
    if not p.exists():
        print("novelty_null: results/novelty_null.json not found "
              "(run experiments/novelty_null.py)")
        return
    data = json.loads(p.read_text())
    if not data:
        print("novelty_null: empty payload")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    for rec in data:
        d = rec["d"]
        deciles = rec["deciles"]
        if not deciles:
            continue
        x = [r["decile"] for r in deciles]
        obs = [r["obs_novel_frac"] for r in deciles]
        nul = [r["null_novel_frac"] for r in deciles]
        c = COLORS_D.get(d, "gray")
        ax.plot(x, obs, marker="o", linestyle="-", color=c,
                label=f"observed d={d}")
        ax.plot(x, nul, marker="s", linestyle="--", color=c, alpha=0.6,
                label=f"null d={d}")
    ax.set_xlabel("firing-rate decile (1=rarest, 10=most frequent)")
    ax.set_ylabel("novelty fraction (best-Jaccard $<$ 0.1)")
    ax.set_title("Activation novelty vs firing-rate-matched null, by decile")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    # Optional dense-vs-dense seed-variation calibration.
    dvd_path = Path("results/novelty_null_dense_vs_dense.json")
    dvd = None
    if dvd_path.exists():
        dvd_data = json.loads(dvd_path.read_text())
        if dvd_data:
            dvd = dvd_data[0]

    ax2 = axes[1]
    ds = [rec["d"] for rec in data]
    obs_o = [rec["obs_novel_frac"] for rec in data]
    nul_o = [rec["null_novel_frac"] for rec in data]
    x_pos = np.arange(len(ds))
    width = 0.36
    ax2.bar(x_pos - width / 2, obs_o, width, label="observed",
            color=[COLORS_D.get(d, "gray") for d in ds])
    ax2.bar(x_pos + width / 2, nul_o, width, label="null",
            color=[COLORS_D.get(d, "gray") for d in ds],
            alpha=0.45, hatch="//")
    if dvd is not None:
        ax2.axhline(dvd["obs_novel_frac"], color="black", linestyle=":",
                    linewidth=1.5,
                    label=f"dense seed-variation ({dvd['obs_novel_frac']:.2f})")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"d={d}" for d in ds])
    ax2.set_ylabel("overall novelty fraction")
    ax2.set_title("Overall")
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=7, loc="lower left")

    fig.suptitle("Firing-rate-matched novelty null (Expander vs dense tied, $k{=}64$, seed 0)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "novelty_null.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "novelty_null.pdf",
                        paper_dir / "novelty_null.pdf")


def support_structure_3panel(db: list[dict]) -> None:
    """3-panel Expander-SAE vs Clustered-sparse comparison for the paper,
    plus the Expander-SAE (d=m) and Dense-SAE references.

    Panels: rel-err, CE recovered, dead-feature fraction (log scale).
    Pruned-retuned dense is excluded; Clustered-sparse uses dashed lines.
    """
    ds_shown = [7, 50, 200]
    rows_clustered = [e for e in _encoder_entries(db)
                      if e["architecture"] == "clustered_sparse"
                      and e["n"] == 4096 and e["d"] in ds_shown
                      and "_b" not in e.get("id", "")]
    rows_expander = [e for e in _encoder_entries(db)
                     if e["architecture"] == "expander_tied"
                     and e["n"] == 4096 and e["d"] in ds_shown
                     and "_b" not in e.get("id", "")]
    rows_warm = [e for e in _encoder_entries(db)
                 if e["architecture"] == "dense_warmtied" and e["n"] == 4096]

    # Match the styling used by ``expander_vs_pruned_retuned_dense``: one
    # fixed colour per architecture, marker distinguishes d, line style
    # distinguishes Expander (solid) vs Clustered (dashed) vs Dense (solid).
    EXPANDER_COLOR  = "#1f77b4"  # blue
    CLUSTERED_COLOR = "#d62728"  # red
    d_marker = {7: "o", 50: "s", 200: "^"}

    def _plot_family(ax, rows, metric, *, color, linestyle, label_prefix, alpha=1.0):
        agg = _aggregate_seeds(rows, metric)
        by_arch_d = defaultdict(list)
        for (arch, m, n, d, k), s in agg.items():
            by_arch_d[(arch, d)].append((k, s))
        for (arch, d) in sorted(by_arch_d, key=lambda kd: _legend_sort_key(*kd)):
            pts = sorted(by_arch_d[(arch, d)], key=lambda r: r[0])
            ks = [p[0] for p in pts]
            mu = [p[1]["mean"] for p in pts]
            ax.plot(ks, mu,
                    marker=d_marker.get(d, "o"),
                    linestyle=linestyle, color=color,
                    label=f"{label_prefix} (d={d})",
                    markersize=5, linewidth=1.5, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("rel_err_mean", "rel. err",              "Reconstruction error", False),
        ("ce_recovered", "CE-loss recovered",     "CE recovered",         False),
        ("dead_frac",    "dead-feature fraction", "Dead features",        True),
    ]

    for ax, (metric, ylabel, title, log_y) in zip(axes, panels):
        # Expander-SAE family — one fixed blue, solid, markers distinguish d.
        _plot_family(ax, rows_expander, metric,
                     color=EXPANDER_COLOR, linestyle="-",
                     label_prefix="Expander-SAE")
        # Clustered-sparse — fixed red, dashed, markers distinguish d.
        _plot_family(ax, rows_clustered, metric,
                     color=CLUSTERED_COLOR, linestyle="--",
                     label_prefix="Clustered-sparse", alpha=0.9)
        # Dense-SAE — black reference (no d annotation needed).
        agg_warm = _aggregate_seeds(rows_warm, metric)
        if agg_warm:
            pts = sorted((key[4], s) for key, s in agg_warm.items())
            ks = [p[0] for p in pts]
            mu = [p[1]["mean"] for p in pts]
            ax.plot(ks, mu, marker="D", linestyle="-",
                    color=DENSE_SAE_COLOR, label="Dense-SAE",
                    markersize=5, linewidth=1.5)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")

    axes[0].legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle("Support-structure controls (m=512, n=4096; mean over seeds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "support_structure_3panel.pdf")
    plt.close(fig)
    paper_dir = Path("paper/figures")
    if paper_dir.exists():
        import shutil
        shutil.copyfile(FIG_DIR / "support_structure_3panel.pdf",
                        paper_dir / "support_structure_3panel.pdf")


def synthetic_omp_recovery(_db: list[dict]) -> None:
    """Reads results/synthetic_db.json (disjoint from benchmark_db.json)."""
    synth_path = "results/synthetic_db.json"
    if not Path(synth_path).exists():
        print("synthetic_omp_recovery: results/synthetic_db.json missing")
        return
    import json as _json
    with open(synth_path) as f:
        synth = _json.load(f)
    if not synth:
        print("synthetic_omp_recovery: empty synthetic DB")
        return

    sigmas = sorted({e["sigma"] for e in synth})
    ds = sorted({e["d"] for e in synth})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5.5 * len(sigmas), 4.0),
                             sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    for ax, sig in zip(axes, sigmas):
        for d in ds:
            rows = [e for e in synth if e["d"] == d and e["sigma"] == sig]
            by_k = defaultdict(list)
            for e in rows:
                by_k[e["k_synth"]].append(e["metrics"]["support_recovery_rate"])
            ks_l = sorted(by_k)
            mu = [float(np.mean(by_k[k])) for k in ks_l]
            sem = [float(np.std(by_k[k]) / np.sqrt(max(len(by_k[k]), 1))) for k in ks_l]
            ax.errorbar(ks_l, mu, yerr=sem, marker="o", label=f"d={d}",
                        color=COLORS_D.get(d, "gray"))
        ax.set_xlabel("$k_{\\mathrm{synth}}$")
        ax.set_title(f"σ = {sig}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")
    axes[0].set_ylabel("exact support recovery rate")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Synthetic OMP recovery on d-regular Expander decoders (m=512, n=4096)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_omp_recovery.pdf")
    plt.close(fig)


def synthetic_omp_reconstruction_error(_db: list[dict]) -> None:
    synth_path = "results/synthetic_db.json"
    if not Path(synth_path).exists():
        print("synthetic_omp_reconstruction_error: missing synthetic DB")
        return
    import json as _json
    with open(synth_path) as f:
        synth = _json.load(f)
    if not synth:
        return

    sigmas = sorted({e["sigma"] for e in synth})
    ds = sorted({e["d"] for e in synth})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5.5 * len(sigmas), 4.0),
                             sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    for ax, sig in zip(axes, sigmas):
        for d in ds:
            rows = [e for e in synth if e["d"] == d and e["sigma"] == sig]
            by_k = defaultdict(list)
            for e in rows:
                by_k[e["k_synth"]].append(e["metrics"]["recon_err_mean"])
            ks_l = sorted(by_k)
            mu = [float(np.mean(by_k[k])) for k in ks_l]
            sem = [float(np.std(by_k[k]) / np.sqrt(max(len(by_k[k]), 1))) for k in ks_l]
            ax.errorbar(ks_l, mu, yerr=sem, marker="o", label=f"d={d}",
                        color=COLORS_D.get(d, "gray"))
        ax.set_xlabel("$k_{\\mathrm{synth}}$")
        ax.set_title(f"σ = {sig}")
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_ylabel("relative reconstruction error")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Synthetic OMP reconstruction error on d-regular Expander decoders")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_omp_reconstruction_error.pdf")
    plt.close(fig)


FIGURES = {
    1:  ("relerr vs k",              relerr_vs_k),
    2:  ("CE vs k",                  ce_vs_k),
    3:  ("dead-feature vs k",        dead_frac_vs_k),
    4:  ("CE vs unique params",      ce_vs_uniq_params),
    5:  ("learning curves",          learning_curves),
    6:  ("novelty vs d",             novelty_vs_d),
    7:  ("algorithm comparison",     algorithm_comparison),
    8:  ("matched-params comparison", matched_params_comparison),
    9:  ("Jaccard novelty vs k",     novelty_vs_k),
    10: ("decoder-cos novelty vs k", decoder_cos_novelty_vs_k),
    11: ("token entropy vs k",       token_entropy_vs_k),
    12: ("test-time dead vs k",      firing_rate_vs_k),
    13: ("geometry diagnostics",     geometry_diagnostics),
    14: ("Expander vs clustered-sparse",    expander_vs_clustered_sparse),
    15: ("Expander vs pruned-retuned dense", expander_vs_pruned_retuned_dense),
    16: ("synthetic OMP recovery",         synthetic_omp_recovery),
    17: ("synthetic OMP reconstruction",   synthetic_omp_reconstruction_error),
    18: ("support-structure 3-panel",      support_structure_3panel),
    19: ("firing-rate-matched novelty null", novelty_null),
    20: ("encoder-only rel-err vs k",      relerr_vs_k_encoder_only),
    21: ("active-support collisions",      active_support_collisions),
    22: ("cross-layer replication",        cross_layer_table),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fig", type=int, nargs="*", default=list(FIGURES.keys()))
    args = p.parse_args()
    db = _load_db()
    for num in args.fig:
        name, fn = FIGURES[num]
        print(f"[{num}] {name}...")
        fn(db)
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
