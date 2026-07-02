"""Plot the per-architecture feature-coherence histogram from the
LLM-eval JSON.

Run: ``venv/bin/python results/plot_feature_coherence.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "results" / "feature_dashboard_eval.json"
FIG_PATH = ROOT / "results" / "figures" / "feature_coherence_histogram.pdf"


def main():
    data = json.loads(JSON_PATH.read_text())
    summary = data["summary"]
    arch_labels = data["config"]["archs"]
    inter_rho = data["inter_rater"]["spearman_rho"]

    # Use overall (averaged across two judges) per-feature mean coherence,
    # binned to integers 1..5.
    per_arch_scores: dict[str, list[float]] = {a: [] for a in arch_labels}
    for rec in data["features"]:
        per_arch_scores[rec["arch_label"]].append(rec["coherence_overall_mean"])

    fig, axes = plt.subplots(1, len(arch_labels), figsize=(11, 3.0),
                             sharey=True)
    bins = np.arange(0.5, 6.0, 1.0)  # 1..5 integer bins

    for ax, label in zip(axes, arch_labels):
        scores = np.array(per_arch_scores[label], dtype=float)
        s = summary[label]
        # Integer histogram (rounded to nearest int for the visual).
        rounded = np.round(scores).astype(int)
        counts = [int(((rounded == b)).sum()) for b in (1, 2, 3, 4, 5)]
        ax.bar([1, 2, 3, 4, 5], counts, width=0.8, edgecolor="black")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("coherence (1-5)")
        ax.set_ylim(0, max(20, max(counts) + 2))
        ax.set_title(f"{label}\nmean = {s['mean']:.2f} $\\pm$ {s['sem']:.2f}, "
                     f"concept = {100*s['concept_frac']:.0f}\\%",
                     fontsize=10)

    axes[0].set_ylabel("# features")
    fig.suptitle(
        f"Blinded LLM evaluation of feature dashboards "
        f"(Pythia-70M layer 3, $k{{=}}64$, seed 0; "
        f"inter-judge Spearman $\\rho = {inter_rho:.2f}$)",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"wrote {FIG_PATH}")


if __name__ == "__main__":
    main()
