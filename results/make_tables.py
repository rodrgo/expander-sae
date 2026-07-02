"""Emit LaTeX table fragments from benchmark_db.json.

Outputs (under results/tables/):
  main_results.tex          — primary paper table (rel_err + CE + dead%)
  pareto.tex                — compact version of the Pareto figure
  practical.tex             — storage, timing, param counts
  data_efficiency.tex       — (d, data_budget) grid w/ samples-to-asymptote col
  encoder_mask_ablation.tex — from imported encoder_mask entries

Usage:
    python results/make_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_EFF_D, DATA_EFF_K, M, N

DB_PATH = "results/benchmark_db.json"
TABLES_DIR = Path("results/tables/")
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _load_db() -> list[dict]:
    return json.loads(Path(DB_PATH).read_text())


def _fmt_mean_std(vs: list[float], digits: int = 3) -> str:
    if not vs:
        return "---"
    mu, sd = np.mean(vs), np.std(vs)
    return rf"${mu:.{digits}f}\pm{sd:.{digits}f}$"


def _samples_to_asymptote(db: list[dict], arch: str, d: int, n: int, k: int,
                          threshold: float = 0.9) -> int | None:
    """Smallest data_budget at which rel_err is within (1-threshold) of the
    best (lowest) rel_err achieved for this (arch, n, d, k) across budgets."""
    rows = [e for e in db if e["inference_method"] == "encoder"
            and e["architecture"] == arch and e["d"] == d
            and e["n"] == n and e["k"] == k]
    if not rows:
        return None
    by_budget = defaultdict(list)
    for e in rows:
        by_budget[e["training"]["data_budget"]].append(e["metrics"]["rel_err_mean"])
    means = {b: float(np.mean(v)) for b, v in by_budget.items()}
    best = min(means.values())
    target = best + (1.0 - threshold) * abs(best)
    for b in sorted(means):
        if means[b] <= target:
            return b
    return None


def main_results(db: list[dict]) -> str:
    """Row per (arch, d) at k=DATA_EFF_K, columns for rel_err (encoder, OMP) + CE + dead."""
    k = DATA_EFF_K
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Arch & $d$ & Unique params & Rel.\ err.\ (enc) & Rel.\ err.\ (OMP) & CE recovered & Dead \% \\",
        r"\midrule",
    ]

    def add_row(arch: str, d: int, n: int) -> None:
        enc_rows = [e for e in db if e["inference_method"] == "encoder"
                    and e["architecture"] == arch and e["d"] == d
                    and e["n"] == n and e["k"] == k]
        omp_rows = [e for e in db if e["inference_method"] == "omp"
                    and e["architecture"] == arch and e["d"] == d
                    and e["n"] == n and e["k"] == k]
        if not enc_rows:
            return
        enc_vals = [e["metrics"]["rel_err_mean"] for e in enc_rows]
        omp_vals = [e["metrics"]["rel_err_mean"] for e in omp_rows]
        ce_vals = [e["metrics"].get("ce_recovered") for e in enc_rows
                   if e["metrics"].get("ce_recovered") is not None]
        dead_vals = [e["metrics"]["dead_frac"] * 100 for e in enc_rows]
        uniq = enc_rows[0]["practical"].get("unique_params", "---")
        lines.append(
            f"{arch.replace('_', ' ')} & {d} & {uniq:,} & "
            f"{_fmt_mean_std(enc_vals)} & {_fmt_mean_std(omp_vals)} & "
            f"{_fmt_mean_std(ce_vals) if ce_vals else '---'} & "
            f"{_fmt_mean_std(dead_vals, 1)} \\\\"
        )

    for d in sorted({e["d"] for e in db if e["architecture"] == "expander_tied"}):
        add_row("expander_tied", d, N)
    add_row("dense_tied", M, N)
    add_row("dense_warmtied", M, N)
    add_row("dense_randinit", M, N)

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def pareto_table(db: list[dict]) -> str:
    rows = []
    for e in db:
        if e["inference_method"] != "encoder":
            continue
        if e["metrics"].get("ce_recovered") is None:
            continue
        rows.append({
            "arch": e["architecture"], "n": e["n"], "d": e["d"], "k": e["k"],
            "uniq": e["practical"].get("unique_params"),
            "ce": e["metrics"]["ce_recovered"],
            "relerr": e["metrics"]["rel_err_mean"],
        })
    rows.sort(key=lambda r: (r["uniq"] or 0))
    lines = [r"\begin{tabular}{lrrrr}",
             r"\toprule",
             r"Arch & $n$ & $k$ & Unique params & CE recovered \\",
             r"\midrule"]
    for r in rows:
        lines.append(
            f"{r['arch'].replace('_', ' ')} & {r['n']} & {r['k']} & "
            f"{r['uniq']:,} & {r['ce']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def practical_table(db: list[dict]) -> str:
    entries = [e for e in db if e["inference_method"] == "encoder"
               and e["k"] == DATA_EFF_K and e["seed"] == 0
               and e["practical"].get("storage_decoder_kb") is not None]
    entries.sort(key=lambda e: (e["architecture"], e["d"]))
    lines = [r"\begin{tabular}{lrrrr}",
             r"\toprule",
             r"Arch & $d$ & Decoder params & Storage (KB) & Enc ms/sample \\",
             r"\midrule"]
    for e in entries:
        lines.append(
            f"{e['architecture'].replace('_', ' ')} & {e['d']} & "
            f"{e['practical']['decoder_params']:,} & "
            f"{e['practical']['storage_decoder_kb']:.0f} & "
            f"{e['practical']['inference_ms_per_sample']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def data_efficiency_table(db: list[dict]) -> str:
    """k=DATA_EFF_K grid. Rows: (arch, d). Cols: data budgets. Final col: samples-to-asymptote (90%)."""
    from config import DATA_BUDGETS
    rows = []
    for d in DATA_EFF_D:
        rows.append(("expander_tied", d, N))
    rows.append(("dense_tied", M, N))
    budgets = sorted(DATA_BUDGETS)

    header = "Arch & $d$ & " + " & ".join(f"{b//1000}k" for b in budgets) + r" & \#-samples@90\% \\"
    lines = [
        r"\begin{tabular}{ll" + "r" * len(budgets) + r"l}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for arch, d, n in rows:
        cells = []
        for b in budgets:
            vals = [e["metrics"]["rel_err_mean"] for e in db
                    if e["inference_method"] == "encoder"
                    and e["architecture"] == arch and e["d"] == d
                    and e["n"] == n and e["k"] == DATA_EFF_K
                    and e["training"]["data_budget"] == b]
            cells.append(f"{np.mean(vals):.3f}" if vals else "---")
        asy = _samples_to_asymptote(db, arch, d, n, DATA_EFF_K)
        asy_s = f"{asy//1000}k" if asy is not None else "---"
        lines.append(f"{arch.replace('_',' ')} & {d} & " + " & ".join(cells) + f" & {asy_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def encoder_mask_ablation_table(db: list[dict]) -> str:
    """Compare encoder-mode variants at (d, k) pairs present in the ablation data."""
    variants = {
        "expander_encmask_none": "dense encoder",
        "expander_encmask_tied": "tied-support (headline)",
        "expander_encmask_indep": "independent d-regular",
    }
    rows = [e for e in db if e["architecture"] in variants
            and e["inference_method"] == "encoder"]
    if not rows:
        return "% (no imported encoder_mask data)"

    by_key = defaultdict(list)
    for e in rows:
        by_key[(e["d"], e["k"], e["architecture"])].append(e)

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"$d$ & $k$ & Encoder variant & Rel.\ err.\ & CE recovered \\",
        r"\midrule",
    ]
    for (d, k, arch) in sorted(by_key, key=lambda kk: (kk[0], kk[1], kk[2])):
        entries = by_key[(d, k, arch)]
        errs = [e["metrics"]["rel_err_mean"] for e in entries]
        ces = [e["metrics"].get("ce_recovered") for e in entries
               if e["metrics"].get("ce_recovered") is not None]
        lines.append(
            f"{d} & {k} & {variants[arch]} & "
            f"{_fmt_mean_std(errs)} & "
            f"{_fmt_mean_std(ces) if ces else '---'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=str, default=None,
                   choices=[None, "main", "pareto", "practical",
                            "data_efficiency", "encoder_mask"])
    args = p.parse_args()

    db = _load_db()
    writers = {
        "main": ("main_results.tex", main_results),
        "pareto": ("pareto.tex", pareto_table),
        "practical": ("practical.tex", practical_table),
        "data_efficiency": ("data_efficiency.tex", data_efficiency_table),
        "encoder_mask": ("encoder_mask_ablation.tex", encoder_mask_ablation_table),
    }
    to_write = [args.table] if args.table else list(writers)
    for key in to_write:
        fname, fn = writers[key]
        Path(TABLES_DIR / fname).write_text(fn(db))
        print(f"wrote {TABLES_DIR / fname}")


if __name__ == "__main__":
    main()
