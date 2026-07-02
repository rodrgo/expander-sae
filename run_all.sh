#!/usr/bin/env bash
# run_all.sh — reproduce the Expander SAE paper.
#
# Three tiers:
#   --figures  (default)  Rebuild every figure and table from the shipped
#                         results/*.json. No GPU, no Modal, runs in seconds.
#   --smoke               Train one tiny SAE on synthetic data locally and
#                         plot it. Proves the train -> infer -> plot path works
#                         with no Modal/GPU/HF.
#   --full                Re-run the entire pipeline from scratch on Modal GPUs.
#                         Requires a Modal account, GPU quota, an HF token for
#                         the gated Llama-3.2-1B run, and (optionally) OpenAI +
#                         Anthropic keys for the blinded coherence eval.
#
# Always run from the repository root.
set -euo pipefail
MODE="${1:---figures}"

# Prefer `python`, fall back to `python3`. Override with PYTHON=... if needed.
PY="${PYTHON:-$(command -v python || command -v python3)}"
if [ -z "$PY" ]; then echo "No python interpreter found." >&2; exit 1; fi

plots() {
    "$PY" results/plot_all.py
    "$PY" results/make_tables.py
    "$PY" results/plot_feature_coherence.py
    echo
    echo "Figures: results/figures/    Tables: results/tables/"
}

run_figures() {
    echo "=== Figures/tables tier (no GPU) ==="
    plots
}

run_smoke() {
    echo "=== Smoke tier (local, synthetic, no Modal) ==="
    "$PY" experiments/make_synthetic_data.py
    "$PY" experiments/training_sweep.py --smoke
    "$PY" experiments/inference_sweep.py --methods omp
    "$PY" results/plot_all.py --fig 4
    echo "Smoke run complete — see results/figures/."
}

run_full() {
    echo "=== Full pipeline tier (Modal GPU) ==="

    # 1. Extract Pythia-70M layer-3 activations to the Modal volume.
    modal run experiments/extract_activations.py::main

    # 2. Train every (arch, m, n, d, k, seed) config.
    modal run experiments/training_sweep.py::sweep

    # 3. OMP / NIHT / CoSaMP on every trained model (local, CPU).
    "$PY" experiments/inference_sweep.py

    # 4. CE-loss recovered (Modal GPU).
    modal run experiments/ce_loss_sweep.py::main

    # 5. Cross-model / cross-layer scaling (Modal GPU; Llama needs HF_TOKEN).
    modal run experiments/scaling_pythia160m.py::main
    modal run experiments/scaling_qwen2_5_3b.py::main
    modal run experiments/scaling_llama32_1b.py::main
    modal run --detach experiments/scaling_omp_ce.py::main --tag qwen25_3b_layer12  --iterative
    modal run --detach experiments/scaling_omp_ce.py::main --tag llama32_1b_layer12 --iterative

    # 6. Diagnostics + analysis (local).
    "$PY" experiments/feature_analysis.py
    "$PY" experiments/practical_benchmarks.py
    "$PY" experiments/learning_curves_sweep.py
    "$PY" experiments/geometry_diagnostics.py
    "$PY" experiments/synthetic_omp_recovery.py
    "$PY" experiments/active_support_collisions.py
    "$PY" experiments/novelty_null.py
    "$PY" experiments/feature_dashboard_eval.py   # needs OPENAI_API_KEY + ANTHROPIC_API_KEY

    # 7. Structured-OMP decoding benchmarks (Appendix C throughput tables).
    #    Each writes a results/*.csv (not committed); the Appendix C tables
    #    (encoder-vs-OMP-variants and the block-size L sweep) are typeset by
    #    hand from these. structured_omp_throughput runs locally on CPU; the
    #    rest are Modal A10G. profile_oneshot.py is a per-op profiling helper
    #    that feeds no table and is not run here.
    "$PY" experiments/structured_omp_throughput.py          # vanilla vs structured OMP (CPU rows)
    modal run experiments/batched_omp_gpu_throughput.py::main   # batched structured OMP+QR (GPU rows)
    modal run experiments/gomp_sweep.py::main                   # generalised (parallel) OMP L-sweep
    modal run experiments/multiblock_cholesky_sweep.py::main    # Cholesky-refit variant
    modal run experiments/multiblock_L_sweep.py::main           # block-size L Pareto sweep (d=7)

    # 8. Figures + tables from the freshly populated database.
    plots
}

case "$MODE" in
    --figures) run_figures ;;
    --smoke)   run_smoke ;;
    --full)    run_full ;;
    *) echo "usage: $0 [--figures | --smoke | --full]"; exit 1 ;;
esac
