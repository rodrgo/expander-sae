# Expander Sparse Autoencoders

Reproduction code for **Expander Sparse Autoencoders: Parameter-Efficient Sparse
Dictionaries for Mechanistic Interpretability**.

Expander SAEs are tied-weight TopK sparse autoencoders whose decoder support is
fixed by the adjacency matrix of a left **d-regular expander graph**. This gives a
one-parameter family indexed by the column degree `d` that trades reconstruction
fidelity for a large reduction in learned decoder values (`dn` instead of `mn`),
while keeping the sparse-coding problem `(m, n, k)` fixed. The repo trains the
architecture, decodes it with OMP / NIHT / CoSaMP (including GPU-batched structured
OMP), and reproduces every figure and table in the paper.

## Quickstart — rebuild every figure and table (no GPU)

The precomputed results are shipped in `results/*.json`, so all 16 figures and the
table fragments rebuild on a laptop in seconds:

```bash
pip install -r requirements.txt   # numpy / scipy / matplotlib suffice for this tier
./run_all.sh                      # == ./run_all.sh --figures
```

Outputs land in `results/figures/` and `results/tables/`.

## Reproduction tiers

`run_all.sh` has three tiers (always run it from the repo root):

| Tier | Command | Needs | What it does |
|------|---------|-------|--------------|
| Figures | `./run_all.sh --figures` | CPU only | Rebuilds all figures/tables from shipped JSON |
| Smoke | `./run_all.sh --smoke` | CPU only | Trains one tiny SAE on synthetic data → infer → plot (proves the path works) |
| Full | `./run_all.sh --full` | Modal GPU + HF token + API keys | Re-runs the entire pipeline from scratch |

### Full pipeline prerequisites

- **Modal** — the GPU steps (activation extraction, training, CE-loss, scaling,
  decoding benchmarks) run on [Modal](https://modal.com). Run `modal token new` once.
- **HuggingFace token** — the gated Llama-3.2-1B scaling run needs `HF_TOKEN`.
  Register it as a Modal secret: `modal secret create huggingface HF_TOKEN=hf_xxx`.
- **OpenAI + Anthropic keys** — only for the blinded LLM feature-coherence eval
  (`feature_dashboard_eval.py`), ~$5-7 of API spend. Copy `.env.example` to `.env`.

The full pipeline writes everything back into `results/benchmark_db.json` and the
per-experiment JSONs; the figures tier then reads them. All sweep scripts are
idempotent — a rerun only redoes configs missing from the database.

## Repository layout

```
config.py                 Sweep grids, training defaults, null-space gate
db.py                     Concurrency-safe JSON benchmark database
run_all.sh                Three-tier reproduction runner

models/                   ExpanderSAE + dense / clustered / pruned-retuned baselines
inference/                OMP, NIHT, CoSaMP + GPU-batched structured-OMP decoders
kernels/triton/           Structured-OMP decoding kernels (Appendix C)
experiments/              Activation extraction, training, inference, CE-loss,
                          scaling, per-figure diagnostics, and the Appendix C
                          GPU throughput benchmarks (structured / gOMP / Cholesky)
results/
  plot_all.py             benchmark_db.json + diagnostics JSON -> figures
  make_tables.py          benchmark_db.json -> LaTeX table fragments
  plot_feature_coherence.py
  *.json                  Shipped precomputed results (the figures tier reads these)
  raw/*_encoder_learning_curve.npy   Learning-curve trajectories for one figure
```

## Where each artifact comes from

**Figures** (`results/figures/`) are all built by `results/plot_all.py`, except the
feature-coherence histogram (`results/plot_feature_coherence.py`):

| Figure(s) | Data source |
|-----------|-------------|
| Storage–fidelity frontier (`ce_vs_k`, `relerr_vs_k*`) | `benchmark_db.json` |
| Support-structure controls, matched-params, pruned-retuned | `benchmark_db.json` |
| Novelty (`novelty_vs_k`, `decoder_cos`, `token_entropy`), geometry | `benchmark_db.json` |
| Firing-rate-matched novelty null | `novelty_null*.json` |
| Active-support collisions | `active_support_collisions.json` |
| Learning curves | `benchmark_db.json` + `raw/*_learning_curve.npy` |
| Synthetic OMP recovery / rel-err | `synthetic_db.json` |
| Feature-coherence histogram | `feature_dashboard_eval.json` |

**Auto-generated tables** — `results/make_tables.py` emits LaTeX fragments to
`results/tables/` (`main_results`, `pareto`, `practical`, `data_efficiency`,
`encoder_mask_ablation`), all from `benchmark_db.json`.

**Hand-transcribed tables** — a few paper tables are typeset by hand from the raw
per-experiment outputs rather than emitted by a generator. The repo ships those raw
outputs as the source of record so the numbers are auditable:

| Table | Transcribed from |
|-------|------------------|
| Cross-model / cross-layer / Qwen replication | `*_replication.json`, `cross_layer_replication.json` |
| Encoder-vs-iterative-OMP at scale | `*_omp_ce.json` |
| Appendix C throughput (encoder vs OMP variants; block-size *L* sweep) | `results/*.csv` written by the Appendix C throughput scripts — regenerated by the full tier, not committed |

Upstream producers (full tier): `training_sweep.py` and `ce_loss_sweep.py` populate
`benchmark_db.json`; `inference_sweep.py` adds the OMP/NIHT/CoSaMP rows;
`feature_analysis.py` adds novelty/entropy; `geometry_diagnostics.py` adds the
certificate ratios; `scaling_*.py` and `scaling_omp_ce.py` write the replication and
OMP-at-scale JSONs. The Appendix C throughput scripts
(`structured_omp_throughput.py`, `batched_omp_gpu_throughput.py`, `gomp_sweep.py`,
`multiblock_cholesky_sweep.py`, `multiblock_L_sweep.py`) write `results/*.csv` for
the hand-transcribed throughput tables; `profile_oneshot.py` is a per-op profiling
helper that feeds no table.

## Key hyperparameters

- Headline model: **Pythia-70M-deduped**, residual stream at layer 3 (`m=512`,
  `n=4096`). Scaling runs override these per model (Pythia-160M, Qwen2.5-3B,
  Llama-3.2-1B).
- Degrees `d ∈ {7, 30, 50, 100, 200}`; sparsity `k ∈ {16, 32, 64, 128}`; seeds 0/1/2.
- Optimiser: Adam, lr `3e-4 → 1e-5` cosine, batch 256, 5000 steps, grad-clip 1.0.
- Sparsity via TopK (no ℓ1 penalty). Dead-feature resampling every 1000 steps.
- Null-space gate: configurations with `n ≤ 2k` are excluded from the main frontier.

## License

MIT — see [LICENSE](LICENSE).
