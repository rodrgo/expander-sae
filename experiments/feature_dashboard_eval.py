"""Blinded LLM evaluation of feature dashboards.

Tests whether Expander-SAE features are as interpretable as Dense-SAE
features. Samples 25 features per architecture (Expander-SAE $d{=}7$,
Expander-SAE $d{=}200$, Dense-SAE) at the headline configuration
(Pythia-70M layer 3, $m{=}512$, $n{=}4096$, $k{=}64$, seed 0),
stratified to match firing-rate quartiles. For each feature renders a
top-15-activations dashboard with a globally-shuffled anonymous ID
(`F01`..`F75`) so the LLM judge cannot infer the architecture.
Two judges (Claude Sonnet 4.5 + GPT-4o) score each dashboard at
temperature 0, three calls each. Outputs aggregate score
distributions and inter-rater agreement.

Run: ``venv/bin/python experiments/feature_dashboard_eval.py``
Cost: ~$5-7 in API spend total.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_entry, load_db
from models import build


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TOKENS_ACTS = ROOT / "data" / "tokens_and_acts.npz"
RESULTS_JSON = ROOT / "results" / "feature_dashboard_eval.json"
CACHE_JSON = ROOT / "results" / "feature_dashboard_eval_cache.json"
FIG_PATH = ROOT / "results" / "figures" / "feature_coherence_histogram.pdf"

ARCHS = [
    # (label, db_arch, d)
    ("Expander d=7",   "expander_tied",  7),
    ("Expander d=200", "expander_tied",  200),
    ("Dense-SAE",      "dense_warmtied", 512),
]
M, N, K, SEED = 512, 4096, 64, 0

N_FEATURES_PER_ARCH = 25
N_QUARTILES = 4
TOP_K_EXAMPLES = 15
CONTEXT_TOKENS = 16

CALLS_PER_FEATURE_PER_JUDGE = 3
JUDGE_MODELS = {
    "claude": "claude-sonnet-4-5",
    "gpt4o":  "gpt-4o-2024-08-06",
}

PROMPT_TEMPLATE = """\
You will see a "feature dashboard" from a sparse autoencoder probe of a \
language model. The dashboard shows the 15 input contexts where this \
feature activated most strongly, with the target token highlighted as \
**TOKEN**.

Your job is to rate how *coherent* the activation pattern is.

Rubric:
  - Briefly describe what tokens or contexts cause this feature to fire.
  - Rate the coherence of the activation pattern on a 1-5 scale:
      5 = there is a clear, unifying theme across the activations
      4 = there is a theme but with some noise
      3 = there is a partial theme on a subset of activations
      2 = activations are mostly unrelated, with hints of theme
      1 = activations seem unrelated
  - Guess the concept this feature represents in 1-3 words, or write \
"no clear concept" if the firing pattern doesn't suggest one.

Respond as JSON with fields: description (string), coherence (int 1-5), \
concept (string).

Dashboard:

{dashboard_markdown}
"""


# ----------------------------------------------------------------------
# Env / API key loading
# ----------------------------------------------------------------------

def _load_dotenv():
    """Read .env in repo root if present and set env vars."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ----------------------------------------------------------------------
# Stage 1: encode activations through SAEs and pick features
# ----------------------------------------------------------------------

def _encode_real(model, acts_np: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Return real-valued (T, n) feature activations."""
    model = model.eval()
    T = len(acts_np)
    out = np.zeros((T, model.n), dtype=np.float32)
    tensor = torch.from_numpy(acts_np).float()
    with torch.no_grad():
        for i in range(0, T, batch):
            chunk = tensor[i:i + batch]
            _, h = model(chunk)
            out[i:i + batch] = h.numpy().astype(np.float32)
    return out


def _stratified_sample(firing_rate: np.ndarray, n_per_quartile: list[int],
                       rng: np.random.Generator) -> list[int]:
    """Quartile-stratified sample of alive features.

    n_per_quartile is a length-4 list with the counts per quartile.
    Returns a list of feature indices (sum(n_per_quartile) entries).
    """
    alive = np.flatnonzero(firing_rate > 0)
    if len(alive) < sum(n_per_quartile):
        raise RuntimeError(f"Not enough alive features: "
                           f"{len(alive)} alive, need {sum(n_per_quartile)}")
    rates = firing_rate[alive]
    edges = np.quantile(rates, [0.0, 0.25, 0.5, 0.75, 1.0])
    picked = []
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        if q == 3:
            in_q = alive[(rates >= lo) & (rates <= hi)]
        else:
            in_q = alive[(rates >= lo) & (rates < hi)]
        if len(in_q) < n_per_quartile[q]:
            # Fall back to all features in this quartile
            picked.extend(in_q.tolist())
        else:
            picked.extend(rng.choice(in_q, size=n_per_quartile[q],
                                     replace=False).tolist())
    return picked


def _build_quartile_targets(total: int) -> list[int]:
    """Distribute `total` picks across 4 quartiles, biasing the
    overflow into the higher quartiles (more features there)."""
    base = total // 4
    rem = total - base * 4
    targets = [base, base, base, base]
    for i in range(rem):
        targets[3 - i] += 1  # spillover from top quartile down
    return targets


# ----------------------------------------------------------------------
# Stage 2: top-K activating-example extraction & dashboard rendering
# ----------------------------------------------------------------------

def _extract_top_k_examples(
    feature_acts_flat: np.ndarray,   # (T,) for a single feature, real-valued
    tokens: np.ndarray,              # (n_seqs, seq_len)
    k: int = TOP_K_EXAMPLES,
    ctx: int = CONTEXT_TOKENS,
) -> list[dict]:
    """Top-k activations for a feature; return list of context dicts."""
    n_seqs, seq_len = tokens.shape
    # Take only abs-value top-k positions; we want strong activations
    # (positive or negative).
    top_idx = np.argsort(-np.abs(feature_acts_flat))[:k]
    out: list[dict] = []
    for flat_idx in top_idx:
        seq_idx = int(flat_idx // seq_len)
        pos = int(flat_idx % seq_len)
        lo = max(0, pos - ctx)
        hi = min(seq_len, pos + ctx + 1)
        out.append({
            "seq_idx":     seq_idx,
            "pos":         pos,
            "context_left":  tokens[seq_idx, lo:pos].tolist(),
            "target_token":  int(tokens[seq_idx, pos]),
            "context_right": tokens[seq_idx, pos + 1:hi].tolist(),
            "activation":  float(feature_acts_flat[flat_idx]),
        })
    return out


def _render_dashboard(feature_id: str, examples: list[dict], tokenizer
                     ) -> str:
    """Render the markdown dashboard. Strips control chars from decoded
    text to avoid JSON-confusion in LLM output."""
    def _decode(ids: list[int]) -> str:
        s = tokenizer.decode(ids, skip_special_tokens=True)
        # Strip newlines / weird whitespace to keep dashboards on one line
        # per example.
        s = re.sub(r"\s+", " ", s).strip()
        return s

    lines = [
        f"# Feature {feature_id}",
        "",
        "Top 15 activations (descending by activation magnitude):",
        "",
    ]
    for i, ex in enumerate(examples, 1):
        left = _decode(ex["context_left"]) or "(start of sequence)"
        target = _decode([ex["target_token"]])
        right = _decode(ex["context_right"]) or "(end of sequence)"
        act = ex["activation"]
        lines.append(
            f"  {i:2d}. [act = {act:+.2f}] ...{left}... **{target}** "
            f"...{right}..."
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Stage 3: LLM judges with disk caching
# ----------------------------------------------------------------------

class _LLMCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def key(self, feature_id: str, judge: str, call_idx: int) -> str:
        return f"{feature_id}|{judge}|{call_idx}"

    def get(self, feature_id: str, judge: str, call_idx: int) -> Optional[dict]:
        return self.data.get(self.key(feature_id, judge, call_idx))

    def set(self, feature_id: str, judge: str, call_idx: int, value: dict):
        self.data[self.key(feature_id, judge, call_idx)] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def _parse_response(text: str) -> dict:
    """Pull a JSON dict with {description, coherence, concept} out of an
    LLM response. Tolerates leading/trailing prose and code fences."""
    # First try direct JSON parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "coherence" in obj:
            return _normalise_dict(obj)
    except json.JSONDecodeError:
        pass
    # Strip code fences.
    m = re.search(r"\{[^{}]*\"coherence\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return _normalise_dict(obj)
        except json.JSONDecodeError:
            pass
    # Last resort: regex pull out the integer.
    m = re.search(r"coherence[\"\s:]+([1-5])", text)
    if m:
        return {"description": text[:200], "coherence": int(m.group(1)),
                "concept": "PARSE_FAIL", "raw": text}
    return {"description": text[:200], "coherence": None,
            "concept": "PARSE_FAIL", "raw": text}


def _normalise_dict(obj: dict) -> dict:
    return {
        "description": str(obj.get("description", "")),
        "coherence":   int(obj["coherence"]),
        "concept":     str(obj.get("concept", "")).strip().lower(),
    }


def _call_anthropic(prompt: str, model: str, seed: int) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text if resp.content else ""
    return _parse_response(text)


def _call_openai(prompt: str, model: str, seed: int) -> dict:
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=400,
        temperature=0.0,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    return _parse_response(text)


# ----------------------------------------------------------------------
# Stage 4: main pipeline
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="Build dashboards but skip LLM calls (debug mode).")
    ap.add_argument("--n-features", type=int, default=N_FEATURES_PER_ARCH)
    args = ap.parse_args()

    _load_dotenv()
    if not args.no_llm:
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY not set")
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    print("loading tokenizer + tokens + activations...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    payload = np.load(TOKENS_ACTS)
    tokens, acts = payload["tokens"], payload["activations"]
    n_seqs, seq_len = tokens.shape
    print(f"  tokens: {tokens.shape}, activations: {acts.shape}")
    assert acts.shape[0] == n_seqs * seq_len, \
        f"flat-activation count ({acts.shape[0]}) != n_seqs*seq_len " \
        f"({n_seqs*seq_len})"

    db = load_db()

    # --- Per-arch: load model, encode, sample features, render dashboards
    quartile_targets = _build_quartile_targets(args.n_features)
    print(f"per-arch quartile targets: {quartile_targets}")
    rng = np.random.default_rng(42)

    all_features: list[dict] = []
    for label, arch, d in ARCHS:
        entry = get_entry(db, arch, M, N, d, K, SEED, "encoder")
        if entry is None:
            raise RuntimeError(f"No DB entry for {arch} d={d}")
        ckpt = entry["model_path"]
        print(f"\n[{label}] loading {ckpt}")
        model = build(arch, m=M, n=N, d=d, k=K, seed=SEED)
        model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True))
        model = model.eval()
        print(f"[{label}] encoding 128k tokens through SAE...")
        h = _encode_real(model, acts)               # (T, n)
        firing_rate = (h != 0).mean(axis=0)
        n_alive = int((firing_rate > 0).sum())
        print(f"[{label}] alive={n_alive}/{model.n}, "
              f"firing-rate stats: mean={firing_rate.mean():.4f}, "
              f"median={np.median(firing_rate):.4f}, "
              f"max={firing_rate.max():.4f}")
        picks = _stratified_sample(firing_rate, quartile_targets, rng)
        print(f"[{label}] picked {len(picks)} features (target "
              f"{sum(quartile_targets)})")
        for j in picks:
            examples = _extract_top_k_examples(h[:, j], tokens)
            all_features.append({
                "arch_label":   label,
                "arch":         arch,
                "d":            d,
                "feature_idx":  int(j),
                "firing_rate":  float(firing_rate[j]),
                "examples":     examples,
            })
        del model, h
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nbuilt {len(all_features)} feature records")

    # --- Globally shuffle feature IDs so nothing in the dashboards
    # reveals the architecture order.
    perm = rng.permutation(len(all_features))
    for new_id, idx in enumerate(perm, 1):
        all_features[idx]["feature_id"] = f"F{new_id:02d}"

    # --- Render dashboards
    for rec in all_features:
        rec["dashboard"] = _render_dashboard(
            rec["feature_id"], rec["examples"], tok)

    # --- Sanity-grep: no architecture/d should appear in any dashboard
    for rec in all_features:
        leak = re.search(r"d=\d|expander|dense|firing|seed\s*\d",
                         rec["dashboard"], re.IGNORECASE)
        if leak is not None:
            print(f"WARNING: possible leak in {rec['feature_id']}: "
                  f"{leak.group(0)}")

    # --- Print 2-3 sample dashboards for human eyeball
    print("\n=== Sample dashboards (eyeball check before LLM calls) ===")
    for rec in all_features[:3]:
        print(rec["dashboard"][:600] + "\n...")

    if args.no_llm:
        # Still write the records so we can inspect.
        out = {
            "config": {"archs": [a[0] for a in ARCHS],
                       "n_per_arch": args.n_features},
            "features": [{k: v for k, v in r.items() if k != "examples"}
                         for r in all_features],
        }
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_JSON.write_text(json.dumps(out, indent=2))
        print(f"\n[--no-llm] wrote {RESULTS_JSON} with dashboards only")
        return

    # --- Stage 3: LLM judges
    cache = _LLMCache(CACHE_JSON)
    judge_callers = {
        "claude": _call_anthropic,
        "gpt4o":  _call_openai,
    }

    n_calls_total = (len(all_features) * len(JUDGE_MODELS)
                     * CALLS_PER_FEATURE_PER_JUDGE)
    n_done = 0
    t0 = time.time()
    parse_fails = 0
    for rec in all_features:
        prompt = PROMPT_TEMPLATE.format(dashboard_markdown=rec["dashboard"])
        rec["judges"] = {}
        for judge, model_id in JUDGE_MODELS.items():
            calls = []
            for c in range(CALLS_PER_FEATURE_PER_JUDGE):
                cached = cache.get(rec["feature_id"], judge, c)
                if cached is not None:
                    calls.append(cached)
                    n_done += 1
                    continue
                try:
                    resp = judge_callers[judge](prompt, model_id, seed=c + 1)
                except Exception as exc:
                    print(f"  [{rec['feature_id']}] {judge} call {c} "
                          f"FAILED: {exc}")
                    resp = {"description": str(exc), "coherence": None,
                            "concept": "API_FAIL"}
                if resp.get("concept") == "PARSE_FAIL":
                    parse_fails += 1
                cache.set(rec["feature_id"], judge, c, resp)
                calls.append(resp)
                n_done += 1
                if n_done % 10 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 0.1)
                    eta = (n_calls_total - n_done) / max(rate, 0.1)
                    print(f"  {n_done}/{n_calls_total} calls "
                          f"({rate:.1f}/s, eta {eta:.0f}s)")
            rec["judges"][judge] = calls

    print(f"\nparse failures: {parse_fails} / {n_calls_total}")

    # --- Stage 5: aggregate
    arch_scores = {label: [] for label, *_ in ARCHS}
    judge_means = {judge: {} for judge in JUDGE_MODELS}
    for rec in all_features:
        per_judge = []
        for judge in JUDGE_MODELS:
            calls = rec["judges"][judge]
            scores = [c["coherence"] for c in calls
                      if c.get("coherence") is not None]
            mean = float(np.mean(scores)) if scores else float("nan")
            per_judge.append(mean)
            judge_means[judge][rec["feature_id"]] = mean
        rec["coherence_judge_mean"] = per_judge
        rec["coherence_overall_mean"] = float(np.mean(per_judge))
        arch_scores[rec["arch_label"]].append(rec["coherence_overall_mean"])

    summary = {}
    for label, scores in arch_scores.items():
        scores_arr = np.array(scores, dtype=float)
        summary[label] = {
            "n":       int(len(scores_arr)),
            "mean":    float(scores_arr.mean()),
            "sem":     float(scores_arr.std(ddof=1) / np.sqrt(len(scores_arr))),
            "median":  float(np.median(scores_arr)),
            "histogram_1to5": [
                int(((scores_arr >= b - 0.5) & (scores_arr < b + 0.5)).sum())
                for b in (1, 2, 3, 4, 5)
            ],
        }
        # Bootstrap 95% CI
        boots = [scores_arr[rng.integers(0, len(scores_arr), len(scores_arr))].mean()
                 for _ in range(2000)]
        summary[label]["ci95"] = [float(np.quantile(boots, 0.025)),
                                  float(np.quantile(boots, 0.975))]

    # Inter-judge Spearman.
    from scipy.stats import spearmanr
    fids = [r["feature_id"] for r in all_features]
    claude_arr = np.array([judge_means["claude"][f] for f in fids])
    gpt4o_arr = np.array([judge_means["gpt4o"][f] for f in fids])
    rho, p = spearmanr(claude_arr, gpt4o_arr,
                       nan_policy="omit")
    inter_rater = {"spearman_rho": float(rho), "p_value": float(p)}

    # Concept-coverage.
    for rec in all_features:
        labels = []
        for judge in JUDGE_MODELS:
            for c in rec["judges"][judge]:
                concept = (c.get("concept") or "").lower()
                if concept and concept not in (
                        "no clear concept", "parse_fail", "api_fail"):
                    labels.append(concept)
        rec["concept_labels"] = labels
        rec["has_concept"] = len(labels) > 0
    for label, scores in arch_scores.items():
        recs_for_arch = [r for r in all_features if r["arch_label"] == label]
        summary[label]["concept_frac"] = float(
            np.mean([r["has_concept"] for r in recs_for_arch]))

    print("\n=== Summary ===")
    for label, s in summary.items():
        print(f"  {label:20}  mean={s['mean']:.2f}  sem={s['sem']:.2f}  "
              f"ci95=[{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]  "
              f"concept_frac={s['concept_frac']:.2f}  "
              f"hist={s['histogram_1to5']}")
    print(f"  inter-judge Spearman rho = {rho:.3f} (p = {p:.3g})")

    # --- Save
    out = {
        "config": {
            "archs": [a[0] for a in ARCHS],
            "n_per_arch": args.n_features,
            "calls_per_judge": CALLS_PER_FEATURE_PER_JUDGE,
            "judges": JUDGE_MODELS,
        },
        "summary": summary,
        "inter_rater": inter_rater,
        "features": all_features,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Drop the dashboards from the saved JSON (they're large and
    # reproducible from feature_idx + tokens). Keep examples for
    # later inspection.
    for rec in out["features"]:
        rec.pop("dashboard", None)
    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
