# Beyond Redaction — reference implementation (Review 2 skeleton)

This is a working starter implementation of the pipeline described in
`Revised_Title_and_Objectives.md` and `Review2_Datasets_Model_Evaluation.md`.
It is organized so each file maps directly onto one objective, and each
demo block (`if __name__ == "__main__":`) runs standalone with no API
keys or model downloads required — that's the part that's genuinely
finished. The pieces that need your GLiNER / cloud LLM / TAB dataset
credentials are wired up as pluggable interfaces with clear `TODO`s.

## What's fully implemented (verified to run — see `python3 <file>.py`)

| File | Objective | Status |
|---|---|---|
| `src/mechanism_router.py` | 1 (routing) | **Working.** Rule-based structure classifier + severity model, reproduces the worked example from the proposal exactly. |
| `src/vault.py` | 2 (reversibility) | **Working.** Session-bound one-to-many surrogates, Fernet-encrypted store, covers both TOKENIZE and DP_NOISE branches. |
| `src/dp_noise.py` | 1 / 3 | **Working** (generalization fallback). Metric-DP-in-embedding-space class is written but needs an `embed_fn` (plug in `sentence-transformers`). |
| `src/detection.py` | 1 (detection) | **Working** (`RuleBasedDetector`, for dev/testing). `GLiNERDetector` and `LocalLLMDetector` are written but need `pip install gliner` and an LLM API key respectively. |
| `src/baselines.py` | 5 (comparison) | **Working.** Four baseline systems to run the pipeline against. |
| `src/evaluate.py` | 4 / 5 (metrics) | **Working** math (relative gain, extended metric, latency breakdown). `reidentification_success_rate` needs an attacker LLM wired in. |
| `src/utility_assessor.py` | 3 (utility gate) | **Structurally complete**, needs training data (a batch of (prompt, epsilon, observed quality) triples) before `fit()` is meaningful — use your Objective 4 evaluation runs to generate this. |

Roughly: routing + vault + evaluation math = the reversibility/protection
half of the pipeline, done. Wiring in GLiNER, a cloud LLM, and an
attacker LLM (a few hours of API glue code, not new research) is what
takes this from ~50% to fully running end-to-end — see the roadmap doc
for the exact remaining checklist.

## Install

```bash
pip install -r requirements.txt
# optional, for the real detector instead of RuleBasedDetector:
pip install gliner
```

## Datasets (see the planning doc for full detail)

- **ai4privacy/pii-masking-openpii-1m** (Hugging Face) — training/eval
  data for the detector and router.
  `pip install datasets && python -c "from datasets import load_dataset; load_dataset('ai4privacy/pii-masking-openpii-1m')"`
- **Text Anonymization Benchmark (TAB)** — ground truth for
  `mechanism_router.load_tab_direct_quasi_labels()` and
  `evaluate.router_vs_tab_agreement()`.
  Download: https://github.com/NorskRegnesentral/text-anonymization-benchmark
- **Enron Email Dataset** — realistic enterprise writing style for
  building synthetic task prompts. https://www.cs.cmu.edu/~enron/

## Quick smoke test

```bash
cd src
python3 mechanism_router.py   # prints the routing decision for 6 sample entities
python3 vault.py              # protects two entities, restores them from a fake response
python3 evaluate.py           # prints relative gain vs. the extended metric
```

## Wiring the full pipeline (not yet assembled — this is the other half of the 50%)

`pipeline.py` (not yet written) should: run `HybridDetector.detect()` →
for each entity, `MechanismRouter.route()` → for TOKENIZE entities,
`SessionVault.protect()`; for DP_NOISE entities, `dp_noise.sanitize_entity()`
then also log the pair in the vault → assemble the sanitized prompt →
`utility_assessor.query_with_gate()` → `SessionVault.restore_all()` on
the response.
