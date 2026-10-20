# DP 2.0 Implementation Report

Implemented: risk engine, four protection paths, HMAC/AES-GCM vault,
privacy-budget accounting, pre-LLM leakage checks, cloud/local routing,
response filtering, utility gating, adversarial interfaces, local dataset
adapters, metrics, baselines, and integrated pipeline controls.

The pipeline order is enforced as:

`detect -> assess risk -> apply critical policy -> protect -> account epsilon -> check leakage -> utility gate -> route provider -> filter response -> restore`

The four baselines are `no_privacy`, `static_masking`,
`tokenize_everything`, and `dp_everything`. The proposed system is the
`ProtectionPipeline`, not an additional baseline.

Verified offline with `57 passed` tests. Not executed: real cloud/local model
calls, large TAB/AI4Privacy/Enron runs, and a formal deployment-specific
end-to-end DP proof. Leakage and response detectors are transparent
deterministic baselines, not comprehensive inference detectors. Provider
credentials and dataset paths are intentionally deployment configuration, not
source-controlled values.
