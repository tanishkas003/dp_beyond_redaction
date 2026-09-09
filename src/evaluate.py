"""
evaluate.py
===========
Metrics and comparison harness for Objectives 4 and 5.

Implements:
  - router_vs_tab_agreement(): validates the structure-based
    TOKENIZE-vs-DP_NOISE split against TAB's human-annotated
    direct/quasi-identifier labels -- the empirical grounding for
    Objective 1's routing rule (not just an engineering choice).
  - leakage_prevention_rate(): standard detection-based metric, kept
    for comparability with prior work, but NOT treated as the
    headline privacy number (see re-identification below).
  - reidentification_success_rate(): the real headline metric per
    Objective 4, following RAT-Bench's finding that detection rate
    overstates protection. Runs a pluggable LLM "attacker".
  - reversibility_weighted_relative_gain(): extends the established
    Relative Gain metric (Mattern et al. 2022: RG = Up/Uo - Pp/Po,
    used in POLAR-Bench and others) with a third term for exact-value
    recovery accuracy -- the piece none of the reviewed DP-only
    papers need, because they never attempt exact recovery. This is
    the project's proposed composite metric; report both RG and this
    extension so results are comparable to prior work either way.
  - latency_breakdown(): per-stage timing, per the LLM Gatekeeper
    paper's finding that end-to-end latency alone hides where the
    real adoption blocker is.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Objective 1: router validated against TAB
# ---------------------------------------------------------------------------

def router_vs_tab_agreement(router, tab_records: list[dict]) -> dict:
    """tab_records: output of mechanism_router.load_tab_direct_quasi_labels().
    Compares the router's structure-based mechanism choice against TAB's
    human DIRECT/QUASI labels. Report this as evidence the routing rule
    isn't arbitrary -- it's cross-validated against expert human
    anonymization judgment, which none of the reviewed prior systems do."""
    try:
        from .mechanism_router import is_structured_identifier
    except ImportError:  # pragma: no cover - standalone invocation from src/
        from mechanism_router import is_structured_identifier

    correct = 0
    for rec in tab_records:
        predicted_direct = is_structured_identifier(rec["entity_text"], rec["entity_type"])
        actual_direct = rec["tab_label"] == "DIRECT"
        correct += int(predicted_direct == actual_direct)
    n = max(len(tab_records), 1)
    return {"n": n, "agreement_rate": correct / n}


# ---------------------------------------------------------------------------
# Objective 4: leakage prevention (kept for comparability) + re-identification
# ---------------------------------------------------------------------------

def leakage_prevention_rate(ground_truth_spans: list[str], sanitized_text: str) -> float:
    """Fraction of ground-truth sensitive spans that do NOT appear
    verbatim in the sanitized output. Standard metric -- report it,
    but see reidentification_success_rate() for why this alone
    overstates real protection (RAT-Bench, 2025/2026)."""
    if not ground_truth_spans:
        return 1.0
    prevented = sum(1 for span in ground_truth_spans if span not in sanitized_text)
    return prevented / len(ground_truth_spans)


def reversibility_accuracy(expected_values: Sequence[str], restored_text: str) -> float:
    """Fraction of values that survive a protect -> LLM -> restore round trip.

    Pass only values that were routed through a reversible mechanism.  A value
    counts as restored when it occurs verbatim in the final restored response;
    this correctly records a miss when an LLM paraphrases or drops a token.
    """
    if not expected_values:
        return 1.0
    return sum(value in restored_text for value in expected_values) / len(expected_values)


def utility_retention(original_utility: float, protected_utility: float) -> float:
    """Utility kept relative to the unprotected reference (1.0 is unchanged)."""
    if original_utility <= 0:
        return 1.0 if protected_utility <= 0 else 0.0
    return max(0.0, protected_utility / original_utility)


AttackerFn = "Callable[[str, str], str]"  # (sanitized_text, question) -> guessed answer


def reidentification_success_rate(
    attacker_fn, sanitized_texts: list[str], true_identities: list[str],
    question: str = "Who is this person, or what specific entity does this describe?",
) -> float:
    """Runs an LLM-based attacker against each sanitized text and checks
    whether its guess matches the true (masked) identity/value -- the
    real privacy metric per Objective 4, following RAT-Bench's
    methodology. `attacker_fn` should wrap a call to an LLM with a
    prompt asking it to guess the redacted/tokenized identity from
    context; keep it simple (exact/fuzzy string match against the
    true value) for a first working version."""
    assert len(sanitized_texts) == len(true_identities)
    hits = 0
    for text, truth in zip(sanitized_texts, true_identities):
        guess = attacker_fn(text, question)
        if truth.strip().lower() in guess.strip().lower():
            hits += 1
    return hits / max(len(sanitized_texts), 1)


def quasi_identifier_combination_risk(
    attacker_fn, sanitized_texts: list[str], true_identities: list[str],
    retained_context_fields: list[list[str]],
) -> float:
    """Scoped test for SurrogateShield's "implicit PII" open problem:
    does the COMBINATION of retained, individually-harmless quasi-
    identifiers (job title + city + project name, etc.) let an
    attacker re-identify someone even when no single span is a direct
    identifier? Not a general solution -- a bounded empirical probe,
    as scoped in the project write-up."""
    return reidentification_success_rate(attacker_fn, sanitized_texts, true_identities)


# ---------------------------------------------------------------------------
# Composite privacy-utility metric
# ---------------------------------------------------------------------------

def relative_gain(u_private: float, u_original: float, p_private: float, p_original: float) -> float:
    """Mattern et al. (2022) Relative Gain: RG = Up/Uo - Pp/Po.
    Positive => privacy gained outweighs utility lost. Reported
    alongside the extended metric below for comparability with prior
    work (used in POLAR-Bench and the word-level DP literature)."""
    return (u_private / max(u_original, 1e-9)) - (p_private / max(p_original, 1e-9))


def reversibility_weighted_relative_gain(
    u_private: float, u_original: float, p_private: float, p_original: float,
    reversibility_accuracy: float, weight: float = 0.5,
) -> float:
    """Extension proposed for this project: adds a term for exact-value
    recovery accuracy on tokenized spans (fraction of TOKENIZE-routed
    entities correctly restored in the final response). None of the
    DP-only baselines can score above 0 on this term, since they never
    attempt exact recovery -- this is the metric that should make the
    hybrid router's advantage visible in a single number, alongside
    the standard RG for comparability."""
    base = relative_gain(u_private, u_original, p_private, p_original)
    return base + weight * reversibility_accuracy


# ---------------------------------------------------------------------------
# Objective 5: per-stage latency
# ---------------------------------------------------------------------------

@dataclass
class LatencyBreakdown:
    stages: dict = field(default_factory=dict)

    REQUIRED_STAGES = ("detection", "routing", "protection", "vault_operations", "llm_generation", "restoration")

    @contextmanager
    def measure(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[stage_name] = self.stages.get(stage_name, 0.0) + (time.perf_counter() - start)

    def total(self) -> float:
        return sum(self.stages.values())

    def report(self) -> dict:
        total = self.total() or 1e-9
        return {name: {"seconds": t, "pct_of_total": t / total} for name, t in self.stages.items()}

    def complete_report(self) -> dict:
        """Report all standard stages, using zero for stages not instrumented."""
        total = self.total() or 1e-9
        names = dict.fromkeys((*self.REQUIRED_STAGES, *self.stages))
        return {name: {"seconds": self.stages.get(name, 0.0),
                       "pct_of_total": self.stages.get(name, 0.0) / total}
                for name in names}


# Explicit aliases make notebook/report code read naturally.
calculate_leakage_prevention_rate = leakage_prevention_rate
calculate_reidentification_success_rate = reidentification_success_rate
calculate_reversibility_accuracy = reversibility_accuracy
calculate_utility_retention = utility_retention


if __name__ == "__main__":
    print(relative_gain(u_private=0.8, u_original=1.0, p_private=0.1, p_original=1.0))
    print(reversibility_weighted_relative_gain(
        u_private=0.8, u_original=1.0, p_private=0.1, p_original=1.0,
        reversibility_accuracy=0.97,
    ))
