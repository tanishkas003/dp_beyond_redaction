"""
mechanism_router.py
====================
The Reversibility-Aware Mechanism Router -- the core unique artifact of
this project.

For every entity detected in a prompt, this module answers two questions
in sequence:

  1. RETAIN vs PROTECT   -- does the LLM actually need this value to
                             answer the task well? (utility relevance)
  2. TOKENIZE vs DP_NOISE -- for anything flagged PROTECT, does the
                             *employee's own downstream use* require the
                             exact original value back (tokenize), or is
                             approximate semantic context enough (DP-noise)?

Question 2 is answered by structure, not by a severity score: discrete /
structured identifiers (SSNs, employee IDs, invoice numbers, project
codenames) have no meaningful "nearby" value for a DP mechanism to
perturb toward, and enterprise workflows frequently need the literal
original value echoed back correctly (a report referencing invoice
#INV-2026-00452 is *wrong*, not just imprecise, if the number is
approximated). Free-text quasi-identifiers (job titles, cities, rough
dates) do have meaningful semantic neighbors and rarely need to come
back byte-for-byte correct, so DP/abstraction is both feasible and
sufficient there.

Severity only tunes strength *within* whichever branch structure
already selected (epsilon for DP-noise, vault access-strictness for
tokenized values) -- see dp_noise.py and vault.py.

Ground truth for training / validating the TOKENIZE-vs-DP_NOISE split
comes from the Text Anonymization Benchmark (TAB, Pilan et al. 2022),
which is the only public corpus with human-expert direct-identifier vs
quasi-identifier labels. See train_router_on_tab() below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

EXACT_RECOVERY_KEYWORDS = {
    "exact",
    "original",
    "verbatim",
    "byte-for-byte",
    "unchanged",
    "precise",
    "specific",
    "correct identifier",
    "exact value",
}

APPROXIMATE_CONTEXT_KEYWORDS = {
    "summarize",
    "summary",
    "overview",
    "general",
    "analyze",
    "analysis",
    "describe",
    "insight",
    "trend",
    "aggregate",
}
# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------

class UtilityDecision(str, Enum):
    RETAIN = "RETAIN"
    PROTECT = "PROTECT"


class Mechanism(str, Enum):
    NONE = "NONE"          # retained, no protection applied
    TOKENIZE = "TOKENIZE"  # exact, reversible substitution
    DP_NOISE = "DP_NOISE"  # approximate, calibrated substitution


@dataclass
class RoutingDecision:
    entity_text: str
    entity_type: str
    utility: UtilityDecision
    mechanism: Mechanism
    severity: float          # 0-1, tunes strength within the chosen mechanism
    reason: str               # short justification, useful for the report/demo


# ---------------------------------------------------------------------------
# Step 2 (structure): regex-based structured-identifier detector
# ---------------------------------------------------------------------------
# These patterns decide "does this span look like a discrete, structured
# identifier with no natural semantic neighbors?" Anything that matches is
# routed to TOKENIZE regardless of its NER entity type. Everything else that
# still needs protecting is free text -> DP_NOISE.

_STRUCTURED_PATTERNS: dict[str, re.Pattern] = {
    "SSN":          re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD":  re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "EMAIL":        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE":        re.compile(r"\b\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    "EMPLOYEE_ID":  re.compile(r"\b[A-Z]{2,5}-?\d{4,8}\b"),
    "PROJECT_CODE": re.compile(r"\bProject\s+[A-Z][a-z]+\b"),
    "INVOICE_NO":   re.compile(r"\b(?:INV|PO|SO)-?\d{4,10}\b", re.IGNORECASE),
    "API_KEY":      re.compile(r"\b(?:sk|pk|key)-[A-Za-z0-9]{16,}\b"),
    "IP_ADDRESS":   re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# NER types that are almost always free-text quasi-identifiers rather than
# discrete identifiers, used as a fallback when no regex pattern matches.
_QUASI_NER_TYPES = {"JOB_TITLE", "LOCATION", "GPE", "DATE", "ORG", "NORP"}


def is_structured_identifier(entity_text: str, entity_type: str) -> bool:
    """Structure check used to pick TOKENIZE vs DP_NOISE. Regex first
    (highest precision for genuinely discrete identifiers), NER type as
    a fallback signal only when no pattern matches."""
    for _, pattern in _STRUCTURED_PATTERNS.items():
        if pattern.search(entity_text):
            return True
    if entity_type in {"PERSON", "ID", "CODE"}:
        # names and internal codes are treated as structured by default:
        # an employee's own name usually needs to come back exactly right
        return True
    return False


# ---------------------------------------------------------------------------
# Step 1: RETAIN vs PROTECT (utility-relevance judgment)
# ---------------------------------------------------------------------------
# Pluggable: default is a conservative heuristic (protect everything the
# detector flagged); swap in an LLM-based judge for the full pipeline by
# passing a callable of the same signature.

RetainJudge = Callable[[str, str, str], UtilityDecision]

ExactRecoveryJudge = Callable[[str, str, str], bool]


def default_retain_judge(entity_text: str, entity_type: str, context: str) -> UtilityDecision:
    """Conservative default: anything the detector flagged gets PROTECT
    unless it's a generic/non-identifying type. Replace with
    llm_retain_judge (below) for the full decision-aware pipeline
    described in Objective 1."""
    _NON_SENSITIVE_TYPES = {"TASK_VERB", "GENERIC_NOUN"}
    if entity_type in _NON_SENSITIVE_TYPES:
        return UtilityDecision.RETAIN
    return UtilityDecision.PROTECT


def default_exact_recovery_judge(
    entity_text: str,
    entity_type: str,
    context: str
) -> bool:
    """
    Decide whether the original entity value must be recoverable exactly
    after downstream LLM processing.

    The decision considers both:
    1. Explicit downstream requirements in the context.
    2. The inherent structure of the entity.
    """

    entity_type = entity_type.upper()
    context_lower = context.lower()

    # --------------------------------------------------
    # 1. Explicit downstream request for exact recovery
    # --------------------------------------------------

    if any(
        keyword in context_lower
        for keyword in EXACT_RECOVERY_KEYWORDS
    ):
        return True

    # --------------------------------------------------
    # 2. Explicit downstream use where approximation is sufficient
    # --------------------------------------------------

    if any(
        keyword in context_lower
        for keyword in APPROXIMATE_CONTEXT_KEYWORDS
    ):
        return False

    # --------------------------------------------------
    # 3. Structured identifiers normally need exact recovery
    # --------------------------------------------------

    if is_structured_identifier(entity_text, entity_type):
        return True

    # --------------------------------------------------
    # 4. Default: approximate representation is sufficient
    # --------------------------------------------------

    return False


def make_llm_retain_judge(llm_call: Callable[[str], str]) -> RetainJudge:
    """Factory for an LLM-backed RETAIN/PROTECT judge. `llm_call` should
    be a function that takes a prompt string and returns the model's
    raw text response (wire this to your provider of choice -- see
    README for the Qwen2.5-7B / Ollama setup used in this project)."""

    PROMPT_TEMPLATE = (
        "You are deciding whether a detail in an employee's prompt to an "
        "AI assistant is needed for the assistant to answer well.\n"
        "Detail: \"{entity}\" (type: {etype})\n"
        "Surrounding text: \"{context}\"\n"
        "Answer with exactly one word: RETAIN if removing or masking this "
        "detail would make the assistant's answer noticeably worse, or "
        "PROTECT if the task can be answered just as well without the "
        "exact detail."
    )

    def judge(entity_text: str, entity_type: str, context: str) -> UtilityDecision:
        prompt = PROMPT_TEMPLATE.format(entity=entity_text, etype=entity_type, context=context)
        raw = llm_call(prompt).strip().upper()
        return UtilityDecision.RETAIN if "RETAIN" in raw else UtilityDecision.PROTECT

    return judge


# ---------------------------------------------------------------------------
# Trainable severity model (validated against TAB direct/quasi labels)
# ---------------------------------------------------------------------------

class TrainableSeverityModel:
    """Small sklearn classifier that scores severity (0-1) within the
    branch structure already selected. Trained on simple, cheap
    features -- no GPU, no fine-tuning. Optional: if untrained, falls
    back to a fixed default severity per entity type."""

    _DEFAULT_SEVERITY = {
        "SSN": 0.95, "CREDIT_CARD": 0.95, "API_KEY": 0.9, "EMAIL": 0.6,
        "PHONE": 0.6, "EMPLOYEE_ID": 0.7, "PROJECT_CODE": 0.5,
        "INVOICE_NO": 0.4, "PERSON": 0.7, "JOB_TITLE": 0.3,
        "LOCATION": 0.3, "DATE": 0.2, "ORG": 0.3,
    }

    def __init__(self):
        self.model = None  # sklearn estimator, set by fit()

    def fit(self, features: np.ndarray, severity_labels: np.ndarray) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(max_depth=4)
        self.model.fit(features, severity_labels)

    def score(self, entity_text: str, entity_type: str, features: Optional[np.ndarray] = None) -> float:
        if self.model is not None and features is not None:
            return float(np.clip(self.model.predict(features.reshape(1, -1))[0], 0.0, 1.0))
        return self._DEFAULT_SEVERITY.get(entity_type, 0.5)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class MechanismRouter:
    def __init__(
        self,
        retain_judge: RetainJudge = default_retain_judge,
        exact_recovery_judge: ExactRecoveryJudge = default_exact_recovery_judge,
        severity_model: Optional[TrainableSeverityModel] = None,
    ):
        self.retain_judge = retain_judge
        self.exact_recovery_judge = exact_recovery_judge
        self.severity_model = severity_model or TrainableSeverityModel()

    def route(self, entity_text: str, entity_type: str, context: str = "") -> RoutingDecision:
        utility = self.retain_judge(entity_text, entity_type, context)

        if utility is UtilityDecision.RETAIN:
            return RoutingDecision(
                entity_text=entity_text, entity_type=entity_type,
                utility=utility, mechanism=Mechanism.NONE, severity=0.0,
                reason="Task-relevant; retained in context.",
            )

        requires_exact_recovery = self.exact_recovery_judge(
            entity_text,
            entity_type,
            context
        )

        mechanism = (
            Mechanism.TOKENIZE
            if requires_exact_recovery
            else Mechanism.DP_NOISE
        )

        severity = self.severity_model.score(entity_text, entity_type)

        reason = (
            "Exact recovery required for downstream use; "
            "routed to reversible tokenization."
            if requires_exact_recovery
            else
            "Exact recovery is not required; approximate semantic context "
            "is sufficient, so DP-noise is used."
        )

        return RoutingDecision(
            entity_text=entity_text, entity_type=entity_type,
            utility=utility, mechanism=mechanism, severity=severity, reason=reason,
        )


# ---------------------------------------------------------------------------
# TAB-based training / validation helper
# ---------------------------------------------------------------------------

def load_tab_direct_quasi_labels(tab_json_path: str) -> list[dict]:
    """Parse a TAB (Text Anonymization Benchmark) annotation file into
    (entity_text, entity_type, tab_label) records, where tab_label is
    'DIRECT' or 'QUASI' per the human annotators. Use this to validate
    that is_structured_identifier()'s TOKENIZE/DP_NOISE split agrees
    with expert human judgment -- this is the empirical grounding for
    Objective 1's routing rule (see evaluate.py:
    router_vs_tab_agreement()).

    TAB download: https://github.com/NorskRegnesentral/text-anonymization-benchmark
    """
    import json

    with open(tab_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for doc in raw:
        text = doc["text"]
        for ann in doc.get("annotations", {}).values():
            for span in ann.get("entity_mentions", []):
                records.append({
                    "entity_text": text[span["start_offset"]:span["end_offset"]],
                    "entity_type": span.get("entity_type", "UNKNOWN"),
                    "tab_label": "DIRECT" if span.get("identifier_type") == "DIRECT" else "QUASI",
                })
    return records


if __name__ == "__main__":
    router = MechanismRouter()
    demo_entities = [
        ("Priya Sharma", "PERSON"),
        ("123-45-6789", "SSN"),
        ("Project Falcon", "PROJECT_CODE"),
        ("senior data analyst", "JOB_TITLE"),
        ("Chennai office", "LOCATION"),
        ("March 15, 2026", "DATE"),
    ]
    for text, etype in demo_entities:
        decision = router.route(text, etype, context="performance review draft")
        print(f"{text!r:25} -> {decision.mechanism.value:10} (severity={decision.severity:.2f}) -- {decision.reason}")
