"""
detection.py
============
Hybrid entity detection: a precision-oriented NER pass (GLiNER) plus a
recall-oriented local-LLM pass, combined -- per LegalGuardian's finding
that this hybrid outperforms either alone (Demir et al. 2025). This is
the first stage of the pipeline, feeding mechanism_router.py.

GLiNER is not installed by default in this skeleton (it pulls in
torch + transformers, heavy for a quick local run) -- install it with
`pip install gliner` and swap in GLiNERDetector for real detection.
The interface below lets you develop and test the router/vault/DP
modules against the RuleBasedDetector first, then swap in the real
model without touching downstream code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass
class DetectedEntity:
    text: str
    entity_type: str
    start: int
    end: int
    source: str  # "regex", "gliner", or "llm" -- useful for precision/recall analysis


class EntityDetector(Protocol):
    def detect(self, text: str) -> list[DetectedEntity]: ...


# ---------------------------------------------------------------------------
# Fast, dependency-free detector for development and unit tests
# ---------------------------------------------------------------------------

_DEV_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PROJECT_CODE": re.compile(r"\bProject\s+[A-Z][a-z]+\b"),
    "DATE": re.compile(r"\b[A-Z][a-z]+ \d{1,2},? \d{4}\b"),
}


class RuleBasedDetector:
    """Regex-only detector. High precision, low recall by design --
    used as the development fallback and as the "static masking"
    baseline in evaluate.py / baselines.py."""

    def detect(self, text: str) -> list[DetectedEntity]:
        found = []
        for etype, pattern in _DEV_PATTERNS.items():
            for m in pattern.finditer(text):
                found.append(DetectedEntity(m.group(), etype, m.start(), m.end(), source="regex"))
        return found


# ---------------------------------------------------------------------------
# Real hybrid detector (GLiNER + local LLM), wire up before evaluation
# ---------------------------------------------------------------------------

class GLiNERDetector:
    """Wraps GLiNER (pip install gliner) for zero-shot NER over a
    custom label set relevant to enterprise prompts. High precision on
    named-entity-shaped spans."""

    LABELS = [
        "person", "location", "organization", "job title", "date",
        "project name", "email", "phone number", "employee id",
    ]

    def __init__(self, model_name: str = "urchade/gliner_multi-v2.1"):
        from gliner import GLiNER  # local import: heavy optional dependency
        self.model = GLiNER.from_pretrained(model_name)

    def detect(self, text: str) -> list[DetectedEntity]:
        raw = self.model.predict_entities(text, self.LABELS)
        return [
            DetectedEntity(r["text"], r["label"].upper().replace(" ", "_"),
                            r["start"], r["end"], source="gliner")
            for r in raw
        ]


class LocalLLMDetector:
    """Second-pass, recall-oriented detector: prompts a small local LLM
    (e.g. Qwen2.5-7B via Ollama) to flag anything GLiNER missed --
    nested entities, implicit references, informal internal codenames.
    `llm_call` is injected so this works with any provider."""

    PROMPT_TEMPLATE = (
        "List any sensitive or identifying details in the text below that "
        "are NOT already in this list: {already_found}.\n"
        "Return one per line as `TEXT | TYPE`, or `NONE` if there aren't any.\n\n"
        "Text: {text}"
    )

    def __init__(self, llm_call):
        self.llm_call = llm_call

    def detect(self, text: str, already_found: Iterable[str] = ()) -> list[DetectedEntity]:
        prompt = self.PROMPT_TEMPLATE.format(already_found=list(already_found), text=text)
        raw = self.llm_call(prompt).strip()
        found = []
        if raw.upper() != "NONE":
            for line in raw.splitlines():
                if "|" not in line:
                    continue
                span_text, etype = [p.strip() for p in line.split("|", 1)]
                start = text.find(span_text)
                if start == -1:
                    continue
                found.append(DetectedEntity(span_text, etype.upper(), start,
                                             start + len(span_text), source="llm"))
        return found


class HybridDetector:
    """Combines GLiNER (precision) + LocalLLMDetector (recall), per
    LegalGuardian's recommendation. Deduplicates overlapping spans,
    preferring the GLiNER hit when both agree."""

    def __init__(self, primary: EntityDetector, secondary: LocalLLMDetector | None = None):
        self.primary = primary
        self.secondary = secondary

    def detect(self, text: str) -> list[DetectedEntity]:
        primary_hits = self.primary.detect(text)
        if self.secondary is None:
            return primary_hits
        already = [e.text for e in primary_hits]
        secondary_hits = self.secondary.detect(text, already_found=already)
        # drop secondary hits that overlap an existing primary hit
        merged = list(primary_hits)
        for hit in secondary_hits:
            if not any(hit.start < p.end and hit.end > p.start for p in primary_hits):
                merged.append(hit)
        return sorted(merged, key=lambda e: e.start)


if __name__ == "__main__":
    sample = "Hi, I'm Priya Sharma, SSN 123-45-6789, working on Project Falcon. Meeting on March 15, 2026."
    detector = RuleBasedDetector()
    for e in detector.detect(sample):
        print(e)
