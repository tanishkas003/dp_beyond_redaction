"""
baselines.py
============
The comparison systems for Objective 5. Each one deliberately embodies
a single-mechanism approach from the literature, so that the empirical
comparison in evaluate.py can show *why* the hybrid, reversibility-
aware router (mechanism_router.py) beats picking one mechanism for
everything -- not just assert it.

  - no_privacy          : the raw prompt, unmodified (upper bound on
                           utility, zero privacy -- reference point)
  - static_regex_mask    : classic DLP-style approach (what
                           Microsoft Purview-style sensitivity labels
                           and most enterprise DLP tools do): detect via
                           regex/dictionary, replace with a fixed
                           generic placeholder. No reversibility, no
                           utility-awareness, no severity tuning.
  - fixed_tokenize_all   : plain tokenization of every detected entity
                           (the "Hide and Seek" baseline every paper
                           reviewed compares against) -- no session
                           binding, no structure-based routing, so
                           quasi-identifiers lose their semantic
                           usefulness unnecessarily.
  - dp_only              : DP-noise applied to EVERY detected entity,
                           including structured identifiers -- this is
                           the ablation that empirically demonstrates
                           Carpentier et al.'s finding that word-level
                           DP breaks structured spans (an SSN noised in
                           embedding space becomes garbage, not a valid
                           SSN-shaped string).
"""

from __future__ import annotations

from detection import EntityDetector, DetectedEntity


def no_privacy(text: str, detector: EntityDetector) -> str:
    return text


def static_regex_mask(text: str, detector: EntityDetector, placeholder: str = "[REDACTED]") -> str:
    entities = sorted(detector.detect(text), key=lambda e: e.start, reverse=True)
    for e in entities:
        text = text[:e.start] + placeholder + text[e.end:]
    return text


def fixed_tokenize_all(text: str, detector: EntityDetector) -> tuple[str, dict]:
    """Static, reused-token baseline -- deliberately NOT session-bound,
    to reproduce the linkability weakness CON-QA measured in Hide-and-
    Seek. Same entity text always maps to the same token."""
    fixed_map: dict[str, str] = {}
    counters: dict[str, int] = {}
    entities = sorted(detector.detect(text), key=lambda e: e.start, reverse=True)
    for e in entities:
        if e.text not in fixed_map:
            counters[e.entity_type] = counters.get(e.entity_type, 0) + 1
            fixed_map[e.text] = f"[{e.entity_type}_{counters[e.entity_type]}]"
        text = text[:e.start] + fixed_map[e.text] + text[e.end:]
    return text, fixed_map


def dp_only(text: str, detector: EntityDetector, dp_sanitize_fn, epsilon: float) -> str:
    """Applies the DP-noise mechanism to every detected entity
    regardless of structure -- the ablation for "only DP" mentioned in
    Objective 5's comparison."""
    entities = sorted(detector.detect(text), key=lambda e: e.start, reverse=True)
    for e in entities:
        replacement = dp_sanitize_fn(e.text, e.entity_type)
        text = text[:e.start] + replacement + text[e.end:]
    return text
