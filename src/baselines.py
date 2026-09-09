"""Reference privacy baselines used to evaluate the hybrid pipeline.

All baselines accept the project's detector so that comparisons use the same
set of detected spans. Replacements are made right-to-left, preserving offsets.
"""
from __future__ import annotations

from typing import Callable

try:
    from .detection import EntityDetector
except ImportError:  # pragma: no cover
    from detection import EntityDetector


def _entities_right_to_left(text: str, detector: EntityDetector):
    """Return non-overlapping spans in safe replacement order."""
    entities = sorted(detector.detect(text), key=lambda entity: entity.start, reverse=True)
    accepted, occupied_start = [], len(text)
    for entity in entities:
        if entity.end <= occupied_start:
            accepted.append(entity)
            occupied_start = entity.start
    return accepted


def no_privacy(text: str, detector: EntityDetector | None = None) -> str:
    """No-protection upper-utility / zero-privacy reference."""
    return text


def static_masking(text: str, detector: EntityDetector, placeholder: str | None = None) -> str:
    """Replace every detected entity with a non-reversible static tag.

    By default tags retain only the entity class (``[EMAIL]``, ``[SSN]``),
    rather than a value-specific token.  Supplying ``placeholder`` produces
    the classic one-tag DLP baseline (for example ``[REDACTED]``).
    """
    for entity in _entities_right_to_left(text, detector):
        replacement = placeholder if placeholder is not None else f"[{entity.entity_type.upper()}]"
        text = text[:entity.start] + replacement + text[entity.end:]
    return text


def tokenize_everything(text: str, detector: EntityDetector) -> tuple[str, dict[str, str]]:
    """Tokenize every entity with stable typed placeholders.

    The mapping is intentionally plaintext and local to this call: this is a
    baseline rather than the project's encrypted session vault.
    """
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    for entity in _entities_right_to_left(text, detector):
        if entity.text not in mapping:
            entity_type = entity.entity_type.upper()
            counters[entity_type] = counters.get(entity_type, 0) + 1
            mapping[entity.text] = f"[{entity_type}_{counters[entity_type]}]"
        text = text[:entity.start] + mapping[entity.text] + text[entity.end:]
    return text, mapping


def dp_everything(text: str, detector: EntityDetector, dp_sanitize_fn: Callable[..., str] | None = None,
                  epsilon: float = 5.0) -> str:
    """Generalize every detected entity, including structured identifiers."""
    if dp_sanitize_fn is None:
        try:
            from .dp_noise import generalize
        except ImportError:  # pragma: no cover
            from dp_noise import generalize
        dp_sanitize_fn = generalize
    for entity in _entities_right_to_left(text, detector):
        try:
            replacement = dp_sanitize_fn(entity.text, entity.entity_type, epsilon)
        except TypeError:
            replacement = dp_sanitize_fn(entity.text, entity.entity_type)
        text = text[:entity.start] + replacement + text[entity.end:]
    return text


# Names retained for compatibility with the original skeleton.
static_regex_mask = static_masking
fixed_tokenize_all = tokenize_everything
dp_only = dp_everything