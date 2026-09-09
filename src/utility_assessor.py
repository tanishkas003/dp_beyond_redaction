"""A lightweight, reference-free utility gate for protected prompts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class UtilityFeatures:
    semantic_similarity: float
    epsilon: float
    protected_span_ratio: float
    sanitized_length_ratio: float

    def to_array(self) -> np.ndarray:
        return np.array([self.semantic_similarity, self.epsilon,
                         self.protected_span_ratio, self.sanitized_length_ratio], dtype=float)


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    predicted_utility: float
    features: UtilityFeatures
    reason: str


def _token_jaccard(left: str, right: str) -> float:
    """Dependency-free similarity fallback suitable for a pre-LLM gate."""
    left_tokens, right_tokens = set(re.findall(r"\w+", left.lower())), set(re.findall(r"\w+", right.lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def extract_features(original: str, sanitized: str, epsilon: float = 0.0,
                     n_protected: int = 0, n_total_tokens: Optional[int] = None,
                     embed_fn: Optional[Callable[[str], np.ndarray]] = None) -> UtilityFeatures:
    """Extract local features; an embedding function is optional."""
    if embed_fn is None:
        similarity = _token_jaccard(original, sanitized)
    else:
        original_vector, sanitized_vector = np.asarray(embed_fn(original), dtype=float), np.asarray(embed_fn(sanitized), dtype=float)
        denominator = np.linalg.norm(original_vector) * np.linalg.norm(sanitized_vector)
        similarity = float(np.dot(original_vector, sanitized_vector) / denominator) if denominator else 0.0
        similarity = float(np.clip(similarity, 0.0, 1.0))
    total = n_total_tokens if n_total_tokens is not None else len(re.findall(r"\w+", original))
    return UtilityFeatures(similarity, max(float(epsilon), 0.0),
                           min(max(n_protected / max(total, 1), 0.0), 1.0),
                           len(sanitized) / max(len(original), 1))


class UtilityAssessor:
    """Transparent feature-and-threshold gate; no training data is required."""
    def __init__(self, quality_threshold: float = 0.6, similarity_weight: float = 0.8,
                 protection_penalty: float = 0.15, length_penalty: float = 0.05):
        if not 0 <= quality_threshold <= 1:
            raise ValueError("quality_threshold must be between 0 and 1")
        self.quality_threshold = quality_threshold
        self.similarity_weight = similarity_weight
        self.protection_penalty = protection_penalty
        self.length_penalty = length_penalty

    def predict(self, features: UtilityFeatures) -> float:
        length_drift = min(abs(1.0 - features.sanitized_length_ratio), 1.0)
        # Epsilon affects the already-observed sanitized text; it is not itself
        # a quality score, so it is deliberately not double-counted here.
        score = (self.similarity_weight * features.semantic_similarity + (1 - self.similarity_weight)
                 - self.protection_penalty * features.protected_span_ratio - self.length_penalty * length_drift)
        return float(np.clip(score, 0.0, 1.0))

    def gate(self, features: UtilityFeatures) -> bool:
        return self.predict(features) >= self.quality_threshold

    def assess(self, features: UtilityFeatures) -> GateDecision:
        score = self.predict(features)
        allowed = score >= self.quality_threshold
        return GateDecision(allowed, score, features, "utility meets threshold" if allowed else "utility below threshold")


def query_with_gate(original_prompt: str, sanitize_fn: Callable[[str, float], str], assessor: UtilityAssessor,
                    embed_fn: Optional[Callable[[str], np.ndarray]], cloud_llm_call: Callable[[str], str],
                    local_fallback_call: Optional[Callable[[str], str]] = None,
                    epsilon_schedule: Optional[list[float]] = None, n_protected: int = 0,
                    n_total_tokens: Optional[int] = None):
    """Query with the first allowed prompt, otherwise use the configured fallback.

    Returns ``(response, epsilon, gate_decision)``; fallback responses carry
    ``epsilon=None`` when a local fallback is used.
    """
    schedule = epsilon_schedule or [1.0, 3.0, 6.0, 10.0]
    if not schedule:
        raise ValueError("epsilon_schedule must not be empty")
    last_decision = None
    for epsilon in schedule:
        sanitized = sanitize_fn(original_prompt, epsilon)
        features = extract_features(original_prompt, sanitized, epsilon, n_protected, n_total_tokens, embed_fn)
        decision = assessor.assess(features)
        last_decision = decision
        if decision.allowed:
            return cloud_llm_call(sanitized), epsilon, decision
    sanitized = sanitize_fn(original_prompt, schedule[-1])
    if local_fallback_call is not None:
        return local_fallback_call(sanitized), None, last_decision
    return cloud_llm_call(sanitized), schedule[-1], last_decision
