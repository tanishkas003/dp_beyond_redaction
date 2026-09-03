"""
utility_assessor.py
====================
Objective 3: instead of applying DP noise at a fixed epsilon and hoping
the sanitized prompt is still usable, this gate predicts -- BEFORE
spending a cloud LLM call -- whether the response will clear a quality
bar, and only proceeds if it does. Architecture follows Carpentier et
al.'s (2024/2025) utility-assessor middleware: a small local regressor
trained on cheap, reference-free features, not the LLM itself.

The gate is what turns "measure the epsilon-vs-utility tradeoff" into
something actually usable: without it, a caller has no way to know
whether epsilon=3 will produce a great answer or a useless one for
THIS particular prompt, short of paying for the query and finding out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UtilityFeatures:
    semantic_similarity: float   # cosine sim(original, sanitized) via sentence-transformers
    epsilon: float                # DP strength used (0 if no DP applied)
    protected_span_ratio: float   # fraction of tokens that were tokenized/DP-noised
    sanitized_length_ratio: float  # len(sanitized) / len(original)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.semantic_similarity, self.epsilon,
            self.protected_span_ratio, self.sanitized_length_ratio,
        ])


def extract_features(original: str, sanitized: str, epsilon: float,
                      n_protected: int, n_total_tokens: int, embed_fn) -> UtilityFeatures:
    orig_vec = embed_fn(original)
    san_vec = embed_fn(sanitized)
    cos_sim = float(np.dot(orig_vec, san_vec) /
                     (np.linalg.norm(orig_vec) * np.linalg.norm(san_vec) + 1e-9))
    return UtilityFeatures(
        semantic_similarity=cos_sim,
        epsilon=epsilon,
        protected_span_ratio=n_protected / max(n_total_tokens, 1),
        sanitized_length_ratio=len(sanitized) / max(len(original), 1),
    )


class UtilityAssessor:
    """Trained on (features, observed_quality) pairs where
    observed_quality is a downstream task score (e.g. task-success
    label, ROUGE-L against a reference, or human/LLM-judge rating on a
    held-out calibration set built during Objective 4's evaluation
    runs). Falls back to a similarity-only heuristic if untrained."""

    def __init__(self, quality_threshold: float = 0.6):
        self.model = None
        self.quality_threshold = quality_threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(max_depth=4)
        self.model.fit(X, y)

    def predict(self, features: UtilityFeatures) -> float:
        if self.model is not None:
            return float(np.clip(self.model.predict(features.to_array().reshape(1, -1))[0], 0, 1))
        # heuristic fallback: semantic similarity alone, before training data exists
        return features.semantic_similarity

    def gate(self, features: UtilityFeatures) -> bool:
        """Return True if the sanitized prompt is predicted good enough
        to send to the cloud LLM."""
        return self.predict(features) >= self.quality_threshold


def query_with_gate(
    original_prompt: str,
    sanitize_fn,               # (prompt, epsilon) -> sanitized_prompt
    assessor: UtilityAssessor,
    embed_fn,
    cloud_llm_call,            # sanitized_prompt -> response
    local_fallback_call=None,  # sanitized_prompt -> response, used if all retries fail
    epsilon_schedule: Optional[list[float]] = None,
    n_protected: int = 0,
    n_total_tokens: int = 1,
):
    """Objective 3's full loop: try progressively larger epsilon
    (less noise) until the assessor predicts an acceptable response,
    or fall back to a local model rather than send a low-utility
    prompt to the paid cloud API."""
    epsilon_schedule = epsilon_schedule or [1.0, 3.0, 6.0, 10.0]
    for epsilon in epsilon_schedule:
        sanitized = sanitize_fn(original_prompt, epsilon)
        features = extract_features(original_prompt, sanitized, epsilon,
                                     n_protected, n_total_tokens, embed_fn)
        if assessor.gate(features):
            return cloud_llm_call(sanitized), epsilon, features
    if local_fallback_call is not None:
        sanitized = sanitize_fn(original_prompt, epsilon_schedule[-1])
        return local_fallback_call(sanitized), None, None
    # last resort: send the least-noised version anyway, but flag it
    sanitized = sanitize_fn(original_prompt, epsilon_schedule[-1])
    return cloud_llm_call(sanitized), epsilon_schedule[-1], "LOW_CONFIDENCE"
