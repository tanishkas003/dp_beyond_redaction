"""
dp_noise.py
===========
The DP-noise branch: applied only to free-text quasi-identifiers that
the router (mechanism_router.py) decided do NOT need exact reversible
recovery. Implements word-level metric differential privacy in
sentence-embedding space (the same family used by Carpentier et al.'s
prompt-DP work), plus a cheap category-generalization fallback for
when no embedding model is available.

This is deliberately NOT applied to structured identifiers -- see
mechanism_router.is_structured_identifier() for why: there's no
meaningful "nearby" SSN, and Carpentier et al.'s own results show
fixed-epsilon word DP destroys usability on exactly this kind of span.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cheap fallback: category generalization (no embedding model needed)
# ---------------------------------------------------------------------------

_GENERALIZATION_MAP = {
    "LOCATION": lambda v: "a regional office",
    "JOB_TITLE": lambda v: "a professional role",
    "DATE": lambda v: "a recent date",
    "ORG": lambda v: "a partner organization",
}


def generalize(entity_text: str, entity_type: str) -> str:
    fn = _GENERALIZATION_MAP.get(entity_type.upper())
    return fn(entity_text) if fn else "a related detail"


# ---------------------------------------------------------------------------
# Metric DP in embedding space (Laplace mechanism + nearest-neighbor decode)
# ---------------------------------------------------------------------------
# This mirrors the standard word-level metric-DP recipe used across the
# literature (SanText/CusText-style): embed the word, add calibrated
# multivariate Laplace noise scaled by 1/epsilon, then snap to the
# nearest word in a candidate vocabulary. Smaller epsilon = more noise
# = stronger privacy = more semantic drift; this is exactly the
# tradeoff Objective 3's Utility Assessor is built to navigate rather
# than leaving the caller to guess a good epsilon blindly.

@dataclass
class MetricDPConfig:
    epsilon: float = 5.0     # smaller = more private, more distortion
    embedding_dim: int = 384  # matches e.g. all-MiniLM-L6-v2


class MetricDPMechanism:
    def __init__(self, embed_fn, vocabulary: list[str], config: Optional[MetricDPConfig] = None):
        """`embed_fn`: str -> np.ndarray, e.g. a sentence-transformers
        model's .encode(). `vocabulary`: candidate replacement words
        per entity type, ideally domain-appropriate (job titles, city
        names, etc.) -- keep this list large enough that the noised
        point usually lands near something plausible."""
        self.embed_fn = embed_fn
        self.vocabulary = vocabulary
        self.config = config or MetricDPConfig()
        self._vocab_embeddings = np.stack([embed_fn(w) for w in vocabulary])

    def _laplace_noise(self, dim: int, epsilon: float) -> np.ndarray:
        # Multivariate Laplace via the standard direction*magnitude construction
        direction = np.random.normal(size=dim)
        direction /= np.linalg.norm(direction) + 1e-9
        magnitude = np.random.gamma(shape=dim, scale=1.0 / epsilon)
        return direction * magnitude

    def sanitize(self, word: str, epsilon: Optional[float] = None) -> str:
        eps = epsilon or self.config.epsilon
        vec = self.embed_fn(word)
        noisy = vec + self._laplace_noise(vec.shape[0], eps)
        dists = np.linalg.norm(self._vocab_embeddings - noisy, axis=1)
        return self.vocabulary[int(np.argmin(dists))]


def sanitize_entity(entity_text: str, entity_type: str, mechanism: Optional[MetricDPMechanism],
                     epsilon: float) -> str:
    """Top-level function called by the pipeline for every DP_NOISE-routed
    span. Falls back to category generalization if no embedding
    mechanism was configured (fine for a first working version;
    swap in MetricDPMechanism for the full Objective 3 experiment)."""
    if mechanism is None:
        return generalize(entity_text, entity_type)
    return mechanism.sanitize(entity_text, epsilon=epsilon)


if __name__ == "__main__":
    print(generalize("Chennai office", "LOCATION"))
    print(generalize("senior data analyst", "JOB_TITLE"))
