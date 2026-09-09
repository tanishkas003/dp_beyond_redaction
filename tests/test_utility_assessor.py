import numpy as np

from src.utility_assessor import UtilityAssessor, extract_features, query_with_gate


def test_extract_features_without_embeddings_is_bounded():
    features = extract_features("write report for Alice", "write report for [PERSON_1]", n_protected=1)
    assert 0 <= features.semantic_similarity <= 1
    assert features.protected_span_ratio == 0.25


def test_assessor_rejects_low_similarity_prompt():
    assessor = UtilityAssessor(quality_threshold=0.7)
    features = extract_features("summarize quarterly sales", "[REDACTED]", n_protected=3)
    assert assessor.assess(features).allowed is False


def test_query_uses_local_fallback_when_no_candidate_passes():
    response, epsilon, decision = query_with_gate(
        "summarize quarterly sales", lambda prompt, eps: "[REDACTED]",
        UtilityAssessor(quality_threshold=0.9), None, lambda prompt: "cloud",
        local_fallback_call=lambda prompt: "local", epsilon_schedule=[1.0, 2.0], n_protected=2,
    )
    assert (response, epsilon, decision.allowed) == ("local", None, False)


def test_extract_features_accepts_embedding_function():
    features = extract_features("a", "b", embed_fn=lambda text: np.array([1.0, 0.0]) if text == "a" else np.array([0.0, 1.0]))
    assert features.semantic_similarity == 0.0
