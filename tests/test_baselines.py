from src.baselines import dp_everything, no_privacy, static_masking, tokenize_everything
from src.detection import RuleBasedDetector


TEXT = "Email ana@example.com and SSN 123-45-6789."


def test_no_privacy_returns_input_unchanged():
    assert no_privacy(TEXT) == TEXT


def test_static_masking_removes_detected_values():
    protected = static_masking(TEXT, RuleBasedDetector(), placeholder="[MASK]")
    assert "ana@example.com" not in protected
    assert "123-45-6789" not in protected
    assert protected.count("[MASK]") == 2


def test_tokenize_everything_uses_typed_tokens_and_returns_mapping():
    protected, mapping = tokenize_everything(TEXT, RuleBasedDetector())
    assert "[EMAIL_1]" in protected and "[SSN_1]" in protected
    assert mapping["ana@example.com"] == "[EMAIL_1]"


def test_dp_everything_applies_to_all_detected_types():
    protected = dp_everything(TEXT, RuleBasedDetector(), lambda value, entity_type, epsilon: f"<{entity_type}>", 2.0)
    assert protected == "Email <EMAIL> and SSN <SSN>."
