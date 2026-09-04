from src.detection import (
    RuleBasedDetector,
    GLiNERDetector,
    HybridDetector,
)

from src.mechanism_router import MechanismRouter
from src.vault import SessionVault
from src.protection_pipeline import ProtectionPipeline


def build_pipeline():

    detector = HybridDetector(
        primary=RuleBasedDetector(),
        secondary=GLiNERDetector(),
    )

    router = MechanismRouter()

    vault = SessionVault()

    pipeline = ProtectionPipeline(
        detector=detector,
        router=router,
        vault=vault,
    )

    return pipeline, vault


def test_end_to_end_protection_and_restoration():

    text = (
        "Priya Sharma works as a Senior Software Engineer "
        "at Microsoft in Bangalore. "
        "Her SSN is 123-45-6789. "
        "Her employee ID is EMP-12345. "
        "Contact her at priya.sharma@example.com."
    )

    pipeline, vault = build_pipeline()

    # ------------------------------------------
    # STEP 1: Protect the text
    # ------------------------------------------

    result = pipeline.protect(text)

    # Protected text should be different
    assert result.protected_text != text

    # ------------------------------------------
    # STEP 2: Verify entities were processed
    # ------------------------------------------

    assert len(result.protected_entities) > 0

    # ------------------------------------------
    # STEP 3: Verify sensitive originals are gone
    # ------------------------------------------

    assert "Priya Sharma" not in result.protected_text
    assert "123-45-6789" not in result.protected_text
    assert "EMP-12345" not in result.protected_text
    assert "priya.sharma@example.com" not in result.protected_text

    # ------------------------------------------
    # STEP 4: Simulate an LLM response
    # ------------------------------------------

    protected_response = (
        "Here is the employee information: "
        + result.protected_text
    )

    # ------------------------------------------
    # STEP 5: Restore original values
    # ------------------------------------------

    restored_response = vault.restore_all(
        protected_response
    )

    # ------------------------------------------
    # STEP 6: Verify exact recovery
    # ------------------------------------------

    assert "Priya Sharma" in restored_response
    assert "123-45-6789" in restored_response
    assert "EMP-12345" in restored_response
    assert "priya.sharma@example.com" in restored_response

    # ------------------------------------------
    # STEP 7: Cleanup
    # ------------------------------------------

    vault.destroy()