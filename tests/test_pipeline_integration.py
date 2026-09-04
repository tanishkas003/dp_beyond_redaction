from src.detection import (
    RuleBasedDetector,
    GLiNERDetector,
    HybridDetector,
)

from src.mechanism_router import (
    MechanismRouter,
    Mechanism,
)


TEXT = """
Hi, I'm Priya Sharma.

My SSN is 123-45-6789.

I work as a Senior Software Engineer
at Microsoft in Bangalore.

My employee ID is EMP-12345.

The team is working on Project Falcon.

The meeting is scheduled for March 15, 2026.

Contact me at priya.sharma@example.com.
"""


def test_detection_and_routing_pipeline():

    rule_detector = RuleBasedDetector()
    gliner_detector = GLiNERDetector()

    detector = HybridDetector(
        primary=rule_detector,
        secondary=gliner_detector,
    )

    router = MechanismRouter()

    entities = detector.detect(TEXT)

    decisions = {}

    for entity in entities:

        decision = router.route(
            entity.text,
            entity.entity_type,
            TEXT,
        )

        decisions[entity.text] = decision.mechanism

    # Exact recovery → TOKENIZE
    assert decisions["Priya Sharma"] == Mechanism.TOKENIZE
    assert decisions["123-45-6789"] == Mechanism.TOKENIZE
    assert decisions["EMP-12345"] == Mechanism.TOKENIZE
    assert decisions["Project Falcon"] == Mechanism.TOKENIZE
    assert decisions["priya.sharma@example.com"] == Mechanism.TOKENIZE

    # Approximate semantic context → DP_NOISE
    assert decisions["Senior Software Engineer"] == Mechanism.DP_NOISE
    assert decisions["Microsoft"] == Mechanism.DP_NOISE
    assert decisions["Bangalore"] == Mechanism.DP_NOISE
    assert decisions["March 15, 2026"] == Mechanism.DP_NOISE