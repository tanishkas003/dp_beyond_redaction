from src.detection import (
    RuleBasedDetector,
    GLiNERDetector,
    HybridDetector,
)

from src.mechanism_router import MechanismRouter


text = """
Hi, I'm Priya Sharma.

My SSN is 123-45-6789.

I work as a Senior Software Engineer
at Microsoft in Bangalore.

My employee ID is EMP-12345.

The team is working on Project Falcon.

The meeting is scheduled for March 15, 2026.

Contact me at priya.sharma@example.com.
"""


print("Loading detectors...")

rule_detector = RuleBasedDetector()
gliner_detector = GLiNERDetector()

detector = HybridDetector(
    primary=rule_detector,
    secondary=gliner_detector,
)

router = MechanismRouter()


print("\nDETECTION + ROUTING PIPELINE\n")
print("-" * 90)

entities = detector.detect(text)

for entity in entities:

    decision = router.route(
        entity.text,
        entity.entity_type,
        text,
    )

    print(
        f"{entity.text:30} "
        f"| {entity.entity_type:15} "
        f"| {decision.mechanism.value:10} "
        f"| {entity.source}"
    )

print("-" * 90)