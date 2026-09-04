from src.detection import (
    RuleBasedDetector,
    GLiNERDetector,
    HybridDetector,
)

from src.mechanism_router import MechanismRouter
from src.vault import SessionVault

from src.protection_pipeline import ProtectionPipeline


text = """
Priya Sharma works as a Senior Software Engineer
at Microsoft in Bangalore.

Her SSN is 123-45-6789.

Her employee ID is EMP-12345.

The team is working on Project Falcon.

The meeting is scheduled for March 15, 2026.

Contact her at priya.sharma@example.com.
"""


print("Loading Beyond Redaction pipeline...")

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


result = pipeline.protect(text)


print("\n" + "=" * 80)
print("ORIGINAL TEXT")
print("=" * 80)

print(result.original_text)


print("\n" + "=" * 80)
print("PROTECTED TEXT")
print("=" * 80)

print(result.protected_text)


print("\n" + "=" * 80)
print("ENTITY PROTECTION DECISIONS")
print("=" * 80)

for entity in result.protected_entities:

    print(
        f"{entity.original:30} "
        f"| {entity.entity_type:15} "
        f"| {entity.mechanism:10} "
        f"| {entity.surrogate}"
    )