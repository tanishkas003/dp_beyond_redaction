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


print("\n" + "=" * 80)
print("1. ORIGINAL PROMPT")
print("=" * 80)

print(text)


# --------------------------------------------------
# Build pipeline
# --------------------------------------------------

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


# --------------------------------------------------
# Protect
# --------------------------------------------------

result = pipeline.protect(text)

print("\n" + "=" * 80)
print("2. PROTECTED PROMPT")
print("=" * 80)

print(result.protected_text)


# --------------------------------------------------
# Simulated LLM response
# --------------------------------------------------

protected_response = (
    "The employee profile indicates that "
    + result.protected_text
)

print("\n" + "=" * 80)
print("3. SIMULATED LLM RESPONSE")
print("=" * 80)

print(protected_response)


# --------------------------------------------------
# Restore original values
# --------------------------------------------------

final_response = vault.restore_all(protected_response)

print("\n" + "=" * 80)
print("4. RESTORED FINAL RESPONSE")
print("=" * 80)

print(final_response)


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

vault.destroy()