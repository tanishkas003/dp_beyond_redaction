from src.mechanism_router import MechanismRouter

router = MechanismRouter()

examples = [
    ("123-45-6789", "SSN"),
    ("INV-2026-00452", "INVOICE_NO"),
    ("Priya Sharma", "PERSON"),
    ("Senior Software Engineer", "JOB_TITLE"),
    ("Bangalore", "LOCATION"),
    ("March 15, 2026", "DATE"),
]

for entity, entity_type in examples:
    decision = router.route(entity, entity_type)

    print("\nEntity:", entity)
    print("Type:", entity_type)
    print("Utility:", decision.utility)
    print("Mechanism:", decision.mechanism)
    print("Reason:", decision.reason)