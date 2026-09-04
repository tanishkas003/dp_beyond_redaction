from src.mechanism_router import (
    MechanismRouter,
    Mechanism,
    UtilityDecision,
)


def test_ssn_routes_to_tokenize():
    router = MechanismRouter()

    decision = router.route(
        "123-45-6789",
        "SSN"
    )

    assert decision.utility == UtilityDecision.PROTECT
    assert decision.mechanism == Mechanism.TOKENIZE


def test_invoice_routes_to_tokenize():
    router = MechanismRouter()

    decision = router.route(
        "INV-2026-00452",
        "INVOICE_NO"
    )

    assert decision.utility == UtilityDecision.PROTECT
    assert decision.mechanism == Mechanism.TOKENIZE


def test_person_routes_to_tokenize():
    router = MechanismRouter()

    decision = router.route(
        "Priya Sharma",
        "PERSON"
    )

    assert decision.utility == UtilityDecision.PROTECT
    assert decision.mechanism == Mechanism.TOKENIZE


def test_job_title_routes_to_dp_noise():
    router = MechanismRouter()

    decision = router.route(
        "Senior Software Engineer",
        "JOB_TITLE"
    )

    assert decision.utility == UtilityDecision.PROTECT
    assert decision.mechanism == Mechanism.DP_NOISE


def test_location_routes_to_dp_noise():
    router = MechanismRouter()

    decision = router.route(
        "Bangalore",
        "LOCATION"
    )

    assert decision.utility == UtilityDecision.PROTECT
    assert decision.mechanism == Mechanism.DP_NOISE


def test_date_routes_to_dp_noise():
    router = MechanismRouter()

    decision = router.route(
        "March 15, 2026",
        "DATE"
    )

    assert decision.utility == UtilityDecision.PROTECT
    assert decision.mechanism == Mechanism.DP_NOISE


def test_retain_results_in_no_protection():

    def custom_retain_judge(entity_text, entity_type, context):
        return UtilityDecision.RETAIN

    router = MechanismRouter(
        retain_judge=custom_retain_judge
    )

    decision = router.route(
        "Python",
        "GENERIC_NOUN",
        "Explain Python programming."
    )

    assert decision.utility == UtilityDecision.RETAIN
    assert decision.mechanism == Mechanism.NONE
    assert decision.severity == 0.0


def test_custom_exact_recovery_judge_can_change_routing():

    def custom_exact_recovery_judge(
        entity_text,
        entity_type,
        context
    ):
        return True

    router = MechanismRouter(
        exact_recovery_judge=custom_exact_recovery_judge
    )

    decision = router.route(
        "Bangalore",
        "LOCATION",
        "Please include the exact city name."
    )

    assert decision.utility == UtilityDecision.PROTECT
    assert decision.mechanism == Mechanism.TOKENIZE


