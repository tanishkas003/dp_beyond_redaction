from src.mechanism_router import (
    MechanismRouter,
    Mechanism,
)


def test_location_defaults_to_dp_noise():
    router = MechanismRouter()

    decision = router.route(
        "Bangalore",
        "LOCATION",
        "Summarize the employee profile."
    )

    assert decision.mechanism == Mechanism.DP_NOISE


def test_location_requires_exact_recovery_when_requested():
    router = MechanismRouter()

    decision = router.route(
        "Bangalore",
        "LOCATION",
        "Use the exact original city name in the final document."
    )

    assert decision.mechanism == Mechanism.TOKENIZE


def test_job_title_can_use_approximate_representation():
    router = MechanismRouter()

    decision = router.route(
        "Senior Software Engineer",
        "JOB_TITLE",
        "Provide a general summary of the employee profile."
    )

    assert decision.mechanism == Mechanism.DP_NOISE


def test_job_title_can_require_exact_recovery():
    router = MechanismRouter()

    decision = router.route(
        "Senior Software Engineer",
        "JOB_TITLE",
        "Preserve the exact original job title."
    )

    assert decision.mechanism == Mechanism.TOKENIZE