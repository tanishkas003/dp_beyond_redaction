from src.dp_noise import generalize, sanitize_entity


def test_generalize_location():

    result = generalize(
        "Bangalore",
        "LOCATION",
    )

    assert result == "a regional office"


def test_generalize_job_title():

    result = generalize(
        "Senior Software Engineer",
        "JOB_TITLE",
    )

    assert result == "a professional role"


def test_generalize_date():

    result = generalize(
        "March 15, 2026",
        "DATE",
    )

    assert result == "a recent date"


def test_generalize_org():

    result = generalize(
        "Microsoft",
        "ORG",
    )

    assert result == "a partner organization"


def test_unknown_type_returns_generic_value():

    result = generalize(
        "Some Unknown Value",
        "UNKNOWN_TYPE",
    )

    assert result == "a related detail"


def test_sanitize_entity_uses_fallback_when_no_mechanism():

    result = sanitize_entity(
        entity_text="Bangalore",
        entity_type="LOCATION",
        mechanism=None,
        epsilon=5.0,
    )

    assert result == "a regional office"


def test_generalize_is_case_insensitive():

    result = generalize(
        "Bangalore",
        "location",
    )

    assert result == "a regional office"
