from src.vault import SessionVault


def test_protect_and_restore_tokenized_entity():

    vault = SessionVault()

    surrogate = vault.protect(
        "Priya Sharma",
        "PERSON",
        mechanism="TOKENIZE",
    )

    restored = vault.restore(surrogate)

    assert restored == "Priya Sharma"


def test_protect_and_restore_dp_entity():

    vault = SessionVault()

    surrogate = vault.protect(
        "Bangalore",
        "LOCATION",
        mechanism="DP_NOISE",
    )

    restored = vault.restore(surrogate)

    assert restored == "Bangalore"


def test_restore_all():

    vault = SessionVault()

    person_surrogate = vault.protect(
        "Priya Sharma",
        "PERSON",
        mechanism="TOKENIZE",
    )

    location_surrogate = vault.protect(
        "Bangalore",
        "LOCATION",
        mechanism="DP_NOISE",
    )

    response = (
        f"{person_surrogate} works from "
        f"{location_surrogate}."
    )

    restored = vault.restore_all(response)

    assert "Priya Sharma" in restored
    assert "Bangalore" in restored


def test_destroy_clears_vault():

    vault = SessionVault()

    surrogate = vault.protect(
        "Priya Sharma",
        "PERSON",
        mechanism="TOKENIZE",
    )

    assert vault.restore(surrogate) == "Priya Sharma"

    vault.destroy()

    assert vault.restore(surrogate) is None

def test_structured_entities_get_unique_tokens():

    vault = SessionVault()

    ssn = vault.protect(
        "123-45-6789",
        "SSN",
        mechanism="TOKENIZE",
    )

    employee_id = vault.protect(
        "EMP-12345",
        "EMPLOYEE_ID",
        mechanism="TOKENIZE",
    )

    email = vault.protect(
        "priya.sharma@example.com",
        "EMAIL",
        mechanism="TOKENIZE",
    )

    assert ssn != employee_id
    assert employee_id != email
    assert ssn != email

    assert vault.restore(ssn) == "123-45-6789"
    assert vault.restore(employee_id) == "EMP-12345"
    assert vault.restore(email) == "priya.sharma@example.com"

def test_restore_all_restores_structured_placeholders():

    vault = SessionVault()

    ssn = vault.protect(
        "123-45-6789",
        "SSN",
        mechanism="TOKENIZE",
    )

    employee_id = vault.protect(
        "EMP-12345",
        "EMPLOYEE_ID",
        mechanism="TOKENIZE",
    )

    email = vault.protect(
        "priya.sharma@example.com",
        "EMAIL",
        mechanism="TOKENIZE",
    )

    protected_response = (
        f"SSN: {ssn}, "
        f"Employee ID: {employee_id}, "
        f"Email: {email}"
    )

    restored = vault.restore_all(protected_response)

    assert "123-45-6789" in restored
    assert "EMP-12345" in restored
    assert "priya.sharma@example.com" in restored

    assert ssn not in restored
    assert employee_id not in restored
    assert email not in restored


def test_store_pre_generated_mapping():

    vault = SessionVault()

    vault.store_mapping(
        surrogate="[PERSON_1]",
        original="Priya Sharma",
        entity_type="PERSON",
        mechanism="TOKENIZE",
    )

    restored = vault.restore("[PERSON_1]")

    assert restored == "Priya Sharma"