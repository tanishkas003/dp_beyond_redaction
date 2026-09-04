from src.detection import RuleBasedDetector


def get_entities(text):
    detector = RuleBasedDetector()
    return detector.detect(text)


def has_entity(entities, text, entity_type):
    return any(
        entity.text == text
        and entity.entity_type == entity_type
        for entity in entities
    )


def test_ssn_detection():
    entities = get_entities(
        "My SSN is 123-45-6789."
    )

    assert has_entity(
        entities,
        "123-45-6789",
        "SSN"
    )


def test_email_detection():
    entities = get_entities(
        "Contact john.smith@example.com for details."
    )

    assert has_entity(
        entities,
        "john.smith@example.com",
        "EMAIL"
    )


def test_employee_id_detection():
    entities = get_entities(
        "Employee EMP-12345 submitted the report."
    )

    assert has_entity(
        entities,
        "EMP-12345",
        "EMPLOYEE_ID"
    )


def test_invoice_detection():
    entities = get_entities(
        "Please process invoice INV-2026-00452."
    )

    assert has_entity(
        entities,
        "INV-2026-00452",
        "INVOICE_NO"
    )


def test_project_code_detection():
    entities = get_entities(
        "The team is working on Project Falcon."
    )

    assert has_entity(
        entities,
        "Project Falcon",
        "PROJECT_CODE"
    )


def test_project_short_code_detection():
    entities = get_entities(
        "The new work is tracked under PROJ-782."
    )

    assert has_entity(
        entities,
        "PROJ-782",
        "PROJECT_CODE"
    )


def test_date_detection():
    entities = get_entities(
        "The meeting is scheduled for March 15, 2026."
    )

    assert has_entity(
        entities,
        "March 15, 2026",
        "DATE"
    )


def test_ip_address_detection():
    entities = get_entities(
        "The server IP is 192.168.1.25."
    )

    assert has_entity(
        entities,
        "192.168.1.25",
        "IP_ADDRESS"
    )