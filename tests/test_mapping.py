from enigmatic.presidio_ops.mapping import SessionMapping


def test_same_surface_form_reuses_placeholder() -> None:
    mapping = SessionMapping()
    first = mapping.placeholder_for("EMAIL_ADDRESS", "ada@example.com")
    second = mapping.placeholder_for("EMAIL_ADDRESS", "ada@example.com")
    assert first == second
    assert first == "<EMAIL_ADDRESS_1>"


def test_different_entities_and_values_get_separate_indexes() -> None:
    mapping = SessionMapping()
    email = mapping.placeholder_for("EMAIL_ADDRESS", "ada@example.com")
    other = mapping.placeholder_for("EMAIL_ADDRESS", "bob@example.com")
    card = mapping.placeholder_for("CREDIT_CARD", "4111111111111111")
    assert email == "<EMAIL_ADDRESS_1>"
    assert other == "<EMAIL_ADDRESS_2>"
    assert card == "<CREDIT_CARD_1>"
    restored = mapping.restore_complete(f"write to {email} or {other} card {card}")
    assert restored == "write to ada@example.com or bob@example.com card 4111111111111111"


def test_restore_complete_walks_nested_json() -> None:
    mapping = SessionMapping()
    token = mapping.placeholder_for("PHONE_NUMBER", "555-0100")
    payload = {"choices": [{"message": {"content": f"call {token}"}}]}
    restored = mapping.restore_complete_any(payload)
    assert restored == {"choices": [{"message": {"content": "call 555-0100"}}]}
