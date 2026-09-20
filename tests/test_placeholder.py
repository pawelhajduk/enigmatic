from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.presidio_ops.placeholder import PlaceholderOperator


def test_placeholder_operator_uses_session_mapping() -> None:
    mapping = SessionMapping()
    op = PlaceholderOperator()
    first = op.operate("ada@example.com", {"mapping": mapping, "entity_type": "EMAIL_ADDRESS"})
    second = op.operate("ada@example.com", {"mapping": mapping, "entity_type": "EMAIL_ADDRESS"})
    assert first == second == "<EMAIL_ADDRESS_1>"
    assert mapping.restore_complete(first) == "ada@example.com"
