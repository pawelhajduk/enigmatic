from enigmatic.presidio_ops.mapping import SessionMapping
from enigmatic.restore import StreamRestorer


def test_split_placeholder_across_chunks() -> None:
    mapping = SessionMapping()
    token = mapping.placeholder_for("EMAIL_ADDRESS", "ada@example.com")
    assert token == "<EMAIL_ADDRESS_1>"
    restorer = StreamRestorer(mapping)
    first = restorer.feed("Hello <EMAIL_ADDR")
    second = restorer.feed("ESS_1>!")
    assert first == "Hello "
    assert second == "ada@example.com!"
    assert restorer.flush() == ""


def test_longest_match_first() -> None:
    mapping = SessionMapping()
    mapping.placeholder_for("EMAIL_ADDRESS", "short@example.com")
    # Force a longer token that shares a prefix by using a custom reverse entry
    mapping._reverse["<EMAIL_ADDRESS_10>"] = "long@example.com"  # noqa: SLF001
    mapping._forward[("EMAIL_ADDRESS", "long@example.com")] = "<EMAIL_ADDRESS_10>"  # noqa: SLF001
    restorer = StreamRestorer(mapping)
    out = restorer.feed("x <EMAIL_ADDRESS_10> y")
    out += restorer.flush()
    assert "long@example.com" in out
    assert "short@example.com" not in out
