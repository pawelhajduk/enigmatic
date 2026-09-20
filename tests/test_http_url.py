from enigmatic.providers.http import join_url


def test_join_url_does_not_double_v1() -> None:
    assert join_url("https://api.openai.com/v1", "/v1/chat/completions") == "https://api.openai.com/v1/chat/completions"
    assert join_url("https://api.anthropic.com", "/v1/messages") == "https://api.anthropic.com/v1/messages"
