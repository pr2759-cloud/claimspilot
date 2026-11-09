import httpx
import openai
import pytest

from llm import retry
from llm.retry import with_retries


def make_rate_limit_error():
    """Build the same error the OpenAI SDK raises on HTTP 429."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # Replace time.sleep so tests don't actually wait.
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)


def test_retries_then_succeeds():
    calls = {"n": 0}

    @with_retries(max_attempts=4)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise make_rate_limit_error()
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third try


def test_gives_up_after_max_attempts():
    calls = {"n": 0}

    @with_retries(max_attempts=3)
    def always_fails():
        calls["n"] += 1
        raise make_rate_limit_error()

    with pytest.raises(openai.RateLimitError):
        always_fails()
    assert calls["n"] == 3


def test_does_not_retry_other_errors():
    calls = {"n": 0}

    @with_retries(max_attempts=4)
    def broken():
        calls["n"] += 1
        raise ValueError("a bug, not a temporary failure")

    with pytest.raises(ValueError):
        broken()
    assert calls["n"] == 1  # no retries