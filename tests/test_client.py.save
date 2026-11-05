import os
import pytest
from llm.client import complete

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="no key")
def test_anthropic_returns_text():
    r = complete([{"role": "user", "content": "Say OK"}], provider="anthropic", max_tokens=10)
    assert r.text and r.input_tokens > 0
