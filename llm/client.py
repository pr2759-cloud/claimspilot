import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def _split_system(messages):
    """Anthropic takes the system prompt separately; OpenAI takes it as a message."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


def _call_anthropic(messages, max_tokens, temperature):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    model = os.environ["ANTHROPIC_MODEL"]
    system, rest = _split_system(messages)
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": rest}
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    r = client.messages.create(**kwargs)
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, model, r.usage.input_tokens, r.usage.output_tokens


def _call_openai(messages, max_tokens, temperature):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env
    model = os.environ["OPENAI_MODEL"]
    kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    r = client.chat.completions.create(**kwargs)
    text = r.choices[0].message.content or ""
    return text, model, r.usage.prompt_tokens, r.usage.completion_tokens


_BACKENDS = {"anthropic": _call_anthropic, "openai": _call_openai}


def complete(messages, tools=None, schema=None, max_tokens=1024,
             temperature=None, provider=None) -> LLMResponse:
    if tools or schema:
        raise NotImplementedError("tools arrive in Module 3, schema in Module 1")
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    start = time.perf_counter()
    text, model, tin, tout = _BACKENDS[provider](messages, max_tokens, temperature)
    latency = (time.perf_counter() - start) * 1000
    return LLMResponse(text, provider, model, tin, tout, round(latency, 1))