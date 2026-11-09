"""One function, complete(), for every model call, whatever the provider.

Callers send OpenAI-style messages:
    [{"role": "system" | "user" | "assistant", "content": "..."}]
"""
import os
import time
from dataclasses import dataclass

import anthropic
import openai
from dotenv import load_dotenv

from llm.pricing import estimate_cost
from llm.retry import with_retries
from llm.tracing import log_event

load_dotenv()

DEFAULT_MAX_TOKENS = 1024
TIMEOUT_SECONDS = 30.0
SCHEMA_TOOL_NAME = "submit_result"  # name given to the JSON schema for structured output


@dataclass
class LLMResponse:
    text: str                 # plain text, or a JSON string when a schema was given
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None = None  # None when the model's price isn't in pricing.py


# Clients are created on first use, so having only one API key doesn't crash the import.
# max_retries=0 turns off the SDKs' hidden built-in retries: our own retry (Step 3) is
# the only one, so every retry is visible in the console.
_clients: dict = {}


def _anthropic_client() -> anthropic.Anthropic:
    if "anthropic" not in _clients:
        _clients["anthropic"] = anthropic.Anthropic(max_retries=0, timeout=TIMEOUT_SECONDS)
    return _clients["anthropic"]


def _openai_client() -> openai.OpenAI:
    if "openai" not in _clients:
        _clients["openai"] = openai.OpenAI(max_retries=0, timeout=TIMEOUT_SECONDS)
    return _clients["openai"]


def _close_objects(schema):
    """Return a copy of schema with additionalProperties: false on every object.

    Anthropic structured outputs require it; callers shouldn't have to remember.
    """
    if isinstance(schema, list):
        return [_close_objects(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _close_objects(v) for k, v in schema.items()}
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
    return out


@with_retries()
def _call_anthropic(messages, model, schema, temperature, max_tokens):
    # Anthropic takes the system prompt as a separate argument, not as a message.
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    chat = [m for m in messages if m["role"] != "system"]

    kwargs = {"model": model, "max_tokens": max_tokens, "messages": chat}
    if system_parts:
        kwargs["system"] = "\n\n".join(system_parts)
    # temperature is deliberately not sent: the Anthropic SDK (2.x) no longer accepts it,
    # since current Claude models don't support sampling parameters. It is still honoured
    # for OpenAI. Determinism here comes from forced tool output + the schema instead.
    if schema is not None:
        # Native structured outputs. (Forced tool_choice is rejected by current Claude
        # models, so the old "force a submit_result tool" trick no longer works here.)
        kwargs["output_config"] = {
            "format": {"type": "json_schema", "schema": _close_objects(schema)},
        }

    resp = _anthropic_client().messages.create(**kwargs)

    if resp.stop_reason == "refusal":
        raise ValueError(f"Model refused: {resp.stop_details}")
    # With output_config.format, the text block is guaranteed to be valid JSON.
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


# Models that rejected a non-default temperature (e.g. reasoning models like gpt-6-astra).
# Learned on the first 400, so later calls skip the parameter without an extra round trip.
_NO_TEMPERATURE: set[str] = set()


@with_retries()
def _call_openai(messages, model, schema, temperature, max_tokens):
    kwargs = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
    if temperature is not None and model not in _NO_TEMPERATURE:
        kwargs["temperature"] = temperature
    if schema is not None:
        # Native structured outputs. (A forced function tool is rejected by reasoning models
        # on /v1/chat/completions, so the old "force a submit_result tool" trick is gone.)
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": SCHEMA_TOOL_NAME, "schema": _close_objects(schema), "strict": True},
        }

    try:
        resp = _openai_client().chat.completions.create(**kwargs)
    except openai.BadRequestError as e:
        if "temperature" not in kwargs or "temperature" not in str(e):
            raise
        _NO_TEMPERATURE.add(model)
        print(f"[llm] {model} only supports the default temperature; sending without it.")
        del kwargs["temperature"]
        resp = _openai_client().chat.completions.create(**kwargs)

    choice = resp.choices[0]
    if getattr(choice.message, "refusal", None):
        raise ValueError(f"Model refused: {choice.message.refusal}")
    if schema is not None and choice.finish_reason == "length":
        raise ValueError("Output hit max_tokens before the JSON was complete")
    text = choice.message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


def complete(
    messages: list[dict],
    tools=None,
    schema: dict | None = None,
    *,  # everything after this must be passed by name, e.g. provider="openai"
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    trace_id: str | None = None,
    label: str | None = None,
) -> LLMResponse:
    """Call the configured model. Pass schema (a JSON schema dict) to get JSON back.

    provider and model override .env for one call (used for comparisons and tiering).
    trace_id and label are written to the trace log so related calls can be grouped.
    """
    if tools is not None:
        raise NotImplementedError("Tool use arrives in Module 3.")

    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        model = model or os.environ["ANTHROPIC_MODEL"]
        call = _call_anthropic
    elif provider == "openai":
        model = model or os.environ["OPENAI_MODEL"]
        call = _call_openai
    else:
        raise ValueError(f"Unknown provider: {provider}")

    base = {"trace_id": trace_id, "label": label, "provider": provider, "model": model}
    start = time.perf_counter()
    try:
        text, input_tokens, output_tokens = call(messages, model, schema, temperature, max_tokens)
    except Exception as e:
        # Log the failure, then re-raise: the caller decides what to do.
        latency_ms = round((time.perf_counter() - start) * 1000)
        log_event({**base, "status": "error", "error": f"{type(e).__name__}: {e}", "latency_ms": latency_ms})
        raise

    latency_ms = round((time.perf_counter() - start) * 1000)
    cost = estimate_cost(model, input_tokens, output_tokens)
    log_event({
        **base,
        "status": "ok",
        "structured": schema is not None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "output_preview": text[:300],  # Module 4 adds PII redaction before logging
    })
    return LLMResponse(text, provider, model, input_tokens, output_tokens, latency_ms, cost)