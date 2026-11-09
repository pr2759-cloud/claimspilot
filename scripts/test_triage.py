import json

import pytest

from api import triage
from llm.client import LLMResponse

VALID = json.dumps({
    "category": "water_damage",
    "severity": "high",
    "missing_info": ["date of loss"],
    "summary": "A burst pipe flooded the kitchen and the ceiling collapsed.",
})
INVALID = json.dumps({"category": "car stuff", "severity": "huge", "summary": "short"})


def install_fake_model(monkeypatch, outputs):
    """Replace the real model with one that returns `outputs` in order."""
    calls = []

    def fake_complete(messages, **kwargs):
        calls.append(messages)
        text = outputs[len(calls) - 1]
        return LLMResponse(text=text, provider="fake", model="fake",
                           input_tokens=0, output_tokens=0, latency_ms=0)

    monkeypatch.setattr(triage, "complete", fake_complete)
    monkeypatch.setattr(triage, "log_event", lambda event: None)  # keep test runs out of traces.jsonl
    return calls


def test_valid_on_first_try(monkeypatch):
    calls = install_fake_model(monkeypatch, [VALID])
    result = triage.triage_claim("A pipe burst and flooded my kitchen.")
    assert result.category == "water_damage"
    assert len(calls) == 1


def test_retries_once_with_error_fed_back(monkeypatch):
    calls = install_fake_model(monkeypatch, [INVALID, VALID])
    result = triage.triage_claim("A pipe burst and flooded my kitchen.")
    assert result.severity == "high"
    assert len(calls) == 2
    assert "failed validation" in calls[1][-1]["content"]  # the error was sent back


def test_fails_loudly_after_two_bad_outputs(monkeypatch):
    calls = install_fake_model(monkeypatch, [INVALID, INVALID])
    with pytest.raises(triage.TriageError):
        triage.triage_claim("A pipe burst and flooded my kitchen.")
    assert len(calls) == 2  # exactly one retry, never more