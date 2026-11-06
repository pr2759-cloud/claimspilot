"""Claim triage: free-text claim description in, validated structured result out."""
import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from llm.client import complete
from llm.tracing import log_event, new_trace_id

Category = Literal[
    "auto_collision", "auto_theft", "property_damage", "water_damage",
    "fire", "theft_burglary", "liability", "other",
]
Severity = Literal["low", "medium", "high", "critical"]


class TriageResult(BaseModel):
    """The contract: what a valid triage answer looks like."""
    category: Category = Field(description="The single best-fitting claim category.")
    severity: Severity = Field(description="How urgent and serious the claim is.")
    missing_info: list[str] = Field(
        description="Facts an adjuster needs that the description does not give. Empty list if none."
    )
    summary: str = Field(
        min_length=20, max_length=400,
        description="One or two neutral sentences for the adjuster.",
    )


class TriageError(Exception):
    """Raised when the model can't produce a valid result after one retry."""


SYSTEM_PROMPT = """You are a claims intake assistant at an insurance company.
Read the customer's claim description and triage it for a human adjuster.

Rules:
- Use only facts stated in the description. Never invent amounts, dates or names.
- severity: low = minor damage, no injuries, likely under $1,000.
  medium = moderate damage, no injuries.
  high = major damage, a home or vehicle that can't be used, or any injury.
  critical = serious injury, death, or ongoing danger.
- missing_info: short phrases for each fact an adjuster needs that is not stated,
  such as date of loss, location, police report number, photos, estimated cost.
- summary: one or two neutral sentences. Never promise coverage or payment.
- If the text is not an insurance claim, use category "other" and say so in the summary."""

# One worked example (few-shot). It shows the format and a subtle category choice.
EXAMPLE_CLAIM = (
    "Someone broke my car window last night in the office parking lot "
    "and took my laptop bag. I filed a police report."
)
EXAMPLE_ANSWER = {
    "category": "theft_burglary",
    "severity": "medium",
    "missing_info": [
        "police report number", "value of the laptop and bag",
        "window repair estimate", "address of the parking lot",
    ],
    "summary": "The customer's car window was broken overnight in an office parking lot "
               "and a laptop bag was stolen. A police report was filed.",
}


def triage_claim(claim_text: str, *, provider: str | None = None, model: str | None = None) -> TriageResult:
    """Return a validated TriageResult, or raise TriageError after two failed attempts."""
    trace_id = new_trace_id()  # groups both attempts in the trace log
    schema = TriageResult.model_json_schema()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_CLAIM},
        {"role": "assistant", "content": json.dumps(EXAMPLE_ANSWER)},
        {"role": "user", "content": claim_text},
    ]

    last_error = None
    for attempt in (1, 2):
        resp = complete(
            messages, schema=schema, provider=provider, model=model,
            temperature=0, trace_id=trace_id, label=f"triage_attempt_{attempt}",
        )
        try:
            result = TriageResult.model_validate_json(resp.text)
            log_event({"trace_id": trace_id, "label": "triage_result", "status": "valid", "attempts": attempt})
            return result
        except ValidationError as e:
            last_error = e
            log_event({"trace_id": trace_id, "label": "triage_result", "status": "invalid",
                       "attempt": attempt, "error": str(e)[:500]})
            # Feed the bad output and the exact error back, so the model can fix it.
            messages = messages + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": f"Your output failed validation:\n{e}\n"
                                             "Return a corrected result that fixes these errors."},
            ]

    raise TriageError(f"Triage failed validation twice: {last_error}")
