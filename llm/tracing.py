"""Append one JSON line per event to traces.jsonl."""
import json
import os
import time
import uuid
from pathlib import Path

TRACE_FILE = Path(os.getenv("TRACE_FILE", "traces.jsonl"))


def new_trace_id() -> str:
    """A short random id that groups all the calls made for one request."""
    return uuid.uuid4().hex[:12]


def log_event(event: dict) -> None:
    """Write one event as one line of JSON.

    Logging must never crash the app, so errors are printed, not raised.
    """
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    try:
        with TRACE_FILE.open("a", encoding="utf-8") as f:  # "a" = append
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        print(f"[tracing] could not write trace: {e}")
