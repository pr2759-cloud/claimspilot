"""Summarize traces.jsonl: calls, errors, average latency, tokens and cost per model."""
import json
from collections import defaultdict
from pathlib import Path

stats = defaultdict(lambda: {"calls": 0, "errors": 0, "latency": 0, "in": 0, "out": 0, "cost": 0.0})

for line in Path("traces.jsonl").read_text().splitlines():
    event = json.loads(line)
    if "model" not in event:  # skip triage_result events
        continue
    s = stats[(event["provider"], event["model"])]
    s["calls"] += 1
    if event["status"] == "error":
        s["errors"] += 1
        continue
    s["latency"] += event["latency_ms"]
    s["in"] += event["input_tokens"]
    s["out"] += event["output_tokens"]
    s["cost"] += event["cost_usd"] or 0

print(f"{'provider/model':40} {'calls':>5} {'errors':>6} {'avg ms':>7} {'in tok':>7} {'out tok':>7} {'cost $':>8}")
for (provider, model), s in stats.items():
    ok = s["calls"] - s["errors"]
    avg = s["latency"] / ok if ok else 0
    print(f"{provider + '/' + model:40} {s['calls']:5} {s['errors']:6} {avg:7.0f} {s['in']:7} {s['out']:7} {s['cost']:8.4f}")