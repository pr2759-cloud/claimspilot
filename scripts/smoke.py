from llm.client import complete

msgs = [
    {"role": "system", "content": "You are a concise insurance assistant."},
    {"role": "user", "content": "In one sentence, what is a deductible?"},
]
for p in ["anthropic", "openai"]:
    try:
        r = complete(msgs, provider=p)
        print(f"[{p}] {r.model} | {r.input_tokens}->{r.output_tokens} tok | {r.latency_ms} ms\n{r.text}\n")
    except Exception as e:
        print(f"[{p}] failed: {e}\n")
