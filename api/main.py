import os
from fastapi import FastAPI

app = FastAPI(title="ClaimsPilot")

@app.get("/health")
def health():
    return {"status": "ok", "provider": os.getenv("LLM_PROVIDER", "anthropic")}
