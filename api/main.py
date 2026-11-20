import os
from fastapi import FastAPI
import anthropic
import openai
from fastapi import HTTPException
from pydantic import BaseModel, Field

from api.ask import router as ask_router
from api.triage import TriageError, TriageResult, triage_claim

app = FastAPI(title="ClaimsPilot")
app.include_router(ask_router)  # POST /ask (Module 2)

@app.get("/health")
def health():
    return {"status": "ok", "provider": os.getenv("LLM_PROVIDER", "anthropic")}

class TriageRequest(BaseModel):
    # max_length caps cost: a huge input would mean a huge bill.
    claim_text: str = Field(min_length=10, max_length=5000)


@app.post("/triage", response_model=TriageResult)
def triage(req: TriageRequest) -> TriageResult:
    try:
        return triage_claim(req.claim_text)
    except TriageError as e:
        raise HTTPException(status_code=502, detail=f"Model returned invalid output: {e}")
    except (anthropic.APIError, openai.APIError) as e:
        raise HTTPException(status_code=503, detail=f"Model provider unavailable: {type(e).__name__}")