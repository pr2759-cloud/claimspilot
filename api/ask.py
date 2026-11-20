"""POST /ask - policy Q&A with citations.

Identity: for now the caller passes user_id and we look up access_level in
data/users.json. In production the user comes from the auth token, never the body;
say exactly that in the interview.
"""
import json
from functools import lru_cache

import anthropic
import openai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from rag import config
from rag.answer import AnswerError
from rag.pipeline import AskResult, ask

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    user_id: str = Field(min_length=1, max_length=100)


@lru_cache(maxsize=1)
def load_users() -> dict[str, str]:
    """user_id -> access_level. Accepts a list of user objects or a dict keyed by id."""
    data = json.loads(config.USERS_FILE.read_text())
    users = data.values() if isinstance(data, dict) else data
    return {u["user_id"]: u["access_level"] for u in users}


@lru_cache(maxsize=1)
def get_index():
    from rag.embeddings import get_embedder
    from rag.store import PolicyIndex

    return PolicyIndex(get_embedder())


@router.post("/ask", response_model=AskResult)
def ask_endpoint(req: AskRequest) -> AskResult:
    try:
        access_level = load_users().get(req.user_id)
    except FileNotFoundError:
        raise HTTPException(503, "users.json missing")
    if access_level is None:
        raise HTTPException(403, "unknown user")
    try:
        index = get_index()
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(503, f"policy index unavailable: {e}")
    try:
        return ask(req.question, access_level, index)
    except AnswerError as e:
        raise HTTPException(502, f"Model returned invalid output: {e}")
    except (anthropic.APIError, openai.APIError) as e:  # same mapping as /triage
        raise HTTPException(503, f"Model provider unavailable: {type(e).__name__}")
