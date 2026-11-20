"""ask(): retrieve -> rerank -> answer, with timings and an audit trace per stage."""
import time

from pydantic import BaseModel

from rag import config
from rag.answer import Citation, generate_answer
from rag.rerank import llm_rerank
from rag.retrieve import hybrid_retrieve
from llm.tracing import log_event, new_trace_id


class AskResult(BaseModel):
    answer: str
    covered: bool
    citations: list[Citation]
    rerank_degraded: bool = False
    retrieved: list[str] = []          # chunk ids after RRF (debug / eval recall@k)
    used: list[str] = []               # chunk ids sent to the answer prompt
    timings_ms: dict[str, float] = {}
    trace_id: str = ""


def ask(question: str, access_level: str, index) -> AskResult:
    trace_id = new_trace_id()  # groups retrieval, rerank and answer lines in traces.jsonl
    t0 = time.perf_counter()
    fused = hybrid_retrieve(index, question, access_level)
    candidates = [index.chunks[cid] for cid, _ in fused]
    t1 = time.perf_counter()

    kept, scores, degraded = llm_rerank(question, candidates, trace_id=trace_id)
    t2 = time.perf_counter()

    answer, covered, citations = generate_answer(question, kept, trace_id=trace_id)
    t3 = time.perf_counter()

    result = AskResult(
        answer=answer,
        covered=covered,
        citations=citations,
        rerank_degraded=degraded,
        retrieved=[c.chunk_id for c in candidates],
        used=[c.chunk_id for c in kept],
        timings_ms={
            "retrieve": round((t1 - t0) * 1000, 1),
            "rerank": round((t2 - t1) * 1000, 1),
            "answer": round((t3 - t2) * 1000, 1),
        },
        trace_id=trace_id,
    )
    # The question itself is not logged here: Module 4 adds PII redaction first.
    log_event({
        "trace_id": trace_id,
        "label": "rag.ask",
        "status": "ok",
        "access_level": access_level,
        "retrieved": result.retrieved,
        "rerank_scores": scores,
        "used": result.used,
        "cited": [c.chunk_id for c in citations],
        "covered": covered,
        "rerank_degraded": degraded,
        "timings_ms": result.timings_ms,
    })
    return result
