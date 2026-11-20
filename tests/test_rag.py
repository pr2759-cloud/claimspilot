"""Module 2 tests. No API calls: a hashing embedder stands in for real embeddings and
complete() is monkeypatched where rerank.py / answer.py look it up."""
import hashlib
import json
import math

import httpx
import openai
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from types import SimpleNamespace
from rag import answer as answer_mod
from rag import config
from rag import rerank as rerank_mod
from rag.ingest import chunk_all, is_heading
from rag.pipeline import ask
from rag.retrieve import rrf
from rag.store import PolicyIndex, build_index, tokenize


def Resp(text):
    """Only .text is read by rag/, so a namespace stands in for LLMResponse."""
    return SimpleNamespace(text=text)


class HashEmbedder:
    """Bag-of-words hashed into 256 dims. Deterministic, offline, good enough to test plumbing."""
    name = "test:hash256"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for tok in tokenize(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def write_pdf(path, pages):
    c = canvas.Canvas(str(path), pagesize=letter)
    for lines in pages:
        y = 740
        for line in lines:
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()


@pytest.fixture
def index(tmp_path):
    pol = tmp_path / "policies"
    pol.mkdir()
    write_pdf(pol / "homeowners.pdf", [
        ["SECTION 1 DEFINITIONS", "Dwelling means the house on the residence premises."],
        ["4.2 Water Damage", "We cover sudden and accidental discharge of water from a plumbing system.",
         "We do not cover flood, surface water or groundwater seepage."],
        ["5 Deductible", "The deductible for each covered loss is 1000 dollars."],
    ])
    write_pdf(pol / "adjuster_guide.pdf", [
        ["SECTION A FRAUD INDICATORS", "Escalate claims filed within 30 days of policy inception."],
    ])
    (pol / "access.json").write_text(json.dumps({"adjuster_guide.pdf": "adjuster"}))
    chunks = chunk_all(pol)
    build_index(chunks, HashEmbedder(), tmp_path / "index")
    return PolicyIndex(HashEmbedder(), tmp_path / "index")


def fake_complete(rerank_scores=None, answer=None):
    """Return a complete() that answers rerank and answer calls by inspecting the schema."""
    def _complete(messages, schema=None, **kw):
        if "scores" in schema["properties"]:
            n = messages[1]["content"].count("<excerpt ")
            scores = rerank_scores or {}
            return Resp(text=json.dumps({"scores": [scores.get(i, 0) for i in range(n)]}))
        return Resp(text=json.dumps(answer))
    return _complete


@pytest.fixture(autouse=True)
def isolated_trace_file(tmp_path, monkeypatch):
    """Keep test runs out of the real traces.jsonl."""
    import llm.tracing
    monkeypatch.setattr(llm.tracing, "TRACE_FILE", tmp_path / "traces.jsonl")


# ---- ingestion ---------------------------------------------------------------

def test_heading_detection():
    assert is_heading("4.2 Water Damage")
    assert is_heading("SECTION 1 DEFINITIONS")
    assert is_heading("WHAT IS NOT COVERED")
    assert is_heading("COVERAGE C – PERSONAL PROPERTY")  # en dash, as in the real HO-3
    assert not is_heading("We do not cover flood, surface water or groundwater seepage.")


def test_chunks_carry_section_page_and_access(index):
    water = [c for c in index.chunks.values() if c.section == "4.2 Water Damage"]
    assert water and water[0].page == 2 and water[0].doc_id == "homeowners"
    assert water[0].text.startswith("[4.2 Water Damage]")
    guide = [c for c in index.chunks.values() if c.doc_id == "adjuster_guide"]
    assert all(c.access_level == "adjuster" for c in guide)


def test_overlap_windows():
    from rag.ingest import window
    words = [(str(i), 1) for i in range(10)]
    pieces = list(window(words, size=4, overlap=1))
    assert [p[0][0] for p in pieces] == ["0", "3", "6"]
    assert pieces[-1][-1][0] == "9"


# ---- retrieval ---------------------------------------------------------------

def test_tokenizer_keeps_section_numbers():
    assert "4.2" in tokenize("See Section 4.2 for details")


def test_rrf_rewards_agreement():
    fused = dict(rrf([["a", "b", "c"], ["b", "d"]]))
    assert max(fused, key=fused.get) == "b"  # 2nd + 1st beats 1st + absent


def test_bm25_finds_exact_section_number(index):
    hits = index.bm25_search("what does 4.2 say", ["customer"], 5)
    assert index.chunks[hits[0]].section == "4.2 Water Damage"


def test_access_filter_hides_adjuster_docs(index):
    for search in (index.bm25_search, index.vector_search):
        assert not any(index.chunks[c].doc_id == "adjuster_guide"
                       for c in search("fraud indicators escalate", ["customer"], 20))
    assert any(index.chunks[c].doc_id == "adjuster_guide"
               for c in index.bm25_search("fraud indicators", config.allowed_levels("adjuster"), 20))


def test_unknown_access_level_rejected():
    with pytest.raises(ValueError):
        config.allowed_levels("admin")


# ---- end to end with a fake model -----------------------------------------------

def test_answer_with_citations(index, monkeypatch):
    fc = fake_complete({0: 3}, {"covered": True, "answer": "Flood is not covered [1].", "source_ids": [1]})
    monkeypatch.setattr(rerank_mod, "complete", fc)
    monkeypatch.setattr(answer_mod, "complete", fc)
    r = ask("Is flood damage covered?", "customer", index)
    assert r.covered and len(r.citations) == 1
    assert r.citations[0].doc_id == "homeowners" and r.citations[0].page >= 1


def test_invented_source_id_becomes_refusal(index, monkeypatch):
    fc = fake_complete({0: 3}, {"covered": True, "answer": "Yes [7].", "source_ids": [7]})
    monkeypatch.setattr(rerank_mod, "complete", fc)
    monkeypatch.setattr(answer_mod, "complete", fc)
    r = ask("Is flood covered?", "customer", index)
    assert not r.covered and r.answer == config.NOT_COVERED and r.citations == []


def test_nothing_relevant_refuses_without_answer_call(index, monkeypatch):
    calls = []

    def fc(messages, schema=None, **kw):
        calls.append(schema)
        return fake_complete({})(messages, schema=schema)

    monkeypatch.setattr(rerank_mod, "complete", fc)
    monkeypatch.setattr(answer_mod, "complete", fc)
    r = ask("What is the capital of France?", "customer", index)
    assert r.answer == config.NOT_COVERED and len(calls) == 1  # rerank only


def test_rerank_failure_degrades_not_crashes(index, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("provider down")
    monkeypatch.setattr(rerank_mod, "complete", broken)
    monkeypatch.setattr(answer_mod, "complete",
                        fake_complete(answer={"covered": True, "answer": "x [1]", "source_ids": [1]}))
    r = ask("deductible amount", "customer", index)
    assert r.rerank_degraded and r.covered


def test_rerank_wrong_length_and_out_of_range_scores(index, monkeypatch):
    def fc(messages, schema=None, **kw):
        if "scores" in schema["properties"]:
            return Resp(text=json.dumps({"scores": [9, -2]}))  # too short, out of range
        return Resp(text=json.dumps({"covered": True, "answer": "x [1]", "source_ids": [1]}))
    monkeypatch.setattr(rerank_mod, "complete", fc)
    monkeypatch.setattr(answer_mod, "complete", fc)
    r = ask("deductible", "customer", index)
    assert len(r.used) == 1 and not r.rerank_degraded  # 9 clamped to 3; -2 and missing -> 0


def test_invalid_answer_retried_once_then_error(index, monkeypatch):
    monkeypatch.setattr(rerank_mod, "complete", fake_complete({0: 3}))
    calls = []

    def bad(messages, schema=None, **kw):
        calls.append(1)
        return Resp(text="not json")
    monkeypatch.setattr(answer_mod, "complete", bad)
    with pytest.raises(answer_mod.AnswerError):
        ask("deductible", "customer", index)
    assert len(calls) == 2


# ---- API ---------------------------------------------------------------------

def test_api_unknown_user_is_403(index, monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import ask as ask_api

    users = tmp_path / "users.json"
    users.write_text(json.dumps([{"user_id": "u1", "access_level": "customer"}]))
    monkeypatch.setattr(config, "USERS_FILE", users)
    ask_api.load_users.cache_clear()
    monkeypatch.setattr(ask_api, "get_index", lambda: index)
    fc = fake_complete({0: 3}, {"covered": True, "answer": "1000 dollars [1].", "source_ids": [1]})
    monkeypatch.setattr(rerank_mod, "complete", fc)
    monkeypatch.setattr(answer_mod, "complete", fc)

    app = FastAPI()
    app.include_router(ask_api.router)
    client = TestClient(app)
    assert client.post("/ask", json={"question": "deductible?", "user_id": "nobody"}).status_code == 403
    assert client.post("/ask", json={"question": "x", "user_id": "u1"}).status_code == 422
    r = client.post("/ask", json={"question": "What is the deductible?", "user_id": "u1"})
    assert r.status_code == 200 and r.json()["citations"]

    def down(*a, **k):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://x"))
    monkeypatch.setattr(answer_mod, "complete", down)
    assert client.post("/ask", json={"question": "What is the deductible?", "user_id": "u1"}).status_code == 503
