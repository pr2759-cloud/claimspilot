# Module 2: RAG (`POST /ask`)

## Status and next steps

Already done on branch `feature/module-2-rag` (created from `feature/module-1-triage`, because Module 1 isn't merged into `main` yet):
- `rag/`, `api/ask.py`, `scripts/build_index.py`, `scripts/ask.py`, `tests/test_rag.py` added; `POST /ask` wired into `api/main.py`
- Built against the real `complete()` (native structured output, `trace_id`/`label`) and `log_event()`; provider errors map to 503, as `/triage` does
- `data/users.json` rebuilt so its ids match `claims.json` (`u_001`-`u_006` customers, plus `u_101` agent and `u_201` adjuster)
- `data/index/` added to `.gitignore`

Still to do:
1. `.env`: add `EMBEDDING_PROVIDER=openai` if the OpenAI key works, otherwise `local` (then `pip install sentence-transformers`).
2. Optional `data/policies/access.json`, e.g. `{"Condominium_Unit_Owners.pdf": "agent"}`, to see the filter in action. Unlisted PDFs default to `customer`.
3. `python -m scripts.build_index --show-sections`, then check that the sections look like the real headings. If they don't, tune `HEADING_PATTERNS` in `rag/ingest.py`.
4. `python -m scripts.build_index`, then `pytest`.
5. `python -m scripts.ask --file notes/questions.txt --level customer` with 10 questions (6 answerable, 2 exact-section, 2 not covered), then again with `--level adjuster`. Results go to `notes/rag_questions.jsonl` for Module 5.
6. `uvicorn api.main:app --reload` and
   `curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"Is flood damage covered?","user_id":"u_001"}'`
7. Merge Module 1 into `main` first (PR), then open the PR for this branch.

## Request flow

```
api/ask.py        validate body (3-1000 chars) → user_id → access_level (users.json; 403 if unknown)
 → rag/pipeline.py  ask(): times each stage, writes one audit trace
   → retrieve.py      allowed levels → vector top 20 (Chroma, where-filter) + BM25 top 20 → RRF
   → rerank.py        one LLM call scores all 20 (0-3) → keep score ≥1, top 5; on failure use RRF order
   → answer.py        numbered excerpts → structured {covered, answer, source_ids}
                      → code checks ids → builds [doc, section, page] citations, or refuses
```

## Design decisions

- **Sections first, then token windows.** Chunks never cross a section heading, so the citation's section is always correct. Long sections are split into ~500-token windows with 50 overlap. Each chunk is prefixed with its section title, so a question like "what does Section 4.2 exclude" matches on both BM25 and embeddings.
- **Hybrid with RRF.** Dense search catches paraphrase ("water damage" vs "flooding"). BM25 catches exact tokens ("4.2", "HO-3", amounts). The tokenizer keeps `4.2` as one token. RRF fuses by rank only, so the two score scales never need calibrating.
- **Access filter before ranking, not after.** Chroma's `where` filter and a BM25 index per access level mean restricted chunks are never scored. Filtering after top-k could also leave a customer with zero results.
- **Rerank: recall then precision.** Retrieval is cheap and casts a wide net. The reranker reads the question and each chunk together, which embeddings never do. It's one batched call. A chunk scored 0 is dropped, and if nothing survives, the system refuses **without** calling the answer model.
- **Graceful degradation.** If the reranker fails, fall back to RRF order with `rerank_degraded=true`. If the answer model fails, retry once with the error, then return 502.
- **Citations are built by code, not written by the model.** The model returns excerpt numbers. Code drops numbers that weren't provided and builds citations from metadata. "Covered" with no valid citation becomes a refusal. So a citation can't be invented, and an ungrounded answer can't get through.
- **Embedder pinned to the index.** The index stores which embedder built it, and a mismatch fails loudly (rather than returning quietly wrong results).
- **Retrieved text is data.** Both prompts say so, and excerpts are wrapped in tags. Module 4 hardens this.
- **The audit trace logs chunk ids, scores and citations, but not the question** (it may contain PII; Module 4 adds redaction).

## Talking points

11. "Hybrid retrieval because dense vectors miss exact identifiers like 'Section 4.2'. BM25 catches them, and RRF merges the two without calibrating scores."
12. "I enforce access control in the retrieval query, not on the output. A customer's search never scores an adjuster-only chunk."
13. "The model never writes citations. It points at excerpt numbers, code checks them, and builds `[doc, section, page]` from metadata. An answer with no valid citation is converted to 'not covered'."
14. "Retrieve wide for recall, rerank narrow for precision. If the reranker finds nothing relevant, we refuse before paying for generation."
15. "Three RAG failure modes, each with a control: wrong chunk retrieved (hybrid + rerank, measured by recall@5 in Module 5), right chunk ignored (faithfulness judge), invented answer (structured citations + refusal)."
16. **Regulated client:** local embeddings via sentence-transformers keep the data on-premises. Chroma can be swapped for pgvector inside their VPC. Access level lives in chunk metadata. Every answer is traced to the chunk ids it used. Re-indexing is a full rebuild for now; at scale it becomes incremental by document hash.

## Known limits (good to raise yourself in the interview)

- Heading detection is a heuristic; scanned PDFs would need OCR, and tables lose their structure.
- The BM25 index is rebuilt in memory at start-up. That's fine for 5 PDFs, but at 50k documents you'd use a search engine (OpenSearch, or Postgres full-text search).
- One LLM rerank call adds latency (likely the biggest share; check `timings_ms`). A cross-encoder (e.g. `bge-reranker`) would be faster and cheaper.
- Identity comes from the request body. In production it comes from the auth token.
