"""Hybrid retrieval: dense top 20 + BM25 top 20, merged with reciprocal rank fusion.

Why hybrid: embeddings capture meaning ("water damage" ~ "flooding") but blur exact
tokens ("Section 4.2", "HO-3", a deductible amount). BM25 is the opposite. RRF merges
the two using ranks only, so their incomparable score scales never need calibrating.
"""
from rag import config


def rrf(rankings: list[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """score(d) = sum over rankings of 1 / (k + rank). A doc ranked well by both wins."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(index, query: str, access_level: str, top_n: int = 20) -> list[tuple[str, float]]:
    levels = config.allowed_levels(access_level)
    dense = index.vector_search(query, levels, config.K_VECTOR)
    sparse = index.bm25_search(query, levels, config.K_BM25)
    return rrf([dense, sparse])[:top_n]
