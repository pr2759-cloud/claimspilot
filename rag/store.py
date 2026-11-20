"""Index: Chroma for dense vectors, rank_bm25 for keywords, one chunks.jsonl as the
source of truth for chunk text and metadata.

Both searches apply the access filter BEFORE ranking, so a customer's query never
even scores an adjuster-only chunk (filtering after top-k would also starve results).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from rag import config
from rag.ingest import Chunk

# Keeps "4.2" and "a-12" as single tokens so "Section 4.2" matches exactly.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_index(chunks: list[Chunk], embedder, index_dir: Path = config.INDEX_DIR) -> int:
    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        client.delete_collection(config.COLLECTION)  # rebuild from scratch: simple and reproducible
    except Exception:
        pass
    col = client.create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine", "embedder": embedder.name}
    )
    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        col.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embedder.embed([c.text for c in batch]),
            metadatas=[
                {"doc_id": c.doc_id, "section": c.section, "page": c.page, "access_level": c.access_level}
                for c in batch
            ],
        )
    with open(index_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    return len(chunks)


class PolicyIndex:
    def __init__(self, embedder, index_dir: Path = config.INDEX_DIR):
        chunks_file = index_dir / "chunks.jsonl"
        if not chunks_file.exists():
            raise FileNotFoundError(f"no index at {index_dir}; run scripts/build_index.py")
        self.embedder = embedder
        self.chunks: dict[str, Chunk] = {}
        for line in chunks_file.read_text().splitlines():
            c = Chunk(**json.loads(line))
            self.chunks[c.chunk_id] = c
        client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        self.col = client.get_collection(config.COLLECTION)
        built_with = (self.col.metadata or {}).get("embedder")
        if built_with != embedder.name:
            raise RuntimeError(
                f"index built with {built_with}, querying with {embedder.name}; rebuild the index"
            )

    def vector_search(self, query: str, levels: list[str], k: int) -> list[str]:
        res = self.col.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=k,
            where={"access_level": {"$in": levels}},
        )
        return res["ids"][0]

    @lru_cache(maxsize=8)
    def _bm25(self, levels: tuple[str, ...]):
        ids = [cid for cid, c in self.chunks.items() if c.access_level in levels]
        return ids, (BM25Okapi([tokenize(self.chunks[i].text) for i in ids]) if ids else None)

    def bm25_search(self, query: str, levels: list[str], k: int) -> list[str]:
        ids, bm25 = self._bm25(tuple(sorted(levels)))
        if bm25 is None:
            return []
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, s in ranked[:k] if s > 0]  # zero score = no shared terms
