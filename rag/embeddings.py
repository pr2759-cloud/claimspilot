"""Embedding backends. Anthropic has no embeddings API, so the choice is OpenAI
(text-embedding-3-small, hosted) or sentence-transformers (local, nothing leaves the
machine - the answer to "where do embeddings live?" for a regulated client).

Set EMBEDDING_PROVIDER=openai|local in .env. Index and query MUST use the same
embedder; the index records which one built it and refuses a mismatch.
"""
import os
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # stay well under request limits
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + 100])
            out.extend(d.embedding for d in resp.data)
        return out


class LocalEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.name = f"local:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder() -> Embedder:
    provider = os.getenv("EMBEDDING_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "local")
    if provider == "openai":
        return OpenAIEmbedder(os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    if provider == "local":
        return LocalEmbedder(os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    raise ValueError(f"EMBEDDING_PROVIDER must be openai or local, got {provider!r}")
