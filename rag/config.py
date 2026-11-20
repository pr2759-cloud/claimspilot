"""Module 2 settings. Paths and knobs in one place so the pipeline code stays readable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = ROOT / "data" / "policies"
ACCESS_MANIFEST = POLICIES_DIR / "access.json"   # optional: {"file.pdf": "agent", ...}
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "data" / "index"))
USERS_FILE = ROOT / "data" / "users.json"

# Chunking: ~500 tokens with 50 overlap. Tokens estimated as words / 0.75,
# which is close enough for English prose and avoids a tokenizer dependency.
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50
WORDS_PER_TOKEN = 0.75

# Retrieval
K_VECTOR = 20        # dense candidates
K_BM25 = 20          # keyword candidates
RRF_K = 60           # standard constant from the RRF paper
RERANK_TOP_K = 5     # chunks that reach the answer prompt
RERANK_MIN_SCORE = 1 # 0 = irrelevant; below this a chunk is dropped

COLLECTION = "policies"
NOT_COVERED = "This is not covered in these documents."

# Access levels, lowest to highest. A caller sees documents at or below their level.
ACCESS_LEVELS = ["customer", "agent", "adjuster"]


def allowed_levels(access_level: str) -> list[str]:
    if access_level not in ACCESS_LEVELS:
        raise ValueError(f"unknown access_level: {access_level!r}")
    return ACCESS_LEVELS[: ACCESS_LEVELS.index(access_level) + 1]
