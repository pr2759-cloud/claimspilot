"""Parse -> section-aware chunks.

Pipeline position: ingest -> PARSE -> CHUNK -> embed -> index -> retrieve -> rerank -> generate.

Each chunk remembers doc_id, section, page (where it starts) and access_level, because
the answer must cite [doc, section, page] and retrieval must filter by access level.
"""
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from rag import config

# pypdf warns loudly about fonts it can only partly decode; text still extracts.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Heading heuristics. Real policy PDFs are messy, so this is deliberately simple and
# tunable: print the sections ingest finds (scripts/build_index.py --show-sections)
# and adjust these patterns to your documents.
HEADING_PATTERNS = [
    re.compile(r"^(section|part|article|schedule|endorsement)\s+[\w.\-]+\b.{0,80}$", re.I),
    re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][^.]{2,80}$"),          # "4.2 Exclusions"
    re.compile(r"^[A-Z][A-Z0-9 ,&'/\-–—]{3,60}$"),               # "WHAT IS NOT COVERED", "COVERAGE C – PERSONAL PROPERTY"
]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    page: int          # 1-based page where the chunk starts
    page_end: int
    access_level: str
    text: str          # chunk body, prefixed with its section title

    def to_dict(self) -> dict:
        return asdict(self)


def is_heading(line: str) -> bool:
    line = line.strip()
    if len(line) < 4 or len(line) > 90 or line.endswith((",", ";", ":")):  # ":" = list lead-in
        return False
    if line.endswith((".", "-")) or re.search(r"\b(or|and)$", line):
        return False  # sentence end, hyphenated line break, or list item lead-in
    if re.match(r"^\d", line) and len(line.split()) > 7:
        return False  # numbered heading = number + at most 6 words; longer is a list item
    return any(p.match(line) for p in HEADING_PATTERNS)


def load_access_manifest(policies_dir: Path) -> dict[str, str]:
    path = policies_dir / "access.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text())
    for name, level in manifest.items():
        config.allowed_levels(level)  # raises on typos like "adjustor"
    return manifest


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def split_sections(pages: list[tuple[int, str]]) -> list[tuple[str, list[tuple[str, int]]]]:
    """Return [(section_title, [(word, page), ...]), ...] in document order."""
    sections: list[tuple[str, list[tuple[str, int]]]] = []
    title, words = "Preamble", []
    for page_no, text in pages:
        for line in text.splitlines():
            if is_heading(line):
                if words:
                    sections.append((title, words))
                title, words = line.strip(), []
            else:
                words.extend((w, page_no) for w in line.split())
    if words:
        sections.append((title, words))
    return sections


def window(words: list[tuple[str, int]], size: int, overlap: int):
    """Sliding windows over words. Overlap keeps a sentence that straddles a boundary
    retrievable from either side."""
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if piece:
            yield piece
        if start + size >= len(words):
            break


def chunk_document(pdf_path: Path, access_level: str = "customer") -> list[Chunk]:
    doc_id = pdf_path.stem
    size = int(config.CHUNK_TOKENS * config.WORDS_PER_TOKEN)
    overlap = int(config.CHUNK_OVERLAP * config.WORDS_PER_TOKEN)
    chunks: list[Chunk] = []
    for title, words in split_sections(extract_pages(pdf_path)):
        for piece in window(words, size, overlap):
            body = " ".join(w for w, _ in piece)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{len(chunks):04d}",
                    doc_id=doc_id,
                    section=title,
                    page=piece[0][1],
                    page_end=piece[-1][1],
                    access_level=access_level,
                    # Section title in the text helps both BM25 and embeddings match
                    # questions like "what does Section 4 exclude".
                    text=f"[{title}] {body}",
                )
            )
    return chunks


def chunk_all(policies_dir: Path = config.POLICIES_DIR) -> list[Chunk]:
    manifest = load_access_manifest(policies_dir)
    pdfs = sorted(policies_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {policies_dir}")
    out: list[Chunk] = []
    for pdf in pdfs:
        out.extend(chunk_document(pdf, manifest.get(pdf.name, "customer")))
    return out
