"""Parse + chunk every PDF in data/policies/, embed, and (re)build the index.

    python -m scripts.build_index                 # build
    python -m scripts.build_index --show-sections # inspect detected sections, no embedding
"""
import argparse
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

from rag.ingest import chunk_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-sections", action="store_true")
    args = ap.parse_args()

    chunks = chunk_all()
    per_doc = Counter(c.doc_id for c in chunks)
    print(f"{len(chunks)} chunks from {len(per_doc)} documents")
    for doc, n in per_doc.items():
        level = next(c.access_level for c in chunks if c.doc_id == doc)
        print(f"  {doc}: {n} chunks, access_level={level}")

    if args.show_sections:
        seen = set()
        for c in chunks:
            if (c.doc_id, c.section) not in seen:
                seen.add((c.doc_id, c.section))
                print(f"  {c.doc_id} p.{c.page}: {c.section}")
        return

    from rag.embeddings import get_embedder
    from rag.store import build_index

    emb = get_embedder()
    print(f"embedding with {emb.name} ...")
    build_index(chunks, emb)
    print("index built")


if __name__ == "__main__":
    main()
