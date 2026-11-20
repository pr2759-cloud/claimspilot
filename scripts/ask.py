"""Ask questions from the terminal and log them for Module 5's test set.

    python -m scripts.ask "Is flood damage covered?" --level customer
    python -m scripts.ask --file notes/questions.txt --level adjuster

Every run appends to notes/rag_questions.jsonl: question, level, answer, citations,
retrieved chunk ids. Add an "expected" field by hand later and it becomes an eval case.
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag.embeddings import get_embedder  # noqa: E402
from rag.pipeline import ask  # noqa: E402
from rag.store import PolicyIndex  # noqa: E402

LOG = Path(__file__).resolve().parent.parent / "notes" / "rag_questions.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--file", help="one question per line")
    ap.add_argument("--level", default="customer", choices=["customer", "agent", "adjuster"])
    args = ap.parse_args()

    questions = [args.question] if args.question else []
    if args.file:
        questions += [q.strip() for q in Path(args.file).read_text().splitlines() if q.strip()]
    if not questions:
        ap.error("give a question or --file")

    index = PolicyIndex(get_embedder())
    LOG.parent.mkdir(exist_ok=True)
    for q in questions:
        r = ask(q, args.level, index)
        print(f"\nQ: {q}\nA: {r.answer}")
        for c in r.citations:
            print(f"   [{c.n}] {c.label()}")
        print(f"   covered={r.covered} degraded={r.rerank_degraded} timings={r.timings_ms}")
        with open(LOG, "a") as f:
            f.write(json.dumps({"question": q, "access_level": args.level, **r.model_dump()}) + "\n")


if __name__ == "__main__":
    main()
