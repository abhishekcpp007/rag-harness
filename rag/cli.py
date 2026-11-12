"""Command line wrapper. `rag ingest`, `rag search`, `rag ask`, `rag serve`."""
from __future__ import annotations

import argparse

from .answer import answer_question
from .config import SETTINGS
from .embedding import get_embedder
from .ingest import ingest
from .llm import get_llm
from .retrieve import Retriever
from .store import Store


def _retriever(s):
    store = Store(s.db_path)
    return Retriever(store=store, embedder=get_embedder(s), settings=s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest")

    ps = sub.add_parser("search")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=5)

    pa = sub.add_parser("ask")
    pa.add_argument("question")

    pv = sub.add_parser("serve")
    pv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    s = SETTINGS

    if args.cmd == "ingest":
        ingest(s, verbose=True)
        return 0

    if args.cmd == "search":
        for h in _retriever(s).search(args.query, args.k):
            print(f"{h.score:.5f}  {h.chunk_id:<34} {h.heading_path}")
            print(f"        {h.text[:150].splitlines()[0] if h.text else ''}")
        return 0

    if args.cmd == "ask":
        a = answer_question(args.question, _retriever(s), get_llm(s.llm), s)
        print(a.text)
        if a.citations:
            print()
            for c in a.citations:
                print(f"  [{c['n']}] {c['doc_id']} :: {c['heading_path']}")
        return 0

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("rag.api:app", host="0.0.0.0", port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
