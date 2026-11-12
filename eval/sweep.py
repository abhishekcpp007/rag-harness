"""Write the abstain threshold curve for both embedding backends."""
from __future__ import annotations

import json
import os

from eval.run_eval import load_golden, sweep_abstain
from rag.config import SETTINGS

HERE = os.path.dirname(__file__)

if __name__ == "__main__":
    out = {}
    for emb in ("hash", "fastembed"):
        s = SETTINGS.with_(embedder=emb, db_path=os.path.join("data", f"sweep-{emb}.sqlite3"))
        out[emb] = sweep_abstain(s, load_golden())
        c = out[emb]["chosen"]
        print(f"{emb:<10} threshold {c['threshold']} under a 12.5% wrong-refusal budget: "
              f"catches {c['abstained_on_unanswerable']:.0%} of unanswerable, "
              f"wrongly refuses {c['abstained_on_answerable']:.0%}")
    path = os.path.join(HERE, "results", "abstain-sweep.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"-> {path}")
