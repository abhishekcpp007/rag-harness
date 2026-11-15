"""Which differences in the README tables survive a bootstrap, and which were noise all along.

Reads the committed result files, lines their rows up by question id, and answers two questions.
How wide is the interval on each headline number, and does each configuration's gap against the
default survive once the question-to-question variation is accounted for.

Run:
    python -m eval.significance            # tables to stdout, JSON to eval/results/
"""
from __future__ import annotations

import json
import os

from eval.stats import ALPHA, RESAMPLES, holm, mean_ci, paired_delta
from eval.table import LABEL, ORDER, load

HERE = os.path.dirname(__file__)
# Metrics worth an interval, and the population each one averages over. "answerable" is the 35
# questions with an answer in the corpus, "unanswerable" the 7 without.
METRICS = [
    ("chunk_mrr", "answerable"),
    ("chunk_recall@1", "answerable"),
    ("mrr", "answerable"),
    ("citation_support", "answerable"),
    ("answer_ok", "answerable"),
    ("abstained", "unanswerable"),
]
HEADLINE = "chunk_mrr"
KINDS = ["identifier", "semantic", "procedural", "factual"]
# Comparisons that are not against the default and were decided before looking at the numbers,
# so they are one test each rather than a family and carry no Holm adjustment.
PAIRS = [
    ("hash-hybrid", "hash-only", "does BM25 fusion rescue a weak encoder"),
    ("cross-lex", "cross-rerank", "does rank fusion beat the cross encoder on its own"),
]


def rows_for(result: dict, population: str) -> list[dict]:
    rows = result["rows"]
    if population == "unanswerable":
        return [r for r in rows if r["kind"] == "unanswerable"]
    return [r for r in rows if r["kind"] != "unanswerable"]


def series(result: dict, metric: str, population: str) -> tuple[list[str], list]:
    rows = rows_for(result, population)
    rows.sort(key=lambda r: r["id"])
    return [r["id"] for r in rows], [r.get(metric) for r in rows]


def aligned(a: dict, b: dict, metric: str, population: str) -> tuple[list, list]:
    ids_a, va = series(a, metric, population)
    ids_b, vb = series(b, metric, population)
    if ids_a != ids_b:
        raise ValueError(
            f"the two runs answered different question sets, so a paired comparison would "
            f"pair the wrong questions: {set(ids_a) ^ set(ids_b)}"
        )
    return va, vb


def main() -> None:
    available = [n for n in ORDER
                 if os.path.exists(os.path.join(HERE, "results", f"{n}.json"))]
    results = {n: load(n) for n in available}
    out: dict = {"resamples": RESAMPLES, "alpha": ALPHA, "intervals": {}, "vs_default": {}}

    print(f"Percentile bootstrap, {RESAMPLES} resamples, {int((1 - ALPHA) * 100)} percent "
          f"intervals.\n")
    print("| run | " + " | ".join(m for m, _ in METRICS) + " |")
    print("|---" * (len(METRICS) + 1) + "|")
    for name in available:
        cells = []
        out["intervals"][name] = {}
        for metric, population in METRICS:
            _, values = series(results[name], metric, population)
            ci = mean_ci(values)
            out["intervals"][name][metric] = ci
            cells.append(f"{ci['mean']:.3f} [{ci['lo']:.2f}, {ci['hi']:.2f}]")
        print(f"| {LABEL.get(name, name)} | " + " | ".join(cells) + " |")

    base = results["default"]
    print(f"\n\nPaired against the default, on {HEADLINE}. Same resampled question ids both "
          f"sides.\n")
    deltas, pvals = {}, {}
    for name in available:
        if name == "default":
            continue
        va, vb = aligned(results[name], base, HEADLINE, "answerable")
        d = paired_delta(va, vb)
        deltas[name] = d
        pvals[name] = d["p"]
    adjusted = holm(pvals)

    print("| run | delta vs default | 95% interval | questions that move | p | p after Holm "
          "| real |")
    print("|---|---|---|---|---|---|---|")
    for name in sorted(deltas, key=lambda n: deltas[n]["delta"]):
        d = deltas[name]
        a = adjusted[name]
        verdict = "yes" if a["significant"] else "not shown"
        print(f"| {LABEL.get(name, name)} | {d['delta']:+.4f} | "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['questions_differing']}/{d['n']} | "
              f"{d['p']:.3f} | {a['p_holm']:.3f} | {verdict} |")
        out["vs_default"][name] = {**d, **a}

    print("\n\nPre-specified comparisons, one test each, no family adjustment.\n")
    print("| comparison | what it asks | delta | 95% interval | questions that move | p |")
    print("|---|---|---|---|---|---|")
    out["pairs"] = {}
    for a_name, b_name, question in PAIRS:
        if a_name not in results or b_name not in results:
            continue
        va, vb = aligned(results[a_name], results[b_name], HEADLINE, "answerable")
        d = paired_delta(va, vb)
        out["pairs"][f"{a_name} vs {b_name}"] = {**d, "asks": question}
        print(f"| {a_name} vs {b_name} | {question} | {d['delta']:+.4f} | "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['questions_differing']}/{d['n']} | "
              f"{d['p']:.3f} |")

    print(f"\n\nBy question kind, {HEADLINE} for the default. These are the smallest samples in "
          f"the repo.\n")
    print("| kind | n | chunk MRR | 95% interval |")
    print("|---|---|---|---|")
    out["by_kind"] = {}
    for kind in KINDS:
        rows = [r for r in base["rows"] if r["kind"] == kind]
        ci = mean_ci([r[HEADLINE] for r in rows])
        out["by_kind"][kind] = ci
        print(f"| {kind} | {ci['n']} | {ci['mean']:.3f} | [{ci['lo']:.2f}, {ci['hi']:.2f}] |")

    path = os.path.join(HERE, "results", "significance.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
