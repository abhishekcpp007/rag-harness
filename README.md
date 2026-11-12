# rag-harness

A question answering service over a small engineering handbook, plus the eval harness that says
whether it actually works.

The service half is ordinary. Chunk the markdown, embed it, retrieve, answer with citations. The
harness is the half worth reading. Every choice in the retrieval path has a measured number next
to it, and two of those numbers came out the opposite way from what I expected before running them.

```bash
make install        # uv venv, editable install, fastembed for real embeddings
make index          # 12 documents -> 54 chunks
make test           # 29 tests, no network, no model download
make eval           # 7 configurations over 42 questions
make tables         # the markdown below, regenerated from eval/results/
```

Ask it something:

```bash
$ .venv/bin/python -m rag.cli ask "I got NW-5001 back, should I retry the charge?"
| NW-5001 | Acquirer timeout | Do not retry the charge, poll its state | [1]
| NW-5002 | Ledger write failed, charge rolled back | Safe to retry with the same idempotency key | [1]

  [1] 03-error-codes :: Error Codes > Platform errors, 500 range

$ .venv/bin/python -m rag.cli ask "how do I issue a virtual card to an employee"
I could not find that in the handbook.
```

The second one is the interesting case. Nothing in the corpus is about virtual cards, retrieval
still returns its five best guesses, and the service refuses rather than letting a model write a
paragraph on top of them.

The default generator is extractive, so it quotes the handbook instead of writing prose. That is
deliberate and it has a visible cost, shown further down.

## The corpus

Twelve markdown files for a made up card acquiring platform called Northwind. Charges, webhooks,
settlement, payouts, a runbook, an error code table, a glossary. It's written rather than scraped,
which means there is no licence question and the gold labels are exact.

It also has the shape that makes retrieval hard in the way real handbooks are hard. Error codes
like `NW-4021` carry almost no semantic signal. A runbook step reads "retry twice, then page the
on-call" and means nothing without the heading above it. Three different files mention settlement
timing and only one of them answers a question about it.

## The eval set

42 questions in `eval/golden.yaml`, each labelled with the exact chunk ids that contain the answer.
35 are answerable. 7 are about things the corpus has never heard of, and the only correct behaviour
on those is to refuse.

Questions are tagged by kind so a score can be sliced:

| kind | n | what it stresses |
|---|---|---|
| identifier | 6 | exact codes, `NW-4021`, test card numbers |
| semantic | 9 | paraphrase with almost no shared vocabulary |
| procedural | 8 | runbook steps and ordering |
| factual | 12 | one specific number or name |
| unanswerable | 7 | refusing |

Gold labels are chunk ids, not document ids. With only twelve documents, document level recall
saturates near 1.0 and stops telling you anything.

## What the numbers say

Regenerate with `make eval && make tables`. Every row reindexes from scratch.

| run | chunk recall@1 | chunk recall@3 | chunk MRR | answer ok | citations on target | ms |
|---|---|---|---|---|---|---|
| default (bge-small, dense, lexical rerank) | 0.80 | 0.89 | 0.850 | 0.60 | 0.74 | 31 |
| default minus the lexical rerank | 0.80 | 0.89 | 0.834 | 0.60 | 0.74 | 23 |
| default minus the heading path | 0.80 | 0.86 | 0.841 | 0.57 | 0.70 | 26 |
| default plus BM25 fusion | 0.71 | 0.83 | 0.793 | 0.51 | 0.69 | 27 |
| BM25 alone | 0.66 | 0.83 | 0.739 | 0.51 | 0.66 | 4 |
| hash embedder, dense only | 0.60 | 0.74 | 0.667 | 0.51 | 0.66 | 4 |
| hash embedder plus BM25 fusion | 0.66 | 0.83 | 0.739 | 0.54 | 0.66 | 4 |

Chunk MRR split by question kind:

| run | identifier | semantic | procedural | factual |
|---|---|---|---|---|
| default (bge-small, dense, lexical rerank) | 1.00 | 0.56 | 0.84 | 1.00 |
| default minus the lexical rerank | 1.00 | 0.52 | 0.82 | 1.00 |
| default minus the heading path | 1.00 | 0.44 | 0.94 | 1.00 |
| default plus BM25 fusion | 1.00 | 0.44 | 0.78 | 0.96 |
| BM25 alone | 1.00 | 0.34 | 0.67 | 0.96 |
| hash embedder, dense only | 1.00 | 0.28 | 0.54 | 0.88 |
| hash embedder plus BM25 fusion | 1.00 | 0.34 | 0.73 | 0.92 |

Refusal behaviour, at the shipped threshold of 0.34:

| run | refused, cannot be answered | refused, could have been answered |
|---|---|---|
| default (bge-small, dense, lexical rerank) | 0.71 | 0.11 |
| default minus the lexical rerank | 0.71 | 0.11 |
| default minus the heading path | 0.71 | 0.09 |
| default plus BM25 fusion | 0.57 | 0.09 |
| BM25 alone | 0.57 | 0.09 |
| hash embedder, dense only | 0.57 | 0.11 |
| hash embedder plus BM25 fusion | 0.57 | 0.09 |

The millisecond column is wall clock on a laptop including the embedding call, so treat
it as an order of magnitude and not a benchmark.

## Hybrid retrieval lost, and that is the point

I built BM25 and reciprocal rank fusion first, because hybrid retrieval is what everybody
recommends. Then I measured it against plain dense retrieval on the same index.

Fusing BM25 into a `bge-small-en-v1.5` index moved chunk MRR from **0.850 down to 0.793**, and
answer quality from 0.60 to 0.51. It lost on semantic questions, 0.56 down to 0.44, and on
procedural ones, 0.84 down to 0.78. It won nowhere.

The folklore argument for hybrid is identifiers. That argument does not reproduce here. Identifier
MRR is 1.00 for every configuration in the table, including the deliberately weak hash embedder,
because `NW-4021` survives subword tokenisation well enough that a dense encoder has no trouble
with it.

So BM25 is off by default. It earns its place when the encoder is weak, and the table shows exactly
that: with the hash embedder, adding BM25 moves MRR from **0.667 up to 0.739**. One flag turns it
back on.

```bash
RAG_USE_SPARSE=1 python -m rag.cli search "NW-4021"
```

BM25 stays in the codebase for a second reason anyway. The abstain signal is built on it.

## Refusing to answer

The first abstain signal I tried was the fused retrieval score, and it does not work at all. RRF
scores a chunk purely on its rank, so the top hit for a question about Apple Pay scores the same
as the top hit for a question the handbook answers perfectly. On the golden set the two
distributions sat on top of each other, medians 0.0377 and 0.0348, with the best unanswerable
query scoring above the median answerable one.

What does separate them is a blend of the best BM25 score and how much of the question's vocabulary
appears in the retrieved text. A question about virtual cards matches no rare term in the corpus,
so both halves collapse together.

The threshold is not a guess. `eval/sweep.py` walks it from 0 to 1 and writes the whole curve to
`eval/results/abstain-sweep.json`. Picking a point needs a stated preference, and mine is that
wrongly refusing a question the handbook can answer is the failure a user actually complains about.
So that gets a budget of one in eight, and under that budget the threshold is **0.34**, which
catches 71 percent of unanswerable questions while wrongly refusing 11 percent of answerable ones.

Push it to 0.60 and refusals reach 100 percent of the unanswerable set, at the cost of refusing 37
percent of real questions. The curve is committed so you can disagree with my choice and read your
own off it.

## Citations are checked, not requested

Asking a model for citations gets you citations. Checking them is a separate job, and it happens
in `rag/answer.py` after generation.

Every sentence has to carry a `[n]` pointing at a context block that was really in the prompt. A
sentence citing nothing gets dropped. A sentence citing block 9 when only 5 blocks were sent gets
dropped, which is the quiet failure mode where a model invents a plausible footnote. If dropping
leaves nothing behind, the answer turns into a refusal.

The eval measures whether the surviving citations point at a gold chunk's document, which is the
`citations on target` column. At 0.74 it is the weakest number in the table and the most honest
one. An answer can be correct and still point the reader at the wrong page.

## Chunking

Two rules, both measurable.

**The heading path travels with the text.** A paragraph reading "retry twice, then page the
on-call" is embedded as `Runbook: Settlement stall > Recovery steps :: retry twice, then page the
on-call`. Turning that off costs 0.03 answer quality and most of the semantic slice, 0.56 down to
0.44.

**Code fences and tables are atomic.** Splitting a fence in half produces a chunk that reads like
config and embeds like prose. `tests/test_chunking.py` pins it by chunking a document at a window
small enough to force the split and asserting the backtick count stays even.

## What the harness caught

Four things, and none of them would have shown up by reading a ranking table.

**The abstain signal was secretly coupled to a feature flag.** `confidence()` read its BM25 term
off `hit.signals["sparse"]`, which only exists when sparse retrieval is switched on. Turning BM25
off halved the confidence number, and the service started refusing 77 percent of the questions it
was retrieving correctly. The retrieval metrics got better in the same run, which is what makes
this one nasty. It now reads BM25 from the index directly and
`tests/test_confidence.py::test_confidence_does_not_move_when_sparse_retrieval_is_switched_off`
holds it there.

**The answer generator was cutting sentences in half.** The corpus is hard wrapped at about 100
columns, and the sentence splitter worked per line, so answers came out as "refused every time it
was asked. [2]" with the subject stranded on the line above. Unwrapping paragraphs before splitting
moved answer quality from **0.486 to 0.600** on its own, measured by reverting just that change.
It is the single largest improvement in the project and it is not a retrieval change at all.

**An eval run reported zeros instead of failing.** After BM25 moved to off by default, the
`--no-dense` ablation left no retriever running at all. The run finished, wrote a result file, and
reported 0.000 for every metric with a straight face. A harness that stays quiet while measuring
nothing is worse than one that crashes, so building a retriever with both signals off now raises.

**A docstring claimed the opposite of what the code did.** I had written that RRF at k=60 lets one
confident retriever beat two lukewarm ones. It doesn't. A chunk ranked fourth by both scores
2/64 = 0.031 and beats a chunk ranked first by one and missing from the other at 1/61 = 0.016. The
test failed, the docstring was wrong, and `tests/test_fusion.py` now pins the behaviour at both
k=60 and k=0.

## Things I measured and then left alone

Light plural stemming was a wash. It lifts citations on target from 0.710 to 0.742 and drops chunk
MRR from 0.864 to 0.850. It stays in because the lexical reranker and the abstain overlap both
want `runs` and `run` to be the same token, but it is not carrying the result.

The lexical reranker is worth 0.016 MRR over no reranker at all. A cross encoder would do better
and costs a model load plus roughly 40ms a query. On 54 chunks that trade is not worth making, so
the heavier model is left out until the eval asks for it.

## Honest limits

The generator is extractive by default. It picks sentences out of the retrieved chunks and attaches
the chunk they came from, so it cannot hallucinate, and `answer ok` of 0.60 is a floor rather than a
ceiling. Here is what the floor looks like when it fails:

```bash
$ .venv/bin/python -m rag.cli ask "what time does the settlement sweep run in India"
There is no manual sweep, and asking for one has been refused every time it was asked. [1]
India traffic runs out of `ap-south-1` and Singapore traffic out of `ap-southeast-1`. [5]

  [1] 05-settlement :: Settlement > Batch cutoff
```

Retrieval did its job. Block 1 is the right chunk and the sentence next to the one it picked reads
"The sweep runs at 23:40 IST". The scorer went wrong because `time` in the question matched `every
time it was asked`, and a sentence picker with no idea what a question is cannot tell those apart.
Any instruction following model fixes this. The harness exists so the fix can be proven rather
than assumed. Retrieval is finding the right chunk 89 percent of the time at k=3, so most of the gap
between 0.89 and 0.60 belongs to the generator. Set `RAG_LLM=anthropic` with a key and the same
harness scores a real model against the same floor. I have not run that, because I do not want to
publish numbers I cannot regenerate on a clean checkout.

The abstain threshold was read off a curve computed on this same golden set. That makes the
refusal numbers an upper bound, not a prediction about a fresh corpus. With 42 questions a held out
split would be too small to mean anything, so I would rather state the bias than hide it behind a
80/20 line.

The vector store is one float32 matrix held in memory with the text in SQLite. That is a BLAS call
per query and it is fine to somewhere around a hundred thousand chunks. Past that it wants a real
vector database, and `rag/store.py` is the only file that would change.

Two environment pairs are supported for embeddings and generation but only one path is exercised in
CI, which runs the hash backend so it stays off the network.

## Layout

```
rag/
  chunking.py     heading aware markdown splitter, atomic code fences
  tokenize.py     shared tokeniser, identifier splitting, small plural rule
  bm25.py         Okapi BM25 with an explain() for when a ranking looks wrong
  embedding.py    hash | fastembed | openai behind one interface
  store.py        SQLite for text, one numpy matrix for vectors
  retrieve.py     dense, sparse, reciprocal rank fusion, lexical rerank
  confidence.py   the abstain signal, and what had to be thrown away first
  answer.py       context assembly, citation enforcement, refusal
  llm.py          extractive | anthropic | openai
  api.py          FastAPI, 6 routes including /explain
corpus/           12 markdown documents, 54 chunks
eval/
  golden.yaml     42 questions with gold chunk ids
  metrics.py      recall@k, MRR, nDCG, citation support
  run_eval.py     one config in, one result file out
  sweep.py        abstain threshold curve
  table.py        the markdown tables above
  results/        committed, so a diff shows a regression
tests/            29 tests, hash backend, no network
```

## API

```bash
make serve
curl localhost:8000/health
curl "localhost:8000/search?q=settlement+sweep&k=3"
curl -X POST localhost:8000/answer -H 'content-type: application/json' \
     -d '{"question":"when does the settlement sweep run"}'
curl "localhost:8000/explain?q=NW-4021&chunk_id=03-error-codes%231"
```

`/explain` prints the per term BM25 contribution for one chunk. It is the first thing I reach for
when a result looks wrong, and it has saved more time than any amount of logging.

`docker compose up --build` builds the image, indexes at build time so the first request stays off
the network, and serves on 8000.

## Configuration

Everything is an environment variable, listed in `.env.example` and typed in `rag/config.py`.

| variable | default | notes |
|---|---|---|
| `RAG_EMBEDDER` | `fastembed` | `hash` needs no download, `openai` needs a key |
| `RAG_USE_SPARSE` | `0` | BM25 fusion, see the table above before turning it on |
| `RAG_RERANK` | `lexical` | or `none` |
| `RAG_ABSTAIN_CONFIDENCE` | `0.34` | read off `eval/results/abstain-sweep.json` |
| `RAG_CHUNK_WORDS` | `180` | window, measured in words |
| `RAG_LLM` | `extractive` | or `anthropic`, or `openai` |
