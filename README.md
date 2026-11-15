# rag-harness

A question answering service over a small engineering handbook, plus the eval harness that says
whether it actually works.

The service half is ordinary. Chunk the markdown, embed it, retrieve, answer with citations. The
harness is the half worth reading. Every choice in the retrieval path has a measured number next
to it, and the ones worth your time are the ones that came out the opposite way from what I
expected before running them. Two of those are the reason the default configuration is what it is.

There are two other benchmarks in here beside the quality one. `bench/scale.py` sweeps corpus
size to find where each index stops working, and the answer was not the index I expected. And
`bench/load.py` points concurrent clients at the running service, which is how three bugs that a
green test suite cannot see got found.

```bash
make install        # uv venv, editable install, fastembed for real embeddings
make index          # 12 documents -> 54 chunks
make test           # 127 tests, no network, no model download
make eval           # 11 configurations over 42 questions
make tables         # the markdown below, regenerated from eval/results/
make docker-test    # the same lint and suite inside the image, with no network
make bench          # where each index stops working, sparse and dense, sizes to 250k
make latency        # per-query cost of every retrieval configuration
make load           # 16 concurrent clients at a running service, non-zero if any fail
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

| run | chunk recall@1 | chunk recall@3 | chunk MRR | answer ok | citations on target |
|---|---|---|---|---|---|
| default (bge-small, dense, PRF expansion, lexical rerank) | 0.83 | 0.89 | 0.864 | 0.60 | 0.75 |
| default minus the query expansion | 0.80 | 0.89 | 0.850 | 0.60 | 0.74 |
| default minus the lexical rerank | 0.77 | 0.89 | 0.820 | 0.60 | 0.76 |
| default with a cross encoder instead of the lexical rerank | 0.74 | 0.89 | 0.810 | 0.60 | 0.75 |
| default with the cross encoder rank-fused into the lexical one | 0.74 | 0.89 | 0.821 | 0.54 | 0.72 |
| default minus the heading path | 0.80 | 0.86 | 0.830 | 0.57 | 0.67 |
| default plus BM25 fusion | 0.71 | 0.83 | 0.793 | 0.51 | 0.69 |
| BM25 alone | 0.66 | 0.83 | 0.739 | 0.51 | 0.66 |
| hash embedder, dense only | 0.60 | 0.71 | 0.670 | 0.51 | 0.64 |
| hash embedder plus BM25 fusion | 0.66 | 0.80 | 0.724 | 0.54 | 0.64 |

Chunk MRR split by question kind:

| run | identifier | semantic | procedural | factual |
|---|---|---|---|---|
| default (bge-small, dense, PRF expansion, lexical rerank) | 1.00 | 0.56 | 0.91 | 1.00 |
| default minus the query expansion | 1.00 | 0.56 | 0.84 | 1.00 |
| default minus the lexical rerank | 1.00 | 0.52 | 0.75 | 1.00 |
| default with a cross encoder instead of the lexical rerank | 0.92 | 0.50 | 0.85 | 0.96 |
| default with the cross encoder rank-fused into the lexical one | 0.92 | 0.56 | 0.84 | 0.96 |
| default minus the heading path | 1.00 | 0.39 | 0.94 | 1.00 |
| default plus BM25 fusion | 1.00 | 0.44 | 0.78 | 0.96 |
| BM25 alone | 1.00 | 0.34 | 0.67 | 0.96 |
| hash embedder, dense only | 1.00 | 0.30 | 0.56 | 0.85 |
| hash embedder plus BM25 fusion | 1.00 | 0.31 | 0.69 | 0.92 |

Refusal behaviour, at the shipped threshold of 0.34:

| run | refused, cannot be answered | refused, could have been answered |
|---|---|---|
| default (bge-small, dense, PRF expansion, lexical rerank) | 0.71 | 0.09 |
| default minus the query expansion | 0.71 | 0.11 |
| default minus the lexical rerank | 0.71 | 0.11 |
| default with a cross encoder instead of the lexical rerank | 0.57 | 0.09 |
| default with the cross encoder rank-fused into the lexical one | 0.71 | 0.09 |
| default minus the heading path | 0.71 | 0.09 |
| default plus BM25 fusion | 0.57 | 0.09 |
| BM25 alone | 0.57 | 0.09 |
| hash embedder, dense only | 0.57 | 0.09 |
| hash embedder plus BM25 fusion | 0.57 | 0.09 |

There is no latency column in these tables, and there used to be. The eval runs each question
once, so its millisecond figure was a single sample with the generator and the metric bookkeeping
folded in. Two ablations that do an identical amount of retrieval work came out 15ms and 54ms
apart on a contended container, which is a number noisy enough to make the ranking figures next to
it look less trustworthy than they are. Latency gets its own benchmark, further down.

## Hybrid retrieval lost, and that is the point

I built BM25 and reciprocal rank fusion first, because hybrid retrieval is what everybody
recommends. Then I measured it against plain dense retrieval on the same index.

Fusing BM25 into a `bge-small-en-v1.5` index moved chunk MRR from **0.864 down to 0.793**, and
answer quality from 0.60 to 0.51. It lost on semantic questions, 0.56 down to 0.44, and on
procedural ones, 0.91 down to 0.78. It won nowhere.

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
0.42.

**Code fences and tables are atomic.** Splitting a fence in half produces a chunk that reads like
config and embeds like prose. `tests/test_chunking.py` pins it by chunking a document at a window
small enough to force the split and asserting the backtick count stays even.

## What the harness caught

Six things, and none of them would have shown up by reading a ranking table.

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

**A run could swap its encoder and keep the old label.** `get_embedder` falls back to the hash
backend when fastembed is missing, prints one line, and carries on. That's the right call for the
service, because a fresh clone still answers questions. It's the wrong call for the eval. The first
time I ran the ablations inside Docker, a leftover `RAG_EMBEDDER=hash` in the test image turned all
seven of them into hash runs, and each result file was written under its usual name. The numbers
were real. The labels weren't. `eval/run_eval.py` now fails when the backend that ran isn't the one
that was asked for, and `eval/table.py` won't render a row whose file disagrees with the label
above it.

**A flag had never once worked.** `--abstain-score` set a field called `abstain_score`. The setting
is called `abstain_confidence`, so every use of that flag raised TypeError, and nothing noticed
because nothing in the repo called it. It's `--abstain-confidence` now, with a test that drives it
through `main()`.

## The query expansion that fixed the wrong category

The worst column in the by-kind table has always been semantic: questions phrased the way a person
actually asks them. Someone types "what stops a double tap on the pay button from charging twice"
and the handbook says "idempotency key". The two share no vocabulary, so no amount of reranking
helps, because the right chunk never makes the candidate list in the first place.

The standard answer to vocabulary mismatch is pseudo-relevance feedback, and it needs no model.
Retrieve once, take the top few chunks, average their vectors, pull the query vector a little way
toward that average, retrieve again. One extra matrix multiply against an index that is already
in memory.

It worked, and then it worked on the wrong category. Chunk MRR went from **0.850 to 0.864** and
recall@1 from 0.80 to 0.83, but semantic MRR did not move at all, 0.56 before and 0.56 after. The
whole gain came from procedural questions, **0.84 up to 0.91**. In hindsight that is the obvious
result: feedback can only amplify a neighbourhood the first pass already found, and on the
semantic failures the first pass lands in the wrong document entirely. Procedural questions were
landing in the right document and the wrong chunk, which is exactly the error a centroid fixes.

The gain is not a knob I tuned onto the golden set. Every beta between 0.2 and 0.8 lands chunk MRR
between 0.857 and 0.871, so the shipped 0.4 sits in the middle of a flat region rather than on a
peak. Feedback depth mattered more: 3 chunks beat 5 and 8, because the fourth and fifth are
usually already off topic on a corpus this small.

```bash
RAG_EXPANSION=none python -m rag.cli ask "how do I replay a failed webhook"
```

## The cross encoder does not earn its place here

The previous version of this README said a cross encoder "would do better" than the lexical
rerank and left it unmeasured. That is the kind of claim this repo exists to stop making, so it
is measured now, across the whole golden set, at three settings.

| reranker | chunk MRR | identifier | semantic |
|---|---|---|---|
| lexical, the default | 0.864 | 1.00 | 0.56 |
| `Xenova/ms-marco-MiniLM-L-6-v2` | 0.810 | 0.92 | 0.50 |
| the same model, rank-fused with lexical | 0.821 | 0.92 | 0.56 |
| `BAAI/bge-reranker-base` | 0.822 | 1.00 | 0.44 |

It loses, and the by-kind split says why. The cross encoder is trained on web passages where
meaning carries the signal, so it reads `NW-5001` as noise and promotes the prose chunk that
talks around the error code over the table row that defines it. Identifier MRR drops from 1.00 to
0.92 and factual from 1.00 to 0.96. Those are the two categories this corpus is mostly made of.

Chaining the two did not rescue it either, and the first attempt is worth recording. Running the
cross encoder and then applying the lexical bonus to its scores produced an ordering identical to
the cross encoder alone, byte for byte, because a sigmoid over logits saturates and the gap
between the top two probabilities is wider than a 40 percent bonus can close. Fusing the two
*rankings* with RRF does work as a mechanism, and it recovers semantic MRR to 0.56, but it still
lands below the lexical pass at 87 times the latency, which the table two sections down prices
properly.

So `cross` and `cross+lexical` both ship, both off. The code is in `rag/rerank.py` and one
environment variable turns either on. The table is the argument for leaving them off.

## The index everyone replaces was not the slow one

The plan for this phase was a vector index. A flat matrix times a query vector is fine at 54
chunks and obviously would not be fine at a million, so the next step was to find the size where
it breaks and put an approximate index behind it. `bench/scale.py` sweeps corpus size over both
halves of the search, with synthetic documents resampled from the real corpus so the term
distribution is not invented.

The dense half was never the problem.

| chunks | flat matrix | matrix size |
|---|---|---|
| 1,000 | 0.026 ms | 1.5 MB |
| 10,000 | 0.352 ms | 15.4 MB |
| 50,000 | 2.143 ms | 76.8 MB |
| 100,000 | 4.646 ms | 153.6 MB |
| 250,000 | 12.839 ms | 384.0 MB |

Twelve milliseconds for a quarter of a million chunks, in one BLAS call, with no index to build
and nothing to keep in sync. Meanwhile BM25 was scoring every document in Python:

| chunks | inverted index | the loop it replaced | speedup |
|---|---|---|---|
| 1,000 | 0.064 ms | 0.595 ms | 9.3x |
| 10,000 | 0.515 ms | 9.283 ms | 18.0x |
| 50,000 | 3.101 ms | 46.793 ms | 15.1x |

At fifty thousand chunks the sparse side was 22 times slower than the dense side it was fused
with. The part I had planned to replace was answering in 2ms and the part I had not thought about
was taking 47. So `rag/bm25.py` is an inverted index now. A term maps to the documents holding
it, scoring touches only the documents a query term appears in, the arithmetic is numpy rather
than nested loops, and the length normalizer is computed once at build time instead of once per
term per document. Both versions are still linear in corpus size, because a common term's
posting list grows with the corpus too. What changed is the constant, by more than an order of
magnitude.

The formula did not change, and that is checked rather than claimed. The scorer this replaced is
written out longhand in `tests/test_bm25.py` and every golden question is scored both ways, with
the ranking compared position by position and the scores to twelve digits. A rewrite for speed
that is allowed to move a ranking is not a rewrite, it is a new retriever.

Two smaller things fell out of the same sweep. Indexing was spending most of its time in the
tokenizer, and a word expands the same way every time it appears, so `rag/tokenize.py` caches per
word and indexing got 3.8x faster on real prose at a 99.5 percent hit rate. And fusion was
fetching its candidate pool from SQLite one row at a time, twenty round trips to build five
results, which is now one statement.

## What the benchmark's own vectors decided

The approximate index did get built, and the first run said it was useless: recall@10 of 0.152 at
250,000 chunks, giving up 85 percent of the correct top ten to be fast at something the flat
matrix was already doing in 12ms. That number was wrong, and the generator was why.

Random Gaussian vectors sit at cosine 0 from each other. Real ones do not. Measured off the 54
real bge-small chunks in this repo, two chunks of the same document sit at 0.710 and two chunks of
different documents at 0.620. Nothing is anywhere near 0. The model puts the whole corpus inside a
narrow cone, which is harder on a nearest-neighbour graph in one way, since every candidate looks
about equally close, and much easier in another, since the graph has real structure to walk.

So the sweep fits its vectors to those two measured numbers and reports the similarities it
actually produced. Same index, same parameters, same corpus size, two geometries:

| 250,000 chunks, ef | fitted to the real corpus | isotropic noise |
|---|---|---|
| 32 | 0.350 ms, recall 0.832 | 0.477 ms, recall 0.152 |
| 64 | 0.497 ms, recall 0.960 | 0.880 ms, recall 0.188 |
| 128 | 0.691 ms, recall 1.000 | 1.684 ms, recall 0.268 |
| 256 | 1.175 ms, recall 1.000 | 2.856 ms, recall 0.320 |

On vectors shaped like the real ones, the graph gives up nothing at ef=128 and runs 18 times
faster than exact search. On noise it is a disaster at every setting. A benchmark written the
obvious way would have told me to never use an approximate index, with numbers, and the numbers
would have been about my random number generator.

Which is why there is no approximate index in `rag/`. Not because it loses: at a quarter of a
million chunks it wins outright. Because this corpus has 54 chunks, the flat matrix holds to
250,000 of them in 12ms, and the honest crossover is somewhere past where any of this is going to
run. `bench/scale.py` is in the repo so the next person can find their own crossover instead of
taking mine.

## What each configuration costs per query

Latency used to be a column in the eval table and it is its own benchmark now, for the reason
further up: one sample per question is not a measurement. `bench/latency.py` builds a retriever
per configuration, warms it, then times every query on its own and reports the median over the
repeats. Search only, with the answer path measured separately, because those are two questions.

| configuration | search p50 | search p95 | answer p50 |
|---|---|---|---|
| default, lexical rerank and PRF | 4.75 ms | 6.23 ms | 5.14 ms |
| no query expansion | 4.83 ms | 6.83 ms | 4.71 ms |
| no rerank | 4.38 ms | 6.31 ms | 4.80 ms |
| plus BM25 fusion | 4.84 ms | 6.46 ms | 5.37 ms |
| BM25 alone | 0.32 ms | 0.39 ms | 0.65 ms |
| hash embedder | 0.35 ms | 0.43 ms | 0.64 ms |
| cross encoder, ms-marco MiniLM | 467 ms | 555 ms | |
| cross encoder rank-fused with lexical | 414 ms | 487 ms | |
| cross encoder, bge-reranker-base | 3670 ms | 4397 ms | |

The first four rows say the same thing, and it is the most useful number in the table. Taking the
reranker out saves 0.37ms. Adding BM25 fusion costs 0.09ms. Switching off query expansion is
inside the noise and came out slower. Every one of those is a rounding error on a 4.4ms floor,
and that floor is one bge-small forward pass over the query. At this corpus size the retrieval
is free and the encoder is the whole bill, which is why the hash rows run in a third of a
millisecond: no model, no forward pass.

It also prices the cross encoder honestly. 98 times slower than the cheap pass it loses to, and
`bge-reranker-base` 770 times slower, in exchange for a ranking the eval table says is worse.

## Three bugs that needed a second thread

Everything in this repo was written and tested one call at a time. FastAPI runs a plain `def`
route in a worker threadpool, so that is not how it runs. `bench/load.py` points 16 concurrent
clients at the service and exits non-zero if any request failed, and the first time it ran, 176
of 200 requests failed.

```
"status_codes": {"0": 76, "200": 24, "500": 100},
"failures": [["Internal Server Error", 100],
             ["ReadError: [Errno 54] Connection reset by peer", 76]]
```

All one error, in the server log a hundred times: `SQLite objects created in a thread can only be
used in that same thread`. The whole service shared one connection. Every unit test passed,
because a unit test is one thread, and so was every curl I had run by hand. The connection is per
thread now and the matrix stays shared, since it is written once at ingest and only read after.

The second bug I had written myself ten minutes earlier. The inverted index reuses one score
accumulator across queries rather than allocating a corpus-length array every search, which is a
sensible thing to do right up until two threads add into the same slots. That one produces a
quietly wrong ranking rather than an exception, which is worse. The buffer is per thread.

The third was the cold start. With the index built lazily on first request, 16 clients arrive,
none of them find a retriever, and each one loads the encoder for itself:

| first request behaviour | p50 | p95 | max |
|---|---|---|---|
| lazy, no lock | 86.2 ms | 2718.7 ms | 2729.1 ms |
| lazy, one build at a time | 35.0 ms | 400.5 ms | 431.2 ms |
| built during startup | 35.2 ms | 57.3 ms | 77.2 ms |

A lock around the build does most of it. Building during the startup hook moves the rest off the
request path and makes the readiness probe mean something, because the port does not answer until
the index is loaded.

Warm, with all three fixed:

```
/search   16 clients, 400 requests   429 rps   p50 35.2ms  p95 57.3ms  p99 68.9ms   0 failed
/answer   16 clients, 200 requests   347 rps   p50 43.2ms  p95 72.6ms  p99 82.0ms   0 failed
```

The p50 of 35ms against 4.8ms for a single query is queueing, not work. Four CPU bound requests at a
time is what one Python process gives you, and the fix for that is more processes rather than
anything in this code.

`tests/test_concurrency.py` pins all of it without needing a running service, and the CI docker
job now runs the real load test against the shipped image. That step exists because it used to
fail.

## Things I measured and then left alone

Light plural stemming was a wash, measured before query expansion went in: it lifted citations on
target from 0.710 to 0.742 and dropped chunk MRR from 0.864 to 0.850. It stays in because the
lexical reranker and the abstain overlap both want `runs` and `run` to be the same token, but it
is not carrying the result.

Two bigger sentence encoders lost to the small one. `BAAI/bge-base-en-v1.5` scored 0.817 chunk MRR
and `thenlper/gte-base` scored 0.795, against 0.864 for `bge-small-en-v1.5`, on the same index and
the same questions. bge-base did lift answer quality to 0.63 and citations to 0.74, so the
headline is not that small is better everywhere, only that four times the parameters bought
nothing in ranking here. Fifty-four short chunks of terse reference prose is not the regime those
models are bigger for.

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

The vector store is one float32 matrix held in memory with the text in SQLite. `bench/scale.py`
says that is one BLAS call and 12.8ms at 250,000 chunks against 384MB of RAM, so the limit is
memory before it is latency. Past that it wants a real vector database, and `rag/store.py` is the
only file that would change.

No query cache. The obvious one is an LRU over the whole search, and the reason it is absent is
that nothing I can measure says it would hit: the golden set has no repeated question, and a
handbook gets asked 42 different things rather than the same thing 42 times. A cache whose hit
rate I cannot state is a cache I cannot size.

The rate limiter is off by default and keeps its buckets in process memory, so behind more than
one replica each one limits separately. Real per-tenant limiting belongs at the gateway.

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
  retrieve.py     dense, sparse, reciprocal rank fusion, PRF query expansion
  rerank.py       none | lexical | cross encoder | rank fusion of two of them
  confidence.py   the abstain signal, and what had to be thrown away first
  answer.py       context assembly, citation enforcement, refusal
  llm.py          extractive | anthropic | openai
  observability.py JSON request logs, Prometheus metrics, token bucket limiter
  api.py          FastAPI, 7 routes including /explain and /metrics
corpus/           12 markdown documents, 54 chunks
bench/
  scale.py        corpus size sweep, inverted vs brute force, flat vs approximate
  latency.py      per-query cost per retrieval configuration, median over repeats
  load.py         concurrent clients against a running service, exits non-zero on any failure
  results/        committed
eval/
  golden.yaml     42 questions with gold chunk ids
  metrics.py      recall@k, MRR, nDCG, citation support
  run_eval.py     one config in, one result file out
  run_all.sh      the ablation list, shared by make eval and the container run
  sweep.py        abstain threshold curve
  table.py        the markdown tables above
  results/        committed, so a diff shows a regression
tests/            127 tests, hash backend, no network. 42 of them are the BM25 parity check
                  running over every golden question
```

## API

```bash
make serve
curl localhost:8000/health
curl "localhost:8000/search?q=settlement+sweep&k=3"
curl -X POST localhost:8000/answer -H 'content-type: application/json' \
     -d '{"question":"when does the settlement sweep run"}'
curl "localhost:8000/explain?q=NW-4021&chunk_id=03-error-codes%231"
curl localhost:8000/metrics
```

`/search` returns the milliseconds each stage took alongside the hits, so a slow query says which
stage was slow without a profiler. Every response carries an `x-request-id`, echoing the caller's
if it sent one, and every request writes one JSON line to stdout with that id in it.

`/explain` prints the per term BM25 contribution for one chunk. It is the first thing I reach for
when a result looks wrong, and it has saved more time than any amount of logging.

`docker compose up --build` builds the image, indexes at build time so the first request stays off
the network, and serves on 8000.

## Running the suite in a container

```bash
docker build --target test -t rag-harness:test .
docker run --rm --network none rag-harness:test    # ruff, then 127 tests
```

The test stage carries the dev dependencies and the suite. The image you ship carries neither, and
it's about 80MB smaller for it. `--network none` isn't decoration: the tests run on the hash
embedder, so they download nothing, and the flag is what proves it rather than me saying so.

The numbers committed under `eval/results/` were produced inside the container, with the model
cache mounted in so the run needed no network:

```bash
docker build --target runtime -t rag-harness:runtime .
docker run --rm -v "$PWD/eval/results:/app/eval/results" \
  -v "$TMPDIR/fastembed_cache:/tmp/fastembed_cache" rag-harness:runtime \
  sh eval/run_all.sh
```

Run the same eleven ablations on a macOS host instead and every summary metric comes out
identical, to the last digit, in all eleven. Of the 462 question runs that makes, six retrieve a
different order, and all six differ at ranks three through five with the gold chunk still first,
so no metric moves. Two chunks scoring within a float of each other land the other way round on a
different BLAS build. The default configuration agrees question for question.

Speed does not travel the same way. Docker Desktop gives the container a fraction of this host's
single core throughput, so every timing in this README comes from the host and from
`bench/latency.py` or `bench/scale.py`, never from the container that produced the quality
numbers.

## Configuration

Everything is an environment variable, listed in `.env.example` and typed in `rag/config.py`.

| variable | default | notes |
|---|---|---|
| `RAG_EMBEDDER` | `fastembed` | `hash` needs no download, `openai` needs a key |
| `RAG_USE_SPARSE` | `0` | BM25 fusion, see the table above before turning it on |
| `RAG_RERANK` | `lexical` | or `none`, `cross`, `cross+lexical`; see the table before paying for a cross encoder |
| `RAG_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | only read when `RAG_RERANK` starts with `cross` |
| `RAG_EXPANSION` | `prf` | or `none` to switch query expansion off |
| `RAG_PRF_DOCS` | `3` | how many first pass chunks the feedback vector averages |
| `RAG_PRF_BETA` | `0.4` | how far the query moves toward them |
| `RAG_ABSTAIN_CONFIDENCE` | `0.34` | read off `eval/results/abstain-sweep.json` |
| `RAG_CHUNK_WORDS` | `180` | window, measured in words |
| `RAG_LLM` | `extractive` | or `anthropic`, or `openai` |
| `RAG_LOG_FORMAT` | `json` | or `text` when a human is tailing the logs |
| `RAG_RATE_LIMIT_RPS` | `0` | off. Switching it on means the load test measures the limiter |
| `RAG_RATE_LIMIT_BURST` | `20` | bucket depth, only read when the rps is above zero |
