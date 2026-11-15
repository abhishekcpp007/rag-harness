#!/bin/sh
# Every ablation the README tables are built from, in one list, so `make eval` on a host and a
# `docker run` inside the image cannot drift apart. PY picks the interpreter: the venv's on a
# host, the image's own on a container, where there is no venv and no make.
set -eu
PY="${PY:-python}"

$PY -m eval.run_eval --name default
$PY -m eval.run_eval --name no-expansion --expansion none
$PY -m eval.run_eval --name no-rerank --no-rerank
$PY -m eval.run_eval --name cross-rerank --rerank cross
$PY -m eval.run_eval --name cross-lex --rerank cross+lexical
$PY -m eval.run_eval --name cross-bge --rerank cross --rerank-model BAAI/bge-reranker-base
$PY -m eval.run_eval --name no-headings --no-headings
$PY -m eval.run_eval --name hybrid --with-sparse
$PY -m eval.run_eval --name sparse-only --no-dense --with-sparse
$PY -m eval.run_eval --name hash-only --embedder hash
$PY -m eval.run_eval --name hash-hybrid --embedder hash --with-sparse
$PY -m eval.sweep
