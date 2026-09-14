#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for retrieve-phase detection.

The load-bearing property is PRECISION, not recall. A false positive makes
PhaseProfiler recommend int8 HNSW quantization and NUMA index sharding for what
was actually a file search — confidently wrong hardware advice. A false negative
just leaves the command as ``act``, which is what it used to be anyway.

Run: poetry run pytest src/benchmarks/terminal_bench/test_retrieval.py -q
"""
from __future__ import annotations

import pytest

from src.benchmarks.terminal_bench.retrieval import (
    classify_phase,
    is_retrieval,
    retrieval_signals,
)


# ── vector / ANN indexes ────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "python -c 'import faiss; faiss.IndexFlatL2(768)'",
    "python build_index.py --index-type hnsw",
    "python -m hnswlib_bench --dim 384",
    "qdrant-cli collection create docs",
    "python query.py --vector-store milvus",
    "psql -c \"SELECT id FROM docs ORDER BY emb <-> '[1,2]' LIMIT 5\" && python pgvector_query.py",
    "python -c 'import chromadb; chromadb.Client()'",
    "python search.py --embeddings out.npy",
])
def test_vector_workloads_are_retrieve(cmd):
    assert classify_phase(cmd) == "retrieve", cmd
    assert retrieval_signals(cmd), "must record why it was classified"


# ── lexical / inverted-index retrieval ──────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "python -c 'from rank_bm25 import BM25Okapi'",
    "python retrieve.py --scorer bm25",
    "python -c 'from sklearn.feature_extraction.text import TfidfVectorizer'",
    "python score.py --method tf-idf",
    "curl -s localhost:9200/docs/_search -d '{\"query\":{}}' # elasticsearch",
    "python -m opensearchpy.bulk_index corpus.jsonl",
    "java -jar lucene-cli.jar index ./corpus",
    "python build.py --inverted-index",
])
def test_lexical_retrieval_is_retrieve(cmd):
    """BM25/TF-IDF/inverted indexes are retrieval, exactly as vector search is."""
    assert classify_phase(cmd) == "retrieve", cmd


# ── embeddings and reranking ────────────────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "python -c 'from sentence_transformers import SentenceTransformer'",
    "python embed.py && python -c 'model.encode(docs)' # fastembed",
    "python rerank.py --model cross_encoder/ms-marco",
    "python -m colbert.index --collection docs.tsv",
    "python -m mteb run -t SciFact",
    "python -m beir.retrieval.evaluate",
])
def test_embedding_and_rerank_are_retrieve(cmd):
    assert classify_phase(cmd) == "retrieve", cmd


# ── the boundary: filesystem inspection is NOT retrieval ────────────────────

@pytest.mark.parametrize("cmd", [
    "grep -rn 'def main' /app",
    "find / -name '*.py' -maxdepth 3",
    "cat /app/README.md",
    "ls -la /app",
    "rg --json 'TODO' src/",
    "awk '{print $2}' data.txt | sort | uniq -c",
    "sed -i 's/foo/bar/' input.tex",
    "head -100 main.log | grep Overfull",
])
def test_filesystem_inspection_is_act(cmd):
    """This is the whole point of the boundary — shell search stays `act`."""
    assert classify_phase(cmd) == "act", cmd
    assert retrieval_signals(cmd) == []


def test_grepping_for_a_retrieval_word_is_still_act():
    """`grep -r faiss .` is someone searching a repo, not querying an index.

    Without the leading-binary guard the word "faiss" in the argument would
    reclassify a plain file search as vector retrieval.
    """
    assert classify_phase("grep -rn faiss /app") == "act"
    assert classify_phase("find / -name '*hnsw*'") == "act"
    assert classify_phase("cat bm25_notes.md") == "act"


@pytest.mark.parametrize("cmd", [
    "pdflatex main.tex",
    "make -j8",
    "gcc -O2 sim.c -o sim",
    "pytest -q",
    "apt-get install -y curl",
    "python train.py --epochs 3",
    "echo 1 > /logs/verifier/reward.txt",
])
def test_ordinary_work_is_act(cmd):
    assert classify_phase(cmd) == "act", cmd


# ── precision guards: near-miss words must not fire ─────────────────────────

@pytest.mark.parametrize("cmd", [
    "python -c 'print(\"chromatic aberration\")'",     # chroma
    "echo 'this is annoying' > note.txt",              # annoy
    "python plot.py --vector-graphics out.svg",         # 'vector' alone
    "python -c 'v = [1,2,3]  # a vector of results'",   # 'vector' in prose
    "./solr_backup_notes.sh",                          # solr inside a filename
])
def test_lookalike_words_do_not_fire(cmd):
    """\\b guards and call-shaped patterns keep prose from becoming a phase."""
    assert classify_phase(cmd) == "act", cmd


def test_empty_and_whitespace_are_act():
    for cmd in ("", "   ", "\n"):
        assert classify_phase(cmd) == "act"
        assert is_retrieval(cmd) is False


# ── shell wrappers must not hide the real binary ────────────────────────────

def test_wrappers_are_skipped_when_finding_the_binary():
    # A wrapped grep is still a grep.
    assert classify_phase("sudo timeout 30 grep -r faiss /app") == "act"
    assert classify_phase("bash -lc 'grep -rn hnsw .'") == "act"
    # A wrapped retrieval command is still retrieval.
    assert classify_phase("sudo python -c 'import faiss'") == "retrieve"
    assert classify_phase("time python -m mteb run") == "retrieve"


# ── the reason is recorded, not just the verdict ─────────────────────────────

def test_signals_name_the_kind_of_retrieval():
    assert "vector_index" in retrieval_signals("python -c 'import faiss'")
    assert "lexical_index" in retrieval_signals("python -c 'from rank_bm25 import BM25Okapi'")
    assert "embedding" in retrieval_signals(
        "python -c 'from sentence_transformers import SentenceTransformer'")
    assert "rerank" in retrieval_signals("python rerank.py --model cross_encoder/x")
    assert "retrieval_benchmark" in retrieval_signals("python -m mteb run -t SciFact")


def test_signals_are_deduplicated():
    sig = retrieval_signals("python -c 'import faiss, hnswlib; import qdrant'")
    assert sig.count("vector_index") == 1


# ── the phase is a first-class citizen downstream ───────────────────────────

def test_retrieve_is_in_the_pipeline_and_has_solutions():
    """The analyzer must already know this phase, or tagging it changes nothing."""
    from src.analyzers.phase_profiler import (
        PHASE_LABELS, PHASE_ORDER, PHASE_SOLUTIONS,
    )
    assert "retrieve" in PHASE_ORDER
    assert PHASE_ORDER.index("retrieve") == 1, "Retrieve is phase 02"
    assert "Retrieve" in PHASE_LABELS["retrieve"]
    # These recommendations are exactly what a false positive would misapply.
    assert PHASE_SOLUTIONS["retrieve"]["working_set_overflow"]
