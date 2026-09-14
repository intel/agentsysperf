#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Detect the ``retrieve`` phase in shell commands.

Phase 02 (Retrieve — Context Enrichment) is a first-class stage of the agentic
pipeline: vector search, embedding generation, reranking, and lexical/semantic
search over an index. It has its own hardware signature (working-set overflow,
memory-bandwidth saturation, cross-NUMA index traffic) and its own optimizations
(``PHASE_SOLUTIONS["retrieve"]``: NUMA-aware index partitioning, int8 HNSW
quantization, passage-level RAG). Those recommendations are only meaningful if
the span really was retrieval.

**What counts.** Semantic and index-backed retrieval:

* vector / ANN indexes — FAISS, HNSW, ScaNN, Annoy, Qdrant, Milvus, Weaviate,
  Chroma, LanceDB, pgvector, Pinecone
* embedding generation — ``sentence_transformers``, ``*.encode()``, embedding
  endpoints
* lexical retrieval and inverted indexes — BM25, TF-IDF, Elasticsearch,
  OpenSearch, Solr, Lucene, Whoosh, Tantivy, Xapian
* reranking — cross-encoders, ColBERT
* retrieval benchmark harnesses — MTEB, BEIR

**What does NOT count.** Filesystem inspection: ``grep``, ``rg``, ``find``,
``locate``, ``cat``, ``ls``, ``awk``, ``sed``. These are shell *actions* and
belong to ``act``. This boundary is the whole point: tagging a ``grep`` as
retrieval would make PhaseProfiler recommend int8 HNSW quantization for a file
search, which is worse than saying nothing.

Because a false positive produces confidently wrong hardware advice, the
matcher is deliberately **high-precision, low-recall**: it fires only on
explicit retrieval tooling, and anything ambiguous stays ``act``.
"""

from __future__ import annotations

import re
from typing import Final, List, Tuple

# ── Signals ─────────────────────────────────────────────────────────────────
#
# Each entry is (compiled pattern, short reason). The reason is recorded on the
# span so a reviewer can audit WHY a command was called retrieval — a phase
# attribution nobody can explain is not evidence.
#
# \b guards keep these from firing inside unrelated words: "chroma" must not
# match "chromatic", "annoy" must not match "annoying", "solr" must not match a
# path fragment. Library names are matched as import/CLI tokens.

_VECTOR_STORES: Final = (
    r"faiss|hnswlib|hnsw|scann|annoy|nmslib|"
    r"qdrant|milvus|weaviate|chromadb|chroma_client|lancedb|"
    r"pgvector|pinecone|vespa|vald|marqo|usearch"
)
# Vector-store names appear inside identifiers too (hnswlib_bench,
# faiss_index.py), where \b does not fire because "_" is a word character.
# Anchor on a non-word or underscore boundary instead.
_VECTOR_IN_IDENT: Final = rf"(?:^|[^A-Za-z0-9]|_)(?:{_VECTOR_STORES})(?:[^A-Za-z0-9]|_|$)"
_LEXICAL_ENGINES: Final = (
    r"elasticsearch|opensearch(?:py)?|solr|lucene|whoosh|tantivy|xapian|"
    r"rank_bm25|bm25|meilisearch|typesense|sphinxsearch"
)
_EMBEDDERS: Final = (
    r"sentence_transformers|sentencetransformer|"
    r"instructor_embedding|fastembed|text2vec|openai\.embeddings|"
    r"transformers\.AutoModel(?!ForCausalLM)"
)
_RERANKERS: Final = r"cross_encoder|crossencoder|colbert|rerank(?:er|ing)?"
_BENCHMARKS: Final = r"\bmteb\b|\bbeir\b"

_RETRIEVAL_SIGNALS: Final[Tuple[Tuple[re.Pattern, str], ...]] = (
    (re.compile(_VECTOR_IN_IDENT, re.IGNORECASE), "vector_index"),
    # pgvector / Postgres distance operators: <-> (L2), <=> (cosine), <#> (inner
    # product). A SQL ORDER BY on one of these IS an ANN query.
    (re.compile(r"<->|<=>|<#>"), "vector_search"),
    (re.compile(rf"\b(?:{_LEXICAL_ENGINES})\b", re.IGNORECASE), "lexical_index"),
    (re.compile(rf"(?:{_EMBEDDERS})", re.IGNORECASE), "embedding"),
    (re.compile(rf"\b(?:{_RERANKERS})\b", re.IGNORECASE), "rerank"),
    (re.compile(_BENCHMARKS, re.IGNORECASE), "retrieval_benchmark"),
    # Generic but unambiguous: TF-IDF and inverted indexes, however spelled.
    (re.compile(r"\btf[-_ ]?idf\b|TfidfVectorizer", re.IGNORECASE), "lexical_index"),
    (re.compile(r"\binverted[-_ ]index\b", re.IGNORECASE), "lexical_index"),
    # Embedding/vector-search API surfaces, not bare words: require the call or
    # flag shape so prose like "a vector of results" cannot match.
    (re.compile(r"\b(?:similarity_search|vector_search|knn_search|ann_search)\b",
                re.IGNORECASE), "vector_search"),
    (re.compile(r"\bembeddings?\.(?:create|encode|embed)\b", re.IGNORECASE), "embedding"),
    (re.compile(r"--(?:embed|embeddings|vector-store|index-type)\b",
                re.IGNORECASE), "vector_index"),
)

# Commands that are filesystem inspection, never retrieval. Checked FIRST so a
# path or filename that happens to contain a signal word cannot reclassify a
# plain file search — `grep -r faiss .` is someone looking for the string
# "faiss" in a repo, which is act, not a vector query.
_FILESYSTEM_BINARIES: Final[frozenset] = frozenset({
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "ripgrep",
    "find", "locate", "mlocate", "fd", "which", "whereis",
    "cat", "head", "tail", "less", "more", "ls", "tree", "stat",
    "awk", "sed", "cut", "sort", "uniq", "wc", "tr", "diff",
})

_PASSTHROUGH: Final[frozenset] = frozenset({
    "sudo", "env", "nice", "time", "timeout", "strace", "nohup", "bash", "sh",
    "-c", "-lc",
})


def _leading_binary(command: str) -> str:
    """First meaningful binary name, skipping shell wrappers and their args.

    Numeric tokens are skipped too: in ``sudo timeout 30 grep -r faiss /app``
    the ``30`` is ``timeout``'s duration, and stopping there would miss the real
    binary (``grep``) and let the argument "faiss" reclassify a file search as
    vector retrieval.
    """
    for token in command.strip().split():
        base = token.rsplit("/", 1)[-1]
        if base in _PASSTHROUGH or base.startswith("-"):
            continue
        # timeout/nice take a numeric operand before the command.
        if base.replace(".", "", 1).isdigit():
            continue
        # A quoted wrapper body: `bash -lc 'grep -rn hnsw .'` — look inside.
        stripped = base.strip("'\"")
        if stripped != base and stripped:
            return _leading_binary(stripped)
        return base
    return ""


def retrieval_signals(command: str) -> List[str]:
    """Return the retrieval signals in *command*, or [] if it is not retrieval.

    Filesystem inspection returns [] regardless of content, so ``grep -r faiss``
    is not mistaken for a vector query.
    """
    if not command or not command.strip():
        return []
    if _leading_binary(command) in _FILESYSTEM_BINARIES:
        return []
    found: List[str] = []
    for pattern, reason in _RETRIEVAL_SIGNALS:
        if pattern.search(command) and reason not in found:
            found.append(reason)
    return found


def is_retrieval(command: str) -> bool:
    """True when *command* performs semantic or index-backed retrieval."""
    return bool(retrieval_signals(command))


def classify_phase(command: str) -> str:
    """Return the pipeline phase for a shell *command*: ``retrieve`` or ``act``.

    Only these two are possible here — ``admit`` and ``commit`` are owned by the
    adapter (env provisioning, oracle scoring) and ``reason`` by the LLM call,
    none of which run shell commands.
    """
    return "retrieve" if is_retrieval(command) else "act"


__all__ = ["classify_phase", "is_retrieval", "retrieval_signals"]
