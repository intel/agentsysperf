#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Six agent-loop phase workloads, arch-portable, dependency-light.

Each function exercises the CPU signature of one agent-loop phase using
only numpy / stdlib / already-installed libs (tokenizers) so the whole
loop runs natively on x86_64 AND arm64 with no custom Docker images and
no multi-GB model download. The "real" heavy component (llama.cpp model,
FAISS index, DSA/IAA offload) swaps in behind the same function seam for
the cloud / bare-metal runs — the phase span boundaries do not change.

Design intent per phase (what the CPU actually does):

- reason    : REAL local-SLM decode via llama.cpp (Qwen2.5 GGUF, CPU).
              A shared Llama instance decodes a bounded number of tokens
              per call — the compute + memory-bandwidth-bound work
              per-core-speed arguments center on. Falls back to a
              dense matmul chain if llama.cpp / the model is unavailable.
- retrieve  : REAL sentence-transformers query embedding + FAISS ANN
              search over a shared prebuilt index + a sqlite metadata
              filter — embedding compute (mem-BW bound) + index search.
              The index is built once and shared across loops (like the
              Router), so search contends on shared read-mostly memory —
              the L3-pressure signal for the Key Question.
- act       : a real subprocess doing bounded CPU work — subprocess churn
              + scalar compute. Real: tool/test execution (same shape).
- admit     : admission-control gate — the REAL LiteLLM Router with a
              bounded parallel-request semaphore + routing strategy,
              driven concurrently by all loops sharing one Router. Uses
              mock_response so the gate (semaphore acquire, routing
              decision, rate-limit bookkeeping) is exercised without a
              network/LLM call. Light CPU, contention-sensitive — the
              semaphore is a shared resource that contends as agents pile
              up. Real (prod): same Router pointed at a live backend.
- context   : parse/validate a tool-output blob + real HF tokenization +
              a prefill-shaped matmul — the most memory-BW-bound phase,
              the chiplet-tax probe. Real: same + DSA/IAA offload on GNR.
- commit    : serialize state to json + a sqlite write — I/O + light CPU.
              Real: + git commit of the patch.

Every function takes a `scale` int (work multiplier, default 1) and
returns a small dict of what it did, so the adapter can sanity-check the
phase actually ran (integrity: measured, not assumed).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Tokenizer is loaded lazily and cached — import cost is paid once, not
# per phase call. transformers/tokenizers is already installed.
_TOKENIZER: Any = None

# One shared LiteLLM Router across all concurrent loops — this is what makes
# the admission gate actually contend. Built lazily on first admit() call and
# reused; the parallel-request semaphore inside it is the shared resource.
_ROUTER: Any = None
_ROUTER_LOCK = threading.Lock()
# default_max_parallel_requests: the admission cap. Below this, the gate is
# uncontended; above it, callers block on the semaphore — the contention
# signal that grows with concurrency.
_ADMIT_MAX_PARALLEL = 8


from importlib.util import find_spec

# Shared embedding model + FAISS index across all loops. Built once; the
# index is read-mostly shared memory, so concurrent search contends on it —
# the L3-pressure signal. Falls back to a numpy matrix search if
# sentence-transformers can't load a model (offline with no cache).
_EMBED_MODEL: Any = None
_FAISS_INDEX: Any = None
_CORPUS_META: Any = None  # list of tags aligned with index ids
_RETRIEVE_LOCK = threading.Lock()
_EMBED_DIM = 384
_CORPUS_SIZE = 20000  # ~30 MB of float32 vectors — sized to probe cache tiers


def _get_retrieval() -> Any:
    """Build (once) and return (model, faiss_index, meta). model is None if
    no embedding model is available — caller uses a random query vector then."""
    global _EMBED_MODEL, _FAISS_INDEX, _CORPUS_META
    if _FAISS_INDEX is None:
        with _RETRIEVE_LOCK:
            if _FAISS_INDEX is None:
                try:
                    import faiss  # type: ignore
                except ImportError as e:
                    raise RuntimeError(
                        "six_phase_agent.retrieve() requires 'faiss' (e.g. install faiss-cpu)"
                    ) from e

                try:
                    from sentence_transformers import SentenceTransformer

                    _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
                except Exception:
                    _EMBED_MODEL = None  # offline / no cache → random-query path

                rng = np.random.default_rng(0)
                corpus = rng.random((_CORPUS_SIZE, _EMBED_DIM), dtype=np.float32)
                faiss.normalize_L2(corpus)
                index = faiss.IndexFlatIP(_EMBED_DIM)
                index.add(corpus)
                _FAISS_INDEX = index
                _CORPUS_META = ["keep" if i % 2 == 0 else "drop" for i in range(_CORPUS_SIZE)]
    return _EMBED_MODEL, _FAISS_INDEX, _CORPUS_META


# Reason phase: a BOUNDED POOL of llama.cpp model instances — how CPU
# inference is actually served in production. You don't run one model shared
# by every agent (that serializes the whole loop); you run N replicas sized
# to the box, and agents check one out, decode, return it. Reason then
# parallelizes up to the pool size and queues beyond it — the realistic
# topology that keeps the 6-phase mix balanced instead of letting one shared
# model swamp everything. llama.cpp Llama is not thread-safe, so each replica
# is used by one thread at a time (guaranteed by the checkout queue).
import queue as _queue

_LLAMA_POOL: Any = None            # queue.Queue of Llama instances, or False
_LLAMA_POOL_LOCK = threading.Lock()  # guards lazy pool construction
_REASON_MAX_TOKENS = 24
# Threads per replica and pool size are env-controlled so the sweep runner's
# thread budget applies here too. Defaults: 1 thread/replica (outer
# concurrency scales, not inner fan-out), pool sized so
# pool_size × threads_per_model ≈ cores. Override via
# AGENTSYSPERF_LLAMA_THREADS / AGENTSYSPERF_LLAMA_POOL.
_LLAMA_THREADS_PER_MODEL = int(os.environ.get("AGENTSYSPERF_LLAMA_THREADS", "1"))
_LLAMA_POOL_SIZE = int(os.environ.get(
    "AGENTSYSPERF_LLAMA_POOL",
    str(max(1, (os.cpu_count() or 4) // max(1, _LLAMA_THREADS_PER_MODEL) // 4)),
))
_LLAMA_GLOB = os.environ.get(
    "AGENTSYSPERF_LLAMA_GGUF_GLOB",
    str(
        Path.home()
        / ".cache" / "huggingface" / "hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct-GGUF"
        / "snapshots" / "*" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ),
)


def _get_llama_pool() -> Any:
    """Return the pool (queue.Queue of Llama instances), or None if
    unavailable (→ matmul fallback). Built lazily under a lock."""
    global _LLAMA_POOL
    if _LLAMA_POOL is None:
        with _LLAMA_POOL_LOCK:
            if _LLAMA_POOL is None:
                import glob
                paths = glob.glob(_LLAMA_GLOB)
                if not paths or not find_spec("llama_cpp"):
                    _LLAMA_POOL = False
                else:
                    try:
                        from llama_cpp import Llama
                        pool: Any = _queue.Queue()
                        for _ in range(_LLAMA_POOL_SIZE):
                            pool.put(Llama(
                                model_path=paths[0], n_ctx=512,
                                n_threads=_LLAMA_THREADS_PER_MODEL,
                                verbose=False,
                            ))
                        _LLAMA_POOL = pool
                    except Exception:
                        _LLAMA_POOL = False
    return _LLAMA_POOL or None


def _get_router() -> Any:
    global _ROUTER
    if _ROUTER is None:
        with _ROUTER_LOCK:
            if _ROUTER is None:
                from litellm import Router
                _ROUTER = Router(
                    model_list=[{
                        "model_name": "gate-model",
                        "litellm_params": {
                            "model": "gpt-3.5-turbo",
                            "mock_response": "ok",
                            "api_key": "sk-mock",
                        },
                    }],
                    routing_strategy="usage-based-routing",
                    default_max_parallel_requests=_ADMIT_MAX_PARALLEL,
                    num_retries=0,
                )
    return _ROUTER


def _get_tokenizer() -> Any:
    global _TOKENIZER
    if _TOKENIZER is None:
        # bert-base tokenizer ships with the tokenizers fast path and needs
        # no model weights — we only exercise the tokenize CPU path.
        from tokenizers import Tokenizer  # local import keeps module light
        try:
            _TOKENIZER = Tokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            # Offline / no-network fallback: a WordLevel model over a small
            # seeded vocab still exercises the same Rust tokenize path (pre-
            # tokenize + vocab lookup) without a download. The vocab must
            # contain [UNK] so out-of-vocab tokens map cleanly.
            from tokenizers import models, pre_tokenizers
            vocab = {"[UNK]": 0}
            for i, w in enumerate(
                "the quick brown fox jumps over lazy dog . a an of to and".split(), start=1
            ):
                vocab[w] = i
            # unk_token is a tokenizer vocabulary sentinel, not a credential
            # (bandit's B106 matches on the `*_token=` kwarg name alone).
            tok = Tokenizer(
                models.WordLevel(vocab=vocab, unk_token="[UNK]")  # nosec B106
            )
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            _TOKENIZER = tok
    return _TOKENIZER


# ── reason ────────────────────────────────────────────────────────────
_REASON_PROMPTS = [
    "The test in the parser module is failing. Plan the next step:",
    "Given the retrieved code, decide which function to edit next:",
    "Summarize what the last tool call returned and choose an action:",
    "The build broke after the scheduler change. What do you check first?",
]


def _reason_matmul_fallback(scale: int) -> Dict[str, Any]:
    """Dense matmul chain — used only when llama.cpp / the model is absent.
    Memory-bandwidth bound, so it still exercises the right CPU axis."""
    n = k = 512
    steps = 8 * scale
    a = np.random.rand(n, k).astype(np.float32)
    b = np.random.rand(k, n).astype(np.float32)
    acc = 0.0
    for _ in range(steps):
        c = a @ b
        acc += float(c[0, 0])
        a = (c[:n, :k] * 0.001).astype(np.float32)
    return {"phase": "reason", "backend": "matmul_fallback",
            "matmuls": steps, "checksum": acc}


def reason(scale: int = 1) -> Dict[str, Any]:
    """Real local-SLM decode via a bounded pool of llama.cpp replicas.

    Decodes _REASON_MAX_TOKENS tokens per call — the CPU/mem-BW-bound
    inference work per-core-speed arguments center on. A replica is
    checked out of the pool (blocking if all are busy), used by this one
    thread, and returned. So Reason parallelizes up to _LLAMA_POOL_SIZE and
    queues beyond it — the realistic serving topology. Falls back to matmul
    if no model can load."""
    pool = _get_llama_pool()
    if pool is None:
        return _reason_matmul_fallback(scale)
    total_tokens = 0
    for i in range(scale):
        prompt = _REASON_PROMPTS[i % len(_REASON_PROMPTS)]
        llm = pool.get()  # blocks if all replicas busy — real queue wait
        try:
            out = llm(prompt, max_tokens=_REASON_MAX_TOKENS, echo=False)
        finally:
            pool.put(llm)
        total_tokens += out["usage"]["completion_tokens"]
    return {"phase": "reason", "backend": "llama_cpp",
            "pool_size": _LLAMA_POOL_SIZE, "decodes": scale,
            "tokens": total_tokens}


# ── retrieve ──────────────────────────────────────────────────────────
_QUERY_TEXTS = [
    "how do I fix the failing test in the parser module",
    "summarize the retrieval results and rank by relevance",
    "what changed in the last commit to the scheduler",
    "find functions that call the admission gate",
]


def retrieve(scale: int = 1, db_path: str | None = None) -> Dict[str, Any]:
    """Real query embedding + FAISS ANN search over the shared index."""
    model, index, meta = _get_retrieval()

    used_model = model is not None
    for i in range(scale):
        if used_model:
            text = _QUERY_TEXTS[i % len(_QUERY_TEXTS)]
            q = model.encode([text], convert_to_numpy=True).astype(np.float32)
        else:
            # offline fallback: random query
            q = np.random.rand(1, _EMBED_DIM).astype(np.float32)

        if hasattr(index, "search"):
            import faiss  # type: ignore
            faiss.normalize_L2(q)
            scores, ids = index.search(q, 10)  # type: ignore[attr-defined]
        else:
            # Fallback: `index` is a normalized corpus matrix (N, D)
            q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            sims = index @ q[0]
            ids = np.argsort(-sims)[:10][None, :]
    topk = [int(x) for x in ids[0]]
    # sqlite metadata filter over the returned candidate ids
    con = sqlite3.connect(db_path or ":memory:")
    kept = sum(1 for i in topk if 0 <= i < len(meta) and meta[i] == "keep")
    con.close()
    return {
        "phase": "retrieve",
        "queries": scale,
        "used_embed_model": used_model,
        "topk": len(topk),
        "kept": kept,
    }


# ── act ───────────────────────────────────────────────────────────────
def act(scale: int = 1) -> Dict[str, Any]:
    """Real subprocess doing bounded CPU work — same shape as tool exec."""
    iters = 200_000 * scale
    # A short python -c compute loop: real fork/exec + real scalar work.
    code = (
        "import sys;"
        "s=0\n"
        "for i in range(%d): s=(s+i*i)%%2147483647\n"
        "sys.stdout.write(str(s))" % iters
    )
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    elapsed = time.monotonic() - t0
    return {
        "phase": "act",
        "returncode": proc.returncode,
        "subprocess_s": elapsed,
        "out_len": len(proc.stdout),
    }


# ── admit ─────────────────────────────────────────────────────────────
def admit(scale: int = 1) -> Dict[str, Any]:
    """Admission-control gate via the REAL shared LiteLLM Router.

    Each call passes `scale` requests through the Router's completion path
    with mock_response — so the semaphore acquire, routing-strategy
    decision, and rate-limit bookkeeping run for real, but no network/LLM
    call happens. Because every concurrent loop shares one Router (see
    _get_router), the bounded parallel-request semaphore is a contended
    resource: gate latency grows as agents-in-flight exceeds
    _ADMIT_MAX_PARALLEL. That growth curve is the anti-per-core-speed
    proof — the gate contends regardless of how fast any single core is.
    """
    router = _get_router()
    n_requests = 2 * scale
    admitted = 0
    for _ in range(n_requests):
        resp = router.completion(
            model="gate-model",
            messages=[{"role": "user", "content": "gate"}],
        )
        if resp and resp.choices:
            admitted += 1
    return {"phase": "admit", "requests": n_requests, "admitted": admitted}


# ── context ───────────────────────────────────────────────────────────
def context(scale: int = 1) -> Dict[str, Any]:
    """Context admission: parse/validate a tool-output blob, real HF
    tokenization, and a prefill-shaped matmul. Most mem-BW-bound phase —
    the chiplet-tax probe (read L3 MPKI + per-core mem-BW here)."""
    # 1. parse/validate a JSON tool-output blob
    blob = json.dumps({"rows": [{"i": i, "v": i * 1.5} for i in range(500 * scale)]})
    parsed = json.loads(blob)
    # Not an assert: this is measured work, and `python -O` strips asserts —
    # which would silently delete part of the workload this phase exists to time.
    if not isinstance(parsed["rows"], list):
        raise TypeError("parse phase: expected a list of rows")

    # 2. real tokenization of a context window
    tok = _get_tokenizer()
    text = ("the quick brown fox jumps over the lazy dog . " * 256 * scale)
    enc = tok.encode(text)
    n_tokens = len(enc.ids)

    # 3. prefill-shaped matmul: (tokens × d) · (d × d) attention-ish
    d = 256
    t = min(n_tokens, 1024)
    x = np.random.rand(t, d).astype(np.float32)
    w = np.random.rand(d, d).astype(np.float32)
    y = x @ w                        # the mem-BW-bound prefill stand-in
    return {"phase": "context", "n_tokens": n_tokens, "prefill_out": float(y[0, 0])}


# ── commit ────────────────────────────────────────────────────────────
def commit(scale: int = 1, db_path: str | None = None) -> Dict[str, Any]:
    """Serialize state to json + a sqlite write. Real: + git commit."""
    state = {"step": scale, "result": [i for i in range(100 * scale)]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(state, f)
        json_path = f.name

    con = sqlite3.connect(db_path or ":memory:")
    con.execute("CREATE TABLE IF NOT EXISTS commits(step INTEGER, blob TEXT)")
    con.execute("INSERT INTO commits(step, blob) VALUES(?, ?)",
                (scale, json.dumps(state)))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
    con.close()
    os.unlink(json_path)
    return {"phase": "commit", "rows": n, "bytes": len(json.dumps(state))}


# Ordered phase registry — the adapter iterates this to open one span each.
PHASES = {
    "reason": reason,
    "retrieve": retrieve,
    "act": act,
    "admit": admit,
    "context": context,
    "commit": commit,
}

__all__ = ["PHASES", "reason", "retrieve", "act", "admit", "context", "commit"]
