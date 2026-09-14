#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""OpenAI-compatible record/replay proxy for deterministic agentic benchmarks.

This is the data-plane seam that isolates the Xeon silicon signal from LLM
variance. An agent (Harbor/Terminus-2 in a container, or AgentSysPerf's own
LiteLLM loop) is pointed at this proxy via ``OPENAI_API_BASE``. The proxy
either forwards to a live upstream and captures the trajectory (record), or
serves byte-identical responses from a fixture (replay) so run-over-run
variance drops from ~±15% to ~±2% and the silicon difference becomes visible.

Promoted from ``harness/scripts/replay_proxy.py`` (colleague's AWS-free local
port of agentic-benchmark-4) into the package so the sweep runner and the
:class:`~src.replay.manager.ReplayProxy` manager can drive it directly.

Modes:
  off      Return an empty assistant response instantly (no upstream, no
           latency). Isolates orchestration + tool execution from LLM time.
  replay   Serve responses from a fixture keyed by (trial_key, turn_index).
           Optional latency injection preserves wave-scheduling behaviour.
  record   Forward to upstream, capture the trajectory to a fixture file.

Trial identification (the determinism mechanism):
  trial_key  = sha256(first user message, canonicalized) truncated to 16 hex.
               Canonicalization strips per-trial volatile strings (hostnames,
               timestamps, UUIDs, tmp dirs) so the SAME task hashes identically
               across replay hosts.
  turn_index = count of assistant messages already in request.messages.

Run standalone:
  python -m src.replay.proxy --mode off --port 4001
  python -m src.replay.proxy --mode replay --fixture fixture.jsonl --port 4001
  python -m src.replay.proxy --mode record --upstream http://localhost:4000 --port 4001
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

from src.replay.fixture import FIXTURE_MISS_MARKER, FIXTURE_MISS_STATUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("src.replay.proxy")

app = FastAPI()

# Runtime state — set by main() from CLI args (or env when launched by the
# ReplayProxy manager as a subprocess).
_MODE = "off"
_UPSTREAM_URL = ""
_FIXTURE_PATH: Path | None = None
_STRICT_MISS = True
_INJECT_LATENCY = False
_REPLAY_TRIAL = None  # If set, always serve from this trial key (flexible mode).

# In-memory fixture index: {trial_key: {turn_index: entry}}
_fixture = defaultdict(dict)
_write_lock = threading.Lock()

# Canonicalization patterns — strip per-trial volatile strings so same-task
# prompts hash identically across replay hosts.
#
# CRITICAL — these replacement strings are part of what gets hashed into the
# trial_key, so they MUST match byte-for-byte the proxy that RECORDED the
# fixture being replayed. These are the canonical strip patterns for
# Terminal-Bench; other benchmarks use pluggable keyers (see keying.py).
_CANON_STRIPS = [
    (re.compile(r"Current terminal state:.*", re.DOTALL), "[TERMINAL_STATE_STRIPPED]"),
    (re.compile(r"[a-z]+@[a-f0-9]{8,}:[\S]*[#$]"), "[SHELL_PROMPT]"),
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "[TIMESTAMP]"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), "[UUID]"),
    (re.compile(r"__[A-Za-z0-9]{7}\b"), "__[TRIAL_SFX]"),
    # NOT a filesystem path on this host, and deliberately NOT rooted at
    # tempfile.gettempdir(): this is a canonicalization TOKEN that is hashed
    # into trial_key, and it matches tmpdir paths that appeared in the
    # RECORDED transcript (inside the agent's container, where /tmp is /tmp).
    # Making it depend on this host's TMPDIR would change every trial_key and
    # miss every fixture lookup. Leave the literal alone. See the CRITICAL note
    # at the top of _CANON_STRIPS.
    # hashed token, not a path
    (re.compile(r"/tmp/tmp[A-Za-z0-9_]+"), "/tmp/[TMP]"),  # nosec B108
]


def _canon(text: str) -> str:
    for pat, repl in _CANON_STRIPS:
        text = pat.sub(repl, text)
    return text


def _miss_detail(trial_key: str, turn: int, *, flexible: bool = False) -> str:
    """Build a miss message that names what was asked for and what exists.

    The trial_key alone is not actionable — it is a hash. Listing the turns held
    for that trial separates "wrong task" (trial absent entirely) from "ran off
    the end of a recorded trajectory" (trial present, turn beyond its last).
    """
    have = sorted(_fixture.get(trial_key, {}).keys())
    if not have:
        known = sorted(_fixture)
        detail = (f"trial not in fixture at all; {len(known)} trial(s) loaded"
                  f" from {_FIXTURE_PATH}: {known}")
    else:
        detail = f"trial has turns {have[0]}..{have[-1]} ({len(have)} entries), asked for {turn}"
    prefix = "FLEX " if flexible else ""
    return (f"{prefix}{FIXTURE_MISS_MARKER}: trial={trial_key} turn={turn} — {detail}. "
            f"This is permanent, not transient: retrying cannot help.")


def _trial_key(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
            return hashlib.sha256(_canon(content).encode("utf-8", "replace")).hexdigest()[:16]
    return "no_user_msg"


def _turn_index(messages: list) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def _load_fixture(path: Path | None) -> int:
    if not path or not path.exists():
        log.warning(f"Fixture {path} not found; replay will miss on every call.")
        return 0
    n = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                _fixture[e["trial_key"]][e["turn"]] = e
                n += 1
            except Exception as ex:
                log.warning(f"Bad fixture line: {ex}")
    log.info(f"Loaded {n} entries across {len(_fixture)} trials from {path}")
    return n


def _empty_response(model: str = "off-mode") -> dict:
    return {
        "id": "chatcmpl-off",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _response_to_sse(resp: dict) -> StreamingResponse:
    cid = resp.get("id", "chatcmpl-replay")
    created = resp.get("created", int(time.time()))
    model = resp.get("model", "replay")
    choices_in = resp.get("choices", [])

    async def gen():
        for i, ch in enumerate(choices_in):
            role = ch.get("message", {}).get("role", "assistant")
            yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": i, "delta": {"role": role}, "finish_reason": None}]})}\n\n'
        for i, ch in enumerate(choices_in):
            content = ch.get("message", {}).get("content", "") or ""
            if content:
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": i, "delta": {"content": content}, "finish_reason": None}]})}\n\n'
            tool_calls = ch.get("message", {}).get("tool_calls")
            if tool_calls:
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": i, "delta": {"tool_calls": tool_calls}, "finish_reason": None}]})}\n\n'
        for i, ch in enumerate(choices_in):
            yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": i, "delta": {}, "finish_reason": ch.get("finish_reason", "stop")}]})}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# --- Mode handlers ---

async def _handle_off(body: dict):
    """Return an empty response instantly. The agent still executes tools."""
    wants_stream = bool(body.get("stream", False))
    trial_key = _trial_key(body.get("messages", []))
    turn = _turn_index(body.get("messages", []))
    log.info(f"OFF trial={trial_key} turn={turn}")
    resp = _empty_response()
    if wants_stream:
        return _response_to_sse(resp)
    return resp


async def _handle_replay(body: dict):
    wants_stream = bool(body.get("stream", False))
    trial_key = _trial_key(body.get("messages", []))
    turn = _turn_index(body.get("messages", []))

    # Flexible mode: serve from a specific fixture trial regardless of prompt hash.
    if _REPLAY_TRIAL:
        entry = _fixture.get(_REPLAY_TRIAL, {}).get(turn)
        if entry is None:
            detail = _miss_detail(_REPLAY_TRIAL, turn, flexible=True)
            log.error(detail)
            if _STRICT_MISS:
                raise HTTPException(FIXTURE_MISS_STATUS, detail)
            resp = _empty_response()
        else:
            if _INJECT_LATENCY:
                await asyncio.sleep(entry["latency_ms"] / 1000.0)
            resp = entry["response"]
            log.info(f'FLEX trial={_REPLAY_TRIAL} turn={turn} lat={entry["latency_ms"]}ms')
    else:
        entry = _fixture.get(trial_key, {}).get(turn)
        if entry is None:
            detail = _miss_detail(trial_key, turn)
            log.error(detail)
            if _STRICT_MISS:
                raise HTTPException(FIXTURE_MISS_STATUS, detail)
            resp = _empty_response()
        else:
            if _INJECT_LATENCY:
                await asyncio.sleep(entry["latency_ms"] / 1000.0)
            resp = entry["response"]
            log.info(f'REPLAY trial={trial_key} turn={turn} lat={entry["latency_ms"]}ms')

    if wants_stream:
        return _response_to_sse(resp)
    return resp


async def _handle_record(body: dict, headers: dict):
    wants_stream = bool(body.get("stream", False))
    upstream_req = {**body, "stream": False}
    upstream_req.pop("stream_options", None)
    upstream_req["temperature"] = 0.0  # deterministic sampling on the record pass

    # Build the upstream URL without doubling /v1: if the configured upstream
    # already ends in /v1 (e.g. an OpenAI-compatible gateway base URL), append
    # only /chat/completions; otherwise add the full /v1/chat/completions.
    base = _UPSTREAM_URL
    url = (f"{base}/chat/completions" if base.endswith("/v1")
           else f"{base}/v1/chat/completions")
    fwd_headers = {k: v for k, v in headers.items()
                   if k.lower() not in ("host", "content-length", "accept-encoding")}
    # The agent only has the proxy's dummy key. Replace Authorization with the
    # REAL upstream credential (set on the proxy process env by the run driver),
    # so record-mode reaches an authenticated gateway (e.g. Bedrock).
    _upstream_key = os.environ.get("AGENTSYSPERF_UPSTREAM_KEY") or os.environ.get("OPENAI_API_KEY")
    if _upstream_key:
        fwd_headers = {k: v for k, v in fwd_headers.items() if k.lower() != "authorization"}
        fwd_headers["Authorization"] = f"Bearer {_upstream_key}"

    t0 = time.time()
    # trust_env=True (default): the upstream LLM gateway may be an EXTERNAL host
    # reachable ONLY via the corporate HTTP(S)_PROXY (e.g. Bedrock on this
    # network has no direct egress). NO_PROXY must list localhost so the
    # inbound agent->proxy hop is NOT proxied, but the proxy->upstream hop is.
    async with httpx.AsyncClient(timeout=900) as client:
        r = await client.post(url, json=upstream_req, headers=fwd_headers)
    latency_ms = int((time.time() - t0) * 1000)
    r.raise_for_status()
    resp = r.json()

    trial_key = _trial_key(body.get("messages", []))
    turn = _turn_index(body.get("messages", []))
    entry = {
        "trial_key": trial_key, "turn": turn,
        "latency_ms": latency_ms, "wants_stream": wants_stream,
        "response": resp, "recorded_at": int(time.time()),
    }
    with _write_lock:
        _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _FIXTURE_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        _fixture[trial_key][turn] = entry
    log.info(f"RECORD trial={trial_key} turn={turn} lat={latency_ms}ms")

    if wants_stream:
        return _response_to_sse(resp)
    return resp


# --- Endpoints ---

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "agentsysperf-proxy", "object": "model", "owned_by": "local"}
    ]}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    headers = dict(req.headers)
    try:
        if _MODE == "off":
            return await _handle_off(body)
        elif _MODE == "replay":
            return await _handle_replay(body)
        elif _MODE == "record":
            return await _handle_record(body, headers)
        else:
            raise HTTPException(500, f"unknown mode: {_MODE}")
    except HTTPException:
        raise
    except Exception as e:
        log.exception("proxy error")
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})


@app.get("/healthz")
async def healthz():
    return {
        "mode": _MODE,
        "fixture_trials": len(_fixture),
        "fixture_entries": sum(len(v) for v in _fixture.values()),
        "upstream": _UPSTREAM_URL if _MODE == "record" else None,
    }


def main(argv: list[str] | None = None) -> None:
    global _MODE, _UPSTREAM_URL, _FIXTURE_PATH, _STRICT_MISS, _INJECT_LATENCY, _REPLAY_TRIAL

    p = argparse.ArgumentParser(description="AgentSysPerf record/replay proxy")
    p.add_argument("--mode", choices=["off", "replay", "record"], default="off")
    p.add_argument("--port", type=int, default=4001)
    # All interfaces by default because the agent under test frequently runs
    # inside a container and reaches this proxy over the docker bridge gateway,
    # which a loopback bind does not answer. It serves recorded LLM fixtures and
    # holds no credentials. Pass --host 127.0.0.1 for host-only agent runs.
    # container agents dial the bridge gateway
    p.add_argument("--host", default="0.0.0.0")  # nosec B104
    p.add_argument("--fixture", type=Path, default=None)
    p.add_argument("--upstream", default="http://127.0.0.1:4000")
    p.add_argument("--inject-latency", action="store_true", default=False)
    p.add_argument("--no-strict", action="store_true", default=False)
    p.add_argument("--replay-trial", default=None,
                   help="Serve responses from this specific fixture trial key "
                        "(ignores prompt hash matching). Use for flexible replay.")
    args = p.parse_args(argv)

    _MODE = args.mode
    _UPSTREAM_URL = args.upstream.rstrip("/")
    _FIXTURE_PATH = args.fixture
    _STRICT_MISS = not args.no_strict
    _INJECT_LATENCY = args.inject_latency
    _REPLAY_TRIAL = args.replay_trial

    if _MODE in ("replay", "record") and _FIXTURE_PATH:
        _load_fixture(_FIXTURE_PATH)

    if _REPLAY_TRIAL:
        if _REPLAY_TRIAL in _fixture:
            log.info(f"Flexible replay: serving from trial={_REPLAY_TRIAL} "
                     f"({len(_fixture[_REPLAY_TRIAL])} turns)")
        else:
            log.warning(f"Flexible replay: trial={_REPLAY_TRIAL} not found in fixture!")
            log.info(f"Available trials: {sorted(_fixture.keys())}")

    log.info(f"Starting proxy: mode={_MODE} port={args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
