#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
OpenAI-compatible record/replay proxy for deterministic agentic benchmarks.

Extracted from agentic-benchmark-4 for local use. Eliminates LLM variance
when characterizing CPU behavior under agentic workloads.

Modes:
  off      Return empty assistant response instantly (no latency injection).
           Isolates orchestration + tool execution from LLM time.
  replay   Serve responses from fixture keyed by (trial_key, turn_index).
           Optional latency injection to preserve scheduling behavior.
  record   Forward to upstream, capture trajectory to fixture file.

Trial identification:
  trial_key  = sha256(first user message, canonicalized) truncated to 16 hex
  turn_index = count of assistant messages in request.messages

Run:
  # LLM OFF mode (fastest, characterize everything-but-inference):
  python replay_proxy.py --mode off --port 4001

  # Replay mode (deterministic LLM responses):
  python replay_proxy.py --mode replay --fixture /path/to/fixture.jsonl --port 4001

  # Record mode (capture new trajectory):
  python replay_proxy.py --mode record --upstream http://localhost:4000 --port 4001
"""
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('proxy')

app = FastAPI()

# Runtime state — set by CLI args
_MODE = 'off'
_UPSTREAM_URL = ''
_FIXTURE_PATH = None
_STRICT_MISS = True
_INJECT_LATENCY = False
_REPLAY_TRIAL = None  # If set, always serve from this trial key (flexible mode)

# In-memory fixture index: {trial_key: {turn_index: entry}}
_fixture = defaultdict(dict)
_write_lock = threading.Lock()

# Turn counter per trial for "off" mode (tracks conversation state)
_off_turn_counter = defaultdict(int)

# Canonicalization patterns — strip per-trial volatile strings so same-task
# prompts hash identically across replay hosts.
_CANON_STRIPS = [
    (re.compile(r'Current terminal state:.*', re.DOTALL), '[TERMINAL_STATE]'),
    (re.compile(r'[a-z]+@[a-f0-9]{8,}:[\S]*[#$]'), '[SHELL_PROMPT]'),
    (re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?'), '[TS]'),
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), '[UUID]'),
    (re.compile(r'__[A-Za-z0-9]{7}\b'), '__[SFX]'),
    # A canonicalization TOKEN hashed into trial_key, not a path on this host.
    # Must stay the literal "/tmp/[TMP]": rooting it at tempfile.gettempdir()
    # would change every trial_key under a non-default TMPDIR and miss every
    # fixture lookup. See src/replay/proxy.py for the full note.
    # hashed token, not a path
    (re.compile(r'/tmp/tmp[A-Za-z0-9_]+'), '/tmp/[TMP]'),  # nosec B108
]


def _canon(text: str) -> str:
    for pat, repl in _CANON_STRIPS:
        text = pat.sub(repl, text)
    return text


def _trial_key(messages: list) -> str:
    for m in messages:
        if m.get('role') == 'user':
            content = m.get('content', '')
            if isinstance(content, list):
                content = ''.join(p.get('text', '') for p in content if p.get('type') == 'text')
            return hashlib.sha256(_canon(content).encode('utf-8', 'replace')).hexdigest()[:16]
    return 'no_user_msg'


def _turn_index(messages: list) -> int:
    return sum(1 for m in messages if m.get('role') == 'assistant')


def _load_fixture(path: Path) -> int:
    if not path or not path.exists():
        log.warning(f'Fixture {path} not found; replay will miss on every call.')
        return 0
    n = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                _fixture[e['trial_key']][e['turn']] = e
                n += 1
            except Exception as ex:
                log.warning(f'Bad fixture line: {ex}')
    log.info(f'Loaded {n} entries across {len(_fixture)} trials from {path}')
    return n


def _empty_response(model: str = 'off-mode') -> dict:
    return {
        'id': 'chatcmpl-off',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': ''},
            'finish_reason': 'stop',
        }],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
    }


def _response_to_sse(resp: dict) -> StreamingResponse:
    cid = resp.get('id', 'chatcmpl-replay')
    created = resp.get('created', int(time.time()))
    model = resp.get('model', 'replay')
    choices_in = resp.get('choices', [])

    async def gen():
        for i, ch in enumerate(choices_in):
            role = ch.get('message', {}).get('role', 'assistant')
            yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": i, "delta": {"role": role}, "finish_reason": None}]})}\n\n'
        for i, ch in enumerate(choices_in):
            content = ch.get('message', {}).get('content', '') or ''
            if content:
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": i, "delta": {"content": content}, "finish_reason": None}]})}\n\n'
            tool_calls = ch.get('message', {}).get('tool_calls')
            if tool_calls:
                yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": i, "delta": {"tool_calls": tool_calls}, "finish_reason": None}]})}\n\n'
        for i, ch in enumerate(choices_in):
            yield f'data: {json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": i, "delta": {}, "finish_reason": ch.get("finish_reason", "stop")}]})}\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


# --- Mode handlers ---

async def _handle_off(body: dict):
    """Return empty response instantly. Agent will still execute tools."""
    wants_stream = bool(body.get('stream', False))
    trial_key = _trial_key(body.get('messages', []))
    turn = _turn_index(body.get('messages', []))
    log.info(f'OFF trial={trial_key} turn={turn}')
    resp = _empty_response()
    if wants_stream:
        return _response_to_sse(resp)
    return resp


async def _handle_replay(body: dict):
    wants_stream = bool(body.get('stream', False))
    trial_key = _trial_key(body.get('messages', []))
    turn = _turn_index(body.get('messages', []))

    # Flexible mode: serve from a specific fixture trial regardless of prompt hash
    if _REPLAY_TRIAL:
        entry = _fixture.get(_REPLAY_TRIAL, {}).get(turn)
        if entry is None:
            max_turn = max(_fixture.get(_REPLAY_TRIAL, {}).keys(), default=-1)
            log.warning(f'FLEX MISS trial={_REPLAY_TRIAL} turn={turn} (max_turn={max_turn})')
            if _STRICT_MISS:
                raise HTTPException(503, f'fixture miss: trial={_REPLAY_TRIAL} turn={turn}')
            resp = _empty_response()
        else:
            if _INJECT_LATENCY:
                await asyncio.sleep(entry['latency_ms'] / 1000.0)
            resp = entry['response']
            log.info(f'FLEX trial={_REPLAY_TRIAL} turn={turn} lat={entry["latency_ms"]}ms')
    else:
        entry = _fixture.get(trial_key, {}).get(turn)
        if entry is None:
            log.warning(f'MISS trial={trial_key} turn={turn} '
                        f'(have={sorted(_fixture.get(trial_key, {}).keys())})')
            if _STRICT_MISS:
                raise HTTPException(503, f'fixture miss: trial={trial_key} turn={turn}')
            resp = _empty_response()
        else:
            if _INJECT_LATENCY:
                await asyncio.sleep(entry['latency_ms'] / 1000.0)
            resp = entry['response']
            log.info(f'REPLAY trial={trial_key} turn={turn} lat={entry["latency_ms"]}ms')

    if wants_stream:
        return _response_to_sse(resp)
    return resp


async def _handle_record(body: dict, headers: dict):
    wants_stream = bool(body.get('stream', False))
    upstream_req = {**body, 'stream': False}
    upstream_req.pop('stream_options', None)
    upstream_req['temperature'] = 0.0

    url = f'{_UPSTREAM_URL}/v1/chat/completions'
    fwd_headers = {k: v for k, v in headers.items()
                   if k.lower() not in ('host', 'content-length', 'accept-encoding')}

    t0 = time.time()
    async with httpx.AsyncClient(timeout=900) as client:
        r = await client.post(url, json=upstream_req, headers=fwd_headers)
    latency_ms = int((time.time() - t0) * 1000)
    r.raise_for_status()
    resp = r.json()

    trial_key = _trial_key(body.get('messages', []))
    turn = _turn_index(body.get('messages', []))
    entry = {
        'trial_key': trial_key, 'turn': turn,
        'latency_ms': latency_ms, 'wants_stream': wants_stream,
        'response': resp, 'recorded_at': int(time.time()),
    }
    with _write_lock:
        _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _FIXTURE_PATH.open('a') as f:
            f.write(json.dumps(entry) + '\n')
        _fixture[trial_key][turn] = entry
    log.info(f'RECORD trial={trial_key} turn={turn} lat={latency_ms}ms')

    if wants_stream:
        return _response_to_sse(resp)
    return resp


# --- Endpoints ---

@app.get('/v1/models')
async def list_models():
    return {'object': 'list', 'data': [
        {'id': 'agentsysperf-proxy', 'object': 'model', 'owned_by': 'local'}
    ]}


@app.post('/v1/chat/completions')
async def chat_completions(req: Request):
    body = await req.json()
    headers = dict(req.headers)
    try:
        if _MODE == 'off':
            return await _handle_off(body)
        elif _MODE == 'replay':
            return await _handle_replay(body)
        elif _MODE == 'record':
            return await _handle_record(body, headers)
        else:
            raise HTTPException(500, f'unknown mode: {_MODE}')
    except HTTPException:
        raise
    except Exception as e:
        log.exception('proxy error')
        return JSONResponse(status_code=500, content={'error': {'message': str(e)}})


@app.get('/healthz')
async def healthz():
    return {
        'mode': _MODE,
        'fixture_trials': len(_fixture),
        'fixture_entries': sum(len(v) for v in _fixture.values()),
        'upstream': _UPSTREAM_URL if _MODE == 'record' else None,
    }


def main():
    global _MODE, _UPSTREAM_URL, _FIXTURE_PATH, _STRICT_MISS, _INJECT_LATENCY, _REPLAY_TRIAL

    p = argparse.ArgumentParser(description='AgentSysPerf replay proxy')
    p.add_argument('--mode', choices=['off', 'replay', 'record'], default='off')
    p.add_argument('--port', type=int, default=4001)
    # See src/replay/proxy.py: containerized agents reach this over the docker
    # bridge gateway, which a loopback bind does not answer. Fixtures only, no
    # credentials. Pass --host 127.0.0.1 for host-only agent runs.
    # container agents dial the bridge gateway
    p.add_argument('--host', default='0.0.0.0')  # nosec B104
    p.add_argument('--fixture', type=Path, default=None)
    p.add_argument('--upstream', default='http://127.0.0.1:4000')
    p.add_argument('--inject-latency', action='store_true', default=False)
    p.add_argument('--no-strict', action='store_true', default=False)
    p.add_argument('--replay-trial', default=None,
                   help='Serve responses from this specific fixture trial key '
                        '(ignores prompt hash matching). Use for flexible replay.')
    args = p.parse_args()

    _MODE = args.mode
    _UPSTREAM_URL = args.upstream.rstrip('/')
    _FIXTURE_PATH = args.fixture
    _STRICT_MISS = not args.no_strict
    _INJECT_LATENCY = args.inject_latency
    _REPLAY_TRIAL = args.replay_trial

    if _MODE in ('replay', 'record') and _FIXTURE_PATH:
        _load_fixture(_FIXTURE_PATH)

    if _REPLAY_TRIAL:
        if _REPLAY_TRIAL in _fixture:
            log.info(f'Flexible replay: serving from trial={_REPLAY_TRIAL} '
                     f'({len(_fixture[_REPLAY_TRIAL])} turns)')
        else:
            log.warning(f'Flexible replay: trial={_REPLAY_TRIAL} not found in fixture!')
            log.info(f'Available trials: {sorted(_fixture.keys())}')

    log.info(f'Starting proxy: mode={_MODE} port={args.port}')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
