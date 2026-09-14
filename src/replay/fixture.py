#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Fixture I/O + validation for the record/replay proxy.

A fixture is a JSONL file, one entry per recorded LLM turn. Each entry is the
unit the proxy serves on replay, keyed by ``(trial_key, turn)``:

    {
      "trial_key": "<16-hex sha256 of canonicalized first user message>",
      "turn": <int, count of assistant messages already in the request>,
      "latency_ms": <int, wall-time of the recorded upstream call>,
      "wants_stream": <bool, whether the recording client asked to stream>,
      "response": { ...full OpenAI chat.completion JSON... },
      "recorded_at": <int, unix seconds>
    }

The schema is intentionally response-only: the request is reconstructable from
the agent's deterministic message history during replay, so storing it would
just double the fixture size.

CRITICAL — fixtures are agent-specific. ``trial_key`` hashes the first user
message and ``turn`` counts assistant messages, so a fixture recorded by one
agent (e.g. Harbor/Terminus-2) will MISS on a different agent (e.g. AgentSysPerf's
own LiteLLM loop) whose prompts and turn structure differ. Record the fixture
with the EXACT agent you will replay, and run :func:`validate_fixture` plus a
one-task dry run before a full sweep.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# A fixture miss is PERMANENT: the file on disk does not grow mid-run, so the
# same (trial_key, turn) misses forever. The proxy must therefore report it with
# a NON-retryable status, and consumers must treat it as fatal for the trial.
#
# This was 503, which HTTP defines as transient and which every
# OpenAI-compatible client duly retried: a single miss at turn 0 became 200
# identical retries and the run exhausted max_turns before failing. 400 maps to
# BadRequestError, which no client retries.
#
# These live here rather than in proxy.py so the agent loop can import the
# marker without pulling in fastapi/uvicorn and proxy.py's import-time
# logging.basicConfig().
FIXTURE_MISS_STATUS = 400
FIXTURE_MISS_MARKER = "fixture miss"


@dataclass(frozen=True)
class FixtureStats:
    """Summary of a fixture's coverage, used for the pre-sweep quality gate."""

    path: Path
    n_entries: int
    n_trials: int
    avg_turns: float
    single_turn_trials: int
    trials: Dict[str, int]  # trial_key -> number of turns


def load_fixture(path: Path) -> Dict[str, Dict[int, dict]]:
    """Load a JSONL fixture into the in-memory index ``{trial_key: {turn: entry}}``.

    Skips blank and malformed lines (logged by the proxy at load time); this
    helper is the pure parse used by tooling and tests.
    """
    index: Dict[str, Dict[int, dict]] = defaultdict(dict)
    if not path.exists():
        return index
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                index[e["trial_key"]][e["turn"]] = e
            except (ValueError, KeyError):
                continue
    return index


def fixture_stats(path: Path) -> FixtureStats:
    """Compute coverage statistics for a fixture file."""
    index = load_fixture(path)
    trials = {k: len(v) for k, v in index.items()}
    n_trials = len(trials)
    n_entries = sum(trials.values())
    avg_turns = (n_entries / n_trials) if n_trials else 0.0
    single = sum(1 for n in trials.values() if n <= 1)
    return FixtureStats(
        path=path,
        n_entries=n_entries,
        n_trials=n_trials,
        avg_turns=avg_turns,
        single_turn_trials=single,
        trials=trials,
    )


def validate_fixture(
    path: Path,
    *,
    min_trials: int = 1,
    min_avg_turns: float = 1.0,
) -> Tuple[bool, List[str], FixtureStats]:
    """Gate a fixture before replay.

    Returns ``(ok, issues, stats)``. A thin fixture (few trials, ~1 turn each)
    usually means the record pass aborted early — replaying it would silently
    under-exercise the agent loop. Defaults are permissive (MVP records its own
    fixture); raise ``min_trials`` / ``min_avg_turns`` for production fixtures.
    """
    stats = fixture_stats(path)
    issues: List[str] = []
    if not path.exists():
        issues.append(f"fixture file {path} does not exist")
        return False, issues, stats
    if stats.n_entries == 0:
        issues.append("fixture is empty (no parseable entries)")
    if stats.n_trials < min_trials:
        issues.append(f"only {stats.n_trials} trials (want >= {min_trials})")
    if stats.avg_turns < min_avg_turns:
        issues.append(
            f"avg turns/trial {stats.avg_turns:.1f} (want >= {min_avg_turns}) — "
            f"record pass may have aborted early"
        )
    return len(issues) == 0, issues, stats


__all__ = ["FixtureStats", "load_fixture", "fixture_stats", "validate_fixture"]
