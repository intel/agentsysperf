#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for the dashboard's missing-vs-zero readers (`_num` / `_fmt`).

`d.get("key", 0)` turns an absent measurement into a plausible reading. That has
produced three user-visible bugs: a throughput chart that plotted a flat line at
zero (wrong column name), num_commands stored as NULL for every run, and a TMA
panel that drew four empty bars from NaN. A missing measurement and a measured
zero are different findings and must render differently.

The helpers live in demo_app.py, which executes Streamlit page code at import,
so they are loaded by source extraction rather than imported.

Run: poetry run pytest src/dashboard/test_no_silent_zeros.py -q
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

_DEMO = Path(__file__).resolve().parents[2] / "demo_app.py"


def _load_helpers():
    """Exec just the helper block out of demo_app.py (no Streamlit at import)."""
    src = _DEMO.read_text()
    start = src.index("_MISSING = ")
    end = src.index("@st.cache_data", start)
    ns = {"json": json, "re": re, "time": time}
    exec(src[start:end], ns)
    return ns


@pytest.fixture(scope="module")
def helpers():
    return _load_helpers()


# ── _num: absence stays absent ──────────────────────────────────────────────

def test_missing_key_is_none_not_zero(helpers):
    assert helpers["_num"]({}, "throughput") is None


def test_explicit_null_is_none(helpers):
    assert helpers["_num"]({"throughput": None}, "throughput") is None


def test_measured_zero_survives(helpers):
    """The whole point: 0.0 is a reading and must NOT be confused with absence.

    An idle socket reading 0 MiB/s of memory bandwidth is a finding.
    """
    assert helpers["_num"]({"mem_bw": 0.0}, "mem_bw") == 0.0
    assert helpers["_num"]({"mem_bw": 0}, "mem_bw") == 0


def test_nan_is_treated_as_unmeasured(helpers):
    """NaN passes `is not None` — this is what drew four empty TMA bars."""
    assert helpers["_num"]({"tma": float("nan")}, "tma") is None


def test_numeric_strings_are_read(helpers):
    assert helpers["_num"]({"x": "12.5"}, "x") == 12.5


def test_unparseable_value_is_none(helpers):
    assert helpers["_num"]({"x": "banana"}, "x") is None
    assert helpers["_num"]({"x": []}, "x") is None


def test_bool_is_not_a_measurement(helpers):
    """bool is an int in Python; True must not silently become 1.0."""
    assert helpers["_num"]({"x": True}, "x") is None


def test_none_row_is_safe(helpers):
    assert helpers["_num"](None, "x") is None


def test_default_is_honoured(helpers):
    assert helpers["_num"]({}, "x", 7) == 7


def test_reads_object_attributes_too(helpers):
    class _Row:
        throughput = 3.5
    assert helpers["_num"](_Row(), "throughput") == 3.5


# ── _fmt: absence renders as an em-dash ─────────────────────────────────────

def test_fmt_missing_renders_em_dash(helpers):
    assert helpers["_fmt"]({}, "x") == "—"
    assert helpers["_fmt"]({"x": None}, "x") == "—"
    assert helpers["_fmt"]({"x": float("nan")}, "x") == "—"


def test_fmt_zero_renders_zero_not_em_dash(helpers):
    assert helpers["_fmt"]({"x": 0.0}, "x") == "0.0"


def test_fmt_respects_format_and_scale(helpers):
    assert helpers["_fmt"]({"x": 2.345}, "x", fmt="{:.2f}") == "2.35"
    assert helpers["_fmt"]({"x": 2048}, "x", fmt="{:.0f}", scale=1 / 1024) == "2"


# ── the keys that actually broke ────────────────────────────────────────────

@pytest.mark.parametrize("key", [
    "aggregate_throughput_turns_per_s",
    "mean_throughput_turns_per_s",
    "total_duration_s",
    "emon_metrics_count",
    "logical_cpus",
    "throughput_trials_per_min",   # the wrong column name behind the flat line
])
def test_known_absent_keys_do_not_become_zero(helpers, key):
    """Every one of these was read with `.get(key, 0)` against data lacking it."""
    assert helpers["_num"]({"density": 1.0}, key) is None


def test_demo_app_no_longer_defaults_these_keys_to_zero():
    """Guard the source itself: the old pattern must not creep back in."""
    src = _DEMO.read_text()
    offenders = re.findall(
        r'\.get\(\s*"(aggregate_throughput_turns_per_s|mean_throughput_turns_per_s'
        r'|total_duration_s|emon_metrics_count|logical_cpus'
        r'|throughput_trials_per_min)"\s*,\s*0(?:\.0)?\s*\)',
        src,
    )
    assert not offenders, f"silent-zero reads reintroduced: {sorted(set(offenders))}"
