#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
AgentSysPerf Demo Dashboard
=========================
Run with: streamlit run demo_app.py --server.port 7860 --server.address 0.0.0.0
Service URLs are derived from the host (see _service_host), so this works on any
box — set AGENTSYSPERF_HOST to override the advertised hostname.
"""

import os
import socket
import sys
from pathlib import Path as _Path
# Make the repo importable when running this script directly, without a
# hardcoded absolute path (works for any checkout location).
sys.path.insert(0, str(_Path(__file__).resolve().parent))

import json
import re
import time
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="AgentSysPerf Demo",
    page_icon="⚡",
    layout="wide",
)


def _service_host() -> str:
    """Host to advertise for this box's own services (demo/live/grafana).

    Derived, not hardcoded: AGENTSYSPERF_HOST override > FQDN > hostname. The
    dashboard runs ON the host it links to, so the running box IS the answer.
    """
    h = os.environ.get("AGENTSYSPERF_HOST")
    if h:
        return h
    try:
        fqdn = socket.getfqdn()
        if fqdn and fqdn != "localhost":
            return fqdn
    except Exception:
        pass
    return socket.gethostname()


def _langfuse_host() -> str:
    """Langfuse may live on ANOTHER box — prefer the real $LANGFUSE_HOST, else
    fall back to this host's :7862."""
    return os.environ.get("LANGFUSE_HOST") or f"http://{_service_host()}:7862"


def _emon_analyzer():
    """Resolve the EMON analyzer instance from the plugin registry, or None.

    EMON left the vendor-neutral core in the 0.1.0 split; it is restored by the
    Intel-only `agentsysperf-emon` plugin, which re-registers the `emon`
    analyzer entry point. So "Hardware Analysis" keeps its place in the nav but
    only lights up its EmonAnalyzer findings when that plugin is installed.
    """
    try:
        from src.protocols import discover_analyzers
        return discover_analyzers().get("emon")
    except Exception:
        return None


_HOST = _service_host()

# ─── Load Data ───────────────────────────────────────────────────────────

@st.cache_data
def load_platform():
    from src.platform import detect_platform
    p = detect_platform()
    return {
        "model_name": p.model_name,
        "microarchitecture": p.microarchitecture,
        "physical_cores": p.physical_cores,
        "logical_cpus": p.logical_cpus,
        "numa_nodes": p.numa_nodes,
        "l3_per_node_mb": p.l3_per_node // (1024 * 1024),
        "l3_budget_mb": p.l3_budget_mb,
        "dram_bw_per_node_gbs": round(p.dram_bw_per_node_gbs, 1),
        "dram_bw_total_gbs": round(p.dram_bw_total_gbs, 1),
        "dram_bw_source": p.dram_bw_source,
        "has_amx": p.has_amx,
        "has_avx512": p.has_avx512,
        "vendor": p.vendor,
    }


@st.cache_data
def load_preflight():
    from src.preflight import SystemCheck
    check = SystemCheck()
    report = check.run()
    return {
        "checks": [
            {"name": c.name, "status": c.status.value, "current": c.current_value, "expected": c.expected_value}
            for c in report.checks
        ],
        "passed": report.passed_count,
        "total": len(report.checks),
    }


# Source records via the data-access layer (store-first, verbatim JSON
# fallback) instead of a hardcoded path. The transform logic below is
# unchanged — only the SOURCE of `raw` moved. See src/dashboard_data.py.
import src.dashboard_data as _data
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

_SYNTHETIC_FALLBACK = [Path(f"{_TMP}/agentsysperf_scratch/synthetic_l1_l3")]


# ── Provenance labelling (P1) ────────────────────────────────────────────────
# Panels on one page can describe unrelated experiments. The landing page showed
# a 10-task LLM correctness run in its KPIs and, directly below, a scaling curve
# from ONE task (circuit-fibsqrt) driven by the LLM-free `oracle` agent on a
# 16-core cpuset — with nothing saying so. Every data panel states what produced
# it, so no chart can be read as describing a different experiment.

def _prov_age(ts) -> str:
    """Human age of an epoch-seconds timestamp, or '' if unusable."""
    if not ts:
        return ""
    try:
        age = time.time() - float(ts)
    except (TypeError, ValueError):
        return ""
    if age < 0:
        return ""
    if age < 3600:
        return f"{age / 60:.0f} min ago"
    if age < 86400:
        return f"{age / 3600:.0f} h ago"
    return f"{age / 86400:.0f} d ago"


def _prov_caption(*parts: str) -> str:
    """Join provenance fragments, dropping empties, as one caption line."""
    return " · ".join(p for p in parts if p)


def _prov_model(model: str) -> str:
    """Shorten a provider-qualified model id for a caption.

    ``bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0`` is unreadable
    inline; keep the family and version, drop the routing prefix and date.
    """
    if not model:
        return "?"
    short = model.rsplit("/", 1)[-1]
    for prefix in ("us.anthropic.", "anthropic.", "us."):
        if short.startswith(prefix):
            short = short[len(prefix):]
    # Trim a trailing Bedrock date/revision suffix (…-20250929-v1:0).
    return re.sub(r"-\d{8}(-v\d+[:.]\d+)?$", "", short)


# ── No silent zeros (P4) ─────────────────────────────────────────────────────
# `d.get("key", 0)` turns a missing key into a plausible-looking measurement.
# This has produced three user-visible bugs already:
#   * throughput_trials_per_min — the stored column is throughput_per_min, so
#     every point plotted as 0 and the headline chart was a flat line reading as
#     "throughput does not scale on this box"
#   * num_commands — never written, so NULL for every run ever stored, reading as
#     "the agent ran no commands" while 292 command spans existed
#   * metric_TMA_* — all-NaN on a SKU with no TMA nodes, drawn as four empty bars
# A missing measurement and a measured zero are different findings. Read through
# _num()/_fmt() so absence stays visible.

_MISSING = "—"


def _num(row, key, default=None):
    """Read a numeric field, returning None (not 0) when it is absent or null.

    Use for anything that will be plotted, summed, or averaged. A None keeps the
    point out of the series instead of pinning it to the axis.
    """
    if row is None:
        return default
    val = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
    if val is None:
        return default
    if isinstance(val, bool):  # bool is an int; almost never the intent here
        return default
    if isinstance(val, (int, float)):
        # NaN passes `is not None` but is not a reading — see the TMA panel.
        return default if val != val else val
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _fmt(row, key, *, fmt="{:.1f}", scale=1.0, missing=_MISSING):
    """Format a numeric field for display, or the em-dash when not measured."""
    val = _num(row, key)
    if val is None:
        return missing
    return fmt.format(val * scale)


def _jmeta(row: dict) -> dict:
    """Return a row's ``metadata`` as a dict, whether stored as JSON or dict."""
    raw = (row or {}).get("metadata")
    if isinstance(raw, str):
        try:
            return json.loads(raw) or {}
        except ValueError:
            return {}
    return raw or {}


def _sweep_provenance(meta: dict, points: list) -> str:
    """Describe a sweep: task(s), agent, density basis, size, age.

    The task name and cpuset live in the SWEEP row's metadata (key ``task``),
    while per-cell fields like cores_in_cpuset live on the points, so both are
    consulted. Note the sweep row stores the agent under ``model``.
    """
    smd = _jmeta(meta)
    pmd = _jmeta(points[0]) if points else {}
    task = (smd.get("task") or smd.get("task_id")
            or pmd.get("task") or pmd.get("task_id") or "?")
    agent = (meta.get("agent") or smd.get("agent")
             or meta.get("model") or pmd.get("agent") or "?")
    basis = meta.get("vcpu_basis") or pmd.get("cores_in_cpuset")
    cpuset = smd.get("cpuset") or pmd.get("cpuset")
    reps = len({p.get("replicate") for p in points}) if points else 0
    agent_note = f"agent `{agent}`"
    if agent == "oracle":
        # The single most misread fact on the page: no model is involved.
        agent_note += " (runs the task's solve.sh — NO LLM)"
    return _prov_caption(
        f"task `{task}`",
        agent_note,
        f"{len(points)} points, {reps} replicate(s)",
        f"basis {basis} cores" + (f" (cpuset {cpuset})" if cpuset else "") if basis else "",
        _prov_age(meta.get("created_at")),
    )

# Map the sidebar's display labels to store benchmark_id slugs. The page
# dispatch keys stay the display labels (unchanged); only store lookups use the
# slug. (Resolves the slug-vs-label hazard flagged in the storage plan.)
_BENCH_SLUG = {
    "Terminal-Bench": "terminal-bench",
    "Tau-Bench": "tau-bench",
    "SWE-Bench": "swe-bench",
}


@st.cache_data
def load_benchmarks():
    raw = _data.get_records("terminal-bench", fallback_paths=_SYNTHETIC_FALLBACK)
    if not raw:
        return []

    spans = {}
    for r in raw:
        sid = r["span_id"]
        if sid not in spans:
            spans[sid] = {}
        spans[sid][r["layer"]] = r["payload"]

    benchmarks = []
    for sid, layers in spans.items():
        task = sid.split("::")[-1]
        l1 = layers.get("l1", {})
        l3 = layers.get("l3", {})
        benchmarks.append({
            "task": task,
            "duration_ms": round(l1.get("duration_us", 0) / 1000, 0),
            "cpu_time_s": round(l1.get("cpu_time_s", 0), 2),
            "cpu_pct": round(l1.get("cpu_pct_mean", 0), 1),
            "rss_mb": round(l1.get("rss_kb_peak", 0) / 1024, 0),
            "ipc": round(l3.get("ipc", 0), 2),
            "cache_miss_pct": round(l3.get("cache_miss_pct", 0), 1),
        })

    return benchmarks


@st.cache_data
def load_analysis():
    from src.analyzers.memory_bandwidth import MemoryBandwidthAnalyzer
    from src.protocols import MeasurementRecord

    raw = _data.get_records("terminal-bench", fallback_paths=_SYNTHETIC_FALLBACK)
    if not raw:
        return []

    records = [MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"]) for r in raw]

    analyzer = MemoryBandwidthAnalyzer()
    results = []
    for r in analyzer.analyze(records):
        task = r.span_id.split("::")[-1]
        patterns = r.evidence.get("patterns_matched", [])
        solutions = r.evidence.get("solutions", [])
        results.append({
            "task": task,
            "verdict": r.verdict,
            "confidence": round(r.confidence * 100),
            "patterns": [p["pattern"] for p in patterns],
            "solutions": [{"name": s["name"], "category": s["category"], "impact": s["expected_impact"]} for s in solutions[:4]],
        })

    return results


# ─── Sidebar ─────────────────────────────────────────────────────────────

st.sidebar.title("AgentSysPerf Demo")
st.sidebar.markdown("---")

_NAV_PAGES = {
    "Dashboard": [],
    "Platform": [
        "Platform Discovery",
        "Pre-flight Validation",
        "Architecture",
    ],
    "Terminal-Bench": [
        "Hardware Baselines",
        "Agent Performance",
        "Hardware Analysis",
        "Scaling",
        "Recommendations",
    ],
    "Tau-Bench": [
        "Agent Performance",
        "Hardware Analysis",
        "Scaling",
        "Recommendations",
    ],
    "SWE-Bench": [
        "Agent Performance",
        "Hardware Analysis",
        "Scaling",
        "Recommendations",
    ],
    "Observability": [
        "Langfuse Traces",
        "Grafana Dashboard",
    ],
    "Workflows": [
        "Spec Decode Workflow",
    ],
}

# Tree-style navigation with +/- expand/collapse
st.sidebar.markdown(
    """<style>
    div[data-testid="stSidebar"] [data-testid="stButton"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 2px 8px !important;
        min-height: 1.8rem !important;
        font-family: monospace;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: rgba(151, 166, 195, 0.15) !important;
    }
    div[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        background: rgba(255, 75, 75, 0.1) !important;
        color: #ff4b4b !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# Initialize session state for expanded sections and active page
if "nav_expanded" not in st.session_state:
    st.session_state.nav_expanded = {"Platform": True}
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Dashboard"

_sections = list(_NAV_PAGES.keys())

for _section in _sections:
    _pages = _NAV_PAGES[_section]

    if not _pages:
        # Standalone leaf item (no children) — clicking navigates directly
        _is_active = st.session_state.nav_page == _section
        _prefix = "▸ " if _is_active else "  "
        if st.sidebar.button(
            f"{_prefix}{_section}",
            key=f"nav_sec_{_section}",
            use_container_width=True,
            type="primary" if _is_active else "secondary",
        ):
            st.session_state.nav_page = _section
            st.rerun()
        continue

    _is_expanded = st.session_state.nav_expanded.get(_section, False)
    _toggle_icon = "−" if _is_expanded else "+"

    if st.sidebar.button(
        f"{_toggle_icon}  {_section}",
        key=f"nav_sec_{_section}",
        use_container_width=True,
    ):
        st.session_state.nav_expanded[_section] = not _is_expanded
        _is_expanded = not _is_expanded
        st.rerun()

    if _is_expanded:
        for _page in _pages:
            _page_key = f"{_section}::{_page}"
            _is_active = st.session_state.nav_page == _page_key
            _prefix = "▸ " if _is_active else "  "

            if st.sidebar.button(
                f"{_prefix}{_page}",
                key=f"nav_pg_{_page_key}",
                use_container_width=True,
                type="secondary" if not _is_active else "primary",
            ):
                st.session_state.nav_page = _page_key
                st.rerun()

demo_page = st.session_state.nav_page
_nav_section = demo_page.split("::")[0]

st.sidebar.markdown("---")
# Derive the System/Arch/Cores from the live platform (was hardcoded to a
# specific GNR box). detect_platform() is the source of truth — same data the
# Platform Discovery page shows.
_plat = load_platform()
_sys_name = _plat["model_name"].replace("Intel(R) Xeon(R) ", "").replace("INTEL(R) XEON(R) ", "").strip()
st.sidebar.markdown(
    f"<small><b>System</b>: {_sys_name}<br>"
    f"<b>Arch</b>: {_plat['microarchitecture']}<br>"
    f"<b>Cores</b>: {_plat['physical_cores']} ({_plat['logical_cpus']} logical)</small>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Services**")
st.sidebar.markdown(f"[Demo UI](http://{_HOST}:7860) · [Live](http://{_HOST}:7861)")
st.sidebar.markdown(f"[Langfuse]({_langfuse_host()}) · [Grafana](http://{_HOST}:3000)")

# ─── Pages ───────────────────────────────────────────────────────────────

if demo_page == "Dashboard":
    _plat_info = load_platform()
    _cpu_short = _plat_info["model_name"].replace("Intel(R) Xeon(R) ", "").replace("INTEL(R) XEON(R) ", "").strip()
    _total_l3 = _plat_info["l3_per_node_mb"] * _plat_info["numa_nodes"]

    st.title("AgentSysPerf Dashboard")
    st.caption(
        f"{_cpu_short} ({_plat_info['microarchitecture']}) · "
        f"{_plat_info['physical_cores']} cores · {_plat_info['numa_nodes']} NUMA nodes · "
        f"{_total_l3} MB L3"
    )

    # ─── Top-level KPIs ──────────────────────────────────────────────────
    # Gather data from store + filesystem
    from src.storage.sqlite_store import SQLiteResultStore
    _dash_store = None
    _dash_runs = []
    _dash_sweeps = []
    try:
        _dash_store = SQLiteResultStore.open(read_only=True)
        _dash_runs = _dash_store.list_runs(limit=100)
        _dash_sweeps = _dash_store.query_sweeps()
    except Exception:
        pass

    _dash_benchmarks_with_data = set()
    _TB2_DIRS = [Path(f"{_TMP}/agentsysperf_emon_tb2"), Path(f"{_TMP}/agentsysperf_phase_emon_tb2"),
                 Path(f"{_TMP}/agentsysperf_phase_tb2_dry")]
    _TAU_DIRS = [Path(f"{_TMP}/agentsysperf_tau_bench"), Path(f"{_TMP}/agentsysperf_emon_tau")]
    _SWE_DIRS = [Path(f"{_TMP}/agentsysperf_swe_bench")]

    for _d in _TB2_DIRS:
        if (_d / "measurement_records.json").exists():
            _dash_benchmarks_with_data.add("Terminal-Bench")
            break
    for _d in _TAU_DIRS:
        if (_d / "measurement_records.json").exists():
            _dash_benchmarks_with_data.add("Tau-Bench")
            break
    for _d in _SWE_DIRS:
        if (_d / "measurement_records.json").exists():
            _dash_benchmarks_with_data.add("SWE-Bench")
            break

    # Sweep verdict. Choose the best-evidenced sweep, not _dash_sweeps[0] —
    # query_sweeps() is unordered, so [0] silently picked a 5-point/1-replicate
    # sweep over a 15-point/3-replicate one and the headline knee described the
    # weaker experiment.
    _dash_knee_str = "—"
    _dash_bottleneck = "—"
    _glance_meta = None
    _glance_pts: list = []
    if _dash_sweeps and _dash_store:
        def _glance_rank(meta):
            p = _dash_store.query_sweep_points(meta["sweep_id"])
            return (len(p), len({x.get("replicate") for x in p}),
                    meta.get("created_at") or 0)

        _glance_meta = max(_dash_sweeps, key=_glance_rank)
        _glance_pts = _dash_store.query_sweep_points(_glance_meta["sweep_id"])
        _sv = _dash_store.query_verdicts(
            _glance_meta["sweep_id"], analyzer_name="scaling",
        )
        if _sv:
            _sv_ev = _sv[0].get("evidence", {})
            _knee = _sv_ev.get("knee", {})
            _dash_knee_str = f"density={_knee.get('density', '?'):g}" if _knee else "—"
            _dash_bottleneck = _sv_ev.get("bottleneck", "—") or "—"

    # "Total Runs" must not count sweep cells. Each density cell writes its own
    # runs row (FK target for its sweep_point), so a 15-cell sweep looked like 15
    # benchmark runs: the store held 23 rows of which 22 were cells and exactly 1
    # was a real benchmark run.
    _dash_cell_ids = {
        f"{m['sweep_id']}" for m in _dash_sweeps
    } if _dash_sweeps else set()

    def _is_sweep_cell(row) -> bool:
        rid = row.get("run_id") or ""
        if any(rid == s or rid.startswith(f"{s}::") for s in _dash_cell_ids):
            return True
        # Fall back on the shape the sweep runner writes when the sweep row
        # itself is gone: no benchmark_id at all.
        return not row.get("benchmark_id")

    _dash_bench_runs = [r for r in _dash_runs if not _is_sweep_cell(r)]

    _kpi1, _kpi2, _kpi3, _kpi4 = st.columns(4)
    _kpi1.metric("Benchmarks", len(_dash_benchmarks_with_data))
    _kpi2.metric("Benchmark Runs", len(_dash_bench_runs),
                 help=f"Excludes {len(_dash_runs) - len(_dash_bench_runs)} "
                      "density-sweep cell rows, which are not benchmark runs.")
    _kpi3.metric("Saturation Knee", _dash_knee_str)
    _kpi4.metric("Top Bottleneck", _dash_bottleneck.replace("_", " ").title())
    # These four numbers do NOT describe one experiment: the left pair counts
    # every stored benchmark run, the right pair comes from a single density
    # sweep. Say so, or the row reads as one result.
    if _glance_meta is not None:
        st.caption(
            "Left two count everything in the store. **Knee and bottleneck come "
            f"from one sweep only** — `{_glance_meta['sweep_id']}`: "
            f"{_sweep_provenance(_glance_meta, _glance_pts)}."
        )
    else:
        st.caption(
            "Left two count everything in the store. No density sweep is stored, "
            "so knee and bottleneck are blank."
        )

    # ─── Benchmark Summary Table ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Benchmark Summary")

    _bench_summary = []
    for _bname, _slug, _dirs in [
        ("Terminal-Bench", "terminal-bench", _TB2_DIRS),
        ("Tau-Bench", "tau-bench", _TAU_DIRS),
        ("SWE-Bench", "swe-bench", _SWE_DIRS),
    ]:
        _row = {"Benchmark": _bname, "Tasks": "—", "Turns": "—",
                "Duration": "—", "LLM %": "—", "Status": "no data"}
        _recs = _data.get_records(_slug, fallback_paths=_dirs, layer="l1")
        _agent_recs = [r for r in _recs if "/" in r.get("span_id", "") and
                       ("turn_" in r["span_id"] or "_llm" in r["span_id"] or "_cmd" in r["span_id"])]
        if _agent_recs:
            _tasks_set = set()
            _llm_dur = 0
            _total_dur = 0
            _llm_count = 0
            _tool_count = 0
            for _r in _agent_recs:
                _p = _r["payload"] if isinstance(_r["payload"], dict) else {}
                _sid = _r["span_id"]
                _parts = _sid.split("/")
                _tasks_set.add(_parts[1] if len(_parts) >= 3 else _parts[0])
                _dur = _p.get("duration_us", 0) / 1000
                _total_dur += _dur
                _phase = _p.get("phase", "")
                if _phase == "reason" or "llm" in _sid:
                    _llm_dur += _dur
                    _llm_count += 1
                elif _phase == "act" or "cmd" in _sid:
                    _tool_count += 1
            _row["Tasks"] = str(len(_tasks_set))
            _row["Turns"] = str(_llm_count + _tool_count)
            _row["Duration"] = f"{_total_dur/1000:.1f}s"
            _row["LLM %"] = f"{_llm_dur/_total_dur*100:.0f}%" if _total_dur > 0 else "—"
            _row["Status"] = "complete"
        elif _recs:
            # Task-level spans only (no per-turn sub-spans). Report the
            # counts we did measure and leave Turns / LLM % as "—" rather
            # than inferring turn structure that was never recorded.
            _row["Tasks"] = str(len({r.get("span_id") for r in _recs if r.get("span_id")}))
            _task_dur = sum(
                (r.get("payload", {}).get("duration_us", 0) / 1e6)
                for r in _recs if isinstance(r.get("payload"), dict)
            )
            if _task_dur > 0:
                _row["Duration"] = f"{_task_dur:.1f}s"
            _row["Status"] = "task-level only"
        _bench_summary.append(_row)

    st.dataframe(_bench_summary, use_container_width=True, hide_index=True,
                 column_config={"Status": st.column_config.TextColumn(width="small")})
    # Name the run behind these rows. They come from measurement records, which
    # may be a different (often older) run than the sweep charted further down.
    _jr = [r for r in _dash_bench_runs
           if (r.get("benchmark_id") or "").startswith("terminal")]
    if _jr:
        _latest = max(_jr, key=lambda r: r.get("start_time") or 0)
        st.caption(_prov_caption(
            f"Latest Terminal-Bench run `{_latest.get('run_id', '?')}`",
            f"model `{_prov_model(_latest.get('model'))}`",
            f"{_latest.get('passed_tasks', '?')}/{_latest.get('total_tasks', '?')} passed",
            _prov_age(_latest.get("start_time")),
            "measured on the full machine, not a pinned cpuset",
        ))

    # ─── Hardware Health ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hardware Health")

    _dash_emon_csv, _dash_emon_src = _data.artifact_source(
        "terminal-bench", kind="emon_csv",
        fallback_dirs=[Path(f"{_TMP}/agentsysperf_mixed4_d192_emon/d192_spread_mixed"),
                       Path(f"{_TMP}/agentsysperf_mixed4_emon/d80_spread_mixed"),
                       Path(f"{_TMP}/agentsysperf_emon_tb2"), Path(f"{_TMP}/emon_tb2_test")],
        # Prefer the wide *details* CSV — _dash_find_metric() searches column
        # names, which only exist there (summary is tall / metrics-as-rows).
        fallback_patterns=("*_system_view_details.csv", "*_socket_view_details.csv",
                           "*_system_view_summary.csv", "*_socket_view_summary.csv"),
    )

    # EMON is collected by a separate root-only script, NOT by `agentsysperf
    # run`, so this panel routinely describes an older and different workload
    # than the KPIs above. Date it and name it, or it reads as current.
    if _dash_emon_csv:
        try:
            _emon_mtime = Path(_dash_emon_csv).stat().st_mtime
        except OSError:
            _emon_mtime = None
        _emon_age = _prov_age(_emon_mtime)
        _stale = bool(_emon_mtime and (time.time() - _emon_mtime) > 86400)
        # "scratch" means the file was found by globbing /tmp, so nothing ties it
        # to any run — it is whatever survived the last reboot. Say that, rather
        # than presenting an orphaned file as this run's hardware profile.
        _unlinked = _dash_emon_src == "scratch"
        _emon_note = _prov_caption(
            f"source `{Path(_dash_emon_csv).name}`",
            _emon_age,
            ("found by scanning `/tmp`, **not linked to any run** — re-run "
             "the EMON collector from the Intel-only agentsysperf-emon plugin "
             "to register it against a "
             "run id" if _unlinked else "registered to this benchmark's latest run"),
            "collected separately from `agentsysperf run`",
        )
        (st.warning if (_stale or _unlinked) else st.caption)(
            (f"**This EMON data is {_emon_age} — it is NOT from the run above.** "
             if _stale else "") + _emon_note
        )
    _hw_col1, _hw_col2 = st.columns(2)

    with _hw_col1:
        if _dash_emon_csv:
            import pandas as pd
            _dash_df = pd.read_csv(_dash_emon_csv)

            def _dash_find_metric(df, keyword):
                # NaN must read as "not measured", not as a value. On Clearwater
                # Forest the CWF pyEDP XML defines ZERO TMA nodes, so pyEDP still
                # emits every metric_TMA_* column but leaves them empty. NaN
                # passes `is not None`, so the panel drew four bars of nothing
                # and looked like a rendering failure rather than an unsupported
                # metric. See memory: "CWF defines zero TMA nodes".
                import math
                for col in df.columns:
                    if keyword.lower() in col.lower():
                        try:
                            val = float(df[col].mean())
                        except (ValueError, TypeError):
                            continue
                        if math.isnan(val):
                            continue
                        return val
                return None

            _tma_search = {
                "Frontend Bound": "tma_frontend_bound(%)",
                "Backend Bound": "tma_backend_bound(%)",
                "Bad Speculation": "tma_bad_speculation(%)",
                "Retiring": "tma_retiring(%)",
            }
            _tma_vals = {}
            for _label, _kw in _tma_search.items():
                _val = _dash_find_metric(_dash_df, _kw)
                if _val is not None:
                    _tma_vals[_label] = _val

            if _tma_vals:
                _tma_colors = {"Frontend Bound": "#ff9800", "Backend Bound": "#d32f2f",
                               "Bad Speculation": "#9c27b0", "Retiring": "#4caf50"}
                _fig_tma = go.Figure(go.Bar(
                    x=list(_tma_vals.values()),
                    y=list(_tma_vals.keys()),
                    orientation="h",
                    marker_color=[_tma_colors.get(k, "#607d8b") for k in _tma_vals.keys()],
                    text=[f"{v:.0f}%" for v in _tma_vals.values()],
                    textposition="outside",
                ))
                _fig_tma.update_layout(
                    title="TMA Level 1 (%)",
                    xaxis_title="%", height=250,
                    margin=dict(l=120, r=50, t=40, b=30),
                )
                st.plotly_chart(_fig_tma, use_container_width=True)
            elif any("tma" in c.lower() for c in _dash_df.columns):
                # Columns present but all-NaN: the SKU defines no TMA nodes.
                # Say which SKU and why, rather than implying a broken chart.
                st.info(
                    "**TMA not available on this CPU.** The EMON CSV carries the "
                    f"`metric_TMA_*` columns but every value is empty — this SKU's "
                    "pyEDP definition contributes no TMA nodes, so there is no "
                    "top-down breakdown to show. Raw frontend/retiring/"
                    "bad-speculation counters still populate; see **Hardware "
                    "Telemetry**."
                )
            else:
                st.info("TMA data not available in EMON CSV columns.")
        else:
            st.info("No EMON data. Run a benchmark with EMON to see TMA breakdown.")

    with _hw_col2:
        # Top findings from EmonAnalyzer
        _dash_ea = _emon_analyzer()
        if _dash_emon_csv and _dash_ea is not None:
            try:
                from src.protocols import MeasurementRecord
                _dash_er = list(_dash_ea.analyze([MeasurementRecord(span_id="dash", layer="emon", payload={"csv_path": str(_dash_emon_csv)})]))
                if _dash_er:
                    _dash_ev = _dash_er[0].evidence
                    _dash_rcs = _dash_ev.get("root_causes", [])
                    if _dash_rcs:
                        st.markdown("**Top Findings**")
                        for _rc in _dash_rcs[:4]:
                            _sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(_rc["severity"], "⚪")
                            st.markdown(f"{_sev_icon} **[{_rc['severity']}]** {_rc['headline']}")
                            st.caption(f"    Fix: {_rc['fix']} | Gain: {_rc['gain_range']}")
                    else:
                        st.success("No hardware bottlenecks detected.")
            except Exception as _e:
                st.warning(f"EmonAnalyzer error: {_e}")
        elif _dash_emon_csv:
            st.caption(
                "Hardware findings need the Intel-only `agentsysperf-emon` "
                "plugin (not installed)."
            )
        else:
            st.info("Run with EMON enabled to surface hardware findings.")

    # ─── Scaling at a Glance ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Scaling at a Glance")

    # Read the store, which is the system of record. This panel used to prefer
    # two hardcoded /tmp/agentsysperf_scaling_*/all_results.json files; both are
    # long gone, so it always fell through to the sweep branch — which then read
    # a column name that does not exist ("throughput_trials_per_min" vs the
    # stored "throughput_per_min"), so every point plotted as 0 and the headline
    # chart was a flat line at zero. Use the same builder as the Scaling page so
    # the two views cannot drift again.
    if _glance_meta is not None and _glance_pts:
        from src.dashboard.scaling_views import throughput_knee_figure

        # _glance_meta / _glance_pts were selected above (best-evidenced sweep),
        # so the KPI knee and this chart always describe the SAME sweep.
        _glance_id = _glance_meta["sweep_id"]
        # State the experiment BEFORE the chart. This panel is not the same
        # experiment as the KPIs or the Benchmark Summary above it.
        st.info(
            f"**This chart is one density sweep, not the benchmark run above.** "
            f"`{_glance_id}` — {_sweep_provenance(_glance_meta, _glance_pts)}. "
            "It answers *how many concurrent agents fit before throughput stops "
            "scaling*, on one task at fixed difficulty. It is not a pass-rate or "
            "model-quality result."
        )
        _gv = _dash_store.query_verdicts(_glance_id, analyzer_name="scaling")
        _gev = _gv[0]["evidence"] if _gv else {}
        st.plotly_chart(
            throughput_knee_figure(
                _glance_pts, knee=_gev.get("knee"), bottleneck=_gev.get("bottleneck"),
            ),
            use_container_width=True,
        )
        if not _gv:
            st.caption(
                "No scaling verdict stored for this sweep, so no knee is marked. "
                "Re-import it (`agentsysperf sweep import <dir>`) to analyze."
            )
        if len(_dash_sweeps) > 1:
            _others = ", ".join(
                f"`{m['sweep_id']}`" for m in _dash_sweeps
                if m["sweep_id"] != _glance_id
            )
            st.caption(
                f"{len(_dash_sweeps)} sweeps stored; showing the best-evidenced "
                f"one. Others: {_others}. Compare them side by side under "
                "**Scaling**."
            )
    elif _glance_meta is not None:
        st.info(f"Sweep `{_glance_meta['sweep_id']}` has no points yet.")
    else:
        st.info(
            "No scaling data in the store yet. Run a sweep "
            "(`harness/scripts/run_density_test_cwf.sh`), then import it with "
            "`agentsysperf sweep import`."
        )

    # ─── Quick Actions ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Quick Actions")
    _qa1, _qa2, _qa3 = st.columns(3)
    with _qa1:
        st.code("agentsysperf sweep run --dry-run", language="bash")
        st.caption("Run concurrency sweep (--dry-run cells are synthetic)")
    with _qa2:
        st.code("python -m experiments.scaling.run_experiment --quick", language="bash")
        st.caption("Run density experiment")
    with _qa3:
        st.code("poetry run agentsysperf db ls", language="bash")
        st.caption("List stored runs")


elif demo_page == "Platform::Platform Discovery":
    st.title("Platform Discovery")
    st.markdown("AgentSysPerf detects hardware capabilities — no manual configuration needed.")

    platform = load_platform()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU", platform["model_name"].replace("Intel(R) Xeon(R) ", ""))
        st.metric("Vendor", platform["vendor"])
    with col2:
        st.metric("Microarchitecture", platform["microarchitecture"])
        st.metric("Cores", f"{platform['physical_cores']} physical / {platform['logical_cpus']} logical")
    with col3:
        st.metric("NUMA Nodes", platform["numa_nodes"])
        st.metric("L3 per Node", f"{platform['l3_per_node_mb']} MB")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("L3 Budget (usable)", f"{platform['l3_budget_mb']} MB")
    with col2:
        # Label the provenance: an estimated peak and a measured one should not
        # look identical, and an unknown platform has no peak at all.
        _bw = platform["dram_bw_per_node_gbs"]
        _src = platform.get("dram_bw_source", "estimated")
        st.metric(
            "DRAM BW / Node",
            f"{_bw} GB/s" if _bw else "unknown",
            help=f"Source: {_src}",
        )
    with col3:
        st.metric("AMX", "Yes" if platform["has_amx"] else "No")
    with col4:
        st.metric("AVX-512", "Yes" if platform["has_avx512"] else "No")

    st.markdown("---")
    st.info("Works on **Intel Xeon**, **AMD EPYC**, and **ARM Neoverse**. Thresholds and analysis adapt automatically.")


elif demo_page == "Platform::Pre-flight Validation":
    st.title("Pre-flight System Validation")
    st.markdown("Validates the host is properly configured before benchmarking.")

    preflight = load_preflight()

    # Summary
    passed = preflight["passed"]
    total = preflight["total"]
    if passed == total:
        st.success(f"All checks passed: {passed}/{total}")
    else:
        st.warning(f"Issues found: {total - passed}/{total} checks failed")

    # Table
    for check in preflight["checks"]:
        status = check["status"]
        if status == "OK":
            icon = "✅"
        elif status == "WARNING":
            icon = "⚠️"
        else:
            icon = "❌"

        col1, col2, col3, col4 = st.columns([1, 4, 4, 4])
        with col1:
            st.write(icon)
        with col2:
            st.write(f"**{check['name']}**")
        with col3:
            st.write(check["current"])
        with col4:
            st.write(check["expected"])

    st.markdown("---")
    st.info("Catches misconfigured governors, missing perf access, wrong kernel settings — things that silently corrupt results.")


elif demo_page == "Terminal-Bench::Hardware Baselines":
    st.title("Terminal-Bench: Hardware Baselines")
    st.markdown("9 synthetic CPU workloads measured with IPC and cache miss rate in 18 seconds.")

    benchmarks = load_benchmarks()

    if not benchmarks:
        st.error("No benchmark data found. Run `examples/run_synthetic_l1_l3.py` first.")
    else:
        # IPC Chart
        tasks = [b["task"] for b in benchmarks]
        ipcs = [b["ipc"] for b in benchmarks]
        cache_miss = [b["cache_miss_pct"] for b in benchmarks]

        col1, col2 = st.columns(2)

        with col1:
            fig_ipc = go.Figure(go.Bar(
                x=ipcs,
                y=tasks,
                orientation="h",
                marker_color=["#d32f2f" if ipc < 2 else "#1976d2" if ipc < 4 else "#388e3c" for ipc in ipcs],
            ))
            fig_ipc.update_layout(
                title="Instructions Per Cycle (IPC)",
                xaxis_title="IPC",
                height=400,
                margin=dict(l=100),
            )
            st.plotly_chart(fig_ipc, use_container_width=True)

        with col2:
            fig_cache = go.Figure(go.Bar(
                x=cache_miss,
                y=tasks,
                orientation="h",
                marker_color=["#d32f2f" if cm > 50 else "#ff9800" if cm > 20 else "#388e3c" for cm in cache_miss],
            ))
            fig_cache.update_layout(
                title="Cache Miss Rate (%)",
                xaxis_title="Cache Miss %",
                height=400,
                margin=dict(l=100),
            )
            st.plotly_chart(fig_cache, use_container_width=True)

        # Data table
        st.markdown("### Detailed Results")
        st.dataframe(
            benchmarks,
            column_config={
                "task": "Workload",
                "duration_ms": st.column_config.NumberColumn("Duration (ms)", format="%.0f"),
                "cpu_time_s": st.column_config.NumberColumn("CPU Time (s)", format="%.2f"),
                "cpu_pct": st.column_config.NumberColumn("CPU %", format="%.1f"),
                "rss_mb": st.column_config.NumberColumn("RSS (MB)", format="%.0f"),
                "ipc": st.column_config.NumberColumn("IPC", format="%.2f"),
                "cache_miss_pct": st.column_config.NumberColumn("Cache Miss %", format="%.1f"),
            },
            use_container_width=True,
        )

        st.markdown("---")
        st.info("**Key insight**: `io` workload (IPC=1.65, 97% cache miss) is clearly memory-bound. `control` (IPC=5.98) is compute-efficient.")

    # ─── Bottleneck Analysis (merged) ────────────────────────────────────
    st.markdown("---")
    st.header("Bottleneck Analysis")
    st.markdown("Identifies memory subsystem patterns and maps them to optimization solutions.")

    analysis = load_analysis()

    if not analysis:
        st.warning("No analysis data. Run benchmarks first.")
    else:
        # Split into bottleneck vs clean
        bottlenecks = [a for a in analysis if a["verdict"] != "no_memory_bottleneck"]
        clean = [a for a in analysis if a["verdict"] == "no_memory_bottleneck"]

        if bottlenecks:
            st.subheader("Bottlenecks Detected")
            for item in bottlenecks:
                with st.expander(f"**{item['task']}** — {item['verdict']} ({item['confidence']}% confidence)", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Patterns:**")
                        for p in item["patterns"]:
                            st.markdown(f"- `{p}`")
                    with col2:
                        st.markdown("**Solutions:**")
                        for s in item["solutions"]:
                            st.markdown(f"- **{s['name']}** [{s['category']}]")
                            st.caption(f"  {s['impact']}")

        if clean:
            st.subheader("No Memory Bottleneck")
            clean_tasks = ", ".join([f"`{a['task']}`" for a in clean])
            st.success(f"Compute-efficient workloads: {clean_tasks}")

        st.markdown("---")

        # Solution mapping summary
        st.subheader("Solution Registry")
        solution_data = [
            {"Solution": "Speculative Decoding", "Category": "algorithmic", "Trigger": "weight_streaming"},
            {"Solution": "Quantization (INT8/INT4)", "Category": "algorithmic", "Trigger": "weight_streaming"},
            {"Solution": "NUMA-Aware Scheduling", "Category": "scheduling", "Trigger": "cross_numa_traffic"},
            {"Solution": "RAG (Context Reduction)", "Category": "architecture", "Trigger": "kv_cache_pressure"},
            {"Solution": "Cache Partitioning (RDT)", "Category": "hardware", "Trigger": "capacity_thrashing"},
            {"Solution": "Memory Interleaving", "Category": "hardware", "Trigger": "bandwidth_saturation"},
            {"Solution": "Data Tiling", "Category": "algorithmic", "Trigger": "working_set_overflow"},
        ]
        st.dataframe(solution_data, use_container_width=True)


elif demo_page == "Terminal-Bench::Agent Performance":
    st.title("Terminal-Bench: Agent Performance")
    st.markdown("Real agent workload: LLM reasoning + tool execution in a terminal environment, measured per-turn.")

    from src.analyzers.phase_profiler import PHASE_LABELS, PHASE_ORDER

    _TB2_DATA_DIRS = [
        Path(f"{_TMP}/agentsysperf_emon_tb2"),
        Path(f"{_TMP}/agentsysperf_phase_emon_tb2"),
        Path(f"{_TMP}/agentsysperf_phase_tb2_dry"),
    ]
    _tb2_records = _data.get_records(
        "terminal-bench",
        fallback_paths=_TB2_DATA_DIRS,
        layer="l1",
    )
    # Select phase-tagged spans by their `phase` payload field, NOT by span-id
    # substrings: `admit` and `commit` fire once per task outside the turn loop,
    # so their ids carry no "turn_"/"_llm"/"_cmd" and a substring filter dropped
    # them entirely. Keying off the tag admits all five pipeline phases.
    #
    # This also excludes the task-level parent span (phase=None), which CONTAINS
    # the phase spans — counting it would double every total.
    _tb2_agent_recs = [
        r for r in _tb2_records
        if "/" in r["span_id"]
        and (r["payload"] if isinstance(r["payload"], dict) else {}).get("phase")
        in PHASE_ORDER
    ]

    if _tb2_agent_recs:
        import pandas as pd

        # Group by task
        _tb2_tasks = {}
        for _r in _tb2_agent_recs:
            _sid = _r["span_id"]
            _p = _r["payload"] if isinstance(_r["payload"], dict) else {}
            _parts = _sid.split("/")
            _task_key = _parts[1] if len(_parts) >= 3 else _parts[0]
            if _task_key not in _tb2_tasks:
                _tb2_tasks[_task_key] = {
                    "llm_calls": 0, "tool_calls": 0,
                    "llm_dur_ms": 0, "tool_dur_ms": 0, "total_dur_ms": 0,
                    "cpu_pct_sum": 0, "cpu_count": 0,
                    "rss_kb_peak": 0,
                    "turn_latencies": [],
                    # Per-phase wall time and span counts, keyed by phase name.
                    # The old two-bucket split (LLM vs Tool) had no home for
                    # admit/retrieve/commit, so their time vanished from every
                    # total while still being counted in total_dur_ms.
                    "phase_dur_ms": {_ph: 0.0 for _ph in PHASE_ORDER},
                    "phase_counts": {_ph: 0 for _ph in PHASE_ORDER},
                }
            _tm = _tb2_tasks[_task_key]
            _dur_ms = _p.get("duration_us", 0) / 1000
            _tm["total_dur_ms"] += _dur_ms
            _tm["turn_latencies"].append(_dur_ms)

            _phase = _p.get("phase", "")
            _tm["phase_dur_ms"][_phase] += _dur_ms
            _tm["phase_counts"][_phase] += 1
            # LLM/Tool remain as named aliases for reason/act — the two phases
            # the summary metrics call out — but they are no longer the only
            # buckets, so nothing is dropped.
            if _phase == "reason":
                _tm["llm_calls"] += 1
                _tm["llm_dur_ms"] += _dur_ms
            elif _phase == "act":
                _tm["tool_calls"] += 1
                _tm["tool_dur_ms"] += _dur_ms

            _cpu = _p.get("cpu_pct_mean", 0)
            if _cpu:
                _tm["cpu_pct_sum"] += _cpu
                _tm["cpu_count"] += 1
            _rss = _p.get("rss_kb_peak", 0)
            if _rss > _tm["rss_kb_peak"]:
                _tm["rss_kb_peak"] = _rss

        # Summary metrics
        _n_tasks = len(_tb2_tasks)
        _total_llm = sum(t["llm_calls"] for t in _tb2_tasks.values())
        _total_tool = sum(t["tool_calls"] for t in _tb2_tasks.values())
        _total_dur = sum(t["total_dur_ms"] for t in _tb2_tasks.values())
        _total_llm_dur = sum(t["llm_dur_ms"] for t in _tb2_tasks.values())
        _avg_cpu = (sum(t["cpu_pct_sum"] for t in _tb2_tasks.values()) /
                    max(sum(t["cpu_count"] for t in _tb2_tasks.values()), 1))
        # Wall time per phase across all tasks — the five numbers that must sum
        # to _total_dur. Anything unaccounted for is a bug, not rounding.
        _phase_totals = {
            _ph: sum(t["phase_dur_ms"][_ph] for t in _tb2_tasks.values())
            for _ph in PHASE_ORDER
        }

        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        _sc1.metric("Agent Tasks", _n_tasks)
        # One `reason` span per turn — llm+tool double-counted every turn.
        _sc2.metric("Total Turns", _total_llm)
        _sc3.metric("Total Duration", f"{_total_dur/1000:.1f}s")
        _sc4.metric("Avg CPU %", f"{_avg_cpu:.0f}%")

        _sc5, _sc6, _sc7, _sc8 = st.columns(4)
        _sc5.metric("LLM Calls", _total_llm)
        _sc6.metric("Tool Calls", _total_tool)
        _llm_pct = (_total_llm_dur / _total_dur * 100) if _total_dur > 0 else 0
        _sc7.metric("LLM Time %", f"{_llm_pct:.0f}%")
        _avg_turns = _total_llm / _n_tasks if _n_tasks else 0
        _sc8.metric("Avg Turns/Task", f"{_avg_turns:.1f}")

        # Wall time by phase — all five, so the parts sum to the whole.
        st.markdown("#### Wall Time by Phase")
        _ph_rows = [
            {
                "Phase": PHASE_LABELS.get(_ph, _ph.title()),
                "Spans": sum(t["phase_counts"][_ph] for t in _tb2_tasks.values()),
                "Wall (ms)": round(_phase_totals[_ph], 1),
                "Wall %": round(100.0 * _phase_totals[_ph] / _total_dur, 1)
                if _total_dur else 0.0,
            }
            for _ph in PHASE_ORDER
            if sum(t["phase_counts"][_ph] for t in _tb2_tasks.values())
        ]
        st.dataframe(_ph_rows, use_container_width=True, hide_index=True)

        # Per-task table — one column per phase that actually occurred, so the
        # row's phase columns account for its Duration.
        st.markdown("#### Per-Task Breakdown")
        _present = [
            _ph for _ph in PHASE_ORDER
            if sum(t["phase_counts"][_ph] for t in _tb2_tasks.values())
        ]
        _task_rows = []
        for _tk, _tv in sorted(_tb2_tasks.items()):
            _row = {"Task": _tk, "Turns": _tv["llm_calls"],
                    "Duration (ms)": round(_tv["total_dur_ms"], 0)}
            for _ph in _present:
                _row[f"{_ph} (ms)"] = round(_tv["phase_dur_ms"][_ph], 1)
            _row["Avg CPU %"] = round(_tv["cpu_pct_sum"] / max(_tv["cpu_count"], 1), 1)
            _row["Peak RSS (MB)"] = round(_tv["rss_kb_peak"] / 1024, 1)
            _task_rows.append(_row)
        st.dataframe(_task_rows, use_container_width=True, hide_index=True)

        # Per-turn latency chart, stacked by phase. Turn number comes from the
        # span id (turn_<n>_*), not from enumeration order: the phase spans of
        # one turn must stack on ONE bar, and admit/commit have no turn at all.
        st.markdown("#### Per-Turn Latency")
        # Same palette the Phase Analysis section uses, so a phase is one colour
        # everywhere in the app.
        _phase_colors_tb2 = {
            "admit": "#9e9e9e",
            "retrieve": "#2196f3",
            "reason": "#f44336",
            "act": "#4caf50",
            "commit": "#ff9800",
        }
        _turn_data = []
        for _r in _tb2_agent_recs:
            _sid = _r["span_id"]
            _parts = _sid.split("/")
            _p = _r["payload"] if isinstance(_r["payload"], dict) else {}
            _phase = _p.get("phase", "")
            _turn_no = None
            for _seg in _sid.split("/")[-1].split("_"):
                if _seg.isdigit():
                    _turn_no = int(_seg)
                    break
            _turn_data.append({
                "Task": _parts[1] if len(_parts) >= 3 else _parts[0],
                "Turn": _turn_no if _turn_no is not None else -1,
                "Duration (ms)": _p.get("duration_us", 0) / 1000,
                "Phase": _phase,
            })

        _turn_df = pd.DataFrame(_turn_data)
        # Per-task spans (admit/commit) carry no turn number; show them as a
        # separate labelled category instead of silently folding them into turn 0.
        _turn_df["Turn"] = _turn_df["Turn"].apply(
            lambda t: "per-task" if t < 0 else str(t)
        )
        _fig_turns = go.Figure()
        for _ph in PHASE_ORDER:
            _sub = _turn_df[_turn_df["Phase"] == _ph]
            if not _sub.empty:
                _fig_turns.add_trace(go.Bar(
                    x=_sub["Turn"], y=_sub["Duration (ms)"],
                    name=PHASE_LABELS.get(_ph, _ph.title()),
                    marker_color=_phase_colors_tb2.get(_ph, "#607d8b"),
                ))
        _fig_turns.update_layout(
            title="Turn-by-Turn Latency (all 5 phases)",
            xaxis_title="Turn #", yaxis_title="Duration (ms)",
            barmode="stack", height=350,
            margin=dict(l=50, r=30, t=40, b=30),
        )
        st.plotly_chart(_fig_turns, use_container_width=True)

        # Time split pie chart — every phase, so the slices total 100% of
        # measured wall time rather than 88% of it.
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            _pie_phases = [_ph for _ph in PHASE_ORDER if _phase_totals[_ph] > 0]
            _fig_pie = go.Figure(go.Pie(
                labels=[PHASE_LABELS.get(_ph, _ph.title()) for _ph in _pie_phases],
                values=[_phase_totals[_ph] for _ph in _pie_phases],
                marker_colors=[_phase_colors_tb2[_ph] for _ph in _pie_phases],
                hole=0.4,
            ))
            _fig_pie.update_layout(title="Time Distribution", height=300,
                                   margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(_fig_pie, use_container_width=True)

        with _tc2:
            # Latency distribution histogram
            _turn_only = _turn_df[_turn_df["Turn"] != "per-task"]
            _all_lats = (
                _turn_only.groupby(["Task", "Turn"])["Duration (ms)"].sum().tolist()
                if not _turn_only.empty else []
            )
            if _all_lats:
                _fig_hist = go.Figure(go.Histogram(
                    x=_all_lats, nbinsx=20,
                    marker_color="#1976d2", opacity=0.7,
                ))
                _fig_hist.update_layout(title="Turn Latency Distribution",
                                        xaxis_title="Duration (ms)", yaxis_title="Count",
                                        height=300, margin=dict(l=50, r=30, t=40, b=30))
                st.plotly_chart(_fig_hist, use_container_width=True)

    else:
        st.info(
            "No TB2 agent task data available. Run a benchmark to collect agent metrics:\n\n"
            "```\nagentsysperf run -b terminal-bench --num-tasks 2 --model gpt-4o-mini\n```"
        )

    # ─── Phase Analysis (merged) ─────────────────────────────────────────
    st.markdown("---")
    st.header("Agentic Pipeline Phase Analysis")
    st.markdown("""
    Decomposes agentic workloads into 5 pipeline phases and characterizes
    each phase's hardware signature. Maps patterns to Intel Xeon optimizations.
    """)

    from src.analyzers.phase_profiler import PhaseProfiler, PHASE_LABELS

    _PHASE_DATA_DIRS = {
        "Terminal-Bench": [
            Path(f"{_TMP}/agentsysperf_emon_tb2"),
            Path(f"{_TMP}/agentsysperf_phase_tb2"),
            Path(f"{_TMP}/agentsysperf_phase_tb2_dry"),
            Path(f"{_TMP}/agentsysperf_tb2_benchmark"),
        ],
        "Tau-Bench": [
            Path(f"{_TMP}/agentsysperf_tau_bench"),
            Path(f"{_TMP}/agentsysperf_phase_tau"),
            Path(f"{_TMP}/agentsysperf_scaling"),
        ],
        "SWE-Bench": [
            Path(f"{_TMP}/agentsysperf_swe_bench"),
            Path(f"{_TMP}/agentsysperf_phase_swe"),
        ],
    }

    _raw = _data.get_records(
        _BENCH_SLUG.get(_nav_section, _nav_section),
        fallback_paths=_PHASE_DATA_DIRS.get(_nav_section, []),
    )
    _phase_src = _data.latest_run_id(_BENCH_SLUG.get(_nav_section, _nav_section)) or "legacy JSON"

    if not _raw:
        _bench_lower = _nav_section.lower().replace("-", "_")
        st.warning(
            f"No phase-tagged data available for **{_nav_section}**. Run:\n\n"
            f"```\npoetry run python examples/run_phase_profiler_{_bench_lower}.py --dry-run\n```"
        )
    else:
        from src.protocols import MeasurementRecord
        _records = [MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"]) for r in _raw]

        _profiler = PhaseProfiler()
        _results = list(_profiler.analyze(_records))

        if not _results:
            st.error("PhaseProfiler produced no results from the loaded data.")
        else:
            _r = _results[0]
            _bd = _r.evidence.get("phase_breakdown", {})
            _inflection = _r.evidence.get("inflection")

            st.markdown("### 5-Phase Agentic Pipeline")
            st.code("""
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ 01 ADMIT │──▶│02 RETRIEVE│──▶│ 03 REASON│──▶│  04 ACT  │──▶│05 COMMIT │
    │ Auth/    │   │ Vector/  │   │   LLM    │   │ Tool/API │   │Writeback │
    │ Policy   │   │ Rerank   │   │ Inference│   │ Code Exec│   │  Audit   │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
            """, language="text")

            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            with _mc1:
                _dominant = _r.verdict.replace("phase_profile_", "")
                st.metric("Dominant Phase", _dominant.title())
            with _mc2:
                st.metric("Total Wall-Clock", f"{_r.evidence['total_wall_ms']:.0f} ms")
            with _mc3:
                st.metric("Iterations", _r.evidence.get("iteration_count", 0))
            with _mc4:
                if _inflection:
                    st.metric("Inflection", f"iter {_inflection['iteration']}")
                else:
                    st.metric("Inflection", "Not reached")

            st.markdown("---")

            col1, col2 = st.columns(2)

            _phase_colors = {
                "admit": "#9e9e9e",
                "retrieve": "#2196f3",
                "reason": "#f44336",
                "act": "#4caf50",
                "commit": "#ff9800",
            }

            with col1:
                _phases = list(_bd.keys())
                _wall_vals = [_bd[p]["wall_pct"] for p in _phases]
                _colors = [_phase_colors.get(p, "#607d8b") for p in _phases]

                fig_pie = go.Figure(go.Pie(
                    labels=[p.title() for p in _phases],
                    values=_wall_vals,
                    marker=dict(colors=_colors),
                    textinfo="label+percent",
                    hole=0.3,
                ))
                fig_pie.update_layout(
                    title="Wall-Clock Distribution by Phase",
                    height=350,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                _ipc_v = []
                _miss_v = []
                _labels = []
                _sizes = []
                _clrs = []
                for _p in _phases:
                    _d = _bd[_p]
                    if _d["avg_ipc"] is not None and _d["avg_cache_miss_pct"] is not None:
                        _ipc_v.append(_d["avg_ipc"])
                        _miss_v.append(_d["avg_cache_miss_pct"])
                        _labels.append(_p.title())
                        _sizes.append(max(_d["wall_pct"] * 0.8, 10))
                        _clrs.append(_phase_colors.get(_p, "#607d8b"))

                if _ipc_v:
                    fig_scatter = go.Figure(go.Scatter(
                        x=_ipc_v, y=_miss_v,
                        mode="markers+text",
                        text=_labels,
                        textposition="top center",
                        marker=dict(size=_sizes, color=_clrs, opacity=0.8),
                    ))
                    fig_scatter.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.4,
                                         annotation_text="Memory-bound threshold")
                    fig_scatter.add_vline(x=2.0, line_dash="dash", line_color="gray", opacity=0.4,
                                         annotation_text="Compute-efficient threshold")
                    fig_scatter.update_layout(
                        title="Hardware Signature per Phase",
                        xaxis_title="IPC",
                        yaxis_title="Cache Miss %",
                        height=350,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("No L3 perf counter data for hardware signature plot. Enable perf access.")

            st.markdown("### Phase Details")
            _tbl = []
            for _p, _d in _bd.items():
                _tbl.append({
                    "Phase": PHASE_LABELS.get(_p, _p.title()),
                    "Wall %": _d["wall_pct"],
                    "CPU %": _d["cpu_pct"],
                    "Duration (ms)": _d["wall_ms"],
                    "Avg IPC": round(_d["avg_ipc"], 2) if _d["avg_ipc"] is not None else None,
                    "Cache Miss %": round(_d["avg_cache_miss_pct"], 1) if _d["avg_cache_miss_pct"] is not None else None,
                    "HW Pattern": _d["pattern"],
                    "Spans": _d["span_count"],
                })
            st.dataframe(_tbl, use_container_width=True)

            if _inflection:
                st.markdown("---")
                st.markdown("### Inflection Point Detected")
                st.error(
                    f"At **iteration {_inflection['iteration']}**, cumulative orchestration time "
                    f"(Retrieve + Act + Commit = {_inflection['cumulative_other_s']:.2f}s) exceeded "
                    f"cumulative inference time ({_inflection['cumulative_reason_s']:.2f}s). "
                    f"Ratio: **{_inflection['ratio']:.2f}x**.\n\n"
                    f"**Implication**: CPU optimization of non-inference phases now yields more ROI "
                    f"than model-level optimizations (quantization, speculative decoding)."
                )

            st.markdown("---")
            st.markdown("### Per-Phase Optimization Map")
            _sol_data = []
            for _p, _sols in _r.evidence.get("phase_solutions", {}).items():
                if _sols:
                    _sol_data.append({
                        "Phase": _p.title(),
                        "Pattern": _bd.get(_p, {}).get("pattern", "—"),
                        "Solutions": "; ".join(_sols),
                    })
            if _sol_data:
                st.dataframe(_sol_data, use_container_width=True)

            if _r.recommendations:
                st.markdown("### Key Findings")
                for _rec in _r.recommendations:
                    st.info(_rec)

            st.caption(f"Data source: `{_phase_src}`")


elif demo_page == "Tau-Bench::Agent Performance":
    st.title("Tau-Bench: Agent Performance")
    st.markdown("""
    Tau-Bench evaluates LLM agents on retail, airline, and telecom customer service tasks.
    Measures task success rate, turn count, and latency under real tool-use workflows.
    """)

    _tau_results_dir = None
    for _cand in [
        Path(f"{_TMP}/agentsysperf_tau_bench"),
        Path(f"{_TMP}/agentsysperf_scaling_tau"),
    ]:
        if (_cand / "measurement_records.json").exists() or list(_cand.glob("*results*.json")):
            _tau_results_dir = _cand
            break

    if _tau_results_dir is None:
        st.warning(
            "No Tau-Bench results available. Run:\n\n"
            "```bash\npoetry run python -m src.benchmarks.tau_bench.adapter "
            "--env retail --model Qwen/Qwen3-Coder-30B --num-tasks 5\n```"
        )
    else:
        _tau_files = list(_tau_results_dir.glob("*results*.json")) + list(_tau_results_dir.glob("*tau*.json"))
        if _tau_files:
            _tau_data = json.loads(_tau_files[0].read_text())
            if isinstance(_tau_data, list):
                st.dataframe(_tau_data, use_container_width=True)
            elif isinstance(_tau_data, dict):
                st.json(_tau_data)
        else:
            _mr_file = _tau_results_dir / "measurement_records.json"
            if _mr_file.exists():
                _raw = json.loads(_mr_file.read_text())
                _spans = {}
                for _r in _raw:
                    _sid = _r["span_id"]
                    if _sid not in _spans:
                        _spans[_sid] = {}
                    _spans[_sid][_r["layer"]] = _r["payload"]

                _task_rows = []
                for _sid, _layers in _spans.items():
                    _l1 = _layers.get("l1", {})
                    _task_rows.append({
                        "Task": _sid.split("::")[-1] if "::" in _sid else _sid,
                        "Duration (ms)": round(_l1.get("duration_us", 0) / 1000, 0),
                        "CPU %": round(_l1.get("cpu_pct_mean", 0), 1),
                        "RSS (MB)": round(_l1.get("rss_kb_peak", 0) / 1024, 0),
                    })
                if _task_rows:
                    st.dataframe(_task_rows, use_container_width=True)
                else:
                    st.info("Measurement records found but no task spans extracted.")
            else:
                st.info(f"Results directory found at `{_tau_results_dir}` but no parseable data.")

    st.markdown("---")
    st.markdown("### Tau-Bench Environments")
    st.dataframe([
        {"Environment": "retail", "Tasks": 115, "Domain": "E-commerce returns, exchanges, order management"},
        {"Environment": "airline", "Tasks": 50, "Domain": "Flight booking, cancellation, rebooking"},
        {"Environment": "telecom", "Tasks": 30, "Domain": "Plan changes, billing disputes, support"},
    ], use_container_width=True, hide_index=True)

    # ─── Phase Analysis (merged) ─────────────────────────────────────────
    st.markdown("---")
    st.header("Agentic Pipeline Phase Analysis")
    st.markdown("""
    Decomposes agentic workloads into 5 pipeline phases and characterizes
    each phase's hardware signature. Maps patterns to Intel Xeon optimizations.
    """)

    from src.analyzers.phase_profiler import PhaseProfiler, PHASE_LABELS

    _PHASE_DATA_DIRS_TAU = [
        Path(f"{_TMP}/agentsysperf_tau_bench"),
        Path(f"{_TMP}/agentsysperf_phase_tau"),
        Path(f"{_TMP}/agentsysperf_scaling"),
    ]

    _raw = _data.get_records(
        _BENCH_SLUG.get(_nav_section, _nav_section),
        fallback_paths=_PHASE_DATA_DIRS_TAU,
    )
    _phase_src = _data.latest_run_id(_BENCH_SLUG.get(_nav_section, _nav_section)) or "legacy JSON"

    if not _raw:
        st.warning(
            "No phase-tagged data available for **Tau-Bench**. Run:\n\n"
            "```\npoetry run python examples/run_phase_profiler_tau_bench.py --dry-run\n```"
        )
    else:
        from src.protocols import MeasurementRecord
        _records = [MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"]) for r in _raw]

        _profiler = PhaseProfiler()
        _results = list(_profiler.analyze(_records))

        if not _results:
            st.error("PhaseProfiler produced no results from the loaded data.")
        else:
            _r = _results[0]
            _bd = _r.evidence.get("phase_breakdown", {})
            _inflection = _r.evidence.get("inflection")

            st.markdown("### 5-Phase Agentic Pipeline")
            st.code("""
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ 01 ADMIT │──▶│02 RETRIEVE│──▶│ 03 REASON│──▶│  04 ACT  │──▶│05 COMMIT │
    │ Auth/    │   │ Vector/  │   │   LLM    │   │ Tool/API │   │Writeback │
    │ Policy   │   │ Rerank   │   │ Inference│   │ Code Exec│   │  Audit   │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
            """, language="text")

            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            with _mc1:
                _dominant = _r.verdict.replace("phase_profile_", "")
                st.metric("Dominant Phase", _dominant.title())
            with _mc2:
                st.metric("Total Wall-Clock", f"{_r.evidence['total_wall_ms']:.0f} ms")
            with _mc3:
                st.metric("Iterations", _r.evidence.get("iteration_count", 0))
            with _mc4:
                if _inflection:
                    st.metric("Inflection", f"iter {_inflection['iteration']}")
                else:
                    st.metric("Inflection", "Not reached")

            st.markdown("---")

            col1, col2 = st.columns(2)

            _phase_colors = {
                "admit": "#9e9e9e",
                "retrieve": "#2196f3",
                "reason": "#f44336",
                "act": "#4caf50",
                "commit": "#ff9800",
            }

            with col1:
                _phases = list(_bd.keys())
                _wall_vals = [_bd[p]["wall_pct"] for p in _phases]
                _colors = [_phase_colors.get(p, "#607d8b") for p in _phases]

                fig_pie = go.Figure(go.Pie(
                    labels=[p.title() for p in _phases],
                    values=_wall_vals,
                    marker=dict(colors=_colors),
                    textinfo="label+percent",
                    hole=0.3,
                ))
                fig_pie.update_layout(
                    title="Wall-Clock Distribution by Phase",
                    height=350,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                _ipc_v = []
                _miss_v = []
                _labels = []
                _sizes = []
                _clrs = []
                for _p in _phases:
                    _d = _bd[_p]
                    if _d["avg_ipc"] is not None and _d["avg_cache_miss_pct"] is not None:
                        _ipc_v.append(_d["avg_ipc"])
                        _miss_v.append(_d["avg_cache_miss_pct"])
                        _labels.append(_p.title())
                        _sizes.append(max(_d["wall_pct"] * 0.8, 10))
                        _clrs.append(_phase_colors.get(_p, "#607d8b"))

                if _ipc_v:
                    fig_scatter = go.Figure(go.Scatter(
                        x=_ipc_v, y=_miss_v,
                        mode="markers+text",
                        text=_labels,
                        textposition="top center",
                        marker=dict(size=_sizes, color=_clrs, opacity=0.8),
                    ))
                    fig_scatter.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.4,
                                         annotation_text="Memory-bound threshold")
                    fig_scatter.add_vline(x=2.0, line_dash="dash", line_color="gray", opacity=0.4,
                                         annotation_text="Compute-efficient threshold")
                    fig_scatter.update_layout(
                        title="Hardware Signature per Phase",
                        xaxis_title="IPC",
                        yaxis_title="Cache Miss %",
                        height=350,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("No L3 perf counter data for hardware signature plot. Enable perf access.")

            st.markdown("### Phase Details")
            _tbl = []
            for _p, _d in _bd.items():
                _tbl.append({
                    "Phase": PHASE_LABELS.get(_p, _p.title()),
                    "Wall %": _d["wall_pct"],
                    "CPU %": _d["cpu_pct"],
                    "Duration (ms)": _d["wall_ms"],
                    "Avg IPC": round(_d["avg_ipc"], 2) if _d["avg_ipc"] is not None else None,
                    "Cache Miss %": round(_d["avg_cache_miss_pct"], 1) if _d["avg_cache_miss_pct"] is not None else None,
                    "HW Pattern": _d["pattern"],
                    "Spans": _d["span_count"],
                })
            st.dataframe(_tbl, use_container_width=True)

            if _inflection:
                st.markdown("---")
                st.markdown("### Inflection Point Detected")
                st.error(
                    f"At **iteration {_inflection['iteration']}**, cumulative orchestration time "
                    f"(Retrieve + Act + Commit = {_inflection['cumulative_other_s']:.2f}s) exceeded "
                    f"cumulative inference time ({_inflection['cumulative_reason_s']:.2f}s). "
                    f"Ratio: **{_inflection['ratio']:.2f}x**.\n\n"
                    f"**Implication**: CPU optimization of non-inference phases now yields more ROI "
                    f"than model-level optimizations (quantization, speculative decoding)."
                )

            st.markdown("---")
            st.markdown("### Per-Phase Optimization Map")
            _sol_data = []
            for _p, _sols in _r.evidence.get("phase_solutions", {}).items():
                if _sols:
                    _sol_data.append({
                        "Phase": _p.title(),
                        "Pattern": _bd.get(_p, {}).get("pattern", "—"),
                        "Solutions": "; ".join(_sols),
                    })
            if _sol_data:
                st.dataframe(_sol_data, use_container_width=True)

            if _r.recommendations:
                st.markdown("### Key Findings")
                for _rec in _r.recommendations:
                    st.info(_rec)

            st.caption(f"Data source: `{_phase_src}`")


elif demo_page == "SWE-Bench::Agent Performance":
    st.title("SWE-Bench: Agent Performance")
    st.markdown("""
    SWE-Bench evaluates agents on real GitHub issues from popular Python repositories.
    Measures patch generation accuracy, test pass rate, and completion time.
    """)

    _swe_results_dir = None
    for _cand in [
        Path(f"{_TMP}/agentsysperf_swe_bench"),
        Path(f"{_TMP}/agentsysperf_scratch/swe_bench"),
    ]:
        if _cand.exists() and (list(_cand.glob("*.json")) or list(_cand.glob("*results*"))):
            _swe_results_dir = _cand
            break

    if _swe_results_dir is None:
        st.warning(
            "No SWE-Bench results available. Run:\n\n"
            "```bash\npoetry run python -m src.benchmarks.swe_bench.adapter "
            "--dataset princeton-nlp/SWE-bench_Lite --split test --num-tasks 5\n```"
        )
    else:
        # dict.fromkeys dedupes: a file named e.g. swe_bench_results.json
        # matches both globs, which would otherwise list it twice.
        _swe_files = list(dict.fromkeys(
            list(_swe_results_dir.glob("*results*.json"))
            + list(_swe_results_dir.glob("*swe*.json"))
        ))
        if _swe_files:
            _swe_data = json.loads(_swe_files[0].read_text())
            if isinstance(_swe_data, list):
                st.dataframe(_swe_data, use_container_width=True)
            elif isinstance(_swe_data, dict):
                st.json(_swe_data)
        else:
            st.info(f"Results directory found at `{_swe_results_dir}` but no parseable result files.")

    st.markdown("---")
    st.markdown("### SWE-Bench Datasets")
    st.dataframe([
        # Instances must be uniformly typed: mixing int with the string "2,294"
        # makes Arrow fail the column conversion and the table renders as an
        # exception instead of data.
        {"Dataset": "SWE-bench_Lite", "Instances": 300, "Repos": "12 popular Python repos"},
        {"Dataset": "SWE-bench_Verified", "Instances": 500, "Repos": "Human-verified subset"},
        {"Dataset": "SWE-bench (full)", "Instances": 2294, "Repos": "12 repos, all difficulty levels"},
    ], use_container_width=True, hide_index=True)

    # ─── Phase Analysis (merged) ─────────────────────────────────────────
    st.markdown("---")
    st.header("Agentic Pipeline Phase Analysis")
    st.markdown("""
    Decomposes agentic workloads into 5 pipeline phases and characterizes
    each phase's hardware signature. Maps patterns to Intel Xeon optimizations.
    """)

    from src.analyzers.phase_profiler import PhaseProfiler, PHASE_LABELS

    _PHASE_DATA_DIRS_SWE = [
        Path(f"{_TMP}/agentsysperf_swe_bench"),
        Path(f"{_TMP}/agentsysperf_phase_swe"),
    ]

    _raw = _data.get_records(
        _BENCH_SLUG.get(_nav_section, _nav_section),
        fallback_paths=_PHASE_DATA_DIRS_SWE,
    )
    _phase_src = _data.latest_run_id(_BENCH_SLUG.get(_nav_section, _nav_section)) or "legacy JSON"

    if not _raw:
        st.warning(
            "No phase-tagged data available for **SWE-Bench**. Run:\n\n"
            "```\npoetry run python examples/run_phase_profiler_swe_bench.py --dry-run\n```"
        )
    else:
        from src.protocols import MeasurementRecord
        _records = [MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"]) for r in _raw]

        _profiler = PhaseProfiler()
        _results = list(_profiler.analyze(_records))

        if not _results:
            st.error("PhaseProfiler produced no results from the loaded data.")
        else:
            _r = _results[0]
            _bd = _r.evidence.get("phase_breakdown", {})
            _inflection = _r.evidence.get("inflection")

            st.markdown("### 5-Phase Agentic Pipeline")
            st.code("""
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ 01 ADMIT │──▶│02 RETRIEVE│──▶│ 03 REASON│──▶│  04 ACT  │──▶│05 COMMIT │
    │ Auth/    │   │ Vector/  │   │   LLM    │   │ Tool/API │   │Writeback │
    │ Policy   │   │ Rerank   │   │ Inference│   │ Code Exec│   │  Audit   │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
            """, language="text")

            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            with _mc1:
                _dominant = _r.verdict.replace("phase_profile_", "")
                st.metric("Dominant Phase", _dominant.title())
            with _mc2:
                st.metric("Total Wall-Clock", f"{_r.evidence['total_wall_ms']:.0f} ms")
            with _mc3:
                st.metric("Iterations", _r.evidence.get("iteration_count", 0))
            with _mc4:
                if _inflection:
                    st.metric("Inflection", f"iter {_inflection['iteration']}")
                else:
                    st.metric("Inflection", "Not reached")

            st.markdown("---")

            col1, col2 = st.columns(2)

            _phase_colors = {
                "admit": "#9e9e9e",
                "retrieve": "#2196f3",
                "reason": "#f44336",
                "act": "#4caf50",
                "commit": "#ff9800",
            }

            with col1:
                _phases = list(_bd.keys())
                _wall_vals = [_bd[p]["wall_pct"] for p in _phases]
                _colors = [_phase_colors.get(p, "#607d8b") for p in _phases]

                fig_pie = go.Figure(go.Pie(
                    labels=[p.title() for p in _phases],
                    values=_wall_vals,
                    marker=dict(colors=_colors),
                    textinfo="label+percent",
                    hole=0.3,
                ))
                fig_pie.update_layout(
                    title="Wall-Clock Distribution by Phase",
                    height=350,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                _ipc_v = []
                _miss_v = []
                _labels = []
                _sizes = []
                _clrs = []
                for _p in _phases:
                    _d = _bd[_p]
                    if _d["avg_ipc"] is not None and _d["avg_cache_miss_pct"] is not None:
                        _ipc_v.append(_d["avg_ipc"])
                        _miss_v.append(_d["avg_cache_miss_pct"])
                        _labels.append(_p.title())
                        _sizes.append(max(_d["wall_pct"] * 0.8, 10))
                        _clrs.append(_phase_colors.get(_p, "#607d8b"))

                if _ipc_v:
                    fig_scatter = go.Figure(go.Scatter(
                        x=_ipc_v, y=_miss_v,
                        mode="markers+text",
                        text=_labels,
                        textposition="top center",
                        marker=dict(size=_sizes, color=_clrs, opacity=0.8),
                    ))
                    fig_scatter.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.4,
                                         annotation_text="Memory-bound threshold")
                    fig_scatter.add_vline(x=2.0, line_dash="dash", line_color="gray", opacity=0.4,
                                         annotation_text="Compute-efficient threshold")
                    fig_scatter.update_layout(
                        title="Hardware Signature per Phase",
                        xaxis_title="IPC",
                        yaxis_title="Cache Miss %",
                        height=350,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("No L3 perf counter data for hardware signature plot. Enable perf access.")

            st.markdown("### Phase Details")
            _tbl = []
            for _p, _d in _bd.items():
                _tbl.append({
                    "Phase": PHASE_LABELS.get(_p, _p.title()),
                    "Wall %": _d["wall_pct"],
                    "CPU %": _d["cpu_pct"],
                    "Duration (ms)": _d["wall_ms"],
                    "Avg IPC": round(_d["avg_ipc"], 2) if _d["avg_ipc"] is not None else None,
                    "Cache Miss %": round(_d["avg_cache_miss_pct"], 1) if _d["avg_cache_miss_pct"] is not None else None,
                    "HW Pattern": _d["pattern"],
                    "Spans": _d["span_count"],
                })
            st.dataframe(_tbl, use_container_width=True)

            if _inflection:
                st.markdown("---")
                st.markdown("### Inflection Point Detected")
                st.error(
                    f"At **iteration {_inflection['iteration']}**, cumulative orchestration time "
                    f"(Retrieve + Act + Commit = {_inflection['cumulative_other_s']:.2f}s) exceeded "
                    f"cumulative inference time ({_inflection['cumulative_reason_s']:.2f}s). "
                    f"Ratio: **{_inflection['ratio']:.2f}x**.\n\n"
                    f"**Implication**: CPU optimization of non-inference phases now yields more ROI "
                    f"than model-level optimizations (quantization, speculative decoding)."
                )

            st.markdown("---")
            st.markdown("### Per-Phase Optimization Map")
            _sol_data = []
            for _p, _sols in _r.evidence.get("phase_solutions", {}).items():
                if _sols:
                    _sol_data.append({
                        "Phase": _p.title(),
                        "Pattern": _bd.get(_p, {}).get("pattern", "—"),
                        "Solutions": "; ".join(_sols),
                    })
            if _sol_data:
                st.dataframe(_sol_data, use_container_width=True)

            if _r.recommendations:
                st.markdown("### Key Findings")
                for _rec in _r.recommendations:
                    st.info(_rec)

            st.caption(f"Data source: `{_phase_src}`")


elif demo_page in ("Terminal-Bench::Hardware Analysis", "Tau-Bench::Hardware Analysis", "SWE-Bench::Hardware Analysis"):
    st.title(f"{_nav_section}: EMON Top-Down Microarchitecture Analysis")
    st.markdown("""
    Intel EMON collects hardware performance counters during benchmark execution.
    pyEDP post-processes the raw data into TMA (Top-down Microarchitecture Analysis) metrics
    that pinpoint whether workloads are frontend-bound, backend-bound, or retiring efficiently.
    """)

    # ─── Benchmark KPIs (from measurement records) ───────────────────────
    _EMON_KPI_DIRS = {
        "Terminal-Bench": [
            Path(f"{_TMP}/agentsysperf_emon_tb2"),
            Path(f"{_TMP}/agentsysperf_phase_emon_tb2"),
            Path(f"{_TMP}/agentsysperf_phase_tb2_dry"),
        ],
        "Tau-Bench": [
            Path(f"{_TMP}/agentsysperf_tau_bench"),
            Path(f"{_TMP}/agentsysperf_emon_tau"),
        ],
        "SWE-Bench": [
            Path(f"{_TMP}/agentsysperf_swe_emon"),
            Path(f"{_TMP}/agentsysperf_swe_bench"),
        ],
    }
    _kpi_records = _data.get_records(
        _BENCH_SLUG.get(_nav_section, _nav_section),
        fallback_paths=_EMON_KPI_DIRS.get(_nav_section, []),
        layer="l1",
    )
    if _kpi_records:
        st.markdown("### Benchmark KPIs")
        # Extract per-task metrics from L1 records
        _tasks_map = {}
        for _r in _kpi_records:
            _sid = _r["span_id"]
            _p = _r["payload"] if isinstance(_r["payload"], dict) else {}
            # Group by task: span_id format is "benchmark/task_name/turn_N_type"
            _parts = _sid.split("/")
            _task_key = "/".join(_parts[:2]) if len(_parts) >= 3 else _sid
            if _task_key not in _tasks_map:
                _tasks_map[_task_key] = {"turns": 0, "llm_turns": 0, "cmd_turns": 0,
                                         "total_dur_ms": 0, "llm_dur_ms": 0, "cmd_dur_ms": 0,
                                         "cpu_pct_sum": 0, "cpu_count": 0}
            _tm = _tasks_map[_task_key]
            _dur_ms = _p.get("duration_us", 0) / 1000
            _tm["turns"] += 1
            _tm["total_dur_ms"] += _dur_ms
            _phase = _p.get("phase", "")
            if _phase == "reason" or "llm" in _sid:
                _tm["llm_turns"] += 1
                _tm["llm_dur_ms"] += _dur_ms
            elif _phase == "act" or "cmd" in _sid:
                _tm["cmd_turns"] += 1
                _tm["cmd_dur_ms"] += _dur_ms
            _cpu = _p.get("cpu_pct_mean", 0)
            if _cpu:
                _tm["cpu_pct_sum"] += _cpu
                _tm["cpu_count"] += 1

        _n_tasks = len(_tasks_map)
        _total_turns = sum(t["turns"] for t in _tasks_map.values())
        _total_dur = sum(t["total_dur_ms"] for t in _tasks_map.values())
        _total_llm_dur = sum(t["llm_dur_ms"] for t in _tasks_map.values())
        _total_cmd_dur = sum(t["cmd_dur_ms"] for t in _tasks_map.values())
        _avg_cpu = (sum(t["cpu_pct_sum"] for t in _tasks_map.values()) /
                    max(sum(t["cpu_count"] for t in _tasks_map.values()), 1))

        _kc1, _kc2, _kc3, _kc4 = st.columns(4)
        _kc1.metric("Tasks", _n_tasks)
        _kc2.metric("Total Turns", _total_turns)
        _kc3.metric("Total Duration", f"{_total_dur/1000:.1f}s")
        _kc4.metric("Avg CPU %", f"{_avg_cpu:.0f}%")

        _kc5, _kc6, _kc7, _kc8 = st.columns(4)
        _kc5.metric("LLM Calls", sum(t["llm_turns"] for t in _tasks_map.values()))
        _kc6.metric("Tool Calls", sum(t["cmd_turns"] for t in _tasks_map.values()))
        _llm_pct = (_total_llm_dur / _total_dur * 100) if _total_dur > 0 else 0
        _kc7.metric("LLM Time %", f"{_llm_pct:.0f}%")
        _avg_turns = _total_turns / _n_tasks if _n_tasks else 0
        _kc8.metric("Avg Turns/Task", f"{_avg_turns:.1f}")

        # Per-task latency breakdown table
        if _n_tasks <= 20:
            _task_rows = []
            for _tk, _tv in sorted(_tasks_map.items()):
                _task_name = _tk.split("/")[-1] if "/" in _tk else _tk
                _task_rows.append({
                    "Task": _task_name,
                    "Turns": _tv["turns"],
                    "Duration (ms)": round(_tv["total_dur_ms"], 0),
                    "LLM (ms)": round(_tv["llm_dur_ms"], 0),
                    "Tool (ms)": round(_tv["cmd_dur_ms"], 0),
                    "Avg CPU %": round(_tv["cpu_pct_sum"] / max(_tv["cpu_count"], 1), 1),
                })
            st.dataframe(_task_rows, use_container_width=True, hide_index=True)

        # Per-turn latency chart
        _turn_durs = []
        for _r in _kpi_records:
            _p = _r["payload"] if isinstance(_r["payload"], dict) else {}
            _dur_ms = _p.get("duration_us", 0) / 1000
            _phase = _p.get("phase", "unknown")
            if _phase == "reason" or "llm" in _r["span_id"]:
                _phase = "LLM Inference"
            elif _phase == "act" or "cmd" in _r["span_id"]:
                _phase = "Tool Execution"
            _turn_durs.append({"Turn": len(_turn_durs), "Duration (ms)": _dur_ms, "Phase": _phase})

        if _turn_durs:
            import pandas as pd
            _turn_df = pd.DataFrame(_turn_durs)
            _fig_kpi = go.Figure()
            for _ph in _turn_df["Phase"].unique():
                _sub = _turn_df[_turn_df["Phase"] == _ph]
                _fig_kpi.add_trace(go.Bar(x=_sub["Turn"], y=_sub["Duration (ms)"], name=_ph))
            _fig_kpi.update_layout(
                title="Per-Turn Latency Breakdown",
                xaxis_title="Turn #",
                yaxis_title="Duration (ms)",
                barmode="stack",
                height=300,
                margin=dict(l=50, r=30, t=40, b=30),
            )
            st.plotly_chart(_fig_kpi, use_container_width=True)

        st.markdown("---")

    # Look for EMON CSV results (benchmark-specific directories)
    _EMON_DATA_DIRS = {
        "Terminal-Bench": [
            # Mixed-workload density EMON (core-filtered TMA at d80 and d192).
            Path(f"{_TMP}/agentsysperf_mixed4_d192_emon/d192_spread_mixed"),
            Path(f"{_TMP}/agentsysperf_mixed4_emon/d80_spread_mixed"),
            Path(f"{_TMP}/agentsysperf_emon_tb2"),
            Path(f"{_TMP}/emon_tb2_test"),
        ],
        "Tau-Bench": [
            Path(f"{_TMP}/agentsysperf_emon_tau"),
            Path(f"{_TMP}/agentsysperf_tau_bench"),
        ],
        "SWE-Bench": [
            Path(f"{_TMP}/agentsysperf_swe_emon"),
            Path(f"{_TMP}/agentsysperf_swe_bench"),
        ],
    }

    # Store-first (artifacts registered for the latest run), verbatim glob
    # fallback. EmonAnalyzer needs a real on-disk CSV, so get_artifact_path
    # only returns existing files.
    _emon_csv = _data.get_artifact_path(
        _BENCH_SLUG.get(_nav_section, _nav_section),
        kind="emon_csv",
        fallback_dirs=_EMON_DATA_DIRS.get(_nav_section, []),
        # Prefer the *details* (wide) CSV: this tab's _find_metric() searches
        # COLUMN names, which only exist in the wide per-sample layout. The
        # *summary* CSV is tall (metrics as rows) and yields no column matches.
        fallback_patterns=("*_system_view_details.csv", "*_socket_view_details.csv",
                           "*_system_view_summary.csv", "*_socket_view_summary.csv"),
    )

    if _emon_csv is None:
        st.warning(
            "No EMON data available. EMON TMA analysis requires the separate "
            "Intel-only agentsysperf-emon plugin (Intel host + SEP driver)."
        )
        st.markdown("---")
        st.markdown("### 5-Step EMON Pipeline")
        st.code("""
    ┌──────────────────────────────────────────────────────────────┐
    │ Step 1: LOAD        Read pyEDP CSV into metric DataFrame     │
    ├──────────────────────────────────────────────────────────────┤
    │ Step 2: TRIAGE      Classify workload (compute/memory/mixed) │
    ├──────────────────────────────────────────────────────────────┤
    │ Step 3: LAYER       TMA L1→L2→L3 drill-down                 │
    │                     Frontend Bound → Fetch Latency?          │
    │                     Backend Bound  → Memory Bound?           │
    │                     Memory Bound   → L1/L2/L3/DRAM?         │
    ├──────────────────────────────────────────────────────────────┤
    │ Step 4: FINDINGS    Actionable bottleneck list               │
    │                     severity + confidence + evidence         │
    ├──────────────────────────────────────────────────────────────┤
    │ Step 5: REPORT      Human-readable summary + recommendations│
    └──────────────────────────────────────────────────────────────┘
        """, language="text")
    else:
        import pandas as pd

        _df = pd.read_csv(_emon_csv)
        _num_cols = len(_df.columns)
        _num_rows = len(_df)
        st.success(f"Loaded {_num_cols} metrics x {_num_rows} samples from `{_emon_csv.name}`")

        def _find_metric(df, keyword):
            """Find a column containing keyword and return its mean value."""
            for col in df.columns:
                if keyword.lower() in col.lower():
                    try:
                        return float(df[col].mean())
                    except (ValueError, TypeError):
                        pass
            return None

        # Try to extract TMA L1 metrics (column names like "metric_TMA_Frontend_Bound(%)")
        _tma_search = {
            "Frontend_Bound": "tma_frontend_bound(%)",
            "Backend_Bound": "tma_backend_bound(%)",
            "Bad_Speculation": "tma_bad_speculation(%)",
            "Retiring": "tma_retiring(%)",
        }
        _tma_vals = {}
        for _label, _keyword in _tma_search.items():
            _val = _find_metric(_df, _keyword)
            if _val is not None:
                _tma_vals[_label] = _val

        if _tma_vals:
            st.markdown("### TMA Level 1 Breakdown")
            _cols = st.columns(len(_tma_vals))
            _colors = {"Frontend_Bound": "#ff9800", "Backend_Bound": "#d32f2f",
                       "Bad_Speculation": "#9c27b0", "Retiring": "#4caf50"}
            for _i, (_k, _v) in enumerate(_tma_vals.items()):
                with _cols[_i]:
                    st.metric(_k.replace("_", " "), f"{_v:.1f}%")

            # TMA bar chart
            fig_tma = go.Figure(go.Bar(
                x=list(_tma_vals.values()),
                y=[k.replace("_", " ") for k in _tma_vals.keys()],
                orientation="h",
                marker_color=[_colors.get(k, "#607d8b") for k in _tma_vals.keys()],
                text=[f"{v:.1f}%" for v in _tma_vals.values()],
                textposition="outside",
            ))
            fig_tma.update_layout(
                title="TMA Level 1 (% of pipeline slots)",
                xaxis_title="%",
                height=250,
                margin=dict(l=120, r=50, t=40, b=30),
            )
            st.plotly_chart(fig_tma, use_container_width=True)

        # Key derived metrics (search column names)
        st.markdown("### Key Metrics")
        _key_search = {
            "CPI": "metric_cpi",
            "IPC": "core ipc",
            "LLC MPI": "llc mpi",
            "L1D MPI": "l1d mpi",
            "NUMA Local %": "numa %_reads addressed to local",
            "NUMA Remote %": "numa %_reads addressed to remote",
            "Mem BW (MB/s)": "memory bandwidth total",
            "CPU Util %": "cpu utilization %",
        }
        _found_metrics = {}
        for _label, _keyword in _key_search.items():
            _val = _find_metric(_df, _keyword)
            if _val is not None:
                _found_metrics[_label] = _val

        if _found_metrics:
            _mcols = st.columns(min(len(_found_metrics), 4))
            for _i, (_k, _v) in enumerate(_found_metrics.items()):
                with _mcols[_i % 4]:
                    st.metric(_k, f"{_v:.2f}")

        # Run EmonAnalyzer (5-step pipeline)
        st.markdown("---")
        st.markdown("### EmonAnalyzer — 5-Layer Analysis")
        try:
            from src.protocols import MeasurementRecord
            _analyzer = _emon_analyzer()
            if _analyzer is None:
                st.info(
                    "EMON findings require the Intel-only **agentsysperf-emon** "
                    "plugin, which is not installed. The metric tables above are "
                    "read straight from the EMON CSV; install the plugin to run "
                    "the 5-step EmonAnalyzer pipeline (workload triage + ranked "
                    "findings)."
                )
                _emon_results = []
            else:
                _record = MeasurementRecord(span_id="demo", layer="emon", payload={"csv_path": str(_emon_csv)})
                _emon_results = list(_analyzer.analyze([_record]))
            if _emon_results:
                for _er in _emon_results:
                    _ev = _er.evidence

                    # Workload classification
                    _wclass = _ev.get("workload_class", "unknown")
                    _wdesc = _ev.get("workload_description", "")
                    _hw_bn = str(_ev.get("signals", {}).get("hw_bottleneck", "") or "")
                    _hw_tag = f" | HW bottleneck: **{_hw_bn.replace('_', ' ')}**" if _hw_bn else ""
                    st.markdown(f"**Workload class:** `{_wclass}` — {_wdesc}{_hw_tag}")
                    _acols = st.columns(3)
                    _acols[0].metric("Total Findings", _ev.get("total_findings", 0))
                    _acols[1].metric("Actionable", _ev.get("actionable_findings", 0))
                    _acols[2].metric("Addressable Gain", _ev.get("addressable_gain", "N/A"))

                    # Root cause ranking table
                    st.markdown("#### Root Cause Ranking")
                    _root_causes = _ev.get("root_causes", [])
                    if _root_causes:
                        _rc_rows = []
                        for _rc in _root_causes:
                            _rc_rows.append({
                                "Rank": _rc["rank"],
                                "Severity": _rc["severity"],
                                "Category": _rc["category"],
                                "Headline": _rc["headline"],
                                "Gain": _rc["gain_range"],
                                "Fix": _rc["fix"],
                            })
                        st.dataframe(_rc_rows, use_container_width=True, hide_index=True)

                    # Per-layer breakdown
                    _layer_names = {
                        "1": "Performance",
                        "2": "TMA",
                        "3": "Memory",
                        "4": "Coherency",
                        "5": "Serialization",
                        "6": "I/O",
                        "7": "Power",
                        "10": "Tail Latency",
                    }
                    _layer_summary = _ev.get("layer_summary", {})
                    _layers_run = _ev.get("layers_run", {})
                    st.markdown("#### Per-Layer Breakdown")
                    for _lnum_str, _lname in sorted(_layer_names.items(), key=lambda x: int(x[0])):
                        _findings = _layer_summary.get(_lnum_str, [])
                        _run_info = _layers_run.get(_lnum_str, {})
                        _was_run = _run_info.get("ran", _lnum_str in _layer_summary)
                        _skip_reason = _run_info.get("reason", "")

                        if _findings:
                            with st.expander(f"Layer {_lnum_str}: {_lname} ({len(_findings)} findings)", expanded=False):
                                for _f in _findings:
                                    _sev = _f["severity"]
                                    _icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "INFO": "⚪"}.get(_sev, "⚪")
                                    _gain = _f.get("gain_pct", (0, 0, 0))
                                    st.markdown(
                                        f"{_icon} **[{_sev}]** {_f['category']}: {_f['headline']}  \n"
                                        f"&nbsp;&nbsp;&nbsp;&nbsp;Gain estimate: {_gain[0]}–{_gain[2]}% (central: {_gain[1]}%)"
                                    )
                        elif _was_run:
                            st.markdown(f"**Layer {_lnum_str}: {_lname}** — ran, no findings (all metrics within thresholds)")
                        else:
                            st.markdown(f"**Layer {_lnum_str}: {_lname}** — skipped ({_skip_reason or 'not relevant for this workload class'})")

                    # Recommendations
                    if _er.recommendations:
                        st.markdown("#### Recommendations")
                        for _rec in _er.recommendations:
                            st.markdown(f"- {_rec}")
            else:
                st.info("No actionable findings from EMON data.")
        except Exception as _e:
            st.error(f"EmonAnalyzer error: {_e}")

        st.caption(f"Data source: `{_emon_csv}`")


elif demo_page in ("Terminal-Bench::Scaling", "Tau-Bench::Scaling", "SWE-Bench::Scaling"):
    st.title(f"{_nav_section}: Scaling Analysis")
    st.markdown("Concurrency sweeps, density studies, and hardware resource scaling under agent load.")

    _scaling_tab_cs, _scaling_tab_chw, _scaling_tab_ds, _scaling_tab_hw = st.tabs(["Concurrency Sweep", "Concurrency HW Study", "Synth Density Study", "HW Resources"])

    # ═══ Tab 1: Concurrency Sweep ════════════════════════════════════════
    with _scaling_tab_cs:
        st.caption(
            "How many agents per vCPU before this Xeon saturates? Sweeps "
            "density = concurrency ÷ vCPU and finds the knee + bottleneck "
            "(ScalingAnalyzer). Normalized by core count, so the knee is comparable "
            "across SKUs. ↔ the Density Study page does the multi-dimensional why."
        )

        from src.storage.sqlite_store import SQLiteResultStore
        from src.dashboard.scaling_views import (
            limiting_factor_band_figure,
            bottleneck_heatstrip_figure,
            throughput_knee_figure,
            throughput_compare_figure,
            efficiency_figure,
            efficiency_compare_figure,
            cpu_runqueue_figure,
            cpu_runqueue_compare_figure,
            saturation_signature_figure,
            per_task_profile_figure,
            resource_spectrum_figure,
            signature_compare_figure,
            BOTTLENECK_STYLE,
        )

        # Resolve a store that HAS sweeps: canonical store first (read-only), then
        # legacy /tmp dirs. Returns (store, [sweep rows newest-first]) or None.
        def _cs_find_store():
            try:
                _s = SQLiteResultStore.open(read_only=True)
                if _s.query_sweeps():
                    return _s, _s.query_sweeps()
            except Exception:
                pass
            for _cand in [Path(f"{_TMP}/agentsysperf_sweep"), Path(f"{_TMP}/agentsysperf_sweep_dry")]:
                if (_cand / "agentsysperf_results.db").exists():
                    try:
                        _s = SQLiteResultStore(_cand, read_only=True)
                        if _s.query_sweeps():
                            return _s, _s.query_sweeps()
                    except Exception:
                        pass
            return None

        # Is a sweep synthetic (--dry-run) vs measured? Prefer the persisted
        # data_source flag (in metadata JSON); fall back to a heuristic for sweeps
        # recorded before the flag existed (no replay fixture + flat iowait).
        def _cs_is_synthetic(meta, points):
            ds = (meta.get("metadata") or {}).get("data_source")
            if ds:
                return ds == "synthetic"
            if meta.get("replay_fixture"):
                return False
            return all((p.get("iowait_pct_avg") or 0) == 0 for p in points) if points else False

        # Bundle everything one sweep needs to render (single OR as a compare side).
        def _cs_load(meta):
            sid = meta["sweep_id"]
            pts = _cs_store.query_sweep_points(sid)
            vs = _cs_store.query_verdicts(sid, analyzer_name="scaling")
            v = vs[0] if vs else None
            ev = v["evidence"] if v else {}
            return {
                "meta": meta, "sweep_id": sid, "points": pts, "verdict": v,
                "evidence": ev, "knee": ev.get("knee"), "bottleneck": ev.get("bottleneck"),
                "per_task": ev.get("per_task") or {},
                "vcpu_basis": meta.get("vcpu_basis") or (ev.get("knee") or {}).get("vcpu_basis") or 0,
                "synthetic": _cs_is_synthetic(meta, pts),
            }

        # Render the prominent verdict block for one sweep: synthetic badge, colored
        # bottleneck callout, always-visible recommendations, at-knee evidence strip.
        def _cs_render_verdict(d, *, compact=False):
            m = d["meta"]
            if d["synthetic"]:
                st.warning("⚗️ **Synthetic sweep (--dry-run)** — modeled cells, illustrative only. "
                           "Not a hardware measurement; no per-task breakdown.")
            v, knee, bn = d["verdict"], d["knee"], d["bottleneck"]
            if v is None:
                st.info("No scaling verdict for this sweep.")
                return
            color, label = BOTTLENECK_STYLE.get(bn or "", ("#616161", bn or "—"))
            # Colored bottleneck callout band (prominent, not a bare metric chip).
            st.markdown(
                f"<div style='background:{color}1a;border-left:5px solid {color};"
                f"padding:10px 14px;border-radius:4px;margin:6px 0'>"
                f"<b style='color:{color}'>Bottleneck: {label}</b> — "
                f"verdict <code>{v['verdict']}</code> · confidence {v['confidence']:.0%}</div>",
                unsafe_allow_html=True,
            )
            if knee:
                _cells = st.columns(4 if not compact else 2)
                _cells[0].metric("Knee density", f"{knee['density']:g}")
                _cells[1].metric("Agents @ knee", knee.get("concurrency", "—"))
                if not compact:
                    _cells[2].metric("Throughput @ knee", knee.get("throughput_at_knee", "—"))
                    _cells[3].metric("Eff. (work/core)", knee.get("efficiency_at_knee", "—"))
            else:
                st.caption("No knee within the swept density range (headroom remained, or "
                           "insufficient points) — the box did not saturate here.")
            # Recommendations — always visible (was buried in a collapsed expander).
            if v.get("recommendations"):
                st.markdown("**Recommendations**")
                for _rec in v["recommendations"]:
                    st.markdown(f"- {_rec}")

        _cs_found = _cs_find_store()

        if _cs_found is None:
            st.info(
                "No concurrency sweep found. Run:\n\n"
                "```\nagentsysperf sweep run --dry-run\n```\n\n"
                "`--dry-run` cells are synthetic modeled points (badged as such). "
                "For measurements, run a real sweep with `--fixture <fixture.jsonl>`. "
                "Then reload."
            )
        else:
            _cs_store, _cs_sweeps = _cs_found

            def _cs_label(sw):
                _llm = "replay" if sw.get("replay_fixture") else "off"
                _syn = " · ⚗️synthetic" if _cs_is_synthetic(sw, None) else ""
                # Task and agent belong in the label, not just the sweep_id: sweep
                # shape and the observed knee depend on both, and cross-task comparisons
                # should be interpreted on a quota-normalized axis (e.g. quota_demand)
                # when tasks declare different per-container CPU quotas (see docs/adr/0001).
                _m = sw.get("metadata") or {}
                _task = _m.get("task")
                _agent = _m.get("agent") or sw.get("model")
                _what = f" · {_task}" if _task else ""
                _who = f" · agent {_agent}" if _agent else ""
                return (f"{sw['sweep_id']}{_what}{_who} · "
                        f"basis {sw.get('vcpu_basis','?')} · "
                        f"NUMA {sw.get('numa_policy','?')} · LLM {_llm}{_syn}")

            # Order the most-renderable sweep first. Sorting by recency alone
            # put an aborted 0-point sweep at index 0, so the tab opened on
            # empty charts while a populated sweep sat further down the list.
            # Rank by (has scaling verdict, number of points) so a sweep with
            # a knee/bottleneck beats a 1-point stub. Stable sort, so recency
            # still orders sweeps that tie.
            def _cs_rank(sw):
                _sid = sw["sweep_id"]
                _n = len(_cs_store.query_sweep_points(_sid))
                _has_v = bool(_cs_store.query_verdicts(_sid, analyzer_name="scaling"))
                return (0 if _has_v else 1, -_n)

            _cs_sweeps = sorted(_cs_sweeps, key=_cs_rank)
            _cs_by_label = {_cs_label(sw): sw for sw in _cs_sweeps}
            _cs_labels = list(_cs_by_label.keys())

            _cs_view = st.radio("View", ["Single sweep", "Compare two sweeps"],
                                horizontal=True)

            # ── SINGLE ───────────────────────────────────────────────────────
            if _cs_view == "Single sweep":
                _pick = st.selectbox("Sweep", _cs_labels, index=0)
                d = _cs_load(_cs_by_label[_pick])
                st.caption(
                    f"Sweep `{d['sweep_id']}` · {d['meta'].get('hardware_sku','?')} · "
                    f"basis {d['meta'].get('vcpu_basis','?')} {d['meta'].get('vcpu_basis_kind','')} · "
                    f"NUMA {d['meta'].get('numa_policy','?')} · "
                    f"LLM {('replay: ' + str(d['meta'].get('replay_fixture'))) if d['meta'].get('replay_fixture') else 'off'}"
                )
                _cs_render_verdict(d)

                # ── Resource spectrum ────────────────────────────────────────
                # The panel that answers "why does it knee there", which no
                # existing figure could: the other charts show throughput and
                # CPU, so a task that saturates WITHOUT filling its CPU quota
                # looked identical to one that pegged its cores.
                st.markdown("#### What this workload does to the hardware")

                from src.analyzers.task_signature import (
                    signature_for_sweep, explain, COMPONENTS, PROVENANCE,
                )
                _sig = signature_for_sweep(d["points"])

                # WHAT WAS RUN — before any number. A reader hitting "n=2" with no
                # definition cannot start; the bullets below say "n=2" and "quota
                # demand 0.125" and neither means anything without this block.
                if _sig:
                    _m = d["meta"].get("metadata") or {}
                    _task = _m.get("task") or "unknown task"
                    _tcpus = next((p for p in d["points"]
                                   if (p.get("metadata") or {}).get("task_cpus")), None)
                    _tcpus = ((_tcpus.get("metadata") or {}).get("task_cpus")
                              if _tcpus else None)
                    _cores = d["vcpu_basis"]
                    _ns = sorted({p.concurrency for p in _sig})
                    _what = (
                        f"**The experiment.** `{_task}` was run repeatedly on "
                        f"**{_cores} pinned CPU cores** (`{_m.get('cpuset', '?')}`), "
                        f"with **{_ns[0]} to {_ns[-1]} copies running at the same "
                        f"time** — that count is what **n** means below. Each copy "
                        f"is one Docker container."
                    )
                    if _tcpus:
                        _what += (
                            f" The task's own config asks for **{_tcpus:g} "
                            f"CPU{'s' if _tcpus != 1 else ''} per copy**, and the "
                            f"kernel enforces that as a hard cap."
                        )
                    st.markdown(_what)
                    if _tcpus:
                        st.markdown(
                            f"So **quota demand = n × {_tcpus:g} ÷ {_cores}**: the share "
                            f"of the pinned cores that all the running copies are "
                            f"*entitled* to. **1.0 means they have collectively been "
                            f"promised every core there is** — the point past which the "
                            f"machine is over-committed. This is the x-axis on the chart, "
                            f"and it is used instead of raw agent count so that two "
                            f"tasks asking for different amounts of CPU can be compared "
                            f"on the same scale."
                        )
                    st.markdown(
                        "**How to read the three findings below.** The first says where "
                        "each copy ran fastest. The second says where throughput per "
                        "copy dropped more than 20% — the saturation point. The third "
                        "says *why*, by comparing what the copies were entitled to "
                        "against what they actually used."
                    )
                    st.markdown("---")

                if _sig:
                    # Reading comes BEFORE the chart: it is the conclusion, and a
                    # reader who stops after one screen should still leave with it.
                    for _line in explain(_sig):
                        st.markdown(f"- {_line}")

                    # Thin replication is a precision warning, not a footnote, and
                    # it doubles as the key for the markers-only encoding: a
                    # single-sample point is drawn without a connecting line
                    # because the shape between points is unknown.
                    _thin = [p for p in _sig if p.replicates < 3]
                    if _thin:
                        st.warning(
                            "Single- or double-sample points at "
                            + ", ".join(f"n={p.concurrency} ({p.replicates} rep)"
                                        for p in _thin)
                            + ". These have no variance estimate, so the knee "
                              "location is not established. Points are drawn as "
                              "markers without a connecting line for this reason."
                        )
                    # "Work finished per agent (vs its own best)" replaced
                    # "per-agent throughput retained", which nobody could parse.
                    # The chart still needs the reading rule stated, because every
                    # line is a % of its OWN peak — that makes the SHAPES
                    # comparable but not the magnitudes, and a reader who misses
                    # that will compare heights across lines.
                    # Panels are grouped by RESOURCE so each one is single-unit and
                    # can show real values. That is the whole reason to split them:
                    # a merged chart needed a "% of own peak" rescale that made
                    # heights meaningless.
                    st.caption(
                        "One panel per resource, all sharing the same x-axis, so a "
                        "change in one can be lined up against the others. Values "
                        "are **real** — percentages where the metric is a share, "
                        "GB/s and MB/s where it is a rate — not rescaled. "
                        "A break in a line means that resource was **not measured** "
                        "there; it never means zero. Hollow markers are points with "
                        "fewer than 3 repeats."
                    )
                    st.plotly_chart(resource_spectrum_figure(_sig),
                                    use_container_width=True)
                    st.caption(
                        "**\"Work finished per agent\"** (thick grey, in the CPU "
                        "panel) is the outcome; every other line is a candidate "
                        "explanation for it. It answers: *if one agent completed N tasks/min "
                        "when the machine was least loaded, what fraction of N is "
                        "it still completing now?* 100% = as fast as this task "
                        "ever went here; 60% = each agent is 40% slower. Total "
                        "work still rises with more agents — this is per-agent "
                        "speed, which is what a latency-sensitive deployment "
                        "cares about."
                    )

                    # Scope travels with every number. Without it a socket-wide
                    # bandwidth column and a cpuset-scoped CPU column look like
                    # the same kind of measurement, and a reader treats both as
                    # per-cell — which is wrong for four of the six.
                    _SCOPE = {
                        "declared": "declared",
                        "measured_cpuset": "cpuset·measured",
                        "derived_cpuset": "cpuset·derived",
                        "measured_socket": "socket·incl. other tenants",
                        "measured_host": "host·incl. other tenants",
                    }
                    with st.expander("Per-operating-point numbers, with scope"):
                        st.dataframe(
                            [{"n": p.concurrency, "reps": p.replicates,
                              **{f"{k} [{_SCOPE.get(PROVENANCE[k], PROVENANCE[k])}]":
                                 p.components.get(k) for k in COMPONENTS},
                              "retained": p.retention,
                              "throttled%": p.throttled_pct,
                              "unmeasured": ", ".join(p.missing) or "—"}
                             for p in _sig],
                            use_container_width=True, hide_index=True)
                        st.caption(
                            "`declared` comes from the task's own task.toml, not a "
                            "measurement. `cpuset` numbers cover only the pinned "
                            "cores. `socket`/`host` numbers include any other "
                            "workload on this machine and are **not** attributable "
                            "to this sweep alone. Blank = unmeasured, which is "
                            "different from a measured zero."
                        )
                else:
                    st.info("No signature points — this sweep predates the "
                            "resource-spectrum fields.")


                # The density-axis charts move BELOW the spectrum and behind an
                # expander. They are still the right charts, but their x-axis is
                # agents-per-vCPU, which is not comparable across tasks whose
                # declared cpus differ — the confound this section exists to
                # correct. Leading with them re-introduces the misreading.
                if d["points"]:
                    _lcpu = _num(d["evidence"], "logical_cpus")
                    if _lcpu is None:
                        _lcpu = _num(d, "vcpu_basis")
                    if _lcpu is None:
                        _lcpu = 1
                    st.markdown("#### What's limiting the box")
                    st.caption(
                        "x here is agents ÷ vCPU, **not** quota demand. Valid "
                        "within one task; not comparable across tasks whose "
                        "declared `cpus` differ."
                    )
                    # Why Memory and I/O never appear as bands. Without this a
                    # reader concludes those resources were not looked at, when in
                    # fact they were measured and did not cross a threshold — a
                    # very different statement, and the honest one.
                    _mem_min = min((p.get("mem_avail_mb_min") or 0)
                                   for p in d["points"]) or 0
                    _iow_max = max((p.get("iowait_pct_avg") or 0)
                                   for p in d["points"]) or 0
                    with st.expander(
                            "Why are there no Memory or I/O bands?", expanded=False):
                        st.markdown(
                            f"Both were measured; neither came close to a "
                            f"saturation threshold on this machine.\n\n"
                            f"- **Memory capacity:** the lowest free memory seen "
                            f"was **{_mem_min / 1024:,.0f} GB**, against a "
                            f"pressure threshold of 2 GB — about "
                            f"**{(_mem_min / 2048):,.0f}×** clear. This box has "
                            f"755 GB; this workload cannot approach it.\n"
                            f"- **I/O wait:** peaked at **{_iow_max:.3f}%** of CPU "
                            f"time against a 15% threshold — effectively zero.\n\n"
                            f"**Memory *bandwidth* is a different story and it does "
                            f"move** — see the resource spectrum above, where it "
                            f"rises with agent count. It is deliberately not scored "
                            f"as a band: scoring needs a denominator, and this "
                            f"platform's DRAM peak is *estimated* (614 GB/s inferred "
                            f"from the microarchitecture, not measured on this "
                            f"silicon). Dividing a real reading by a guessed ceiling "
                            f"would produce a percentage that looks measured. So "
                            f"bandwidth is shown in GB/s for you to judge, rather "
                            f"than as a saturation claim."
                        )
                    st.plotly_chart(bottleneck_heatstrip_figure(d["points"], logical_cpus=_lcpu),
                                    use_container_width=True)
                    st.plotly_chart(
                        limiting_factor_band_figure(d["points"], logical_cpus=_lcpu,
                                                    knee=d["knee"], bottleneck=d["bottleneck"]),
                        use_container_width=True)
                    st.plotly_chart(throughput_knee_figure(d["points"], knee=d["knee"],
                                                            bottleneck=d["bottleneck"]),
                                    use_container_width=True)
                    st.plotly_chart(efficiency_figure(d["points"], d["vcpu_basis"],
                                                      knee=d["knee"], bottleneck=d["bottleneck"]),
                                    use_container_width=True)
                    st.plotly_chart(cpu_runqueue_figure(d["points"]), use_container_width=True)
                    # Raw proof signals behind an expander (the band/heatstrip above
                    # are the exec-legible reframe of these).
                    with st.expander("Raw saturation signals (ctx-switch / iowait / memory / runqueue)"):
                        st.plotly_chart(
                            saturation_signature_figure(
                                d["points"], logical_cpus=_lcpu,
                                bottleneck=d["bottleneck"], knee=d["knee"]),
                            use_container_width=True)

                # Per-task — explicit empty-state instead of silent omission.
                st.markdown("#### Per-Task Profile")
                if d["per_task"]:
                    _task = st.selectbox("Task", sorted(d["per_task"].keys()))
                    _t = d["per_task"][_task]
                    st.plotly_chart(
                        per_task_profile_figure(_task, _t.get("curve", []),
                                                knee_density=_t.get("knee_density"),
                                                bottleneck=_t.get("bottleneck")),
                        use_container_width=True)
                elif d["synthetic"]:
                    st.info("Synthetic sweeps have no per-task breakdown — run a real "
                            "(non-dry-run) sweep to profile individual tasks.")
                else:
                    st.caption("No per-task data recorded for this sweep.")

                with st.expander("Sweep points (raw)"):
                    st.dataframe(d["points"], use_container_width=True)

            # ── COMPARE ──────────────────────────────────────────────────────
            else:
                _ca, _cb = st.columns(2)
                with _ca:
                    _pa = st.selectbox("Sweep A", _cs_labels, index=0)
                with _cb:
                    _opts_b = [l for l in _cs_labels if l != _pa] or _cs_labels
                    _pb = st.selectbox("Sweep B", _opts_b, index=0)
                A, B = _cs_load(_cs_by_label[_pa]), _cs_load(_cs_by_label[_pb])

                # One-line auto diff.
                def _kd(x):
                    return x["knee"]["density"] if x["knee"] else None
                if _kd(A) and _kd(B):
                    _delta = (_kd(B) - _kd(A)) / _kd(A) * 100
                    st.markdown(f"**Knee moved {_delta:+.0f}%** (A density {_kd(A):g} → B {_kd(B):g}) · "
                                f"bottleneck **{A['bottleneck']}** → **{B['bottleneck']}**")

                # Mirrored verdict cards.
                _cA, _cB = st.columns(2)
                with _cA:
                    st.markdown(f"##### A · `{A['sweep_id']}`")
                    _cs_render_verdict(A, compact=True)
                with _cB:
                    st.markdown(f"##### B · `{B['sweep_id']}`")
                    _cs_render_verdict(B, compact=True)

                # Delta table.
                def _row(name, a, b):
                    return {"metric": name, "A": a, "B": b}
                _ka, _kb = A["knee"] or {}, B["knee"] or {}
                import pandas as _pd
                _dt = [
                    _row("knee density", _ka.get("density"), _kb.get("density")),
                    _row("agents @ knee", _ka.get("concurrency"), _kb.get("concurrency")),
                    _row("throughput @ knee", _ka.get("throughput_at_knee"), _kb.get("throughput_at_knee")),
                    _row("efficiency (work/core)", _ka.get("efficiency_at_knee"), _kb.get("efficiency_at_knee")),
                    _row("p95 @ knee (s)", _ka.get("p95_at_knee"), _kb.get("p95_at_knee")),
                    _row("bottleneck", A["bottleneck"], B["bottleneck"]),
                    _row("confidence", A["verdict"]["confidence"] if A["verdict"] else None,
                         B["verdict"]["confidence"] if B["verdict"] else None),
                ]
                st.markdown("#### Comparison")
                st.dataframe(_pd.DataFrame(_dt), use_container_width=True, hide_index=True)

                # Overlaid curves: throughput, efficiency, CPU/runqueue.
                _sw = [
                    {"label": "A", "points": A["points"], "knee": A["knee"], "vcpu_basis": A["vcpu_basis"]},
                    {"label": "B", "points": B["points"], "knee": B["knee"], "vcpu_basis": B["vcpu_basis"]},
                ]
                st.plotly_chart(throughput_compare_figure(_sw), use_container_width=True)
                st.plotly_chart(efficiency_compare_figure(_sw), use_container_width=True)
                st.plotly_chart(cpu_runqueue_compare_figure(_sw), use_container_width=True)

                # Per-task side-by-side when BOTH have it.
                _shared = sorted(set(A["per_task"]) & set(B["per_task"]))
                if _shared:
                    st.markdown("#### Per-task (side by side)")
                    _pt = st.selectbox("Task", _shared)
                    _pa2, _pb2 = st.columns(2)
                    with _pa2:
                        _ta = A["per_task"][_pt]
                        st.plotly_chart(per_task_profile_figure("A · " + _pt, _ta.get("curve", []),
                                        knee_density=_ta.get("knee_density"), bottleneck=_ta.get("bottleneck")),
                                        use_container_width=True)
                    with _pb2:
                        _tb = B["per_task"][_pt]
                        st.plotly_chart(per_task_profile_figure("B · " + _pt, _tb.get("curve", []),
                                        knee_density=_tb.get("knee_density"), bottleneck=_tb.get("bottleneck")),
                                        use_container_width=True)
                elif A["synthetic"] or B["synthetic"]:
                    st.caption("Per-task side-by-side unavailable — at least one sweep is synthetic.")

                # ── Signature comparison ─────────────────────────────────────
                # The chart that resolves the originating question. Plotted
                # against agent count, two tasks with different declared `cpus`
                # appear to knee at 2x different densities; against quota demand
                # both sit at 1.0 and the difference collapses. What remains is
                # the genuine difference in HOW they approach it.
                st.markdown("#### Task signature comparison")
                from src.analyzers.task_signature import (
                    signature_for_sweep as _sfs, explain as _sexp,
                )
                _sa, _sb = _sfs(A["points"]), _sfs(B["points"])
                if _sa and _sb:
                    st.plotly_chart(
                        signature_compare_figure([
                            {"label": (A["meta"].get("metadata") or {}).get("task")
                                      or A["sweep_id"], "points": _sa},
                            {"label": (B["meta"].get("metadata") or {}).get("task")
                                      or B["sweep_id"], "points": _sb},
                        ]),
                        use_container_width=True)
                    _ca, _cb = st.columns(2)
                    with _ca:
                        st.markdown("**A**")
                        for _l in _sexp(_sa):
                            st.markdown(f"- {_l}")
                    with _cb:
                        st.markdown("**B**")
                        for _l in _sexp(_sb):
                            st.markdown(f"- {_l}")
                else:
                    st.caption("Signature comparison needs both sweeps to carry "
                               "quota/resource fields (re-run with the current runner).")


    # ═══ Tab 2: Concurrency HW Study ═══════════════════════════════════════════
    with _scaling_tab_chw:
        st.markdown("""
        **EMON-based hardware analysis of the concurrency sweep.** Shows how TMA
        bottleneck categories (Frontend Bound, Backend Bound, Memory Latency, etc.)
        shift as agent density increases during real TB2 tasks via Harbor.
        """)

        # Look for EMON data from sweep cells — stored per cell in the sweep output dir
        _chw_sweep_dirs = [
            # Mixed-workload density EMON cells (d{n}_spread_mixed with core-filtered TMA).
            Path(f"{_TMP}/agentsysperf_mixed4_d192_emon"),
            Path(f"{_TMP}/agentsysperf_mixed4_emon"),
            Path(f"{_TMP}/agentsysperf_sweep"),
            Path(f"{_TMP}/agentsysperf_sweep_dry"),
        ]
        _chw_emon_cells = []

        # Also check the canonical store for sweep metadata
        _chw_sweep_id = None
        if _cs_found is not None:
            if _cs_sweeps:
                _chw_sweep_id = _cs_sweeps[0].get("sweep_id")
            _chw_sweep_dirs.insert(0, _cs_store.output_dir)

        for _chw_dir in _chw_sweep_dirs:
            if not _chw_dir.exists():
                continue
            # Each cell is d{density}_r{replicate}/ with emon.dat or emon_edp CSVs
            for _cell_dir in sorted(_chw_dir.iterdir()):
                if not _cell_dir.is_dir():
                    continue
                _emon_dat = _cell_dir / "emon.dat"
                _emon_csv = None
                # Check for post-processed CSVs
                for _suf in ("_system_view_details.csv", "_system_view_summary.csv"):
                    _cand_csv = list(_cell_dir.glob(f"*{_suf}"))
                    if _cand_csv:
                        _emon_csv = _cand_csv[0]
                        break
                if _emon_dat.exists() or _emon_csv:
                    # Parse density from dir name: sweep cells are d{n}_r{rep}
                    # (e.g. d0.5_r0), mixed-density cells are d{n}_spread_mixed.
                    _dname = _cell_dir.name
                    _density_val = None
                    if _dname.startswith("d"):
                        _dtok = _dname[1:].split("_")[0]
                        try:
                            _density_val = float(_dtok)
                        except ValueError:
                            pass
                    _chw_emon_cells.append({
                        "dir": _cell_dir,
                        "density": _density_val,
                        "emon_dat": _emon_dat if _emon_dat.exists() else None,
                        "emon_csv": _emon_csv,
                        "name": _dname,
                    })
        # Deliberately no early break: the d80 and d192 cells live in
        # different dirs, so stopping at the first non-empty one left the
        # "TMA Breakdown by Density" chart with a single density point.
        # Dedupe by cell name, keeping the earlier (higher-precedence) dir.
        _chw_seen = set()
        _chw_emon_cells = [
            _c for _c in _chw_emon_cells
            if not (_c["name"] in _chw_seen or _chw_seen.add(_c["name"]))
        ]

        if not _chw_emon_cells:
            st.info(
                "No EMON hardware data from concurrency sweep cells. Run:\n\n"
                "```\nagentsysperf sweep run --emon --fixture <fixture.jsonl>\n```\n\n"
                "The `--emon` flag enables EMON collection during each sweep cell.\n"
                "After the sweep, post-process with pyEDP to generate TMA metric CSVs."
            )
        else:
            st.success(f"Found EMON data for {len(_chw_emon_cells)} sweep cells")

            # Show which cells have raw vs processed data
            _chw_summary = []
            for _cell in _chw_emon_cells:
                _chw_summary.append({
                    "Cell": _cell["name"],
                    "Density": _cell["density"],
                    "EMON .dat": "Yes" if _cell["emon_dat"] else "No",
                    "Metrics CSV": _cell["emon_csv"].name if _cell["emon_csv"] else "Not processed",
                })
            st.dataframe(_chw_summary, use_container_width=True)

            # If we have processed CSVs, show TMA breakdown per density
            _chw_with_csv = [c for c in _chw_emon_cells if c["emon_csv"] is not None]
            if _chw_with_csv:
                st.markdown("### TMA Breakdown by Density")

                import pandas as pd

                _tma_rows = []
                for _cell in sorted(_chw_with_csv, key=lambda c: c["density"] or 0):
                    try:
                        _df = pd.read_csv(_cell["emon_csv"])
                        _cols = _df.columns.tolist()

                        # Extract TMA top-level metrics. Match the exact pyEDP
                        # metric column, NOT a substring: "frontend" also matches
                        # the 10 raw FRONTEND_RETIRED.* event counters and
                        # PERF_METRICS.FRONTEND_BOUND, which are absolute counts.
                        # Averaging those in turned 8.5% into 1.3e10.
                        def _tma_mean(_metric):
                            _name = f"metric_TMA_{_metric}(%)"
                            if _name not in _cols:
                                return None
                            _v = _df[_name].mean()
                            return None if pd.isna(_v) else float(_v)

                        _fe = _tma_mean("Frontend_Bound")
                        _be = _tma_mean("Backend_Bound")
                        _ret = _tma_mean("Retiring")
                        _bs = _tma_mean("Bad_Speculation")
                        # IPC
                        _ipc_cols = [c for c in _cols if "core IPC" in c.lower() or "core_ipc" in c.lower() or c == "metric_core IPC"]
                        _ipc = _df[_ipc_cols].mean().mean() if _ipc_cols else None
                        # CPU utilization
                        _cpu_cols = [c for c in _cols if "CPU utilization %" in c or "cpu_utilization" in c.lower()]
                        _cpu_util = _df[_cpu_cols].mean().mean() if _cpu_cols else None

                        # The metric_TMA_*(%) columns are already percentages —
                        # no 0-1 rescale (that would inflate a real 0.8% to 80%).
                        _tma_rows.append({
                            "Density": _cell["density"],
                            "Frontend Bound %": round(_fe, 1) if _fe is not None else None,
                            "Backend Bound %": round(_be, 1) if _be is not None else None,
                            "Retiring %": round(_ret, 1) if _ret is not None else None,
                            "Bad Speculation %": round(_bs, 1) if _bs is not None else None,
                            "IPC": round(_ipc, 3) if _ipc else None,
                            "CPU Util %": round(_cpu_util, 1) if _cpu_util else None,
                        })
                    except Exception as _e:
                        st.warning(f"Could not parse {_cell['emon_csv'].name}: {_e}")

                if _tma_rows:
                    st.dataframe(_tma_rows, use_container_width=True)

                    # Stacked bar chart of TMA breakdown
                    _tma_densities = [r["Density"] for r in _tma_rows]
                    fig_tma = go.Figure()
                    for _cat, _color in [
                        ("Frontend Bound %", "#ff6b6b"),
                        ("Backend Bound %", "#ffa94d"),
                        ("Retiring %", "#51cf66"),
                        ("Bad Speculation %", "#845ef7"),
                    ]:
                        _vals = [r.get(_cat) or 0 for r in _tma_rows]
                        if any(v > 0 for v in _vals):
                            fig_tma.add_trace(go.Bar(
                                x=[str(d) for d in _tma_densities],
                                y=_vals,
                                name=_cat.replace(" %", ""),
                                marker_color=_color,
                            ))
                    fig_tma.update_layout(
                        barmode="stack",
                        title="TMA Top-Level Breakdown vs Agent Density",
                        xaxis_title="Agent Density",
                        yaxis_title="% of Pipeline Slots",
                        height=400,
                    )
                    st.plotly_chart(fig_tma, use_container_width=True)

                    # IPC degradation chart
                    _ipc_vals = [r.get("IPC") for r in _tma_rows]
                    if any(v is not None for v in _ipc_vals):
                        fig_ipc = go.Figure(go.Scatter(
                            x=[str(d) for d in _tma_densities],
                            y=_ipc_vals,
                            mode="lines+markers",
                            marker=dict(size=10, color="#339af0"),
                            line=dict(width=2),
                        ))
                        fig_ipc.update_layout(
                            title="IPC Degradation with Agent Density",
                            xaxis_title="Agent Density",
                            yaxis_title="Instructions Per Cycle",
                            height=300,
                        )
                        st.plotly_chart(fig_ipc, use_container_width=True)

            # If only raw .dat files exist, show instructions
            _chw_unprocessed = [c for c in _chw_emon_cells if c["emon_dat"] and not c["emon_csv"]]
            if _chw_unprocessed:
                with st.expander(f"{len(_chw_unprocessed)} cells with raw EMON data (not yet post-processed)"):
                    st.markdown(
                        "Run pyEDP to convert `.dat` files to metric CSVs:\n\n"
                        "```bash\n"
                        "for d in /tmp/agentsysperf_sweep/d*_r*/; do\n"
                        "  python -m pyedp.mpp --socket-view \\\n"
                        "    -i \"$d/emon.dat\" -o \"$d/emon.csv\" \\\n"
                        "    -m \"$SEP_DIR\"/config/edp/\"$EMON_DB\".xml \\\n"
                        "    -f \"$SEP_DIR\"/config/edp/chart_format_\"$EMON_DB\".txt\n"
                        "done\n"
                        "```"
                    )
                    st.dataframe(
                        [{"Cell": c["name"], "Density": c["density"],
                          "Size (KB)": c["emon_dat"].stat().st_size // 1024}
                         for c in _chw_unprocessed],
                        use_container_width=True,
                    )

    # ═══ Tab 3: Synth Density Study ════════════════════════════════════════════
    with _scaling_tab_ds:
        st.markdown("""
        Measures how per-agent throughput degrades as **absolute agent count** rises
        on shared CPU resources, swept across NUMA placements and phase mixes (with
        EMON TMA). One shared vLLM server + N concurrent agents, NUMA-aware pinning.
        """)

        # Architecture diagram — detect platform dynamically
        try:
            from src.platform.detect import detect_platform
            _plat = detect_platform()
            _plat_name = f"{_plat.uarch} ({_plat.physical_cores}C, {_plat.numa_nodes} NUMA)"
        except Exception:
            _plat_name = "Detected Platform"

        st.markdown("### Experiment Architecture")
        st.code(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     {_plat_name:<42}│
    ├──────────────────────────────────┬──────────────────────────────────┤
    │   Agent Workers (pinned)         │   Orchestrator + Collection      │
    │                                  │                                  │
    │   ┌──────────┐  ┌──────────┐    │   ┌──────────────────┐           │
    │   │ Agent 1  │  │ Agent 2  │    │   │ Orchestrator     │           │
    │   │ (cpuset) │  │ (cpuset) │    │   │ EMON collector   │           │
    │   │  ...     │  │  ...     │    │   │ pyEDP post-proc  │           │
    │   │ Agent N  │  │          │    │   │ PhaseProfiler    │           │
    │   └──────────┘  └──────────┘    │   └──────────────────┘           │
    └──────────────────────────────────┴──────────────────────────────────┘
        """, language="text")
        st.caption(
            f"Current host: **{_plat_name}**. Each agent is pinned to a cpuset; "
            f"the orchestrator + EMON collection runs on reserved cores. "
            f"Run `python -m experiments.scaling.run_experiment --quick` to populate."
        )

        # Look for scaling results (benchmark-specific directories)
        _SCALING_DATA_DIRS = {
            "Terminal-Bench": [
                # Mixed-workload density sweep (io_heavy+raytrace+interpreter+linalg),
                # d48-d192 with EMON TMA at d80/d192. Preferred: the older _scaling_tb
                # symlinks point at a deleted run_13min_v5 target.
                Path(f"{_TMP}/agentsysperf_scaling_mixed"),
                Path(f"{_TMP}/agentsysperf_scaling_tb"),
                Path(f"{_TMP}/agentsysperf_scaling"),
                Path(f"{_TMP}/agentsysperf_e2e_test_v3/scaling_run"),
                Path(f"{_TMP}/agentsysperf_e2e_test_v2/scaling_run"),
            ],
            "Tau-Bench": [
                Path(f"{_TMP}/agentsysperf_scaling_tau"),
            ],
            "SWE-Bench": [
                Path(f"{_TMP}/agentsysperf_swe_bench"),
                Path(f"{_TMP}/agentsysperf_scaling_swe"),
            ],
        }

        _scaling_data = None
        _scaling_plots_dir = None
        for _sdir in _SCALING_DATA_DIRS.get(_nav_section, []):
            _all_results_file = _sdir / "all_results.json"
            if _all_results_file.exists():
                try:
                    _scaling_data = json.loads(_all_results_file.read_text())
                    _scaling_plots_dir = _sdir / "plots"
                    break
                except Exception:
                    pass
            # Also check per-config results.json
            if _sdir.exists() and not _scaling_data:
                _per_config = sorted(_sdir.glob("*/results.json"))
                if _per_config:
                    _scaling_data = []
                    for _rf in _per_config:
                        try:
                            _scaling_data.append(json.loads(_rf.read_text()))
                        except Exception:
                            pass
                    if _scaling_data:
                        _scaling_plots_dir = _sdir / "plots"
                        break
                    else:
                        _scaling_data = None

        if not _scaling_data:
            st.warning(
                "No scaling experiment data. Run:\n\n"
                "```\npython -m experiments.scaling.run_experiment --quick\n```\n\n"
                "For a fuller study: `--medium` (8 densities) or `--full` (all placements + mixes)."
            )

            # Show configuration
            st.markdown("---")
            st.markdown("### vLLM CPU Configuration (reference)")
            _cfg_data = [
                {"Setting": "VLLM_CPU_SGL_KERNEL", "Value": "1", "Purpose": "x86 small-batch optimized kernel"},
                {"Setting": "--dtype", "Value": "bfloat16", "Purpose": "Stable on CPU (float16 is not)"},
                {"Setting": "--tensor-parallel-size", "Value": "2", "Purpose": "One rank per NUMA node (shared mode)"},
                {"Setting": "VLLM_CPU_KVCACHE_SPACE", "Value": "40 GiB", "Purpose": "Sized for 100+ step agentic conversations"},
                {"Setting": "VLLM_CPU_OMP_THREADS_BIND", "Value": "0-31|32-63", "Purpose": "Pin TP ranks to NUMA nodes"},
                {"Setting": "--max-num-seqs", "Value": "384", "Purpose": "Support up to 32 concurrent agents"},
                {"Setting": "--block-size", "Value": "128", "Purpose": "Optimal for CPU (multiples of 32)"},
            ]
            st.dataframe(_cfg_data, use_container_width=True)

            st.markdown("### Density Levels")
            st.markdown("Sweep: **1, 2, 4, 8, 12, 16, 24, 32** concurrent agents")
            st.markdown("Each agent gets dedicated CPU cores (4 per agent by default) on NUMA node 2.")
        else:
            import pandas as pd

            _all_results = _scaling_data

            if _all_results:
                st.success(f"Loaded {len(_all_results)} experiment configurations")

                # Extract density vs throughput. None, not 0, for an absent
                # field: plotly skips a None point, whereas a 0 pins the line to
                # the axis and reads as a measured collapse in throughput.
                _densities = [r["config"]["density"] for r in _all_results]
                _throughputs = [_num(r, "aggregate_throughput_turns_per_s")
                                for r in _all_results]
                _mean_turns = [_num(r, "mean_throughput_turns_per_s")
                               for r in _all_results]
                if not any(v is not None for v in _throughputs):
                    st.warning(
                        "None of these results carry "
                        "`aggregate_throughput_turns_per_s`, so the throughput "
                        "chart below is empty rather than zero. Check the runner "
                        "that produced this file."
                    )

                col1, col2 = st.columns(2)

                with col1:
                    fig_tp = go.Figure()
                    fig_tp.add_trace(go.Scatter(
                        x=_densities, y=_throughputs,
                        mode="lines+markers",
                        name="Aggregate",
                        marker=dict(size=10),
                    ))
                    fig_tp.add_trace(go.Scatter(
                        x=_densities, y=_mean_turns,
                        mode="lines+markers",
                        name="Per-Agent Mean",
                        marker=dict(size=8),
                        line=dict(dash="dash"),
                    ))
                    fig_tp.update_layout(
                        title="Throughput vs Agent Density",
                        xaxis_title="Concurrent Agents",
                        yaxis_title="Throughput (turns/s)",
                        height=350,
                    )
                    st.plotly_chart(fig_tp, use_container_width=True)

                with col2:
                    # Per-agent degradation
                    if _mean_turns and _mean_turns[0] > 0:
                        _baseline = _mean_turns[0]
                        _degradation = [t / _baseline * 100 for t in _mean_turns]
                        fig_deg = go.Figure(go.Bar(
                            x=_densities, y=_degradation,
                            marker_color=["#4caf50" if d > 80 else "#ff9800" if d > 50 else "#d32f2f" for d in _degradation],
                            text=[f"{d:.0f}%" for d in _degradation],
                            textposition="outside",
                        ))
                        fig_deg.update_layout(
                            title="Per-Agent Performance Retention (%)",
                            xaxis_title="Concurrent Agents",
                            yaxis_title="% of Baseline",
                            height=350,
                        )
                        fig_deg.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
                        st.plotly_chart(fig_deg, use_container_width=True)

                # EMON metrics per density (if available)
                _has_emon = any((_num(r, "emon_metrics_count") or 0) > 0
                                for r in _all_results)
                if _has_emon:
                    st.markdown("---")
                    st.markdown("### EMON Metrics per Density Level")
                    _emon_summary = []
                    for _r in _all_results:
                        _emon_summary.append({
                            "Density": _r["config"]["density"],
                            "EMON Metrics": _fmt(_r, "emon_metrics_count", fmt="{:.0f}"),
                            "Duration (s)": _fmt(_r, "total_duration_s"),
                            "Placement": _r["config"]["placement"],
                        })
                    st.dataframe(_emon_summary, use_container_width=True)

                # Results table
                st.markdown("---")
                st.markdown("### Run Summary")
                _summary = []
                for _r in _all_results:
                    _agents = _r.get("agents", [])
                    _ok = sum(1 for a in _agents if not a.get("error"))
                    _summary.append({
                        "Density": _r["config"]["density"],
                        "Placement": _r["config"]["placement"],
                        "Agents OK": f"{_ok}/{_r['config']['density']}",
                        "Aggregate (turns/s)": _fmt(_r, "aggregate_throughput_turns_per_s",
                                                    fmt="{:.2f}"),
                        "Per-Agent (turns/s)": _fmt(_r, "mean_throughput_turns_per_s",
                                                    fmt="{:.2f}"),
                        "Duration (s)": _fmt(_r, "total_duration_s"),
                    })
                st.dataframe(_summary, use_container_width=True)

            # Display generated plots if available
            if _scaling_plots_dir and _scaling_plots_dir.exists():
                st.markdown("---")
                st.markdown("### Generated Charts")
                _plot_files = sorted(_scaling_plots_dir.glob("*.png"))
                if _plot_files:
                    for _i in range(0, len(_plot_files), 2):
                        _cols = st.columns(2)
                        for _j, _col in enumerate(_cols):
                            _idx = _i + _j
                            if _idx < len(_plot_files):
                                with _col:
                                    st.image(str(_plot_files[_idx]),
                                             caption=_plot_files[_idx].stem.replace("_", " ").title())

            # Show analysis report findings if available
            _report_file = (_scaling_plots_dir.parent / "analysis_report.json") if _scaling_plots_dir else None
            if _report_file and _report_file.exists():
                _report = json.loads(_report_file.read_text())
                _findings = _report.get("findings", [])
                if _findings:
                    st.markdown("---")
                    st.markdown("### Key Findings")
                    for _f in _findings:
                        st.markdown(f"- {_f}")

                _inflections = _report.get("inflection_points", [])
                if _inflections:
                    st.markdown("### Contention Inflection Points")
                    for _inf in _inflections:
                        st.warning(
                            f"**Density {_inf['density']}** ({_inf['curve']}): "
                            + "; ".join(_inf.get("reasons", []))
                        )

    # ═══ Tab 3: HW Resources ═════════════════════════════════════════════
    with _scaling_tab_hw:
        st.markdown("""
        Answers: **"At N concurrent agents, what hardware resource is the bottleneck?"**

        Unlike the agent-density sweep (Concurrency Sweep / Synth Density Study) which measures
        *throughput degradation*, this tab shows the **CPU microarchitecture root cause**
        at each density point — LLC capacity, memory bandwidth, cross-NUMA coherency,
        or core contention — using real EMON PMU counters collected per density level.
        """)

        st.markdown("### Experiment Framework: `experiments/scaling/`")
        st.code("""
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  experiments/scaling/ — HW Resource Scaling Framework                     │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  config.py        Topology (SNC3 96C), density levels, NUMA placements   │
    │  pinning.py       CPU affinity assignment (cores/agent, node isolation)   │
    │  agent_worker.py  Per-agent workload: phase mix (reason/act/balanced)     │
    │  orchestrator.py  Spawn N agents + EMON collection per density point     │
    │  analysis.py      Scaling curves, inflection detection, NUMA comparison   │
    │  plotting.py      Charts: throughput, IPC, LLC MPKI, TMA heatmap         │
    │                                                                          │
    │  run_experiment.py     Full matrix runner (--quick / --medium / --full)   │
    │  launch_vllm_taubench.py   Option A: shared vLLM + N tau-bench agents    │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

    Density sweep:  1 → 2 → 4 → 8 → 12 → 16 → 24 → 32 agents
    Placements:     intra_node | cross_node | spread
    Phase mixes:    compute_heavy | io_heavy | balanced | mixed
    Per density:    EMON EDP collection → pyEDP → TMA + LLC MPKI + IPC
        """, language="text")

        # Look for scaling experiment results
        _HW_SCALING_DIRS = {
            "Terminal-Bench": [
                Path(f"{_TMP}/agentsysperf_scaling_tb"),
                Path(f"{_TMP}/agentsysperf_scaling"),
            ],
            "Tau-Bench": [
                Path(f"{_TMP}/agentsysperf_scaling_tau"),
            ],
            "SWE-Bench": [
                Path(f"{_TMP}/agentsysperf_swe_bench"),
                Path(f"{_TMP}/agentsysperf_scaling_swe"),
            ],
        }

        _hw_report = None
        _hw_results_dir = None
        for _cand in _HW_SCALING_DIRS.get(_nav_section, []):
            # Check for analysis_report.json (produced by analysis.py)
            _report_file = _cand / "analysis_report.json"
            if _report_file.exists():
                try:
                    _hw_report = json.loads(_report_file.read_text())
                    _hw_results_dir = _cand
                    break
                except Exception:
                    pass
            # Check subdirectories (timestamped runs)
            if _cand.exists():
                for _subdir in sorted(_cand.iterdir(), reverse=True):
                    if _subdir.is_dir() and (_subdir / "analysis_report.json").exists():
                        try:
                            _hw_report = json.loads((_subdir / "analysis_report.json").read_text())
                            _hw_results_dir = _subdir
                            break
                        except Exception:
                            pass
                    elif _subdir.is_dir() and (_subdir / "all_results.json").exists():
                        try:
                            sys.path.insert(0, str(Path(__file__).resolve().parent))
                            from experiments.scaling.analysis import analyze, generate_report
                            _analysis = analyze(_subdir)
                            _hw_report = generate_report(_analysis)
                            _hw_results_dir = _subdir
                            break
                        except Exception:
                            pass
                if _hw_report:
                    break

        if _hw_report is None:
            st.warning(
                f"No HW scaling data for **{_nav_section}**. Run:\n\n"
                "```bash\n"
                "# Quick test (2 densities, ~5 min):\n"
                "python -m experiments.scaling.run_experiment --quick\n\n"
                "# With real vLLM + tau-bench:\n"
                "python -m experiments.scaling.launch_vllm_taubench --sweep 1,2,4,8,16\n\n"
                "# Full matrix (all densities x placements x mixes):\n"
                "python -m experiments.scaling.run_experiment --full\n"
                "```"
            )

            st.markdown("---")
            st.markdown("### What This Measures (per density point)")
            _hw_metrics_table = [
                {"Metric": "IPC (Instructions/Cycle)", "Source": "perf_events", "Indicates": "Core efficiency — drops signal contention"},
                {"Metric": "LLC MPKI (L3 misses/1K insn)", "Source": "perf_events", "Indicates": "Cache capacity pressure — jumps at working-set overflow"},
                {"Metric": "Cache Miss %", "Source": "perf_events", "Indicates": "L1+LLC combined miss rate"},
                {"Metric": "Memory BW (GB/s)", "Source": "EMON TMA", "Indicates": "DRAM bandwidth saturation"},
                {"Metric": "NUMA Penalty %", "Source": "cross_node vs intra_node", "Indicates": "Cross-socket coherency cost"},
                {"Metric": "TMA Frontend/Backend/Retiring", "Source": "EMON pyEDP CSV", "Indicates": "Pipeline slot utilization breakdown"},
                {"Metric": "Context Switches", "Source": "perf_events", "Indicates": "Scheduler contention at high density"},
            ]
            st.dataframe(_hw_metrics_table, use_container_width=True, hide_index=True)

            st.markdown("### Inflection Detection Logic")
            st.code("""
    Contention inflection = FIRST density where ANY of:
      • LLC MPKI > 2× baseline (density=1)    → cache capacity wall
      • Normalized throughput < 70% baseline   → throughput cliff
      • IPC < 60% of baseline                  → core/memory stall

    Root-cause attribution:
      LLC MPKI spike + IPC drop   → L3 capacity overflow (working set > LLC/N)
      IPC drop + NUMA penalty     → cross-node coherency (snoop filter thrash)
      High context switches       → scheduler oversubscription
      TMA Backend_Bound rising    → memory bandwidth wall
            """, language="text")
        else:
            import pandas as pd

            st.success(f"Loaded analysis from `{_hw_results_dir}`")

            # Summary metrics
            _hw_summary = _hw_report.get("summary", {})
            _s1, _s2, _s3 = st.columns(3)
            _s1.metric("Configs Analyzed", _hw_summary.get("total_configs", 0))
            _s2.metric("Inflection Points", _hw_summary.get("inflection_points_found", 0))
            _s3.metric("TMA Data", "Yes" if _hw_summary.get("tma_data_available") else "No")

            # Key findings
            _hw_findings = _hw_report.get("findings", [])
            if _hw_findings:
                st.markdown("### Key Findings")
                for _f in _hw_findings:
                    st.markdown(f"- {_f}")

            # ─── Inflection Points ────────────────────────────────────────
            _hw_inflections = _hw_report.get("inflection_points", [])
            if _hw_inflections:
                st.markdown("---")
                st.markdown("### Contention Inflection Points")
                st.markdown("First density where hardware resource saturation is detected:")
                _inf_rows = []
                for _inf in _hw_inflections:
                    _inf_rows.append({
                        "Configuration": _inf["curve"],
                        "Inflection Density": _inf["density"],
                        "Throughput (vs baseline)": f"{_inf['normalized_throughput']:.0%}",
                        "LLC MPKI": f"{_inf['llc_mpki']:.1f}",
                        "IPC": f"{_inf['ipc']:.3f}",
                        "Root Cause": "; ".join(_inf["reasons"]),
                    })
                st.dataframe(_inf_rows, use_container_width=True, hide_index=True)

            # ─── Scaling Curves (Plotly) ──────────────────────────────────
            _hw_curves = _hw_report.get("scaling_curves", {})
            if _hw_curves:
                st.markdown("---")
                st.markdown("### Scaling Curves")

                _tab_tp, _tab_ipc, _tab_llc, _tab_lat = st.tabs([
                    "Throughput", "IPC", "LLC MPKI", "Latency"
                ])

                _placement_colors = {
                    "intra_node": "#d32f2f", "cross_node": "#1976d2", "spread": "#388e3c",
                }
                _mix_dashes = {
                    "compute_heavy": "solid", "io_heavy": "dash",
                    "balanced": "dot", "mixed": "dashdot",
                }

                def _curve_style(key):
                    color = "#607d8b"
                    dash = "solid"
                    for pk, pc in _placement_colors.items():
                        if pk in key:
                            color = pc
                            break
                    for mk, md in _mix_dashes.items():
                        if mk in key:
                            dash = md
                            break
                    return color, dash

                with _tab_tp:
                    _fig_tp = go.Figure()
                    for _ck, _pts in sorted(_hw_curves.items()):
                        _color, _dash = _curve_style(_ck)
                        _fig_tp.add_trace(go.Scatter(
                            x=[p["density"] for p in _pts],
                            y=[p["normalized_throughput"] for p in _pts],
                            mode="lines+markers", name=_ck,
                            line=dict(color=_color, dash=_dash),
                        ))
                    _fig_tp.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)
                    _fig_tp.add_hline(y=0.7, line_dash="dot", line_color="red", opacity=0.3,
                                      annotation_text="70% threshold")
                    _fig_tp.update_layout(
                        title="Per-Agent Throughput Degradation vs Density",
                        xaxis_title="Agent Density (N concurrent agents)",
                        yaxis_title="Normalized Per-Agent Throughput",
                        yaxis_range=[0, 1.15], height=450,
                    )
                    st.plotly_chart(_fig_tp, use_container_width=True)

                with _tab_ipc:
                    _fig_ipc = go.Figure()
                    for _ck, _pts in sorted(_hw_curves.items()):
                        _color, _dash = _curve_style(_ck)
                        _fig_ipc.add_trace(go.Scatter(
                            x=[p["density"] for p in _pts],
                            y=[p["ipc"] for p in _pts],
                            mode="lines+markers", name=_ck,
                            line=dict(color=_color, dash=_dash),
                        ))
                    _fig_ipc.update_layout(
                        title="IPC Degradation Under Contention",
                        xaxis_title="Agent Density (N)",
                        yaxis_title="Instructions Per Cycle (IPC)",
                        height=400,
                    )
                    st.plotly_chart(_fig_ipc, use_container_width=True)

                with _tab_llc:
                    _fig_llc = go.Figure()
                    for _ck, _pts in sorted(_hw_curves.items()):
                        _color, _dash = _curve_style(_ck)
                        _mpki_vals = [p["llc_mpki"] for p in _pts]
                        if all(m == 0 for m in _mpki_vals):
                            continue
                        _fig_llc.add_trace(go.Scatter(
                            x=[p["density"] for p in _pts],
                            y=_mpki_vals,
                            mode="lines+markers", name=_ck,
                            line=dict(color=_color, dash=_dash),
                        ))
                    _fig_llc.update_layout(
                        title="L3 Cache Pressure (LLC Misses Per Kilo Instructions)",
                        xaxis_title="Agent Density (N)",
                        yaxis_title="LLC MPKI",
                        yaxis_type="log" if any(
                            p["llc_mpki"] > 10 for _pts in _hw_curves.values() for p in _pts
                        ) else "linear",
                        height=400,
                    )
                    st.plotly_chart(_fig_llc, use_container_width=True)

                with _tab_lat:
                    _fig_lat = go.Figure()
                    for _ck, _pts in sorted(_hw_curves.items()):
                        _color, _dash = _curve_style(_ck)
                        _fig_lat.add_trace(go.Scatter(
                            x=[p["density"] for p in _pts],
                            y=[p["p50_ms"] for p in _pts],
                            mode="lines+markers", name=f"{_ck} (p50)",
                            line=dict(color=_color, dash=_dash),
                        ))
                        _fig_lat.add_trace(go.Scatter(
                            x=[p["density"] for p in _pts],
                            y=[p["p95_ms"] for p in _pts],
                            mode="lines+markers", name=f"{_ck} (p95)",
                            line=dict(color=_color, dash="dot"),
                            opacity=0.6,
                        ))
                    _fig_lat.update_layout(
                        title="Turn Completion Latency (p50 + p95)",
                        xaxis_title="Agent Density (N)",
                        yaxis_title="Latency (ms)",
                        height=400,
                    )
                    st.plotly_chart(_fig_lat, use_container_width=True)

            # ─── NUMA Comparison ──────────────────────────────────────────
            _hw_numa = _hw_report.get("numa_comparison", [])
            if _hw_numa:
                st.markdown("---")
                st.markdown("### NUMA Placement Comparison")
                st.markdown(
                    "Shows throughput impact of agent placement strategy at each density. "
                    "NUMA penalty = how much throughput drops from cross-node coherency traffic."
                )

                _numa_densities = sorted(set(d["density"] for d in _hw_numa))
                _placements_present = ["intra_node", "cross_node", "spread"]

                _fig_numa = go.Figure()
                for _pl in _placements_present:
                    _tp_vals = []
                    for _d in _numa_densities:
                        entry = next((e for e in _hw_numa if e["density"] == _d), None)
                        if entry and _pl in entry:
                            _tp_vals.append(entry[_pl]["throughput"])
                        else:
                            _tp_vals.append(0)
                    if any(v > 0 for v in _tp_vals):
                        _fig_numa.add_trace(go.Bar(
                            x=[str(d) for d in _numa_densities],
                            y=_tp_vals,
                            name=_pl,
                            marker_color=_placement_colors.get(_pl, "#607d8b"),
                        ))
                _fig_numa.update_layout(
                    title="NUMA Placement Effect on Per-Agent Throughput",
                    xaxis_title="Agent Density",
                    yaxis_title="Per-Agent Throughput (turns/s)",
                    barmode="group", height=400,
                )
                st.plotly_chart(_fig_numa, use_container_width=True)

                # NUMA penalty table
                _numa_penalty_rows = [d for d in _hw_numa if "numa_penalty_pct" in d]
                if _numa_penalty_rows:
                    st.markdown("**NUMA Penalty (cross_node vs intra_node):**")
                    _np_table = [{"Density": d["density"], "Phase Mix": d["phase_mix"],
                                  "NUMA Penalty": f"{d['numa_penalty_pct']:.1f}%"}
                                 for d in _numa_penalty_rows]
                    st.dataframe(_np_table, use_container_width=True, hide_index=True)

            # ─── Phase Breakdown ──────────────────────────────────────────
            _hw_phases = _hw_report.get("phase_breakdown", [])
            if _hw_phases:
                st.markdown("---")
                st.markdown("### Phase Time Breakdown vs Density")
                st.markdown(
                    "How wall-clock time splits between Reason (LLM inference), "
                    "Act (tool/code execution), and Overhead (scheduling/orchestration) "
                    "as agent count increases."
                )

                # Group by placement_mix
                from collections import defaultdict
                _phase_groups = defaultdict(list)
                for _p in _hw_phases:
                    _phase_groups[f"{_p['placement']}_{_p['phase_mix']}"].append(_p)

                _selected_group = st.selectbox(
                    "Configuration", sorted(_phase_groups.keys()),
                    key="hw_phase_group",
                )

                _group_pts = sorted(_phase_groups[_selected_group], key=lambda p: p["density"])
                _fig_phase = go.Figure()
                _fig_phase.add_trace(go.Bar(
                    x=[str(p["density"]) for p in _group_pts],
                    y=[p["reason_pct"] for p in _group_pts],
                    name="Reason (LLM)", marker_color="#f44336",
                ))
                _fig_phase.add_trace(go.Bar(
                    x=[str(p["density"]) for p in _group_pts],
                    y=[p["act_pct"] for p in _group_pts],
                    name="Act (Tool)", marker_color="#4caf50",
                ))
                _fig_phase.add_trace(go.Bar(
                    x=[str(p["density"]) for p in _group_pts],
                    y=[p["overhead_pct"] for p in _group_pts],
                    name="Overhead", marker_color="#9e9e9e",
                ))
                _fig_phase.update_layout(
                    barmode="stack",
                    title=f"Phase Breakdown: {_selected_group}",
                    xaxis_title="Agent Density",
                    yaxis_title="% Wall-Clock Time",
                    yaxis_range=[0, 100],
                    height=400,
                )
                st.plotly_chart(_fig_phase, use_container_width=True)

            # ─── TMA Heatmap ─────────────────────────────────────────────
            _hw_tma = _hw_report.get("tma_heatmap")
            if _hw_tma:
                st.markdown("---")
                st.markdown("### TMA Microarchitecture Heatmap vs Density")
                st.markdown(
                    "Shows how pipeline slot utilization shifts as agents are added. "
                    "Rising Backend_Bound/Memory_Bound → bandwidth wall. "
                    "Rising Frontend_Bound → instruction fetch/decode pressure."
                )

                _tma_keys = ["Frontend_Bound", "Backend_Bound", "Bad_Speculation",
                             "Retiring", "Memory_Bound", "Core_Bound"]
                _available_keys = [k for k in _tma_keys if any(k in d for d in _hw_tma)]
                _tma_densities = sorted(set(d["density"] for d in _hw_tma))

                if _available_keys and _tma_densities:
                    _z_matrix = []
                    for _d in _tma_densities:
                        entries = [e for e in _hw_tma if e["density"] == _d]
                        row = [entries[0].get(k, 0) if entries else 0 for k in _available_keys]
                        _z_matrix.append(row)

                    _fig_tma = go.Figure(go.Heatmap(
                        z=_z_matrix,
                        x=[k.replace("_", " ") for k in _available_keys],
                        y=[str(d) for d in _tma_densities],
                        colorscale="RdYlGn_r",
                        zmin=0, zmax=100,
                        text=[[f"{v:.0f}" for v in row] for row in _z_matrix],
                        texttemplate="%{text}%",
                        colorbar_title="% Slots",
                    ))
                    _fig_tma.update_layout(
                        title="TMA Pipeline Slot Breakdown (% of total slots)",
                        xaxis_title="TMA Category",
                        yaxis_title="Agent Density",
                        height=max(300, len(_tma_densities) * 50),
                    )
                    st.plotly_chart(_fig_tma, use_container_width=True)

            # ─── Root Cause Attribution ───────────────────────────────────
            if _hw_inflections:
                st.markdown("---")
                st.markdown("### Root Cause Attribution")
                st.markdown(
                    "At each inflection point, what hardware resource saturated first:"
                )

                for _inf in _hw_inflections:
                    _reasons = _inf["reasons"]
                    _cause = "Unknown"
                    _explanation = ""

                    if any("LLC_MPKI" in r for r in _reasons):
                        if any("IPC" in r for r in _reasons):
                            _cause = "LLC Capacity Overflow"
                            _explanation = (
                                f"Working set at density={_inf['density']} exceeds LLC/{_inf['density']} per agent. "
                                f"LLC MPKI={_inf['llc_mpki']:.1f} causes memory stalls → IPC={_inf['ipc']:.3f}."
                            )
                        else:
                            _cause = "Cache Pressure (early)"
                            _explanation = (
                                f"LLC MPKI rising ({_inf['llc_mpki']:.1f}) but IPC not yet collapsed — "
                                f"prefetchers still hiding latency. Next density level will cliff."
                            )
                    elif any("IPC" in r for r in _reasons):
                        _cause = "Core/Memory Stall"
                        _explanation = (
                            f"IPC dropped to {_inf['ipc']:.3f} without proportional LLC MPKI rise — "
                            f"likely memory bandwidth saturation or serialization (lock contention)."
                        )
                    elif any("throughput" in r for r in _reasons):
                        _cause = "Throughput Cliff"
                        _explanation = (
                            f"Per-agent throughput at {_inf['normalized_throughput']:.0%} of baseline. "
                            f"Combined scheduler oversubscription + memory contention."
                        )

                    with st.expander(f"**{_inf['curve']}** — density {_inf['density']}: {_cause}", expanded=True):
                        st.markdown(_explanation)
                        st.markdown(f"**Evidence:** {'; '.join(_reasons)}")

            # Baselines
            _hw_baselines = _hw_report.get("baselines", {})
            if _hw_baselines:
                st.markdown("---")
                st.markdown("### Baseline (density=1) Reference")
                _bl_rows = []
                for _bk, _bv in sorted(_hw_baselines.items()):
                    _bl_rows.append({
                        "Config": _bk,
                        "Throughput (turns/s)": f"{_bv['throughput']:.3f}",
                        "IPC": f"{_bv['ipc']:.3f}",
                        "LLC MPKI": f"{_bv['llc_mpki']:.2f}",
                        "p50 (ms)": f"{_bv['p50_ms']:.1f}",
                        "Reason %": f"{_bv['reason_pct']:.1f}",
                    })
                st.dataframe(_bl_rows, use_container_width=True, hide_index=True)

            st.caption(f"Data source: `{_hw_results_dir}`")


elif demo_page == "Observability::Langfuse Traces":
    st.title("Langfuse LLM Trace Analytics")
    st.markdown("""
    Langfuse captures per-step LLM analytics (tokens, cost, prompt/output, latency)
    via LiteLLM's built-in callback. Traces correlate with AgentSysPerf hardware
    measurements through a shared `run_id`.
    """)

    # Architecture diagram
    st.markdown("### Integration Architecture")
    st.code("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    AgentSysPerf Agent Loop                              │
    │                                                                     │
    │   litellm.completion(model, messages, metadata={run_id, task_id})   │
    │           │                                    │                     │
    │           ▼                                    ▼                     │
    │   ┌───────────────┐                   ┌───────────────────┐         │
    │   │ LLM Response  │                   │ Langfuse Callback │         │
    │   │ (agent loop)  │                   │ (auto-captures):  │         │
    │   └───────────────┘                   │  • model          │         │
    │                                       │  • prompt/output  │         │
    │                                       │  • tokens in/out  │         │
    │                                       │  • latency        │         │
    │                                       │  • cost (auto)    │         │
    │                                       │  • run_id/task_id │         │
    │                                       └─────────┬─────────┘         │
    └─────────────────────────────────────────────────┼───────────────────┘
                                                      │
                                                      ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     Langfuse (Self-hosted)                           │
    │                                                                     │
    │   Sessions ──► Traces ──► Generations (per LLM call)                │
    │                              │                                       │
    │   Dashboards: tokens/cost per task, latency percentiles,            │
    │               model comparison, per-step waterfall                   │
    │                                                                     │
    │   Join key: run_id ←──── correlates with ────► Grafana (hardware)   │
    └─────────────────────────────────────────────────────────────────────┘
    """, language="text")

    # Connection status
    st.markdown("---")
    st.markdown("### Connection Status")

    import os as _os
    _lf_host = _os.environ.get("LANGFUSE_HOST", "")
    _lf_key = _os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    _lf_secret = _os.environ.get("LANGFUSE_SECRET_KEY", "")
    _lf_enabled = _os.environ.get("AGENTSYSPERF_LANGFUSE", "").strip() in ("1", "true", "yes")

    _col1, _col2, _col3, _col4 = st.columns(4)
    with _col1:
        if _lf_enabled:
            st.metric("AGENTSYSPERF_LANGFUSE", "Enabled")
        else:
            st.metric("AGENTSYSPERF_LANGFUSE", "Disabled")
    with _col2:
        st.metric("LANGFUSE_HOST", _lf_host or "(not set)")
    with _col3:
        st.metric("Public Key", _lf_key[:8] + "..." if len(_lf_key) > 8 else _lf_key or "(not set)")
    with _col4:
        st.metric("Secret Key", "***" if _lf_secret else "(not set)")

    if _lf_enabled and _lf_host:
        st.success(f"Langfuse tracing is **active**. Traces flow to `{_lf_host}`.")
        st.markdown(f"### [Open Langfuse Dashboard]({_lf_host})")
    elif _lf_enabled and not _lf_host:
        st.error("AGENTSYSPERF_LANGFUSE=1 but LANGFUSE_HOST is not set. Traces will fail.")
    else:
        st.warning(
            "Langfuse tracing is **disabled**. To enable:\n\n"
            "```bash\n"
            "export AGENTSYSPERF_LANGFUSE=1\n"
            f"export LANGFUSE_HOST={_langfuse_host()}\n"
            "export LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
            "export LANGFUSE_SECRET_KEY=sk-lf-...\n"
            "```"
        )

    # Try to fetch recent traces from Langfuse API
    st.markdown("---")
    st.markdown("### Recent Traces")

    if _lf_host and _lf_key and _lf_secret:
        try:
            import requests
            _resp = requests.get(
                f"{_lf_host.rstrip('/')}/api/public/traces",
                params={"limit": 10, "orderBy": "timestamp.DESC"},
                auth=(_lf_key, _lf_secret),
                timeout=5,
            )
            if _resp.status_code == 200:
                _traces = _resp.json().get("data", [])
                if _traces:
                    _trace_table = []
                    for _tr in _traces:
                        _trace_table.append({
                            "Trace ID": _tr.get("id", "")[:12] + "...",
                            "Name": _tr.get("name", "—"),
                            "Session": _tr.get("sessionId", "—"),
                            "Timestamp": _tr.get("timestamp", "")[:19],
                            "Tokens (in)": _tr.get("usage", {}).get("promptTokens", 0) if _tr.get("usage") else 0,
                            "Tokens (out)": _tr.get("usage", {}).get("completionTokens", 0) if _tr.get("usage") else 0,
                            "Cost ($)": round(_tr.get("calculatedTotalCost", 0) or 0, 4),
                            "Latency (s)": round((_tr.get("latency", 0) or 0) / 1000, 2),
                        })
                    st.dataframe(_trace_table, use_container_width=True)

                    # Summary metrics
                    _total_cost = sum(t.get("Cost ($)", 0) for t in _trace_table)
                    _total_tokens = sum(t.get("Tokens (in)", 0) + t.get("Tokens (out)", 0) for t in _trace_table)
                    _avg_latency = sum(t.get("Latency (s)", 0) for t in _trace_table) / len(_trace_table)

                    _sc1, _sc2, _sc3 = st.columns(3)
                    with _sc1:
                        st.metric("Total Cost (last 10)", f"${_total_cost:.4f}")
                    with _sc2:
                        st.metric("Total Tokens (last 10)", f"{_total_tokens:,}")
                    with _sc3:
                        st.metric("Avg Latency", f"{_avg_latency:.2f}s")
                else:
                    st.info("Connected to Langfuse but no traces found yet. Run a benchmark to generate traces.")
            elif _resp.status_code == 401:
                st.error("Authentication failed. Check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.")
            else:
                st.warning(f"Langfuse API returned {_resp.status_code}: {_resp.text[:200]}")
        except ImportError:
            st.info("Install `requests` to fetch live traces from Langfuse API.")
        except Exception as _e:
            st.warning(f"Could not connect to Langfuse at `{_lf_host}`: {_e}")
    else:
        st.info("Set Langfuse credentials to see live trace data here.")

    # What gets captured
    st.markdown("---")
    st.markdown("### What Langfuse Captures")

    _capture_data = [
        {"Field": "model", "Source": "litellm.completion()", "Example": "Qwen/Qwen3-Coder-30B-A3B-Instruct"},
        {"Field": "prompt (input)", "Source": "messages array", "Example": "System + conversation history"},
        {"Field": "completion (output)", "Source": "LLM response", "Example": "Agent reasoning + tool calls"},
        {"Field": "promptTokens", "Source": "usage.prompt_tokens", "Example": "1,245"},
        {"Field": "completionTokens", "Source": "usage.completion_tokens", "Example": "387"},
        {"Field": "latency", "Source": "wall-clock generation time", "Example": "2.3s"},
        {"Field": "cost", "Source": "auto-computed from model pricing", "Example": "$0.0023"},
        {"Field": "trace_name", "Source": "metadata.task_id", "Example": "retail_task_042"},
        {"Field": "session_id", "Source": "metadata.task_id", "Example": "retail_task_042"},
        {"Field": "trace_user_id", "Source": "metadata.run_id", "Example": "run_2026-06-09_14:30"},
        {"Field": "tags", "Source": "metadata.tags", "Example": '["agentsysperf", "run:..."]'},
    ]
    st.dataframe(_capture_data, use_container_width=True)

    # Correlation story
    st.markdown("---")
    st.markdown("### Two-Pane Correlation")
    st.code("""
    ┌─────────────────────────────┐     shared run_id     ┌─────────────────────────────┐
    │        LANGFUSE             │◄───────────────────────►│        GRAFANA              │
    │                             │                         │                             │
    │  • Per-step token counts    │                         │  • IPC during that step     │
    │  • Per-step cost ($)        │                         │  • Cache miss rate           │
    │  • Prompt / completion text │                         │  • TMA breakdown            │
    │  • Latency waterfall        │                         │  • Memory bandwidth         │
    │  • Task success/fail score  │                         │  • NUMA cross-traffic       │
    │  • Model comparison         │                         │  • Phase time distribution  │
    └─────────────────────────────┘                         └─────────────────────────────┘

    Question: "Why is task_042 slow?"
    → Langfuse: turn 7 generated 2,400 tokens (10x normal) — context explosion
    → Grafana:  during turn 7, IPC dropped to 0.3, LLC MPKI spiked to 900 — KV cache thrashing
    → Action:   reduce max_model_len or enable chunked prefill
    """, language="text")

    st.info(
        "**Setup**: Self-host Langfuse via Docker Compose (MIT license). "
        "Components: langfuse-web + langfuse-worker + Postgres + ClickHouse + Redis + S3/minio."
    )


elif demo_page == "Observability::Grafana Dashboard":
    st.title("Live Grafana Dashboard")
    st.markdown("Every benchmark run auto-exports to Prometheus. Grafana provides real-time visualization.")

    grafana_url = f"http://{_HOST}:3000/d/agentsysperf-main/agentsysperf-benchmark-dashboard"

    st.markdown(f"### [Click here to open Grafana Dashboard]({grafana_url})")
    st.caption("No login required: the shipped Grafana config enables anonymous access (GF_AUTH_DISABLE_LOGIN_FORM=true).")

    st.markdown("---")
    st.markdown("### Architecture")
    st.code("""
AgentSysPerf Benchmark Run
        │
        ▼
┌──────────────────────┐
│  Metrics Server      │  ← localhost:9101/metrics
│  (Prometheus format) │
└──────────┬───────────┘
           │ scrape every 5s
           ▼
┌──────────────────────┐
│  Prometheus          │  ← localhost:9090
│  (Time-series DB)    │
└──────────┬───────────┘
           │ query
           ▼
┌──────────────────────┐
│  Grafana             │  ← localhost:3000
│  (Auto-provisioned   │
│   dashboard)         │
└──────────────────────┘
    """, language="text")

    st.markdown("---")
    st.markdown("### Metrics Exported")

    metrics_table = [
        {"Metric": "agentsysperf_task_duration_seconds", "Source": "L1", "Description": "Wall-clock task duration"},
        {"Metric": "agentsysperf_task_cpu_time_seconds", "Source": "L1", "Description": "CPU time consumed"},
        {"Metric": "agentsysperf_task_memory_rss_bytes", "Source": "L1", "Description": "Peak resident set size"},
        {"Metric": "agentsysperf_task_ipc", "Source": "L3", "Description": "Instructions per cycle"},
        {"Metric": "agentsysperf_task_cache_miss_percent", "Source": "L3", "Description": "LLC cache miss rate"},
        {"Metric": "agentsysperf_tma_backend_bound_ratio", "Source": "PerfSpect", "Description": "TMA Backend Bound"},
        {"Metric": "agentsysperf_tma_retiring_ratio", "Source": "PerfSpect", "Description": "TMA Retiring"},
    ]
    st.dataframe(metrics_table, use_container_width=True)

    st.markdown("---")
    st.info("**Stack**: AgentSysPerf → Prometheus (scrape :9101/metrics every 5s) → Grafana (auto-provisioned dashboard). Zero manual wiring.")


elif demo_page == "Workflows::Spec Decode Workflow":
    st.title("Speculative Decoding Workflow")
    st.markdown("End-to-end: measure → detect bottleneck → map solution → apply → verify")

    # Flow diagram
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. MEASURE    Run vLLM decode_batch1                            │
    │               → IPC=0.8, cache miss=94%, RSS=14GB               │
    ├─────────────────────────────────────────────────────────────────┤
    │ 2. DETECT     MemoryBandwidthAnalyzer                           │
    │               → weight_streaming pattern (92% confidence)       │
    ├─────────────────────────────────────────────────────────────────┤
    │ 3. MAP        Solution registry                                 │
    │               → Speculative Decoding, Quantization, Sharding    │
    ├─────────────────────────────────────────────────────────────────┤
    │ 4. APPLY      DFlash SD (vllm PR #44029)                        │
    │               → --speculative-method=dflash --num-spec-tokens=8 │
    ├─────────────────────────────────────────────────────────────────┤
    │ 5. VERIFY     Re-measure with AgentSysPerf                         │
    │               → IPC: 0.8 → 1.9 | Throughput: 19 → 55 tok/s     │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### vLLM Benchmark Tasks")

    from src.benchmarks.vllm_inference import VLLMInferenceAdapter
    adapter = VLLMInferenceAdapter(model="meta-llama/Llama-3.1-8B-Instruct", speculative_model="meta-llama/Llama-3.2-1B-Instruct")
    tasks = list(adapter.list_tasks())
    adapter.teardown()

    task_data = [{"Task": t.id, "Category": t.category, "Method": t.extra.get("spec_method", "-"), "Description": t.extra["description"][:80]} for t in tasks]
    st.dataframe(task_data, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Before (Baseline)")
        st.metric("IPC", "0.8", delta=None)
        st.metric("Cache Miss", "94%")
        st.metric("Throughput", "19 tok/s")
        st.metric("Pattern", "weight_streaming")
    with col2:
        st.markdown("### After (DFlash SD)")
        st.metric("IPC", "1.9", delta="+137%")
        st.metric("Cache Miss", "94%", delta="same (expected)")
        st.metric("Throughput", "55 tok/s", delta="+189%")
        st.metric("Acceptance Rate", "25%")

    st.info("DFlash SD (PR #44029): self-speculative on CPU, no draft model needed. Verifies 3+ tokens per weight load.")


elif demo_page == "Platform::Architecture":
    st.title("Plugin Architecture")
    st.markdown("Pluggable at every layer — external teams add adapters without forking.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Benchmark Adapters")
        st.markdown("""
| Adapter | Domain | Status |
|---------|--------|--------|
| `synthetic_cpu` | CPU workloads | ✅ Implemented |
| `terminal_bench` | Bash/coding tasks | ✅ Implemented |
| `vllm_inference` | LLM serving | ✅ Implemented |
| `tau_bench` | Retail/airline/telecom | ✅ Implemented |
| `swe_bench` | Software engineering | ✅ Implemented |
| `openclaw` | Legal reasoning | 📋 Reference |
""")

    with col2:
        st.markdown("### Measurement Layers")
        st.markdown("""
| Layer | What It Measures | Status |
|-------|-----------------|--------|
| `l1_subspan` | CPU time, RSS, threads | ✅ |
| `l3_perf` | IPC, cache miss, branches | ✅ |
| `emon` | Full TMA via Intel EMON/pyEDP | ✅ |
| `perfspect` | TMA breakdown | ✅ |
""")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Analyzers")
        st.markdown("""
| Analyzer | Detects |
|----------|---------|
| `cpu_bound` | Core vs memory bottleneck |
| `cache` | L3 residency |
| `memory_leak` | RSS growth |
| `memory_bandwidth` | 6 patterns + 11 solutions |
| `emon_analyzer` | 7-layer TMA pipeline + findings |
| `phase_profiler` | Agentic phase decomposition |
| `breakdown` | Per-task attribution |
""")

    with col2:
        st.markdown("### Exporters")
        st.markdown("""
| Exporter | Target |
|----------|--------|
| Prometheus | Push or scrape mode |
| Grafana | Auto-provisioned dashboard |
| SQLite | Local result store |
| PowerPoint | Xeon report generator |
""")

    st.markdown("---")
    st.markdown("### Extension Model")
    st.code("""
# Add a new benchmark — implement 3 methods:
class MyBenchmarkAdapter:
    def list_tasks(self) -> Iterable[TaskSpec]: ...
    def run_task(self, task, agent_invoker) -> TaskResult: ...
    def teardown(self) -> None: ...

# Register in pyproject.toml:
[project.entry-points."agentsysperf.benchmarks"]
my_benchmark = "my_pkg:MyBenchmarkAdapter"
""", language="python")


# ─── Recommendations Page ─────────────────────────────────────────────────

elif demo_page in ("Terminal-Bench::Recommendations", "Tau-Bench::Recommendations", "SWE-Bench::Recommendations"):
    st.title(f"{_nav_section}: Recommendations")
    st.markdown(
        "Consolidated optimization and hardware recommendations derived from "
        "**Bottleneck Analysis**, **EMON TMA Analysis**, and **Phase Profiler** findings."
    )

    # ─── Gather findings from all analysis layers ─────────────────────
    _rec_emon_findings = []
    _rec_phase_findings = []
    _rec_bottleneck_findings = []
    _rec_workload_class = "unknown"

    # EMON findings (store-first, legacy fallback)
    _EMON_FALLBACK_DIRS_REC = {
        # Mixed-workload density cells first (d192 = max density, strongest
        # contention signal), then the legacy single-workload EMON dirs as fallback.
        "Terminal-Bench": [Path(f"{_TMP}/agentsysperf_mixed4_d192_emon/d192_spread_mixed"),
                           Path(f"{_TMP}/agentsysperf_mixed4_emon/d80_spread_mixed"),
                           Path(f"{_TMP}/agentsysperf_emon_tb2"), Path(f"{_TMP}/emon_tb2_test")],
        "Tau-Bench": [Path(f"{_TMP}/agentsysperf_emon_tau"), Path(f"{_TMP}/agentsysperf_tau_bench"), Path(f"{_TMP}/agentsysperf_scaling")],
        "SWE-Bench": [Path(f"{_TMP}/agentsysperf_swe_bench"), Path(f"{_TMP}/agentsysperf_phase_swe")],
    }
    _rec_emon_csv = _data.get_artifact_path(
        _BENCH_SLUG.get(_nav_section, _nav_section),
        kind="emon_csv",
        fallback_dirs=_EMON_FALLBACK_DIRS_REC.get(_nav_section, []),
        fallback_patterns=("*_system_view_summary.csv", "*_system_view_details.csv",
                           "*_socket_view_summary.csv", "*_socket_view_details.csv"),
    )

    if _rec_emon_csv:
        try:
            from src.protocols import MeasurementRecord
            _ea = _emon_analyzer()
            _er_results = (
                list(_ea.analyze([MeasurementRecord(span_id="rec", layer="emon", payload={"csv_path": str(_rec_emon_csv)})]))
                if _ea is not None else []
            )
            if _er_results:
                _ev = _er_results[0].evidence
                _rec_workload_class = _ev.get("workload_class", "unknown")
                for _rc in _ev.get("root_causes", []):
                    _rec_emon_findings.append({
                        "source": "EMON",
                        "severity": _rc["severity"],
                        "category": _rc["category"],
                        "headline": _rc["headline"],
                        "fix": _rc["fix"],
                        "gain": _rc["gain_range"],
                    })
        except Exception:
            pass

    # Phase Profiler findings (store-first, legacy fallback)
    _PHASE_FALLBACK_DIRS_REC = {
        "Terminal-Bench": [Path(f"{_TMP}/agentsysperf_emon_tb2"), Path(f"{_TMP}/agentsysperf_phase_tb2"), Path(f"{_TMP}/agentsysperf_phase_tb2_dry")],
        "Tau-Bench": [Path(f"{_TMP}/agentsysperf_tau_bench"), Path(f"{_TMP}/agentsysperf_phase_tau"), Path(f"{_TMP}/agentsysperf_scaling")],
        "SWE-Bench": [Path(f"{_TMP}/agentsysperf_swe_bench"), Path(f"{_TMP}/agentsysperf_phase_swe")],
    }
    _rec_phase_raw = _data.get_records(
        _BENCH_SLUG.get(_nav_section, _nav_section),
        fallback_paths=_PHASE_FALLBACK_DIRS_REC.get(_nav_section, []),
    )

    if _rec_phase_raw:
        try:
            from src.analyzers.phase_profiler import PhaseProfiler
            from src.protocols import MeasurementRecord
            _records = [MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"]) for r in _rec_phase_raw]
            _pp = PhaseProfiler()
            _pp_results = list(_pp.analyze(_records))
            if _pp_results:
                _pr = _pp_results[0]
                _bd = _pr.evidence.get("phase_breakdown", {})
                _inflection = _pr.evidence.get("inflection")
                # Derive findings from dominant phase + inflection
                _dominant = _pr.verdict.replace("phase_profile_", "")
                if _dominant == "reason":
                    _rec_phase_findings.append({
                        "source": "Phase",
                        "severity": "HIGH",
                        "category": "inference",
                        "headline": "LLM inference dominates pipeline (reason phase)",
                        "fix": "Speculative decoding, quantization, or batch scheduling",
                        "gain": "20-50%",
                    })
                elif _dominant == "act":
                    _rec_phase_findings.append({
                        "source": "Phase",
                        "severity": "HIGH",
                        "category": "tool_execution",
                        "headline": "Tool/code execution dominates pipeline (act phase)",
                        "fix": "Async tool dispatch, Docker pre-warming, I/O optimization",
                        "gain": "15-40%",
                    })
                elif _dominant == "retrieve":
                    _rec_phase_findings.append({
                        "source": "Phase",
                        "severity": "MEDIUM",
                        "category": "retrieval",
                        "headline": "RAG retrieval dominates pipeline",
                        "fix": "Vector index tuning, cache embeddings, reduce context window",
                        "gain": "10-30%",
                    })
                if _inflection:
                    _rec_phase_findings.append({
                        "source": "Phase",
                        "severity": "MEDIUM",
                        "category": "orchestration",
                        "headline": f"Orchestration inflection at iteration {_inflection['iteration']} (ratio {_inflection['ratio']:.1f}x)",
                        "fix": "Reduce agent turns, improve prompt efficiency, early-exit strategies",
                        "gain": "10-25%",
                    })
                # Flag phases with anomalous patterns
                for _phase, _pdata in _bd.items():
                    if _pdata.get("pattern", "").startswith("cpu_spin"):
                        _rec_phase_findings.append({
                            "source": "Phase",
                            "severity": "MEDIUM",
                            "category": "spin_wait",
                            "headline": f"CPU spin-wait detected in '{_phase}' phase",
                            "fix": "Replace busy-wait with epoll/event-driven I/O",
                            "gain": "5-15%",
                        })
        except Exception:
            pass

    # Bottleneck Analysis findings (Terminal-Bench only has this currently)
    if _nav_section == "Terminal-Bench":
        try:
            _bn_analysis = load_analysis()
            for _item in _bn_analysis:
                if _item["verdict"] != "no_memory_bottleneck":
                    for _s in _item.get("solutions", []):
                        _rec_bottleneck_findings.append({
                            "source": "Bottleneck",
                            "severity": "HIGH",
                            "category": _s.get("category", "memory"),
                            "headline": f"{_item['task']}: {_item['verdict']}",
                            "fix": _s["name"],
                            "gain": _s.get("impact", "N/A"),
                        })
        except Exception:
            pass

    # ─── Severity ordering ────────────────────────────────────────────
    _sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    _all_findings = _rec_emon_findings + _rec_phase_findings + _rec_bottleneck_findings
    _all_findings.sort(key=lambda f: _sev_order.get(f["severity"], 5))

    if not _all_findings:
        st.info(
            f"No analysis data available for **{_nav_section}** yet. Run the benchmark with EMON enabled "
            f"to generate recommendations."
        )
    else:
        st.success(f"**{len(_all_findings)}** findings across {len(set(f['source'] for f in _all_findings))} analysis layers "
                   f"| Workload class: **{_rec_workload_class}**")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: Optimizations
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("Optimizations")
    st.markdown(
        "Software configurations, algorithms, and orchestration strategies "
        "mapped to CPU features and workload characteristics."
    )

    if _all_findings:
        # Ranked findings table
        st.subheader("Prioritized Action Plan")
        _opt_rows = []
        for _i, _f in enumerate(_all_findings, 1):
            _opt_rows.append({
                "#": _i,
                "Source": _f["source"],
                "Severity": _f["severity"],
                "Category": _f["category"],
                "Finding": _f["headline"],
                "Recommended Fix": _f["fix"],
                "Expected Gain": _f["gain"],
            })
        st.dataframe(_opt_rows, use_container_width=True, hide_index=True)

    # Software optimizations mapped to CPU features
    st.subheader("CPU Feature → Optimization Mapping")

    _sw_optimizations = [
        {
            "CPU Feature": "AMX (Advanced Matrix Extensions)",
            "Optimization": "INT8/BF16 GEMM acceleration for LLM inference",
            "Applicable When": "Inference-bound (reason phase > 60%)",
            "Config": "vLLM: --dtype bfloat16, oneDNN AMX kernels",
            "Category": "Algorithmic",
        },
        {
            "CPU Feature": "AVX-512 VNNI",
            "Optimization": "Vectorized INT8 quantization (GPTQ/AWQ)",
            "Applicable When": "Weight streaming bottleneck (DRAM BW > 70%)",
            "Config": "quantize model to INT4/INT8, use llama.cpp GGUF",
            "Category": "Algorithmic",
        },
        {
            "CPU Feature": "SNC3 (Sub-NUMA Clustering)",
            "Optimization": "NUMA-aware agent scheduling; pin agents to SNC domains",
            "Applicable When": "Cross-NUMA traffic > 15%, LLC MPI > 0.01",
            "Config": "BIOS: SNC=3, numactl --cpubind, VLLM_CPU_OMP_THREADS_BIND",
            "Category": "Scheduling",
        },
        {
            "CPU Feature": "Intel RDT (Cache Partitioning)",
            "Optimization": "Isolate LLC for inference vs agent workloads",
            "Applicable When": "LLC eviction rate > 30%, capacity thrashing",
            "Config": "pqos -e llc:1=0x00f;llc:2=0xff0",
            "Category": "Hardware",
        },
        {
            "CPU Feature": "Speculative Decoding",
            "Optimization": "Draft model (small) + verify (large) for faster generation",
            "Applicable When": "Single-stream latency matters, reason phase dominant",
            "Config": "vLLM: --speculative-model draft-7B --num-speculative-tokens 5",
            "Category": "Algorithmic",
        },
        {
            "CPU Feature": "Hardware Prefetcher Tuning",
            "Optimization": "Adjust DCU/L2 prefetcher for KV-cache access patterns",
            "Applicable When": "L2 MPI > 0.02, streaming access in attention layers",
            "Config": "MSR 0x1A4: disable DCU IP prefetcher for attention, enable for MLP",
            "Category": "Hardware",
        },
        {
            "CPU Feature": "Transparent Huge Pages (THP)",
            "Optimization": "Reduce dTLB misses for large model allocations",
            "Applicable When": "dTLB miss latency > 10 cycles, model > 4GB",
            "Config": "echo always > /sys/kernel/mm/transparent_hugepage/enabled",
            "Category": "OS/Kernel",
        },
        {
            "CPU Feature": "CXL Memory Expansion",
            "Optimization": "Extend KV-cache capacity to CXL-attached memory",
            "Applicable When": "DRAM capacity limit hit, KV-cache overflow",
            "Config": "numactl --preferred=cxl_node for KV-cache allocations",
            "Category": "Hardware",
        },
    ]
    st.dataframe(_sw_optimizations, use_container_width=True, hide_index=True)

    # Orchestration optimizations
    st.subheader("Orchestration & Scheduling Optimizations")
    _orch_optimizations = [
        {
            "Strategy": "Batch Agent Scheduling",
            "Description": "Group agent requests to maximize vLLM continuous batching",
            "Trigger": "Low GPU/CPU utilization at low concurrency",
            "Expected Impact": "2-4x throughput improvement",
        },
        {
            "Strategy": "Async Tool Dispatch",
            "Description": "Fire tool calls without blocking inference pipeline",
            "Trigger": "Act phase > 30% with I/O-bound tools",
            "Expected Impact": "15-30% latency reduction",
        },
        {
            "Strategy": "Early-Exit Reasoning",
            "Description": "Stop generation when confidence threshold met",
            "Trigger": "Long reasoning chains with diminishing returns after inflection",
            "Expected Impact": "20-40% token reduction",
        },
        {
            "Strategy": "Prompt Compression",
            "Description": "LLMLingua/AutoCompressor to reduce context tokens",
            "Trigger": "Prompt tokens > 4K, frontend bound > 40%",
            "Expected Impact": "30-60% prefill latency reduction",
        },
        {
            "Strategy": "Density-Aware Autoscaler",
            "Description": "Scale agents up to the Kneedle-detected saturation knee",
            "Trigger": "Throughput plateaus at density > 1.5 agents/core",
            "Expected Impact": "Optimal throughput without tail latency degradation",
        },
    ]
    st.dataframe(_orch_optimizations, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: HW Recommendation
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("HW Recommendation")
    st.markdown(
        "CPU and platform configuration recommendations based on workload "
        "characteristics observed during benchmark execution."
    )

    # Determine recommendation based on workload class and findings
    _is_memory_bound = any(f["category"] in ("memory", "bandwidth", "latency") for f in _all_findings)
    _is_compute_bound = any(f["category"] in ("compute", "tma", "retiring") for f in _all_findings)
    _is_io_bound = any(f["category"] in ("io", "ddio") for f in _all_findings)
    _has_coherency = any(f["category"] in ("coherency", "false_sharing") for f in _all_findings)

    st.subheader("Recommended CPU Configuration")

    # Build SKU recommendation
    _sku_rec = {
        "Parameter": [],
        "Recommendation": [],
        "Rationale": [],
    }

    # Core count
    _sku_rec["Parameter"].append("SKU / Core Count")
    if _rec_workload_class in ("agentic", "latency"):
        _sku_rec["Recommendation"].append("Xeon 6900P series (96+ cores)")
        _sku_rec["Rationale"].append("Agentic workloads scale with core count for concurrent agent execution")
    elif _is_memory_bound:
        _sku_rec["Recommendation"].append("Xeon 6900E series (144 cores, efficiency)")
        _sku_rec["Rationale"].append("Memory-bound workloads benefit from more memory channels per core")
    else:
        _sku_rec["Recommendation"].append("Xeon 6700P series (64-86 cores)")
        _sku_rec["Rationale"].append("Balanced compute/memory for mixed workloads")

    # Memory configuration
    _sku_rec["Parameter"].append("Memory Config")
    if _is_memory_bound:
        _sku_rec["Recommendation"].append("12 channels DDR5-5600 MCR, 2DPC")
        _sku_rec["Rationale"].append("Maximum bandwidth for weight streaming and KV-cache")
    else:
        _sku_rec["Recommendation"].append("12 channels DDR5-4800, 1DPC")
        _sku_rec["Rationale"].append("Sufficient bandwidth, lower cost")

    # SNC configuration
    _sku_rec["Parameter"].append("SNC Mode")
    if _has_coherency or _rec_workload_class == "agentic":
        _sku_rec["Recommendation"].append("SNC3 (3 sub-NUMA clusters)")
        _sku_rec["Rationale"].append("Reduces cross-cluster snoops, improves LLC locality for pinned agents")
    else:
        _sku_rec["Recommendation"].append("SNC off (monolithic)")
        _sku_rec["Rationale"].append("Simpler scheduling, no SNC migration overhead")

    # Prefetcher
    _sku_rec["Parameter"].append("HW Prefetcher")
    _sku_rec["Recommendation"].append("DCU Streamer: ON, DCU IP: OFF for attention layers")
    _sku_rec["Rationale"].append("Attention has stride-varying patterns; IP prefetcher pollutes LLC")

    # THP
    _sku_rec["Parameter"].append("Huge Pages")
    _sku_rec["Recommendation"].append("1GB hugepages for model weights, 2MB THP for KV-cache")
    _sku_rec["Rationale"].append("Reduces dTLB pressure for large allocations (>30GB model + cache)")

    # Power
    _sku_rec["Parameter"].append("Power Profile")
    if _rec_workload_class in ("agentic", "latency"):
        _sku_rec["Recommendation"].append("Performance bias: max frequency, C-states limited to C1")
        _sku_rec["Rationale"].append("Latency-sensitive agentic workloads penalized by C-state exit latency")
    else:
        _sku_rec["Recommendation"].append("Balanced bias: EPB=7, C6 enabled")
        _sku_rec["Rationale"].append("Throughput workloads tolerate wakeup latency; saves power between batches")

    # CXL
    _sku_rec["Parameter"].append("CXL Expansion")
    if _is_memory_bound and _rec_workload_class == "agentic":
        _sku_rec["Recommendation"].append("CXL 2.0 Type 3 memory (128-512 GB) for KV-cache overflow")
        _sku_rec["Rationale"].append("Extended agent conversation history without DRAM capacity limit")
    else:
        _sku_rec["Recommendation"].append("Not required")
        _sku_rec["Rationale"].append("DRAM capacity sufficient for current workload")

    # I/O
    _sku_rec["Parameter"].append("I/O & Network")
    if _is_io_bound:
        _sku_rec["Recommendation"].append("DDIO enabled, 100GbE with RDMA for distributed inference")
        _sku_rec["Rationale"].append("I/O bound workload benefits from DDIO direct cache injection")
    else:
        _sku_rec["Recommendation"].append("Standard NVMe + 25GbE sufficient")
        _sku_rec["Rationale"].append("Workload is not I/O constrained")

    st.dataframe(_sku_rec, use_container_width=True, hide_index=True)

    # Platform comparison
    st.subheader("Platform Comparison for Agentic Workloads")
    _platform_cmp = [
        {
            "Platform": "Granite Rapids (Xeon 6900P)",
            "Cores": "96-128",
            "LLC": "288-504 MB",
            "Mem BW": "307 GB/s (12ch DDR5-5600 MCR)",
            "Key Feature": "AMX, SNC3, P-core density",
            "Best For": "High-concurrency agentic inference",
        },
        {
            "Platform": "Granite Rapids (Xeon 6900E)",
            "Cores": "128-144",
            "LLC": "324-504 MB",
            "Mem BW": "307 GB/s",
            "Key Feature": "E-core efficiency, max density",
            "Best For": "Throughput-oriented batch processing",
        },
        {
            "Platform": "Emerald Rapids (Xeon 8592+)",
            "Cores": "64",
            "LLC": "320 MB",
            "Mem BW": "307 GB/s (8ch DDR5-4800)",
            "Key Feature": "Mature ecosystem, wide availability",
            "Best For": "Existing deployments, moderate concurrency",
        },
        {
            "Platform": "Clearwater Forest (Xeon 6P+E)",
            "Cores": "288",
            "LLC": "576 MB",
            "Mem BW": "800+ GB/s (MCDIMM)",
            "Key Feature": "3D stacked E-cores, max throughput",
            "Best For": "Future: max-density agentic serving (2025+)",
        },
    ]
    st.dataframe(_platform_cmp, use_container_width=True, hide_index=True)

    # Workload-specific recommendation summary
    st.subheader("Summary")
    if _rec_workload_class == "agentic":
        st.success(
            "**Agentic workload detected.** Recommended: Xeon 6900P (96+ cores) with SNC3, "
            "DDR5-5600 MCR, 1GB hugepages for model, and performance power profile. "
            "Pin vLLM to NUMA 0-1, agents to NUMA 2 with 4 cores/agent."
        )
    elif _is_memory_bound:
        st.info(
            "**Memory-bound workload.** Recommended: Maximize memory bandwidth — "
            "2DPC DDR5-5600, enable SNC for LLC locality, consider CXL for capacity. "
            "Focus on quantization and speculative decoding to reduce memory pressure."
        )
    elif _is_compute_bound:
        st.info(
            "**Compute-bound workload.** Recommended: High-frequency P-core SKU, "
            "AMX-accelerated inference, aggressive batching to amortize overheads."
        )
    else:
        st.info(
            "**Balanced workload.** Recommended: Standard Granite Rapids configuration "
            "with SNC off, balanced power profile. Focus on software optimizations first."
        )
