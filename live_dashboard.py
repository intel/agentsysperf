#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
AgentSysPerf Live Dashboard
=========================
Run with: streamlit run live_dashboard.py --server.port 7861 --server.address 0.0.0.0
Live URL: http://<this-host>:7861

This dashboard runs benchmarks on demand and updates charts in real-time.
Unlike the demo app (port 7860), this one executes actual workloads.
"""

import sys
from pathlib import Path as _Path
# Make the repo importable when running this script directly, without a
# hardcoded absolute path (works for any checkout location).
sys.path.insert(0, str(_Path(__file__).resolve().parent))

import json
import time
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="AgentSysPerf Live",
    page_icon="📊",
    layout="wide",
)

# ─── State ───────────────────────────────────────────────────────────────

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = []
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ─── Sidebar Controls ────────────────────────────────────────────────────

st.sidebar.title("AgentSysPerf Live")
st.sidebar.markdown("---")

# Benchmark configuration
st.sidebar.subheader("Benchmark Config")
duration = st.sidebar.slider("Duration per task (s)", 1, 10, 2)
selected_tasks = st.sidebar.multiselect(
    "Workloads",
    ["compile", "ml_train", "linalg", "io", "compress", "raytrace", "sat", "interpreter", "control"],
    default=["compile", "ml_train", "linalg", "io", "compress"],
)

# Run button
run_button = st.sidebar.button("▶ Run Benchmark", type="primary", use_container_width=True)

# Clear results
if st.sidebar.button("🗑 Clear Results", use_container_width=True):
    st.session_state.benchmark_results = []
    st.session_state.run_count = 0
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Runs completed**: {st.session_state.run_count}")
st.sidebar.markdown(f"**Data points**: {len(st.session_state.benchmark_results)}")

# ─── Run Benchmark ───────────────────────────────────────────────────────

def run_benchmark(tasks, duration_s):
    """Run AgentSysPerf benchmark and return results."""
    from src.benchmarks.synthetic_cpu import SyntheticCpuAdapter
    from src.measurements.l1_subspan import L1SubSpanMeasurement
    from src.measurements.l3_perf import L3PerfMeasurement
    from src.protocols import TaskSpec
    from src.runner import RunContext, track_span

    class _NoopInvoker:
        def invoke(self, instruction, **kwargs):
            return None

    output_dir = Path(f"{_TMP}/agentsysperf_live/run_{int(time.time())}")
    adapter = SyntheticCpuAdapter()
    invoker = _NoopInvoker()

    ctx = RunContext(
        measurements=[L1SubSpanMeasurement(), L3PerfMeasurement(target="self")],
        output_dir=output_dir,
    )

    results = []
    with ctx:
        time.sleep(0.2)
        for spec in adapter.list_tasks(include=tasks):
            spec = TaskSpec(
                **{**spec.__dict__, "extra": {**spec.extra, "duration_s": duration_s}},
            )
            span_id = f"{ctx.run_id}::{spec.id}"

            with track_span(ctx, span_id, kind="synthetic_cpu", node_id=spec.id):
                adapter.run_task(spec, agent_invoker=invoker)

            l1 = next((r for r in ctx.records if r.span_id == span_id and r.layer == "l1"), None)
            l3 = next((r for r in ctx.records if r.span_id == span_id and r.layer == "l3"), None)

            if l1 and l3:
                results.append({
                    "task": spec.id,
                    "run_id": ctx.run_id,
                    "timestamp": time.time(),
                    "duration_ms": round(l1.payload["duration_us"] / 1000, 0),
                    "cpu_time_s": round(l1.payload.get("cpu_time_s", 0), 2),
                    "cpu_pct": round(l1.payload.get("cpu_pct_mean", 0), 1),
                    "rss_mb": round(l1.payload.get("rss_kb_peak", 0) / 1024, 0),
                    "ipc": round(l3.payload.get("ipc", 0), 2),
                    "cache_miss_pct": round(l3.payload.get("cache_miss_pct", 0), 1),
                })

    adapter.teardown()
    return results


# ─── Main Content ────────────────────────────────────────────────────────

st.title("📊 AgentSysPerf Live Dashboard")

if run_button:
    st.session_state.is_running = True

    with st.spinner(f"Running {len(selected_tasks)} workloads ({duration}s each)..."):
        new_results = run_benchmark(selected_tasks, duration)
        st.session_state.benchmark_results.extend(new_results)
        st.session_state.run_count += 1

    st.session_state.is_running = False
    st.rerun()

# ─── Display Results ─────────────────────────────────────────────────────

results = st.session_state.benchmark_results

if not results:
    st.info("Click **▶ Run Benchmark** in the sidebar to start collecting data.")
    st.stop()

# Latest run results
latest_run_id = results[-1]["run_id"]
latest_results = [r for r in results if r["run_id"] == latest_run_id]

# ─── Summary Metrics ─────────────────────────────────────────────────────

st.subheader("Latest Run Summary")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Tasks", len(latest_results))
with col2:
    avg_ipc = sum(r["ipc"] for r in latest_results) / len(latest_results)
    st.metric("Avg IPC", f"{avg_ipc:.2f}")
with col3:
    avg_cache = sum(r["cache_miss_pct"] for r in latest_results) / len(latest_results)
    st.metric("Avg Cache Miss", f"{avg_cache:.1f}%")
with col4:
    avg_cpu = sum(r["cpu_pct"] for r in latest_results) / len(latest_results)
    st.metric("Avg CPU%", f"{avg_cpu:.1f}%")
with col5:
    total_time = sum(r["duration_ms"] for r in latest_results) / 1000
    st.metric("Total Time", f"{total_time:.1f}s")

# ─── Charts ──────────────────────────────────────────────────────────────

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    # IPC bar chart (latest run)
    tasks = [r["task"] for r in latest_results]
    ipcs = [r["ipc"] for r in latest_results]

    fig_ipc = go.Figure(go.Bar(
        x=ipcs,
        y=tasks,
        orientation="h",
        marker_color=["#d32f2f" if v < 2 else "#1976d2" if v < 4 else "#388e3c" for v in ipcs],
        text=[f"{v:.2f}" for v in ipcs],
        textposition="outside",
    ))
    fig_ipc.update_layout(
        title="Instructions Per Cycle (IPC)",
        xaxis_title="IPC",
        height=350,
        margin=dict(l=100, r=50),
    )
    st.plotly_chart(fig_ipc, use_container_width=True)

with col2:
    # Cache miss bar chart (latest run)
    cache_miss = [r["cache_miss_pct"] for r in latest_results]

    fig_cache = go.Figure(go.Bar(
        x=cache_miss,
        y=tasks,
        orientation="h",
        marker_color=["#d32f2f" if v > 50 else "#ff9800" if v > 20 else "#388e3c" for v in cache_miss],
        text=[f"{v:.1f}%" for v in cache_miss],
        textposition="outside",
    ))
    fig_cache.update_layout(
        title="Cache Miss Rate (%)",
        xaxis_title="Cache Miss %",
        height=350,
        margin=dict(l=100, r=50),
    )
    st.plotly_chart(fig_cache, use_container_width=True)

# ─── Trend Charts (across runs) ─────────────────────────────────────────

if st.session_state.run_count > 1:
    st.markdown("---")
    st.subheader("Trend Across Runs")

    # Group by task, show IPC over time
    all_tasks = list(set(r["task"] for r in results))

    fig_trend = go.Figure()
    for task in sorted(all_tasks):
        task_results = [r for r in results if r["task"] == task]
        fig_trend.add_trace(go.Scatter(
            x=list(range(len(task_results))),
            y=[r["ipc"] for r in task_results],
            mode="lines+markers",
            name=task,
        ))

    fig_trend.update_layout(
        title="IPC Trend Across Runs",
        xaxis_title="Run #",
        yaxis_title="IPC",
        height=350,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ─── Analysis ────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Bottleneck Analysis")

from src.analyzers.memory_bandwidth import MemoryBandwidthAnalyzer
from src.protocols import MeasurementRecord

# Build records from latest results
records = []
for r in latest_results:
    span_id = f"{r['run_id']}::{r['task']}"
    records.append(MeasurementRecord(span_id=span_id, layer="l1", payload={
        "duration_us": r["duration_ms"] * 1000,
        "cpu_time_s": r["cpu_time_s"],
        "cpu_pct_mean": r["cpu_pct"],
        "rss_kb_peak": r["rss_mb"] * 1024,
    }))
    records.append(MeasurementRecord(span_id=span_id, layer="l3", payload={
        "ipc": r["ipc"],
        "cache_miss_pct": r["cache_miss_pct"],
    }))

analyzer = MemoryBandwidthAnalyzer()
analysis_results = list(analyzer.analyze(records))

for ar in analysis_results:
    task = ar.span_id.split("::")[-1]
    if ar.verdict == "no_memory_bottleneck":
        st.success(f"**{task}**: No memory bottleneck (compute-efficient)")
    else:
        patterns = ar.evidence.get("patterns_matched", [])
        solutions = ar.evidence.get("solutions", [])
        with st.expander(f"⚠️ **{task}**: {ar.verdict} ({ar.confidence:.0%})", expanded=False):
            for p in patterns:
                st.markdown(f"- Pattern: `{p['pattern']}` ({p['confidence']:.0%})")
            if solutions:
                st.markdown("**Solutions:**")
                for s in solutions[:3]:
                    st.markdown(f"- {s['name']} [{s['category']}]: {s.get('impact', s.get('expected_impact', ''))}")

# ─── Phase Analysis ──────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Phase Analysis (Agentic Pipeline)")

from src.analyzers.phase_profiler import PhaseProfiler

# Load phase-tagged records from the most recent TB2 run
_phase_records_dir = None
for _candidate in [
    Path(f"{_TMP}/agentsysperf_emon_tb2"),
    Path(f"{_TMP}/agentsysperf_phase_tb2"),
    Path(f"{_TMP}/agentsysperf_phase_tb2_dry"),
    Path(f"{_TMP}/agentsysperf_tb2_benchmark"),
    Path(f"{_TMP}/agentsysperf_tb2_simple"),
]:
    _records_file = _candidate / "measurement_records.json"
    if _records_file.exists():
        _phase_records_dir = _candidate
        break

if _phase_records_dir is not None:
    import json as _json
    _records_file = _phase_records_dir / "measurement_records.json"
    _raw_phase_records = _json.loads(_records_file.read_text())
    _phase_measurement_records = [
        MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"])
        for r in _raw_phase_records
    ]

    _profiler = PhaseProfiler()
    _phase_results = list(_profiler.analyze(_phase_measurement_records))

    if _phase_results:
        _pr = _phase_results[0]
        _breakdown = _pr.evidence.get("phase_breakdown", {})
        _inflection = _pr.evidence.get("inflection")

        # Summary metrics
        _pcol1, _pcol2, _pcol3, _pcol4 = st.columns(4)
        with _pcol1:
            st.metric("Dominant Phase", _pr.verdict.replace("phase_profile_", "").title())
        with _pcol2:
            st.metric("Confidence", f"{_pr.confidence:.0%}")
        with _pcol3:
            st.metric("Phases Detected", len(_breakdown))
        with _pcol4:
            _iters = _pr.evidence.get("iteration_count", 0)
            st.metric("Iterations", _iters)

        st.markdown("")

        # Phase breakdown stacked bar chart
        if _breakdown:
            _phases = list(_breakdown.keys())
            _wall_pcts = [_breakdown[p]["wall_pct"] for p in _phases]
            _cpu_pcts = [_breakdown[p]["cpu_pct"] for p in _phases]

            _phase_colors = {
                "admit": "#9e9e9e",
                "retrieve": "#2196f3",
                "reason": "#f44336",
                "act": "#4caf50",
                "commit": "#ff9800",
            }

            _col_a, _col_b = st.columns(2)

            with _col_a:
                _fig_phase = go.Figure()
                for _p in _phases:
                    _fig_phase.add_trace(go.Bar(
                        name=_p.title(),
                        x=[_breakdown[_p]["wall_pct"]],
                        y=["Wall-Clock"],
                        orientation="h",
                        marker_color=_phase_colors.get(_p, "#607d8b"),
                        text=f"{_breakdown[_p]['wall_pct']:.1f}%",
                        textposition="inside",
                    ))
                    _fig_phase.add_trace(go.Bar(
                        name=_p.title(),
                        x=[_breakdown[_p]["cpu_pct"]],
                        y=["CPU-Time"],
                        orientation="h",
                        marker_color=_phase_colors.get(_p, "#607d8b"),
                        text=f"{_breakdown[_p]['cpu_pct']:.1f}%",
                        textposition="inside",
                        showlegend=False,
                    ))
                _fig_phase.update_layout(
                    title="Phase Time Breakdown",
                    barmode="stack",
                    height=200,
                    margin=dict(l=80, r=20, t=40, b=20),
                    legend=dict(orientation="h", y=-0.3),
                    xaxis_title="%",
                )
                st.plotly_chart(_fig_phase, use_container_width=True)

            with _col_b:
                # Hardware signature per phase (IPC vs Cache Miss)
                _ipc_vals = []
                _miss_vals = []
                _phase_labels = []
                for _p in _phases:
                    _d = _breakdown[_p]
                    if _d["avg_ipc"] is not None and _d["avg_cache_miss_pct"] is not None:
                        _ipc_vals.append(_d["avg_ipc"])
                        _miss_vals.append(_d["avg_cache_miss_pct"])
                        _phase_labels.append(_p.title())

                if _ipc_vals:
                    _fig_hw = go.Figure()
                    _fig_hw.add_trace(go.Scatter(
                        x=_ipc_vals,
                        y=_miss_vals,
                        mode="markers+text",
                        text=_phase_labels,
                        textposition="top center",
                        marker=dict(
                            size=[_breakdown[p.lower()]["wall_pct"] * 0.8 + 10 for p in _phase_labels],
                            color=[_phase_colors.get(p.lower(), "#607d8b") for p in _phase_labels],
                        ),
                    ))
                    _fig_hw.update_layout(
                        title="Hardware Signature (bubble = wall-clock %)",
                        xaxis_title="IPC (higher = compute-efficient)",
                        yaxis_title="Cache Miss % (higher = memory-bound)",
                        height=300,
                        margin=dict(l=60, r=20, t=40, b=40),
                        showlegend=False,
                    )
                    # Add quadrant annotations
                    _fig_hw.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
                    _fig_hw.add_vline(x=2.0, line_dash="dash", line_color="gray", opacity=0.5)
                    st.plotly_chart(_fig_hw, use_container_width=True)
                else:
                    st.info("L3 perf counters not available — run with perf access for IPC/cache-miss data.")

        # Phase details table
        _table_data = []
        for _p, _d in _breakdown.items():
            _table_data.append({
                "Phase": _p.title(),
                "Wall %": _d["wall_pct"],
                "CPU %": _d["cpu_pct"],
                "Wall (ms)": _d["wall_ms"],
                "IPC": _d["avg_ipc"] if _d["avg_ipc"] is not None else "—",
                "Cache Miss %": _d["avg_cache_miss_pct"] if _d["avg_cache_miss_pct"] is not None else "—",
                "Pattern": _d["pattern"],
                "Spans": _d["span_count"],
            })
        st.dataframe(_table_data, use_container_width=True)

        # Inflection point
        if _inflection:
            st.warning(
                f"**Inflection at iteration {_inflection['iteration']}**: "
                f"Orchestration overhead (Retrieve+Act+Commit) exceeds inference time. "
                f"Ratio: {_inflection['ratio']:.2f}x. "
                f"CPU optimization in non-inference phases now has more ROI than model optimization."
            )

        # Solutions
        _phase_solutions = _pr.evidence.get("phase_solutions", {})
        _has_solutions = any(s for s in _phase_solutions.values())
        if _has_solutions:
            with st.expander("Per-Phase Optimization Solutions (Intel Xeon)", expanded=False):
                for _p, _sols in _phase_solutions.items():
                    if _sols:
                        _pattern = _breakdown.get(_p, {}).get("pattern", "")
                        st.markdown(f"**{_p.title()}** (`{_pattern}`): {'; '.join(_sols)}")

        # Recommendations
        if _pr.recommendations:
            with st.expander("Recommendations", expanded=True):
                for _rec in _pr.recommendations:
                    st.markdown(f"- {_rec}")

        st.caption(f"Source: `{_phase_records_dir}`")
    else:
        st.info("Phase-tagged records found but PhaseProfiler produced no results.")
else:
    st.info(
        "No phase-tagged benchmark data available. Run:\n\n"
        "```\npoetry run python examples/run_phase_profiler_tb2.py --dry-run\n```\n\n"
        "Then reload this page."
    )

# ─── EMON TMA ───────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("EMON TMA Analysis")

_emon_csv_path = None
for _emon_dir in [Path(f"{_TMP}/agentsysperf_emon_tb2"), Path(f"{_TMP}/agentsysperf_scaling"), Path(f"{_TMP}/emon_tb2_test")]:
    if not _emon_dir.exists():
        continue
    for _pattern in ("*_system_view_summary.csv", "*_system_view_details.csv",
                     "*_socket_view_summary.csv", "*_socket_view_details.csv"):
        _matches = sorted(_emon_dir.glob(_pattern))
        if _matches:
            _emon_csv_path = _matches[0]
            break
    if _emon_csv_path:
        break

if _emon_csv_path:
    import pandas as pd
    _emon_df = pd.read_csv(_emon_csv_path)
    st.caption(f"Source: `{_emon_csv_path}` ({len(_emon_df.columns)} metrics x {len(_emon_df)} samples)")

    def _find_emon_metric(df, keyword):
        for col in df.columns:
            if keyword.lower() in col.lower():
                try:
                    return float(df[col].mean())
                except (ValueError, TypeError):
                    pass
        return None

    _tma_l1 = {}
    _tma_search = {
        "Frontend_Bound": "tma_frontend_bound(%)",
        "Backend_Bound": "tma_backend_bound(%)",
        "Bad_Speculation": "tma_bad_speculation(%)",
        "Retiring": "tma_retiring(%)",
    }
    for _label, _kw in _tma_search.items():
        _val = _find_emon_metric(_emon_df, _kw)
        if _val is not None:
            _tma_l1[_label] = _val

    if _tma_l1:
        _tma_cols = st.columns(len(_tma_l1))
        for _i, (_k, _v) in enumerate(_tma_l1.items()):
            with _tma_cols[_i]:
                st.metric(_k.replace("_", " "), f"{_v:.1f}%")

        _tma_colors = {"Frontend_Bound": "#ff9800", "Backend_Bound": "#d32f2f",
                       "Bad_Speculation": "#9c27b0", "Retiring": "#4caf50"}
        _fig_tma = go.Figure(go.Bar(
            x=list(_tma_l1.values()),
            y=[k.replace("_", " ") for k in _tma_l1.keys()],
            orientation="h",
            marker_color=[_tma_colors.get(k, "#607d8b") for k in _tma_l1.keys()],
            text=[f"{v:.1f}%" for v in _tma_l1.values()],
            textposition="outside",
        ))
        _fig_tma.update_layout(
            title="TMA Level 1",
            xaxis_title="% Pipeline Slots",
            height=200,
            margin=dict(l=120, r=50, t=40, b=30),
        )
        st.plotly_chart(_fig_tma, use_container_width=True)
    else:
        st.info("EMON CSV loaded but no TMA L1 metrics found.")
else:
    st.info("No EMON data. EMON TMA requires the separate Intel-only agentsysperf-emon plugin.")

# ─── Concurrency-Scaling Sweep ─────────────────────────────────────────────

st.markdown("---")
st.subheader("Concurrency Sweep — agents/vCPU saturation knee")
st.caption(
    "Normalizes density by vCPU and finds the saturation knee + bottleneck via "
    "the ScalingAnalyzer (SQLite sweep_points). Distinct from the demo app's "
    "**Parallel Agent Density Study** (absolute agent count × NUMA × phase mix)."
)

from src.storage.sqlite_store import SQLiteResultStore
from src.dashboard.scaling_views import (
    throughput_knee_figure,
    cpu_runqueue_figure,
    per_task_profile_figure,
    BOTTLENECK_STYLE,
)
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

# Find the most recent sweep. Sweeps now write to the canonical store
# (ResultStore.open()); legacy per-dir /tmp DBs are a fallback. ALWAYS open
# read_only — a viewer must never run migrations or hold the WAL write lock
# (which would crash/contend with a live sweep). query_sweeps() is ordered
# newest-first.
_sweep_store = None
_sweep_id = None
_sweep_meta = {}


def _try_sweep_store(store):
    """Return (store, sweep_id, meta) if it has a sweep, else None."""
    try:
        sweeps = store.query_sweeps()
    except Exception:
        return None
    if sweeps:
        return store, sweeps[0]["sweep_id"], sweeps[0]
    return None

# 1. Canonical store (where P2.5 sweeps land).
try:
    _hit = _try_sweep_store(SQLiteResultStore.open(read_only=True))
except Exception:
    _hit = None
# 2. Legacy per-dir /tmp DBs (older sweeps), read-only.
if _hit is None:
    for _cand in [Path(f"{_TMP}/agentsysperf_sweep"), Path(f"{_TMP}/agentsysperf_sweep_dry")]:
        if (_cand / "agentsysperf_results.db").exists():
            try:
                _hit = _try_sweep_store(SQLiteResultStore(_cand, read_only=True))
            except Exception:
                _hit = None
            if _hit is not None:
                break

if _hit is not None:
    _sweep_store, _sweep_id, _sweep_meta = _hit

if _sweep_store is None:
    st.info(
        "No concurrency sweep found. Run:\n\n"
        "```\nagentsysperf sweep run --dry-run\n```\n\n"
        "`--dry-run` cells are synthetic modeled points (badged as such). "
        "For measurements, run a real sweep with `--fixture <fixture.jsonl>`. "
        "Then reload."
    )
else:
    _points = _sweep_store.query_sweep_points(_sweep_id)
    _verdicts = _sweep_store.query_verdicts(_sweep_id, analyzer_name="scaling")
    _verdict = _verdicts[0] if _verdicts else None
    _evidence = _verdict["evidence"] if _verdict else {}
    _knee = _evidence.get("knee")
    _bottleneck = _evidence.get("bottleneck")

    st.caption(
        f"Sweep `{_sweep_id}` · {_sweep_meta.get('hardware_sku','?')} · "
        f"basis {_sweep_meta.get('vcpu_basis','?')} {_sweep_meta.get('vcpu_basis_kind','')} · "
        f"NUMA {_sweep_meta.get('numa_policy','?')} · "
        f"LLM {('replay: ' + str(_sweep_meta.get('replay_fixture'))) if _sweep_meta.get('replay_fixture') else 'off'}"
    )

    # Verdict summary cards.
    if _verdict:
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            st.metric("Verdict", _verdict["verdict"])
        with _c2:
            _color, _label = BOTTLENECK_STYLE.get(_bottleneck or "", ("#616161", _bottleneck or "—"))
            st.metric("Bottleneck", _label)
        with _c3:
            st.metric("Confidence", f"{_verdict['confidence']:.0%}")
        if _knee:
            st.markdown(
                f"**Knee** at density **{_knee['density']:g}** "
                f"(~{_knee.get('concurrency','?')} concurrent agents), "
                f"throughput {_knee.get('throughput_at_knee','?')} trials/min, "
                f"p95 {_knee.get('p95_at_knee','?')}s."
            )

    # Aggregate scaling curves.
    if _points:
        st.plotly_chart(
            throughput_knee_figure(_points, knee=_knee, bottleneck=_bottleneck),
            use_container_width=True,
        )
        st.plotly_chart(cpu_runqueue_figure(_points), use_container_width=True)

    # Per-task profile (e.g. mteb-retrieve).
    _per_task = _evidence.get("per_task") or {}
    if _per_task:
        st.markdown("#### Per-Task Profile")
        _task = st.selectbox("Task", sorted(_per_task.keys()))
        _t = _per_task[_task]
        st.plotly_chart(
            per_task_profile_figure(
                _task, _t.get("curve", []),
                knee_density=_t.get("knee_density"),
                bottleneck=_t.get("bottleneck"),
            ),
            use_container_width=True,
        )
        _tcolor, _tlabel = BOTTLENECK_STYLE.get(_t.get("bottleneck") or "", ("#616161", _t.get("bottleneck") or "—"))
        st.markdown(
            f"**{_task}**: bottleneck **{_tlabel}**"
            + (f", knee at density {_t['knee_density']:g}." if _t.get("knee_density") else " (no knee within swept range).")
        )

    # Recommendations.
    if _verdict and _verdict.get("recommendations"):
        with st.expander("Scaling Recommendations", expanded=True):
            for _rec in _verdict["recommendations"]:
                st.markdown(f"- {_rec}")

    with st.expander("Sweep points (raw)"):
        st.dataframe(_points, use_container_width=True)

# ─── Raw Data ────────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("Raw Data Table"):
    st.dataframe(latest_results, use_container_width=True)
