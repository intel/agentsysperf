#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""TB2 run for the Grafana dashboard: named light/medium tasks, L1+L3 measured,
per-task wall-clock cap so nothing runs away (last 'first-N' run hit 100+ min).

Selects tasks BY NAME (not first-N), runs each under a hard timeout in a worker
thread, captures L1 (duration/CPU/RSS) + L3 (IPC/cache/branch), stores to SQLite,
runs analyzers, and writes measurement_records.json for the Prometheus exporter.

    export OPENAI_API_KEY=...   # never written to disk by this script
    poetry run python scripts/run_tb2_dashboard_demo.py
"""
from __future__ import annotations

import os
import sys
import time
import concurrent.futures as cf
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.agent_loops.litellm_invoker import LiteLLMAgentInvoker
from src.benchmarks.terminal_bench import TerminalBenchAdapter
from src.benchmarks.terminal_bench.harbor_loader import load_tasks_from_harbor_registry
from src.protocols import discover_measurements, discover_analyzers
from src.runner import RunContext, track_span
from src.storage.sqlite_store import SQLiteResultStore

# 5 representative tasks across CPU profiles; includes a RAG task (mteb-retrieve).
# Harbor filters on the fully-qualified name (org/name), so prefix all entries.
TASK_NAMES = [
    "terminal-bench/modernize-scientific-stack",   # light  — python tooling
    "terminal-bench/log-summary-date-ranges",      # light  — text / I-O
    "terminal-bench/count-dataset-tokens",         # medium — ML data throughput
    "terminal-bench/largest-eigenval",             # medium — numeric compute (high-IPC contrast)
    "terminal-bench/mteb-retrieve",                # medium — RAG (embeddings + retrieval)
]

MODEL = "gpt-4o-mini"
MAX_TURNS = 30
PER_TASK_TIMEOUT_S = 420          # hard cap per task (7 min) — kills runaways
RUN_ID = os.environ.get("TB2_RUN_ID", "dashboard_demo")
OUTPUT_DIR = REPO / "docs" / os.environ.get("TB2_OUTPUT_SUBDIR", "tb2_dashboard_demo")

# Optional comma-separated short-name override (e.g. for a quick 2-task
# validation): TB2_TASKS=modernize-scientific-stack,log-summary-date-ranges
_override = os.environ.get("TB2_TASKS")
if _override:
    TASK_NAMES = [f"terminal-bench/{n.strip().split('/')[-1]}"
                  for n in _override.split(",") if n.strip()]


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== TB2 Dashboard Demo ===\nRun: {RUN_ID}  Model: {MODEL}  "
          f"max_turns={MAX_TURNS}  per-task cap={PER_TASK_TIMEOUT_S}s")
    print("Tasks:", ", ".join(TASK_NAMES), "\n")

    print("1. Loading named tasks from Harbor...")
    tasks = load_tasks_from_harbor_registry(task_names=TASK_NAMES)
    got = [t.task_id.split("/")[-1] for t in tasks]
    print(f"   resolved {len(got)}: {got}")
    missing = set(TASK_NAMES) - set(got)
    if missing:
        print(f"   WARNING: not found in registry, skipping: {sorted(missing)}")
    if not tasks:
        print("ERROR: no tasks resolved", file=sys.stderr)
        return 1

    adapter = TerminalBenchAdapter(dataset_loader=lambda: tasks)
    invoker = LiteLLMAgentInvoker(model=MODEL, max_turns=MAX_TURNS, temperature=0.0)
    measurements = discover_measurements()
    analyzers = discover_analyzers()
    store = SQLiteResultStore(output_dir=OUTPUT_DIR)
    store.store_run_metadata(run_id=RUN_ID, metadata={
        "start_time": int(time.time()),
        "hardware_sku": "Intel Xeon Platinum 8592+",
        "model": MODEL,
    })
    print(f"   measurements={list(measurements)}  analyzers={list(analyzers)}\n")

    print("2. Running (each task isolated, capped, records saved per-task)...\n")
    # Pass RUN_ID so the RunContext, SQLite store, Prometheus labels, AND
    # Langfuse traces all share ONE run_id — required for Phase-3 correlation
    # (otherwise RunContext auto-generates a different id than RUN_ID).
    ctx = RunContext(
        measurements=list(measurements.values()),
        output_dir=OUTPUT_DIR,
        run_id=RUN_ID,
    )
    specs = list(adapter.list_tasks())
    passed = 0

    # One persistent executor for the whole run. We deliberately do NOT use a
    # per-task `with ThreadPoolExecutor(...)` block: its __exit__ calls
    # shutdown(wait=True), which would block on a runaway thread and defeat the
    # per-task timeout. We let a timed-out worker keep running orphaned and move
    # on; shutdown(wait=False) at the end avoids blocking on it.
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        with ctx:
            for spec in specs:
                short = spec.id.split("/")[-1]
                print(f"   ▶ {short}")
                t0 = time.time()
                result = None
                # track_span always opens/closes so L1+L3 are captured even if
                # the task body times out or raises — the span still finalizes.
                with track_span(ctx, spec.id, kind="terminal_bench", node_id=short):
                    fut = ex.submit(adapter.run_task, spec,
                                    agent_invoker=invoker, run_context=ctx)
                    try:
                        result = fut.result(timeout=PER_TASK_TIMEOUT_S)
                    except cf.TimeoutError:
                        print(f"     ⏱ TIMEOUT after {PER_TASK_TIMEOUT_S}s — span still captured")
                    except Exception as e:  # ERROR ISOLATION: one bad task ≠ dead run
                        print(f"     ⚠ ERROR ({type(e).__name__}: {e}) — span still captured")
                dur = time.time() - t0
                ok = bool(result and result.passed)
                passed += ok
                print(f"     {'✓' if ok else '✗'} {dur:.1f}s")
                store.store_task_result(run_id=RUN_ID,
                                        task_id=(result.task_id if result else spec.id),
                                        result={"passed": ok, "duration_s": dur,
                                                "workload_type": "light_medium"})
                # INCREMENTAL STORAGE: persist the records JSON after every task,
                # so a crash on a later task never discards earlier tasks' data.
                ctx._serialize_records()
                print(f"     saved {len(ctx.records)} records so far")
    finally:
        ex.shutdown(wait=False)  # never block on an orphaned runaway worker

    print(f"\n3. Storing & analyzing ({len(ctx.records)} records)...")
    store.store_measurements(run_id=RUN_ID, records=ctx.records)
    # Persist step-level execution trace (StepTrace rows: per-turn LLM + tool
    # events with tokens/cost/exit status).
    steps = ctx.step_traces
    store.store_spans(run_id=RUN_ID, spans=steps)
    print(f"   stored {len(steps)} step-trace rows")
    analysis = []
    for name, an in analyzers.items():
        try:
            analysis.extend(an.analyze(ctx.records))
        except Exception as e:
            print(f"   WARNING: {name} failed: {e}")
    store.store_analysis_results(run_id=RUN_ID, results=analysis)

    # layer coverage — so we know which dashboard panels will populate
    layers = {}
    for r in ctx.records:
        layers[r.layer] = layers.get(r.layer, 0) + 1
    print(f"\n=== Summary ===")
    print(f"Passed: {passed}/{len(specs)}")
    print(f"Records by layer: {layers}")
    print(f"Verdicts: {len(analysis)}")
    print(f"DB: {OUTPUT_DIR}/agentsysperf_results.db")
    print(f"Records JSON: {OUTPUT_DIR}/measurement_records.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
