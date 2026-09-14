#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Push AgentSysPerf measurements to the Pushgateway and emit the Grafana dashboard.

Two outputs:
  1. POST metrics from a run's measurement_records.json to the Pushgateway.
  2. Write the Grafana dashboard JSON (raw `generate()` form, which is what
     Grafana *file provisioning* expects -- NOT save()'s API envelope) into
     monitoring/grafana/dashboards/ so the stack auto-loads it.

Usage:
    poetry run python scripts/push_to_prometheus.py \
        [--records /tmp/agentsysperf_results/measurement_records.json] \
        [--run-id simple_demo] \
        [--pushgateway http://localhost:9091]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.exporters import GrafanaDashboardGenerator, PrometheusExporter
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

DASHBOARD_DIR = REPO / "monitoring" / "grafana" / "dashboards"


def emit_dashboard() -> Path:
    """Write the dashboard in the bare form Grafana file-provisioning loads."""
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out = DASHBOARD_DIR / "agentsysperf_dashboard.json"
    dashboard = GrafanaDashboardGenerator(
        datasource="Prometheus",
        title="AgentSysPerf Benchmark Dashboard",
    ).generate()
    out.write_text(json.dumps(dashboard, indent=2))
    print(f"[dashboard] wrote {out.relative_to(REPO)} ({out.stat().st_size} bytes)")
    return out


def push_from_store(run_id: str, pushgateway: str, dsn: str | None) -> int:
    """Push a run's metrics by reading the canonical ResultStore (P7, opt-in).

    Reuses export_run_from_store, so the exposition text is byte-identical to the
    JSON path — same metric names, labels, task_id/workload derivation, and the
    phase rollup from verdicts/spans. The legacy --records JSON path stays the
    DEFAULT; this is opt-in via --from-store until P10 proves parity and flips
    start.sh.
    """
    from src.storage.sqlite_store import SQLiteResultStore
    store = SQLiteResultStore.open(dsn=dsn, read_only=True)
    exporter = PrometheusExporter(
        mode="push",
        pushgateway_url=pushgateway,
        extra_labels={"sku": "Xeon-EMR-8592+"},
    )
    exporter.export_run_from_store(store, run_id)
    store.close()
    text = exporter.get_metrics_text()
    print(f"[push] store run {run_id!r} -> {len(text.splitlines())} metric lines "
          f"({len(text)} bytes) -> {pushgateway} (job=agentsysperf)")
    return 0


def push(records_path: Path, run_id: str, pushgateway: str) -> int:
    if not records_path.exists():
        print(f"[push] ERROR: records file not found: {records_path}", file=sys.stderr)
        print("       Generate one with: poetry run python run_simple_benchmark.py", file=sys.stderr)
        return 1

    n = len(json.loads(records_path.read_text()))
    exporter = PrometheusExporter(
        mode="push",
        pushgateway_url=pushgateway,
        extra_labels={"sku": "Xeon-EMR-8592+"},
    )
    exporter.export_from_file(records_path, run_id=run_id)

    # If the run's SQLite store sits alongside the records file, also export the
    # per-phase rollup (time breakdown + execution_hw) and per-step tokens/cost
    # that live in analyzer_verdicts / spans — not in measurement_records.json.
    db_path = records_path.parent / "agentsysperf_results.db"
    phase_note = ""
    if db_path.exists():
        try:
            from src.storage.sqlite_store import SQLiteResultStore
            store = SQLiteResultStore(output_dir=db_path.parent)
            verdicts = store.query_verdicts(run_id, analyzer_name="breakdown")
            spans = store.query_spans(run_id)
            store.close()
            exporter.export_phase_metrics(
                run_id=run_id, breakdown_verdicts=verdicts, spans=spans,
            )
            phase_note = f", +phase metrics ({len(verdicts)} verdicts, {len(spans)} spans)"
        except Exception as e:  # phase metrics are best-effort
            phase_note = f", phase metrics skipped ({type(e).__name__})"

    text = exporter.get_metrics_text()
    print(f"[push] {n} records -> {len(text.splitlines())} metric lines "
          f"({len(text)} bytes) -> {pushgateway} (job=agentsysperf, run_id={run_id}){phase_note}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path,
                    default=Path(f"{_TMP}/agentsysperf_results/measurement_records.json"))
    ap.add_argument("--run-id", default="simple_demo")
    ap.add_argument("--pushgateway", default="http://localhost:9091")
    ap.add_argument("--dashboard-only", action="store_true",
                    help="Only emit the dashboard JSON, skip pushing metrics.")
    ap.add_argument("--from-store", action="store_true",
                    help="Read metrics from the canonical ResultStore instead of "
                         "the --records JSON file (opt-in; JSON is still the default).")
    ap.add_argument("--store-dsn", default=None,
                    help="Optional store DSN/URL for --from-store (else $AGENTSYSPERF_HOME).")
    args = ap.parse_args()

    emit_dashboard()
    if args.dashboard_only:
        return 0
    if args.from_store:
        return push_from_store(args.run_id, args.pushgateway, args.store_dsn)
    return push(args.records, args.run_id, args.pushgateway)


if __name__ == "__main__":
    raise SystemExit(main())
