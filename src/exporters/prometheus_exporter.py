#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Prometheus Exporter for AgentSysPerf
==================================

Exposes AgentSysPerf benchmark measurements as Prometheus metrics via an
HTTP endpoint. Supports two modes:

1. **Push mode** — push metrics to a Prometheus Pushgateway after each
   benchmark run (batch jobs).
2. **Server mode** — run an HTTP server that Prometheus scrapes on
   /metrics (long-running or continuous benchmarking).

Metric naming follows Prometheus conventions:
- agentsysperf_task_duration_seconds
- agentsysperf_task_cpu_time_seconds
- agentsysperf_task_cpu_utilization_percent
- agentsysperf_task_memory_rss_bytes
- agentsysperf_task_cache_miss_percent
- agentsysperf_task_ipc
- agentsysperf_tma_frontend_bound_ratio
- agentsysperf_tma_backend_bound_ratio
- agentsysperf_tma_retiring_ratio
- agentsysperf_tma_bad_speculation_ratio

Labels: run_id, task_id, workload, category, node_id
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.protocols import MeasurementRecord
from src.safe_url import require_http_url

logger = logging.getLogger(__name__)


# ─── Prometheus Text Format Builder ─────────────────────────────────────

def _sanitize_metric_name(name: str) -> str:
    """Convert metric name to Prometheus-safe format."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_").lower()


def _format_labels(labels: Dict[str, str]) -> str:
    """Format labels as Prometheus label string."""
    if not labels:
        return ""
    pairs = [f'{k}="{v}"' for k, v in sorted(labels.items()) if v]
    return "{" + ",".join(pairs) + "}"


def _format_metric(name: str, value: float, labels: Dict[str, str],
                   help_text: str = "", metric_type: str = "gauge") -> str:
    """Format a single metric in Prometheus exposition format."""
    lines = []
    if help_text:
        lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")
    lines.append(f"{name}{_format_labels(labels)} {value}")
    return "\n".join(lines)


def _dedupe_help_type(text: str) -> str:
    """Keep only the first ``# HELP``/``# TYPE`` line per metric name.

    The Prometheus exposition format requires that each metric's ``# HELP``
    and ``# TYPE`` lines appear at most once. ``_format_metric`` emits them
    per sample, so a run with N tasks produces N copies per metric name,
    which the Pushgateway's strict parser rejects with HTTP 400. Strip the
    redundant copies, preserving sample order.
    """
    seen_help: set = set()
    seen_type: set = set()
    out: List[str] = []
    for line in text.split("\n"):
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            # "# HELP <name> ..." / "# TYPE <name> ..." -> name is field 2
            parts = line.split(" ", 3)
            name = parts[2] if len(parts) > 2 else ""
            seen = seen_help if line.startswith("# HELP ") else seen_type
            if name in seen:
                continue
            seen.add(name)
        out.append(line)
    return "\n".join(out)


# ─── Prometheus Exporter ─────────────────────────────────────────────────

class PrometheusExporter:
    """Export AgentSysPerf measurements as Prometheus metrics.

    Parameters
    ----------
    mode : str, default="push"
        Export mode: "push" (Pushgateway) or "server" (HTTP /metrics endpoint).
    pushgateway_url : str, optional
        Pushgateway URL for push mode. Default: http://localhost:9091
    server_port : int, default=9101
        Port for the /metrics HTTP server in server mode.
    server_host : str, default="0.0.0.0"
        Bind address for the /metrics HTTP server. All interfaces by default
        because the entire point of server mode is to be scraped by a
        Prometheus running somewhere else — a loopback default would make the
        feature inert for its intended use. Pass "127.0.0.1" when the scraper
        is local (or when running on an untrusted network) to keep the endpoint
        off the wire. Note what is on this endpoint if you do expose it:
        hardware counters, model names and task ids, no credentials.
    job_name : str, default="agentsysperf"
        Prometheus job name label.
    extra_labels : dict, optional
        Additional labels to attach to all metrics (e.g., platform, sku).

    Examples
    --------
    Push mode (after benchmark run):

    >>> exporter = PrometheusExporter(mode="push", pushgateway_url="http://pushgw:9091")
    >>> exporter.export_run(run_id="run-abc", records=measurement_records)

    Server mode (continuous scraping):

    >>> exporter = PrometheusExporter(mode="server", server_port=9101)
    >>> exporter.start()
    >>> # ... run benchmarks, call exporter.update(records) ...
    >>> exporter.stop()
    """

    def __init__(
        self,
        *,
        mode: str = "push",
        pushgateway_url: str = "http://localhost:9091",
        server_port: int = 9101,
        # see server_host in the docstring
        server_host: str = "0.0.0.0",  # nosec B104
        job_name: str = "agentsysperf",
        extra_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._mode = mode
        self._pushgateway_url = pushgateway_url.rstrip("/")
        self._server_port = server_port
        self._server_host = server_host
        self._job_name = job_name
        self._extra_labels = extra_labels or {}
        self._metrics_text = ""
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    # ─── Public API ──────────────────────────────────────────────────

    def export_run(
        self,
        *,
        run_id: str,
        records: Sequence[MeasurementRecord],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Export a complete benchmark run's measurements.

        Parameters
        ----------
        run_id : str
            Unique identifier for this benchmark run.
        records : list of MeasurementRecord
            All measurement records from the run.
        metadata : dict, optional
            Run metadata (platform, timestamp, etc.).
        """
        self._metrics_text = self._build_metrics_text(run_id, records, metadata)

        if self._mode == "push":
            self._push_to_gateway(run_id)
        elif self._mode == "server":
            # Metrics will be served on next scrape
            logger.info(f"Updated metrics for run {run_id} ({len(records)} records)")

    def export_from_file(self, results_file: Path, run_id: str = "unknown") -> None:
        """Export measurements from a saved measurement_records.json file.

        Parameters
        ----------
        results_file : Path
            Path to measurement_records.json from an AgentSysPerf run.
        run_id : str
            Run ID to use in labels.
        """
        with open(results_file) as f:
            raw_records = json.load(f)

        records = [
            MeasurementRecord(
                span_id=r["span_id"],
                layer=r["layer"],
                payload=r["payload"],
            )
            for r in raw_records
        ]

        self.export_run(run_id=run_id, records=records)

    def export_run_from_store(self, store: Any, run_id: str) -> None:
        """Export a run's metrics by reading from a ResultStore (P7).

        The store-backed sibling of :meth:`export_from_file`. It rebuilds the
        SAME exposition text by funnelling through :meth:`export_run` /
        :meth:`export_phase_metrics` — identical metric names, label keys, and
        the span_id→task_id / node_id→workload / kind derivation — so a run
        pushed from the store is byte-for-byte the same as one pushed from JSON.
        Hardware/TMA come from ``query_measurements``; the per-phase rollup +
        per-step tokens/cost come from ``query_verdicts('breakdown')`` +
        ``query_spans`` (the same data the sidecar-DB path used).
        """
        rows = store.query_measurements(run_id)
        records = [
            MeasurementRecord(span_id=r["span_id"], layer=r["layer"], payload=r["payload"])
            for r in rows
        ]
        self.export_run(run_id=run_id, records=records)

        # Phase metrics (best-effort, same as the legacy sidecar path).
        try:
            verdicts = store.query_verdicts(run_id, analyzer_name="breakdown")
            spans = store.query_spans(run_id)
            if verdicts or spans:
                self.export_phase_metrics(
                    run_id=run_id, breakdown_verdicts=verdicts, spans=spans,
                )
        except Exception:  # phase metrics are best-effort
            logger.warning("export_run_from_store: phase metrics skipped for %s",
                          run_id, exc_info=True)

    def export_phase_metrics(
        self,
        *,
        run_id: str,
        breakdown_verdicts: Sequence[Mapping[str, Any]],
        spans: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Append per-phase metrics (from SQLite) and re-push.

        Hardware/time live in measurement_records (already exported via
        export_run); this adds the *categorical* phase rollup that lives in the
        SQLite analyzer_verdicts + spans tables:
          - time breakdown per phase (orchestration/execution/inference)
          - execution-phase hardware (instruction-weighted IPC/cache/branch)
          - per-task token + cost totals (from llm_call spans)

        Call AFTER export_run/export_from_file so the hardware text is present;
        this extends self._metrics_text and re-pushes the combined payload.
        """
        lines = [self._metrics_text.rstrip("\n")] if self._metrics_text else []

        for v in breakdown_verdicts:
            if v.get("analyzer_name") != "breakdown":
                continue
            task_id = v.get("task_id", "unknown")
            ev = v.get("evidence", {}) or {}
            base = {"run_id": run_id, "task_id": task_id, **self._extra_labels}

            # Time fraction per phase.
            for phase in ("inference", "execution", "orchestration"):
                pct = ev.get(f"{phase}_pct")
                if pct is not None:
                    lines.append(_format_metric(
                        "agentsysperf_phase_time_percent", float(pct),
                        {**base, "phase": phase},
                        help_text="Wall-clock time share of this agent phase"))

            # Execution-phase hardware (instruction-weighted). Inference HW is
            # deliberately absent (remote network wait — see breakdown analyzer).
            hw = ev.get("execution_hw") or {}
            for key, metric in (
                ("ipc", "agentsysperf_phase_execution_ipc"),
                ("cache_miss_pct", "agentsysperf_phase_execution_cache_miss_percent"),
                ("branch_miss_pct", "agentsysperf_phase_execution_branch_miss_percent"),
            ):
                if key in hw:
                    lines.append(_format_metric(
                        metric, float(hw[key]),
                        {**base, "phase": "execution"},
                        help_text=f"Execution-phase {key} (instruction-weighted)"))

        # Per-task token + cost totals from llm_call spans.
        tokens_in: Dict[str, int] = {}
        tokens_out: Dict[str, int] = {}
        cost: Dict[str, float] = {}
        for s in spans:
            if s.get("span_kind") != "llm_call":
                continue
            tid = s.get("task_id", "unknown")
            tokens_in[tid] = tokens_in.get(tid, 0) + int(s.get("tokens_in", 0) or 0)
            tokens_out[tid] = tokens_out.get(tid, 0) + int(s.get("tokens_out", 0) or 0)
            cost[tid] = cost.get(tid, 0.0) + float(s.get("cost_usd", 0.0) or 0.0)
        for tid in cost:
            base = {"run_id": run_id, "task_id": tid, "phase": "inference",
                    **self._extra_labels}
            lines.append(_format_metric(
                "agentsysperf_phase_tokens_total", tokens_in[tid],
                {**base, "direction": "in"}, metric_type="counter",
                help_text="Prompt tokens for the inference phase"))
            lines.append(_format_metric(
                "agentsysperf_phase_tokens_total", tokens_out[tid],
                {**base, "direction": "out"}, metric_type="counter",
                help_text="Completion tokens for the inference phase"))
            lines.append(_format_metric(
                "agentsysperf_phase_cost_usd", round(cost[tid], 6), base,
                metric_type="counter",
                help_text="Inference-phase cost in USD"))

        self._metrics_text = _dedupe_help_type("\n\n".join(lines) + "\n")
        if self._mode == "push":
            self._push_to_gateway(run_id)

    def start(self) -> None:
        """Start the HTTP metrics server (server mode only)."""
        if self._mode != "server":
            logger.warning("start() only applies to server mode")
            return

        handler = self._make_handler()
        self._server = HTTPServer((self._server_host, self._server_port), handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        logger.info(
            f"Prometheus metrics server running on "
            f"{self._server_host}:{self._server_port}/metrics"
        )

    def stop(self) -> None:
        """Stop the HTTP metrics server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("Prometheus metrics server stopped")

    def get_metrics_text(self) -> str:
        """Return current metrics in Prometheus exposition format."""
        return self._metrics_text

    # ─── Metrics Building ────────────────────────────────────────────

    def _build_metrics_text(
        self,
        run_id: str,
        records: Sequence[MeasurementRecord],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convert MeasurementRecords to Prometheus exposition format."""
        metric_lines = []

        for record in records:
            base_labels = {
                "run_id": run_id,
                "task_id": record.span_id.split("::")[-1] if "::" in record.span_id else record.span_id,
                "layer": record.layer,
                **self._extra_labels,
            }

            # Add workload/node labels from payload
            if "node_id" in record.payload:
                base_labels["workload"] = record.payload["node_id"]
            if "kind" in record.payload:
                base_labels["kind"] = record.payload["kind"]

            # Export L1 metrics
            if record.layer == "l1":
                metric_lines.extend(self._export_l1(record.payload, base_labels))

            # Export L3 metrics
            elif record.layer == "l3":
                metric_lines.extend(self._export_l3(record.payload, base_labels))

            # Export PerfSpect metrics
            elif record.layer == "perfspect":
                metric_lines.extend(self._export_perfspect(record.payload, base_labels))

        # Add run metadata as info metric
        if metadata:
            info_labels = {**base_labels, **{k: str(v) for k, v in metadata.items()}}
            metric_lines.append(
                _format_metric(
                    "agentsysperf_run_info", 1.0, info_labels,
                    help_text="AgentSysPerf run metadata"
                )
            )

        return _dedupe_help_type("\n\n".join(metric_lines) + "\n")

    def _export_l1(self, payload: Mapping[str, Any], labels: Dict[str, str]) -> List[str]:
        """Export L1 (resource usage) metrics."""
        metrics = []

        if "duration_us" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_duration_seconds",
                payload["duration_us"] / 1_000_000,
                labels,
                help_text="Task wall-clock duration in seconds",
            ))

        if "cpu_time_s" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_cpu_time_seconds",
                payload["cpu_time_s"],
                labels,
                help_text="Task CPU time consumed in seconds",
            ))

        if "cpu_pct_mean" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_cpu_utilization_percent",
                payload["cpu_pct_mean"],
                labels,
                help_text="Mean CPU utilization percentage",
            ))

        if "cpu_pct_peak" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_cpu_utilization_peak_percent",
                payload["cpu_pct_peak"],
                labels,
                help_text="Peak CPU utilization percentage",
            ))

        if "rss_kb_peak" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_memory_rss_bytes",
                payload["rss_kb_peak"] * 1024,
                labels,
                help_text="Peak resident set size in bytes",
            ))

        if "num_threads_peak" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_threads_peak",
                payload["num_threads_peak"],
                labels,
                help_text="Peak thread count during task",
            ))

        return metrics

    def _export_l3(self, payload: Mapping[str, Any], labels: Dict[str, str]) -> List[str]:
        """Export L3 (hardware counter) metrics."""
        metrics = []

        if "ipc" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_ipc",
                payload["ipc"],
                labels,
                help_text="Instructions per cycle",
            ))

        if "cache_miss_pct" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_cache_miss_percent",
                payload["cache_miss_pct"],
                labels,
                help_text="LLC cache miss percentage",
            ))

        if "branch_miss_pct" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_branch_miss_percent",
                payload["branch_miss_pct"],
                labels,
                help_text="Branch misprediction percentage",
            ))

        if "instructions" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_instructions_total",
                payload["instructions"],
                labels,
                help_text="Total instructions retired",
                metric_type="counter",
            ))

        if "cycles" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_cycles_total",
                payload["cycles"],
                labels,
                help_text="Total CPU cycles",
                metric_type="counter",
            ))

        return metrics

    def _export_perfspect(self, payload: Mapping[str, Any], labels: Dict[str, str]) -> List[str]:
        """Export PerfSpect (TMA) metrics."""
        metrics = []

        # TMA L1 buckets
        tma_metrics = {
            "frontend_bound": ("agentsysperf_tma_frontend_bound_ratio", "TMA Frontend Bound ratio"),
            "bad_speculation": ("agentsysperf_tma_bad_speculation_ratio", "TMA Bad Speculation ratio"),
            "backend_bound": ("agentsysperf_tma_backend_bound_ratio", "TMA Backend Bound ratio"),
            "retiring": ("agentsysperf_tma_retiring_ratio", "TMA Retiring ratio"),
        }

        for key, (metric_name, help_text) in tma_metrics.items():
            if key in payload:
                value = payload[key]
                # Normalize to 0-1 ratio if given as percentage
                if value > 1.0:
                    value = value / 100.0
                metrics.append(_format_metric(metric_name, value, labels, help_text=help_text))

        # Memory bandwidth
        if "memory_bandwidth_gbs" in payload:
            metrics.append(_format_metric(
                "agentsysperf_memory_bandwidth_bytes_per_second",
                payload["memory_bandwidth_gbs"] * 1e9,
                labels,
                help_text="Memory bandwidth in bytes per second",
            ))

        # CPI
        if "cpi" in payload:
            metrics.append(_format_metric(
                "agentsysperf_task_cpi",
                payload["cpi"],
                labels,
                help_text="Cycles per instruction",
            ))

        # Classification as info metric
        if "tma_classification" in payload:
            class_labels = {**labels, "classification": payload["tma_classification"]}
            metrics.append(_format_metric(
                "agentsysperf_tma_classification_info",
                1.0,
                class_labels,
                help_text="TMA workload classification",
            ))

        return metrics

    # ─── Push Mode ───────────────────────────────────────────────────

    def _push_to_gateway(self, run_id: str) -> None:
        """Push metrics to Prometheus Pushgateway."""
        url = f"{self._pushgateway_url}/metrics/job/{self._job_name}/run_id/{run_id}"

        try:
            req = Request(
                require_http_url(url),
                data=self._metrics_text.encode("utf-8"),
                method="POST",
            )
            req.add_header("Content-Type", "text/plain; charset=utf-8")
            # scheme gated by require_http_url
            urlopen(req, timeout=10)  # nosec B310
            logger.info(f"Pushed metrics to Pushgateway: {url}")
        except URLError as e:
            logger.error(f"Failed to push to Pushgateway ({url}): {e}")
        except Exception as e:
            logger.error(f"Pushgateway error: {e}")

    # ─── Server Mode ─────────────────────────────────────────────────

    def _make_handler(self):
        """Create HTTP request handler that serves /metrics."""
        exporter = self

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    body = exporter.get_metrics_text().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress default logging

        return MetricsHandler


__all__ = ["PrometheusExporter"]
