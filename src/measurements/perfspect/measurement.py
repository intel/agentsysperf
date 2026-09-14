#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
PerfSpect Measurement Plugin for AgentSysPerf
============================================

Integrates Intel PerfSpect metrics collection into AgentSysPerf's measurement
framework. Provides TMA (Top-down Microarchitecture Analysis) breakdown,
cache metrics, memory bandwidth, and power data per span.

References:
- Intel PerfSpect: https://github.com/intel/PerfSpect
- Intel Top-Down Methodology: https://www.intel.com/content/www/us/en/developer/articles/technical/top-down-microarchitecture-analysis.html
- Yasin, "A Top-Down method for performance analysis and counters architecture", ISPASS 2014

PerfSpect advantages over raw `perf stat`:
- Pre-built TMA metric formulas per microarchitecture
- Compact core + uncore metrics in one collection
- CSV output for easy parsing
- Works without SEP driver (unlike emon)

Prerequisites:
- PerfSpect binary available (~/perfspect/perfspect or on PATH)
- kernel.perf_event_paranoid <= 1
- Linux perf subsystem available
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.protocols import MeasurementRecord

logger = logging.getLogger(__name__)

# Rough seconds PerfSpect spends on configure/prepare/metadata before its first
# collection interval starts. A property of the tool, not of the span, so no
# interval setting tunes it away.
#
# MEASURED, not assumed (perfspect 3.17.0, Xeon Platinum 8592+/EMR, 2026-08-11): launch with
# `--duration 3600`, SIGTERM after N seconds, check for *_metrics_summary.csv.
#   --interval 1: N=7 -> no summary, N=8 -> summary
#   --interval 5: N=11 -> no summary, N=12 -> summary
# Both boundaries give init = floor - interval = 7s, reproduced on repeat runs.
#
# ESTIMATE, used only to word a diagnostic — never to decide whether to keep a
# span. It is not portable: through this plugin's code path the same host needed
# more than 9.5s at --interval 1 where the shell harness needed only 8s, and it
# will shift with core count, sudo-elevation latency and microarchitecture. An
# earlier revision claimed 4s here and hardcoded a `duration_s < 10` gate; that
# number was both wrong and load-bearing, which is the combination to avoid.
_INIT_OVERHEAD_S = 7.0

# PerfSpect's default collection interval (`perfspect metrics --interval`,
# "event collection interval in seconds (default: 5)"). Kept as the default here
# to preserve existing accuracy; lowering it lowers the span floor (12s -> 8s at
# 1s) and still yields memory-bandwidth and TMA metrics in the summary.
_DEFAULT_INTERVAL_S = 5


def _norm_kind(kind: str) -> str:
    """Normalize a span kind so "terminal-bench" and "terminal_bench" match."""
    return kind.replace("-", "_")


def _needs_noroot() -> bool:
    """True when PerfSpect must be told not to elevate.

    PerfSpect defaults to re-exec'ing itself under sudo. On a host where the
    user is not root and sudo needs a password, that fails non-interactively
    with "failed to elevate privileges on local target / no targets remain"
    and NO metric files are written. `--noroot` collects core metrics only
    (no uncore), so only pass it when elevation would actually fail.
    """
    if os.geteuid() == 0:
        return False
    try:
        return subprocess.run(
            ["sudo", "-n", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).returncode != 0
    except (OSError, subprocess.SubprocessError):
        return True


def _pmu_corrupted(metrics: Mapping[str, Any]) -> bool:
    """True when the parsed metrics carry the PMU-contention fingerprint.

    EMON reserves the PMU through the SEP driver. A PerfSpect collection that
    overlaps one still exits cleanly and writes a full summary CSV, but perf
    hands back identical counts for cycles and instructions, so IPC and CPI
    both come out as exactly 1.0 (stddev 0) and CPU utilization reads in the
    tens of millions of percent. Neither tool reports an error, so this has to
    be caught on the values themselves.
    """
    # CPU utilization is a percentage; anything past 100% per-CPU is impossible
    # and the contended reading lands in the tens of millions.
    util = metrics.get("cpu_utilization_pct")
    if isinstance(util, (int, float)) and util > 100.0:
        return True

    # cycles == instructions exactly, to six decimal places, in both
    # directions. A real workload does not sit on 1.000000/1.000000.
    ipc, cpi = metrics.get("ipc"), metrics.get("cpi")
    return ipc == 1.0 and cpi == 1.0


class PerfSpectMeasurement:
    """Measurement plugin that collects PerfSpect metrics per span.

    Collects TMA L1/L2 breakdown, CPI/IPC, cache hit rates, memory
    bandwidth, and power metrics using PerfSpect's `metrics` subcommand.

    Parameters
    ----------
    perfspect_path : str, optional
        Path to perfspect binary. Auto-detected if not specified.
    scope : str, default="system"
        Collection scope: "system", "process", or "cgroup".
    target_pids : list of int, optional
        PIDs to monitor (only with scope="process").
    categories : list of str, optional
        Metric categories to collect. Default: all available.
    interval_s : int, default=5
        PerfSpect collection interval, in seconds. Together with PerfSpect's
        ~7s startup this sets the minimum measurable span (~12s at the default,
        ~8s at ``interval_s=1``); shorter spans emit no record at all. No
        setting brings the floor below ~8s.
    """

    layer = "perfspect"

    def __init__(
        self,
        *,
        perfspect_path: Optional[str] = None,
        scope: str = "system",
        target_pids: Optional[List[int]] = None,
        categories: Optional[List[str]] = None,
        max_duration_s: int = 3600,
        interval_s: int = _DEFAULT_INTERVAL_S,
        collect_kinds: Optional[List[str]] = None,
        noroot: Optional[bool] = None,
    ) -> None:
        self._perfspect_path = perfspect_path or self._find_perfspect()
        self._scope = scope
        self._target_pids = target_pids
        self._categories = categories
        # Derive the minimum measurable span from its two causes instead of
        # hardcoding it, so the number explains itself and tracks the interval.
        self._interval_s = interval_s
        self._min_span_s = _INIT_OVERHEAD_S + interval_s
        # PerfSpect needs init + one interval per collection (see _min_span_s)
        # and spawns a heavy process per span. The agent loop emits many 0-4s
        # sub-spans (kind=inference/execution), so collecting on those wastes
        # spawns and yields nothing. Restrict to long top-level task spans.
        # Match separator-insensitively: span kinds arrive as the benchmark id
        # (run_driver passes kind=cfg.benchmark), and the CLI spells it
        # "terminal-bench" while the entry points spell it "terminal_bench".
        # synthetic_cpu is deliberately NOT in this default set: its tasks target
        # default_duration_s=3.0 and the CLI exposes no way to lengthen them, so
        # every such span falls far below the floor. Instrumenting them spawned a
        # perfspect per task, blocked for the teardown, and produced nothing.
        # Pass collect_kinds=["synthetic_cpu"] explicitly if you drive the
        # adapter from Python with a duration above the floor.
        self._collect_kinds = {
            _norm_kind(k) for k in (
                collect_kinds
                or ("terminal_bench", "task", "swe_bench", "tau_bench")
            )
        }
        # PerfSpect only writes metric files on NATURAL completion of a fixed
        # --duration run (a killed `--duration 0` run yields "no metrics
        # collected"). So we launch with a generous upper-bound duration at
        # span start and terminate-then-parse at span end; PerfSpect flushes a
        # partial summary on its own interval cadence.
        self._max_duration_s = max_duration_s
        self._noroot = _needs_noroot() if noroot is None else noroot
        if self._noroot:
            logger.info(
                "PerfSpect running with --noroot (no passwordless sudo); "
                "core metrics only, uncore/memory-bandwidth metrics unavailable"
            )
        self._output_dir: Optional[Path] = None
        self._active_spans: Dict[str, Dict[str, Any]] = {}
        self._available = self._check_available()

    # ─── Measurement Protocol ────────────────────────────────────────

    def start(self, *, run_id: str = "", output_dir: Optional[Path] = None) -> None:
        """Initialize measurement plugin.

        Signature matches the Measurement protocol (``run_id``/``output_dir``
        keyword args, as L1/L3 declare). PerfSpect collects into its own temp
        dir, so the args are accepted for protocol conformance but unused.
        """
        if not self._available:
            logger.warning(
                "PerfSpect not available; measurements will be empty. "
                "Install from: https://github.com/intel/PerfSpect/releases"
            )
            return

        self._output_dir = Path(tempfile.mkdtemp(prefix="agentsysperf_perfspect_"))
        logger.info(f"PerfSpect measurement ready (output: {self._output_dir})")

    def observe_span(self, span_id: str, kind: str = "", node_id: str = "") -> None:
        """Start PerfSpect collection for a span.

        Only top-level task spans (see ``collect_kinds``) are instrumented;
        short turn-level sub-spans (kind=inference/execution) are skipped
        because PerfSpect can't produce metrics for sub-10s windows.
        """
        if not self._available:
            return
        if _norm_kind(kind) not in self._collect_kinds:
            logger.debug("PerfSpect skipping span %s (kind=%s not in %s)",
                         span_id, kind, sorted(self._collect_kinds))
            return

        span_dir = self._output_dir / span_id.replace("::", "__")
        span_dir.mkdir(parents=True, exist_ok=True)

        # Start perfspect metrics in background
        cmd = self._build_command(span_dir)

        try:
            # IMPORTANT: do NOT use subprocess.PIPE here. PerfSpect streams a
            # progress spinner to stderr; with an unread PIPE it fills the
            # ~64KB buffer, blocks, and never flushes its metric files (the
            # span dir stays empty). Discard the streams instead.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._active_spans[span_id] = {
                "proc": proc,
                "dir": span_dir,
                "start_time": time.time(),
                "kind": kind,
                "node_id": node_id,
            }
            logger.debug(f"PerfSpect started for span {span_id} (pid={proc.pid})")
        except Exception as e:
            logger.error(f"Failed to start PerfSpect for span {span_id}: {e}")

    def finalize_span(self, span_id: str) -> List[MeasurementRecord]:
        """Stop PerfSpect collection and parse results."""
        if span_id not in self._active_spans:
            return []

        span_info = self._active_spans.pop(span_id)
        proc = span_info["proc"]
        span_dir = span_info["dir"]
        start_time = span_info["start_time"]
        duration_s = time.time() - start_time

        # Stop perfspect (SIGTERM triggers the metric-file flush).
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # Files are written *during* shutdown; poll briefly for the summary CSV
        # instead of parsing instantly (avoids a flush race).
        #
        # Deliberately NOT gated on duration_s. The span floor is not a portable
        # constant — measured on one host it moved from 8s to >9.5s just between
        # a shell harness and this code path, and it varies with core count,
        # sudo-elevation latency and microarchitecture. Discarding a span
        # because its duration looked too short would throw away real metrics
        # wherever init is faster than estimated. Whether the summary file was
        # written is ground truth, so let it decide and use duration only to
        # explain the outcome below.
        deadline = time.time() + 10
        while time.time() < deadline:
            if list(span_dir.glob("*_metrics_summary.csv")):
                break
            time.sleep(0.5)

        # Parse results
        metrics = self._parse_metrics(span_dir)

        if not metrics:
            if duration_s < self._min_span_s:
                logger.warning(
                    "PerfSpect: no metrics for span %s — it ran %.1fs, under "
                    "the ~%.0fs this host needs (~%.0fs tool init + one %ds "
                    "interval). Expected for a span this short. The perfspect "
                    "layer is UNMEASURED here, not zero: no record is emitted "
                    "and analyzers see the layer as absent. Use longer tasks, "
                    "or interval_s=1 to lower the floor.",
                    span_id, duration_s, self._min_span_s, _INIT_OVERHEAD_S,
                    self._interval_s,
                )
            else:
                logger.warning(
                    "PerfSpect: no metrics for span %s despite running %.1fs, "
                    "which clears the ~%.0fs floor — this is unexpected. Check "
                    "privilege elevation (--noroot collects no uncore metrics) "
                    "and whether another tool holds the PMU.",
                    span_id, duration_s, self._min_span_s,
                )
            return []

        if _pmu_corrupted(metrics):
            logger.warning(
                "PerfSpect span %s discarded: IPC and CPI both read exactly "
                "1.0, the fingerprint of perf returning identical values for "
                "cycles and instructions. This happens when an exclusive-PMU "
                "collector holds the counters — the two cannot collect at once. "
                "Stop the other collector to get PerfSpect metrics.",
                span_id,
            )
            return []

        # Build measurement record
        payload = {
            "kind": span_info["kind"],
            "node_id": span_info["node_id"],
            "duration_s": duration_s,
            "scope": self._scope,
            **metrics,
        }

        return [
            MeasurementRecord(
                span_id=span_id,
                layer=self.layer,
                payload=payload,
            )
        ]

    def stop(self) -> None:
        """Cleanup all active collections."""
        for span_id in list(self._active_spans.keys()):
            self.finalize_span(span_id)

        if self._output_dir and self._output_dir.exists():
            shutil.rmtree(self._output_dir, ignore_errors=True)

    # ─── Internal Helpers ────────────────────────────────────────────

    def _find_perfspect(self) -> Optional[str]:
        """Auto-detect PerfSpect binary location.

        Search order — PATH first, then the two locations PerfSpect's release
        tarball is commonly unpacked into:
        1. On PATH
        2. ~/perfspect/perfspect
        3. ~/dev/perfspect/perfspect
        """
        # Check PATH
        path_bin = shutil.which("perfspect")
        if path_bin:
            logger.debug(f"Found perfspect on PATH: {path_bin}")
            return path_bin

        # Check common locations
        home = Path.home()
        candidates = [
            Path("perfspect/perfspect"),  # Relative to project root
            Path(".venv/perfspect/perfspect"),  # Relative path to .venv
            home / "perfspect" / "perfspect",
            home / "dev" / "perfspect" / "perfspect",
            Path("/opt/intel/perfspect/perfspect"),
        ]

        logger.debug(f"Searching for perfspect binary in {len(candidates)} locations")
        for candidate in candidates:
            logger.debug(f"Checking: {candidate} (exists={candidate.exists()}, executable={candidate.exists() and os.access(candidate, os.X_OK)})")
            if candidate.exists() and os.access(candidate, os.X_OK):
                logger.info(f"Found perfspect at: {candidate}")
                return str(candidate)

        logger.warning(f"PerfSpect binary not found in any candidate location. Searched: {[str(c) for c in candidates]}")
        return None

    def _check_available(self) -> bool:
        """Check if PerfSpect can run on this system."""
        if not self._perfspect_path:
            logger.warning("PerfSpect binary not found - check search paths with --verbose")
            return False

        logger.debug(f"PerfSpect binary found at: {self._perfspect_path}")

        # Check perf_event_paranoid
        try:
            with open("/proc/sys/kernel/perf_event_paranoid") as f:
                paranoid = int(f.read().strip())
            logger.debug(f"perf_event_paranoid = {paranoid}")
            if paranoid > 1:
                logger.warning(
                    f"perf_event_paranoid={paranoid} (need <=1 for PerfSpect metrics). "
                    f"Fix with: sudo sysctl -w kernel.perf_event_paranoid=1"
                )
                return False
        except (IOError, ValueError) as e:
            logger.debug(f"Could not check perf_event_paranoid: {e}")
            pass

        logger.info(f"PerfSpect is available at: {self._perfspect_path}")
        return True

    def _build_command(self, output_dir: Path) -> List[str]:
        """Build perfspect metrics command.

        Flag semantics per ``perfspect metrics --help``
        (https://github.com/intel/PerfSpect):
        - --duration is unbounded unless set, so always set it
        - --format csv for parsing
        - --scope process + --pids for per-process attribution
        """
        cmd = [
            self._perfspect_path,
            "metrics",
            # A killed `--duration 0` run flushes NOTHING ("no metrics
            # collected"). A fixed --duration run flushes a partial summary
            # when terminated early, so use a generous upper bound and stop it
            # when the span ends.
            "--duration", str(self._max_duration_s),
            # Explicit rather than relying on the tool default, because this
            # value sets the minimum measurable span (see _min_span_s).
            "--interval", str(self._interval_s),
            "--format", "csv",
            "--output", str(output_dir),
            "--scope", self._scope,
            "--noupdate",            # skip the Intel-network update check
        ]

        if self._noroot:
            cmd.append("--noroot")

        if self._scope == "process" and self._target_pids:
            cmd.extend(["--pids", ",".join(str(p) for p in self._target_pids)])

        return cmd

    def _parse_metrics(self, span_dir: Path) -> Dict[str, Any]:
        """Parse PerfSpect's summary CSV into a metric dict.

        PerfSpect writes ``<host>_metrics_summary.csv`` with one row per metric
        and columns ``metric,mean,min,max,stddev``. We take the ``mean`` column.
        Metric names are the verbose PerfSpect labels, e.g. ``IPC``,
        ``TMA_Frontend_Bound(%)``, ``memory bandwidth total (MB/sec)``. We map
        the handful the analyzers/exporter consume to stable keys and also keep
        normalized copies of everything for debugging.
        """
        metrics: Dict[str, Any] = {}

        # Prefer the *summary* file; the per-interval file has a wide,
        # timestamped row format we don't need here.
        summary = list(span_dir.glob("*_metrics_summary.csv"))
        csv_files = summary or [
            p for p in span_dir.glob("*.csv")
            if not p.name.endswith("_metrics.csv")  # avoid the wide per-interval file
        ] or list(span_dir.glob("*.csv"))
        if not csv_files:
            stdout_file = span_dir / "stdout.log"
            if stdout_file.exists():
                return self._parse_text_output(stdout_file)
            return {}

        raw: Dict[str, float] = {}
        try:
            with open(csv_files[0]) as f:
                for row in csv.DictReader(f):
                    name = (row.get("metric") or row.get("Metric") or "").strip()
                    val = row.get("mean", row.get("value", row.get("Value", "")))
                    if not name or val in (None, ""):
                        continue
                    try:
                        parsed = float(val)
                    except (ValueError, TypeError):
                        continue
                    # PerfSpect emits "NaN" for metrics it could not compute on
                    # this part (e.g. TMA_......MEM_Bandwidth(%) on some SKUs),
                    # and float("NaN") does NOT raise. A NaN that reaches an
                    # analyzer fails every `>` comparison silently, which reads
                    # downstream as "measured and below threshold" rather than
                    # "not measured" — the same fabrication as a counter
                    # recorded as 0.0. Drop it so the key is simply absent.
                    if not math.isfinite(parsed):
                        continue
                    raw[name] = parsed
        except Exception as e:
            logger.error(f"Failed to parse PerfSpect summary CSV: {e}")
            return {}

        # Map verbose PerfSpect labels -> stable keys the rest of AgentSysPerf uses.
        # TMA buckets are reported as percentages; store as 0-1 ratios.
        label_map = {
            "TMA_Frontend_Bound(%)": ("frontend_bound", 0.01),
            "TMA_Bad_Speculation(%)": ("bad_speculation", 0.01),
            "TMA_Backend_Bound(%)": ("backend_bound", 0.01),
            # Level-2 bucket (the dots encode TMA depth). Previously unmapped,
            # which left MemoryBandwidthAnalyzer's tma_memory_bound signal
            # unreachable on EVERY platform — that was a gap in this label_map,
            # not a property of the silicon. Mapping it here is automatically
            # platform-correct because of the `if label in raw` guard below: on
            # E-core parts PerfSpect emits no TMA rows at all, so the key stays
            # absent and consumers see None from the data rather than from a
            # hardcoded constant. Verified present on this class of P-core part
            # (TMA_..Memory_Bound(%) = 16.29 on Xeon Platinum 8592+, perfspect 3.17.0).
            "TMA_..Memory_Bound(%)": ("tma_memory_bound", 0.01),
            "TMA_Retiring(%)": ("retiring", 0.01),
            "IPC": ("ipc", 1.0),
            "CPI": ("cpi", 1.0),
            "memory bandwidth total (MB/sec)": ("memory_bandwidth_gbs", 0.001),
            "branch misprediction ratio": ("branch_miss_ratio", 1.0),
        }
        for label, (key, scale) in label_map.items():
            if label in raw:
                metrics[key] = raw[label] * scale

        # Record the collection scope alongside the numbers. "memory bandwidth
        # total" is machine-wide at --scope system, and consumers that divide it
        # by a per-socket or per-node peak need to know that rather than assume.
        metrics["scope"] = self._scope

        # Keep a normalized copy of every raw metric for debugging/extra panels.
        for name, value in raw.items():
            norm = name.strip().lower().replace(" ", "_").replace("%", "pct")
            metrics.setdefault(norm, value)

        return self._derive_summary(metrics)

    def _parse_text_output(self, stdout_file: Path) -> Dict[str, Any]:
        """Fallback parser for text output."""
        metrics = {}
        try:
            content = stdout_file.read_text()
            for line in content.splitlines():
                if ":" in line and not line.startswith("#"):
                    parts = line.split(":", 1)
                    key = parts[0].strip().lower().replace(" ", "_")
                    value = parts[1].strip()
                    try:
                        metrics[key] = float(value.split()[0])
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
        return metrics

    def _derive_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Derive summary insights from raw metrics.

        The TMA level-1 buckets and their meaning are defined by the Top-Down
        Methodology (see module docstring). The cut-offs below are **this
        project's triage heuristics**, not values specified by TMA — they exist
        to label a dominant bucket for reporting, and are deliberately coarse:

        - Frontend_Bound > 20%: instruction fetch/decode bottleneck
        - Backend_Bound > 40%: memory or execution unit stalls
        - Bad_Speculation > 10%: branch mispredictions significant
        - Retiring > 70%: workload is efficient (IPC is the limit)

        Treat the ``tma_classification`` they produce as a pointer to where to
        drill down, not as a measurement. ``tma_dominant_bucket`` is threshold
        free and is the sounder field of the two.
        """
        # TMA L1 classification
        tma_keys = {
            "frontend_bound": metrics.get("frontend_bound", None),
            "bad_speculation": metrics.get("bad_speculation", None),
            "backend_bound": metrics.get("backend_bound", None),
            "retiring": metrics.get("retiring", None),
        }

        if all(v is not None for v in tma_keys.values()):
            # Determine dominant bottleneck
            dominant = max(tma_keys, key=lambda k: tma_keys[k])
            metrics["tma_dominant_bucket"] = dominant

            # TMA buckets are stored as 0-1 ratios (see _parse_metrics), so the
            # percentage thresholds documented above become ratios here.
            if tma_keys["retiring"] > 0.70:
                metrics["tma_classification"] = "efficient"
            elif tma_keys["backend_bound"] > 0.40:
                metrics["tma_classification"] = "backend_bound"
            elif tma_keys["frontend_bound"] > 0.20:
                metrics["tma_classification"] = "frontend_bound"
            elif tma_keys["bad_speculation"] > 0.10:
                metrics["tma_classification"] = "speculation_bound"
            else:
                metrics["tma_classification"] = "balanced"

        # Memory subsystem classification
        mem_bw = metrics.get("dram_bandwidth_gbs", metrics.get("mem_bandwidth_gbs"))
        if mem_bw is not None:
            metrics["memory_bandwidth_gbs"] = mem_bw

        # CPI/IPC
        ipc = metrics.get("ipc", metrics.get("instructions_per_cycle"))
        cpi = metrics.get("cpi", metrics.get("cycles_per_instruction"))
        if ipc is not None:
            metrics["ipc"] = ipc
        if cpi is not None:
            metrics["cpi"] = cpi

        return metrics


__all__ = ["PerfSpectMeasurement"]
