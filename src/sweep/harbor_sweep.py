#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""HarborSweep — run a concurrency-scaling sweep on EMR via Harbor.

The MVP execution mechanism: each density cell runs N agents concurrently using
Harbor's own ``-n`` (container-per-agent) path — the same way the colleague's
AWS study ran, already container-isolated, so per-agent attribution is clean at
the container/node level. For each cell we:

  1. point the agents' ``OPENAI_API_BASE`` at a :class:`ReplayProxy` (so the LLM
     is deterministic and the wall time reflects silicon, not model variance),
  2. sample node telemetry with :class:`L1SystemMeasurement` over the cell,
  3. run ``harbor run -n {concurrency} -k {attempts}`` over the task set,
  4. parse per-trial timings from Harbor ``result.json`` files,
  5. roll the cell up into a ``sweep_points`` row.

After all cells, :class:`ScalingAnalyzer` finds the knee + bottleneck across the
sweep and the verdict is stored. Per-span hardware detail for in-container
agents is out of scope here (the native process-pool runner is the full-scope
path); this runner deliberately works at cell granularity, which is the right
level for density/throughput/knee.

Why Harbor's concurrency and not threads: AgentSysPerf's ``PsutilSampler`` reads
whole-process CPU% and would assign it to every in-process agent identically.
Containers give each agent its own process and the node telemetry captures the
aggregate — the quantity the sweep is actually about.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.analyzers.scaling import ScalingAnalyzer
from src.measurements.l1_system import L1SystemMeasurement
from src.platform.detect import detect_platform
from src.protocols import discover_measurements
from src.replay import ReplayProxy, validate_fixture
from src.runner import RunContext, track_span
from src.storage.sqlite_store import SQLiteResultStore
from src.sweep.spec import SweepSpec

logger = logging.getLogger(__name__)


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * (pct / 100.0)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _docker_bridge_gateway(default: str = "172.17.0.1") -> str:
    """Host IP reachable from inside a Docker container (the bridge gateway).

    Reads the docker0 interface address; falls back to the conventional
    172.17.0.1. This is the address a containerized agent dials to reach a
    host-bound replay proxy.
    """
    try:
        import subprocess as _sp
        out = _sp.run(["ip", "-4", "addr", "show", "docker0"],
                      capture_output=True, text=True, timeout=5).stdout
        for tok in out.split():
            if tok.count(".") == 3 and tok[0].isdigit():
                return tok.split("/")[0]
    except Exception:
        pass
    return default


class HarborSweep:
    """Drive a density sweep over Harbor + TB2, persisting to a ResultStore."""

    def __init__(self, spec: SweepSpec, *, store: Optional[SQLiteResultStore] = None):
        self.spec = spec
        self.platform = detect_platform()
        # Resolve the density basis from the platform if not set explicitly.
        if spec.vcpu_basis is None:
            spec.vcpu_basis = (
                self.platform.physical_cores
                if spec.vcpu_basis_kind == "physical_cores"
                else self.platform.logical_cpus
            ) or (os.cpu_count() or 1)
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        # Default to the single canonical store (the open() seam), NOT a per-dir
        # /tmp DB — otherwise the live dashboard (which reads the canonical store
        # via the DAL) never sees freshly-run sweeps. An explicit store= still
        # wins (tests, callers that want isolation). See P2.5 in the plan.
        self.store = store or SQLiteResultStore.open()
        self.analyzer = ScalingAnalyzer()
        # EMON collection is optional and Intel-only: it lives in the separate,
        # unreleased ``agentsysperf-emon`` plugin, discovered here rather than
        # imported. When spec.emon is set but the plugin is not installed, the
        # sweep runs without EMON (warn-once) instead of failing.
        self._emon = discover_measurements().get("emon") if spec.emon else None
        if spec.emon and self._emon is None:
            logger.warning(
                "EMON collection requested (spec.emon=True) but the 'emon' "
                "measurement plugin is not installed — install the Intel-only "
                "plugin `agentsysperf-emon` to collect EDP during a sweep. "
                "Running this sweep WITHOUT EMON."
            )

    # ── Top-level ────────────────────────────────────────────────────

    def run(self, *, sweep_id: Optional[str] = None, dry_run: bool = False) -> str:
        spec = self.spec
        sweep_id = sweep_id or f"sweep_{int(time.time())}"

        if spec.llm_mode == "replay":
            if spec.fixture is None:
                raise ValueError("replay mode requires spec.fixture")
            ok, issues, stats = validate_fixture(spec.fixture)
            if not ok:
                raise ValueError(f"fixture failed validation: {issues}")
            logger.info("Fixture OK: %d trials / %d entries", stats.n_trials, stats.n_entries)

        self.store.store_sweep_metadata(sweep_id=sweep_id, metadata={
            "created_at": int(time.time()),
            "hardware_sku": self.platform.model_name,
            "vcpu_basis": spec.vcpu_basis,
            "vcpu_basis_kind": spec.vcpu_basis_kind,
            "numa_policy": spec.numa_policy,
            "model": "agentsysperf-proxy",
            "replay_fixture": str(spec.fixture) if spec.fixture else None,
            "benchmark": spec.benchmark,
            # Provenance: dry-run cells are synthetic/illustrative (no Harbor,
            # modeled points), real cells are measured. Stored in the metadata
            # JSON blob so the dashboard can badge synthetic sweeps and not
            # mistake them for measurements. (lands in sweeps.metadata)
            "data_source": "synthetic" if dry_run else "measured",
        })
        # The scaling verdict is stored under run_id=sweep_id (below), and each
        # cell's sweep_point uses run_id="{sweep_id}::d..r..". Both reference
        # runs(run_id) via FKs that become ON DELETE CASCADE in P3 — so the
        # parent runs rows MUST exist or the inserts fail once foreign_keys=ON.
        # Insert the sweep_id run row now; per-cell run rows are inserted in the
        # cell loop before store_sweep_point. (STORAGE_IMPLEMENTATION_PLAN P2.)
        self._ensure_run_row(sweep_id, kind="sweep")

        cells = spec.cells()
        logger.info(
            "Sweep %s: %d cells (densities=%s, basis=%d %s, replicates=%d)",
            sweep_id, len(cells), list(spec.densities), spec.vcpu_basis,
            spec.vcpu_basis_kind, spec.replicates,
        )

        all_points: List[Dict[str, Any]] = []
        per_task_points: Dict[str, List[Dict[str, Any]]] = {}

        for (density, concurrency, replicate) in cells:
            logger.info(
                "── cell: density=%g concurrency=%d replicate=%d ──",
                density, concurrency, replicate,
            )
            if dry_run:
                point = self._synthetic_point(sweep_id, density, concurrency, replicate)
            else:
                point, task_rows = self._run_cell(sweep_id, density, concurrency, replicate)
                for tr in task_rows:
                    per_task_points.setdefault(tr["task"], []).append({**point, **tr})
            # Parent runs row for this cell before its sweep_point (FK target).
            self._ensure_run_row(point["run_id"], kind="sweep_cell", sweep_id=sweep_id)
            self.store.store_sweep_point(sweep_id=sweep_id, point=point)
            all_points.append(point)

        # Sweep-level analysis: knee + bottleneck across all cells.
        result = self.analyzer.analyze_sweep(
            all_points,
            logical_cpus=self.platform.logical_cpus or (os.cpu_count() or 1),
            per_task_points=per_task_points or None,
        )
        if result is not None:
            # Store under the sweep_id (reuse span_id slot as the grouping key).
            from dataclasses import replace
            self.store.store_analysis_results(
                run_id=sweep_id, results=[replace(result, span_id=sweep_id)]
            )
            logger.info("Scaling verdict: %s (conf=%.2f)", result.verdict, result.confidence)
        return sweep_id

    def _ensure_run_row(self, run_id: str, *, kind: str, sweep_id: Optional[str] = None) -> None:
        """Insert a minimal runs row so sweep FKs (run_id) are not dangling.

        Sweeps live ABOVE runs additively: the sweep_id and each cell run_id are
        used as run_ids by sweep_points / the scaling verdict, but harbor_sweep
        never created the matching runs rows. With foreign_keys=ON + the P3
        cascade FKs, those inserts would fail (and a fail-loud migration would
        reject the dangling rows). store_run_metadata upserts, so this is safe to
        call repeatedly. hardware_sku comes from the detected platform, never a
        literal.
        """
        meta = {
            "start_time": int(time.time()),
            "hardware_sku": self.platform.model_name,
            "model": "agentsysperf-proxy",
            "benchmark": self.spec.benchmark,
            "run_kind": kind,
        }
        if sweep_id is not None:
            meta["sweep_id"] = sweep_id
        self.store.store_run_metadata(run_id=run_id, metadata=meta)

    # ── One cell ─────────────────────────────────────────────────────

    def _run_cell(
        self, sweep_id: str, density: float, concurrency: int, replicate: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        spec = self.spec
        run_id = f"{sweep_id}::d{density:g}_r{replicate}"
        cell_dir = spec.output_dir / f"d{density:g}_r{replicate}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        jobs_dir = cell_dir / "jobs"

        l1sys = L1SystemMeasurement(sample_interval_s=spec.sample_interval_s)
        ctx = RunContext(measurements=[l1sys], output_dir=cell_dir)

        # Start EMON collection for the cell — delegated to the discovered
        # Intel-only plugin (self._emon); a no-op when it is not installed.
        emon = self._emon
        if emon is not None:
            try:
                emon.start(run_id=run_id, output_dir=cell_dir)
                logger.info("EMON collecting for cell d%g_r%d", density, replicate)
                time.sleep(1.0)  # warmup
            except Exception as e:  # optional telemetry never fails a cell
                logger.warning("EMON: start failed (%s) — cell runs without EMON", e)
                emon = None

        elapsed = 0.0
        # Terminus-2's LLM call runs HOST-SIDE (in the `harbor run` process), so
        # 127.0.0.1 is the right address to dial. Bind 0.0.0.0 anyway (harmless;
        # also reachable via the bridge gateway should a future agent run
        # containerized). The corp-proxy hijack of this local call is handled in
        # _harbor_run via NO_PROXY=* (httpx honors the wildcard).
        with ReplayProxy(
            mode=spec.llm_mode, fixture=spec.fixture, port=spec.proxy_port,
            host="0.0.0.0", advertise_host="127.0.0.1",  # nosec B104
        ) as proxy, ctx:
            span_id = f"{run_id}::cell"
            with track_span(ctx, span_id, kind="execution", node_id=f"density={density:g}"):
                t0 = time.time()
                harbor_rc = self._harbor_run(jobs_dir, concurrency, proxy.env)
                elapsed = time.time() - t0

        # Stop EMON collection and pull the post-processed metrics-CSV path from
        # the plugin's own record (the plugin owns collection + post-processing).
        emon_csv_path = None
        if emon is not None:
            time.sleep(0.5)  # cooldown
            try:
                for rec in emon.stop():
                    csv = rec.payload.get("csv_path")
                    if csv:
                        emon_csv_path = csv
                        logger.info("EMON: metrics CSV → %s", Path(csv).name)
            except Exception as e:
                logger.warning("EMON: stop/post-process failed (%s)", e)

        # Pull node telemetry rollup for this cell.
        rec = next((r for r in ctx.records
                    if r.layer == "l1_system" and r.span_id == span_id), None)
        tele = rec.payload if rec else {}

        # Per-trial timings from Harbor.
        task_rows = self._collect_task_phases(jobs_dir)
        if not task_rows:
            raise RuntimeError(
                f"harbor run produced no trials for cell density={density:g} "
                f"concurrency={concurrency} replicate={replicate} "
                f"(exit {harbor_rc}); see {jobs_dir.parent / 'harbor_stderr.txt'}"
            )
        if harbor_rc != 0:
            logger.warning(
                "harbor run exited %d but %d trial(s) completed — keeping the cell; "
                "rewards below reflect the failures",
                harbor_rc, len(task_rows),
            )
        latencies = sorted(t["duration_s"] for t in task_rows if t.get("duration_s"))
        completed = len(task_rows)
        p95 = _percentile(latencies, 95)
        throughput = completed / (elapsed / 60.0) if elapsed > 0 else 0.0

        point = {
            "run_id": run_id,
            "density": density,
            "concurrency": concurrency,
            "replicate": replicate,
            "elapsed_s": round(elapsed, 1),
            "throughput_per_min": round(throughput, 2),
            "completed_trials": completed,
            "p95_trial_latency_s": round(p95, 2),
            "cpu_avg": tele.get("cpu_avg"),
            "cpu_p95": tele.get("cpu_p95"),
            "cpu_peak": tele.get("cpu_peak"),
            "runqueue_max": tele.get("runqueue_max"),
            "ctx_sw_per_s": tele.get("ctx_sw_per_s_avg"),
            "mem_avail_mb_min": tele.get("mem_avail_mb_min"),
            "iowait_pct_avg": tele.get("iowait_pct_avg"),
            "emon_csv": str(emon_csv_path) if emon_csv_path else None,
        }
        # Per-task rows carry their own latency for the per-task sweep curve.
        task_curve_rows = [
            {"task": t["task"], "p95_trial_latency_s": t.get("duration_s", 0.0),
             "throughput_per_min": throughput}
            for t in task_rows
        ]
        return point, task_curve_rows

    def _harbor_run(self, jobs_dir: Path, concurrency: int, proxy_env: Dict[str, str]) -> int:
        spec = self.spec
        api_base = proxy_env["OPENAI_API_BASE"]
        # Dataset source: local --path (no registry egress) or -d registry name.
        # With --path to a local dataset dir, -i includes are bare task names;
        # with the registry dataset they're "terminal-bench/<name>".
        import shlex
        if spec.dataset_path:
            dataset_arg = f"--path {shlex.quote(str(spec.dataset_path))}"
            include = " ".join(f"-i {shlex.quote(t.split('/')[-1])}" for t in spec.tasks)
        else:
            dataset_arg = "-d terminal-bench/terminal-bench-2"
            include = " ".join(
                f"-i {shlex.quote('terminal-bench/' + t.split('/')[-1])}" for t in spec.tasks
            )
        # --force-build: build task images locally instead of pulling prebuilt
        # ones from a registry. EMR has no direct Docker Hub egress (pulls time
        # out), so this is required — same fix as the terminal_bench adapter's
        # force_build=True (harbor_environment.py).
        cmd = (
            f"harbor run {dataset_arg} -a terminus-2 "
            f'-m "openai/agentsysperf-proxy" --force-build '
            f"-n {concurrency} -k {spec.attempts} {include} -o {shlex.quote(str(jobs_dir))} "
            f'--ae "OPENAI_API_BASE={api_base}" '
            f'--ae "OPENAI_API_KEY=sk-not-needed-local-only" '
            f'--ae "OPENAI_BASE_URL={api_base}" '
            f'--ae "NO_PROXY=*" --ae "no_proxy=*" '
            f"--agent-timeout-multiplier {spec.agent_timeout_multiplier} "
            f"--agent-setup-timeout-multiplier 3.0 --no-delete -y"
        )
        # Terminus-2's LLM call runs HOST-SIDE via harbor.llms.lite_llm (litellm),
        # in this `harbor run` process, which inherits the corporate HTTP_PROXY.
        # litellm's httpx client routes even the local replay-proxy call
        # (127.0.0.1) through the corp proxy → an HTML error page ("IE friendly
        # error"). httpx does NOT honor NO_PROXY's host/CIDR forms, but it DOES
        # honor the WILDCARD `NO_PROXY=*` (bypass all). Verified: trust_env-bypass
        # is the only thing that reaches the proxy. This replay run needs no corp
        # proxy (LLM is local; tasks --force-build), so bypass everything.
        env = {**os.environ, **proxy_env}
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
        # Ensure the venv bin/ is on PATH so harbor CLI is found
        venv_bin = Path(sys.executable).parent
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        logger.info("harbor: %s", cmd[:140])
        # No shell. The command is a single `harbor run` invocation with no
        # pipeline, redirect, glob or substitution, so shlex.split reproduces
        # exactly the argv /bin/sh would have built (the `*` in NO_PROXY=* is
        # inside quotes either way, so it stays literal) — minus the sh layer
        # that made every interpolated task name and path a metacharacter risk.
        proc = subprocess.run(shlex.split(cmd), env=env, capture_output=True,
                              text=True, timeout=7200)
        (jobs_dir.parent / "harbor_stdout.txt").write_text(proc.stdout)
        (jobs_dir.parent / "harbor_stderr.txt").write_text(proc.stderr)
        if proc.returncode != 0:
            logger.error(
                "harbor run exited %d — see %s",
                proc.returncode,
                jobs_dir.parent / "harbor_stderr.txt",
            )
        return proc.returncode

    @staticmethod
    def _collect_task_phases(jobs_dir: Path) -> List[Dict[str, Any]]:
        """Extract per-trial timing/reward from Harbor result.json files."""
        rows: List[Dict[str, Any]] = []
        if not jobs_dir.exists():
            return rows
        for job_dir in sorted(jobs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            for sub in sorted(job_dir.iterdir()):
                rf = sub / "result.json"
                if not rf.exists():
                    continue
                try:
                    d = json.loads(rf.read_text())
                    ae = d.get("agent_execution") or {}
                    vr = d.get("verifier_result") or {}
                    rewards = vr.get("rewards") or {}
                    reward = rewards.get("reward", 0.0) if isinstance(rewards, dict) else 0.0
                    rows.append({
                        "task": d.get("task_name") or sub.name,
                        "duration_s": ae.get("duration_s", 0.0),
                        "reward": reward,
                        "turns": ae.get("total_turns", 0),
                    })
                except (ValueError, OSError) as e:
                    logger.warning("could not parse %s: %s", rf, e)
        return rows

    # ── Dry-run synthetic cell (no Harbor; smoke-tests the pipeline) ──

    def _synthetic_point(
        self, sweep_id: str, density: float, concurrency: int, replicate: int,
    ) -> Dict[str, Any]:
        """A plausible saturating cell so the pipeline can be exercised offline.

        Throughput rises then flattens; CPU and runqueue climb with density —
        enough for the analyzer to find a knee. NOT real data; dry-run only.
        """
        # Diminishing-returns throughput: saturates around density ~1.
        throughput = 40.0 * (1.0 - 1.0 / (1.0 + 2.0 * density))
        cpu = min(99.0, 30.0 + 45.0 * density)
        return {
            "run_id": f"{sweep_id}::d{density:g}_r{replicate}",
            "density": density, "concurrency": concurrency, "replicate": replicate,
            "elapsed_s": round(120.0 / max(density, 0.1), 1),
            "throughput_per_min": round(throughput, 2),
            "completed_trials": len(self.spec.tasks) * self.spec.attempts,
            "p95_trial_latency_s": round(5.0 + 6.0 * density, 2),
            "cpu_avg": round(cpu, 1),
            "cpu_p95": round(min(100.0, cpu + 5), 1),
            "cpu_peak": round(min(100.0, cpu + 8), 1),
            "runqueue_max": round(concurrency * 0.5, 1),
            "ctx_sw_per_s": 5000.0 + 3000.0 * density,
            "mem_avail_mb_min": 900000.0,
            "iowait_pct_avg": 0.0,
        }


__all__ = ["HarborSweep"]
