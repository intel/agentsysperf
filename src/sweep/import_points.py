#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Import a shell-runner sweep (``point.json`` per cell) into the canonical store.

``harness/scripts/run_density_test_cwf.sh`` — the runner that actually varies
concurrency correctly — writes one ``point.json`` per density cell and nothing
else. The dashboard reads ``sweeps`` / ``sweep_points``. So the working runner is
the one that does not reach the dashboard, and the one that reaches the dashboard
(``harbor_sweep.py``) does not vary its own independent variable. This module
closes that gap without requiring either side to change.

Deliberately generic: any runner that emits ``point.json`` in the documented
shape can be imported, so this is not welded to one script.

See the internal density-sweep fix plan (P1) and
``docs/adr/0001-two-agent-loops-litellm-and-terminus.md``.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# point.json key -> sweep_points column. Everything not listed here (and not
# dropped below) spills into sweep_points.metadata, which store_sweep_point
# already does for unknown keys — so no migration is needed.
_RENAMES = {
    "concurrency_requested": "concurrency",
    "agents_per_core": "density",
    "p95_agent_exec_s": "p95_trial_latency_s",
}

# Bulky per-trial detail: useful on disk, noise in a JSON column.
_DROP = {"trials", "counters", "__cell_dir__"}

# Raw evidence files the runner leaves beside point.json. Registered as
# artifacts so a rollup can be audited by run_id instead of by remembering a
# path — the same reason the EMON CSV is registered. Kind is the stem, so
# get_artifact_path(run_id, kind="perf_csv") resolves the counters behind a
# cell's IPC figure.
_CELL_ARTIFACTS = (
    ("perf.csv", "perf_csv"),
    ("membw.csv", "membw_csv"),
    ("vmstat.txt", "vmstat_txt"),
    ("telemetry.json", "cpuset_telemetry_json"),
    ("containers.json", "container_telemetry_json"),
    ("harbor_stdout.txt", "harbor_stdout"),
    ("harbor_stderr.txt", "harbor_stderr"),
    ("point.json", "sweep_point_json"),
)

# Node telemetry columns. The runner supplies these from cpuset_telemetry.py; a
# sweep taken before that existed has them absent. Left absent rather than
# zero-filled: ScalingAnalyzer treats a missing value as unmeasured, and a
# fabricated 0.0 would read as "no CPU pressure" and classify a saturated cell
# as headroom_remaining.
_TELEMETRY = (
    "cpu_avg", "cpu_p95", "cpu_peak", "runqueue_max",
    "ctx_sw_per_s", "mem_avail_mb_min", "iowait_pct_avg",
)


def _parse_provenance(path: Path) -> Dict[str, str]:
    """Parse the runner's ``provenance.txt`` (``key: value`` lines)."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def _agent_from_provenance(prov: Dict[str, str]) -> Optional[str]:
    """Extract the bare agent name from the provenance line.

    ADR 0001: the agent is a recorded parameter, never implied by which file was
    run. A sweep whose agent is unknown cannot be compared to any other sweep,
    so this is load-bearing rather than cosmetic.
    """
    raw = prov.get("agent")
    if not raw:
        return None
    # "oracle (solve.sh, no LLM)" -> "oracle"; "terminus-2 (replay proxy)" -> "terminus-2"
    return raw.split("(")[0].strip() or None


def _cell_sort_key(p: Path) -> Tuple[int, int]:
    """Sort cells by (concurrency, replicate) from the ``n{N}_r{R}`` dir name."""
    m = re.match(r"n(\d+)_r(\d+)$", p.parent.name)
    return (int(m.group(1)), int(m.group(2))) if m else (1 << 30, 0)


def import_sweep(
    results_dir: Path,
    *,
    sweep_id: Optional[str] = None,
    store: Any = None,
    benchmark: str = "terminal-bench",
    include_failed: bool = False,
    stamp: Optional[int] = None,
) -> Dict[str, Any]:
    """Import every ``point.json`` under *results_dir* into the store.

    Returns a summary dict: ``sweep_id``, ``imported``, ``skipped``, ``vcpu_basis``.

    Cells whose ``cell_status != "ok"`` are skipped unless *include_failed*: a
    cell that failed, or that found a different number of trials than it
    expected, has a wrong throughput denominator. Persisting it would put a
    fabricated operating point on the curve. Skips are logged, never silent.
    """
    results_dir = Path(results_dir)
    points = sorted(results_dir.glob("n*_r*/point.json"), key=_cell_sort_key)
    if not points:
        raise FileNotFoundError(f"no n*_r*/point.json under {results_dir}")

    prov = _parse_provenance(results_dir / "provenance.txt")
    agent = _agent_from_provenance(prov)
    if agent is None:
        logger.warning(
            "%s has no agent in provenance.txt — the sweep will be stored with "
            "agent=unknown and cannot be compared against other sweeps (ADR 0001)",
            results_dir,
        )
        agent = "unknown"

    if store is None:
        from src.storage.sqlite_store import SQLiteResultStore

        store = SQLiteResultStore.open()

    stamp = int(stamp if stamp is not None else time.time())
    sweep_id = sweep_id or f"sweep_{results_dir.name}_{stamp}"

    loaded: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for f in points:
        try:
            d = json.loads(f.read_text())
        except (ValueError, OSError) as e:
            logger.warning("could not read %s: %s", f, e)
            skipped.append({"cell": f.parent.name, "reason": f"unreadable: {e}"})
            continue
        status = d.get("cell_status", "unknown")
        if status != "ok" and not include_failed:
            logger.warning(
                "skipping cell %s: cell_status=%s (n=%s) — a failed cell has a "
                "wrong throughput denominator and is not an operating point",
                f.parent.name, status, d.get("concurrency_requested"),
            )
            skipped.append({"cell": f.parent.name, "reason": f"cell_status={status}"})
            continue
        # Keep the cell directory: the raw perf/membw/vmstat/telemetry files
        # beside point.json are the only way to audit a rollup, and the importer
        # registers them as artifacts below. Stored under a dunder key so it
        # cannot collide with a real point field and is easy to strip.
        d["__cell_dir__"] = f.parent
        loaded.append(d)

    if not loaded:
        raise ValueError(
            f"no usable cells in {results_dir} "
            f"({len(skipped)} skipped; pass include_failed=True to force)"
        )

    # The density basis is the CPUSET size, not the machine's core count. A
    # 16-core pinned run at n=16 is density 1.0 here and 0.056 under
    # SweepSpec's concurrency/288. The basis MUST travel with the rows or the two
    # definitions get plotted on one axis and mean nothing.
    cores = loaded[0].get("cores_in_cpuset")
    cpuset = loaded[0].get("cpuset")
    if not isinstance(cores, int) or cores <= 0:
        raise ValueError(
            "point.json missing/invalid cores_in_cpuset; cannot derive sweep density basis"
        )
    if any(c.get("cores_in_cpuset") != cores for c in loaded):
        raise ValueError(
            "cells disagree on cores_in_cpuset; a sweep must share one density "
            "basis. Import each cpuset as its own sweep."
        )

    store.store_sweep_metadata(sweep_id=sweep_id, metadata={
        "created_at": stamp,
        "hardware_sku": prov.get("model_name") or "",
        "vcpu_basis": cores,
        "vcpu_basis_kind": "cpuset_cores",
        # numa_policy is honest here, unlike SweepSpec's unapplied field: the
        # runner really does pin via a compose-overlay cpuset, verified
        # in-container on GNR (CpusetCpus=43-85).
        "numa_policy": f"cpuset_pinned:{cpuset}" if cpuset else "unpinned",
        "model": agent,
        "replay_fixture": None,
        "benchmark": benchmark,
        "data_source": "measured",
        "agent": agent,
        "source_dir": str(results_dir),
        "cpuset": cpuset,
        "numa_node": loaded[0].get("numa_node"),
        "governor": prov.get("governor"),
        "sep_driver_loaded": prov.get("sep_driver_loaded"),
        "host": prov.get("host"),
        "task": prov.get("task"),
        "cells_skipped": len(skipped),
    })

    # sweep_points.run_id and the scaling verdict both reference runs(run_id).
    # Insert the parent rows first or the FKs dangle (same reason
    # harbor_sweep._ensure_run_row exists).
    def _ensure_run(run_id: str, kind: str) -> None:
        meta = {
            "start_time": stamp,
            "hardware_sku": prov.get("model_name") or "",
            "model": agent,
            "benchmark": benchmark,
            "run_kind": kind,
        }
        if kind == "sweep_cell":
            meta["sweep_id"] = sweep_id
        store.store_run_metadata(run_id=run_id, metadata=meta)

    _ensure_run(sweep_id, "sweep")

    imported = 0
    for d in loaded:
        n = d.get("concurrency_requested")
        rep = d.get("replicate")
        if n is None or rep is None:
            logger.warning(
                "skipping cell with missing concurrency_requested/replicate: %s",
                d.get("cell") or "<unknown>",
            )
            skipped.append({"cell": d.get("cell") or "<unknown>", "reason": "missing required keys"})
            continue
        run_id = f"{sweep_id}::n{n}_r{rep}"
        _ensure_run(run_id, "sweep_cell")
        point: Dict[str, Any] = {"run_id": run_id}
        for k, v in d.items():
            if k in _DROP:
                continue
            point[_RENAMES.get(k, k)] = v
        # Absent, not zero — see _TELEMETRY.
        for k in _TELEMETRY:
            point.setdefault(k, None)
        point["agent"] = agent
        point["data_source"] = "measured"

        store.store_sweep_point(sweep_id=sweep_id, point=point)
        _register_cell_artifacts(store, run_id, d.get("__cell_dir__"))
        imported += 1

    if not any(c.get("cpu_avg") is not None for c in loaded):
        logger.warning(
            "no cell in %s carries cpu_avg — ScalingAnalyzer cannot classify a "
            "bottleneck and every cell will read 'headroom_remaining' regardless "
            "of actual saturation. Re-run with harness/scripts/cpuset_telemetry.py "
            "wired in.",
            results_dir,
        )

    # Run the sweep-level analysis here, not as a separate step. The Python
    # runner (sweep/harbor_sweep.py) calls analyze_sweep itself, but this
    # shell-runner bridge did not — so every imported sweep landed with zero
    # verdicts, and the dashboard rendered no knee marker and no bottleneck
    # label for them. There is no CLI that backfills it: `agentsysperf analyze`
    # reads a results DIRECTORY of MeasurementRecords and only prints, so it can
    # neither see sweep_points nor write to the store.
    verdict = _analyze_imported_sweep(store, sweep_id, loaded)

    logger.info(
        "imported %d cell(s) into sweep %s (basis=%s cores, agent=%s); %d skipped",
        imported, sweep_id, cores, agent, len(skipped),
    )
    return {
        "sweep_id": sweep_id,
        "imported": imported,
        "skipped": skipped,
        "vcpu_basis": cores,
        "agent": agent,
        "verdict": verdict,
    }


def _register_cell_artifacts(store: Any, run_id: str, cell_dir: Optional[Path]) -> int:
    """Register a cell's raw evidence files so they are addressable by run_id.

    Returns the number registered. Absent files are skipped silently: a sweep
    taken before a given collector existed simply has fewer files, which is not
    an error. Never raises — losing an artifact pointer must not fail an import
    whose measurements already landed.
    """
    if cell_dir is None:
        return 0
    n = 0
    for filename, kind in _CELL_ARTIFACTS:
        p = Path(cell_dir) / filename
        if not p.is_file():
            continue
        try:
            store.store_artifact(run_id=run_id, kind=kind, name=filename,
                                 path=p.resolve())
            n += 1
        except Exception:  # noqa: BLE001
            logger.warning("could not register %s for %s", p, run_id, exc_info=True)
    return n


def _analyze_imported_sweep(store: Any, sweep_id: str, loaded: Sequence[Dict[str, Any]]):
    """Store one ScalingAnalyzer verdict for a just-imported sweep.

    Returns the verdict string, or None if the sweep could not be analyzed.
    Never raises: a failed analysis must not lose an import that already
    succeeded.
    """
    from src.analyzers.scaling import ScalingAnalyzer

    points = store.query_sweep_points(sweep_id)
    if not points:
        return None

    # The oversubscription ratio compares runqueue_max against logical_cpus, so
    # the basis must match the SCOPE the runqueue was sampled at. The runner
    # reads /proc/loadavg-style host-wide runqueue, so a 16-core cpuset run on a
    # 288-CPU box must divide by 288 — dividing by 16 declares
    # scheduler_oversubscription on a box that is 94% idle. (Same scope trap
    # fixed in the dashboard's _pressure_scores.)
    host_wide = any(c.get("runqueue_is_host_wide") for c in loaded)
    host_cpus = next(
        (c.get("logical_cpus_host") for c in loaded if c.get("logical_cpus_host")), None
    )
    cpuset_cores = next(
        (c.get("cores_in_cpuset") for c in loaded if c.get("cores_in_cpuset")), None
    )

    if host_wide:
        if not host_cpus:
            logger.warning(
                "sweep %s marks runqueue_is_host_wide but carries no logical_cpus_host; "
                "skipping the scaling verdict to avoid a scope mismatch",
                sweep_id,
            )
            return None
        basis = host_cpus
    else:
        basis = cpuset_cores
    if not basis:
        logger.warning(
            "sweep %s carries neither logical_cpus_host nor cores_in_cpuset; "
            "skipping the scaling verdict rather than guessing a runqueue basis",
            sweep_id,
        )
        return None

    try:
        result = ScalingAnalyzer().analyze_sweep(points, logical_cpus=int(basis))
    except Exception:  # noqa: BLE001 — analysis must not fail the import
        logger.warning("scaling analysis failed for %s", sweep_id, exc_info=True)
        return None
    if result is None:
        return None

    from dataclasses import replace
    try:
        store.store_analysis_results(
            run_id=sweep_id, results=[replace(result, span_id=sweep_id)]
        )
    except Exception:  # noqa: BLE001
        logger.warning("could not store scaling verdict for %s", sweep_id, exc_info=True)
        return None
    logger.info(
        "scaling verdict for %s: %s (conf=%.2f, runqueue basis=%s CPUs%s)",
        sweep_id, result.verdict, result.confidence, basis,
        " host-wide" if (host_wide and host_cpus) else " cpuset",
    )
    return result.verdict


__all__ = ["import_sweep"]
