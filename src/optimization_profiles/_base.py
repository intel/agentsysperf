#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Shared scaffolding for the 6 reference optimization profiles.

All six profiles share the same ``apply()`` projection logic and the
same ``verify_engaged()`` counter-reading + threshold-checking flow.
Centralising those here keeps the per-profile classes to a tight
declaration of axes and verify_counters.

Threshold convention
--------------------
``verify_counters`` maps counter name → numeric threshold. By default
the threshold is a MINIMUM: the counter reading must be ``>=`` it for
the axis to be engaged. A counter name ending in ``_max`` flips this:
the reading must be ``<=`` the threshold. Used for axes that prove
engagement by the *absence* of bad behaviour — e.g.
``numa_remote_access_ratio_max <= 0.15`` proves NUMA-local pinning
worked because remote access stayed low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Mapping, Sequence


class UnsupportedProfileError(RuntimeError):
    """Raised when a profile's required ISA is absent on the running host.

    Deliberately fatal rather than a warning: AgentSysPerf's abort-don't-degrade
    principle means a profile that cannot engage must not produce a labelled
    result. Callers that want to skip such profiles should consult
    ``unsupported_on()`` first rather than catching this.
    """


@dataclass(frozen=True)
class ProfileSpec:
    """The data each reference profile carries.

    ``axes`` is the orthogonal-knob bundle: ISA target, math library,
    runtime, quantization, NUMA policy, page size, core isolation,
    accelerator. ``verify_counters`` declares the hardware-counter
    evidence required to prove engagement. ``notes`` is documentation
    surfaced in the engagement report.
    """

    name: str
    is_baseline: bool
    axes: Mapping[str, str]
    verify_counters: Mapping[str, float]  # name → threshold (see _max convention)
    notes: str = ""
    # Default counter sample window. A profile that needs longer to
    # show stable activity can override.
    verify_window_s: float = 1.0
    # ISA capabilities the host must have for this profile to mean anything.
    # Names match PlatformInfo's has_* fields ("amx" -> has_amx). Empty means
    # the profile is portable. Checked by requires_unavailable_on(); a profile
    # whose prerequisite ISA is absent cannot produce a meaningful measurement,
    # so applying it must fail loudly rather than quietly measure the fallback
    # path and label the result as if the optimization were active.
    requires: Sequence[str] = ()


# ─── Default axis values ──────────────────────────────────────────────
# Kept as a single source of truth so the projection helper doesn't
# duplicate them across profiles.
_AXIS_DEFAULTS: Dict[str, str] = {
    "isa": "avx512",
    "math_library": "reference",
    "runtime": "vanilla",
    "quantization": "fp32",
    "numa": "unpinned",
    "memory_pages": "4k",
    "core_isolation": "shared",
    "accelerator": "none",
}


def project_axes(axes: Mapping[str, str]) -> Dict[str, Dict[str, str]]:
    """Translate axis settings into per-plugin config bundles.

    Returns a dict with ``backend_config`` and ``scheduler_config``
    sub-dicts, mirroring how the profile would land on a real
    InferenceBackend + Scheduler pair. The keys are illustrative; a
    concrete backend plugin defines the authoritative names.
    """
    a = {**_AXIS_DEFAULTS, **dict(axes)}
    return {
        "backend_config": {
            "isa_target": a["isa"],
            "math_library": a["math_library"],
            "runtime": a["runtime"],
            "quantization": a["quantization"],
            "accelerator": a["accelerator"],
        },
        "scheduler_config": {
            "numa_policy": a["numa"],
            "memory_pages": a["memory_pages"],
            "core_isolation": a["core_isolation"],
        },
    }


def evaluate_thresholds(
    *,
    verify_counters: Mapping[str, float],
    measured: Mapping[str, float],
    available_events: Sequence[str],
) -> Dict[str, Any]:
    """Apply the threshold convention and produce an engagement verdict.

    Pure function; no I/O. Caller supplies the measured counter values
    plus the set of events the telemetry plugin advertises (so we can
    distinguish "counter unavailable on this host" from "counter
    available but below threshold").

    Returns a dict matching the OptimizationProfilePlugin.verify_engaged
    contract: ``engaged``, ``counters``, ``thresholds``, ``failures``,
    ``per_axis``.
    """
    failures: list[str] = []
    per_axis: Dict[str, bool] = {}
    available = set(available_events)

    if not verify_counters:
        # Empty verify_counters means the profile claims nothing
        # — trivially engaged. This is how `base` works.
        return {
            "engaged": True,
            "counters": dict(measured),
            "thresholds": {},
            "failures": [],
            "per_axis": {},
        }

    for counter, threshold in verify_counters.items():
        if counter not in available:
            per_axis[counter] = False
            failures.append(
                f"{counter}: not advertised by any installed "
                f"HardwareTelemetryPlugin. Counter names are matched literally "
                f"against available_events, so some installed plugin must "
                f"advertise and compute this exact metric — note it is a "
                f"derived metric name, and not every one is a PMU quantity "
                f"(hugepage / QAT / oneDNN counters come from /proc, sysfs and "
                f"library instrumentation), so a vendor PMU plugin alone may "
                f"not supply it. See docs/measurement_layers.md"
            )
            continue
        val = measured.get(counter)
        if val is None:
            per_axis[counter] = False
            failures.append(
                f"{counter}: telemetry advertised it but did not return a "
                f"reading — check counter availability for the running SKU"
            )
            continue
        if counter.endswith("_max"):
            ok = val <= threshold
            if not ok:
                failures.append(
                    f"{counter}={val:.3f} exceeds max {threshold:.3f} "
                    f"(optimization regressed, not engaged)"
                )
        else:
            ok = val >= threshold
            if not ok:
                failures.append(
                    f"{counter}={val:.3f} below min {threshold:.3f} "
                    f"(claimed optimization did not engage)"
                )
        per_axis[counter] = ok

    return {
        "engaged": len(failures) == 0,
        "counters": dict(measured),
        "thresholds": dict(verify_counters),
        "failures": failures,
        "per_axis": per_axis,
    }


class BaseProfileImpl:
    """Shared body for the 6 reference :class:`OptimizationProfilePlugin` classes.

    Concrete profiles subclass this and set the ``SPEC`` class
    attribute to a :class:`ProfileSpec`. Entry-point discovery
    instantiates plugins with no args, so subclasses must not require
    constructor arguments.
    """

    SPEC: ClassVar[ProfileSpec]  # subclass MUST set

    @property
    def name(self) -> str:
        return self.SPEC.name

    @property
    def is_baseline(self) -> bool:
        return self.SPEC.is_baseline

    @property
    def requires(self) -> Sequence[str]:
        """ISA capabilities this profile needs (e.g. ``("amx",)``)."""
        return self.SPEC.requires

    def unsupported_on(self, platform_info: Any = None) -> list:
        """Return the required capabilities this host lacks.

        Empty list means the profile is applicable. ``platform_info`` defaults
        to live detection; pass an explicit PlatformInfo in tests.
        """
        if not self.SPEC.requires:
            return []
        if platform_info is None:
            from src.platform import detect_platform
            platform_info = detect_platform()
        return [
            cap
            for cap in self.SPEC.requires
            if not getattr(platform_info, f"has_{cap}", False)
        ]

    def apply(self, *, sut_config: Dict[str, Any]) -> Dict[str, Any]:
        """Layer this profile's axes onto the caller's SUT config.

        Returns a new dict; the caller's input is not mutated. The
        projection adds ``backend_config`` and ``scheduler_config``
        sub-dicts plus a ``profile`` field naming the active profile
        for downstream auditability.

        Profiles MUST NOT silently downgrade — if the host can't satisfy the
        profile's required ISA, this raises rather than returning a degraded
        config. Measuring an "amx_only" profile on a host without AMX would
        exercise the fallback path while labelling the result as AMX, which is
        worse than refusing. Engagement is verified separately via
        verify_engaged() after warmup.

        Raises
        ------
        UnsupportedProfileError
            If the host lacks a capability listed in ``SPEC.requires``.
        """
        missing = self.unsupported_on()
        if missing:
            raise UnsupportedProfileError(
                f"Profile {self.SPEC.name!r} requires "
                f"{', '.join(missing)} which this host does not support. "
                f"Refusing to apply: the run would measure the fallback path "
                f"but be labelled as {self.SPEC.name!r}. "
                f"Use a profile without that requirement (e.g. 'base')."
            )

        out = dict(sut_config)
        projection = project_axes(self.SPEC.axes)

        existing_backend = dict(out.get("backend_config", {}))
        existing_backend.update(projection["backend_config"])
        out["backend_config"] = existing_backend

        existing_scheduler = dict(out.get("scheduler_config", {}))
        existing_scheduler.update(projection["scheduler_config"])
        out["scheduler_config"] = existing_scheduler

        out["profile"] = self.SPEC.name
        out["profile_axes"] = dict(self.SPEC.axes)
        return out

    def verify_engaged(
        self,
        *,
        telemetry: Any,  # HardwareTelemetryPlugin; kept Any to avoid import cycle
        run_id: str,
        warmup_done: bool = True,
    ) -> Dict[str, Any]:
        """Read declared counters via ``telemetry``; return engagement verdict.

        The telemetry plugin decides the sampling target (system-wide
        vs per-PID vs cgroup); we don't override that here. A real
        runner would configure telemetry with the SUT's target before
        calling this method.
        """
        del run_id, warmup_done  # currently unused; kept for the contract

        wanted = list(self.SPEC.verify_counters.keys())
        if not wanted:
            return evaluate_thresholds(
                verify_counters={},
                measured={},
                available_events=getattr(telemetry, "available_events", ()),
            )

        reading = telemetry.read_counters(
            wanted, window_s=self.SPEC.verify_window_s,
        )
        return evaluate_thresholds(
            verify_counters=self.SPEC.verify_counters,
            measured=dict(reading.values),
            available_events=getattr(telemetry, "available_events", ()),
        )


__all__ = [
    "ProfileSpec",
    "project_axes",
    "evaluate_thresholds",
    "BaseProfileImpl",
]
