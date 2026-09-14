#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
=============================================================================
AgentSysPerf — Reference Optimization Profiles  (base vs Xeon-optimized)
=============================================================================

Six reference OptimizationProfilePlugin implementations. `base` is the
portable reference; the rest progressively enable Xeon-specific
optimizations so a gain can be ATTRIBUTED to a specific axis rather than
hidden inside one opaque "optimized" bundle.

The profile is swept exactly like routing_strategy: held-constant
everything else, vary only the profile, measure what the tuning buys.

TWO INTEGRITY RULES (enforced by the harness, encoded here):

  1. `base` is a REASONABLE PORTABLE build, never a strawman. It is
     AVX-512 + oneDNN-absent + FP32 + unpinned — i.e. what you actually get
     from a stock `pip install`, not an artificially crippled single-thread
     FP32 reference. is_baseline=True only for this arm; the reporter
     invalidates the study if the baseline is itself heavily optimized.

  2. Every enabled optimization declares the telemetry counter that PROVES
     it engaged, plus a minimum threshold. verify_engaged() reads the real
     counter via HardwareTelemetryPlugin and returns engaged=False if a
     claimed optimization shows ~zero activity. The harness then ABORTS that
     arm — it is never reported as a valid data point. "Built with AMX" is
     not evidence AMX ran.

CONFIDENCE NOTE:
- These are reference profiles for the benchmark, not production configs.
- The verify_counters thresholds (e.g. amx_active_cycle_ratio >= 0.05) are
  STARTING ESTIMATES. Calibrate them against a known-good AMX run on the
  target hardware before trusting an engaged/not-engaged verdict.
- backend_config / scheduler_config keys are illustrative; the concrete
  InferenceBackend / Scheduler plugins define the authoritative key names.
=============================================================================
"""

from __future__ import annotations
from typing import Any

from plugin_contracts import (
    OptimizationProfilePlugin,
    OptimizationProfile,
    ProfileProjection,
    ProfileEngagementReport,
    HardwareTelemetryPlugin,
)


# ---------------------------------------------------------------------------
# Shared projection helper: translate axis settings into plugin configs
# ---------------------------------------------------------------------------
def _project(profile: OptimizationProfile) -> ProfileProjection:
    ax = profile.axes
    backend: dict[str, Any] = {
        "isa_target":       ax.get("isa", "avx512"),
        "math_library":     ax.get("math_library", "reference"),
        "runtime":          ax.get("runtime", "vanilla"),
        "quantization":     ax.get("quantization", "fp32"),
        "accelerator":      ax.get("accelerator", "none"),
    }
    scheduler: dict[str, Any] = {
        "numa_policy":      ax.get("numa", "unpinned"),
        "memory_pages":     ax.get("memory_pages", "4k"),
        "core_isolation":   ax.get("core_isolation", "shared"),
    }
    return ProfileProjection(
        backend_config=backend,
        scheduler_config=scheduler,
        required_telemetry_counters=list(profile.verify_counters.keys()),
    )


async def _verify(
    profile: OptimizationProfile,
    telemetry: HardwareTelemetryPlugin,
    node_id: str,
) -> ProfileEngagementReport:
    """Read the declared counters; an enabled optimization that shows
    ~zero activity fails the arm. _max suffix means 'must stay BELOW'."""
    wanted = list(profile.verify_counters.keys())
    per_axis: dict[str, bool] = {}
    failures: list[str] = []
    measured: dict[str, float] = {}

    if not wanted:
        # base arm: nothing to prove, trivially engaged
        return ProfileEngagementReport(
            profile_name=profile.name, engaged=True,
            per_axis={}, measured_counters={}, failures=[],
        )

    try:
        measured = await telemetry.read_engagement_counters(node_id, wanted)
    except Exception as e:
        return ProfileEngagementReport(
            profile_name=profile.name, engaged=False, per_axis={},
            measured_counters={},
            failures=[f"counter read failed ({e}); cannot trust the label"],
        )

    for counter, threshold in profile.verify_counters.items():
        val = measured.get(counter)
        if val is None:
            per_axis[counter] = False
            failures.append(f"{counter}: unavailable on this hardware")
            continue
        if counter.endswith("_max"):
            ok = val <= threshold
            if not ok:
                failures.append(
                    f"{counter}={val:.3f} exceeds max {threshold:.3f} "
                    f"(optimization regressed, not engaged)")
        else:
            ok = val >= threshold
            if not ok:
                failures.append(
                    f"{counter}={val:.3f} below min {threshold:.3f} "
                    f"(claimed optimization did not engage)")
        per_axis[counter] = ok

    return ProfileEngagementReport(
        profile_name=profile.name,
        engaged=(len(failures) == 0),
        per_axis=per_axis,
        measured_counters=measured,
        failures=failures,
    )


# ===========================================================================
# 1. base — portable reference (NOT a strawman)
# ===========================================================================
class BaseProfile(OptimizationProfilePlugin):
    """Stock `pip install` reality: AVX-512, no oneDNN, FP32, unpinned.
    Deliberately a *reasonable* baseline so speedups aren't inflated."""

    @property
    def profile_name(self) -> str:
        return "base"

    @property
    def is_baseline(self) -> bool:
        return True

    def get_profile(self) -> OptimizationProfile:
        return OptimizationProfile(
            name="base",
            axes={
                "isa": "avx512", "math_library": "reference",
                "runtime": "vanilla", "quantization": "fp32",
                "numa": "unpinned", "memory_pages": "4k",
                "core_isolation": "shared", "accelerator": "none",
            },
            verify_counters={},   # nothing claimed → nothing to verify
            notes="Portable reference. Reporter invalidates study if this "
                  "arm shows AMX/oneDNN activity (would mean a rigged base).",
        )

    def project(self, profile): return _project(profile)
    async def verify_engaged(self, profile, telemetry, node_id):
        return await _verify(profile, telemetry, node_id)


# ===========================================================================
# 2. amx_only — isolate the AMX contribution
# ===========================================================================
class AMXOnlyProfile(OptimizationProfilePlugin):
    @property
    def profile_name(self) -> str: return "amx_only"
    @property
    def is_baseline(self) -> bool: return False

    def get_profile(self) -> OptimizationProfile:
        return OptimizationProfile(
            name="amx_only",
            axes={
                "isa": "amx_tdpbssd", "math_library": "reference",
                "runtime": "vanilla", "quantization": "int8_vnni",
                "numa": "unpinned", "memory_pages": "4k",
                "core_isolation": "shared", "accelerator": "none",
            },
            # AMX must actually run: tile-active cycles must be a real
            # fraction of busy cycles, else the kernel fell back to AVX-512.
            verify_counters={"amx_active_cycle_ratio": 0.05},
            notes="Isolates AMX vs the base ISA. INT8 needed for AMX path.",
        )

    def project(self, profile): return _project(profile)
    async def verify_engaged(self, profile, telemetry, node_id):
        return await _verify(profile, telemetry, node_id)


# ===========================================================================
# 3. amx_onednn — AMX + tuned GEMM library
# ===========================================================================
class AMXoneDNNProfile(OptimizationProfilePlugin):
    @property
    def profile_name(self) -> str: return "amx_onednn"
    @property
    def is_baseline(self) -> bool: return False

    def get_profile(self) -> OptimizationProfile:
        return OptimizationProfile(
            name="amx_onednn",
            axes={
                "isa": "amx_tdpbssd", "math_library": "onednn",
                "runtime": "ipex", "quantization": "int8_vnni",
                "numa": "unpinned", "memory_pages": "4k",
                "core_isolation": "shared", "accelerator": "none",
            },
            verify_counters={
                "amx_active_cycle_ratio": 0.05,
                "onednn_kernel_dispatch_ratio": 0.5,  # most GEMMs via oneDNN
            },
            notes="Adds oneDNN+IPEX on top of AMX. Attributes the library "
                  "contribution separately from raw ISA.",
        )

    def project(self, profile): return _project(profile)
    async def verify_engaged(self, profile, telemetry, node_id):
        return await _verify(profile, telemetry, node_id)


# ===========================================================================
# 4. amx_numa_hugepages — compute + memory-subsystem tuning
# ===========================================================================
class AMXNumaHugepagesProfile(OptimizationProfilePlugin):
    @property
    def profile_name(self) -> str: return "amx_numa_hugepages"
    @property
    def is_baseline(self) -> bool: return False

    def get_profile(self) -> OptimizationProfile:
        return OptimizationProfile(
            name="amx_numa_hugepages",
            axes={
                "isa": "amx_tdpbssd", "math_library": "onednn",
                "runtime": "ipex", "quantization": "int8_vnni",
                "numa": "numa_local", "memory_pages": "hugepages_2m",
                "core_isolation": "isolcpus_affinity", "accelerator": "none",
            },
            verify_counters={
                "amx_active_cycle_ratio": 0.05,
                "hugepage_fault_ratio": 0.5,
                "numa_remote_access_ratio_max": 0.15,   # _max → stay below
            },
            notes="Adds NUMA-local pinning + 2M hugepages + core isolation. "
                  "Isolates the memory-subsystem contribution.",
        )

    def project(self, profile): return _project(profile)
    async def verify_engaged(self, profile, telemetry, node_id):
        return await _verify(profile, telemetry, node_id)


# ===========================================================================
# 5. full_xeon — everything on, including INT4/AMX + accelerator offload
# ===========================================================================
class FullXeonProfile(OptimizationProfilePlugin):
    @property
    def profile_name(self) -> str: return "full_xeon"
    @property
    def is_baseline(self) -> bool: return False

    def get_profile(self) -> OptimizationProfile:
        return OptimizationProfile(
            name="full_xeon",
            axes={
                "isa": "amx_tdpbssd", "math_library": "onednn",
                "runtime": "ipex", "quantization": "int4_amx",
                "numa": "numa_local", "memory_pages": "hugepages_1g",
                "core_isolation": "isolcpus_affinity", "accelerator": "qat",
            },
            verify_counters={
                "amx_active_cycle_ratio": 0.05,
                "onednn_kernel_dispatch_ratio": 0.5,
                "hugepage_fault_ratio": 0.5,
                "numa_remote_access_ratio_max": 0.15,
                "accel_queue_depth_qat": 1.0,   # QAT actually receiving work
            },
            notes="All axes on. The headline 'Xeon-optimized' arm. Only "
                  "valid if EVERY counter clears its threshold.",
        )

    def project(self, profile): return _project(profile)
    async def verify_engaged(self, profile, telemetry, node_id):
        return await _verify(profile, telemetry, node_id)


# ===========================================================================
# 6. openvino_xeon — alternative runtime path (OpenVINO instead of IPEX)
# ===========================================================================
class OpenVINOXeonProfile(OptimizationProfilePlugin):
    @property
    def profile_name(self) -> str: return "openvino_xeon"
    @property
    def is_baseline(self) -> bool: return False

    def get_profile(self) -> OptimizationProfile:
        return OptimizationProfile(
            name="openvino_xeon",
            axes={
                "isa": "amx_tdpbssd", "math_library": "onemkl",
                "runtime": "openvino", "quantization": "int8_vnni",
                "numa": "numa_local", "memory_pages": "hugepages_2m",
                "core_isolation": "isolcpus_affinity", "accelerator": "none",
            },
            verify_counters={
                "amx_active_cycle_ratio": 0.05,
                "openvino_infer_request_ratio": 0.9,   # path actually used
                "hugepage_fault_ratio": 0.5,
            },
            notes="Alternative runtime arm. Lets the study compare IPEX vs "
                  "OpenVINO as a clean swap, everything else equal.",
        )

    def project(self, profile): return _project(profile)
    async def verify_engaged(self, profile, telemetry, node_id):
        return await _verify(profile, telemetry, node_id)


# ===========================================================================
# REGISTRY — exposed under agentsysperf.optimization_profile entry points
# ===========================================================================
REFERENCE_PROFILES = {
    "base":               BaseProfile,
    "amx_only":           AMXOnlyProfile,
    "amx_onednn":         AMXoneDNNProfile,
    "amx_numa_hugepages": AMXNumaHugepagesProfile,
    "full_xeon":          FullXeonProfile,
    "openvino_xeon":      OpenVINOXeonProfile,
}
