#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""The 6 reference :class:`OptimizationProfilePlugin` implementations.

Ported from ``docs/contracts/optimization_profiles.py``. Each profile
declares its axes and the counter evidence required to prove engagement.
``apply()`` and ``verify_engaged()`` come from :class:`BaseProfileImpl`.

Counter naming
--------------
Names like ``amx_active_cycle_ratio``, ``onednn_kernel_dispatch_ratio``,
``hugepage_fault_ratio``, ``numa_remote_access_ratio_max``,
``accel_queue_depth_qat``, ``openvino_infer_request_ratio`` are the
*contract names* the profiles depend on. They are NOT generic Linux
``perf`` events; a vendor-aware HardwareTelemetryPlugin (e.g. one that
wraps Intel PCM, or hooks into the IPEX/OpenVINO runtime) must
advertise them in ``available_events`` and return them from
``read_counters``. The default :class:`PerfStatTelemetry` only
advertises generic events; under that telemetry every amx_* profile
verify_engaged() returns engaged=False with a clear "not advertised"
failure. That is the abort-don't-degrade machinery working correctly.

Threshold values are STARTING ESTIMATES from the design package
(``docs/contracts/optimization_profiles.py``); they need calibration on
real SPR/EMR/GNR silicon before being trusted as engaged/not-engaged
verdicts.
"""

from __future__ import annotations

from src.optimization_profiles._base import BaseProfileImpl, ProfileSpec


class BaseProfile(BaseProfileImpl):
    """Stock ``pip install`` reality: AVX-512, no oneDNN, FP32, unpinned.

    Deliberately a *reasonable* baseline so speedups against it aren't
    inflated. ``is_baseline=True`` is the only profile for which this
    flag is set — the reporter invalidates the study if any
    non-baseline arm shows up with this flag.

    ``verify_counters`` is empty: the profile claims nothing, so
    verify_engaged() trivially passes. The full design has a separate
    ``baseline_purity_check`` that asserts AMX / hugepage / isolcpus
    activity is NOT present on a baseline run; that check belongs in a
    runner-level audit step, not in this profile's verify_engaged.
    """

    SPEC = ProfileSpec(
        name="base",
        is_baseline=True,
        axes={
            "isa": "avx512",
            "math_library": "reference",
            "runtime": "vanilla",
            "quantization": "fp32",
            "numa": "unpinned",
            "memory_pages": "4k",
            "core_isolation": "shared",
            "accelerator": "none",
        },
        verify_counters={},
        notes=(
            "Portable reference. Reporter invalidates the study if this "
            "arm shows AMX/oneDNN activity (would mean a rigged base)."
        ),
    )


class AMXOnlyProfile(BaseProfileImpl):
    """Isolate the AMX contribution: AMX ISA + INT8, otherwise stock.

    INT8/VNNI quantization is required because AMX-TDPBSSD operates on
    INT8 tiles; FP32 would force the kernel back to AVX-512.
    """

    SPEC = ProfileSpec(
        name="amx_only",
        is_baseline=False,
        requires=("amx",),
        axes={
            "isa": "amx_tdpbssd",
            "math_library": "reference",
            "runtime": "vanilla",
            "quantization": "int8_vnni",
            "numa": "unpinned",
            "memory_pages": "4k",
            "core_isolation": "shared",
            "accelerator": "none",
        },
        verify_counters={
            # Tile-active cycles must be a real fraction of busy cycles,
            # else the kernel silently fell back to AVX-512.
            "amx_active_cycle_ratio": 0.05,
        },
        notes="Isolates AMX vs the base ISA. INT8 needed for AMX path.",
    )


class AMXoneDNNProfile(BaseProfileImpl):
    """AMX + tuned GEMM library (oneDNN under IPEX runtime).

    Layered on top of amx_only: same ISA + quantization, but adds
    oneDNN dispatch through Intel Extension for PyTorch. Attribution
    target: oneDNN's contribution beyond the raw ISA.
    """

    SPEC = ProfileSpec(
        name="amx_onednn",
        is_baseline=False,
        requires=("amx",),
        axes={
            "isa": "amx_tdpbssd",
            "math_library": "onednn",
            "runtime": "ipex",
            "quantization": "int8_vnni",
            "numa": "unpinned",
            "memory_pages": "4k",
            "core_isolation": "shared",
            "accelerator": "none",
        },
        verify_counters={
            "amx_active_cycle_ratio": 0.05,
            # Most matmuls should land on oneDNN kernels, not on the
            # framework's fallback path.
            "onednn_kernel_dispatch_ratio": 0.5,
        },
        notes=(
            "Adds oneDNN+IPEX on top of AMX. Attributes the library "
            "contribution separately from raw ISA."
        ),
    )


class AMXNumaHugepagesProfile(BaseProfileImpl):
    """AMX + memory subsystem tuning: NUMA-local + 2M hugepages + isolcpus.

    Layered on top of amx_onednn: keeps the AMX/oneDNN/IPEX/INT8 stack,
    adds the memory-subsystem axes. Attribution target: the
    memory-subsystem contribution at fixed compute config.
    """

    SPEC = ProfileSpec(
        name="amx_numa_hugepages",
        is_baseline=False,
        requires=("amx",),
        axes={
            "isa": "amx_tdpbssd",
            "math_library": "onednn",
            "runtime": "ipex",
            "quantization": "int8_vnni",
            "numa": "numa_local",
            "memory_pages": "hugepages_2m",
            "core_isolation": "isolcpus_affinity",
            "accelerator": "none",
        },
        verify_counters={
            "amx_active_cycle_ratio": 0.05,
            # Most large allocations should land on hugepages, not 4K.
            "hugepage_fault_ratio": 0.5,
            # Cross-socket access should be rare — the _max suffix
            # means "stay below this", not "exceed it".
            "numa_remote_access_ratio_max": 0.15,
        },
        notes=(
            "Adds NUMA-local pinning + 2M hugepages + core isolation. "
            "Isolates the memory-subsystem contribution."
        ),
    )


class FullXeonProfile(BaseProfileImpl):
    """Everything on: INT4-AMX, 1G hugepages, QAT accelerator offload.

    Headline 'Xeon-optimized' arm. Only valid if EVERY counter clears
    its threshold — partial engagement aborts the arm rather than
    reporting under-engaged numbers as if they were the full profile.

    Note: INT4-AMX path requires Granite Rapids (Gen 6) or later for
    full support; on SPR (Gen 4) and EMR (Gen 5) the int4_amx
    quantization will likely fall back to INT8 paths. QAT requires
    a SKU with on-die accelerator.
    """

    SPEC = ProfileSpec(
        name="full_xeon",
        is_baseline=False,
        requires=("amx",),
        axes={
            "isa": "amx_tdpbssd",
            "math_library": "onednn",
            "runtime": "ipex",
            "quantization": "int4_amx",
            "numa": "numa_local",
            "memory_pages": "hugepages_1g",
            "core_isolation": "isolcpus_affinity",
            "accelerator": "qat",
        },
        verify_counters={
            "amx_active_cycle_ratio": 0.05,
            "onednn_kernel_dispatch_ratio": 0.5,
            "hugepage_fault_ratio": 0.5,
            "numa_remote_access_ratio_max": 0.15,
            # QAT actually receiving offloaded work — depth >= 1 means
            # at least one outstanding request was queued during the
            # sample window.
            "accel_queue_depth_qat": 1.0,
        },
        notes=(
            "All axes on. The headline 'Xeon-optimized' arm. Only valid "
            "if EVERY counter clears its threshold."
        ),
    )


class OpenVINOXeonProfile(BaseProfileImpl):
    """Alternative runtime arm: OpenVINO + oneMKL instead of IPEX + oneDNN.

    Lets the study compare IPEX vs OpenVINO as a clean swap with
    everything else equal. AMX still required (OpenVINO uses it).
    Quantization stays at INT8/VNNI; the int4_amx path is IPEX-specific.
    """

    SPEC = ProfileSpec(
        name="openvino_xeon",
        is_baseline=False,
        requires=("amx",),
        axes={
            "isa": "amx_tdpbssd",
            "math_library": "onemkl",
            "runtime": "openvino",
            "quantization": "int8_vnni",
            "numa": "numa_local",
            "memory_pages": "hugepages_2m",
            "core_isolation": "isolcpus_affinity",
            "accelerator": "none",
        },
        verify_counters={
            "amx_active_cycle_ratio": 0.05,
            # Most inferences should run through the OpenVINO path,
            # not framework fallbacks.
            "openvino_infer_request_ratio": 0.9,
            "hugepage_fault_ratio": 0.5,
        },
        notes=(
            "Alternative runtime arm. Lets the study compare IPEX vs "
            "OpenVINO as a clean swap, everything else equal."
        ),
    )


__all__ = [
    "BaseProfile",
    "AMXOnlyProfile",
    "AMXoneDNNProfile",
    "AMXNumaHugepagesProfile",
    "FullXeonProfile",
    "OpenVINOXeonProfile",
]
