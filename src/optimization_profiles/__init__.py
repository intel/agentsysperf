#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Reference :class:`OptimizationProfilePlugin` implementations.

Six profiles ported from ``docs/contracts/optimization_profiles.py``.
``base`` is the portable reference (FP32, AVX-512, unpinned, 4K pages);
the rest layer Xeon-specific axes (AMX, oneDNN/IPEX, NUMA-local pinning,
2M/1G hugepages, isolcpus, INT8/INT4 quantization, QAT offload, OpenVINO
runtime) so a measured speedup can be ATTRIBUTED to a specific axis.

Each profile declares the hardware counters that PROVE it engaged plus a
minimum threshold (or maximum, for ``_max``-suffixed counters). At
verify_engaged time the plugin reads those counters via a
:class:`HardwareTelemetryPlugin` and either reports engaged=True or
returns engaged=False with concrete failure reasons.

Hardware availability (informational; relevant for engagement on real
silicon, not for code correctness):

- ``base`` — every Xeon. Empty verify_counters; trivially engaged.
- ``amx_only`` / ``amx_onednn`` — needs SPR (Gen 4, 2023) or later.
- ``amx_numa_hugepages`` — SPR+ for AMX; NUMA/hugepage axes are OS-level.
- ``full_xeon`` — needs Granite Rapids (Gen 6) or later for INT4-AMX,
  plus a SKU with on-die QAT for the accelerator axis.
- ``openvino_xeon`` — SPR+ for AMX; OpenVINO runtime is software.

On hosts that can't satisfy a profile (Ice Lake / Cascade Lake without
AMX, or any host whose installed HardwareTelemetryPlugin doesn't
advertise the required vendor counters), verify_engaged() returns
engaged=False with explicit failure reasons. That is the
abort-don't-degrade path: profiles fail loudly, never silently.
"""

from src.optimization_profiles.profiles import (
    AMXNumaHugepagesProfile,
    AMXoneDNNProfile,
    AMXOnlyProfile,
    BaseProfile,
    FullXeonProfile,
    OpenVINOXeonProfile,
)

__all__ = [
    "AMXNumaHugepagesProfile",
    "AMXoneDNNProfile",
    "AMXOnlyProfile",
    "BaseProfile",
    "FullXeonProfile",
    "OpenVINOXeonProfile",
]
