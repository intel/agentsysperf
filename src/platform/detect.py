#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
Platform Detection Module
==========================

Auto-discovers hardware capabilities from the running system so that
analyzers, measurements, and preflight checks use real values instead
of hardcoded constants.

Detection sources (in preference order):
1. sysfs / procfs (always available on Linux)
2. lscpu output (fallback parser)
3. MLC baseline (if available — most accurate for BW)
4. Conservative defaults (when detection fails)

Supports: Intel Xeon, AMD EPYC, ARM Neoverse, and generic x86_64/aarch64.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """Detected platform hardware capabilities."""

    # CPU identification
    vendor: str = "unknown"               # GenuineIntel, AuthenticAMD, ARM
    model_name: str = "unknown"
    microarchitecture: str = "unknown"    # GNR, EMR, SPR, Zen4, Neoverse-V2, etc.
    uarch_source: str = "none"            # "cpuid", "sku_string", "none"

    # Raw CPUID identity (x86). 0 when unavailable (e.g. ARM).
    cpu_family: int = 0
    cpu_model: int = 0
    cpu_stepping: int = 0

    # Topology
    physical_cores: int = 0
    logical_cpus: int = 0
    sockets: int = 1
    numa_nodes: int = 1
    cores_per_numa_node: int = 0

    # Cache hierarchy (bytes)
    l1d_per_core: int = 0
    l2_per_core: int = 0
    l3_total: int = 0
    l3_per_node: int = 0                  # L3 available per NUMA node
    l3_usable_per_node: int = 0           # 75% of l3_per_node (safe budget)

    # Memory bandwidth (GB/s)
    dram_bw_per_node_gbs: float = 0.0     # Per-node peak (from MLC or estimate)
    dram_bw_total_gbs: float = 0.0        # Aggregate across all nodes
    dram_bw_source: str = "estimated"     # "mlc", "stream", "estimated", "unknown"

    # Memory latency (ns)
    dram_local_latency_ns: float = 0.0
    dram_remote_latency_ns: float = 0.0
    numa_hop_penalty_ns: float = 0.0

    # Frequency
    base_frequency_ghz: float = 0.0
    max_frequency_ghz: float = 0.0

    # Capabilities
    has_amx: bool = False
    has_avx512: bool = False
    has_avx2: bool = True
    has_numa: bool = False

    # Raw data for debugging
    raw: Dict[str, str] = field(default_factory=dict)

    @property
    def l3_budget_bytes(self) -> int:
        """Safe L3 budget for working set calculations (75% of per-node L3)."""
        return self.l3_usable_per_node

    @property
    def l3_budget_mb(self) -> int:
        """Safe L3 budget in MB."""
        return self.l3_usable_per_node // (1024 * 1024)

    @property
    def dram_bw_is_measured(self) -> bool:
        """True only when peak BW came from a real measurement (MLC/STREAM).

        Analyzers that divide by ``dram_bw_per_node_gbs`` must consult this
        before emitting a utilization-threshold verdict — an estimated peak
        can be off by 3x on an unrecognized platform, and the error is
        one-directional (inflates utilization, so only false positives).
        """
        return self.dram_bw_source in ("mlc", "stream")

    @property
    def uarch_is_known(self) -> bool:
        """True when the microarchitecture was positively identified.

        Prefer this over string comparisons: ``microarchitecture`` carries
        values like ``"Intel (unknown)"`` that are falsely != ``"unknown"``.
        """
        return self.uarch_source in ("cpuid", "sku_string")

    @property
    def is_intel(self) -> bool:
        return self.vendor == "GenuineIntel"

    @property
    def is_amd(self) -> bool:
        return self.vendor == "AuthenticAMD"

    @property
    def is_arm(self) -> bool:
        return "aarch64" in self.vendor.lower() or "arm" in self.vendor.lower()


# Module-level cache
_cached_platform: Optional[PlatformInfo] = None


def detect_platform(*, force_refresh: bool = False) -> PlatformInfo:
    """Detect platform hardware capabilities.

    Results are cached after first call. Use force_refresh=True to re-detect.

    Returns
    -------
    PlatformInfo
        Detected hardware capabilities. Fields that can't be detected
        use conservative defaults.
    """
    global _cached_platform

    if _cached_platform is not None and not force_refresh:
        return _cached_platform

    info = PlatformInfo()

    # Detect in order: CPU → topology → cache → memory → capabilities
    _detect_cpu(info)
    _detect_topology(info)
    _detect_cache(info)
    _detect_memory_bandwidth(info)
    _detect_capabilities(info)
    _detect_frequency(info)

    _cached_platform = info
    bw_str = (
        f"{info.dram_bw_per_node_gbs:.0f} GB/s/node ({info.dram_bw_source})"
        if info.dram_bw_per_node_gbs
        else f"unknown ({info.dram_bw_source})"
    )
    logger.info(
        f"Platform detected: {info.vendor} {info.model_name} "
        f"({info.microarchitecture} via {info.uarch_source}), "
        f"{info.physical_cores} cores, {info.numa_nodes} NUMA nodes, "
        f"L3={info.l3_per_node // (1024*1024)}MB/node, "
        f"BW={bw_str}"
    )

    return info


# ─── Detection Functions ─────────────────────────────────────────────────

def _detect_cpu(info: PlatformInfo) -> None:
    """Detect CPU vendor, model, and microarchitecture."""
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()

        # Key matching must be colon-anchored: `"model" in "model name : ..."`
        # is True, so a substring test silently reads the SKU string into
        # cpu_model and raises ValueError.
        for line in content.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if key == "vendor_id" and info.vendor == "unknown":
                info.vendor = value
            elif key == "model name" and info.model_name == "unknown":
                info.model_name = value
            elif key == "cpu family" and not info.cpu_family:
                info.cpu_family = _safe_int(value)
            elif key == "model" and not info.cpu_model:
                info.cpu_model = _safe_int(value)
            elif key == "stepping" and not info.cpu_stepping:
                info.cpu_stepping = _safe_int(value)
    except IOError:
        pass

    # Identify microarchitecture: CPUID family/model first (works on hosts
    # whose model_name is a generic placeholder, e.g. "Intel(R) Processor"),
    # then fall back to marketing SKU strings.
    info.microarchitecture, info.uarch_source = _identify_microarch(
        info.vendor,
        info.model_name,
        family=info.cpu_family,
        model=info.cpu_model,
    )


def _detect_topology(info: PlatformInfo) -> None:
    """Detect CPU topology: cores, sockets, NUMA nodes."""
    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("CPU(s):"):
                info.logical_cpus = int(line.split(":")[1].strip())
            elif "Core(s) per socket:" in line:
                cores_per_socket = int(line.split(":")[1].strip())
            elif "Socket(s):" in line:
                info.sockets = int(line.split(":")[1].strip())
            elif "NUMA node(s):" in line:
                info.numa_nodes = int(line.split(":")[1].strip())

        info.physical_cores = cores_per_socket * info.sockets
        info.cores_per_numa_node = info.physical_cores // max(info.numa_nodes, 1)
        info.has_numa = info.numa_nodes > 1

    except (subprocess.SubprocessError, ValueError, UnboundLocalError):
        # Fallback: read from /proc/cpuinfo
        try:
            with open("/proc/cpuinfo") as f:
                cores = set()
                for line in f:
                    if "core id" in line:
                        cores.add(line.split(":")[1].strip())
                info.physical_cores = len(cores) if cores else 1
        except IOError:
            info.physical_cores = os.cpu_count() or 1


def _detect_cache(info: PlatformInfo) -> None:
    """Detect cache sizes from sysfs."""
    cache_base = Path("/sys/devices/system/cpu/cpu0/cache")

    if not cache_base.exists():
        # Fallback: try lscpu
        _detect_cache_from_lscpu(info)
        return

    for index_dir in sorted(cache_base.iterdir()):
        if not index_dir.is_dir():
            continue
        try:
            level = (index_dir / "level").read_text().strip()
            type_str = (index_dir / "type").read_text().strip()
            size_str = (index_dir / "size").read_text().strip()

            # Parse size (e.g., "48K", "2048K", "108M")
            size_bytes = _parse_cache_size(size_str)

            if level == "1" and type_str == "Data":
                info.l1d_per_core = size_bytes
            elif level == "2":
                info.l2_per_core = size_bytes
            elif level == "3":
                # L3 shared_cpu_map tells us scope
                info.l3_total = size_bytes * info.sockets
        except (IOError, ValueError):
            continue

    # If L3 wasn't set from sysfs, try lscpu
    if info.l3_total == 0:
        _detect_cache_from_lscpu(info)

    # Compute per-node L3
    if info.l3_total > 0 and info.numa_nodes > 0:
        info.l3_per_node = info.l3_total // info.numa_nodes
    else:
        info.l3_per_node = info.l3_total

    # Safe usable budget: 75% (accounts for OS/contention)
    info.l3_usable_per_node = int(info.l3_per_node * 0.75)


def _detect_cache_from_lscpu(info: PlatformInfo) -> None:
    """Fallback cache detection from lscpu output."""
    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "L1d cache:" in line:
                info.l1d_per_core = _parse_lscpu_cache(line)
            elif "L2 cache:" in line:
                info.l2_per_core = _parse_lscpu_cache(line)
            elif "L3 cache:" in line:
                info.l3_total = _parse_lscpu_cache(line)
    except subprocess.SubprocessError:
        pass


def _detect_memory_bandwidth(info: PlatformInfo) -> None:
    """Detect or estimate DRAM bandwidth.

    Sources (in preference order):
    1. MLC results file (most accurate)
    2. Estimate from DDR generation + channels
    """
    # Try to find MLC results
    mlc_bw = _try_mlc_bandwidth()
    if mlc_bw is not None:
        info.dram_bw_per_node_gbs = mlc_bw / max(info.numa_nodes, 1)
        info.dram_bw_total_gbs = mlc_bw
        info.dram_bw_source = "mlc"
        return

    # Estimate from platform characteristics
    bw_estimate = _estimate_bandwidth(info)
    if bw_estimate is None:
        # Unrecognized platform: leave 0.0 and say so rather than inventing a
        # denominator. Consumers gate on dram_bw_is_measured / a nonzero value.
        info.dram_bw_per_node_gbs = 0.0
        info.dram_bw_total_gbs = 0.0
        info.dram_bw_source = "unknown"
        logger.warning(
            "DRAM peak bandwidth unknown for %s (%s) — bandwidth-utilization "
            "verdicts will be skipped. Provide an MLC baseline or run "
            "`perfspect benchmark --memory` to supply a measured peak.",
            info.model_name,
            info.microarchitecture,
        )
        return

    info.dram_bw_per_node_gbs = bw_estimate / max(info.numa_nodes, 1)
    info.dram_bw_total_gbs = bw_estimate
    info.dram_bw_source = "estimated"


def _detect_capabilities(info: PlatformInfo) -> None:
    """Detect ISA capabilities (AMX, AVX-512, etc.)."""
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()

        flags_line = ""
        for line in content.splitlines():
            if line.startswith("flags"):
                flags_line = line
                break

        info.has_amx = "amx_tile" in flags_line or "amx_bf16" in flags_line
        info.has_avx512 = "avx512f" in flags_line
        info.has_avx2 = "avx2" in flags_line

    except IOError:
        pass


def _detect_frequency(info: PlatformInfo) -> None:
    """Detect CPU frequency."""
    try:
        result = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "CPU max MHz:" in line:
                mhz = float(line.split(":")[1].strip())
                info.max_frequency_ghz = mhz / 1000.0
            elif "CPU MHz:" in line:
                mhz = float(line.split(":")[1].strip())
                if info.base_frequency_ghz == 0:
                    info.base_frequency_ghz = mhz / 1000.0
    except (subprocess.SubprocessError, ValueError):
        pass


# ─── Helper Functions ────────────────────────────────────────────────────

# Intel server CPUID (family, model) -> microarchitecture.
#
# Verified against the kernel's authoritative table,
# arch/x86/include/asm/intel-family.h (linux-hwe-7.0-headers-7.0.0-28), where
# IFM(f, m) encodes family f / model m. Only entries confirmed present in that
# header are listed here.
#
# Deliberately absent: one next-generation server part. The 6.17 and 7.0 headers
# agree on its IFM value but disagree on the macro *name*, so it stays out until
# confirmed on real silicon — see the no-fabricated-codes rule in the README. An
# unlisted part falls through to the SKU-string path and then to
# "Xeon (unknown gen)", which is honest rather than wrong.
_INTEL_CPUID_UARCH: Dict[tuple, str] = {
    (6, 0x55): "Skylake (SKX)",              # also Cascade Lake; stepping-dependent
    (6, 0x6A): "Ice Lake (ICX)",
    (6, 0x6C): "Ice Lake (ICX)",             # Ice Lake-D
    (6, 0x8F): "Sapphire Rapids (SPR)",
    (6, 0xCF): "Emerald Rapids (EMR)",
    (6, 0xAD): "Granite Rapids (GNR)",
    (6, 0xAE): "Granite Rapids (GNR)",       # Granite Rapids-D
    (6, 0xAF): "Sierra Forest (SRF)",        # Crestmont E-cores
    (6, 0xDD): "Clearwater Forest (CWF)",    # Darkmont E-cores
}


def _safe_int(value: str) -> int:
    """Parse an integer from /proc/cpuinfo, returning 0 on garbage."""
    try:
        return int(value)
    except ValueError:
        return 0


def _identify_microarch(
    vendor: str,
    model_name: str,
    *,
    family: int = 0,
    model: int = 0,
) -> tuple:
    """Identify CPU microarchitecture.

    Returns
    -------
    (microarchitecture, source)
        ``source`` is "cpuid" when resolved from CPUID family/model,
        "sku_string" when resolved from the marketing model name, or
        "none" when unresolved.

    Notes
    -----
    The returned string is consumed by lowercase substring matching
    (``"granite" in uarch``) in ``_estimate_bandwidth`` and
    ``measurements/emon/probe.py``, so the ``"Name (ABBR)"`` shape is
    load-bearing — do not change it without updating those call sites.
    """
    model_lower = model_name.lower()

    # CPUID first: reliable on parts whose model_name carries no marketing
    # SKU and on any SKU absent from the tables below.
    if vendor == "GenuineIntel" and family:
        uarch = _INTEL_CPUID_UARCH.get((family, model))
        if uarch:
            return uarch, "cpuid"

    if vendor == "GenuineIntel":
        intel_map = [
            ("6979", "Granite Rapids (GNR)"),
            ("6973", "Granite Rapids (GNR)"),
            ("6972", "Granite Rapids (GNR)"),
            ("6787", "Granite Rapids (GNR)"),
            ("8592", "Emerald Rapids (EMR)"),
            ("8580", "Emerald Rapids (EMR)"),
            ("8490", "Sapphire Rapids (SPR)"),
            ("8480", "Sapphire Rapids (SPR)"),
            ("8380", "Ice Lake (ICX)"),
            ("8360", "Ice Lake (ICX)"),
            ("8280", "Cascade Lake (CLX)"),
            ("8180", "Skylake (SKX)"),
        ]
        for sku, uarch in intel_map:
            if sku in model_lower:
                return uarch, "sku_string"
        if "xeon" in model_lower:
            return "Xeon (unknown gen)", "none"
        return "Intel (unknown)", "none"

    elif vendor == "AuthenticAMD":
        amd_map = [
            ("9754", "Zen 4c (Bergamo)"),
            ("9654", "Zen 4 (Genoa)"),
            ("9554", "Zen 4 (Genoa)"),
            ("9474", "Zen 4 (Genoa)"),
            ("7763", "Zen 3 (Milan)"),
            ("7713", "Zen 3 (Milan)"),
            ("7R13", "Zen 3 (Milan)"),
            ("9B14", "Zen 5 (Turin)"),
        ]
        for sku, uarch in amd_map:
            if sku.lower() in model_lower:
                return uarch, "sku_string"
        if "epyc" in model_lower:
            return "EPYC (unknown gen)", "none"
        return "AMD (unknown)", "none"

    elif "aarch64" in model_lower or "arm" in vendor.lower() or "neoverse" in model_lower:
        if "v2" in model_lower:
            return "Neoverse V2 (Grace)", "sku_string"
        elif "v1" in model_lower:
            return "Neoverse V1 (Graviton3)", "sku_string"
        elif "n2" in model_lower:
            return "Neoverse N2 (Graviton3E)", "sku_string"
        return "ARM (unknown)", "none"

    return "unknown", "none"


def _parse_cache_size(size_str: str) -> int:
    """Parse cache size string (e.g., '48K', '2048K', '108M') to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith("K"):
        return int(size_str[:-1]) * 1024
    elif size_str.endswith("M"):
        return int(size_str[:-1]) * 1024 * 1024
    elif size_str.endswith("G"):
        return int(size_str[:-1]) * 1024 * 1024 * 1024
    return int(size_str)


def _parse_lscpu_cache(line: str) -> int:
    """Parse lscpu cache line (e.g., 'L3 cache: 108 MiB (3 instances)')."""
    value_part = line.split(":")[1].strip()
    # Remove instance info
    value_part = value_part.split("(")[0].strip()
    # Parse number and unit
    match = re.match(r"([\d.]+)\s*(KiB|MiB|GiB|KB|MB|GB|K|M|G)?", value_part)
    if not match:
        return 0

    number = float(match.group(1))
    unit = (match.group(2) or "").upper()

    if unit in ("KIB", "KB", "K"):
        return int(number * 1024)
    elif unit in ("MIB", "MB", "M"):
        return int(number * 1024 * 1024)
    elif unit in ("GIB", "GB", "G"):
        return int(number * 1024 * 1024 * 1024)
    return int(number)


def _try_mlc_bandwidth() -> Optional[float]:
    """Try to read MLC max bandwidth results.

    Looks for MLC output in common locations:
    - ./perf-runs/*/stage-2-mlc/
    - /tmp/mlc_results/
    """
    mlc_paths = [
        Path("perf-runs"),
        Path(f"{_TMP}/mlc_results"),
        Path.home() / "mlc_results",
    ]

    for base in mlc_paths:
        if not base.exists():
            continue
        for f in base.rglob("*bandwidth*"):
            bw = _parse_mlc_bandwidth_file(f)
            if bw is not None:
                return bw

    return None


# MLC "peak injection bandwidth" rows, e.g.
#   ALL Reads        :      230450.5
# Anchored on a labelled row ending in a number, so that unrelated lines
# containing "all" (notably "Installed memory") cannot be mistaken for a
# result. MLC reports MB/s.
_MLC_BW_RE = re.compile(
    r"^\s*(?:ALL\s+Reads|ALL\s+Coverage|Peak\s+Injection\s+Bandwidth)\b[^:]*:\s*"
    r"([\d.]+)\s*$",
    re.IGNORECASE,
)


def _parse_mlc_bandwidth_file(filepath: Path) -> Optional[float]:
    """Parse MLC bandwidth output for peak read BW, in GB/s.

    Returns None rather than a guess when nothing matches: the caller stamps
    the result ``dram_bw_source="mlc"``, so a loose match would launder an
    arbitrary number from the file into a field that claims to be measured.
    """
    try:
        content = filepath.read_text()
    except (IOError, UnicodeDecodeError):
        return None

    for line in content.splitlines():
        match = _MLC_BW_RE.match(line)
        if not match:
            continue
        try:
            mb_per_s = float(match.group(1))
        except ValueError:
            continue
        gb_per_s = mb_per_s / 1000.0  # MLC reports MB/s; this field is GB/s
        # Plausibility band for a server socket: below 10 GB/s means we
        # matched the wrong field, above 100 TB/s means a unit mismatch.
        if 10.0 <= gb_per_s <= 100_000.0:
            return gb_per_s

    return None


def _estimate_bandwidth(info: PlatformInfo) -> Optional[float]:
    """Estimate peak DRAM bandwidth from platform characteristics.

    Conservative estimates based on DDR generation and channel count.
    These are THEORETICAL peaks — actual sustained BW is typically 70-85%.

    Returns None when the platform is not recognized. A fabricated number is
    worse than no number: callers divide by this to produce
    utilization-threshold verdicts, and a too-low peak inflates utilization,
    biasing toward false "bandwidth saturated" conclusions. Returning None
    lets those callers skip the verdict instead of emitting a wrong one.
    """
    if info.vendor == "GenuineIntel":
        uarch = info.microarchitecture.lower()
        if "granite" in uarch or "gnr" in uarch:
            # GNR: DDR5-5600, 8 channels/socket typical
            return 358.0 * info.sockets  # 8ch × 5600MT/s × 8B = 358 GB/s/socket
        elif "emerald" in uarch or "emr" in uarch:
            return 307.0 * info.sockets  # 8ch × 4800MT/s × 8B
        elif "sapphire" in uarch or "spr" in uarch:
            return 307.0 * info.sockets
        elif "ice" in uarch or "icx" in uarch:
            return 204.0 * info.sockets  # 8ch × 3200MT/s × 8B
        elif "clearwater" in uarch or "cwf" in uarch:
            # 12ch × 6400MT/s × 8B. Channel count from the 12 uncore_imc PMUs
            # exposed by the kernel; the 6400MT/s from uncore_imc_0/clockticks
            # measured at 798.1MHz (x8 = 6385MT/s) on a CWF host, 2026-07-28.
            return 614.0 * info.sockets
        elif "sierra" in uarch or "srf" in uarch:
            return 550.0 * info.sockets  # 12ch × 5600MT/s × 8B
        else:
            return None

    elif info.vendor == "AuthenticAMD":
        uarch = info.microarchitecture.lower()
        if "zen 5" in uarch or "turin" in uarch:
            return 461.0 * info.sockets  # 12ch × 4800MT/s × 8B
        elif "zen 4" in uarch or "genoa" in uarch:
            return 461.0 * info.sockets  # 12ch × 4800MT/s × 8B
        elif "zen 3" in uarch or "milan" in uarch:
            return 204.0 * info.sockets  # 8ch × 3200MT/s × 8B
        else:
            return None

    return None


__all__ = ["PlatformInfo", "detect_platform"]
