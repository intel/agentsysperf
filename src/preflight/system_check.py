#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
System Pre-flight Checks for AgentSysPerf
========================================

Validates that a host is properly configured for CPU performance
measurement before running benchmarks.

Each check's pass/fail boundary is taken from the kernel documentation for
the interface it reads:

- ``kernel.perf_event_paranoid`` — Linux ``Documentation/admin-guide/sysctl/kernel.rst``
  and ``Documentation/admin-guide/perf-security.rst``
- ``scaling_governor`` — Linux ``Documentation/admin-guide/pm/cpufreq.rst``
- NUMA topology and tool availability — ``numactl(8)``, ``perf-list(1)``,
  ``cpupower(1)``

Check matrix covers:
- CPU vendor and microarchitecture detection
- Kernel configuration (perf_event_paranoid, MSR, NMI watchdog)
- Required tools (numactl, perf, cpupower)
- Governor setting (must be 'performance' for stable measurements)
- NUMA topology awareness

Design principle: Check-only by default. Report what's present, what's
missing, and what a fix would look like. Never apply changes without
explicit user request.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.platform import PlatformInfo, detect_platform


class CheckStatus(Enum):
    OK = "OK"
    MISSING = "MISSING"
    MISCONFIGURED = "MISCONFIGURED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""

    name: str
    category: str
    status: CheckStatus
    current_value: str
    expected_value: str
    fix_command: str = ""
    notes: str = ""
    # True only for fix_commands that genuinely need shell metacharacters
    # (pipes, command substitution). `agentsysperf preflight --fix` runs
    # everything else with shell=False, so a command that interpolates a
    # runtime value cannot be turned into shell injection. Keep this False
    # unless the command cannot be expressed as an argv list.
    fix_needs_shell: bool = False

    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.OK, CheckStatus.WARNING, CheckStatus.SKIPPED)


@dataclass
class SystemReport:
    """Aggregate report from all pre-flight checks."""

    checks: List[CheckResult] = field(default_factory=list)
    platform_info: Dict[str, str] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """True if all critical checks pass (no MISSING or MISCONFIGURED)."""
        return all(
            c.status not in (CheckStatus.MISSING, CheckStatus.MISCONFIGURED)
            for c in self.checks
            if c.category != "optional"
        )

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> str:
        """Human-readable summary of check results."""
        lines = []
        lines.append("=" * 70)
        lines.append("AgentSysPerf Pre-flight System Check")
        lines.append("=" * 70)

        # Platform info
        if self.platform_info:
            lines.append(f"\nPlatform: {self.platform_info.get('cpu_model', 'Unknown')}")
            lines.append(f"Kernel:   {self.platform_info.get('kernel', 'Unknown')}")
            lines.append(f"OS:       {self.platform_info.get('os', 'Unknown')}")

        # Results table
        lines.append(f"\n{'Check':<35} {'Status':<15} {'Current':<20} {'Expected':<20}")
        lines.append("-" * 90)

        for check in self.checks:
            status_str = check.status.value
            if check.status == CheckStatus.OK:
                marker = "✓"
            elif check.status == CheckStatus.WARNING:
                marker = "⚠"
            else:
                marker = "✗"

            lines.append(
                f"{marker} {check.name:<33} {status_str:<15} "
                f"{check.current_value:<20} {check.expected_value:<20}"
            )

        # Summary
        lines.append("-" * 90)
        lines.append(
            f"Result: {self.passed_count}/{len(self.checks)} passed, "
            f"{self.failed_count} issues"
        )

        if not self.ready:
            lines.append("\nFixes needed:")
            for check in self.checks:
                if not check.passed and check.fix_command:
                    lines.append(f"  {check.name}:")
                    lines.append(f"    {check.fix_command}")

        lines.append("=" * 70)
        return "\n".join(lines)


class SystemCheck:
    """Pre-benchmark system validation.

    Covers the minimum requirements for AgentSysPerf's measurement plugins
    to function correctly. See the module docstring for the documentation
    each check's boundary comes from.

    Parameters
    ----------
    require_perf : bool, default=True
        Whether perf tool availability is required (for L3 measurements).
    require_numa : bool, default=True
        Whether NUMA tools are required.
    require_perfspect : bool, default=False
        Whether PerfSpect binary is required.

    Examples
    --------
    >>> check = SystemCheck()
    >>> report = check.run()
    >>> print(report.summary())
    >>> if not report.ready:
    ...     print("System not ready for benchmarking")
    """

    def __init__(
        self,
        *,
        require_perf: bool = True,
        require_numa: bool = True,
        require_perfspect: bool = False,
        platform_info: Optional[PlatformInfo] = None,
    ) -> None:
        self._require_perf = require_perf
        self._require_numa = require_numa
        self._require_perfspect = require_perfspect
        self._platform = platform_info or detect_platform()

    def run(self) -> SystemReport:
        """Run all pre-flight checks and return report."""
        report = SystemReport()
        report.platform_info = self._collect_platform_info()

        # CPU checks
        report.checks.append(self._check_cpu_vendor())
        report.checks.append(self._check_microarch())

        # Kernel configuration
        report.checks.append(self._check_perf_paranoid())
        report.checks.append(self._check_nmi_watchdog())
        report.checks.append(self._check_msr_module())

        # Governor
        report.checks.append(self._check_governor())

        # Required tools
        report.checks.append(self._check_perf_tool())
        report.checks.append(self._check_numa_tools())

        # Optional tools
        report.checks.append(self._check_perfspect())
        report.checks.append(self._check_observability_tools())

        # Kernel headers
        report.checks.append(self._check_kernel_headers())

        return report

    # ─── Platform Info ───────────────────────────────────────────────

    def _collect_platform_info(self) -> Dict[str, str]:
        """Collect basic platform information."""
        info = {}

        # CPU model
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
        except IOError:
            info["cpu_model"] = "Unknown"

        # Kernel
        info["kernel"] = platform.release()

        # OS
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME"):
                        info["os"] = line.split("=")[1].strip().strip('"')
                        break
        except IOError:
            info["os"] = platform.platform()

        # NUMA nodes
        try:
            result = subprocess.run(
                ["numactl", "--hardware"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if "available" in line:
                    info["numa_nodes"] = line.split(":")[1].strip().split()[0]
                    break
        except (subprocess.SubprocessError, FileNotFoundError):
            info["numa_nodes"] = "unknown"

        return info

    # ─── Individual Checks ───────────────────────────────────────────

    def _check_cpu_vendor(self) -> CheckResult:
        """Check CPU vendor (Intel required for emon/vtune/perfspect/mlc)."""
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "vendor_id" in line:
                        vendor = line.split(":")[1].strip()
                        if vendor == "GenuineIntel":
                            return CheckResult(
                                name="CPU vendor",
                                category="cpu",
                                status=CheckStatus.OK,
                                current_value=vendor,
                                expected_value="GenuineIntel",
                            )
                        else:
                            return CheckResult(
                                name="CPU vendor",
                                category="cpu",
                                status=CheckStatus.WARNING,
                                current_value=vendor,
                                expected_value="GenuineIntel",
                                notes="Non-Intel CPU: emon/vtune/mlc/perfspect unavailable; perf works",
                            )
        except IOError:
            pass

        return CheckResult(
            name="CPU vendor",
            category="cpu",
            status=CheckStatus.WARNING,
            current_value="unknown",
            expected_value="GenuineIntel",
        )

    def _check_microarch(self) -> CheckResult:
        """Detect CPU microarchitecture."""
        uarch = self._platform.microarchitecture
        model = self._platform.model_name

        # Gate on uarch_source, not `uarch != "unknown"`: detection returns
        # sentinel strings like "Intel (unknown)" and "Xeon (unknown gen)" that
        # are literally != "unknown", so the old test reported OK on exactly
        # the hosts where identification had failed.
        if self._platform.uarch_is_known:
            notes = model
            if self._platform.cpu_family:
                notes = (
                    f"{model} [family {self._platform.cpu_family} "
                    f"model {self._platform.cpu_model} "
                    f"stepping {self._platform.cpu_stepping}] "
                    f"via {self._platform.uarch_source}"
                )
            return CheckResult(
                name="Microarchitecture",
                category="cpu",
                status=CheckStatus.OK,
                current_value=uarch,
                expected_value="Known microarchitecture",
                notes=notes,
            )

        notes = (
            f"{model} — unrecognized; DRAM bandwidth peak is unavailable, so "
            f"bandwidth-utilization verdicts will be skipped"
        )
        if self._platform.cpu_family:
            notes = (
                f"{model} [family {self._platform.cpu_family} "
                f"model {self._platform.cpu_model}] not in the CPUID table — "
                f"DRAM bandwidth peak unavailable, so bandwidth-utilization "
                f"verdicts will be skipped"
            )
        return CheckResult(
            name="Microarchitecture",
            category="cpu",
            status=CheckStatus.WARNING,
            current_value=uarch,
            expected_value="Known microarchitecture",
            notes=notes,
        )

    def _check_perf_paranoid(self) -> CheckResult:
        """Check kernel.perf_event_paranoid setting.

        Boundaries follow the values documented in Linux
        ``Documentation/admin-guide/sysctl/kernel.rst``:

        - ``>= 2`` disallows kernel profiling, so TMA/uncore collection fails
          outright — reported as misconfigured.
        - ``>= 1`` disallows CPU-wide event access; per-process profiling still
          works, system-wide collection does not — reported as a warning.
        - ``<= 0`` permits the CPU-wide collection perfspect and emon use.
        """
        try:
            with open("/proc/sys/kernel/perf_event_paranoid") as f:
                value = int(f.read().strip())

            if value <= 0:
                status = CheckStatus.OK
            elif value == 1:
                status = CheckStatus.WARNING
            else:
                status = CheckStatus.MISCONFIGURED

            return CheckResult(
                name="perf_event_paranoid",
                category="kernel",
                status=status,
                current_value=str(value),
                expected_value="<= 0 (ideal), <= 1 (minimum)",
                fix_command="sudo sysctl kernel.perf_event_paranoid=0",
                notes="Controls PMU counter access for non-root users",
            )
        except (IOError, ValueError):
            return CheckResult(
                name="perf_event_paranoid",
                category="kernel",
                status=CheckStatus.MISSING,
                current_value="unreadable",
                expected_value="<= 1",
                fix_command="sudo sysctl kernel.perf_event_paranoid=0",
            )

    def _check_nmi_watchdog(self) -> CheckResult:
        """Check NMI watchdog (steals a PMU counter when enabled)."""
        try:
            with open("/proc/sys/kernel/nmi_watchdog") as f:
                value = int(f.read().strip())

            if value == 0:
                status = CheckStatus.OK
            else:
                status = CheckStatus.WARNING

            return CheckResult(
                name="NMI watchdog",
                category="kernel",
                status=status,
                current_value=str(value),
                expected_value="0 (frees a PMU counter)",
                fix_command="sudo sysctl kernel.nmi_watchdog=0",
                notes="Enabled NMI watchdog consumes one PMU counter",
            )
        except (IOError, ValueError):
            return CheckResult(
                name="NMI watchdog",
                category="kernel",
                status=CheckStatus.SKIPPED,
                current_value="unreadable",
                expected_value="0",
            )

    def _check_msr_module(self) -> CheckResult:
        """Check if MSR kernel module is loaded (needed by mlc, perfspect)."""
        msr_loaded = Path("/dev/cpu/0/msr").exists()

        if msr_loaded:
            return CheckResult(
                name="MSR module",
                category="kernel",
                status=CheckStatus.OK,
                current_value="loaded",
                expected_value="loaded",
            )
        else:
            return CheckResult(
                name="MSR module",
                category="kernel",
                status=CheckStatus.WARNING,
                current_value="not loaded",
                expected_value="loaded",
                fix_command="sudo modprobe msr",
                notes="Required by mlc, some perfspect modes",
            )

    def _check_governor(self) -> CheckResult:
        """Check CPU frequency governor.

        The ``performance`` governor holds the policy at its maximum frequency
        rather than varying it with load (Linux
        ``Documentation/admin-guide/pm/cpufreq.rst``). Any load-varying
        governor moves the clock underneath a measurement, so run-to-run
        comparisons stop being repeatable regardless of workload.
        """
        governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")

        if not governor_path.exists():
            return CheckResult(
                name="CPU governor",
                category="system",
                status=CheckStatus.SKIPPED,
                current_value="no cpufreq",
                expected_value="performance",
                notes="cpufreq not available (fixed frequency or VM)",
            )

        try:
            governor = governor_path.read_text().strip()
            if governor == "performance":
                status = CheckStatus.OK
            else:
                status = CheckStatus.MISCONFIGURED

            return CheckResult(
                name="CPU governor",
                category="system",
                status=status,
                current_value=governor,
                expected_value="performance",
                fix_command="sudo cpupower frequency-set -g performance",
                notes="Non-performance governor causes frequency jitter in measurements",
            )
        except IOError:
            return CheckResult(
                name="CPU governor",
                category="system",
                status=CheckStatus.SKIPPED,
                current_value="unreadable",
                expected_value="performance",
            )

    def _check_perf_tool(self) -> CheckResult:
        """Check if Linux perf tool is available."""
        perf_path = shutil.which("perf")
        category = "tools" if self._require_perf else "optional"

        if perf_path:
            return CheckResult(
                name="perf tool",
                category=category,
                status=CheckStatus.OK,
                current_value=perf_path,
                expected_value="on PATH",
            )
        else:
            status = CheckStatus.MISSING if self._require_perf else CheckStatus.WARNING
            return CheckResult(
                name="perf tool",
                category=category,
                status=status,
                current_value="not found",
                expected_value="on PATH",
                fix_command="sudo apt install linux-tools-$(uname -r)",
                fix_needs_shell=True,  # command substitution
                notes="Required for L3 hardware counter measurements",
            )

    def _check_numa_tools(self) -> CheckResult:
        """Check NUMA tools availability."""
        numactl = shutil.which("numactl")
        category = "tools" if self._require_numa else "optional"

        if numactl:
            return CheckResult(
                name="NUMA tools",
                category=category,
                status=CheckStatus.OK,
                current_value="numactl available",
                expected_value="numactl on PATH",
            )
        else:
            status = CheckStatus.MISSING if self._require_numa else CheckStatus.WARNING
            return CheckResult(
                name="NUMA tools",
                category=category,
                status=status,
                current_value="not found",
                expected_value="numactl on PATH",
                fix_command="sudo apt install numactl",
                notes="Required for NUMA-aware benchmarking",
            )

    def _check_perfspect(self) -> CheckResult:
        """Check PerfSpect availability."""
        category = "tools" if self._require_perfspect else "optional"

        # Same search order as PerfSpectMeasurement._find_perfspect
        candidates = [
            shutil.which("perfspect"),
            Path.home() / "perfspect" / "perfspect",
            Path.home() / "dev" / "perfspect" / "perfspect",
        ]

        for candidate in candidates:
            if candidate and (isinstance(candidate, str) or candidate.exists()):
                path_str = str(candidate)
                return CheckResult(
                    name="PerfSpect",
                    category=category,
                    status=CheckStatus.OK,
                    current_value=path_str,
                    expected_value="perfspect binary",
                )

        status = CheckStatus.MISSING if self._require_perfspect else CheckStatus.WARNING
        return CheckResult(
            name="PerfSpect",
            category=category,
            status=status,
            current_value="not found",
            expected_value="~/perfspect/perfspect or on PATH",
            fix_command="wget -qO- https://github.com/intel/PerfSpect/releases/latest/download/perfspect.tgz | tar xz",
            fix_needs_shell=True,  # pipeline
            notes="Optional but recommended for TMA analysis",
        )

    def _check_observability_tools(self) -> CheckResult:
        """Check baseline observability tools."""
        tools = ["htop", "iostat", "mpstat", "vmstat"]
        present = [t for t in tools if shutil.which(t)]
        missing = [t for t in tools if t not in present]

        if not missing:
            return CheckResult(
                name="Observability tools",
                category="optional",
                status=CheckStatus.OK,
                current_value=f"{len(present)}/{len(tools)} present",
                expected_value="htop, iostat, mpstat, vmstat",
            )
        else:
            return CheckResult(
                name="Observability tools",
                category="optional",
                status=CheckStatus.WARNING,
                current_value=f"missing: {', '.join(missing)}",
                expected_value="htop, iostat, mpstat, vmstat",
                fix_command="sudo apt install sysstat htop",
            )

    def _check_kernel_headers(self) -> CheckResult:
        """Check kernel headers match running kernel."""
        kernel_release = platform.release()
        headers_path = Path(f"/lib/modules/{kernel_release}/build")

        if headers_path.exists():
            return CheckResult(
                name="Kernel headers",
                category="optional",
                status=CheckStatus.OK,
                current_value=f"present ({kernel_release})",
                expected_value=f"match {kernel_release}",
            )
        else:
            return CheckResult(
                name="Kernel headers",
                category="optional",
                status=CheckStatus.WARNING,
                current_value="not found",
                expected_value=f"/lib/modules/{kernel_release}/build",
                fix_command=f"sudo apt install linux-headers-{kernel_release}",
                notes="Needed for SEP driver build (emon/vtune)",
            )

    # Microarch detection delegated to src.platform.detect module


__all__ = ["SystemCheck", "CheckResult", "SystemReport"]
