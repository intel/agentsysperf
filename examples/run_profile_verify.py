#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Demonstrate OptimizationProfilePlugin engagement-verify on this host.

For each discovered profile:
- print the projection produced by ``apply()``
- call ``verify_engaged()`` against the installed perf_stat telemetry
- show the engaged verdict + any failure reasons

On a host without AMX-aware telemetry (i.e. only the default
:class:`PerfStatTelemetry` is installed), every amx_* profile should
fail engagement honestly with "counter not advertised by any installed
HardwareTelemetryPlugin." That is the abort-don't-degrade machinery
working correctly — not a bug.

Run::

    python examples/run_profile_verify.py
"""

from __future__ import annotations

import json
import textwrap

from src.protocols import (
    discover_hardware_telemetry,
    discover_optimization_profiles,
)


def _print_section(title: str) -> None:
    bar = "─" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    profiles = discover_optimization_profiles()
    telemetry_plugins = discover_hardware_telemetry()

    if not telemetry_plugins:
        print("No HardwareTelemetryPlugin installed; verify_engaged would fail.")
        return

    # Use perf_stat as the active telemetry source — that's the one
    # we ship today. Vendor-aware plugins (intel_pcm) are future work.
    telemetry = telemetry_plugins.get("perf_stat") or next(iter(telemetry_plugins.values()))
    print(f"Using telemetry plugin: {telemetry.name}")
    print(f"Available events: {sorted(telemetry.available_events)}")

    sample_sut_config = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "endpoint": "http://localhost:8000/v1",
    }

    for profile_name, profile in sorted(profiles.items()):
        _print_section(f"{profile_name}  (is_baseline={profile.is_baseline})")
        print(f"axes:           {dict(profile.SPEC.axes)}")
        print(f"verify_counters:{dict(profile.SPEC.verify_counters)}")
        if profile.SPEC.notes:
            print("notes:          " + textwrap.fill(
                profile.SPEC.notes, width=64,
                subsequent_indent=" " * 16,
            ))

        applied = profile.apply(sut_config=sample_sut_config)
        print(f"\napply() -> profile={applied['profile']}")
        print(f"  backend_config: {applied['backend_config']}")
        print(f"  scheduler_config: {applied['scheduler_config']}")

        verdict = profile.verify_engaged(
            telemetry=telemetry, run_id="profile-verify-demo",
        )
        print(f"\nverify_engaged() -> engaged={verdict['engaged']}")
        if verdict["counters"]:
            print(f"  counters read:  {verdict['counters']}")
        if verdict["thresholds"]:
            print(f"  thresholds:     {verdict['thresholds']}")
        if verdict["failures"]:
            print("  failures:")
            for f in verdict["failures"]:
                print(textwrap.fill(
                    f, width=68, initial_indent="    - ",
                    subsequent_indent="      ",
                ))


if __name__ == "__main__":
    main()
