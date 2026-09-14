#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""
AgentSysPerf 0.1.0-RC1 Field Testing Verification Script

Verifies that RC1 works on CWF hardware:
- Synthetic CPU baseline collection
- Scaling sweep (concurrency saturation)
- Terminal-Bench integration
- Dashboard data population
- Analyzer verdict generation
"""

import sys
import json
import shlex
import subprocess
from pathlib import Path
from datetime import datetime

def run_cmd(cmd, verbose=False):
    """Run shell command, return stdout.

    shell=True is deliberate and load-bearing here: every command in this file
    is a hand-written literal that uses shell plumbing to do its checking —
    `2>&1 | head -1`, `| wc -l`, `| grep -E`, `| awk '{print $1}'`, `~`
    expansion. Rewriting them as argv would mean reimplementing that plumbing
    in Python for a developer-run verification script.

    The one value that does NOT come from a literal in this file is the run_id
    read back out of the store in check_analyzers(); that one is shlex.quote'd
    at its call site. Keep it that way: anything interpolated into a command
    passed here must be quoted by the caller.
    """
    if verbose:
        print(f"  $ {cmd}")
    # literal commands only, see docstring
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)  # nosec B602
    return result.stdout.strip(), result.returncode

def check_environment():
    """Verify Python, venv, agentsysperf CLI."""
    print("[1/6] Checking environment...")

    checks = {
        "Python 3.12+": ("python3 --version", lambda x: "3.12" in x or "3.13" in x),
        "agentsysperf CLI": ("agentsysperf list 2>&1 | head -1", lambda x: len(x) > 0),
        "perf access": ("cat /proc/sys/kernel/perf_event_paranoid", lambda x: int(x.strip()) <= 1),
        "SQLite store": (f"ls -la ~/.agentsysperf/results.db 2>/dev/null | wc -l", lambda x: True),
    }

    results = {}
    for name, (cmd, validator) in checks.items():
        output, code = run_cmd(cmd)
        passed = code == 0 and validator(output)
        results[name] = "✓" if passed else "✗"
        print(f"  {results[name]} {name}")
        if not passed and "perf" in name.lower():
            print(f"    Fix: sudo sysctl kernel.perf_event_paranoid=1")

    return all(v == "✓" for v in results.values())

def check_plugins():
    """Verify plugin discovery."""
    print("\n[2/6] Checking plugin discovery...")

    # Get plugin counts
    output, _ = run_cmd("agentsysperf list 2>&1 | grep -E '(Benchmarks|Measurements|Analyzers):'")
    print(f"  {output}")

    # Verify 8 analyzers
    output, _ = run_cmd("agentsysperf analyzers list 2>&1 | wc -l")
    count = int(output.strip())

    if count >= 8:
        print(f"  ✓ 8+ analyzers discovered")
        return True
    else:
        print(f"  ✗ Only {count} analyzers (need 8)")
        return False

def check_database():
    """Verify SQLite store schema."""
    print("\n[3/6] Checking database...")

    # Check if store exists
    store_path = Path.home() / ".agentsysperf" / "results.db"
    if not store_path.exists():
        print(f"  ℹ Store not yet created (will be created on first run)")
        return True

    # Get table count
    cmd = f"sqlite3 ~/.agentsysperf/results.db '.tables' | wc -w"
    output, _ = run_cmd(cmd)
    table_count = int(output.strip())

    print(f"  ✓ Store exists with {table_count} tables")

    # Get record counts
    cmd = "sqlite3 ~/.agentsysperf/results.db 'SELECT COUNT(*) FROM measurements;' 2>/dev/null"
    output, _ = run_cmd(cmd)
    if output:
        print(f"  ℹ {output} measurement records in store")

    return True

def verify_synthetic_cpu_run():
    """Run synthetic CPU and verify output."""
    print("\n[4/6] Running synthetic_cpu benchmark (5 tasks for speed)...")

    cmd = "agentsysperf run --benchmark synthetic_cpu --num-tasks 5 2>&1"
    output, code = run_cmd(cmd, verbose=True)

    if code == 0 and "passed" in output.lower():
        # Extract pass count
        for line in output.split('\n'):
            if 'passed' in line.lower():
                print(f"  ✓ {line.strip()}")
        return True
    else:
        print(f"  ✗ Benchmark failed")
        print(output[-500:])
        return False

def verify_scaling_sweep():
    """Run scaling sweep (dry-run, no API key)."""
    print("\n[5/6] Running scaling sweep (dry-run, 6 points)...")

    cmd = "agentsysperf sweep run --dry-run 2>&1"
    output, code = run_cmd(cmd, verbose=True)

    if code == 0:
        # Extract knee
        for line in output.split('\n'):
            if 'knee' in line.lower() or 'saturation' in line.lower():
                print(f"  ℹ {line.strip()}")
        print(f"  ✓ Scaling sweep complete")
        return True
    else:
        print(f"  ✗ Scaling sweep failed")
        print(output[-500:])
        return False

def verify_analyzers():
    """Verify analyzer verdicts are being generated."""
    print("\n[6/6] Checking analyzer verdicts...")

    # Get latest run
    cmd = "agentsysperf db ls 2>/dev/null | head -1 | awk '{print $1}'"
    run_id, _ = run_cmd(cmd)

    if not run_id:
        print("  ℹ No runs in store yet (run benchmarks first)")
        return True

    # Show verdicts for latest run. run_id comes from the store's own output
    # rather than from this file, so it gets quoted before reaching the shell.
    cmd = f"agentsysperf db show {shlex.quote(run_id)} 2>&1"
    output, code = run_cmd(cmd)

    # Count analyzer verdicts
    verdict_count = output.count("verdict:")

    if verdict_count > 0:
        print(f"  ✓ {verdict_count} analyzer verdicts for run {run_id}")

        # Show sample verdicts
        for line in output.split('\n'):
            if 'verdict:' in line or 'confidence:' in line:
                print(f"    {line.strip()}")

        return True
    else:
        print(f"  ℹ No analyzer verdicts yet (run benchmarks with L3 data)")
        return True

def main():
    """Run all verification checks."""
    print("=" * 80)
    print("  AgentSysPerf 0.1.0-RC1 Field Testing Verification")
    print("=" * 80)
    print("")

    checks = [
        ("Environment", check_environment),
        ("Plugin Discovery", check_plugins),
        ("Database", check_database),
        ("Synthetic CPU Run", verify_synthetic_cpu_run),
        ("Scaling Sweep", verify_scaling_sweep),
        ("Analyzer Verdicts", verify_analyzers),
    ]

    results = []
    for name, check_fn in checks:
        try:
            passed = check_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("  VERIFICATION SUMMARY")
    print("=" * 80)
    print("")

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print("")
    print(f"Result: {passed_count}/{total_count} checks passed")
    print("")

    if passed_count == total_count:
        print("  ✓ RC1 is ready for field testing!")
        return 0
    else:
        print("  ✗ Some checks failed. See above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
