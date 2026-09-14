#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Generate a summary report from AgentSysPerf benchmark results."""

import json
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()


def get_system_info():
    """Collect system information."""
    info = {}

    # CPU info
    with open('/proc/cpuinfo') as f:
        lines = f.readlines()
        for line in lines:
            if 'model name' in line:
                info['cpu_model'] = line.split(':')[1].strip()
                break
            if 'cpu cores' in line:
                info['cpu_cores'] = line.split(':')[1].strip()

    # Memory info
    with open('/proc/meminfo') as f:
        for line in f:
            if 'MemTotal' in line:
                mem_kb = int(line.split()[1])
                info['memory_gb'] = round(mem_kb / (1024**2), 1)
                break

    # OS info
    try:
        with open('/etc/os-release') as f:
            for line in f:
                if line.startswith('PRETTY_NAME'):
                    info['os'] = line.split('=')[1].strip().strip('"')
                    break
    except:
        info['os'] = 'Linux'

    # Kernel version
    try:
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        info['kernel'] = result.stdout.strip()
    except:
        info['kernel'] = 'Unknown'

    return info


def main():
    results_file = Path(f"{_TMP}/agentsysperf_results/measurement_records.json")

    if not results_file.exists():
        print("No results found. Run the benchmark first.")
        return

    with open(results_file) as f:
        records = json.load(f)

    system_info = get_system_info()

    print("\n" + "="*80)
    print("AgentSysPerf Benchmark Summary Report")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("SYSTEM INFORMATION")
    print("-" * 80)
    print(f"CPU Model:        {system_info.get('cpu_model', 'Unknown')}")
    print(f"CPU Cores:        {system_info.get('cpu_cores', 'Unknown')}")
    print(f"Total Memory:     {system_info.get('memory_gb', 'Unknown')} GB")
    print(f"Operating System: {system_info.get('os', 'Unknown')}")
    print(f"Kernel Version:   {system_info.get('kernel', 'Unknown')}")

    print("\nBENCHMARK RESULTS")
    print("-" * 80)
    print(f"{'Workload':<15} {'Duration':>12} {'CPU Time':>12} {'Avg CPU%':>12} {'Peak RSS':>12}")
    print("-" * 80)

    total_duration = 0
    total_cpu_time = 0

    for record in records:
        if record['layer'] == 'l1':
            payload = record['payload']
            workload = payload['node_id']
            duration_ms = payload['duration_us'] / 1000
            cpu_time = payload['cpu_time_s']
            cpu_pct = payload['cpu_pct_mean']
            rss_mb = payload['rss_kb_peak'] / 1024

            print(f"{workload:<15} {duration_ms:>10.0f} ms {cpu_time:>10.2f} s {cpu_pct:>10.1f} % {rss_mb:>10.0f} MB")

            total_duration += duration_ms
            total_cpu_time += cpu_time

    print("-" * 80)
    print(f"{'TOTAL':<15} {total_duration:>10.0f} ms {total_cpu_time:>10.2f} s")

    print("\nWORKLOAD DESCRIPTIONS")
    print("-" * 80)
    workloads = {
        'compile': 'Python bytecode compilation (regex-heavy)',
        'ml_train': 'NumPy matrix operations (ML training simulation)',
        'linalg': 'Dense linear algebra operations',
        'io': 'I/O intensive operations with buffered writes',
        'compress': 'Data compression using zlib',
        'raytrace': 'Ray tracing computation',
        'sat': 'Boolean satisfiability solving',
        'interpreter': 'Python interpretation overhead',
        'control': 'Control flow and function call overhead'
    }

    for workload, description in workloads.items():
        print(f"  {workload:<15} {description}")

    print("\nKEY FINDINGS")
    print("-" * 80)

    # Calculate some stats
    cpu_utils = [r['payload']['cpu_pct_mean'] for r in records if r['layer'] == 'l1']
    avg_cpu = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0

    peak_cpus = [r['payload']['cpu_pct_peak'] for r in records if r['layer'] == 'l1']
    max_peak = max(peak_cpus) if peak_cpus else 0

    rss_values = [r['payload']['rss_kb_peak'] / 1024 for r in records if r['layer'] == 'l1']
    max_rss = max(rss_values) if rss_values else 0

    print(f"  • Total benchmark duration: {total_duration/1000:.1f} seconds")
    print(f"  • Total CPU time consumed: {total_cpu_time:.1f} seconds")
    print(f"  • Average CPU utilization: {avg_cpu:.1f}%")
    print(f"  • Peak CPU utilization: {max_peak:.1f}%")
    print(f"  • Maximum memory usage: {max_rss:.0f} MB")
    print(f"  • Number of workloads tested: {len(records)}")

    print("\nNOTES")
    print("-" * 80)
    print("  • CPU% > 100% indicates multi-threaded execution")
    print("  • All workloads ran for 2 seconds each")
    print("  • L1 measurements only (resource usage, no hardware counters)")
    print("  • Hardware counter measurements (L3) require perf installation and permissions")

    print("\n" + "="*80)
    print(f"Full results: {results_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
