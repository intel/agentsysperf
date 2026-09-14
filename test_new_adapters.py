#!/usr/bin/env python3
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Quick test script for new benchmark adapters."""

from src.benchmarks.openclaw import OpenClawAdapter
from src.benchmarks.swe_bench import SWEBenchAdapter
from src.benchmarks.tau_bench import TauBenchAdapter
from src.testing.noop_invoker import NoOpAgentInvoker


def test_openclaw():
    print("\n=== Testing OpenClaw Adapter ===")
    adapter = OpenClawAdapter()
    invoker = NoOpAgentInvoker()

    tasks = list(adapter.list_tasks(limit=3))
    print(f"Loaded {len(tasks)} OpenClaw tasks")

    for i, task in enumerate(tasks[:1], 1):
        print(f"\nTask {i}: {task.id}")
        print(f"Category: {task.category}")
        print(f"Question: {task.extra['question'][:80]}...")

        result = adapter.run_task(task, agent_invoker=invoker)
        print(f"Result: passed={result.passed}, score={result.reward:.2f}")
        print(f"Evaluation: {result.extra.get('evaluation', {})}")

    adapter.teardown()
    print("✓ OpenClaw adapter working")


def test_swe_bench():
    print("\n=== Testing SWE-Bench Adapter ===")
    adapter = SWEBenchAdapter()
    invoker = NoOpAgentInvoker()

    tasks = list(adapter.list_tasks(limit=2))
    print(f"Loaded {len(tasks)} SWE-Bench tasks")

    for i, task in enumerate(tasks[:1], 1):
        print(f"\nTask {i}: {task.id}")
        print(f"Repo: {task.extra['repo']}")
        print(f"Problem: {task.extra['problem_statement'][:80]}...")

        result = adapter.run_task(task, agent_invoker=invoker)
        print(f"Result: passed={result.passed}, score={result.reward:.2f}")
        print(f"Tests: {result.extra.get('tests_passed', 0)}/{result.extra.get('tests_total', 0)}")

    adapter.teardown()
    print("✓ SWE-Bench adapter working")


def test_tau_bench():
    print("\n=== Testing Tau-Bench Adapter ===")
    adapter = TauBenchAdapter(domain="all")
    invoker = NoOpAgentInvoker()

    tasks = list(adapter.list_tasks(limit=3))
    print(f"Loaded {len(tasks)} Tau-Bench tasks")

    for i, task in enumerate(tasks[:1], 1):
        print(f"\nTask {i}: {task.id}")
        print(f"Domain: {task.extra['domain']}")
        print(f"Request: {task.extra['customer_request'][:80]}...")

        result = adapter.run_task(task, agent_invoker=invoker)
        print(f"Result: passed={result.passed}, score={result.reward:.2f}")
        print(f"Actions: {result.extra.get('actions_taken', [])}")

    adapter.teardown()
    print("✓ Tau-Bench adapter working")


def main():
    print("=" * 60)
    print("Testing New AgentSysPerf Benchmark Adapters")
    print("=" * 60)

    try:
        test_openclaw()
        test_swe_bench()
        test_tau_bench()

        print("\n" + "=" * 60)
        print("✅ ALL ADAPTERS WORKING CORRECTLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
