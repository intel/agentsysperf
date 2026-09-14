#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

from __future__ import annotations

import pytest

from src.streams.resources import (
    ResourceBudget,
    pack_task_slots,
    plan_slot_counts,
    plan_task_slots,
    plan_unpinned_slot_counts,
    plan_unpinned_task_slots,
)


def test_resource_budget_formats_non_contiguous_cpuset():
    assert ResourceBudget((0, 1, 2, 7, 8, 12)).cpuset_str() == "0-2,7-8,12"


def test_slot_counts_assign_one_slot_to_every_required_size():
    counts = plan_slot_counts(16, {1: 8, 2: 3, 4: 1})
    assert set(counts) == {1, 2, 4}
    assert all(value >= 1 for value in counts.values())
    assert sum(size * count for size, count in counts.items()) <= 16


def test_slot_counts_rejects_budget_that_cannot_represent_task_mix():
    with pytest.raises(ValueError, match="one slot"):
        plan_slot_counts(6, {1: 3, 2: 2, 4: 1})


def test_pack_slots_is_disjoint_and_numa_local():
    nodes = [list(range(8)), list(range(16, 24))]
    slots = pack_task_slots({1: 2, 2: 2, 4: 1}, nodes)
    all_cpus = [cpu for slot in slots for cpu in slot.budget.cpuset]
    assert len(all_cpus) == len(set(all_cpus))
    assert all(
        set(slot.budget.cpuset) <= set(nodes[0])
        or set(slot.budget.cpuset) <= set(nodes[1])
        for slot in slots
    )


def test_pack_slots_rejects_unplaceable_size_instead_of_downsizing():
    with pytest.raises(ValueError, match="cannot place"):
        pack_task_slots({4: 1}, [list(range(3)), list(range(10, 13))])


def test_plan_task_slots_honors_process_budget():
    slots = plan_task_slots(
        8,
        {1: 4, 2: 2},
        nodes=[list(range(8)), list(range(8, 16))],
    )
    assert sum(len(slot.budget.cpuset) for slot in slots) <= 8
    assert {slot.size for slot in slots} == {1, 2}


def test_plan_task_slots_rejects_more_cores_than_available():
    with pytest.raises(ValueError, match="available CPU budget"):
        plan_task_slots(9, {1: 1}, nodes=[list(range(8))])


def test_unpinned_planner_creates_exactly_requested_oversubscribed_slots():
    slots = plan_unpinned_task_slots(12, {1: 8, 2: 3, 4: 1})
    assert len(slots) == 12
    assert all(not slot.budget.cpuset for slot in slots)
    assert {slot.size for slot in slots} == {1, 2, 4}


def test_unpinned_slot_counts_follow_queued_work():
    counts = plan_unpinned_slot_counts(10, {1: 8, 2: 2})
    assert sum(counts.values()) == 10
    assert counts[1] > counts[2]


def test_unpinned_planner_rejects_slot_cap_below_required_classes():
    with pytest.raises(ValueError, match="one slot"):
        plan_unpinned_slot_counts(2, {1: 1, 2: 1, 4: 1})


def test_unpinned_planner_enforces_configured_max_slots():
    with pytest.raises(ValueError, match="max_slots"):
        plan_unpinned_task_slots(9, {1: 1}, max_slots=8)
