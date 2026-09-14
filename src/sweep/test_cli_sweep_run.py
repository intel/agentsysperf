#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for `agentsysperf sweep run`.

The command is the only supported entry point for a scaling sweep, so these
tests pin the option -> SweepSpec mapping and the guards that stand between a
typo and hours of container work. HarborSweep itself is faked: constructing the
real one probes the platform and opens the canonical results DB, and running it
drives Harbor. What is under test here is the CLI's translation and its
refusals, not the sweep engine (covered by test_sweep_run_rows.py).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from src.cli import app
from src.sweep import DEFAULT_TB2_TASKS

_ENV = {"COLUMNS": "200"}


class _FakeSweep:
    """Stand-in for HarborSweep that records what the CLI handed it."""

    instances: list = []

    # Set by a test to make run() blow up the way a real prerequisite failure would.
    raises: BaseException | None = None

    def __init__(self, spec, **_kwargs):
        self.spec = spec
        # The real __init__ resolves vcpu_basis from the platform; the CLI relies
        # on that having happened before it quotes concurrency numbers.
        if spec.vcpu_basis is None:
            spec.vcpu_basis = 8
        self.store = SimpleNamespace(db_path=Path("/tmp/fake_results.db"))
        self.run_calls: list = []
        _FakeSweep.instances.append(self)

    def run(self, *, sweep_id=None, dry_run=False):
        self.run_calls.append({"sweep_id": sweep_id, "dry_run": dry_run})
        if _FakeSweep.raises is not None:
            raise _FakeSweep.raises
        return sweep_id or "sweep_fake"


@pytest.fixture()
def fake_sweep(monkeypatch):
    _FakeSweep.instances = []
    _FakeSweep.raises = None
    # The CLI does `from src.sweep import HarborSweep` inside the
    # command body, so patching the package attribute is what takes effect.
    monkeypatch.setattr("src.sweep.HarborSweep", _FakeSweep)
    return _FakeSweep


@pytest.fixture()
def fixture_file(tmp_path) -> Path:
    p = tmp_path / "fixture.jsonl"
    p.write_text("{}\n")
    return p


def _invoke(args, **kwargs):
    return CliRunner().invoke(app, ["sweep", "run", *args], env=_ENV, **kwargs)


# ── registration ─────────────────────────────────────────────────────


def test_sweep_run_is_a_registered_subcommand():
    result = CliRunner().invoke(app, ["sweep", "--help"], env=_ENV)
    assert result.exit_code == 0, result.output
    assert "run" in result.output


def test_help_documents_that_dry_run_is_not_a_measurement():
    result = _invoke(["--help"])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    assert "data_source=synthetic" in result.output


# ── refusals that happen before any work ─────────────────────────────


def test_replay_without_fixture_exits_2_and_names_the_remedy(fake_sweep):
    result = _invoke(["--llm-mode", "replay"])
    assert result.exit_code == 2, result.output
    assert "requires --fixture" in result.output
    assert "--llm-mode off" in result.output
    assert fake_sweep.instances == [], "must refuse before constructing a sweep"


def test_missing_fixture_file_is_caught_before_the_confirmation_prompt(fake_sweep):
    result = _invoke(["--llm-mode", "replay", "--fixture", "/tmp/nope_does_not_exist.jsonl"])
    assert result.exit_code == 2, result.output
    assert "does not exist" in result.output
    # The prompt is the thing we are protecting the user from seeing first.
    assert "Proceed?" not in result.output
    assert fake_sweep.instances == []


@pytest.mark.parametrize(
    ("flag", "allowed"),
    [
        ("--llm-mode", "replay, off, record"),
        ("--basis", "physical_cores, logical_cpus"),
        ("--numa", "unpinned, socket_pinned, interleaved"),
    ],
)
def test_bad_enum_value_exits_2_and_lists_the_allowed_set(fake_sweep, flag, allowed):
    # --llm-mode off keeps the replay-needs-a-fixture guard from firing first,
    # except when --llm-mode is itself the flag under test (last value wins).
    args = [flag, "bogus"] + ([] if flag == "--llm-mode" else ["--llm-mode", "off"])
    result = _invoke(args)
    assert result.exit_code == 2, result.output
    assert allowed in result.output
    assert fake_sweep.instances == []


# ── option -> SweepSpec mapping ──────────────────────────────────────


def test_densities_and_replicates_reach_the_spec(fake_sweep):
    result = _invoke(["--dry-run", "-d", "0.5", "-d", "2.0", "--replicates", "3"])
    assert result.exit_code == 0, result.output
    spec = fake_sweep.instances[0].spec
    assert list(spec.densities) == [0.5, 2.0]
    assert spec.replicates == 3
    # densities x replicates, so the plan is 2 points run 3 times each.
    assert len(spec.cells()) == 6


def test_plan_table_shows_concurrency_resolved_against_the_basis(fake_sweep):
    result = _invoke(["--dry-run", "-d", "0.5", "-d", "2.0"])
    assert result.exit_code == 0, result.output
    spec = fake_sweep.instances[0].spec
    # _FakeSweep resolves basis=8, so 0.5 -> 4 and 2.0 -> 16.
    assert spec.concurrency_for(0.5) == 4
    assert spec.concurrency_for(2.0) == 16
    assert "basis=8 physical_cores" in result.output
    for expected in ("4", "16"):
        assert expected in result.output


def test_default_tasks_are_the_fixture_coupled_set(fake_sweep):
    result = _invoke(["--dry-run"])
    assert result.exit_code == 0, result.output
    assert list(fake_sweep.instances[0].spec.tasks) == list(DEFAULT_TB2_TASKS)


def test_explicit_tasks_replace_the_defaults(fake_sweep):
    result = _invoke(["--dry-run", "--task", "terminal-bench/path-tracing"])
    assert result.exit_code == 0, result.output
    assert list(fake_sweep.instances[0].spec.tasks) == ["terminal-bench/path-tracing"]


def test_remaining_options_reach_the_spec(fake_sweep, tmp_path):
    out = tmp_path / "artifacts"
    result = _invoke([
        "--dry-run",
        "--basis", "logical_cpus",
        "--numa", "interleaved",
        "--attempts", "4",
        "--agent-timeout-multiplier", "0.1",
        "--dataset-path", str(tmp_path),
        "--output-dir", str(out),
        "--emon",
    ])
    assert result.exit_code == 0, result.output
    spec = fake_sweep.instances[0].spec
    assert spec.vcpu_basis_kind == "logical_cpus"
    assert spec.numa_policy == "interleaved"
    assert spec.attempts == 4
    assert spec.agent_timeout_multiplier == 0.1
    assert spec.dataset_path == tmp_path
    assert spec.output_dir == out
    assert spec.emon is True


def test_sweep_id_passes_through_to_the_engine(fake_sweep):
    result = _invoke(["--dry-run", "--sweep-id", "my_sweep"])
    assert result.exit_code == 0, result.output
    assert fake_sweep.instances[0].run_calls == [{"sweep_id": "my_sweep", "dry_run": True}]
    assert "my_sweep" in result.output


# ── dry-run provenance ───────────────────────────────────────────────


def test_dry_run_forces_llm_mode_off_and_drops_the_fixture(fake_sweep, fixture_file):
    """A dry run never contacts an LLM, so carrying a replay fixture into the
    spec would misrepresent how the (synthetic) points were produced."""
    result = _invoke(["--dry-run", "--llm-mode", "replay", "--fixture", str(fixture_file)])
    assert result.exit_code == 0, result.output
    spec = fake_sweep.instances[0].spec
    assert spec.llm_mode == "off"
    assert spec.fixture is None
    assert "llm_mode=off" in result.output


def test_dry_run_labels_its_output_synthetic_and_never_prompts(fake_sweep):
    result = _invoke(["--dry-run"])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    assert "data_source=synthetic" in result.output
    assert "(synthetic)" in result.output
    assert "Proceed?" not in result.output
    assert fake_sweep.instances[0].run_calls == [{"sweep_id": None, "dry_run": True}]


def test_emon_under_dry_run_says_it_will_collect_nothing(fake_sweep):
    """EMON is only started by the cell runner, which a dry run never enters."""
    result = _invoke(["--dry-run", "--emon"])
    assert result.exit_code == 0, result.output
    assert "--emon is ignored under --dry-run" in result.output


def test_real_run_is_not_labelled_synthetic(fake_sweep):
    result = _invoke(["--yes", "--llm-mode", "off"])
    assert result.exit_code == 0, result.output
    assert "synthetic" not in result.output
    assert fake_sweep.instances[0].run_calls == [{"sweep_id": None, "dry_run": False}]


# ── the confirmation gate on real runs ───────────────────────────────


def test_real_run_quotes_the_trial_count_and_aborts_when_declined(fake_sweep):
    result = _invoke(["--llm-mode", "off", "-d", "0.5", "--attempts", "2"], input="n\n")
    assert result.exit_code == 1, result.output
    # 1 cell x 10 default tasks x 2 attempts.
    assert "20 agent trials" in result.output
    assert "Aborted." in result.output
    assert fake_sweep.instances[0].run_calls == [], "declining must not start the sweep"


def test_confirming_starts_the_sweep(fake_sweep):
    result = _invoke(["--llm-mode", "off", "-d", "0.5"], input="y\n")
    assert result.exit_code == 0, result.output
    assert fake_sweep.instances[0].run_calls == [{"sweep_id": None, "dry_run": False}]


def test_yes_skips_the_prompt(fake_sweep):
    result = _invoke(["--yes", "--llm-mode", "off", "-d", "0.5"])
    assert result.exit_code == 0, result.output
    assert "Proceed?" not in result.output
    assert fake_sweep.instances[0].run_calls == [{"sweep_id": None, "dry_run": False}]


# ── engine failures surface as messages, not tracebacks ──────────────


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("fixture failed validation: ['bad']"),
        FileNotFoundError("harbor binary not found"),
    ],
)
def test_prerequisite_failures_are_reported_without_a_traceback(fake_sweep, exc):
    fake_sweep.raises = exc
    result = _invoke(["--dry-run"])
    assert result.exit_code == 1, result.output
    assert str(exc) in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
