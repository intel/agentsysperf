#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""P11 verification: MarkdownReportGenerator + `agentsysperf report` CLI.

Run: poetry run pytest src/reporting/test_markdown_report.py -q
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.cli import app
from src.protocols import AnalysisResult, MeasurementRecord, ReportGenerator
from src.reporting import MarkdownReportGenerator
from src.storage.sqlite_store import SQLiteResultStore

runner = CliRunner()


def _seed(store):
    store.store_run_metadata(run_id="r", metadata={
        "start_time": 1, "benchmark_id": "terminal-bench", "model": "gpt-4o-mini",
        "hardware_sku": "Granite Rapids test-sku", "owner_id": "testuser",
        "optimization_profile": "amx_bf16"})
    store.store_task_result(run_id="r", task_id="largest-eigenval",
                           result={"passed": True, "duration_s": 12.5})
    store.store_task_result(run_id="r", task_id="log-summary",
                           result={"passed": False, "duration_s": 4.0})
    store.store_measurements(run_id="r", records=[
        MeasurementRecord(span_id="r::largest-eigenval", layer="l3",
                          payload={"ipc": 3.2, "cache_miss_pct": 5.0})])
    store.store_analysis_results(run_id="r", results=[
        AnalysisResult(analyzer_name="cpu_bound", verdict="core_bound",
                       confidence=0.9, evidence={}, recommendations=[], span_id="largest-eigenval")])
    store._get_connection().commit()


def test_protocol_conformance():
    assert isinstance(MarkdownReportGenerator(), ReportGenerator)


def test_markdown_report_content(tmp_path):
    store = SQLiteResultStore(tmp_path)
    _seed(store)
    out = tmp_path / "r.md"
    MarkdownReportGenerator().generate_report(run_id="r", store=store, output_path=out)
    text = out.read_text()
    # header + summary
    assert "# AgentSysPerf Report" in text and "`r`" in text
    assert "1/2 passed" in text
    assert "Granite Rapids test-sku" in text
    # provenance (only-present rows)
    assert "amx_bf16" in text and "testuser" in text
    # task table with hw metrics joined from measurements
    assert "largest-eigenval" in text and "3.2" in text
    # verdict
    assert "cpu_bound" in text and "core_bound" in text


def test_report_handles_empty_run(tmp_path):
    store = SQLiteResultStore(tmp_path)
    store.store_run_metadata(run_id="empty", metadata={"start_time": 0})
    store._get_connection().commit()
    out = tmp_path / "e.md"
    MarkdownReportGenerator().generate_report(run_id="empty", store=store, output_path=out)
    text = out.read_text()
    assert "No task results recorded" in text


def test_cli_report_md(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    store = SQLiteResultStore.open()
    _seed(store)
    store.close()
    out = tmp_path / "cli.md"
    res = runner.invoke(app, ["report", "r", "--format", "md", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists() and "AgentSysPerf Report" in out.read_text()


def test_cli_report_missing_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    res = runner.invoke(app, ["report", "nope", "--format", "md"])
    assert res.exit_code == 1


def test_cli_report_bad_format(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTSYSPERF_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTSYSPERF_STORE_DSN", raising=False)
    monkeypatch.delenv("AGENTSYSPERF_STORE_URL", raising=False)
    store = SQLiteResultStore.open()
    _seed(store)
    store.close()
    res = runner.invoke(app, ["report", "r", "--format", "html"])
    assert res.exit_code == 1
    assert "Unknown format" in res.output


# ── W1.6: per-task verdicts are distinguishable; no fabricated model ──


def _seed_multitask(store):
    """Two tasks, one per-task analyzer — the shape that rendered as duplicates."""
    store.store_run_metadata(run_id="m", metadata={
        "start_time": 1, "benchmark_id": "synthetic-cpu"})
    for tid in ("linalg", "compile"):
        store.store_task_result(run_id="m", task_id=tid,
                                result={"passed": True, "duration_s": 3.0})
    store.store_analysis_results(run_id="m", results=[
        AnalysisResult(analyzer_name="cpu_bound", verdict="core_bound",
                       confidence=0.92, evidence={}, span_id="linalg"),
        AnalysisResult(analyzer_name="cpu_bound", verdict="core_bound",
                       confidence=0.92, evidence={}, span_id="compile"),
    ])
    store._get_connection().commit()


def test_verdict_rows_name_their_task(tmp_path):
    """Two same-verdict rows must be distinguishable.

    Before: both rows rendered as `| cpu_bound | core_bound | 92% |`, so a
    12-task run looked like a duplication bug rather than per-task results.
    """
    store = SQLiteResultStore(tmp_path)
    _seed_multitask(store)
    out = tmp_path / "m.md"
    MarkdownReportGenerator().generate_report(run_id="m", store=store, output_path=out)

    verdict_rows = [
        ln for ln in out.read_text().splitlines()
        if ln.startswith("|") and "cpu_bound" in ln
    ]
    assert len(verdict_rows) == 2, verdict_rows
    assert len(set(verdict_rows)) == 2, f"rows are indistinguishable: {verdict_rows}"
    assert any("linalg" in r for r in verdict_rows)
    assert any("compile" in r for r in verdict_rows)


def test_no_model_row_when_the_run_called_no_llm(tmp_path):
    """A no-LLM run must not advertise a model.

    `synthetic_cpu` uses NoOpAgentInvoker, but RunConfig.model carries a default
    regardless, so every synthetic report printed `Model: gpt-4o-mini`.
    """
    store = SQLiteResultStore(tmp_path)
    _seed_multitask(store)  # metadata has no "model" key
    out = tmp_path / "m.md"
    MarkdownReportGenerator().generate_report(run_id="m", store=store, output_path=out)

    text = out.read_text()
    assert "**Model:**" not in text
    assert "gpt-4o-mini" not in text


def test_model_row_still_shown_when_a_model_was_used(tmp_path):
    """The omission must be conditional, not a removal of the field."""
    store = SQLiteResultStore(tmp_path)
    _seed(store)
    out = tmp_path / "r.md"
    MarkdownReportGenerator().generate_report(run_id="r", store=store, output_path=out)

    assert "**Model:** gpt-4o-mini" in out.read_text()


def test_run_scoped_verdict_is_labelled_not_blank(tmp_path):
    """A verdict with no task gets an explicit label, not an empty cell."""
    store = SQLiteResultStore(tmp_path)
    store.store_run_metadata(run_id="w", metadata={"start_time": 1})
    store.store_task_result(run_id="w", task_id="t", result={"passed": True})
    store.store_analysis_results(run_id="w", results=[
        AnalysisResult(analyzer_name="scaling", verdict="knee_at_4",
                       confidence=0.8, evidence={})])
    store._get_connection().commit()
    out = tmp_path / "w.md"
    MarkdownReportGenerator().generate_report(run_id="w", store=store, output_path=out)

    row, = [
        ln for ln in out.read_text().splitlines()
        if ln.startswith("|") and "scaling" in ln
    ]
    assert "_run-wide_" in row
