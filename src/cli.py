#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""AgentSysPerf CLI entry point.

Minimal first slice: ``agentsysperf measurements list`` to verify plugin
discovery works end to end. As subsequent merger phases land,
benchmark / profile / telemetry subcommands grow off the same root.

The CLI is a Typer app exposed as the ``agentsysperf`` console script via
``pyproject.toml [project.scripts]``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from src.default_tasks import DEFAULT_TASK_RESOURCE, load_default_task_text
from src.protocols import (
    MeasurementRecord,
    discover_analyzers,
    discover_benchmarks,
    discover_hardware_telemetry,
    discover_measurements,
    discover_optimization_profiles,
)
import tempfile

# Scratch root for run artifacts. gettempdir() honours TMPDIR and falls
# back to "/tmp", so this resolves to the same paths as the hardcoded
# /tmp literals it replaced while no longer pinning output to a
# world-writable directory on systems that configure TMPDIR elsewhere.
_TMP = tempfile.gettempdir()

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="AgentSysPerf: pluggable benchmark suite for agentic AI stacks.",
)
measurements_app = typer.Typer(no_args_is_help=True, help="Measurement plugins (L1, L3 perf, ...)")
benchmarks_app = typer.Typer(no_args_is_help=True, help="Benchmark adapter plugins")
profiles_app = typer.Typer(no_args_is_help=True, help="Optimization-profile plugins")
telemetry_app = typer.Typer(no_args_is_help=True, help="Hardware-telemetry plugins")
analyzers_app = typer.Typer(no_args_is_help=True, help="Analyzer plugins")
db_app = typer.Typer(no_args_is_help=True, help="Inspect/manage the results store (runs, benchmarks)")
sweep_app = typer.Typer(no_args_is_help=True, help="Concurrency-scaling sweeps")
app.add_typer(measurements_app, name="measurements")
app.add_typer(benchmarks_app, name="benchmarks")
app.add_typer(profiles_app, name="profiles")
app.add_typer(telemetry_app, name="telemetry")
app.add_typer(analyzers_app, name="analyzers")
app.add_typer(db_app, name="db")
app.add_typer(sweep_app, name="sweep")

console = Console()


def _configure_logging(verbose: bool) -> None:
    """Route library logging to the terminal.

    Without this the CLI installs no handler at all, so every
    ``logger.warning("... continuing without it")`` a probe emits is discarded
    and a degraded run looks exactly like a clean one. Warnings are shown by
    default; ``--verbose`` drops the floor to INFO, which is where the probes
    explain *why* they skipped.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _render_run_metrics_table(run_id: str, db_path: str) -> None:
    """Render a post-run per-task metrics table from persisted data.

    Reads the just-persisted run from the store and prints a Rich table
    summarising per-task status, timing, CPU utilisation, and (conditionally)
    L3 micro-architectural metrics. This is purely cosmetic; failures here
    must never abort the run.
    """
    from src.storage.sqlite_store import SQLiteResultStore

    store = SQLiteResultStore(db_path=Path(db_path))
    tasks = store.query_tasks(run_id)
    if not tasks:
        return

    measurements = store.query_measurements(run_id)
    # Index L1/L3 measurement payloads by task_id for O(1) lookup.
    l1_by_task: dict[str, dict] = {}
    l3_by_task: dict[str, dict] = {}
    for m in measurements:
        tid = m.get("task_id") or ""
        layer = m.get("layer", "")
        payload = m.get("payload") or {}
        if layer == "l1" and tid:
            l1_by_task.setdefault(tid, {}).update(payload)
        elif layer == "l3" and tid:
            l3_by_task.setdefault(tid, {}).update(payload)

    has_l3 = bool(l3_by_task)

    table = Table(show_lines=False, title_style="bold")
    table.add_column("Task", style="cyan", no_wrap=True, max_width=40)
    table.add_column("Status", justify="center")
    table.add_column("Elapsed (s)", justify="right")
    table.add_column("CPU %", justify="right")
    if has_l3:
        table.add_column("IPC", justify="right")
        table.add_column("Cache Miss %", justify="right")

    total_elapsed = 0.0
    cpu_pcts: list[float] = []
    pass_count = 0

    for t in tasks:
        tid = t["task_id"]
        short_name = tid.split("/")[-1] if "/" in tid else tid
        passed = bool(t.get("passed"))
        if passed:
            pass_count += 1
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"

        dur = t.get("duration_s")
        dur_str = f"{dur:.1f}" if dur is not None else "-"
        if dur is not None:
            total_elapsed += dur

        # L1 metrics
        l1 = l1_by_task.get(tid, {})
        cpu_pct = l1.get("cpu_pct_mean")
        cpu_str = f"{cpu_pct:.1f}" if cpu_pct is not None else "-"
        if cpu_pct is not None:
            cpu_pcts.append(cpu_pct)

        row: list[str] = [short_name, status, dur_str, cpu_str]

        if has_l3:
            l3 = l3_by_task.get(tid, {})
            ipc = l3.get("ipc")
            cache_miss = l3.get("cache_miss_pct")
            row.append(f"{ipc:.2f}" if ipc is not None else "-")
            row.append(f"{cache_miss:.1f}" if cache_miss is not None else "-")

        table.add_row(*row)

    # Summary footer row
    avg_cpu = sum(cpu_pcts) / len(cpu_pcts) if cpu_pcts else 0.0
    throughput = (len(tasks) / total_elapsed * 60) if total_elapsed > 0 else 0.0
    summary_row: list[str] = [
        "[bold]Total/Avg[/bold]",
        f"{pass_count}/{len(tasks)}",
        f"{total_elapsed:.1f}",
        f"{avg_cpu:.1f}" if cpu_pcts else "-",
    ]
    if has_l3:
        summary_row += ["-", "-"]
    table.add_row(*summary_row, end_section=True)

    console.print()
    console.print(table)
    console.print(f"[dim]Throughput: {throughput:.2f} tasks/min[/dim]")
    console.print(
        "[dim]Full dashboard: streamlit run demo_app.py "
        "--server.port 7860 --server.address 0.0.0.0[/dim]"
    )


def _render_plugins(title: str, plugins: dict, *, layer_attr: str | None = None) -> None:
    """Render a table of discovered plugins."""
    table = Table(title=title, show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    if layer_attr:
        table.add_column("Layer", style="magenta")
    table.add_column("Class", style="green")
    table.add_column("Module", style="dim")
    if not plugins:
        console.print(f"[yellow]No {title.lower()} discovered.[/yellow]")
        console.print("[dim]Plugins register via the relevant entry-point group in pyproject.toml.[/dim]")
        return
    for name, instance in sorted(plugins.items()):
        cls = type(instance)
        row = [name]
        if layer_attr:
            attr_value = getattr(instance, layer_attr, "")
            # Convert frozenset to comma-separated string
            if isinstance(attr_value, frozenset):
                attr_value = ", ".join(sorted(attr_value))
            row.append(str(attr_value))
        row += [cls.__name__, cls.__module__]
        table.add_row(*row)
    console.print(table)


@measurements_app.command("list")
def measurements_list() -> None:
    """List discoverable Measurement plugins."""
    _render_plugins(
        "Measurement plugins", discover_measurements(), layer_attr="layer",
    )


@benchmarks_app.command("list")
def benchmarks_list() -> None:
    """List discoverable BenchmarkAdapter plugins."""
    _render_plugins("Benchmark adapters", discover_benchmarks())


@profiles_app.command("list")
def profiles_list() -> None:
    """List discoverable OptimizationProfile plugins.

    The "Applicable" column reflects this host's ISA: a profile requiring AMX
    cannot produce a meaningful measurement on a machine without it, so it is
    reported as unsupported rather than silently measuring the fallback path.
    """
    profiles = discover_optimization_profiles()
    if not profiles:
        _render_plugins("Optimization profiles", profiles)
        return

    from src.platform import detect_platform

    plat = detect_platform()
    table = Table(title="Optimization profiles", show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Requires", style="magenta")
    table.add_column("Applicable here", style="green")
    table.add_column("Class", style="dim")
    for name, prof in sorted(profiles.items()):
        requires = getattr(prof, "requires", ()) or ()
        missing = prof.unsupported_on(plat) if hasattr(prof, "unsupported_on") else []
        applicable = (
            "[green]yes[/green]"
            if not missing
            else f"[red]no — host lacks {', '.join(missing)}[/red]"
        )
        table.add_row(
            name,
            ", ".join(requires) or "—",
            applicable,
            type(prof).__name__,
        )
    console.print(table)
    console.print(
        f"[dim]Host: {plat.microarchitecture} · AMX={'yes' if plat.has_amx else 'no'} "
        f"· AVX-512={'yes' if plat.has_avx512 else 'no'}[/dim]"
    )


@telemetry_app.command("list")
def telemetry_list() -> None:
    """List discoverable HardwareTelemetry plugins and the events they advertise.

    A telemetry plugin is the counter source behind
    ``OptimizationProfilePlugin.verify_engaged()`` — point-in-time reads
    checked against thresholds. Per-span counter *rows* come from a
    Measurement instead (``agentsysperf measurements list``).

    The "Advertised events" column is the load-bearing one: a profile can only
    be engagement-verified for counters some installed plugin advertises. The
    footer names the declared profile counters nothing advertises, so a failed
    ``verify_engaged`` is predictable from this command rather than a surprise
    mid-run.
    """
    plugins = discover_hardware_telemetry()
    if not plugins:
        _render_plugins("Hardware telemetry", plugins)
        return

    table = Table(title="Hardware telemetry", show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Advertised events", style="magenta")
    table.add_column("Class", style="green")
    table.add_column("Module", style="dim")
    advertised: set[str] = set()
    for name, plugin in sorted(plugins.items()):
        events = sorted(getattr(plugin, "available_events", ()) or ())
        advertised.update(events)
        cls = type(plugin)
        table.add_row(name, ", ".join(events) or "—", cls.__name__, cls.__module__)
    console.print(table)
    console.print(
        "[dim]Counter source for OptimizationProfile.verify_engaged(); per-span "
        "counter rows are a Measurement (agentsysperf measurements list).[/dim]"
    )

    # Cross-reference what installed profiles want to verify against what any
    # installed plugin can actually read. Computed, not hardcoded: it stays
    # true if a vendor-aware plugin (e.g. intel_pcm) is installed later.
    wanted: set[str] = set()
    for prof in discover_optimization_profiles().values():
        spec = getattr(prof, "SPEC", None)
        wanted.update(getattr(spec, "verify_counters", None) or ())
    unverifiable = sorted(wanted - advertised)
    if unverifiable:
        console.print(
            f"[yellow]Not advertised by any installed plugin: "
            f"{', '.join(unverifiable)}[/yellow]"
        )
        console.print(
            "[dim]Profiles declaring those counters fail engagement verify with an "
            "explicit \"counter not advertised\" — honest by design, not a bug. "
            "Those are derived metric names, not raw perf events, so closing the "
            "gap needs a plugin that computes them; vendor PMU codes alone are not "
            "enough, and the hugepage / QAT / oneDNN counters are not PMU "
            "quantities at all. See docs/measurement_layers.md.[/dim]"
        )
    elif wanted:
        console.print(
            "[dim]Every counter declared by an installed profile is advertised "
            "by some installed plugin.[/dim]"
        )
    console.print(
        "[dim]Whether this host actually permits the reads: agentsysperf preflight.[/dim]"
    )


@analyzers_app.command("list")
def analyzers_list() -> None:
    """List discoverable Analyzer plugins."""
    _render_plugins("Analyzer plugins", discover_analyzers(), layer_attr="input_layers")


@app.command("list")
def list_all() -> None:
    """List every discoverable plugin (benchmarks, measurements, profiles, telemetry, analyzers).

    Shortcut for the five per-group subcommands. Useful for quick
    "what's installed?" checks; drill into a single group with
    ``agentsysperf benchmarks list``, ``agentsysperf measurements list``,
    etc.
    """
    _render_plugins("Benchmark adapters", discover_benchmarks())
    _render_plugins(
        "Measurement plugins", discover_measurements(), layer_attr="layer",
    )
    _render_plugins("Optimization profiles", discover_optimization_profiles())
    _render_plugins("Hardware telemetry", discover_hardware_telemetry())
    _render_plugins("Analyzer plugins", discover_analyzers())


@app.command("analyze")
def analyze(
    results_dir: Path = typer.Argument(..., help="Directory containing measurement records (output_dir from a run)"),
    format: str = typer.Option("tree", help="Output format: 'tree' or 'json'"),
) -> None:
    """Analyze measurement records and produce insights.

    Loads MeasurementRecords from the specified results directory,
    runs all discovered Analyzer plugins, and displays the insights.

    Example:
        agentsysperf analyze /tmp/agentsysperf_scratch/synthetic_l1_l3/
    """
    # Load records from results_dir
    if not results_dir.exists():
        console.print(f"[red]Error: Directory not found: {results_dir}[/red]")
        raise typer.Exit(1)

    # For the initial implementation, we'll reconstruct records from the run
    # by looking for the known measurement output files
    records = _load_records_from_dir(results_dir)

    if not records:
        console.print(f"[yellow]No measurement records found in {results_dir}[/yellow]")
        console.print("[dim]Tip: Make sure you ran a benchmark with measurements enabled[/dim]")
        raise typer.Exit(1)

    # Discover and run analyzers
    analyzers = discover_analyzers()

    if not analyzers:
        console.print("[yellow]No analyzer plugins discovered[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Analyzing {len(records)} measurement records with {len(analyzers)} analyzers...[/bold]\n")

    all_results = []
    for analyzer_name, analyzer in sorted(analyzers.items()):
        try:
            results = list(analyzer.analyze(records))
            all_results.extend(results)
        except Exception as e:
            console.print(f"[red]Analyzer {analyzer_name} failed: {e}[/red]")

    if not all_results:
        console.print("[yellow]No analysis results produced[/yellow]")
        return

    # Display results
    if format == "json":
        _render_json(all_results)
    else:
        _render_tree(all_results, records)


def _load_records_from_dir(results_dir: Path) -> list[MeasurementRecord]:
    """Load MeasurementRecords from a results directory.

    This is a simplified implementation that reconstructs records from
    the known output files. In production, records would be serialized
    to a structured format (JSON, Parquet, etc.).
    """
    # For now, return empty list - the actual implementation would read
    # from serialized measurement records. The examples will need to be
    # updated to serialize records to JSON.
    records_file = results_dir / "measurement_records.json"
    if records_file.exists():
        with open(records_file) as f:
            data = json.load(f)
            return [MeasurementRecord(**r) for r in data]
    return []


def _render_tree(results: list, records: list[MeasurementRecord]) -> None:
    """Render analysis results as a rich tree."""
    # Group by span_id
    by_span = {}
    run_wide = []

    for result in results:
        if result.span_id:
            by_span.setdefault(result.span_id, []).append(result)
        else:
            run_wide.append(result)

    # Render per-span results
    for span_id in sorted(by_span.keys()):
        span_results = by_span[span_id]
        # Extract task name from span_id (format: run-id::task-id)
        task_name = span_id.split("::")[-1] if "::" in span_id else span_id

        tree = Tree(f"[bold cyan]{task_name}[/bold cyan]")
        for result in span_results:
            confidence_color = "green" if result.confidence > 0.8 else "yellow" if result.confidence > 0.6 else "red"
            branch = tree.add(
                f"[bold]{result.analyzer_name}[/bold]: {result.verdict} "
                f"([{confidence_color}]confidence: {result.confidence:.2f}[/{confidence_color}])"
            )

            # Evidence
            if result.evidence:
                evidence_str = ", ".join(f"{k}={v}" for k, v in list(result.evidence.items())[:3])
                branch.add(f"[dim]Evidence: {evidence_str}[/dim]")

            # Recommendations
            if result.recommendations:
                rec_branch = branch.add("[bold]→[/bold] Recommendations:")
                for rec in result.recommendations[:2]:  # Show top 2
                    rec_branch.add(f"[dim]• {rec}[/dim]")

        console.print(tree)
        console.print()

    # Render run-wide results
    if run_wide:
        tree = Tree("[bold cyan]Run-wide Analysis[/bold cyan]")
        for result in run_wide:
            confidence_color = "green" if result.confidence > 0.8 else "yellow"
            branch = tree.add(
                f"[bold]{result.analyzer_name}[/bold]: {result.verdict} "
                f"([{confidence_color}]confidence: {result.confidence:.2f}[/{confidence_color}])"
            )
            if result.evidence:
                evidence_str = ", ".join(f"{k}={v}" for k, v in result.evidence.items())
                branch.add(f"[dim]Evidence: {evidence_str}[/dim]")
            if result.recommendations:
                for rec in result.recommendations:
                    branch.add(f"[dim]→ {rec}[/dim]")
        console.print(tree)


def _render_json(results: list) -> None:
    """Render analysis results as JSON."""
    output = [
        {
            "analyzer": r.analyzer_name,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "evidence": dict(r.evidence),
            "recommendations": list(r.recommendations),
            "span_id": r.span_id,
        }
        for r in results
    ]
    console.print_json(data=output)


@app.command("analytics")
def analytics(
    benchmark: Optional[str] = typer.Option(None, help="Filter to one benchmark_id"),
) -> None:
    """Cross-run analytical summary via DuckDB (optional 'analytics' extra).

    DuckDB reads the canonical store directly (zero copy) for wide aggregation.
    Falls over to a clear install hint if the extra isn't installed.

    Example:
        agentsysperf analytics
    """
    try:
        from src.storage import duckdb_analytics as dk
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    try:
        n = dk.measurement_count()
        console.print(f"[bold]Measurement rows in store:[/bold] {n}")
        if n < 10_000:
            console.print("[dim](Below ~10-15k rows SQLite is faster; DuckDB pays off as the store grows.)[/dim]")
        rows = dk.ipc_by_benchmark()
        if not rows:
            console.print("[yellow]No measurements with IPC in the store yet.[/yellow]")
            return
        table = Table(title="Mean IPC by benchmark (all runs)")
        for col in ("benchmark", "runs", "avg_ipc", "avg_cache_miss_%"):
            table.add_column(col)
        for b, runs, ipc, miss in rows:
            table.add_row(str(b), str(runs), str(ipc), str(miss))
        console.print(table)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command("export")
def export(
    out: Path = typer.Argument(..., help="Output .parquet path"),
    table: str = typer.Option("measurements", help="Store table to export"),
) -> None:
    """Export a store table to Parquet (publishable / cold-tier bundle).

    Requires the optional 'analytics' extra (DuckDB). Example:
        agentsysperf export /tmp/measurements.parquet
    """
    try:
        from src.storage import duckdb_analytics as dk
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    try:
        written = dk.export_parquet(out, table=table)
        console.print(f"[green]Wrote {written}[/green]")
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


# ─── db: results-store management ─────────────────────────────────────

def _open_store(read_only: bool = True):
    """Open the canonical store. For read-only callers, a missing DB file would
    raise lazily on first query (sqlite ro can't create it) — so for read paths
    we ensure the DB exists by opening it writable once (creates an empty,
    migrated store), which makes `db ls` on a fresh machine print 'No runs'
    instead of crashing."""
    from src.storage.sqlite_store import SQLiteResultStore
    if read_only:
        from src.home import default_db_path, store_dsn
        if not store_dsn() and not default_db_path().exists():
            # Create+migrate an empty store, then reopen read-only.
            SQLiteResultStore.open(read_only=False).close()
    return SQLiteResultStore.open(read_only=read_only)


@db_app.command("ls")
def db_ls(
    benchmark: Optional[str] = typer.Option(None, help="Filter to one benchmark_id"),
    limit: int = typer.Option(50, help="Max runs to show (newest first)"),
) -> None:
    """List runs in the canonical store, newest first.

    Example:
        agentsysperf db ls
        agentsysperf db ls --benchmark terminal-bench
    """
    store = _open_store()
    runs = store.list_runs(benchmark_id=benchmark, limit=limit)
    if not runs:
        console.print("[yellow]No runs in the store.[/yellow]")
        console.print(f"[dim]Store: {store.db_path}[/dim]")
        return
    table = Table(title=f"Runs ({len(runs)})")
    for col in ("run_id", "benchmark", "model", "hardware_sku", "owner", "tasks"):
        table.add_column(col, overflow="fold")
    for r in runs:
        table.add_row(
            str(r.get("run_id")), str(r.get("benchmark_id") or "—"),
            str(r.get("model") or "—"), str(r.get("hardware_sku") or "—"),
            str(r.get("owner_id") or "—"),
            f"{r.get('passed_tasks') or 0}/{r.get('total_tasks') or 0}",
        )
    console.print(table)
    console.print(f"[dim]Store: {store.db_path}[/dim]")


@db_app.command("show")
def db_show(run_id: str = typer.Argument(..., help="Run id to inspect")) -> None:
    """Show one run's metadata, tasks, and measurement-layer coverage.

    Example:
        agentsysperf db show dashboard_demo
    """
    store = _open_store()
    run = store.get_run(run_id)
    if run is None:
        console.print(f"[red]No run {run_id!r} in the store.[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]Run[/bold] {run_id}")
    for k in ("benchmark_id", "model", "hardware_sku", "optimization_profile",
              "owner_id", "host_id", "total_tasks", "passed_tasks", "status"):
        if run.get(k) is not None:
            console.print(f"  {k}: {run[k]}")
    tasks = store.query_tasks(run_id)
    console.print(f"  tasks: {len(tasks)}")
    meas = store.query_measurements(run_id)
    layers: dict = {}
    for m in meas:
        layers[m["layer"]] = layers.get(m["layer"], 0) + 1
    console.print(f"  measurements by layer: {layers or '—'}")


@db_app.command("rm")
def db_rm(
    run_id: str = typer.Argument(..., help="Run id to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a run and cascade its tasks/measurements/spans/verdicts.

    Destructive. Prompts unless --yes. Example:
        agentsysperf db rm bad_run --yes
    """
    store = _open_store(read_only=False)
    if store.get_run(run_id) is None:
        console.print(f"[red]No run {run_id!r} in the store.[/red]")
        raise typer.Exit(1)
    if not yes:
        typer.confirm(f"Delete run {run_id!r} and all its data?", abort=True)
    n = store.delete_run(run_id)
    console.print(f"[green]Deleted {n} run(s).[/green]" if n
                  else "[yellow]Nothing deleted.[/yellow]")


@app.command("run")
def run(
    benchmark: str = typer.Option("terminal-bench", "--benchmark", "-b",
                                  help="Benchmark: terminal-bench | synthetic_cpu"),
    model: str = typer.Option("gpt-4o-mini", help="LiteLLM model (terminal-bench)"),
    num_tasks: Optional[int] = typer.Option(None, "--num-tasks", "-n",
                                            help="Run the first N tasks"),
    full: bool = typer.Option(False, "--full", help="Run all tasks in the dataset"),
    tasks: Optional[str] = typer.Option(None, "--tasks",
                                       help="Comma-separated explicit task names"),
    max_turns: int = typer.Option(200, help="Max agent turns per task"),
    timeout: int = typer.Option(3600, help="Per-task wall-clock cap (seconds)"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run id (default: benchmark+timestamp)"),
    output: Optional[Path] = typer.Option(None, "--output", help="Artifact dir"),
    record: Optional[Path] = typer.Option(None, "--record", help="Record LLM turns to this fixture"),
    replay: Optional[Path] = typer.Option(None, "--replay", help="Replay LLM turns from this fixture (no live LLM, no $)"),
    upstream: Optional[str] = typer.Option(None, "--upstream", help="Real LLM base URL for --record (default: $OPENAI_BASE_URL)"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                 help="Show INFO logs, including why a probe or analyzer skipped"),
) -> None:
    """Run a benchmark and persist it to the canonical store.

    The result is immediately visible to `agentsysperf db ls`, the dashboards
    (via the data-access layer), and `agentsysperf report`.

    Replay journal (the cache): --record once to capture LLM turns, then
    --replay to re-run free — skips the costly/nondeterministic LLM calls while
    still measuring hardware on THIS machine. --replay requires --tasks, naming
    the same tasks (and the same agent) the fixture was recorded with.

    Examples:
        agentsysperf run --benchmark synthetic_cpu --num-tasks 3
        agentsysperf run -b terminal-bench -n 2 --record /tmp/tb2.jsonl   # live, needs OPENAI_API_KEY
        agentsysperf run -b terminal-bench --tasks hello-world --replay /tmp/tb2.jsonl   # free
    """
    from src.run_driver import RunConfig, run_benchmark

    _configure_logging(verbose)

    if record and replay:
        console.print("[red]--record and --replay are mutually exclusive.[/red]")
        raise typer.Exit(1)

    # --replay needs an EXPLICIT task list. -n/--full slice the dataset in its own
    # order, while a fixture holds whichever trials happened to be recorded, keyed
    # by sha256(first user message) — so the two sets coincide only by luck and
    # every non-overlapping task misses. We cannot intersect them for you: the
    # fixture stores trial_key hashes, not task names, and recovering a name would
    # mean building each task's prompt with the same agent that recorded it.
    if replay and not tasks:
        selector = "--full" if full else f"-n {num_tasks}" if num_tasks else "no task selector"
        console.print(
            f"[red]--replay requires --tasks; got {selector}.[/red]\n"
            "  -n/--full pick tasks from the dataset, but the fixture holds a fixed\n"
            "  recorded set. Any task not in it misses, and a miss is fatal.\n"
            f"  Name the recorded tasks explicitly:  --tasks task-a,task-b"
        )
        if replay.exists():
            from src.replay.fixture import fixture_stats

            st = fixture_stats(replay)
            console.print(
                f"  [dim]{replay} holds {st.n_trials} trial(s), {st.n_entries} turn(s), "
                f"avg {st.avg_turns:.1f} turns/trial. Trial keys are hashes, so the\n"
                f"  task names must come from whatever recorded it.[/dim]"
            )
        else:
            console.print(f"  [yellow]Note: {replay} does not exist.[/yellow]")
        raise typer.Exit(1)

    mode = "record" if record else "replay" if replay else "off"
    fixture = record or replay

    cfg = RunConfig(
        benchmark=benchmark, model=model,
        num_tasks=num_tasks, full=full,
        tasks=[t.strip() for t in tasks.split(",")] if tasks else None,
        max_turns=max_turns, timeout_s=timeout,
        run_id=run_id, output_dir=output,
        replay_mode=mode, fixture=fixture, upstream=upstream,
    )
    try:
        summary = run_benchmark(cfg)
    except Exception as e:
        console.print(f"[red]Run failed: {e}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]Run {summary.run_id}: {summary.passed}/{summary.total} passed, "
        f"{summary.records} records.[/green]"
    )
    if summary.coverage is not None:
        cov = summary.coverage
        # Amber when anything was silent: "9 records" alone reads as success,
        # and a 3-of-5 run should not look like a 5-of-5 one.
        partial = bool(cov.measurements_silent or cov.analyzers_silent)
        style = "yellow" if partial else "green"
        for line in cov.as_lines():
            console.print(f"[{style}]{line}[/{style}]")
    console.print(f"[dim]Store: {summary.db_path} · inspect: agentsysperf db show {summary.run_id}[/dim]")

    # Render post-run metrics table from the just-persisted data.
    try:
        _render_run_metrics_table(summary.run_id, summary.db_path)
    except Exception:
        pass  # Table is cosmetic; never fail the run for it.


@app.command("run-streams")
def run_streams(
    tasks: str | None = typer.Option(
        None,
        "--tasks",
        help=(
            "Comma-separated Terminal-Bench task names "
            "(default: clean_tasks_23.txt)"
        ),
    ),
    slots: Optional[int] = typer.Option(
        None,
        "--slots",
        min=1,
        help="Exact number of unpinned task-sized worker slots",
    ),
    total_cores: Optional[int] = typer.Option(
        None,
        "--total-cores",
        min=1,
        help=(
            "Compatibility alias for --slots when unpinned; required CPU budget "
            "when --pin-cores is selected"
        ),
    ),
    pin_cores: bool = typer.Option(
        False,
        "--pin-cores",
        help="Opt into NUMA-local CPU sets and the pinned total-core planner",
    ),
    max_slots: int = typer.Option(
        4096,
        "--max-slots",
        min=1,
        help="Maximum allowed worker slots",
    ),
    stream_multiple: float = typer.Option(
        2.0,
        "--stream-multiple",
        min=0.1,
        help="Target task waves per planned worker slot",
    ),
    model: str = typer.Option(
        "openai/agentsysperf-proxy",
        help="LiteLLM model",
    ),
    replay: Optional[Path] = typer.Option(
        None,
        "--replay",
        help="Replay fixture. Without it, calls use the live model without recording.",
    ),
    timeout: int = typer.Option(3600, "--timeout", min=1,
                                help="Per-task wall-clock cap in seconds"),
    max_turns: int = typer.Option(
        1_000_000,
        "--max-turns",
        min=1,
        help="Maximum agent turns per task",
    ),
    launch_stagger: float = typer.Option(
        0.0,
        "--launch-stagger",
        min=0.0,
        help="Seconds to wait between worker starts",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Base run id (default: task-sized stream plus timestamp)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Artifact directory",
    ),
    dataset: str = typer.Option(
        "terminal-bench/terminal-bench-2",
        "--dataset",
        help="Harbor dataset identifier",
    ),
    ref: str = typer.Option("1", "--ref", help="Harbor dataset revision"),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show INFO logs, including worker scheduling and cleanup details",
    ),
) -> None:
    """Run Terminal-Bench on runc workers sized from each task's metadata.

    The scheduler resolves all selected tasks before launch, pre-pulls their
    images, and routes every task only to a matching-size slot. Scheduling is
    unpinned by default; ``--pin-cores`` restores NUMA-local CPU sets.
    """
    from src.run_driver import RunConfig
    from src.streams import run_task_sized_streams

    _configure_logging(verbose)
    if tasks is None:
        try:
            tasks = load_default_task_text()
        except OSError as exc:
            console.print(
                f"[red]Default task file unavailable: "
                f"{DEFAULT_TASK_RESOURCE} ({exc})[/red]"
            )
            raise typer.Exit(1)

    task_names = [name.strip() for name in tasks.split(",") if name.strip()]
    if not task_names:
        console.print("[red]--tasks must include at least one task name.[/red]")
        raise typer.Exit(1)
    if slots is not None and total_cores is not None:
        console.print("[red]--slots and --total-cores are mutually exclusive.[/red]")
        raise typer.Exit(1)
    if pin_cores and slots is not None:
        console.print("[red]--slots cannot be combined with --pin-cores.[/red]")
        raise typer.Exit(1)
    if pin_cores and total_cores is None:
        console.print("[red]--pin-cores requires --total-cores.[/red]")
        raise typer.Exit(1)
    requested = total_cores if slots is None else slots
    if requested is None:
        console.print("[red]Specify --slots (or deprecated --total-cores).[/red]")
        raise typer.Exit(1)
    if requested > max_slots:
        console.print(
            f"[red]Requested {requested} slots exceeds --max-slots={max_slots}.[/red]"
        )
        raise typer.Exit(1)
    if total_cores is not None and not pin_cores:
        console.print("[yellow]Warning: --total-cores is deprecated; use --slots.[/yellow]")

    cfg = RunConfig(
        benchmark="terminal-bench",
        model=model,
        max_turns=max_turns,
        timeout_s=timeout,
        run_id=run_id,
        output_dir=output,
        replay_mode="replay" if replay else "off",
        fixture=replay,
        dataset_name=dataset,
        dataset_ref=ref,
    )
    _mode = "replaying" if replay else "live"
    print(
        f"Running task-sized streams ({len(task_names)} tasks, "
        f"{requested} slots, mode={_mode})",
        flush=True,
    )
    print(
        "  (tasks launch in parallel; Docker build on first run is slowest)",
        flush=True,
    )
    try:
        results = run_task_sized_streams(
            cfg,
            task_names,
            slots=None if pin_cores else requested,
            total_cores=requested if pin_cores else None,
            pin_cores=pin_cores,
            max_slots=max_slots,
            stream_multiple=stream_multiple,
            launch_stagger_s=launch_stagger,
            on_task_done=lambda item, done, total: console.print(
                f"[dim]{done}/{total}[/dim] {item['task']} "
                f"(slot {item['stream_id']})"
            ),
        )
    except Exception as exc:
        console.print(f"[red]run-streams failed: {exc}[/red]")
        raise typer.Exit(1)

    passed = sum(result["passed"] for result in results)
    execution_failures = [
        result
        for result in results
        if result.get("error") or result.get("execution_failures", 0)
    ]
    oracle_failures = [
        result
        for result in results
        if result["passed"] < result["total"]
        and not result.get("error")
        and not result.get("execution_failures", 0)
    ]
    style = "red" if execution_failures else ("yellow" if oracle_failures else "green")
    console.print(
        f"[{style}]{len(results)} task runs complete: "
        f"{passed}/{len(results)} passed.[/{style}]"
    )
    for result in execution_failures:
        detail = result.get("error") or (
            f"{result.get('execution_failures', 0)} task execution failure(s); "
            f"{result['passed']}/{result['total']} passed"
        )
        console.print(f"[red]{result['task']}: {detail}[/red]")
    for result in oracle_failures:
        outcome = "oracle checks" if result.get("oracle_failures", 0) else "scoring"
        console.print(
            f"[yellow]{result['task']}: "
            f"{result['passed']}/{result['total']} {outcome} passed "
            "(recorded; non-fatal for stream execution)[/yellow]"
        )
    if execution_failures:
        raise typer.Exit(1)


@app.command("report")
def report(
    run_id: str = typer.Argument(..., help="Run id to report on"),
    format: str = typer.Option("md", "--format", help="Report format: 'md' or 'pptx'"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output path (default: <run_id>.<ext>)"),
) -> None:
    """Generate a report for a run from the canonical store.

    Examples:
        agentsysperf report dashboard_demo --format md
        agentsysperf report dashboard_demo --format pptx --out /tmp/report.pptx
    """
    store = _open_store()
    if store.get_run(run_id) is None:
        console.print(f"[red]No run {run_id!r} in the store.[/red]")
        raise typer.Exit(1)

    if format == "md":
        from src.reporting import MarkdownReportGenerator
        gen = MarkdownReportGenerator()
    elif format == "pptx":
        from src.reporting import XeonPowerPointGenerator
        gen = XeonPowerPointGenerator()
    else:
        console.print(f"[red]Unknown format {format!r} (use 'md' or 'pptx').[/red]")
        raise typer.Exit(1)

    ext = "md" if format == "md" else "pptx"
    output_path = out or Path(f"{run_id}.{ext}")
    try:
        written = gen.generate_report(run_id=run_id, store=store, output_path=output_path)
    except Exception as e:
        console.print(f"[red]Report generation failed: {e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Wrote {written}[/green]")


@app.command("preflight")
def preflight(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Apply the suggested fixes (prompts for confirmation; needs sudo).",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="With --fix, skip the confirmation prompt."
    ),
    require_perfspect: bool = typer.Option(
        False, "--require-perfspect", help="Treat a missing PerfSpect as a failure."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Check whether this host is configured for accurate measurement.

    Check-only by default: it reports what is misconfigured and prints the
    command that would fix it, without changing anything. Settings like the
    CPU governor and kernel.perf_event_paranoid materially change measured
    numbers, so this is worth running before any benchmark you intend to cite.

    Examples:
        agentsysperf preflight
        agentsysperf preflight --fix
    """
    from src.preflight.system_check import SystemCheck

    report = SystemCheck(require_perfspect=require_perfspect).run()

    if json_out:
        console.print_json(
            json.dumps(
                {
                    "ready": report.ready,
                    "platform": report.platform_info,
                    "checks": [
                        {
                            "name": c.name,
                            "category": c.category,
                            "status": c.status.value,
                            "current": c.current_value,
                            "expected": c.expected_value,
                            "fix_command": c.fix_command,
                            "notes": c.notes,
                        }
                        for c in report.checks
                    ],
                }
            )
        )
    else:
        console.print(report.summary())

    fixable = [c for c in report.checks if not c.passed and c.fix_command]

    if not fix:
        if fixable:
            console.print(
                f"\n[yellow]{len(fixable)} issue(s) have a known fix. "
                f"Re-run with --fix to apply them.[/yellow]"
            )
        raise typer.Exit(0 if report.ready else 1)

    if not fixable:
        console.print("\n[green]Nothing to fix.[/green]")
        raise typer.Exit(0 if report.ready else 1)

    # Applying changes is explicit and visible: every command is echoed before
    # it runs, so what touched the machine is auditable from the transcript.
    console.print("\n[bold]The following commands will be run:[/bold]")
    for c in fixable:
        console.print(f"  [cyan]{c.fix_command}[/cyan]   [dim]# {c.name}[/dim]")

    if not yes and not typer.confirm("\nApply these changes?", default=False):
        console.print("[yellow]Aborted; nothing was changed.[/yellow]")
        raise typer.Exit(1)

    import shlex
    import subprocess

    # shell=False by default: these run under sudo, and one fix_command
    # interpolates a runtime value (platform.release() into the
    # linux-headers-* package name), so a shell would make that an injection
    # vector. Only the two commands that genuinely need shell metacharacters —
    # a pipeline and a command substitution — declare fix_needs_shell.
    failed = 0
    for c in fixable:
        console.print(f"\n[bold]$ {c.fix_command}[/bold]")
        if c.fix_needs_shell:
            # fix_needs_shell commands are static literals in
            # system_check.py (one pipeline, one command substitution) that
            # interpolate nothing; test_fix_commands.py enforces both that
            # invariant and that the one interpolating command stays argv-only.
            result = subprocess.run(c.fix_command, shell=True)  # nosec B602
        else:
            result = subprocess.run(shlex.split(c.fix_command))
        if result.returncode != 0:
            failed += 1
            console.print(
                f"[red]failed (exit {result.returncode}): {c.name}[/red]"
            )

    # Re-check rather than assuming the fixes took effect.
    console.print("\n[bold]Re-checking...[/bold]")
    after = SystemCheck(require_perfspect=require_perfspect).run()
    console.print(after.summary())
    if failed:
        console.print(f"[red]{failed} fix command(s) failed.[/red]")
    raise typer.Exit(0 if after.ready else 1)


_LLM_MODES = ("replay", "off", "record")
_BASES = ("physical_cores", "logical_cpus")
_NUMA_POLICIES = ("unpinned", "socket_pinned", "interleaved")


def _choice(value: str, allowed: tuple, flag: str) -> str:
    """Validate a string option against its allowed set, or exit with the set."""
    if value not in allowed:
        console.print(
            f"[red]{flag}={value!r} is not one of: {', '.join(allowed)}[/red]"
        )
        raise typer.Exit(2)
    return value


@sweep_app.command("run")
def sweep_run(
    densities: List[float] = typer.Option(
        [0.25, 0.5, 1.0, 1.5, 2.0, 3.0], "--density", "-d",
        help="Density operating point(s); concurrency = round(density x basis). Repeatable.",
    ),
    replicates: int = typer.Option(1, help="Repeats per density point"),
    attempts: int = typer.Option(1, help="Harbor -k: passes through the task set"),
    tasks: Optional[List[str]] = typer.Option(
        None, "--task",
        help="Task id(s). Repeatable. Default: the 10 curated TB2 tasks "
             "(see DEFAULT_TB2_TASKS in src/sweep/spec.py).",
    ),
    fixture: Optional[Path] = typer.Option(
        None, help="Recorded replay fixture (required for --llm-mode replay)",
    ),
    dataset_path: Optional[Path] = typer.Option(
        None, "--dataset-path",
        help="Load tasks from this local dir instead of the Harbor registry "
             "(required on hosts without egress to raw.githubusercontent.com)",
    ),
    llm_mode: str = typer.Option("replay", "--llm-mode", help="replay | off | record"),
    basis: str = typer.Option(
        "physical_cores", "--basis", help="Density basis: physical_cores | logical_cpus",
    ),
    numa: str = typer.Option(
        "unpinned", "--numa", help="NUMA policy: unpinned | socket_pinned | interleaved",
    ),
    agent_timeout_multiplier: float = typer.Option(
        2.0, "--agent-timeout-multiplier", help="Scale each task's agent timeout",
    ),
    output_dir: Path = typer.Option(
        Path(f"{_TMP}/agentsysperf_sweep"), "--output-dir", help="Per-cell artifact dir",
    ),
    emon: bool = typer.Option(False, "--emon", help="Collect EMON EDP per cell"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Produce SYNTHETIC modeled cells instead of running Harbor. Exercises "
             "the pipeline offline; the points are NOT measurements and are stored "
             "with data_source=synthetic so the dashboard badges them.",
    ),
    sweep_id: Optional[str] = typer.Option(None, help="Sweep id (default: sweep_<epoch>)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO logs"),
) -> None:
    """Run a concurrency-scaling sweep and persist it for the dashboard.

    Finds the saturation knee — how many agents per vCPU the box sustains before
    throughput floors — and classifies the bottleneck. Density normalizes the
    x-axis, so a knee found here is comparable to one found on a smaller box.

    A real sweep drives Harbor over containerized tasks at rising concurrency
    and takes hours; `--dry-run` exercises the same pipeline in seconds with
    modeled points that are explicitly labelled synthetic.

    Replay is deterministic but strict: the fixture must be recorded with the
    SAME agent you replay, and only tasks present in it will hit.

    Examples:
        agentsysperf sweep run --dry-run
        agentsysperf sweep run --fixture path/to/fixture.jsonl -d 0.5 -d 1.0 -d 2.0
    """
    _configure_logging(verbose)
    from src.sweep import DEFAULT_TB2_TASKS, HarborSweep, SweepSpec

    llm_mode = _choice(llm_mode, _LLM_MODES, "--llm-mode")
    basis = _choice(basis, _BASES, "--basis")
    numa = _choice(numa, _NUMA_POLICIES, "--numa")

    # A dry run never contacts an LLM, so it must not carry a replay fixture
    # into the spec — mirrors examples/run_scaling_sweep_tb2.py.
    effective_llm_mode = "off" if dry_run else llm_mode
    spec = SweepSpec(
        densities=densities,
        replicates=replicates,
        attempts=attempts,
        tasks=list(tasks) if tasks else list(DEFAULT_TB2_TASKS),
        vcpu_basis_kind=basis,
        llm_mode=effective_llm_mode,
        fixture=None if effective_llm_mode == "off" else fixture,
        dataset_path=dataset_path,
        agent_timeout_multiplier=agent_timeout_multiplier,
        numa_policy=numa,
        emon=emon,
        output_dir=output_dir,
    )

    # Fail before doing any work, with the remedy, rather than partway through.
    if effective_llm_mode == "replay":
        if spec.fixture is None:
            console.print(
                "[red]--llm-mode replay requires --fixture[/red]\n"
                "Record one with the same agent you intend to replay, or use "
                "--llm-mode off, or --dry-run for a synthetic pipeline check."
            )
            raise typer.Exit(2)
        # HarborSweep.run() validates the fixture's contents, but only after the
        # confirmation prompt. Catch a missing file here so a typo'd path does
        # not first ask the user to approve hours of work.
        if not spec.fixture.exists():
            console.print(f"[red]--fixture {spec.fixture} does not exist[/red]")
            raise typer.Exit(2)

    # HarborSweep resolves vcpu_basis from the platform in __init__, so build it
    # before quoting concurrency numbers back to the user.
    sweep = HarborSweep(spec)
    cells = spec.cells()

    table = Table(title=f"Sweep plan ({len(cells)} cells)")
    table.add_column("density", justify="right")
    table.add_column("concurrency", justify="right")
    table.add_column("replicates", justify="right")
    for d in spec.densities:
        table.add_row(f"{d:g}", str(spec.concurrency_for(d)), str(replicates))
    console.print(table)
    console.print(
        f"[dim]basis={spec.vcpu_basis} {spec.vcpu_basis_kind} · "
        f"tasks={len(spec.tasks)} · attempts={attempts} · numa={numa} · "
        f"llm_mode={effective_llm_mode}[/dim]"
    )

    if dry_run:
        console.print(
            "[yellow]--dry-run: cells are SYNTHETIC modeled points, not "
            "measurements.[/yellow] Stored with data_source=synthetic; the "
            "dashboard badges them. Do not cite them as results."
        )
        if emon:
            # A dry run never enters the cell runner, which is the only place
            # EMON is started — so --emon would collect nothing at all. Say so
            # rather than let the user wait for hardware data that cannot come.
            console.print(
                "[yellow]--emon is ignored under --dry-run[/yellow] (no cell is "
                "actually executed, so there is nothing to profile). Drop "
                "--dry-run for EMON collection."
            )
    else:
        trials = len(cells) * len(spec.tasks) * attempts
        console.print(
            f"[yellow]Real sweep: {len(cells)} cells x {len(spec.tasks)} tasks "
            f"x {attempts} attempt(s) = {trials} agent trials under Harbor.[/yellow] "
            "This typically runs for hours and needs Harbor plus a container runtime."
        )
        if not yes and not typer.confirm("Proceed?", default=False):
            console.print("Aborted.")
            raise typer.Exit(1)

    try:
        resolved_id = sweep.run(sweep_id=sweep_id, dry_run=dry_run)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(
        f"\nSweep complete: [bold]{resolved_id}[/bold]"
        f"{' [yellow](synthetic)[/yellow]' if dry_run else ''}"
    )
    console.print(f"[dim]DB: {sweep.store.db_path}[/dim]")
    console.print("View: Scaling → Concurrency Sweep in demo_app.")


@sweep_app.command("import")
def sweep_import(
    results_dir: Path = typer.Argument(..., help="Dir holding n*_r*/point.json cells"),
    sweep_id: Optional[str] = typer.Option(None, help="Sweep id (default: derived from dir + timestamp)"),
    benchmark: str = typer.Option("terminal-bench", help="Benchmark slug for the sweep"),
    include_failed: bool = typer.Option(
        False, "--include-failed",
        help="Import cells whose cell_status != ok (their throughput denominator is wrong)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO logs"),
) -> None:
    """Import a shell-runner sweep into the store so the dashboard can read it.

    `harness/scripts/run_density_test_cwf.sh` writes one `point.json` per density
    cell; the dashboard reads the `sweeps`/`sweep_points` tables. This bridges
    them, preserving the agent name, the cpuset, and the density basis.

    The density basis is the CPUSET size, not the machine's core count — a
    16-core pinned run at 16 agents is density 1.0, not 0.056. `vcpu_basis_kind`
    is recorded as `cpuset_cores` so the two definitions are never confused.

    Example:
        agentsysperf sweep import harness/results/density_cwf_16core
    """
    _configure_logging(verbose)
    from src.sweep.import_points import import_sweep

    try:
        summary = import_sweep(
            results_dir, sweep_id=sweep_id, benchmark=benchmark,
            include_failed=include_failed,
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(
        f"Imported [bold]{summary['imported']}[/bold] cell(s) into sweep "
        f"[bold]{summary['sweep_id']}[/bold] "
        f"(agent={summary['agent']}, basis={summary['vcpu_basis']} cpuset cores)"
    )
    for s in summary["skipped"]:
        console.print(f"  [yellow]skipped {s['cell']}: {s['reason']}[/yellow]")
    console.print("View: Scaling → Concurrency Sweep in demo_app.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
