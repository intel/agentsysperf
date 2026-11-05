#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""The code in docs/PLUGIN_DEVELOPMENT.md runs against src/protocols.py.

Plugin authors copy these snippets, so they are executed rather than eyeballed.
The 1.0.0 guide built `MeasurementRecord(run_id=...)` (a TypeError: the record
has only span_id, layer and payload) and documented Measurement, adapter and
profile methods that no runner calls.

A Python block whose first line is `# my_pkg/<file>.py` or
`# my_plugin/<file>.py` is written to that path and imported, the way an
installed plugin would be; every other block runs as a standalone script once
those modules exist. Each plugin class must then match its Protocol: the same
attributes, the same parameter names and kinds, constructible with no
arguments (that is how discovery builds it), and it must work when driven the
way the runner drives it. The toml blocks' entry points must name real groups
and resolve to the documented classes.
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys
import tomllib
from pathlib import Path

import pytest

from src import protocols
from src.protocols import (
    AnalysisResult,
    Analyzer,
    BenchmarkAdapter,
    CounterReading,
    Measurement,
    MeasurementRecord,
    OptimizationProfilePlugin,
    TaskResult,
    TaskSpec,
)
from src.runner import RunContext, track_span
from src.testing.noop_invoker import NoOpAgentInvoker

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "PLUGIN_DEVELOPMENT.md"
_FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.DOTALL | re.MULTILINE)
_MODULE_HEADER = re.compile(r"#\s*((?:my_pkg|my_plugin)/[\w/]+\.py)\s*$")
_PLUGIN_PACKAGES = ("my_pkg", "my_plugin")

# Protocol -> (required attributes, required methods), as src/protocols.py has them.
_CONTRACTS = {
    Analyzer: (("name", "input_layers"), ("analyze",)),
    Measurement: (("name", "layer"), ("start", "observe_span", "finalize_span", "stop")),
    BenchmarkAdapter: (("name", "version"), ("list_tasks", "run_task", "teardown")),
    OptimizationProfilePlugin: (("name", "is_baseline"), ("apply", "verify_engaged")),
}


def _blocks(lang: str) -> list[str]:
    return [body for tag, body in _FENCE.findall(GUIDE.read_text()) if tag == lang]


@pytest.fixture
def plugin_modules(tmp_path, monkeypatch):
    """Write the guide's my_pkg/my_plugin files under tmp_path; import them."""
    paths = []
    for body in _blocks("python"):
        m = _MODULE_HEADER.match(body.splitlines()[0])
        if m:
            path = tmp_path / m.group(1)
            path.parent.mkdir(parents=True, exist_ok=True)
            (path.parent / "__init__.py").touch()
            path.write_text(body)
            paths.append(m.group(1))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name.split(".")[0] in _PLUGIN_PACKAGES:
            monkeypatch.delitem(sys.modules, name)
    modules = {}
    for rel in paths:
        name = rel[: -len(".py")].replace("/", ".").removesuffix(".__init__")
        modules[name] = importlib.import_module(name)
    yield modules
    for name in list(sys.modules):
        if name.split(".")[0] in _PLUGIN_PACKAGES:
            del sys.modules[name]


def _plugin_classes(modules):
    for module in modules.values():
        for obj in vars(module).values():
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                yield obj


def _contract_of(cls):
    for proto, (_attrs, methods) in _CONTRACTS.items():
        if all(callable(getattr(cls, m, None)) for m in methods):
            return proto
    return None


def _params(func):
    return [(p.name, p.kind) for p in inspect.signature(func).parameters.values()
            if p.name != "self"]


def test_every_python_block_compiles():
    blocks = _blocks("python")
    assert len(blocks) >= 10
    for i, body in enumerate(blocks):
        compile(body, f"{GUIDE.name}[python block {i}]", "exec")


def test_plugin_classes_match_their_protocols(plugin_modules):
    seen = set()
    for cls in _plugin_classes(plugin_modules):
        proto = _contract_of(cls)
        if proto is None:
            continue
        seen.add(proto)
        attrs, methods = _CONTRACTS[proto]
        instance = cls()  # discovery constructs plugins with no arguments
        for attr in attrs:
            assert hasattr(instance, attr), f"{cls.__name__} lacks {attr!r}"
        for method in methods:
            assert _params(getattr(cls, method)) == _params(getattr(proto, method)), (
                f"{cls.__name__}.{method}{inspect.signature(getattr(cls, method))} "
                f"does not match {proto.__name__}.{method}"
                f"{inspect.signature(getattr(proto, method))}"
            )
        assert isinstance(instance, proto)
    assert seen == set(_CONTRACTS), f"guide has no example for {set(_CONTRACTS) - seen}"


def test_measurements_emit_records_under_a_runner(plugin_modules, tmp_path):
    found = [c for c in _plugin_classes(plugin_modules) if _contract_of(c) is Measurement]
    assert found
    for cls in found:
        probe = cls()
        with RunContext(measurements=[probe], output_dir=tmp_path / cls.__name__) as ctx:
            with track_span(ctx, "run::task-1", kind="task", node_id="task-1"):
                pass
        mine = [r for r in ctx.records if r.layer == probe.layer]
        assert mine, f"{cls.__name__} emitted no {probe.layer!r} record"
        assert all(isinstance(r, MeasurementRecord) and r.span_id == "run::task-1" for r in mine)


def test_analyzers_take_records_without_raising(plugin_modules):
    found = [c for c in _plugin_classes(plugin_modules) if _contract_of(c) is Analyzer]
    assert found
    payload = {"ipc": 0.8, "cache_miss_pct": 60.0, "elapsed_s": 1.0, "some_metric": 1}
    for cls in found:
        analyzer = cls()
        assert list(analyzer.analyze([])) == []
        records = [MeasurementRecord(span_id=f"s{i}", layer=layer, payload=payload)
                   for i, layer in enumerate(sorted(analyzer.input_layers))]
        records.append(MeasurementRecord(span_id="other", layer="unrelated", payload={}))
        results = list(analyzer.analyze(records, context=None))
        assert all(isinstance(r, AnalysisResult) for r in results)
        assert all(r.analyzer_name == analyzer.name for r in results)


def test_adapters_run_a_task(plugin_modules):
    found = [c for c in _plugin_classes(plugin_modules) if _contract_of(c) is BenchmarkAdapter]
    assert found
    for cls in found:
        adapter = cls()
        tasks = list(adapter.list_tasks(limit=1))
        assert len(tasks) == 1 and isinstance(tasks[0], TaskSpec)
        assert [t.id for t in adapter.list_tasks(include=[tasks[0].id])] == [tasks[0].id]
        assert tasks[0].id not in [t.id for t in adapter.list_tasks(exclude=[tasks[0].id])]
        result = adapter.run_task(tasks[0], agent_invoker=NoOpAgentInvoker(sleep_duration=0))
        assert isinstance(result, TaskResult) and result.task_id == tasks[0].id
        assert result.error is None, result.error
        adapter.teardown()


class _FakeTelemetry:
    """Advertises the given metric names and reads 1.0 for each."""

    name = "fake"

    def __init__(self, events):
        self.available_events = frozenset(events)

    def read_counters(self, events, *, window_s=1.0, target=None):
        return CounterReading(
            values={e: 1.0 for e in events if e in self.available_events},
            window_s=window_s,
        )


def test_profiles_apply_and_verify(plugin_modules):
    found = [c for c in _plugin_classes(plugin_modules)
             if _contract_of(c) is OptimizationProfilePlugin]
    assert found
    for cls in found:
        profile = cls()
        assert isinstance(profile.apply(sut_config={"model": "m"}), dict)
        wanted = getattr(profile, "verify_counters", {})
        verdict = profile.verify_engaged(telemetry=_FakeTelemetry(wanted), run_id="r")
        assert {"engaged", "counters", "thresholds", "failures"} <= set(verdict)
        assert verdict["engaged"] is True, verdict["failures"]
        if wanted:  # an unadvertised counter must abort, not pass
            with pytest.raises(Exception):
                profile.verify_engaged(telemetry=_FakeTelemetry(()), run_id="r")


def test_standalone_blocks_run(plugin_modules, tmp_path, monkeypatch):
    """Blocks that are not plugin files (Step 1, the unit-test template, the
    driver script) run as written, and any test_* function they define passes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.runner._TMP", str(tmp_path))  # RunContext's default output root
    ran = 0
    for i, body in enumerate(_blocks("python")):
        if _MODULE_HEADER.match(body.splitlines()[0]):
            continue
        namespace = {"__name__": f"guide_block_{i}"}
        exec(compile(body, f"{GUIDE.name}[python block {i}]", "exec"), namespace)
        for name, func in namespace.items():
            if name.startswith("test_") and callable(func):
                func()
        ran += 1
    assert ran >= 3


def test_entry_points_name_real_groups_and_classes(plugin_modules):
    groups = {v for k, v in vars(protocols).items() if k.startswith("ENTRY_POINT_")}
    checked = 0
    for body in _blocks("toml"):
        for group, entries in tomllib.loads(body).get("project", {}).get("entry-points", {}).items():
            assert group in groups, f"unknown entry-point group {group!r}"
            for name, target in entries.items():
                module_name, _, attr = target.partition(":")
                cls = getattr(importlib.import_module(module_name), attr)
                assert cls.name == name, f"{target}: name {cls.name!r} != entry point {name!r}"
                checked += 1
    assert checked >= 6
