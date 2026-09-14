#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for platform detection — the module that manufactures the numbers
analyzers divide by.

Focus is on the failure modes that are silent in production: a wrong
microarchitecture, a fabricated bandwidth peak, and a parser that launders an
arbitrary number into a field labelled "measured". Each of those produces a
confident wrong verdict rather than an error, so they are pinned here.

CPUID values are from the kernel's authoritative table,
arch/x86/include/asm/intel-family.h.
"""

from __future__ import annotations

from src.platform.detect import (
    PlatformInfo,
    _estimate_bandwidth,
    _identify_microarch,
    _parse_cache_size,
    _parse_lscpu_cache,
    _parse_mlc_bandwidth_file,
    _safe_int,
)


# ─── Microarchitecture identification ────────────────────────────────────

def test_cpuid_identifies_clearwater_forest():
    """A model name carrying no marketing SKU must not defeat identification.

    This is the case that motivated CPUID matching: family 6 / model 0xDD
    (INTEL_ATOM_DARKMONT_X) on a host whose model_name matches no entry in
    the SKU table. SKU-string matching cannot resolve this, and previously
    returned "Intel (unknown)",
    which then fabricated a bandwidth peak ~3x too low.
    """
    uarch, source = _identify_microarch(
        "GenuineIntel", "Intel(R) Processor", family=6, model=0xDD
    )
    assert uarch == "Clearwater Forest (CWF)"
    assert source == "cpuid"


def test_cpuid_table_matches_kernel_intel_family_h():
    """Pin the (family, model) -> uarch mapping against intel-family.h."""
    expected = {
        (6, 0x6A): "Ice Lake (ICX)",
        (6, 0x8F): "Sapphire Rapids (SPR)",
        (6, 0xCF): "Emerald Rapids (EMR)",
        (6, 0xAD): "Granite Rapids (GNR)",
        (6, 0xAF): "Sierra Forest (SRF)",
        (6, 0xDD): "Clearwater Forest (CWF)",
    }
    for (family, model), uarch in expected.items():
        got, source = _identify_microarch("GenuineIntel", "whatever", family=family, model=model)
        assert got == uarch, f"family {family} model {model:#x}"
        assert source == "cpuid"


def test_cpuid_wins_over_stale_sku_string():
    """CPUID is authoritative when both could match.

    A relabelled part can carry a model string from a different generation;
    the silicon ID is the one that is not marketing.
    """
    uarch, source = _identify_microarch(
        "GenuineIntel", "Intel(R) Xeon(R) Platinum 8592+", family=6, model=0xAD
    )
    assert uarch == "Granite Rapids (GNR)"
    assert source == "cpuid"


def test_sku_string_fallback_when_cpuid_unknown():
    """An unlisted CPUID still resolves via the marketing name."""
    uarch, source = _identify_microarch(
        "GenuineIntel", "Intel(R) Xeon(R) Platinum 8592+", family=6, model=0x99
    )
    assert uarch == "Emerald Rapids (EMR)"
    assert source == "sku_string"


def test_unknown_intel_reports_source_none():
    """An unrecognized part must be labelled unresolved, not guessed at.

    ``uarch_is_known`` exists because the returned string is "Intel (unknown)",
    which is deceptively != "unknown" — the bug that made preflight report OK
    on exactly the hosts where detection had failed.
    """
    uarch, source = _identify_microarch("GenuineIntel", "Intel(R) Processor", family=6, model=0x01)
    assert source == "none"
    assert not PlatformInfo(microarchitecture=uarch, uarch_source=source).uarch_is_known


def test_default_uarch_source_is_the_unresolved_sentinel():
    """A directly-constructed PlatformInfo must not claim a source it lacks.

    The default has to be one of the three documented values, or reports render
    "via unknown" and callers face a fourth state the predicate never anticipated.
    """
    info = PlatformInfo()
    assert info.uarch_source == "none"
    assert not info.uarch_is_known


def test_amd_and_arm_still_resolve():
    """CPUID matching is Intel-only; other vendors keep the string path."""
    uarch, source = _identify_microarch("AuthenticAMD", "AMD EPYC 9654 96-Core Processor")
    assert uarch == "Zen 4 (Genoa)"
    assert source == "sku_string"

    uarch, source = _identify_microarch("ARM", "Neoverse-V2")
    assert uarch == "Neoverse V2 (Grace)"
    assert source == "sku_string"


def test_uarch_format_stays_parenthesized_abbreviation():
    """Downstream code substring-matches this string; the shape is load-bearing.

    ``_estimate_bandwidth`` and measurements/emon/probe.py both test for
    lowercase fragments like "granite" / "gnr", so a format change here breaks
    bandwidth estimation and EDP metric-file selection silently.
    """
    uarch, _ = _identify_microarch("GenuineIntel", "x", family=6, model=0xAD)
    assert uarch.endswith(")") and "(" in uarch
    lowered = uarch.lower()
    assert "granite" in lowered and "gnr" in lowered


# ─── Bandwidth estimation ────────────────────────────────────────────────

def test_unknown_platform_has_no_fabricated_bandwidth():
    """The core integrity property: no invented denominator.

    Previously an unrecognized Intel part returned 200.0 GB/s/socket. On a
    1-socket SNC3 host that became 66.7 GB/s/node against a real ~205, so
    utilization read ~3x high and the analyzer emitted false
    "bandwidth_saturation". The error is one-directional, hence only false
    positives — which is why None is required rather than a safer constant.
    """
    info = PlatformInfo(vendor="GenuineIntel", microarchitecture="Intel (unknown)", sockets=1)
    assert _estimate_bandwidth(info) is None

    info = PlatformInfo(vendor="AuthenticAMD", microarchitecture="AMD (unknown)", sockets=1)
    assert _estimate_bandwidth(info) is None

    info = PlatformInfo(vendor="Unknown", microarchitecture="unknown", sockets=1)
    assert _estimate_bandwidth(info) is None


def test_known_platform_bandwidth_scales_with_sockets():
    one = PlatformInfo(vendor="GenuineIntel", microarchitecture="Granite Rapids (GNR)", sockets=1)
    two = PlatformInfo(vendor="GenuineIntel", microarchitecture="Granite Rapids (GNR)", sockets=2)
    assert _estimate_bandwidth(two) == 2 * _estimate_bandwidth(one)


def test_clearwater_forest_bandwidth_is_plausible():
    """CWF: 12 channels x DDR5-6400. Derived from the 12 uncore_imc PMUs and a
    measured uncore_imc_0/clockticks of 798.1MHz (x8 = 6385 MT/s)."""
    info = PlatformInfo(
        vendor="GenuineIntel", microarchitecture="Clearwater Forest (CWF)", sockets=1
    )
    bw = _estimate_bandwidth(info)
    assert 550.0 <= bw <= 700.0


def test_bandwidth_source_distinguishes_estimate_from_measurement():
    """An estimate must never be reported as measured."""
    assert not PlatformInfo(dram_bw_source="estimated").dram_bw_is_measured
    assert not PlatformInfo(dram_bw_source="unknown").dram_bw_is_measured
    assert PlatformInfo(dram_bw_source="mlc").dram_bw_is_measured
    assert PlatformInfo(dram_bw_source="stream").dram_bw_is_measured


# ─── MLC parsing ─────────────────────────────────────────────────────────

def test_mlc_parser_rejects_unrelated_lines(tmp_path):
    """Regression: the old parser matched any line containing "all".

    "Installed memory" contains "all", so on a 288-core host it returned 288.0
    (the core count) stamped dram_bw_source="mlc" — a fabricated number wearing
    a measured label.
    """
    f = tmp_path / "bandwidth.txt"
    f.write_text(
        "Intel(R) Memory Latency Checker\n"
        "Installed memory: 288 GB\n"
        "Measuring idle latencies...\n"
    )
    assert _parse_mlc_bandwidth_file(f) is None


def test_mlc_parser_reads_peak_and_converts_units(tmp_path):
    """MLC reports MB/s; the field is GB/s."""
    f = tmp_path / "bandwidth.txt"
    f.write_text(
        "Measuring Peak Injection Memory Bandwidths\n"
        "ALL Reads        :      230450.5\n"
    )
    bw = _parse_mlc_bandwidth_file(f)
    assert bw is not None
    assert abs(bw - 230.45) < 0.01


def test_mlc_parser_rejects_implausible_values(tmp_path):
    """Out-of-band values mean the wrong field matched or units are off."""
    f = tmp_path / "bandwidth.txt"
    f.write_text("ALL Reads        :      12.0\n")   # 0.012 GB/s
    assert _parse_mlc_bandwidth_file(f) is None


def test_mlc_parser_survives_missing_and_binary_files(tmp_path):
    assert _parse_mlc_bandwidth_file(tmp_path / "nope.txt") is None
    binary = tmp_path / "bandwidth.bin"
    binary.write_bytes(b"\x80\x81\xfe\xff")
    assert _parse_mlc_bandwidth_file(binary) is None


# ─── Small parsers ───────────────────────────────────────────────────────

def test_safe_int_tolerates_garbage():
    """/proc/cpuinfo values feed straight into int(); garbage must not raise.

    Notably guards the "model" vs "model name" trap: a substring key match
    would hand a SKU string to this function.
    """
    assert _safe_int("221") == 221
    assert _safe_int("Intel(R) Processor") == 0
    assert _safe_int("") == 0


def test_parse_cache_size_units():
    assert _parse_cache_size("48K") == 48 * 1024
    assert _parse_cache_size("2048K") == 2048 * 1024
    assert _parse_cache_size("108M") == 108 * 1024 * 1024


def test_parse_lscpu_cache_strips_instance_count():
    assert _parse_lscpu_cache("L3 cache: 108 MiB (3 instances)") == 108 * 1024 * 1024
    assert _parse_lscpu_cache("L1d cache: 48 KiB") == 48 * 1024


# ─── Live host smoke test ────────────────────────────────────────────────

def test_detect_platform_is_self_consistent():
    """Runs against whatever host executes the suite; asserts invariants only."""
    from src.platform.detect import detect_platform

    info = detect_platform(force_refresh=True)

    assert info.physical_cores > 0
    assert info.numa_nodes >= 1
    assert info.sockets >= 1

    # Provenance must be one of the declared values, and must agree with the
    # convenience predicates the rest of the codebase gates on.
    assert info.uarch_source in ("cpuid", "sku_string", "none")
    assert info.dram_bw_source in ("mlc", "stream", "estimated", "unknown")
    assert info.uarch_is_known == (info.uarch_source in ("cpuid", "sku_string"))

    # An unknown peak must be 0.0, never a leftover fabricated value.
    if info.dram_bw_source == "unknown":
        assert info.dram_bw_total_gbs == 0.0
        assert info.dram_bw_per_node_gbs == 0.0
    else:
        assert info.dram_bw_total_gbs > 0.0
        # Per-node is the total divided across nodes.
        assert abs(
            info.dram_bw_per_node_gbs - info.dram_bw_total_gbs / info.numa_nodes
        ) < 0.01
