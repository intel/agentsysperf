# Platform Capability & Bandwidth Policy

How AgentSysPerf behaves on hardware it was not written on — GNR, CWF, SRF,
EPYC, ARM — and what it is allowed to claim on each.

Written after a Clearwater Forest bring-up (2026-07-28) surfaced a compounding
9.2x error in bandwidth-utilization verdicts on an unrecognized platform.

## The principle

> A missing number must stay missing. A fabricated number that looks like a
> measurement is worse than an error, because nothing downstream can tell.

AgentSysPerf already states this as *measured-not-assumed* and
*abort-don't-degrade*. This document applies it to platform detection.

## 1. Microarchitecture identification

**Policy: identify by CPUID family/model. Marketing strings are a fallback, not
the primary key.**

`platform/detect.py` keys on `(cpu_family, cpu_model)` from `/proc/cpuinfo`,
matched against a table verified against the kernel's
`arch/x86/include/asm/intel-family.h`. SKU-string matching remains as a
secondary path for parts absent from the table.

Why: some hosts report a `model name` carrying no marketing SKU at all
(e.g. `Intel(R) Processor`). Every marketing-substring match fails, and
the old code returned `"Intel (unknown)"` — which then selected a fabricated
bandwidth default. On a CWF host this produced a peak 3x too low.

Rules:
- **Never add a CPUID pair that has not been verified** against kernel headers
  or confirmed on real silicon. One next-generation part is deliberately
  *absent*: the 6.17 and 7.0 headers agree on its IFM value but disagree on the
  macro name, so the naming is unconfirmed. An unlisted part degrades to
  `"Xeon (unknown gen)"`, which is honest. A wrong entry is not.
- **Gate on `uarch_source`, never on the string.** `microarchitecture` carries
  sentinels like `"Intel (unknown)"` that are literally `!= "unknown"`. Use
  `PlatformInfo.uarch_is_known`. The preflight check previously used the string
  test and therefore reported OK on exactly the hosts where detection failed.
- **The `"Name (ABBR)"` format is load-bearing.** `_estimate_bandwidth` and
  `measurements/emon/probe.py` substring-match lowercase fragments (`"granite"`,
  `"gnr"`). Changing the shape silently breaks bandwidth estimation and EDP
  metric-file selection.

## 2. Memory bandwidth

**Policy: three provenance tiers, and the tier is always recorded.**

| `dram_bw_source` | Meaning | May drive a utilization verdict? |
|---|---|---|
| `mlc`, `stream` | Measured on this host | Yes |
| `estimated` | Derived from channel count x DDR rate for a *recognized* uarch | Yes, with the estimate labelled |
| `unknown` | Platform not recognized | **No** — peak is `0.0`, verdict is skipped |

There is no "conservative default" tier. The old `200.0 * sockets` fallback was
the entire bug: on 1-socket SNC3 it became 66.7 GB/s/node against a real ~205.

Consumers **must** check before dividing:

```python
if platform.dram_bw_total_gbs > 0:
    utilization = measured_bw / platform.dram_bw_total_gbs
else:
    ...  # skip the verdict; log why
```

`PlatformInfo.dram_bw_is_measured` distinguishes tier 1 from tier 2 for callers
that need a genuinely measured peak.

### Scope must match

**Policy: divide a bandwidth number by a peak of the same scope.**

PerfSpect's `memory bandwidth total (MB/sec)` at `--scope system` is
**machine-wide**. Dividing it by a **per-node** peak overstates utilization by
the NUMA node count — 3x under SNC3, independent of any hardware knowledge.
Combined with a fabricated denominator this reached **180% utilization** on
this host, which still silently produced a "severe" verdict.

The PerfSpect plugin now records `scope` alongside its metrics so consumers
cannot guess wrong. Compare system-scope numerators against
`dram_bw_total_gbs`, not `dram_bw_per_node_gbs`.

Note on per-node attribution: all 12 `uncore_imc_*` PMUs on this host report
`cpumask 0` (socket-wide). Per-NUMA-node bandwidth under SNC is therefore an
*assumption*, not a measurement. Do not present it as measured.

### Getting a real peak

Preference order:
1. `sudo perfspect benchmark --memory` — measures latency and bandwidth directly.
2. Intel MLC — parsed from `perf-runs/`, `/tmp/mlc_results/`, `~/mlc_results/`.
3. Uncore IMC counters (`cas_count_read`/`cas_count_write`) — readable **without
   root** at `perf_event_paranoid <= 1`. These give real *utilization*, and
   `clockticks` yields the DDR transfer rate, from which a theoretical ceiling
   follows. Not a sustained peak.

The MLC parser requires a labelled row (`ALL Reads`, `Peak Injection
Bandwidth`) and a plausibility band. It previously matched any line containing
`"all"` — including `"Installed memory"` — and returned the core count (288.0)
stamped `source="mlc"`.

## 3. Optimization profiles and ISA prerequisites

**Policy: a profile whose required ISA is absent refuses to apply.**

Five of the six reference profiles target AMX. On a host without AMX (any
E-core part — SRF, CWF — or non-Intel), an AMX profile would exercise the
AVX-512 or scalar fallback while labelling the result `amx_only`. That is not a
degraded measurement; it is a mislabelled one, and it would silently corrupt
any GNR-vs-CWF comparison.

Each `ProfileSpec` declares `requires` (matched against `PlatformInfo.has_*`):

- `unsupported_on(platform_info)` -> list of missing capabilities. Callers that
  want to *skip* inapplicable profiles consult this first.
- `apply()` raises `UnsupportedProfileError` when a requirement is unmet.
- `agentsysperf profiles list` shows applicability for the current host.

`base` declares no requirements — it is the portable comparison point, and is
the correct arm to use on a machine without AMX.

**What to run on a non-AMX host:** `base` only, and report it as a
single-configuration characterization rather than a profile sweep. A one-arm
"sweep" is not a comparison. The AMX profiles are not broken on such hosts —
they are inapplicable, and the tool now says so instead of producing numbers.

Caveat, unchanged by this document: these profiles are currently **inert** in
the run path. `run_driver.py` never applies a profile; only
`examples/run_profile_verify.py` exercises them. The gate above constrains what
happens when they are wired up; it does not wire them up.

## 4. Run provenance

**Policy: record the knobs that change the numbers.**

`run_driver._run_metadata` captures, per run:

- **Platform** — uarch + `uarch_source`, CPUID family/model/stepping, cores,
  sockets, NUMA nodes, `dram_bw_total_gbs`, `dram_bw_source`.
- **Environment** — `cpu_governor`, `perf_event_paranoid`, `nmi_watchdog`,
  `turbo_enabled`.

Misconfiguration **warns but does not block** — a run under
`governor=powersave` still completes, and the report says so. Rationale: a
blocked run yields nothing, while a labelled run stays interpretable. The
markdown report flags a non-`performance` governor inline.

`hardware_sku` has no schema DEFAULT (migration 0008). It was
`'Intel Xeon Platinum 8592+'` — the SKU of the host the schema was written on —
so any run that did not set it was labelled Emerald Rapids regardless of the
silicon it ran on. `numa_policy` is `NULL` rather than `"unpinned"`, because
nothing in the run path pins anything; the sweep path still sets it where it
genuinely applies.

Run `agentsysperf preflight` before any run you intend to cite; `--fix` applies
the sudo remediations after showing them and prompting.

## 5. Adding a new platform

1. Read `/proc/cpuinfo` for `cpu family` / `model`.
2. Confirm the pair in `arch/x86/include/asm/intel-family.h` (or vendor docs).
   Cross-check with `perfspect metrics --list`, which identifies parts by CPUID.
3. Add to `_INTEL_CPUID_UARCH` in `platform/detect.py`.
4. Add a bandwidth branch in `_estimate_bandwidth` **only** with a defensible
   channel-count x transfer-rate derivation; cite it in a comment. Otherwise
   leave it out and let it report `unknown`.
5. Add the pair to `test_cpuid_table_matches_kernel_intel_family_h`.
6. Run `agentsysperf preflight` and confirm the uarch resolves.

Do not skip step 4's "otherwise" clause. An absent estimate degrades one
verdict; a wrong estimate corrupts every verdict that divides by it.

## Known gaps

- **TMA is unavailable on E-core parts.** `perfspect metrics --list` reports 47
  metrics on CWF and **zero** TMA buckets, so `tma_memory_bound` is legitimately
  `None` there. Analyzers must not depend on TMA as their only corroborating
  signal.

  Two distinct problems used to be conflated here, and the fix is worth
  recording because it is the shape this policy exists to prevent. The PerfSpect
  plugin's `label_map` mapped only the four *top-level* TMA buckets, so no
  memory-bound key was produced **on any platform** — a gap in that map, not a
  property of the silicon. `memory_bandwidth.py` then hardcoded
  `"tma_memory_bound": None` and cited the CWF observation as the reason,
  which baked one platform's capability into a platform benchmark and silently
  disabled the TMA clauses of `cross_numa_traffic` and `bandwidth_saturation`
  everywhere. The plugin now maps `TMA_..Memory_Bound(%)` (verified present at
  16.29% on a P-core host, perfspect 3.17.0) and the analyzer reads it. Absence
  is now produced by the data: `_parse_metrics` only sets keys for labels
  actually present in the CSV, so parts without TMA rows yield no key and the
  signal reads `None` because it was not measured. **Never restore a hardcoded
  capability constant** — gate on the presence of the measurement instead.
- **NaN is a measurement outcome, not a value.** PerfSpect emits `NaN` for
  metrics it cannot compute on a given part (e.g. `TMA_......MEM_Bandwidth(%)`),
  and `float("NaN")` does not raise. A stored NaN fails every `>` comparison
  silently, which reads downstream as "measured and below threshold". Non-finite
  values are dropped at parse so the key is simply absent.
- **PerfSpect metrics need root**, but the plugin's `_check_available()` only
  tests for the binary and `perf_event_paranoid`, so it reports
  `available: True` as non-root and then collects nothing.
- **Analyzer thresholds are uncalibrated** (0.7 / 0.5 utilization, 30 / 40 TMA).
  Deferred by the v1.0 plan (quality-8 / features-3 / positioning-5); the fixes
  here make the *inputs* honest, not the thresholds correct.
