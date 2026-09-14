# Xeon Hardware Metrics Catalog — for Agentic-Stage Perf Characterization

Reference list for the `HardwareTelemetryPlugin` implementation. The goal is
to give every agentic stage (`classify`, `plan`, `summarize`, `rag`,
`rerank`, `revise`, `format`) a **perf fingerprint** so the WSS hypothesis is
*measured*, not asserted — e.g. "summarize → memory-bound, AMX-accelerated"
vs "classify → core-efficient, cache-resident".

## How to read this catalog

Each metric lists: what it tells you, the **granularity at which it is
hardware-true** (not just where a tool will print a number), the Intel tool,
and the raw PMU event family. Granularity tags:

- **[core]** — genuinely per-core (per-logical-CPU PMU counter)
- **[socket]** — genuinely per-socket (uncore / IMC / RAPL). Per-core
  attribution of these is a *heuristic*, not hardware truth.
- **[system]** — aggregate only

> Calibrate every threshold against a known-good run on the target SKU before
> trusting a verdict. Event names below are families; exact event codes are
> microarchitecture-specific (Sapphire/Emerald/Granite Rapids differ; codes for
> parts not yet launched are not public — do not hard-code them from this doc).

---

## 1. Top-down Microarchitecture Analysis (TMA) — the headline classifier

This is *the* answer to "is this stage memory-bound or compute-bound." Levels
sum to 100% of pipeline slots.

| Metric | Tells you | Gran. | Tool | Event family |
|---|---|---|---|---|
| L1: Retiring | useful work done (good) | [core] | toplev `-l1`, VTune uArch, emon+EDP | `UOPS_RETIRED.*`, `*SLOTS` |
| L1: Bad Speculation | wasted on mispredicts | [core] | same | `BR_MISP_RETIRED.*` |
| L1: Frontend Bound | starved for instructions | [core] | same | `IDQ_*`, `*FETCH*` |
| L1: Backend Bound | stalled on execution/memory | [core] | same | `*STALLS*` |
| L2: Backend → **Memory Bound** | stalled on cache/DRAM | [core] | toplev `-l2` | `CYCLE_ACTIVITY.*` |
| L2: Backend → **Core Bound** | stalled on execution ports/divider | [core] | toplev `-l2` | `EXE_ACTIVITY.*` |
| L3: Memory → L1/L2/L3/DRAM/Store Bound | which tier is the bottleneck | [core] | toplev `-l3` | `MEM_LOAD_RETIRED.*` |

**Caveat:** L3+ TMA needs many events → PMU counter multiplexing → scaling
error. Use `-l1`/`-l2` for the continuous pass; reserve `-l3`+ for the
targeted characterization rerun (the Mode B discussion).

---

## 2. Core efficiency

| Metric | Tells you | Gran. | Tool | Event family |
|---|---|---|---|---|
| IPC / CPI | instructions per cycle | [core] | perf, PCM | `INST_RETIRED.ANY` / `CPU_CLK_UNHALTED.THREAD` |
| Effective frequency | actual GHz under load (turbo/throttle) | [core] | turbostat, PCM | `CPU_CLK_UNHALTED.THREAD/REF_TSC` × TSC |
| Pipeline slot utilization | scheduler pressure | [core] | toplev | `*SLOTS` |
| Branch MPKI | mispredicts per 1k instr | [core] | perf | `BR_MISP_RETIRED.ALL_BRANCHES` |

---

## 3. Cache hierarchy — directly tests the WSS hypothesis

The WSS routing rule table (`configs/wss_rules.yaml`) claims tiny→L1/L2,
small→L2, medium→L3, large→DRAM. These counters *verify* that per stage.

| Metric | Tells you | Gran. | Tool | Event family |
|---|---|---|---|---|
| L1D MPKI | L1 miss rate | [core] | perf | `MEM_LOAD_RETIRED.L1_MISS` |
| L2 MPKI / hit ratio | L2 working-set fit | [core] | perf, PCM | `MEM_LOAD_RETIRED.L2_HIT/MISS`, `L2_RQSTS.*` |
| LLC (L3) MPKI / hit ratio | L3 working-set fit | [core]→[socket]* | perf, PCM | `MEM_LOAD_RETIRED.L3_HIT/MISS`, `LONGEST_LAT_CACHE.MISS` |
| LLC occupancy (CMT) | how much L3 a stage actually holds | [socket] | PCM, resctrl/RDT | Cache Monitoring Technology |
| L2/L3 prefetcher effectiveness | prefetch help vs pollute | [core] | perf | `*PREFETCH*`, `SW_PREFETCH_ACCESS.*` |

\* L3 is physically shared per socket; per-core L3 numbers are an
attribution heuristic — clean only when the stage is core-pinned (the
`core_isolation` optimization axis is the precondition).

---

## 4. Memory bandwidth & latency — [socket], never truly per-core

| Metric | Tells you | Gran. | Tool | Event family |
|---|---|---|---|---|
| Mem BW read/write (GB/s) | DRAM demand of the stage | [socket], per-channel | `pcm-memory` | IMC `UNC_M_CAS_COUNT.RD/WR` |
| Loaded memory latency (ns) | latency under the stage's own load | [socket] | `pcm-latency` (online), MLC (offline) | uncore |
| DRAM-bound % | fraction of stalls waiting on DRAM | [core] | toplev `-l3` | TMA Memory→DRAM |
| MRDIMM/DDR5 effective rate | are you near the BW ceiling | [socket] | pcm-memory | IMC |
| Memory read/write ratio | access pattern shape | [socket] | pcm-memory | IMC |

---

## 5. Interconnect & NUMA — [socket], multi-socket runs

| Metric | Tells you | Gran. | Tool | Event family |
|---|---|---|---|---|
| UPI/UXI inter-socket traffic (GB/s) | cross-socket chatter | [socket] | PCM | `UNC_UPI_*` |
| UPI utilization % | interconnect saturation | [socket] | PCM | `UNC_UPI_*` |
| NUMA remote/local access ratio | placement quality | [core]→[socket] | PCM-NUMA, perf | `OFFCORE_RESPONSE.*` (remote vs local) |

`numa_remote_access_ratio` is already the engagement counter for the
`amx_numa_hugepages` / `full_xeon` optimization profiles — same source.

---

## 6. AMX / ISA path — the differentiator for this whole effort

| Metric | Tells you | Gran. | Tool | Event family |
|---|---|---|---|---|
| AMX active cycle ratio | did the stage actually use AMX | [core] | perf, emon | AMX/TMUL engine-busy events* |
| AVX-512 vs scalar/AVX2 mix | vector path taken | [core] | perf | `FP_ARITH_INST_RETIRED.*` |
| TMUL / tile load-store activity | AMX tile pressure | [core] | perf, emon | tile load/store events* |

\* AMX PMU coverage is **limited and generation-specific**. The
`amx_active_cycle_ratio` proxy (already defined for optimization-profile
verification) is the practical signal; treat absolute AMX-op counts as
indicative, not exact, until calibrated on the target SKU.

---

## 7. Power & thermal — the perf-per-watt argument

This is what backs the CPU-vs-GPU economic comparison that runs through the
whole design.

| Metric | Tells you | Gran. | Tool | Event family |
|---|---|---|---|---|
| Package power (W) | socket draw during the stage | [socket] | RAPL via perf, PCM, turbostat | `power/energy-pkg/` |
| DRAM power (W) | memory subsystem draw | [socket] | RAPL | `power/energy-ram/` |
| Perf-per-watt | tokens·s⁻¹ / W — the real efficiency metric | [socket]/[system] | derived | throughput ÷ pkg power |
| Frequency/RAPL throttle events | did power cap distort the run | [core]/[socket] | turbostat, PCM | RAPL throttle, `CORE_POWER.*` |
| Temperature / PROCHOT | thermal throttling present | [socket] | PCM, turbostat | thermal status |

---

## 8. On-die accelerators (QAT / DSA / IAA)

| Metric | Tells you | Gran. | Tool | Source |
|---|---|---|---|---|
| QAT/DSA/IAA queue depth | offload actually receiving work | [socket] | accel-config, idxd sysfs, PCM | engine queues |
| Accelerator throughput | offload contribution | [socket] | accel-config | engine stats |

`accel_queue_depth_qat` is already the `full_xeon` engagement counter — same
source feeds both the optimization gate and stage characterization.

---

## 9. OS / scheduler context — makes the HW counters interpretable

| Metric | Tells you | Gran. | Tool | Source |
|---|---|---|---|---|
| Context switches / migrations | did the stage stay put (attribution validity) | [core] | perf software events, sched tracepoints | `sched:*` |
| Run-queue depth | scheduler pressure | [core]/[system] | perf sched, /proc | runqueue |
| Major/minor + hugepage faults | memory behavior | [core] | perf | `page-faults`, `huge` alloc |
| C-state / P-state residency | idle/turbo behavior between stages | [core]/[socket] | turbostat, PCM | MSR residency |

`hugepage_fault_ratio` is already the engagement counter for the hugepages
optimization axis.

---

## 10. The stage fingerprint (what you compute per agentic stage)

For each `StepTrace` window, join the samples in `[started_at, finished_at]`
on the cores the stage ran on, and emit:

```
stage_fingerprint = {
  tma_l1:            {retiring, frontend, backend, bad_spec}      # %
  tma_backend_split: {core_bound, memory_bound}                   # %
  ipc:               float
  dominant_cache_tier: L1|L2|L3|DRAM   # by MPKI
  mem_bw_gbs:        float             # socket, attributed
  amx_active_ratio:  float
  pkg_power_w:       float
  verdict:           "memory-bound, AMX-accel" | "core-efficient,
                      L2-resident" | ...
}
```

Expected, to be confirmed by measurement (not assumed):

| Stage | Predicted verdict (WSS hypothesis) |
|---|---|
| classify / format | Retiring-dominated, high IPC, L2-resident, ~0 AMX |
| plan | mixed, L2/L3 |
| rag | Backend/Memory, IO-bound, low AMX |
| rerank | Core-bound, L2-resident, some AMX |
| summarize / revise | Backend/Memory-bound, high DRAM BW, AMX-accel |

If the measured fingerprints **don't** match this table, that is a finding —
the WSS routing rule table needs recalibration, and the study should say so.

---

## Cross-cutting honest caveats

1. **Sampling interval vs stage duration.** A 3 ms `classify` against a 100 ms
   perf interval yields ≤1 sample — meaningless alone. Accumulate over many
   task repetitions per stage type, or use a fine interval only in the
   targeted characterization pass.
2. **Per-core attribution of shared resources is a heuristic.** L3, memory
   BW, UPI, package power are per-socket. They map to a stage cleanly *only*
   when that stage is core-pinned — i.e. `core_isolation: isolcpus_affinity`
   plus WSS routing pinning the stage type to a known pool. This is why
   stage characterization is only meaningful with routing + isolation on.
3. **Counter multiplexing.** More events than physical PMU counters → time-
   multiplexed → scaling error. Keep the continuous pass at TMA L1/L2; push
   L3+ and AMX detail to the targeted rerun.
4. **Observer effect.** emon/VTune system-wide collection perturbs the very
   latency you measure. Continuous pass = perf-stat-interval + PCM (light);
   deep pass = emon/toplev-l3 and is explicitly *not* a timed measurement.
5. **Event codes are microarchitecture-specific.** Treat every event family
   here as a name to resolve per SKU, not a portable code. Codes for unlaunched
   parts are not public; do not fabricate them from this doc.
