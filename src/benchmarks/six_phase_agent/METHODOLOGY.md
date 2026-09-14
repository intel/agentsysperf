# 6-Phase Agent Benchmark — Methodology Notes

## Thread budget (load-bearing — read before running or interpreting a sweep)

A concurrency sweep measures how the system behaves as the number of
*concurrent agents* (outer concurrency `c`) rises. That signal is only valid
if each agent uses a **bounded** number of CPU threads. Otherwise the phase
libraries fan out internally and you measure oversubscription, not concurrency.

### The confound (real bug we hit, 2026-07)

The phase workloads call libraries that default to using **all cores** for a
single call:

| Library | Phase | Default threads (192-core box) |
|---------|-------|-------------------------------|
| torch (via sentence-transformers) | retrieve | **96** |
| OpenBLAS/MKL (via numpy) | context, reason-fallback | up to all cores |
| llama.cpp `n_threads` | reason | as configured × pool size |

With no thread control, one agent already spawned ~600–700 process threads.
Under outer concurrency `c`, total threads ≈ `c × 96` — quadratic
oversubscription of a 192-core box. The observable effect:

- **Throughput COLLAPSED** as `c` rose (0.38 → 0.40 → 0.17 loops/s) instead of
  scaling. Looks like "the platform saturates immediately."
- **Per-phase latency ballooned uniformly** at high `c` (all phases ~equal),
  which is easy to misread as "the loop rebalances across phases — platform
  balance beats per-core speed."

Both are **measurement artifacts of thread oversubscription**, not
architectural findings. Publishing either would have been wrong.

### How it was diagnosed

1. **Isolated per-phase thread scaling** — ran each phase alone at c=1 vs
   c=10 threads. Every phase scaled fine or better in isolation → ruled out a
   GIL / per-phase bug.
2. **Thread-count probe** — `psutil.Process().num_threads()` during one call
   showed ~600 threads and `torch.get_num_threads() == 96`. Root cause found.

### The fix (now in the runner)

`examples/run_six_phase_sweep.py` sets, **before importing numpy/torch/faiss**:

```
OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS
  = NUMEXPR_NUM_THREADS = VECLIB_MAXIMUM_THREADS = 1   (default)
torch.set_num_threads(1)
```

and `phases.py` sizes the llama.cpp pool from the same budget
(`AGENTSYSPERF_LLAMA_THREADS`, default 1 thread/replica).

Result after the fix — a real density knee, throughput scales then saturates:

| c | throughput (loops/s) | Reason share |
|---|---------------------|--------------|
| 1 | 0.34 | 94% |
| 5 | 0.47 | 38% |
| 10 | **1.18 (peak)** | 34% |
| 25 | 0.57 (past knee) | 19% |

### The rule

**Inner-library threads × outer concurrency must not exceed core count.**
For a concurrency sweep, pin inner libraries to 1 thread so *outer concurrency*
is the variable under test. To study a different per-agent thread budget, set
`AGENTSYSPERF_INNER_THREADS=k` (and re-derive `pool_size × threads ≈ cores`),
but keep the product bounded and **report the setting** — it changes every
number in the sweep.

## Measurement isolation

Run only **one** sweep on the box at a time. Overlapping benchmark processes
contend for the same cores and contaminate results (also observed during this
work — a background sweep inflated a foreground sweep's latencies). The runner
does not enforce exclusivity; the operator must.

## What is real vs stand-in (as of 2026-07)

| Phase | Backend | Real? |
|-------|---------|-------|
| reason | llama.cpp Qwen2.5-0.5B, bounded replica pool | ✅ real |
| retrieve | sentence-transformers + FAISS over 20K vectors | ✅ real |
| admit | LiteLLM Router, shared, parallel-request semaphore | ✅ real |
| act | real subprocess (bounded compute loop) | partial |
| context | real HF tokenizer + prefill-shaped matmul | partial |
| commit | sqlite + json serialize | stand-in |

Absolute numbers are for **harness validation**, not publication, until the
partial/stand-in phases are replaced and the sweep runs on hardware with L3
perf-counter access (for the cache/mem-BW attribution).

## Claims this single-box benchmark does NOT support

- **"Per-core speed doesn't matter at scale."** Not shown. Reason's *share*
  falls with concurrency, but its *absolute* latency stays large; a faster core
  still shrinks it. Proving the per-core-speed argument requires comparing a
  fast-fewer-core vs. slow-more-core CPU (or Arm vs. Xeon) — one box cannot.
- **Cache/chiplet-tax effects.** Needs L3/mem-BW counters (perf access), absent
  in the current sandbox.
