# Getting Started with AgentSysPerf — Day 1 to Day 4

This guide walks you through your first week with AgentSysPerf, from setup to understanding insights.

## Day 1: Install & Explore

### 1a. Prerequisites

```bash
# Check Python version (must be 3.12 or 3.13)
python3 --version

# If not 3.12+, install it
apt install python3.12-venv  # Ubuntu/Debian
# or: brew install python@3.12  # macOS
```

### 1b. Clone & Setup

```bash
git clone https://github.com/intel-sandbox/agentsysperf.git
cd agentsysperf

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e .
```

### 1c. Discover Plugins

```bash
agentsysperf list
# Output: 6 benchmarks, 4 measurements, 7 analyzers, 1 telemetry
```

See each group:

```bash
agentsysperf benchmarks list      # Available workloads
agentsysperf measurements list    # Available probes (L1, L3, perfspect, etc.)
agentsysperf analyzers list       # Available insights (cpu_bound, cache, scaling, etc.)
agentsysperf profiles list        # Available tuning profiles
agentsysperf telemetry list       # Counter sources + which profile counters are unverifiable here
```

---

## Day 2: Run Your First Benchmark

### 2a. Synthetic CPU (No API key, no Docker)

Fastest way to populate the demo app with baseline data:

```bash
agentsysperf run --benchmark synthetic_cpu --num-tasks 9
# Takes ~3 minutes. Output: Hardware Baselines tab filled.
```

Check the run:

```bash
agentsysperf db ls                      # List runs
agentsysperf db show synthetic_cpu_*    # Details of one run
```

### 2b. Scale Experiment (Still offline)

Populate the Concurrency Sweep tab:

```bash
agentsysperf sweep run --dry-run
# Seconds. Prints the plan, then writes one cell per density point.
```

**`--dry-run` cells are SYNTHETIC modeled points, not measurements.** No agents,
no LLM, no containers — the numbers come from a model, and are persisted with
`data_source=synthetic` so the dashboard badges the sweep and suppresses its
per-task comparison. The point is to exercise the pipeline and light up the tab;
never cite these as results.

A real sweep drops `--dry-run` and needs Harbor, a container runtime, and a
replay fixture recorded with the same agent you replay. It runs for hours, so
the command quotes the trial count and asks for confirmation first:

```bash
agentsysperf sweep run --fixture path/to/fixture.jsonl -d 0.5 -d 1.0 -d 2.0
```

See `agentsysperf sweep run --help` for the full option set (basis, NUMA policy,
attempts, EMON collection, local dataset path).

### 2c. Launch Dashboard

```bash
streamlit run demo_app.py --server.port 7860 --server.address 0.0.0.0
# Browse: http://<host>:7860
```

**Tabs you should see populated:**
- ✅ Hardware Baselines (from synthetic_cpu — real measurements)
- ⚗️ Scaling/Concurrency Sweep (from the `--dry-run` sweep — **synthetic**, badged as such)
- ✅ Hardware Analysis (system detection + profiles)
- ⚠️ Agent Performance (empty; needs real terminal-bench data)
- ⚠️ Recommendations (partial; needs agent runs)

**You're done with Day 2.** You have a working benchmark suite running offline.

---

## Day 3: Understand Measurements & Analyzers

### 3a. Read the Docs

Key documents:

| Document | Purpose |
|----------|---------|
| `docs/ANALYZER_GUIDE.md` | Explain the 7 core analyzers (+ the Intel-only emon plugin), use cases, verdicts, decision trees |
| `docs/methodology/DESIGN.md` | Deep design: integrity principles, measurement layers, error handling |

### 3b. Understand Measurement Layers

```
L1 (Always available):    Per-span CPU %, memory RSS, elapsed time
L3 (Needs perf access):   Per-span IPC, cache miss rates, branch miss rates
L5 (Needs Intel tools):   Per-span TMA, RAPL, PMU events (optional)
```

Check which layers are available on your system:

```bash
# Read a run
agentsysperf db show synthetic_cpu_* | grep -A 20 measurements

# Output example:
# measurements:
#   l1: 9 records (per-span CPU %, memory)
#   l3: 9 records (per-span IPC, cache miss)
#   l1_system: 5 records (node CPU/runqueue/memory)
```

### 3c. Understand Analyzers

**Quick decision tree:**

```
My goal: Optimize _________?

├─ Latency of one agent
│  └─ Measure L1+L3, read: cpu_bound (core-bound vs memory-bound?)
│     ├─ If core_bound → look for hotspots; optimize hot loops
│     └─ If memory_bound → read memory_bandwidth analyzer; tune NUMA/cache/quantization
│
├─ Concurrency headroom
│  └─ Run scaling sweep, read: scaling (knee = agents/vCPU saturation point?)
│     ├─ If knee < 1.0 → saturated; optimize single agent OR reduce concurrency
│     └─ If knee > 1.0 → has headroom; optimize memory/I/O subsystem
│
├─ Per-phase resource breakdown (Reason vs Act vs Retrieve?)
│  └─ Tag spans with phase=, run benchmark, read: phase_profiler + breakdown
│     ├─ If Reason has low IPC → optimize LLM inference (quantization, batching)
│     └─ If Retrieve has high cache miss → optimize embeddings (NUMA, quantization)
│
└─ Memory leaks?
   └─ Measure L1, read: memory_leak
      ├─ If leak_detected → investigate object retention
      └─ If stable → memory is OK
```

**The 3 most useful analyzers for agentic workloads:**

1. **cpu_bound** — Is my bottleneck core, memory, or frontend? (Use this first)
2. **scaling** — Where do I saturate? (Use this to find headroom)
3. **phase_profiler** — Which phase (Reason/Act/Retrieve) is slowest? (Use this to prioritize optimization)

---

## Day 4: Run Real Agent Benchmarks

### 4a. Get an API Key

To run Terminal-Bench (the real agentic workload), you need an LLM:

```bash
export OPENAI_API_KEY=sk-...
```

Or use a local LLM server (vLLM, llama-cpp-python).

### 4b. Run Terminal-Bench

```bash
agentsysperf run -b terminal-bench --num-tasks 2 --model gpt-4o-mini
# Takes ~2–5 minutes (depends on LLM latency).
# Result: Agent Performance & Recommendations tabs now filled.
```

### 4c. Record & Replay (Optional)

Record an LLM trajectory once, replay it free thereafter:

```bash
# Step 1: Record (costs tokens, time, API calls)
agentsysperf run -b terminal-bench -n 2 --record /tmp/fixture.jsonl

# Step 2: Replay (free, deterministic)
agentsysperf run -b terminal-bench -n 2 --replay /tmp/fixture.jsonl

# Step 3: Replay again (still free)
agentsysperf run -b terminal-bench -n 2 --replay /tmp/fixture.jsonl
```

### 4d. Generate Report

```bash
agentsysperf report <run_id> --format md
agentsysperf report <run_id> --format pptx
```

Includes measurements, analyzer verdicts, and recommendations.

---

## Decision Tree: "What Should I Measure?"

### Goal: Understand why my agent is slow

**Step 1: Measure L1 + L3**
```bash
agentsysperf run -b my_benchmark --num-tasks 3
```

**Step 2: Read verdicts**
```bash
agentsysperf db show <run_id>
# Look for analyzer verdicts: cpu_bound? cache? memory_leak?
```

**Step 3: Decide next action**

| Verdict | Interpretation | Next Step |
|---------|-----------------|-----------|
| `core_bound` | CPU frontend/backend saturated | Profile with VTune; check for hotspots |
| `memory_bound` | Memory subsystem limiting throughput | Run scaling sweep; see MemoryBandwidthAnalyzer patterns |
| `frontend_starved` | Branch prediction exhausted | Reduce conditional branches; use PGO |
| `cache_l3_bound` | L3 is the bottleneck | Tune data layout; use cache partitioning (RDT) |
| `memory_leak` | RSS growing | Check object lifetime; profile allocations |

### Goal: Find where I can add more agents before saturation

**Step 1: Run concurrency sweep**
```bash
# Drop --dry-run for a real (measured) sweep; --dry-run cells are synthetic and
# their knee is a property of the model, not of your box.
agentsysperf sweep run --dry-run
```

**Step 2: Read scaling analyzer verdict**
```bash
# CLI output includes knee (saturation point) and bottleneck
# Example: "knee=1.5 agents/vCPU, bottleneck=cpu_bound"
```

**Step 3: Decide**

| Knee | Interpretation | Action |
|------|-----------------|--------|
| `< 1.0` | Saturated at < 1 agent/vCPU | Optimize single agent (quantization, AMX, better algorithm) |
| `1.0–2.0` | Moderate headroom | Tune workload distribution; NUMA pinning |
| `> 2.0` | Lots of headroom | Check I/O, network, or external service latency |

### Goal: Understand per-phase resource consumption

**Step 1: Tag spans with phases**
```python
with track_span("inference", phase="reason"):
    response = agent.invoke(query)

with track_span("tools", phase="act"):
    result = tool.execute()
```

**Step 2: Run benchmark**
```bash
agentsysperf run -b my_benchmark --num-tasks 5
```

**Step 3: Read breakdown + phase_profiler**
```bash
agentsysperf db show <run_id>
# Look for: per-phase wall-clock %, CPU %, IPC, cache miss
```

**Step 4: Optimize highest-impact phase**

| Phase | High IPC (1.5+) | Low IPC (< 1.0) | Recommendation |
|-------|-----------------|-----------------|-----------------|
| reason | ✓ | ✗ | Inference is efficient; not bottleneck |
| reason | ✗ | ✓ | Memory-bound; try quantization, batching |
| act | ✓ | ✗ | Tool execution is efficient; not bottleneck |
| act | ✗ | ✓ | Memory-bound tool execution; tune I/O, caching |
| retrieve | ✗ | ✓ | Embedding/search bottleneck; optimize index |

---

## Quick Commands Reference

```bash
# Discovery
agentsysperf list
agentsysperf benchmarks list

# Benchmarking
agentsysperf run --benchmark synthetic_cpu --num-tasks 9
agentsysperf run -b terminal-bench --num-tasks 2 --model gpt-4o-mini
agentsysperf sweep run --dry-run              # synthetic cells (pipeline check)
agentsysperf sweep run --fixture <f.jsonl>    # real, measured sweep (hours)

# Analysis
agentsysperf db ls                            # List runs
agentsysperf db show <run_id>                 # Details
agentsysperf analyze /path/to/measurements/  # Re-analyze
agentsysperf report <run_id> --format md     # Generate report

# Dashboard
streamlit run demo_app.py --server.port 7860 --server.address 0.0.0.0

# Recording & Replay
agentsysperf run -b terminal-bench -n 2 --record /tmp/fixture.jsonl
agentsysperf run -b terminal-bench -n 2 --replay /tmp/fixture.jsonl
```

---

## Troubleshooting: Common Issues

### "measurements 3/5 active" — Why is L3 missing?

**Cause:** `perf` access requires `kernel.perf_event_paranoid <= 1` or root.

**Fix:**
```bash
# Option A: Lower perf_event_paranoid (requires sudo)
sudo sysctl kernel.perf_event_paranoid=1

# Option B: Run as root
sudo agentsysperf run --benchmark synthetic_cpu --num-tasks 3

# Option C: Check permissions
cat /proc/sys/kernel/perf_event_paranoid
# Output: 2 or higher means `perf` blocked for non-root
```

### "ERROR: run_task: agent_invoker not found"

**Cause:** You're running a benchmark that needs an LLM (Terminal-Bench), but no agent provider is configured.

**Fix:**
```bash
# Use synthetic_cpu (no LLM needed)
agentsysperf run --benchmark synthetic_cpu --num-tasks 3

# Or set up LLM (OpenAI)
export OPENAI_API_KEY=sk-...
agentsysperf run -b terminal-bench --num-tasks 2 --model gpt-4o-mini
```

### "Dashboard shows empty tabs"

**Cause:** No data in the store, or wrong data source.

**Check:**
```bash
# Is the store empty?
agentsysperf db ls
# Output: (empty) — yes

# Populate it
agentsysperf run --benchmark synthetic_cpu --num-tasks 3
agentsysperf sweep run --dry-run

# Restart dashboard
streamlit run demo_app.py --server.port 7860 --server.address 0.0.0.0
```

### "verify_engaged() failed: counter not advertised"

**Cause:** You tried to use an Xeon optimization profile (e.g., `amx_only`), but your system doesn't support it or telemetry can't read the counter.

**Fix:**
```bash
# Use the base profile (always works)
agentsysperf run --benchmark synthetic_cpu --num-tasks 3 --profile base

# Or check your hardware
lscpu | grep -i avx
lscpu | grep -i amx
```

### EMON hardware analysis

Deep Intel TMA/EDP telemetry (EMON) is not part of the vendor-neutral core. It
is provided by the separate, unreleased, Intel-only `agentsysperf-emon` plugin,
which ships its own setup and troubleshooting notes (SEP driver, pyEDP
dependencies, collection drivers). Install it on an Intel host to register the
`emon` measurement and analyzer.

### Terminal-Bench: "Docker compose build" fails with network timeout

**Cause:** Docker Compose v5+ requires the `buildx` plugin. Without it, compose
falls back to the legacy builder which does not reliably inherit proxy settings
from `~/.docker/config.json`. The symptom is "Could not connect to
archive.ubuntu.com" during `apt-get update` inside the container build.

**Verify:**
```bash
docker buildx version        # should print a version; "unknown command" = missing
docker compose version       # v5+ requires buildx
```

**Fix:**
```bash
# Option A: Install buildx plugin (Ubuntu/Debian)
sudo apt-get install docker-buildx-plugin

# Option B: Manual install
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/buildx/releases/latest/download/buildx-v0.20.1.linux-amd64 \
  -o ~/.docker/cli-plugins/docker-buildx
chmod +x ~/.docker/cli-plugins/docker-buildx
```

After installing, verify: `docker buildx version` prints a version, and
`agentsysperf run -b terminal-bench --num-tasks 1` can build the task image.

**Corporate proxy note:** If your host uses a corporate proxy, ensure
`~/.docker/config.json` has the proxy configured:
```json
{
  "proxies": {
    "default": {
      "httpProxy": "http://your-proxy:port",
      "httpsProxy": "http://your-proxy:port",
      "noProxy": "localhost,127.0.0.1"
    }
  }
}
```

### L3 perf warnings: "physically impossible" counter values

**Cause:** Another process holds the PMU (Performance Monitoring Unit). The
common culprit is a stale `perf stat -a` process; any exclusive-PMU collector
(such as the optional Intel-only EMON plugin) does the same.

**Diagnose:**
```bash
# Check for stale perf processes
ps aux | grep "perf stat"
```

**Fix:**
```bash
# Kill the stale perf process, or stop the other exclusive-PMU collector
kill <pid>
```

The warnings are correct behavior: agentsysperf detects and rejects impossible
counter values rather than reporting bad data. On a clean host with no PMU
contention, these warnings do not appear.

---

## Next Steps

1. **Read the Analyzer Guide** (`docs/ANALYZER_GUIDE.md`) to understand all the analyzers
2. **Explore examples** (`examples/`) for advanced workflows
3. **Write a plugin** (`docs/PLUGIN_DEVELOPMENT.md`) to extend the suite
4. **Check methodology** (`docs/methodology/DESIGN.md`) for deep design

---

## Resources

| Resource | Purpose |
|----------|---------|
| `README.md` | Overview, quick start, all commands |
| `docs/ANALYZER_GUIDE.md` | Deep dive into the 7 core analyzers (+ emon plugin) |
| `docs/PLUGIN_DEVELOPMENT.md` | How to write plugins |
| `docs/methodology/DESIGN.md` | 16-section design doc |
| `src/protocols.py` | Protocol definitions (source of truth) |
| GitHub Issues | Report bugs, request features |

---

## Quick Wins (30-minute tasks)

- [ ] Run synthetic_cpu, check measurements
- [ ] Run scaling sweep, understand knee
- [ ] Read one analyzer (e.g., cpu_bound)
- [ ] Tag a span with `phase="reason"`, run benchmark, read phase_profiler verdict
- [ ] Generate a report in Markdown

---

Happy benchmarking! Questions? See the README's *Integrity principles* section, and
`docs/methodology/DESIGN.md` for the design rationale.
