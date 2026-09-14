#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Multi-agent density test — Clearwater Forest (288 cores, 3 NUMA nodes, no SMT).
#
# Ported from run_density_test.sh (Xeon 6787P / GNR SNC3, 2026-05-21), which is
# the script that produced harness/results/density_test/. That script's three
# load-bearing decisions are preserved verbatim, because they are the reason it
# produced a real curve where src/sweep/harbor_sweep.py cannot:
#
#   1. `--n-tasks N` alongside `-n N`. harbor's -n is an asyncio.Semaphore over
#      (n_tasks x attempts); passing -n alone caps peak concurrency at the task
#      count. harbor_sweep.py omits --n-tasks, which is why every density cell
#      from 0.25 to 3.0 runs the same ~10 agents.
#   2. Pinning via an --extra-docker-compose overlay carrying the singular
#      `cpuset` key. numactl/taskset on the harbor client does NOT reach the
#      containers (the daemon does not inherit client affinity), and
#      SweepSpec.numa_policy is recorded but never applied.
#   3. `perf stat -C <range>` — system-wide over a core range. Per-process
#      hardware counters are broken host-wide on this box while the SEP driver
#      (sepint5) is loaded; -C and -a still work.
#
# WHAT CHANGED FOR THIS HOST
#   - CPU range: GNR node1 was 43-85 (43 cores). CWF nodes are 0-95 / 96-191 /
#     192-287 (96 cores each, no SMT, so no HT siblings to co-schedule).
#   - Agent: `-a oracle` instead of terminus-2 + replay proxy. The oracle agent
#     runs each task's own solution/solve.sh: deterministic, no LLM, no network,
#     $0. Measured on overfull-hbox: reward 1.0, 123.7s/trial, of which
#     agent_execution 68.9s (55.7%) is real local CPU work.
#     Use REPLAY=1 to fall back to the GNR-style terminus-2 + replay-proxy path.
#   - Paths are repo-relative rather than absolute per-user paths.
#   - Densities extend past 16, and a hard ceiling guard is enforced (see below).
#
# CEILING: Docker's default address pool allows ~28 concurrent compose projects
# on this host (/etc/docker/daemon.json is absent). harbor creates one project
# per trial, so N>27 fails with "all predefined address pools have been fully
# subnetted". The GNR run topped out at 16 and never hit this. To go higher, add
#   {"default-address-pools":[{"base":"10.200.0.0/14","size":24}]}
# to /etc/docker/daemon.json and restart dockerd. Do NOT use 10.0.0.0/8 — on a
# host whose own LAN falls inside it (as this one's does) that blackholes SSH.
# Verify any pool against `ip -4 route` first.
#
# Usage:
#   ./run_density_test_cwf.sh                      # 1,2,4,8,16 on node0
#   NODE=1 ./run_density_test_cwf.sh               # pin to NUMA node1
#   DENSITIES="4 8 16 24" ./run_density_test_cwf.sh
#   TASK=distribution-search ./run_density_test_cwf.sh
#   MAX_CONCURRENCY=64 ./run_density_test_cwf.sh   # after fixing the pool
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARBOR="${HARBOR:-$REPO/.venv/bin/harbor}"
TASK="${TASK:-overfull-hbox}"
NODE="${NODE:-0}"
DENSITIES="${DENSITIES:-1 2 4 8 16}"
REPLICATES="${REPLICATES:-1}"
OUTDIR="${OUTDIR:-$REPO/harness/results/density_cwf}"
REPLAY="${REPLAY:-0}"
PROXY_PORT="${PROXY_PORT:-4001}"
FIXTURE="${FIXTURE:-$REPO//path/to/your/fixture.jsonl}"

# Concurrency ceiling. 27 is the measured Docker address-pool limit; raise only
# after configuring default-address-pools (see header).
MAX_CONCURRENCY="${MAX_CONCURRENCY:-27}"

# Exactly 4 core events, because CWF exposes exactly 4 generic PMCs
# (/sys/bus/event_source/devices/cpu/caps/ -> 4). Asking for more forces the
# kernel to time-slice: measured 64% enabled with 10 events, 81% with 4 while
# another session ran, 100% with 4 alone. A multiplexed counter is a scaled
# estimate, and the estimate is silent — the CSV looks identical.
#
# These 4 are the only ones anything downstream reads (ipc from cycles+
# instructions, cache_miss_pct from cache-references+cache-misses). The previous
# list carried branch-instructions, branch-misses, L1-dcache-load-misses,
# LLC-load-misses, context-switches and page-faults which NOTHING consumed —
# they cost 36 percentage points of counter accuracy for values nobody read.
# Add an event here only by removing another, or by accepting multiplexing
# explicitly.
PERF_EVENTS="cycles,instructions,cache-references,cache-misses"

# ─── Resolve host topology from the kernel, never hardcoded ──────────────────
# CPUS may be overridden to pin a SUB-RANGE of a node. That is the cheap way to
# reach high agents-per-core without exceeding the ~27 Docker address-pool
# ceiling: 16 agents on 96 cores is 0.17/core, but on 16 cores it is 1.0/core.
# Any override must be a subset of the target node, or the run silently spans
# NUMA nodes and the "pinned to node N" label becomes false.
CPULIST_FILE="/sys/devices/system/node/node${NODE}/cpulist"
if [[ ! -r "$CPULIST_FILE" ]]; then
    echo "ERROR: no NUMA node $NODE on this host. Available:" >&2
    ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | xargs -n1 basename >&2
    exit 1
fi
NODE_CPUS="$(tr -d '[:space:]' < "$CPULIST_FILE")"
CPUS="${CPUS:-$NODE_CPUS}"

if [[ "$CPUS" != "$NODE_CPUS" ]]; then
    # Verify containment rather than trusting the caller.
    python3 - "$CPUS" "$NODE_CPUS" "$NODE" <<'PYCHK' || exit 1
import sys
def expand(spec):
    out = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-'); out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out
want, node, n = expand(sys.argv[1]), expand(sys.argv[2]), sys.argv[3]
stray = sorted(want - node)
if stray:
    print(f"ERROR: CPUS={sys.argv[1]} includes CPUs outside NUMA node {n}: "
          f"{stray[:8]}{'...' if len(stray) > 8 else ''}", file=sys.stderr)
    print(f"       node{n} is {sys.argv[2]}. A cross-node cpuset invalidates the "
          f"'pinned' label.", file=sys.stderr)
    raise SystemExit(1)
print(f"   CPUS override: {sys.argv[1]} ({len(want)} of {len(node)} cores on node {n})")
PYCHK
fi
NODE_CORES="$(python3 -c "
import sys
n=0
for part in '$CPUS'.split(','):
    if '-' in part:
        a,b=part.split('-'); n+=int(b)-int(a)+1
    elif part: n+=1
print(n)
")"

# ─── Resolve the task package directory (content-addressed under the cache) ──
TASK_DIR="$(ls -d "$HOME"/.cache/harbor/tasks/packages/terminal-bench/"$TASK"/*/ 2>/dev/null | head -1 || true)"
if [[ -z "$TASK_DIR" ]]; then
    echo "ERROR: task '$TASK' not found in the harbor cache." >&2
    echo "Available:" >&2
    ls "$HOME"/.cache/harbor/tasks/packages/terminal-bench/ 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi

# Per-container resource quota the task declares in [environment]. harbor turns
# `cpus` into a real cgroup cpu.max, so this bounds what one agent can consume
# regardless of how many cores the cpuset has. Read it rather than assuming 1.
TASK_CPUS="$(awk -F'=' '/^[[:space:]]*cpus[[:space:]]*=/{gsub(/[^0-9.]/,"",$2); print $2; exit}' "$TASK_DIR/task.toml" 2>/dev/null)"
TASK_MEMORY_MB="$(awk -F'=' '/^[[:space:]]*memory_mb[[:space:]]*=/{gsub(/[^0-9.]/,"",$2); print $2; exit}' "$TASK_DIR/task.toml" 2>/dev/null)"
if [[ -z "$TASK_CPUS" ]]; then
    # Do not silently default to 1 — a wrong quota silently corrupts the
    # quota-demand axis, which is the axis cross-task curves are plotted on.
    echo "ERROR: could not read [environment] cpus from $TASK_DIR/task.toml." >&2
    echo "       Cross-task density comparison needs it; refusing to guess." >&2
    exit 1
fi

mkdir -p "$OUTDIR"
COMPOSE="$OUTDIR/cpuset_node${NODE}.yml"
# The singular `cpuset` key is what Compose v5 accepts; cpuset_cpus/cpuset_mems
# are rejected. This is the same mechanism harness/configs/numa_node1.yml used
# on GNR, where CpusetCpus=43-85 was verified in the container.
cat > "$COMPOSE" <<EOF
services:
  main:
    cpuset: "$CPUS"
EOF

PROXY_PID=""
start_proxy() {
    [[ "$REPLAY" == "1" ]] || return 0
    echo "Starting replay proxy (flexible mode)..."
    # NOTE: stdout/stderr go to a FILE, not a pipe. An undrained PIPE wedges the
    # proxy after ~845 requests (measured); to a file it survives 4000+.
    "$REPO/.venv/bin/python" "$REPO/harness/scripts/replay_proxy.py" \
        --mode replay --fixture "$FIXTURE" --replay-trial ab4183d383a978b2 \
        --no-strict --port "$PROXY_PORT" > "$OUTDIR/proxy.log" 2>&1 &
    PROXY_PID=$!
    for _ in $(seq 1 15); do
        if curl -sf "http://localhost:$PROXY_PORT/healthz" >/dev/null 2>&1; then
            echo "Proxy ready (PID $PROXY_PID)."; return 0
        fi
        sleep 1
    done
    echo "ERROR: proxy failed to start; see $OUTDIR/proxy.log" >&2
    return 1
}
stop_proxy() {
    [[ -n "$PROXY_PID" ]] || return 0
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
    echo "Proxy stopped."
}

# ─── Scoped teardown between cells ───────────────────────────────────────────
#
# Removes ONLY this task's harbor containers and networks, then verifies they are
# gone before returning.
#
# The previous version filtered on `label=com.docker.compose.project` with no
# value, i.e. ANY compose project on the box, and force-removed all of them. On a
# shared host that deletes a colleague's containers mid-run. It only ever looked
# harmless because the current tenants (autoclaw, tpcds-SF6000) use plain
# `docker run` and carry no compose label — luck, not scoping.
#
# It also never removed NETWORKS. harbor creates one compose project per trial,
# each with its own bridge network, and Docker's default address pool allows only
# ~28 concurrent projects (the reason MAX_CONCURRENCY=27 exists). Leaked networks
# consume that budget across cells, so a long sweep fails late with "all
# predefined address pools have been fully subnetted" for reasons unrelated to the
# cell that finally trips it. Measured on this host: 4 orphaned
# *__<trial>__env_default networks were already resident before this fix.
#
# Naming is deterministic, verified by starting a real trial and reading it back:
#   container : <task_hash32>__<trial>__env-main-1
#   network   : <task_hash32>__<trial>__env_default
#   project   : <task_hash32>__<trial>__env
# where task_hash32 is the first 32 chars of the content-addressed task-dir name.
# Anchoring on that prefix cannot match another task, another benchmark, or
# another tenant.
TASK_PROJECT_PREFIX="$(basename "${TASK_DIR%/}")"
TASK_PROJECT_PREFIX="${TASK_PROJECT_PREFIX:0:32}"
if [[ ${#TASK_PROJECT_PREFIX} -ne 32 ]]; then
    # Refuse rather than fall back to an unscoped filter: a wrong pattern either
    # matches nothing (silently stops cleaning up, leaks accumulate) or matches
    # too much (deletes other tenants' work). Both are worse than stopping.
    echo "ERROR: could not derive a 32-char compose-project prefix from $TASK_DIR" >&2
    echo "       Refusing to run: teardown would be unscoped." >&2
    exit 1
fi

reap_containers() {
    local deadline=$((SECONDS + 120)) c n
    while :; do
        # -a so exited-but-not-removed containers are caught too.
        c="$(docker ps -a --filter "name=^${TASK_PROJECT_PREFIX}__" -q 2>/dev/null)"
        n="$(docker network ls --filter "name=^${TASK_PROJECT_PREFIX}__" -q 2>/dev/null)"
        [[ -z "$c" && -z "$n" ]] && return 0

        [[ -n "$c" ]] && xargs -r docker rm -f <<< "$c" >/dev/null 2>&1
        # Networks only after their containers are gone, else removal fails with
        # "has active endpoints" and the network survives.
        [[ -n "$n" ]] && xargs -r docker network rm <<< "$n" >/dev/null 2>&1

        if (( SECONDS >= deadline )); then
            # Report rather than abort: leftovers shrink the address-pool budget
            # for later cells, which is worth knowing, but the measurement that
            # already completed is still valid.
            echo "   WARNING: teardown incomplete after 120s —" \
                 "$(wc -w <<< "$c") container(s), $(wc -w <<< "$n") network(s) remain." \
                 "Later cells may hit the ~28-project address-pool ceiling." >&2
            return 0
        fi
        # Docker can briefly report a container attached after harbor returns.
        sleep 0.25
    done
}

run_density() {
    local N=$1 REP=$2
    local LABEL="n${N}_r${REP}"
    local CELL="$OUTDIR/$LABEL"

    if (( N > MAX_CONCURRENCY )); then
        echo "SKIP n=$N — exceeds MAX_CONCURRENCY=$MAX_CONCURRENCY (Docker address pool)." \
            | tee -a "$OUTDIR/skipped.txt"
        return 0
    fi

    echo ""
    echo "── density: $N concurrent agents (replicate $REP) ──"
    echo "   node$NODE cpuset=$CPUS (${NODE_CORES} cores) · agents/core = $(python3 -c "print(f'{$N/$NODE_CORES:.4f}')")"

    # Start from a clean cell. harbor writes each invocation under a NEW
    # timestamped subdir of -o, so a re-run leaves the previous run's
    # result.json in place and the recursive glob below counts BOTH — measured:
    # a re-run at N=1 reported "completed 2/1". Stale data inflating the trial
    # count is worse than losing it, so archive rather than merge.
    if [[ -d "$CELL" ]]; then
        local STAMP
        STAMP="$(date +%Y%m%d-%H%M%S)"
        echo "   (existing cell dir → ${LABEL}.prev-${STAMP})"
        mv "$CELL" "${CELL}.prev-${STAMP}"
    fi
    mkdir -p "$CELL"
    reap_containers

    # perf stat -C over the pinned range: system-wide on those CPUs, which works
    # even though per-process HW counters are broken while sepint5 is loaded.
    # No fixed -- sleep window: bracket the actual harbor run so the counter
    # window and the measured wall clock are the same interval.
    perf stat -C "$CPUS" -e "$PERF_EVENTS" -x, -o "$CELL/perf.csv" -- \
        bash -c '
            trap "" INT
            while [[ ! -f "$1" ]]; do sleep 0.2; done
        ' _ "$CELL/.harbor_done" &
    PERF_PID=$!

    # Memory bandwidth from the integrated memory controller. A SECOND perf
    # session on purpose: uncore events live on a different PMU than the core
    # events above, so they do not compete for the 4 generic core PMCs — both
    # sessions report 100% enabled. Verified on this host, no root and no EMON.
    #
    # SCOPE CAVEAT, load-bearing: uncore_imc is SOCKET-scoped. `-C 0-15` does NOT
    # restrict it (measured: -C 0-15 returns a figure comparable to -a), so this
    # is whole-socket traffic including other tenants, not the cpuset's share.
    # Named *_socket so that cannot be misread as per-cell attribution.
    perf stat -a -e uncore_imc/cas_count_read/,uncore_imc/cas_count_write/ \
        -x, -o "$CELL/membw.csv" -- \
        bash -c '
            trap "" INT
            while [[ ! -f "$1" ]]; do sleep 0.2; done
        ' _ "$CELL/.harbor_done" &
    MEMBW_PID=$!

    vmstat 1 > "$CELL/vmstat.txt" 2>&1 &
    VMSTAT_PID=$!
    # Node telemetry scoped to the CPUSET, not the host. ScalingAnalyzer
    # thresholds CPU_SATURATION_AVG=80 / PEAK=95; a host-aggregate reading on a
    # 16-of-288-core cpuset understates load ~18x (measured 4.27% cpuset vs
    # 0.43% host), so every cell would classify headroom_remaining even when
    # saturated. Populates cpu_avg/p95/peak, iowait, ctx_sw, runqueue, mem.
    "$REPO/.venv/bin/python" "$REPO/harness/scripts/cpuset_telemetry.py" \
        --cpus "$CPUS" --interval 1.0 --out "$CELL/telemetry.json" \
        > "$CELL/telemetry.log" 2>&1 &
    TELEM_PID=$!

    # Per-container cgroup v2: exact CPU-seconds, peak memory, disk bytes, and
    # CFS throttling per trial. This is the only per-AGENT attribution in the
    # sweep — everything else is cpuset- or socket-scoped — and it needs no PMU,
    # so it is unaffected by the SEP driver breaking per-process perf here.
    #
    # --image-filter is not optional on this shared box: a bare cgroup scan
    # picked up two other tenants' long-running containers and folded their
    # cumulative counters into the cell aggregates.
    "$REPO/.venv/bin/python" "$REPO/harness/scripts/container_telemetry.py" \
        --interval 1.0 --image-filter alexgshaw/ --out "$CELL/containers.json" \
        > "$CELL/containers.log" 2>&1 &
    CTR_PID=$!

    # Network baseline. harbor creates one Docker bridge per trial, so summing
    # every br-*/docker0 interface and taking the delta over the cell attributes
    # container traffic to THIS cell — the only network signal the sweep has, and
    # previously it had none at all. Counts other tenants' bridges too, hence the
    # *_host naming downstream.
    local NET0 NET1
    NET0="$(awk '/br-|docker0/{r+=$2; t+=$10} END{print (r+0)" "(t+0)}' /proc/net/dev)"

    local T0 T1 RC=0
    T0=$(date +%s.%N)
    set +e
    if [[ "$REPLAY" == "1" ]]; then
        OPENAI_API_KEY="sk-not-needed-local-only" NO_PROXY="*" no_proxy="*" \
        "$HARBOR" run -p "$TASK_DIR" -a terminus-2 -m "openai/gpt-4" \
            -n "$N" --n-tasks "$N" -k "$N" --yes \
            --extra-docker-compose "$COMPOSE" \
            --agent-timeout-multiplier 0.05 \
            --ak "api_base=http://localhost:$PROXY_PORT/v1" \
            -o "$CELL/jobs" \
            > "$CELL/harbor_stdout.txt" 2> "$CELL/harbor_stderr.txt"
        RC=$?
    else
        # Oracle agent: runs the task's own solution/solve.sh. Deterministic,
        # no LLM, no network, $0. -k N gives N attempts of the one task, so the
        # trial list is N and `-n N` is actually reachable.
        "$HARBOR" run -p "$TASK_DIR" -a oracle \
            -n "$N" -k "$N" --yes \
            --extra-docker-compose "$COMPOSE" \
            -o "$CELL/jobs" \
            > "$CELL/harbor_stdout.txt" 2> "$CELL/harbor_stderr.txt"
        RC=$?
    fi
    set -e
    T1=$(date +%s.%N)
    NET1="$(awk '/br-|docker0/{r+=$2; t+=$10} END{print (r+0)" "(t+0)}' /proc/net/dev)"

    touch "$CELL/.harbor_done"
    wait "$PERF_PID" 2>/dev/null || true
    wait "$MEMBW_PID" 2>/dev/null || true
    # SIGINT (not SIGKILL): the sampler traps it and writes its rollup. Killing
    # it hard would leave telemetry.json absent and silently drop the cell's
    # bottleneck evidence.
    kill -INT "$TELEM_PID" 2>/dev/null || true
    wait "$TELEM_PID" 2>/dev/null || true
    kill -INT "$CTR_PID" 2>/dev/null || true
    wait "$CTR_PID" 2>/dev/null || true
    kill "$VMSTAT_PID" 2>/dev/null || true
    wait "$VMSTAT_PID" 2>/dev/null || true
    rm -f "$CELL/.harbor_done"

    # Record the returncode. harbor has 18 `raise SystemExit(1)` sites and
    # harbor_sweep.py never checks it — a total failure there is persisted as a
    # legitimate zero-throughput operating point.
    echo "$RC" > "$CELL/harbor_returncode.txt"
    if (( RC != 0 )); then
        echo "   WARNING: harbor exited $RC — cell marked failed, not a data point."
    fi

    python3 - "$CELL" "$N" "$REP" "$NODE" "$CPUS" "$NODE_CORES" "$T0" "$T1" "$RC" \
             "$TASK" "$TASK_CPUS" "$TASK_MEMORY_MB" "$NET0" "$NET1" <<'PY'
import json, os, sys, glob
from datetime import datetime as dt
cell, n, rep, node, cpus, cores, t0, t1, rc = sys.argv[1:10]
task, task_cpus, task_mem = sys.argv[10], float(sys.argv[11]), sys.argv[12]
_net0, _net1 = sys.argv[13].split(), sys.argv[14].split()
n, rep, cores, rc = int(n), int(rep), int(cores), int(rc)
elapsed = float(t1) - float(t0)

def secs(d, key):
    p = d.get(key) or {}
    s, e = p.get("started_at"), p.get("finished_at")
    if not s or not e:            # 1 of 43 real files has both None
        return None
    return (dt.fromisoformat(e) - dt.fromisoformat(s)).total_seconds()

def _pytest_s(result_path):
    """Seconds pytest itself ran, from harbor's ctrf.json (epoch start/stop).

    Returns None when the file is absent or malformed — a missing split must not
    silently become 0.0, which would read as "no prologue" and invert the
    conclusion.
    """
    c = result_path.replace("result.json", "verifier/ctrf.json")
    try:
        s = json.load(open(c))["results"]["summary"]
        return round(float(s["stop"]) - float(s["start"]), 3)
    except (OSError, ValueError, KeyError, TypeError):
        return None

trials = []
for f in glob.glob(f"{cell}/jobs/**/result.json", recursive=True):
    try:
        d = json.load(open(f))
    except (ValueError, OSError):
        continue
    if "verifier_result" not in d:        # skip the job-level rollup
        continue
    rewards = (d.get("verifier_result") or {}).get("rewards") or {}
    # agent_execution has ONLY started_at/finished_at — there is no duration_s
    # key in any harbor version. Compute it; do not read it.
    trials.append({
        "task": d.get("task_name"),
        "reward": rewards.get("reward") if isinstance(rewards, dict) else None,
        "total_s": (
            (dt.fromisoformat(d["finished_at"]) - dt.fromisoformat(d["started_at"])).total_seconds()
            if d.get("started_at") and d.get("finished_at") else None),
        "environment_setup_s": secs(d, "environment_setup"),
        "agent_setup_s": secs(d, "agent_setup"),
        "agent_execution_s": secs(d, "agent_execution"),
        "verifier_s": secs(d, "verifier"),
        # Split the verifier into its PROLOGUE (apt-get update, curl-install uv,
        # image work) and the pytest run that does the actual verification.
        # harbor's pytest writes /logs/verifier/ctrf.json with epoch start/stop,
        # so this needs no new instrumentation — the file is already on disk in
        # every trial. Worth separating because the prologue is setup cost that
        # scales with concurrency for reasons unrelated to the workload, and the
        # two tasks differ sharply: fib's prologue is ~9-10% of its verifier,
        # hbox's is ~29-35%.
        "verifier_pytest_s": _pytest_s(f),
        "exception": bool(d.get("exception_info")),
    })

def pct(vals, p):
    """Linear-interpolating percentile. The repo's _percentile returns the max
    for any n<=20, so a field labelled p95 was actually p100."""
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)

lat = [t["agent_execution_s"] for t in trials]
completed = sum(1 for t in trials if not t["exception"])
rewards = [t["reward"] for t in trials if t["reward"] is not None]

point = {
    "concurrency_requested": n,
    "concurrency_reachable": min(n, len(trials)) if trials else 0,
    "replicate": rep,
    "numa_node": int(node), "cpuset": cpus, "cores_in_cpuset": cores,
    "agents_per_core": round(n / cores, 4),
    # The task's declared per-container quota, and the derived axis that makes
    # cross-task curves comparable. agents_per_core alone is NOT comparable
    # across tasks whose `cpus` differ: measured, fib (cpus=1) and hbox (cpus=2)
    # both knee at quota_demand == 1.0 but at agents_per_core 1.0 vs 0.5.
    # Plot cross-task curves on quota_demand; keep agents_per_core for
    # within-task reporting.
    "task": task,
    "task_cpus": task_cpus,
    "task_memory_mb": float(task_mem) if task_mem else None,
    "quota_demand": round(n * task_cpus / cores, 4),
    "quota_granted_cores": round(n * task_cpus, 2),
    "elapsed_s": round(elapsed, 2),
    "harbor_returncode": rc,
    # A cell is only usable when harbor succeeded AND exactly the expected number
    # of trials landed. More than expected means stale results merged in; fewer
    # means trials died. Either way the throughput denominator is wrong, so the
    # cell must not read as "ok".
    # `len(trials) == n` alone is NOT enough: a trial can land a result.json and
    # still have exception_info set, in which case it contributes nothing to
    # `completed` but its wall time is inside `elapsed_s`. Measured: one hung
    # trial (750s agent_execution vs ~50s normal) at n=24 pushed elapsed_s to
    # 774s and collapsed throughput 10.4 -> 1.8/min while the cell still read
    # "ok". Throughput = completed/elapsed, so a partial cell is a fabricated
    # operating point and must not reach the curve.
    "cell_status": (
        "ok" if rc == 0 and len(trials) == n and completed == n
        else "failed" if rc != 0 or not trials
        else "trial_count_mismatch" if len(trials) != n
        else "incomplete_trials"
    ),
    "expected_trials": n,
    "completed_trials": completed,
    "trials_found": len(trials),
    "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
    "throughput_per_min": round(completed / (elapsed / 60.0), 3) if elapsed > 0 else None,
    "p50_agent_exec_s": pct(lat, 50) and round(pct(lat, 50), 2),
    "p95_agent_exec_s": pct(lat, 95) and round(pct(lat, 95), 2),
    "max_agent_exec_s": max((x for x in lat if x is not None), default=None),
    # Provisioning is separable here, unlike in harbor_sweep's single span.
    "p50_env_setup_s": pct([t["environment_setup_s"] for t in trials], 50),
    # Whole-trial wall time, not just the agent phase. The signature's
    # serialization component needs it: CPU-seconds per trial are compared
    # against the trial's FULL duration (env setup + agent + verifier), because
    # off-CPU time in any phase is still off-CPU. Using agent_exec alone would
    # attribute verifier wait to the agent.
    "p50_trial_total_s": pct([t["total_s"] for t in trials], 50),
    "p95_trial_total_s": pct([t["total_s"] for t in trials], 95),
    "p50_verifier_s": pct([t["verifier_s"] for t in trials], 50),
    # Verifier split: pytest is the real verification, prologue is setup
    # (apt/curl/image). Prologue fraction is the honest way to say how much of a
    # cell's "verifier time" is not verification at all.
    "p50_verifier_pytest_s": pct([t["verifier_pytest_s"] for t in trials], 50),
    "p50_verifier_prologue_s": (
        round(_v50 - _p50, 3)
        if (_v50 := pct([t["verifier_s"] for t in trials], 50)) is not None
        and (_p50 := pct([t["verifier_pytest_s"] for t in trials], 50)) is not None
        else None
    ),
    "trials": trials,
}

# ── Fold in the perf counters, WITH the multiplexing fraction ────────────────
# 10 events on a limited number of PMCs means the kernel time-slices them:
# measured 47% enabled on this host, i.e. each value is a scaled estimate from
# under half the window. Publishing counters without that fraction is how a
# multiplexed reading gets mistaken for a direct one. `<not supported>` events
# (L1-dcache-load-misses on CWF E-cores) are recorded as None, never 0.
perf_csv = f"{cell}/perf.csv"
counters, enabled = {}, []
try:
    for line in open(perf_csv):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if len(f) < 5:
            continue
        # perf -x, field order: value, unit, EVENT, counter-run-time, %enabled.
        # The event name is f[2]; f[3] is the run-time in ns, not a name.
        raw, event, pct_enabled = f[0].strip(), f[2].strip(), f[4].strip()
        if raw in ("<not supported>", "<not counted>"):
            counters[event] = None
            continue
        try:
            counters[event] = float(raw)
            enabled.append(float(pct_enabled))
        except ValueError:
            counters[event] = None
except OSError:
    pass

cyc, ins = counters.get("cycles"), counters.get("instructions")
refs, miss = counters.get("cache-references"), counters.get("cache-misses")
point["counters"] = counters
point["counter_enabled_pct_min"] = round(min(enabled), 1) if enabled else None
point["counters_multiplexed"] = bool(enabled and min(enabled) < 99.0)
point["ipc"] = round(ins / cyc, 4) if cyc and ins else None
point["cache_miss_pct"] = round(100.0 * miss / refs, 3) if refs and miss is not None else None

# INSTRUCTIONS per completed trial is the governor-independent work metric.
# Absolute wall clock is not comparable across machines under `powersave`.
point["instructions_per_trial"] = round(ins / completed, 1) if ins and completed else None

# Cycles per trial is DELIBERATELY NOT REPORTED as a work metric. `perf -C 0-95`
# counts all 96 cores for the whole window whether or not they are busy, so at
# low occupancy `cycles` is dominated by idle cores: measured 0.05 GHz average
# per core at n=1 rising to 0.20 at n=16. Dividing that by trial count made
# "cycles/trial" FALL 4.4e11 -> 1.5e11 as concurrency rose, which reads like a 3x
# efficiency win and is purely an artifact of the denominator. Instructions do
# not have this problem (idle cores retire ~nothing) and were stable at ~0.8x
# baseline from n=2 on, which is the real "identical work per trial" evidence.
# The raw cycles count stays in counters[] for anyone who wants occupancy.
point["cycles_total_cpuset"] = cyc
point["avg_ghz_per_core_in_cpuset"] = (
    round(cyc / elapsed / cores / 1e9, 3) if cyc and elapsed and cores else None
)

# ── Fold in cpuset-scoped node telemetry ─────────────────────────────────────
# These are the six sweep_points columns ScalingAnalyzer._classify_bottleneck
# reads. Absent (not zero) when the sampler produced nothing: scaling.py treats
# a missing value as unmeasured, whereas a fabricated 0.0 reads as "no pressure"
# and would classify a saturated cell as headroom_remaining.
try:
    tel = json.load(open(f"{cell}/telemetry.json"))
except (OSError, ValueError):
    tel = {}
for k in ("cpu_avg", "cpu_p95", "cpu_peak", "iowait_pct_avg",
          "ctx_sw_per_s", "runqueue_max", "mem_avail_mb_min"):
    point[k] = tel.get(k)
point["cpu_scope"] = "cpuset" if tel else None
point["cpu_avg_host"] = tel.get("cpu_avg_host")
point["runqueue_is_host_wide"] = tel.get("runqueue_is_host_wide")
# The machine's total core count, needed because runqueue_max is host-wide
# (procs_running has no per-cpu form) while cores_in_cpuset is the pinned subset.
# Without this a consumer divides a 288-core queue depth by 16 and reports ~8x
# oversubscription that the agents did not cause.
point["logical_cpus_host"] = os.cpu_count()
point["telemetry_samples"] = tel.get("n_samples")

# Quota fill: of the CPU the containers were GRANTED, how much did they use on
# average? cpu_avg is cpuset-scoped %, so cpu_avg/100*cores = cores actually
# busy. Dividing by granted cores says whether the agents are quota-saturated or
# idle inside their allowance. Measured on the powersave-era data: fib fell
# 0.79 -> 0.54 across the ladder while hbox sat flat at 0.19-0.26 — hbox does
# NOT use its 2-CPU allowance on average, so its knee is not average-quota
# saturation. Distinguishing "hit the ceiling" from "waiting on something else"
# is the whole point of this field.
_granted = point.get("quota_granted_cores")
if tel.get("cpu_avg") is not None and _granted:
    point["quota_fill"] = round((tel["cpu_avg"] / 100.0 * cores) / _granted, 4)
    point["cores_busy_avg"] = round(tel["cpu_avg"] / 100.0 * cores, 3)
else:
    point["quota_fill"] = None
    point["cores_busy_avg"] = None

# ── Network, from the Docker bridge counters ──────────────────────────────────
# Plan item (4). Nothing in the sweep captured network before this — the coverage
# matrix had an empty row. harbor creates one bridge per trial, so the delta of
# every br-*/docker0 rx/tx across the cell is container traffic for this cell.
#
# Two honest limits, both in the field names: it sums ALL bridges, so on a shared
# box other tenants' containers are included (*_host); and a single-trial probe
# measured 0 MB for both tasks because the task images are pre-cached, so
# apt-get/curl in the verifier prologue hit nothing. A zero here means "no
# network traffic", not "not measured" — the two are distinguishable because a
# parse failure yields None.
try:
    _rx = (float(_net1[0]) - float(_net0[0])) / 1048576.0
    _tx = (float(_net1[1]) - float(_net0[1])) / 1048576.0
    point["net_rx_mb_host"] = round(_rx, 3)
    point["net_tx_mb_host"] = round(_tx, 3)
    point["net_total_mb_s_host"] = (
        round((_rx + _tx) / elapsed, 4) if elapsed > 0 else None
    )
except (IndexError, ValueError):
    point["net_rx_mb_host"] = None
    point["net_tx_mb_host"] = None
    point["net_total_mb_s_host"] = None

# ── Per-container cgroup telemetry ────────────────────────────────────────────
# The only per-AGENT attribution in the sweep. ctr_throttled_pct_periods is the
# field that separates "hit its CPU ceiling in bursts" from "genuinely idle,
# serialized on something else" — an average quota_fill cannot tell those apart,
# and they are different architectural stories.
try:
    ctr = json.load(open(f"{cell}/containers.json"))
except (OSError, ValueError):
    ctr = {}
for k in ("ctr_quota_cpus", "ctr_cpu_seconds_total", "ctr_quota_fill_mean",
          "ctr_throttled_pct_periods_mean", "ctr_throttled_pct_periods_max",
          "ctr_throttled_seconds_total", "ctr_memory_peak_mb_max",
          "ctr_disk_write_mb_total", "ctr_disk_read_mb_total",
          "ctr_cpu_pressure_full_avg10_max"):
    point[k] = ctr.get(k)
point["ctr_containers_seen"] = ctr.get("n_containers_seen")
# A container that starts AND finishes between two ticks is missed entirely, so
# the aggregates only mean something if we saw roughly the expected count. Flag
# rather than silently average over a subset.
if ctr and ctr.get("n_containers_seen", 0) < n:
    print(f"   WARNING: container sampler saw {ctr['n_containers_seen']} of {n} "
          f"containers — per-agent aggregates cover a subset only")

# ── Memory bandwidth from the IMC (uncore) ────────────────────────────────────
# perf reports cas_count_* already scaled to MiB (it applies the 64-byte cache
# line multiplier), so the value IS MiB, not a raw count. Divide by the cell wall
# to get MiB/s. SOCKET-scoped — see the caveat at the launch site.
_bw = {}
try:
    for line in open(f"{cell}/membw.csv"):
        if line.startswith("#") or not line.strip():
            continue
        fld = line.split(",")
        if len(fld) < 5:
            continue
        raw, unit, ev = fld[0].strip(), fld[1].strip(), fld[2].strip()
        if raw in ("<not supported>", "<not counted>"):
            _bw[ev] = None
            continue
        try:
            _bw[ev] = float(raw) if unit in ("MiB", "") else float(raw)
        except ValueError:
            _bw[ev] = None
except OSError:
    pass
_rd = _bw.get("uncore_imc/cas_count_read/")
_wr = _bw.get("uncore_imc/cas_count_write/")
# `if _rd` would store a MEASURED zero as None — the same falsy-vs-None
# conflation the signature analyzer guards against, but here in the PRODUCER,
# where it destroys the distinction before anything downstream can see it. A
# genuine 0 MiB reading (idle socket) is a finding; absence is a gap.
point["mem_read_mib_socket"] = round(_rd, 1) if _rd is not None else None
point["mem_write_mib_socket"] = round(_wr, 1) if _wr is not None else None
point["mem_bw_mib_s_socket"] = (
    round((_rd + _wr) / elapsed, 1)
    if (_rd is not None and _wr is not None and elapsed > 0) else None
)
point["mem_bw_scope"] = "socket" if _rd is not None else None

# ── Disk I/O from the vmstat we already collect ───────────────────────────────
# vmstat.txt has been written for every cell since this runner existed and was
# parsed by NOTHING. `bi`/`bo` are blocks-in/blocks-out per second (1 KiB
# blocks), which is the only disk-throughput signal in the sweep — iowait_pct
# alone cannot distinguish "waiting on disk" from "not doing I/O at all", and on
# this box iowait never exceeded 2.4% against a 15% threshold, so io_bound was
# unreachable. Retroactively this separates the two tasks sharply: hbox writes
# ~8x more than fib (271 vs 32 MB/s at n=24).
# Host-wide (vmstat has no cgroup scope), so on a shared box it includes other
# tenants — recorded as *_host to keep that visible.
_bi, _bo = [], []
try:
    for line in open(f"{cell}/vmstat.txt"):
        f_ = line.split()
        # Data rows start with a digit; the header repeats and must be skipped.
        if len(f_) >= 10 and f_[0].isdigit():
            _bi.append(float(f_[8]))
            _bo.append(float(f_[9]))
except OSError:
    pass
if _bi:
    # Drop the first row: vmstat's first line is an average since boot, not an
    # interval sample, and would swamp a short cell.
    _bi, _bo = _bi[1:] or _bi, _bo[1:] or _bo
    point["disk_read_mb_s_host"] = round(sum(_bi) / len(_bi) / 1024.0, 3)
    point["disk_write_mb_s_host"] = round(sum(_bo) / len(_bo) / 1024.0, 3)
    point["disk_write_mb_s_peak_host"] = round(max(_bo) / 1024.0, 3)
    point["vmstat_samples"] = len(_bo)
else:
    point["disk_read_mb_s_host"] = None
    point["disk_write_mb_s_host"] = None
    point["disk_write_mb_s_peak_host"] = None

if not tel:
    print("   WARNING: no telemetry.json — bottleneck class will be unmeasured")
json.dump(point, open(f"{cell}/point.json", "w"), indent=2)
print(f"   completed {completed}/{n}"
      f" · mean reward {point['mean_reward']}"
      f" · thr {point['throughput_per_min']}/min"
      f" · p50 {point['p50_agent_exec_s']}s · p95 {point['p95_agent_exec_s']}s")
if point["ipc"] is not None:
    mux = (f", MULTIPLEXED {point['counter_enabled_pct_min']}% enabled"
           if point["counters_multiplexed"] else "")
    # Guard every value: a cell where completed==0 leaves instructions_per_trial
    # None (it divides by completed), and ':.3g' on None raises TypeError, which
    # killed a whole run after 6 of 15 cells. A progress print must never be able
    # to abort the measurement it is reporting on.
    _ipt = point.get("instructions_per_trial")
    print(f"   IPC {point['ipc']} · cache-miss {point['cache_miss_pct']}%"
          f" · {f'{_ipt:.3g}' if _ipt else 'n/a'} instr/trial"
          f" · {point.get('avg_ghz_per_core_in_cpuset')}GHz/core avg{mux}")
if point["cell_status"] == "trial_count_mismatch":
    print(f"   WARNING: found {point['trials_found']} result.json but expected {n}"
          f" — throughput denominator is wrong; cell not usable.")
elif point["cell_status"] == "incomplete_trials":
    slow = max((t["agent_execution_s"] or 0) for t in trials)
    print(f"   WARNING: {completed}/{n} trials completed ({n - completed} raised);"
          f" slowest agent_execution {slow:.0f}s. elapsed_s includes the failed"
          f" trial's wall time, so throughput is understated; cell not usable.")
if point["concurrency_reachable"] < n:
    print(f"   WARNING: only {point['concurrency_reachable']} trials existed —"
          f" requested concurrency {n} was NOT reachable.")
PY

    docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
        > "$CELL/docker_stats.txt" 2>/dev/null || true
    reap_containers
}

# ─── Provenance: record the environment, don't assume it ─────────────────────
{
    echo "date: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo "model_name: $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2}')"
    echo "cores_total: $(nproc)"
    echo "numa_nodes: $(ls -d /sys/devices/system/node/node[0-9]* | wc -l)"
    echo "node${NODE}_cpulist: $NODE_CPUS"
    echo "cpuset_used: $CPUS  (${NODE_CORES} cores)$([[ "$CPUS" != "$NODE_CPUS" ]] && echo '  [SUB-RANGE OVERRIDE]')"
    echo "density_denominator: ${NODE_CORES} cores (agents_per_core is relative to the CPUSET, not to 288)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
    echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
    # NOTE: `lsmod | grep -q` is wrong under `set -o pipefail` — grep -q exits on
    # first match, lsmod takes SIGPIPE, and the pipeline reports failure, so the
    # probe silently answers "no" while the driver IS loaded. Read the file.
    echo "sep_driver_loaded: $(grep -cE '^sepint' /proc/modules | grep -qv '^0$' && echo yes || echo no)"
    echo "agent: $([[ "$REPLAY" == 1 ]] && echo 'terminus-2 (replay proxy)' || echo 'oracle (solve.sh, no LLM)')"
    echo "task: $TASK"
    echo "task_dir: $TASK_DIR"
    # The per-container CPU quota the task DECLARES, and that harbor enforces as
    # cgroup cpu.max. This is load-bearing for any cross-task comparison: two
    # tasks at the same agent count are NOT at the same load if their `cpus`
    # differ. Measured: overfull-hbox declares cpus=2 -> cpu.max "200000 100000",
    # circuit-fibsqrt declares cpus=1 -> "100000 100000". Both tasks knee where
    # n*cpus/cores == 1.0, i.e. the apparent "2x different density" was entirely
    # the 2x allocation. Without this field the sweep varies two axes and labels
    # one.
    echo "task_cpus: $TASK_CPUS"
    echo "task_memory_mb: $TASK_MEMORY_MB"
    echo "densities: $DENSITIES"
    echo "replicates: $REPLICATES"
    echo "max_concurrency: $MAX_CONCURRENCY"
    echo "harbor: $("$HARBOR" --version 2>&1 | tail -1)"
    echo "docker_address_pools: $(python3 -c "
import json,os
p='/etc/docker/daemon.json'
print(json.load(open(p)).get('default-address-pools','(none)') if os.path.exists(p) else '(daemon.json absent -> ~28 project ceiling)')
")"
} > "$OUTDIR/provenance.txt"
cat "$OUTDIR/provenance.txt"

if [[ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" != "performance" ]]; then
    echo ""
    echo "NOTE: governor is not 'performance'. Absolute wall-clock numbers are not"
    echo "      comparable across machines. Normalize to cycles/instructions per"
    echo "      task (perf.csv) for any cross-SKU claim."
fi

# ─── Pre-flight: refuse to start if something else holds the PMU ─────────────
# A foreign `perf stat` steals counter slots, and the kernel silently
# time-slices the events instead of erroring. Measured: two orphaned
# system-wide `perf stat -a` processes (9 events each), leaked from earlier
# l3_perf runs, held every counter at 47% enabled — so EVERY microarchitectural
# value in two completed sweeps (30 cells) was a scaled estimate from under half
# the window. Killing them restored 100% enabled. The failure is silent and the
# data looks plausible, which is exactly why this must be a hard gate rather
# than a warning.
# Match the perf BINARY, not any shell whose command line happens to contain the
# string "perf stat" — `pgrep -f` matches the whole cmdline, so a wrapper shell
# (including this script's own launcher) trips it and the gate fires with an
# empty PID list. `pgrep -x perf` matches the executable name only.
_foreign_perf="$(pgrep -x perf 2>/dev/null || true)"
if [[ -n "$_foreign_perf" ]]; then
    echo "ERROR: another 'perf' process is running and will steal PMU slots:" >&2
    for _p in $_foreign_perf; do
        sed 's/\x00/ /g' "/proc/$_p/cmdline" 2>/dev/null | sed "s/^/  $_p /" >&2; echo >&2
    done
    echo "" >&2
    echo "  Counters would be silently MULTIPLEXED (measured 47% enabled with two" >&2
    echo "  orphans present) and every IPC/cache figure would be a scaled estimate." >&2
    echo "  Kill the offending PIDs, then re-run. Override with ALLOW_FOREIGN_PERF=1" >&2
    echo "  if you accept multiplexed counters." >&2
    [[ "${ALLOW_FOREIGN_PERF:-0}" == "1" ]] || exit 1
    echo "  ALLOW_FOREIGN_PERF=1 set — continuing with multiplexed counters." >&2
fi

# EMON holds the PMU exclusively via the SEP driver; a concurrent perf session
# reads ~1e18 on every counter while both tools exit 0.
if pgrep -x emon >/dev/null 2>&1; then
    echo "ERROR: emon is running. It reserves the PMU through the SEP driver, so" >&2
    echo "       perf counters here would be garbage (~1e18 per event). Stop it:" >&2
    echo "       /opt/intel/sep/bin64/emon -stop" >&2
    exit 1
fi

# ─── Pre-pull the task image so a build never lands inside a timed cell ──────
#
# harbor builds or pulls the task image on first use, INSIDE the window this
# script brackets with perf and vmstat. That inflates whichever cell happens to
# run first — a one-off image pull attributed to a density point, and on a
# 3-replicate sweep it lands entirely in rep 1. The counters are equally
# polluted: the build's compile/IO work is counted as the workload's.
#
# Adopted from the streams path, which does this in
# benchmarks/terminal_bench/prebuild.py before any measured container starts.
# Cheap and idempotent: if the image is local, `docker image inspect` succeeds
# and nothing happens.
TASK_IMAGE="$(awk -F'=' '/^[[:space:]]*docker_image[[:space:]]*=/{gsub(/[ "'"'"']/,"",$2); print $2; exit}' \
    "$TASK_DIR/task.toml" 2>/dev/null)"
if [[ -n "$TASK_IMAGE" ]]; then
    if docker image inspect "$TASK_IMAGE" >/dev/null 2>&1; then
        echo "image: $TASK_IMAGE (already local)"
    else
        echo "image: $TASK_IMAGE — pulling BEFORE the first timed cell..."
        if ! docker pull "$TASK_IMAGE" >/dev/null 2>&1; then
            # Not fatal: harbor can still build from the Dockerfile. But say so,
            # because that build WILL be inside cell 1's counter window.
            echo "   WARNING: pre-pull failed. harbor will build on first use," \
                 "so the first cell's wall time and counters include that build." >&2
        fi
    fi
else
    echo "image: (task.toml declares no docker_image — harbor will build in-cell)" >&2
fi

start_proxy
trap stop_proxy EXIT

for rep in $(seq 1 "$REPLICATES"); do
    for n in $DENSITIES; do
        run_density "$n" "$rep"
    done
done

echo ""
echo "── summary ──"
python3 - "$OUTDIR" <<'PY'
import json, glob, sys
rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/n*_r*/point.json")):
    rows.append(json.load(open(f)))
if not rows:
    print("no cells completed"); raise SystemExit
def s(v, nd=None):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if nd is not None else str(v)

hdr = (f"{'agents':>7} {'per-core':>9} {'quota':>7} {'fill':>6} {'status':>21}"
       f" {'done':>7} {'reward':>7}"
       f" {'thr/min':>8} {'per-agt':>8} {'ret%':>6} {'p50 s':>8} {'p95 s':>8}"
       f" {'cpu%':>6} {'thr%':>6} {'IPC':>6} {'memGB/s':>8} {'wrMB/s':>7}"
       f" {'pyt s':>7} {'%enab':>6}")
print(hdr); print("-" * len(hdr))
srt = sorted(rows, key=lambda x: (x["concurrency_requested"], x["replicate"]))

# Retention is measured against the PEAK per-agent cell, not the lowest-
# concurrency cell. Measured on the 16-core sweep: n=1 and n=2 sit at 85% and 80%
# of peak because at 0.101 GHz/core the box is near-idle and those cells are
# dominated by container startup and powersave ramp, not steady-state work.
# Baselining on n=1 therefore OVERSTATES scaling. Per-agent throughput is the
# knee signal; aggregate throughput rises monotonically and hides the knee.
def per_agent(r):
    n, thr = r["concurrency_requested"], r["throughput_per_min"]
    return thr / n if thr and n else None

peak = max((r for r in srt if per_agent(r) is not None),
           key=per_agent, default=None)
peak_pa = per_agent(peak) if peak else None

for r in srt:
    n, pa = r["concurrency_requested"], per_agent(r)
    ret = 100.0 * pa / peak_pa if (pa and peak_pa) else None
    print(f"{n:>7} {r['agents_per_core']:>9.4f}"
          f" {s(r.get('quota_demand'), 3):>7} {s(r.get('quota_fill'), 2):>6}"
          f" {r['cell_status']:>21}"
          f" {str(r['completed_trials'])+'/'+str(r['expected_trials']):>7}"
          f" {s(r['mean_reward']):>7} {s(r['throughput_per_min']):>8}"
          f" {s(pa, 4):>8} {s(ret, 0):>6}"
          f" {s(r['p50_agent_exec_s']):>8} {s(r['p95_agent_exec_s']):>8}"
          f" {s(r.get('cpu_avg'), 1):>6}"
          f" {s(r.get('ctr_throttled_pct_periods_mean'), 1):>6}"
          f" {s(r.get('ipc')):>6}"
          f" {(f'{v/1024:.2f}' if (v := r.get('mem_bw_mib_s_socket')) else '-'):>8}"
          f" {s(r.get('disk_write_mb_s_host'), 1):>7}"
          f" {s(r.get('p50_verifier_pytest_s'), 1):>7}"
          f" {s(r.get('counter_enabled_pct_min'), 1):>6}")

if peak:
    print(f"\nKNEE: per-agent throughput peaks at n={peak['concurrency_requested']}"
          f" ({peak['agents_per_core']:.3g} agents/core, {peak_pa:.4f}/min/agent)."
          f" ret% is relative to THAT cell, not to n=1.")
    ramp = [r for r in srt if r["agents_per_core"] < 0.1]
    if ramp:
        print(f"      {len(ramp)} cell(s) below 0.1 agents/core are ramp region"
              f" (near-idle box, startup-dominated), not steady state.")
reps = {r["concurrency_requested"]: 0 for r in srt}
for r in srt:
    reps[r["concurrency_requested"]] += 1
if max(reps.values()) < 3:
    print(f"      Single replicate at some/all points (max {max(reps.values())}):"
          f" no variance estimate, so the knee has no confidence interval."
          f" Use REPLICATES=3 for a published curve.")

bad = [r for r in rows if r["concurrency_reachable"] < r["concurrency_requested"]]
if bad:
    print(f"\nWARNING: {len(bad)} cell(s) did not reach requested concurrency —"
          f" the density axis did not actually vary there.")
unusable = [r for r in rows if r["cell_status"] != "ok"]
if unusable:
    print(f"WARNING: {len(unusable)} cell(s) not usable: "
          + ", ".join(f"n={r['concurrency_requested']}({r['cell_status']})" for r in unusable))
mux = [r for r in rows if r.get("counters_multiplexed")]
if mux:
    lo = min(r["counter_enabled_pct_min"] for r in mux)
    print(f"NOTE: counters were MULTIPLEXED in {len(mux)} cell(s) (min {lo}% enabled)."
          f" Event values are scaled estimates, not direct counts.")
PY
echo ""
echo "Per-cell counters: $OUTDIR/n*/perf.csv    rollups: $OUTDIR/n*/point.json"
echo "Provenance:        $OUTDIR/provenance.txt"
