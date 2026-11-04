#!/usr/bin/env python3
"""Run the microVM density matrix: one vmfleet.py cell after another.

    run_experiment.py --quick                 # 1,4 x burst,staggered x cpu,dsaiaa, 120 s cells
    run_experiment.py --screen                # phase 1: none/cpu/dsaiaa over every density and
                                              # launch pattern, then dsa/iaa at 1,8,24,36
    run_experiment.py --confirm --densities 1,16,36   # phase 2: every arm and pattern, 3 reps
    run_experiment.py --full                  # every arm x density x pattern
    run_experiment.py --densities 1,8 --engines cpu,dsaiaa --launch burst --regime real --duration 900

Before each cell the socket's 2 MiB huge page pool is sized to the cell's
guests (vmfleet.py refuses a cell the pool cannot hold) and the node's free
memory beside the pool is checked against what the VMMs, the daemons and
the node-bound store need, and the templates are read back into the page
cache (growing the pool evicts them); after the cell the pool must come
back whole and nothing of the cell may be left running. A cell
whose result.json exists is skipped, so an interrupted run resumes with the
same command; --rerun repeats them. Refused and failed cells are retried on
the next run. The driver and the controller run on the socket that is not
measured; the sandboxes bind themselves to the measured socket.

Cells are ordered rep -> engine (none, cpu, iaa, dsa, dsaiaa) -> density ->
launch pattern, so the arms without kernel copy offload finish first and
every rep of a confirmation run spans the whole matrix.

Output: <output>/<label>/ per cell (vmfleet.py's result.json plus cell.log),
<output>/run.json (arguments, host.env, binary versions) and
<output>/cells.json (one record per cell: status, exit code, huge page pool,
timing, the printed summary), rewritten after every cell.
"""
import argparse
import atexit
import fcntl
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VMFLEET = f"{HERE}/vmfleet.py"
sys.path.insert(0, HERE)
from vmfleet import ENGINES  # noqa: E402  (which template codec each engine restores from)
ENGINE_ORDER = ["none", "cpu", "iaa", "dsa", "dsaiaa"]
LAUNCHES = ["burst", "staggered"]
# 2 GiB guests: 2 GiB of pool plus about 1.1 GiB of VMM, daemon and checkpoints each, so a 125 GiB
# socket holds 36 beside the kernel and a reserve (40 does not fit)
DENSITIES = [1, 2, 4, 8, 16, 24, 32, 36]
SCREEN_ACCEL_DENSITIES = [1, 8, 24, 36]
CELL_OVERHEAD_S = 90          # pool resize, preflight, teardown, settle
# node memory outside the huge page pool that a cell needs, in GiB per sandbox: the VMM's own pages,
# the resident daemon, and its checkpoints in the node-bound store (the raw reference image, the base
# checkpoint and the save: 0.63-0.77 GiB at the peak of the smoke cells), plus a reserve
VMM_GIB = 0.1
DAEMON_GIB = 0.15
STORE_GIB = 0.75
RESERVE_GIB = 4.0


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def read_env(path):
    env = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k] = v.strip("'\"")
    return env


def read_int(path):
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


def node_cpus(node):
    cpus = []
    for part in open(f"/sys/devices/system/node/node{node}/cpulist").read().strip().split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def node_mem_gib(node):
    """MemFree and the reclaimable file cache of a node, in GiB."""
    m = {}
    for line in open(f"/sys/devices/system/node/node{node}/meminfo"):
        f = line.split()
        if len(f) >= 4:
            m[f[2].rstrip(":")] = int(f[3]) / 2**20
    return m.get("MemFree", 0.0), m.get("FilePages", 0.0)


def mem_need_gib(c):
    per = VMM_GIB + (0.0 if c["engine"] == "none" else DAEMON_GIB + STORE_GIB)
    return c["density"] * per + RESERVE_GIB


def other_nodes(node):
    nodes = sorted(int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit())
    return [n for n in nodes if n != node and node_cpus(n)]


class Pool:
    """The measured socket's 2 MiB huge page pool."""

    def __init__(self, node):
        self.node = node
        self.base = f"/sys/devices/system/node/node{node}/hugepages/hugepages-2048kB"
        self.nr0 = self.nr()

    def nr(self):
        return read_int(f"{self.base}/nr_hugepages") or 0

    def free(self):
        return read_int(f"{self.base}/free_hugepages") or 0

    def write(self, pages):
        r = subprocess.run(["sudo", "-n", "tee", f"{self.base}/nr_hugepages"], input=f"{pages}\n".encode(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0:
            log(f"  pool: cannot write nr_hugepages: {r.stderr.decode().strip()}")
        return self.nr()

    def settle(self, timeout=60.0):
        """Wait until every page of the pool is free again (the cell's guests are gone)."""
        t0 = time.time()
        while self.free() < self.nr():
            if time.time() - t0 > timeout:
                return False
            time.sleep(1.0)
        return True

    def size(self, pages):
        """Resize to `pages`, growing through compaction and a cache drop when the first attempt
        falls short. Returns the pages the kernel granted."""
        if not self.settle():
            log(f"  pool: {self.nr() - self.free()} pages still in use after 60 s")
        if self.nr() == pages:
            return pages
        got = self.write(pages)
        if got < pages:
            log(f"  pool: got {got} of {pages} pages, compacting and dropping caches")
            subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 1 > /proc/sys/vm/compact_memory; "
                            "echo 3 > /proc/sys/vm/drop_caches"], stderr=subprocess.DEVNULL)
            got = self.write(pages)
        return got

    def restore(self):
        if self.nr() != self.nr0:
            self.settle()
            log(f"pool: restoring node {self.node} to {self.nr0} pages (got {self.write(self.nr0)})")


def cell_label(density, engine, launch, regime, rep, reps):
    label = f"n{density}_{engine}_{launch}_{regime.replace('.', '')}"
    return f"{label}_r{rep}" if reps > 1 else label


def build_cells(a):
    cells = []
    for rep in range(1, a.reps + 1):
        for engine in ENGINE_ORDER:
            if engine not in a.engines:
                continue
            densities = a.densities
            if a.mode == "screen" and engine in ("dsa", "iaa"):
                densities = [d for d in SCREEN_ACCEL_DENSITIES if d in a.densities]
            for density in densities:
                for launch in LAUNCHES:
                    if launch not in a.launch:
                        continue
                    cells.append(dict(label=cell_label(density, engine, launch, a.regime, rep, a.reps), density=density,
                                      engine=engine, launch=launch, regime=a.regime, rep=rep))
    return cells


def leftovers(label):
    """Pids of processes still holding the cell's socket directory or sandbox disks on their command line."""
    marks = (f"/tmp/vmd-{label[:24]}/".encode(), f"/sandboxes/{label}/".encode())
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read()
        except OSError:
            continue
        if any(m in cmd for m in marks):
            pids.append(int(d))
    return pids


class Driver:
    def __init__(self, a, env):
        self.a = a
        self.env = env
        self.home = env["VMD_HOME"]
        self.node = int(env["VMD_SOCKET"])
        self.guest_pages = int(env.get("GUEST_MEM", "2G").rstrip("G")) * 512
        self.out = a.output
        self.pool = Pool(self.node)
        self.cells = build_cells(a)
        self.records = {}
        self.child = None
        self.stopping = False

    # ---- bookkeeping
    def cell_dir(self, c):
        return f"{self.out}/{c['label']}"

    def status(self, c):
        d = self.cell_dir(c)
        if os.path.exists(f"{d}/result.json"):
            return "done"
        if os.path.exists(f"{d}/refused.json"):
            return "refused"
        return "pending"

    def write_cells(self):
        recs = []
        for c in self.cells:
            r = dict(c)
            r.update(self.records.get(c["label"], {"status": self.status(c)}))
            recs.append(r)
        tmp = f"{self.out}/cells.json.tmp"
        json.dump({"output": self.out, "cells": recs}, open(tmp, "w"), indent=1)
        os.replace(tmp, f"{self.out}/cells.json")

    def write_run(self):
        run = {"args": vars(self.a), "argv": sys.argv, "hostname": socket.gethostname(), "kernel": os.uname().release,
               "t_start": time.time(), "host_env": self.env, "node": self.node, "pool_pages_before": self.pool.nr0,
               "driver_cpus": sorted(os.sched_getaffinity(0))}
        try:
            run["bin_versions"] = open(f"{self.env['VMD_BIN']}/VERSIONS").read().strip().splitlines()
        except OSError:
            pass
        json.dump(run, open(f"{self.out}/run.json", "w"), indent=1)

    def warm_templates(self, c):
        """Read the cell's templates and template disks back into the page cache: growing the
        huge page pool evicts them, and a cold first restore would be charged to the cell."""
        fmt = ENGINES[c["engine"]]["fmt"]
        root = f"{self.home}/templates"
        files = []
        for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            if not d.endswith(f"_{c['regime']}"):
                continue
            try:
                man = json.load(open(f"{root}/{d}/manifest.json"))
            except (OSError, ValueError):
                continue
            if fmt in man.get("formats", {}):
                fd = man["formats"][fmt]["dir"]
                files += [f"{fd}/{f}" for f in sorted(os.listdir(fd))]
            if man.get("disk"):
                files.append(man["disk"])
        t0 = time.time()
        n = 0
        for f in files:
            try:
                with open(f, "rb") as fh:
                    while fh.read(1 << 24):
                        pass
                n += os.path.getsize(f)
            except OSError:
                pass
        return n / 2**30, time.time() - t0

    # ---- one cell
    def run_cell(self, c):
        a = self.a
        d = self.cell_dir(c)
        os.makedirs(d, exist_ok=True)
        rec = {"status": "running", "t_start": time.time()}
        self.records[c["label"]] = rec
        self.write_cells()
        want = self.guest_pages * (c["density"] + 1)
        if not a.no_pool:
            got = self.pool.size(want)
            rec["pool_pages"] = {"wanted": want, "got": got}
            if got < want:
                log(f"  pool holds {got} of {want} pages: vmfleet.py will refuse the cell")
        free, cache = node_mem_gib(self.node)
        need = mem_need_gib(c)
        rec["node_mem_gib"] = {"free": round(free, 1), "file_cache": round(cache, 1), "need": round(need, 1)}
        if free + cache < need and not a.force_memory:
            why = (f"node {self.node} has {free:.0f} GiB free + {cache:.0f} GiB file cache outside the huge page pool, "
                   f"the cell needs about {need:.0f} (--force-memory overrides)")
            log(f"  refused: {why}")
            json.dump({"label": c["label"], "refused": why}, open(f"{d}/refused.json", "w"))
            rec.update(status="refused", refused=why, rc=3, t_end=time.time(), summary=[])
            return rec
        gib, secs = self.warm_templates(c)
        rec["warm"] = {"gib": round(gib, 2), "secs": round(secs, 1)}
        log(f"  pool {self.pool.nr()} pages, node {self.node} {free:.0f} GiB free, templates warmed ({gib:.1f} GiB in {secs:.1f} s)")
        cmd = [sys.executable, VMFLEET, "--density", str(c["density"]), "--engine", c["engine"], "--launch", c["launch"],
               "--regime", c["regime"], "--duration", str(a.duration), "--grace", str(a.grace), "--label", c["label"],
               "--out", d] + shlex.split(a.cell_args)
        rec["cmd"] = cmd
        summary = []
        with open(f"{d}/cell.log", "a") as logf:
            logf.write(f"=== {time.strftime('%F %T')} {' '.join(shlex.quote(x) for x in cmd)}\n")
            logf.flush()
            self.child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ,
                                          VMD_HOME=self.home), start_new_session=True)
            for raw in self.child.stdout:
                line = raw.decode(errors="replace")
                logf.write(line)
                logf.flush()
                s = line.rstrip("\n")
                if s.startswith("== ") or (summary and s.startswith("   ")):
                    summary.append(s)
                    print(s, flush=True)
                elif "CELL_REFUSED" in s or "Traceback" in s or s.startswith("  grace over"):
                    print("  " + s, flush=True)
                elif a.verbose:
                    print("  | " + s, flush=True)
            rc = self.child.wait()
            self.child = None
        rec["rc"] = rc
        rec["t_end"] = time.time()
        rec["summary"] = summary
        if rc == 0 and os.path.exists(f"{d}/result.json"):
            rec["status"] = "done"
        elif rc == 3:
            rec["status"] = "refused"
            try:
                rec["refused"] = json.load(open(f"{d}/refused.json"))["refused"]
            except (OSError, KeyError, ValueError):
                pass
        elif self.stopping:
            rec["status"] = "interrupted"
        else:
            rec["status"] = "failed"
        left = leftovers(c["label"])
        if left:
            log(f"  {len(left)} processes of the cell still alive: killing {left}")
            for pid in left:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            rec["leftovers"] = left
        if not self.pool.settle():
            rec["pool_not_returned"] = self.pool.nr() - self.pool.free()
            log(f"  pool: {rec['pool_not_returned']} pages still in use after the cell")
        return rec

    # ---- the matrix
    def run(self):
        a = self.a
        os.makedirs(self.out, exist_ok=True)
        todo = [c for c in self.cells if a.rerun or self.status(c) != "done"]
        per_cell = a.duration + a.grace + CELL_OVERHEAD_S
        log(f"matrix {a.mode}: {len(self.cells)} cells, {len(todo)} to run, about {len(todo) * per_cell / 3600:.1f} h "
            f"({a.duration} s cells + {a.grace} s grace), output {self.out}")
        log(f"  densities {a.densities}, engines {a.engines}, launch {a.launch}, regime {a.regime}, reps {a.reps}; "
            f"socket {self.node}, driver on cpus {min(os.sched_getaffinity(0))}-{max(os.sched_getaffinity(0))}")
        if a.dry_run:
            for i, c in enumerate(self.cells, 1):
                st = self.status(c)
                mark = "skip" if st == "done" and not a.rerun else "run "
                print(f"  {i:3d} {mark} {c['label']:<32} pool {self.guest_pages * (c['density'] + 1):>6} pages, "
                      f"{mem_need_gib(c):4.0f} GiB beside it  ({st})")
            return 0
        self.write_run()
        self.write_cells()
        t0 = time.time()
        n = 0
        counts = {}
        for i, c in enumerate(self.cells, 1):
            st = self.status(c)
            if st == "done" and not a.rerun:
                log(f"[{i}/{len(self.cells)}] {c['label']}: done already, skipped")
                counts["skipped"] = counts.get("skipped", 0) + 1
                continue
            if self.stopping:
                break
            remaining = (len(todo) - n) * per_cell
            log(f"[{i}/{len(self.cells)}] {c['label']}: N={c['density']} {c['engine']} {c['launch']} "
                f"({remaining / 3600:.1f} h left)")
            rec = self.run_cell(c)
            n += 1
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            log(f"  -> {rec['status']} (rc {rec['rc']}, {rec['t_end'] - rec['t_start']:.0f} s)" +
                (f": {rec['refused']}" if rec.get("refused") else ""))
            self.write_cells()
            if rec["status"] == "failed" and a.stop_on_fail:
                log("stopping on the failed cell (--stop-on-fail)")
                break
            if self.stopping:
                break
            if i < len(self.cells):
                time.sleep(a.settle)
        log(f"matrix {a.mode}: {' '.join(f'{k} {v}' for k, v in sorted(counts.items()))} in "
            f"{(time.time() - t0) / 3600:.2f} h; cells.json in {self.out}")
        return 130 if self.stopping else (1 if counts.get("failed") else 0)

    def stop(self, *_):
        if self.stopping:
            return
        self.stopping = True
        if self.child and self.child.poll() is None:
            log("stopping the running cell")
            self.child.send_signal(signal.SIGTERM)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="smoke: densities 1,4 x both patterns x cpu,dsaiaa, 120 s cells")
    mode.add_argument("--screen", action="store_true",
                      help="phase 1: none,cpu,dsaiaa at every density and pattern, then dsa,iaa at 1,8,24,36")
    mode.add_argument("--confirm", action="store_true",
                      help="phase 2: every engine and pattern at --densities (default 1,36: add the knee), 3 reps")
    mode.add_argument("--full", action="store_true", help="every engine x density x pattern")
    ap.add_argument("--densities", default=None, help="comma-separated slot counts (default 1,2,4,8,16,24,32,36)")
    ap.add_argument("--engines", default=None, help="comma-separated engines (none,cpu,dsa,iaa,dsaiaa)")
    ap.add_argument("--launch", default=None, help="comma-separated launch patterns (burst,staggered)")
    ap.add_argument("--regime", default="x0.1", help="template think regime (x0.1 | real)")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--duration", type=int, default=None, help="seconds of launching sessions per cell (600; real: 900)")
    ap.add_argument("--grace", type=int, default=120, help="seconds for in-flight sessions after the duration")
    ap.add_argument("--settle", type=float, default=10.0, help="seconds between cells")
    ap.add_argument("--cell-args", default="", help="extra vmfleet.py arguments, e.g. '--workers 4 --ckpt-every 2'")
    ap.add_argument("--output", default=None, help="run directory (default $VMD_HOME/results/<mode>)")
    ap.add_argument("--rerun", action="store_true", help="repeat cells that have a result.json")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--no-pool", action="store_true", help="leave the huge page pool as it is")
    ap.add_argument("--force-memory", action="store_true", help="run a cell the node's free memory looks too small for")
    ap.add_argument("--no-pin", action="store_true", help="do not pin the driver and controller to another socket")
    ap.add_argument("--dry-run", action="store_true", help="list the cells and exit")
    ap.add_argument("--verbose", action="store_true", help="echo every controller line")
    a = ap.parse_args()

    home = os.environ.get("VMD_HOME") or f"{os.environ.get('AGENTSYSPERF_HOME', os.path.expanduser('~/.agentsysperf'))}/vm_density"
    envp = f"{home}/host.env"
    if not os.path.exists(envp):
        sys.exit(f"no {envp}: run host_setup.sh first")
    env = read_env(envp)
    env.setdefault("VMD_HOME", home)
    if not os.path.exists(VMFLEET):
        sys.exit(f"{VMFLEET} missing")

    if a.quick:
        a.mode = "quick"
        defaults = dict(densities=[1, 4], engines=["cpu", "dsaiaa"], launch=LAUNCHES, reps=1, duration=120)
    elif a.screen:
        a.mode = "screen"
        defaults = dict(densities=DENSITIES, engines=ENGINE_ORDER, launch=LAUNCHES, reps=1, duration=600)
    elif a.confirm:
        a.mode = "confirm"
        defaults = dict(densities=[1, 36], engines=ENGINE_ORDER, launch=LAUNCHES, reps=3, duration=600)
    elif a.full:
        a.mode = "full"
        defaults = dict(densities=DENSITIES, engines=ENGINE_ORDER, launch=LAUNCHES, reps=1, duration=600)
    else:
        a.mode = "custom"
        defaults = dict(densities=[1, 2, 4, 8], engines=["cpu", "dsaiaa"], launch=LAUNCHES, reps=1, duration=600)
    a.densities = [int(x) for x in a.densities.split(",")] if a.densities else defaults["densities"]
    a.engines = a.engines.split(",") if a.engines else defaults["engines"]
    a.launch = a.launch.split(",") if a.launch else defaults["launch"]
    a.reps = a.reps if a.reps is not None else defaults["reps"]
    if a.duration is None:
        a.duration = 900 if a.regime == "real" and a.mode != "quick" else defaults["duration"]
    for e in a.engines:
        if e not in ENGINE_ORDER:
            sys.exit(f"unknown engine {e}")
    for l in a.launch:
        if l not in LAUNCHES:
            sys.exit(f"unknown launch pattern {l}")
    if any(d < 1 for d in a.densities) or a.reps < 1:
        sys.exit("densities and reps must be positive")
    a.output = a.output or f"{home}/results/{a.mode}"

    node = int(env["VMD_SOCKET"])
    if not a.no_pin:
        others = other_nodes(node)
        if others:
            os.sched_setaffinity(0, node_cpus(others[0]))
    os.makedirs(home, exist_ok=True)
    lockf = open(f"{home}/.matrix.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another matrix run holds the lock on this host")
    drv = Driver(a, env)
    if not a.dry_run and not a.no_pool:
        atexit.register(drv.pool.restore)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(s) != signal.SIG_IGN:      # keep nohup's immunity
            signal.signal(s, drv.stop)
    sys.exit(drv.run())


if __name__ == "__main__":
    main()
