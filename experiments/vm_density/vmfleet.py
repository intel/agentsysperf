#!/usr/bin/env python3
"""One cell of the microVM density experiment.

Restores N agent sandboxes from warm templates on one socket, drives each
through a recorded Terminal-Bench session (a checkpoint after every turn, a
save at the end), recycles the slot with the next session until the cell's
duration is up, and writes result.json.

    vmfleet.py --density N --engine none|cpu|dsa|iaa|dsaiaa --launch burst|staggered
               [--regime x0.1|real] [--duration 600] [--label name] ...

Reads $VMD_HOME/host.env (host_setup.sh) and the templates of images.sh.

Launch patterns: `burst` starts every slot at once with no bound on the
restores in flight, so the first wave of restores, turns and checkpoints lands
on the accelerators and the store together; `staggered` spaces the starts by
--stagger seconds (default: the mean session length divided by N, at least
0.5 s) and bounds the restores in flight (--max-restores). A slot that
finishes its session starts the next one right away in both patterns.

Engines: `none` restores from the lz4 template and runs the session with no
daemon; `cpu`, `dsa`, `iaa` and `dsaiaa` attach a resident offload daemon per
sandbox (classification and diff compare on the CPU or on DSA, lz4 or IAA
compression), take a full checkpoint after the restore, a diff after every
turn and a full save at the end. The dsa engines also restore with UFFDIO_COPY
and switch the kernel providers' offloading on for the cell.
"""
import argparse
import atexit
import fcntl
import http.client
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

CLK = os.sysconf("SC_CLK_TCK")
DRAIN_SECS = 0.05
API_BUSY_S = 10.0         # how long pause/resume retry while the VMM answers 409 (migration worker still owns the VM)
STALL_S = 90.0            # a session with no step for this long is nudged with a resume, then failed
STEP_RE = re.compile(rb"AGENT_STEP_DONE (\d+) (\S+) seed=(\d+) secs=([0-9.]+)")
WL2_RE = re.compile(rb"AGENT_WL2 (\S+) seed=(\d+) secs=[0-9.]+ rc=(\d+) out=(\d*) err=(\d*)")
KV_RE = re.compile(r"\b([a-z_]+)=([0-9][0-9.]*)\b")
CKPT_RE = re.compile(r"Checkpoint (\d+) (.*)")
TIME = ["/usr/bin/time", "-f", "%e %U %S", "-o"] if os.path.exists("/usr/bin/time") else None
UFFD_P = "/sys/kernel/uffd_dma"
DCBM_P = "/sys/module/dcbm/parameters"

ENGINES = {
    # daemon: resident offload daemon per sandbox; codec/classify: its settings;
    # fmt: the template codec restored from; uffd_copy: restore installs pages with
    # UFFDIO_COPY (DSA in the kernel when the providers offload); offload: the
    # providers' offloading knob for the cell.
    "none": dict(daemon=False, codec=None, classify=None, fmt="lz4", uffd_copy=False, offload=0),
    "cpu": dict(daemon=True, codec="lz4", classify="cpu", fmt="lz4", uffd_copy=False, offload=0),
    "dsa": dict(daemon=True, codec="lz4", classify="dsa", fmt="lz4", uffd_copy=True, offload=1),
    "iaa": dict(daemon=True, codec="qpl-hardware-static-async", classify="cpu", fmt="qpl",
                uffd_copy=False, offload=0),
    "dsaiaa": dict(daemon=True, codec="qpl-hardware-static-async", classify="dsa", fmt="qpl",
                   uffd_copy=True, offload=1),
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_env(path):
    env = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k] = v.strip("'\"")
    return env


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def fmt_ms(x):
    return "-" if x is None else f"{x * 1000:.0f}"


# ---------------------------------------------------------------- host helpers
class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.unix_path)
        self.sock = s


def ch_api(sock, path, body=None, timeout=30):
    """PUT /api/v1/<path> on the VMM's API socket; returns the HTTP status (int) or an error string."""
    try:
        c = UnixHTTP(sock, timeout)
        c.request("PUT", f"/api/v1/{path}", body=body,
                  headers={"Content-Type": "application/json"} if body else {})
        r = c.getresponse()
        r.read()
        c.close()
        return r.status
    except (OSError, http.client.HTTPException) as e:
        return f"error {e.__class__.__name__}"


def ch_api_busy(sock, path, body=None, budget=API_BUSY_S):
    """ch_api, retried while the VMM answers 409: the migration worker of the
    previous send-migration still owns the VM for a moment after the daemon
    has written the checkpoint.  Returns (status, retries, seconds waited)."""
    t0 = time.time()
    n = 0
    while True:
        st = ch_api(sock, path, body)
        if st != 409 or time.time() - t0 > budget:
            return st, n, time.time() - t0
        n += 1
        time.sleep(0.0005)


def proc_ticks(path):
    f = open(path).read()
    fl = f[f.rindex(")") + 2:].split()
    return (int(fl[11]) + int(fl[12])) / CLK


def proc_cpu(pid):
    """(total, vcpu) CPU seconds of a process: the process total (exited threads
    included) and the part spent by threads named vcpu*."""
    try:
        tot = proc_ticks(f"/proc/{pid}/stat")
        tasks = os.listdir(f"/proc/{pid}/task")
    except (OSError, ValueError, IndexError):
        return None, None
    vcpu = 0.0
    for t in tasks:
        try:
            if open(f"/proc/{pid}/task/{t}/comm").read().startswith("vcpu"):
                vcpu += proc_ticks(f"/proc/{pid}/task/{t}/stat")
        except (OSError, ValueError, IndexError):
            pass
    return tot, vcpu


def read_int(path):
    try:
        return int(open(path).read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def provider_counters():
    out = {}
    for k in ("copies_done", "copies_failed", "copies_refused", "copies_efault", "bytes_dma", "bytes_cpu",
              "admit_failures", "refused_saturated"):
        v = read_int(f"{UFFD_P}/{k}")
        if v is not None:
            out[f"uffd_{k}"] = v
    for k in ("folios_cleared", "clear_failures", "admit_failures", "batches_refused", "clears_batched"):
        v = read_int(f"{DCBM_P}/{k}")
        if v is not None:
            out[f"dcbm_{k}"] = v
    return out


def set_offloading(want):
    """The providers' offloading knobs (sudo); returns what they read back."""
    got = {}
    for p in (f"{UFFD_P}/offloading", f"{DCBM_P}/offloading"):
        if not os.path.exists(p):
            continue
        subprocess.run(["sudo", "-n", "tee", p], input=f"{want}\n".encode(), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
        try:
            got[p] = open(p).read().strip()
        except OSError:
            got[p] = None
    return got


def node_cpus(node):
    spec = open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()
    cpus = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def cpu_busy(cpus):
    """(busy, total) jiffies summed over the given CPUs from /proc/stat."""
    busy = total = 0
    want = {f"cpu{c}" for c in cpus}
    for line in open("/proc/stat"):
        f = line.split()
        if f[0] in want:
            vals = [int(x) for x in f[1:]]
            idle = vals[3] + vals[4]
            total += sum(vals)
            busy += sum(vals) - idle
    return busy, total


def node_free_mib(node):
    for line in open(f"/sys/devices/system/node/node{node}/meminfo"):
        if "MemFree" in line:
            return int(line.split()[3]) // 1024
    return None


def hp_free(node):
    return read_int(f"/sys/devices/system/node/node{node}/hugepages/hugepages-2048kB/free_hugepages") or 0


def kill_pid(pid, sig=signal.SIGKILL, wait=20.0):
    try:
        os.kill(pid, sig)
    except OSError:
        return
    end = time.time() + wait
    while time.time() < end and os.path.exists(f"/proc/{pid}"):
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        time.sleep(0.02)


# ---------------------------------------------------------------- the cell
class Cell:
    def __init__(self, a, env):
        self.a = a
        self.env = env
        self.eng = ENGINES[a.engine]
        self.node = int(env["VMD_SOCKET"])
        self.bin = env["VMD_BIN"]
        self.ch = f"{self.bin}/cloud-hypervisor"
        self.remote = f"{self.bin}/ch-remote"
        self.od = f"{self.bin}/offload_daemon"
        self.disks = env["VMD_DISKS"]
        self.store = env["VMD_STORE"]
        self.tmpl_root = f"{env['VMD_HOME']}/templates"
        self.kernel_offload = env.get("KERNEL_OFFLOAD") == "1"
        self.uffd_copy = self.eng["uffd_copy"] and self.kernel_offload
        self.out = a.out or f"{env['VMD_HOME']}/results/{a.label}"
        self.sockdir = f"/tmp/vmd-{a.label[:24]}"
        self.sbdisks = f"{self.disks}/sandboxes/{a.label}"
        self.sbstore = f"{self.store}/{a.label}"
        self.penv = dict(os.environ, DTO_WQ_LIST=env.get("DTO_WQ_LIST", ""), DTO_IS_NUMA_AWARE="1",
                         DTO_LOG_LEVEL="1", LD_LIBRARY_PATH=self.bin, RUST_LOG="info")
        self.numa = ["numactl", f"-N{self.node}", f"-m{self.node}"]
        self.sessions = a.sessions.split(",") if a.sessions else sorted(
            d[: -len(f"_{a.regime}")] for d in os.listdir(self.tmpl_root) if d.endswith(f"_{a.regime}"))
        self.templates = {}
        self.think = {}            # session -> {turn: think seconds after scaling and cap}
        self.lock = threading.Lock()
        self.sandboxes = []        # finished incarnations (records)
        self.steps = []            # [slot, gen, session, k, turn, t_host, secs, rc, out, err]
        self.ckpts = []
        self.samples = []
        self.live = {}             # slot -> Sandbox in flight
        self.pids = set()          # every child pid we started (cleanup)
        self.next_session = 0
        self.t_start = self.t_end = None
        self.stop = threading.Event()
        self.restore_sem = None
        self.refused = None

    # ---- preflight
    def load_templates(self):
        need = f"{self.eng['fmt']}"
        for s in self.sessions:
            d = f"{self.tmpl_root}/{s}_{self.a.regime}"
            m = f"{d}/manifest.json"
            if not os.path.exists(m):
                return f"template {d} missing (images.sh templates)"
            man = json.load(open(m))
            if need not in man["formats"]:
                return f"template {d} has no {need} format"
            man["restore_dir"] = man["formats"][need]["dir"]
            self.templates[s] = man
            traj = json.load(open(f"{self.env['CH_SRC']}/benchmark/agent/traj/{s}.traj.json"))
            scale = float(self.a.regime[1:]) if self.a.regime.startswith("x") else 1.0
            cap = float(self.a.think_cap)
            th = {}
            for x in traj["turns"]:
                ms = x.get("latency_ms")
                if ms is not None:
                    t = ms / 1000.0 * scale
                    th[x["turn"]] = min(t, cap) if cap > 0 else t
            self.think[s] = th
            man["session_secs"] = sum(th.values())
        return None

    def preflight(self):
        a = self.a
        why = self.load_templates()
        if why:
            return why
        pages = int(self.env.get("GUEST_MEM", "2G").rstrip("G")) * 512 * (a.density + 1)
        if hp_free(self.node) < pages:
            return f"node {self.node} has {hp_free(self.node)} free 2 MiB pages, {a.density} guests need {pages}"
        if self.eng["classify"] == "dsa" or self.eng["fmt"] == "qpl":
            for w in self.env.get("DTO_WQ_LIST", "").split(";"):
                if w and not os.path.exists(f"/dev/dsa/{w}"):
                    return f"DSA user WQ /dev/dsa/{w} missing"
            if self.eng["classify"] == "dsa" and not self.env.get("DTO_WQ_LIST"):
                return "DTO_WQ_LIST is empty in host.env"
        if self.eng["fmt"] == "qpl" and not os.path.isdir("/dev/iax"):
            return "no /dev/iax user WQs for the IAA codec"
        if self.eng["uffd_copy"] and not self.kernel_offload:
            log("  note: no uffd_dma/dcbm providers on this kernel: the dsa engine restores on the CPU (kernel_offload=0)")
        if not os.path.isdir(self.store):
            return f"store {self.store} missing (host_setup.sh store)"
        free_gib = shutil.disk_usage(self.store).free / 2**30
        if self.eng["daemon"] and free_gib < a.density * 0.75:
            return f"store has {free_gib:.0f} GiB free, {a.density} daemons want about {a.density * 0.75:.0f}"
        for f in (self.ch, self.remote, self.od, self.env["GUEST_KERNEL"]):
            if not os.path.exists(f):
                return f"{f} missing"
        return None

    # ---- lifecycle pieces
    def pick_session(self):
        with self.lock:
            s = self.sessions[self.next_session % len(self.sessions)]
            self.next_session += 1
        return s

    def launch_vmm(self, sb):
        """The snapshot carries the template's disk and console paths: the VMM
        runs in its own user+mount namespace with this sandbox's copies bound
        over them (no privileges needed)."""
        man = sb["man"]
        subprocess.run(["cp", "--reflink=always", man["disk"], sb["disk"]], check=True)
        open(sb["console"], "wb").close()
        cmd = (f"mount --bind '{sb['disk']}' '{man['disk']}' && mount --bind '{sb['console']}' '{man['console']}' && "
               f"exec {' '.join(self.numa)} '{self.ch}' --api-socket '{sb['sock']}'")
        p = subprocess.Popen(["unshare", "-Urm", "--propagation", "private", "sh", "-c", cmd],
                             stdout=open(sb["chlog"], "wb"), stderr=subprocess.STDOUT)
        with self.lock:
            self.pids.add(p.pid)
        return p

    def restore(self, sb):
        rec = sb["rec"]
        for _ in range(400):
            if os.path.exists(sb["sock"]):
                break
            time.sleep(0.025)
        else:
            rec["failed"] = "VMM api socket never appeared"
            return False
        recv = subprocess.Popen([self.remote, "--api-socket", sb["sock"], "receive-migration",
                                 f"receiver_url=unix:{sb['rdata']}"], stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        for _ in range(400):
            if os.path.exists(sb["rdata"]):
                break
            time.sleep(0.025)
        else:
            rec["failed"] = "receive-migration socket never appeared"
            recv.kill()
            return False
        args = ["--resume", "--hugetlb", "--workers", str(self.a.workers), "--dsa-depth", "32"]
        if self.uffd_copy:
            args.append("--uffd-copy")
        tf = f"{self.out}/rt{sb['slot']}.{sb['gen']}.txt"
        t0 = time.time()
        r = subprocess.run([*self.numa, *(TIME + [tf] if TIME else []), self.od, "restore",
                            "--socket", sb["rdata"], "--input-dir", sb["man"]["restore_dir"], *args],
                           env=self.penv, capture_output=True, text=True)
        t1 = time.time()
        try:
            recv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            recv.kill()
        rs = {"wall": t1 - t0, "rc": r.returncode, "uffd_copy": self.uffd_copy}
        try:
            e, u, sy = open(tf).read().split()[-3:]
            rs["cpu"] = float(u) + float(sy)
        except (OSError, ValueError):
            rs["cpu"] = None
        logtxt = r.stdout + r.stderr
        open(f"{self.out}/rd{sb['slot']}.{sb['gen']}.log", "w").write(logtxt)
        for k, v in KV_RE.findall(logtxt):
            if k in ("copies_on_dsa", "copies_refused", "copies_cpu", "decompress_ms", "total_ms", "map_us_total",
                     "populate_ms", "chunks", "compressed"):
                rs[k] = float(v)
        rec["restore"] = rs
        if r.returncode != 0:
            rec["failed"] = "restore rc %d: %s" % (r.returncode, (logtxt.strip().splitlines() or ["?"])[-1][:160])
            return False
        rec["t_restored"] = t1
        return True

    def start_daemon(self, sb):
        root = f"{self.sbstore}/s{sb['slot']}.g{sb['gen']}"
        os.makedirs(root, exist_ok=True)
        dsock = f"{self.sockdir}/s{sb['slot']}.g{sb['gen']}.od"
        if os.path.exists(dsock):
            os.unlink(dsock)
        dlog = f"{self.out}/od{sb['slot']}.{sb['gen']}.log"
        p = subprocess.Popen([*self.numa, self.od, "serve", "--socket", dsock, "--output-root", root,
                              "--compression", self.eng["codec"], "--chunk-size", "2097152",
                              "--workers", str(self.a.workers), "--submitters", str(self.a.submitters),
                              "--classify", self.eng["classify"], "--reference-dir", f"{root}/ref",
                              "--diff-compare", self.eng["classify"]],
                             env=self.penv, stdout=open(dlog, "wb"), stderr=subprocess.STDOUT)
        with self.lock:
            self.pids.add(p.pid)
        for _ in range(400):
            if os.path.exists(dsock):
                break
            time.sleep(0.05)
        else:
            sb["rec"]["failed"] = "offload daemon did not open its socket"
            return False
        sb["daemon"] = {"proc": p, "sock": dsock, "log": dlog, "root": root}
        return True

    def checkpoint(self, sb, kind, k, t_ev, turn=None):
        """Pause the guest, send one checkpoint to its resident daemon, resume.
        The daemon's "Checkpoint N" line is the completion signal: ch-remote
        returns when the migration is dispatched, not when it is written."""
        d = sb["daemon"]
        rec = {"slot": sb["slot"], "gen": sb["gen"], "session": sb["session"], "kind": kind, "k": k, "t_event": t_ev}
        if kind == "step":
            rec["turn"] = turn
            rec["think"] = self.think[sb["session"]].get(turn) if turn is not None else None
        t0 = time.time()
        st, n, w = ch_api_busy(sb["sock"], "vm.pause")
        if n:
            rec["pause_busy"] = [n, round(w, 4)]
        if st != 204:
            rec["failed"] = f"pause {st}"
            return rec
        try:
            off = os.path.getsize(d["log"])
        except OSError:
            off = 0
        url = f"destination_url=unix:{d['sock']},memory_mode=memfds,preserve_source=on,dirty_log=consume"
        t1 = time.time()
        deadline = t1 + 60
        while True:
            r = subprocess.run([self.remote, "--api-socket", sb["sock"], "send-migration", url],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            if "already in progress" in r.stderr and time.time() < deadline:
                time.sleep(0.002)
                continue
            rec["failed"] = "send-migration: " + (r.stderr.strip().splitlines() or ["?"])[-1][:120]
            break
        line = None
        if "failed" not in rec:
            with open(d["log"], "rb") as f:
                f.seek(off)
                buf = b""
                while time.time() < deadline + 120:
                    chunk = f.read()
                    if chunk:
                        buf += chunk
                        for l in buf.decode(errors="replace").splitlines():
                            m = CKPT_RE.search(l)
                            if m:
                                line = m.group(0)
                                break
                        if line:
                            break
                    time.sleep(0.0005)
            if line is None:
                rec["failed"] = "timeout waiting for the daemon"
        t2 = time.time()
        rec["resume_http"], n, w = ch_api_busy(sb["sock"], "vm.resume")
        if n:
            rec["resume_busy"] = [n, round(w, 4)]
        if rec["resume_http"] != 204:
            rec["failed"] = f"resume {rec['resume_http']}"
        t3 = time.time()
        rec["wall"] = t2 - t1
        rec["pause_wall"] = t3 - t0
        rec["after_step"] = t0 - t_ev
        if line:
            rec["line"] = line[:300]
            for key, v in KV_RE.findall(line):
                rec[key] = float(v)
            m = re.search(r"\bkind=(\w+)", line)
            if m:
                rec["type"] = m.group(1)
            if "failed" in line:
                rec["failed"] = "daemon: " + line[:160]
        if rec.get("think") is not None:
            rec["overlap"] = rec["after_step"] + rec["pause_wall"] > rec["think"]
        return rec

    def save(self, sb):
        """One full checkpoint of the sandbox at teardown (the session's save), by a one-shot daemon."""
        out = f"{self.sbstore}/s{sb['slot']}.g{sb['gen']}.save"
        osock = f"{self.sockdir}/s{sb['slot']}.g{sb['gen']}.save"
        shutil.rmtree(out, ignore_errors=True)
        if os.path.exists(osock):
            os.unlink(osock)
        tf = f"{self.out}/st{sb['slot']}.{sb['gen']}.txt"
        rec = {}
        st = ch_api(sb["sock"], "vm.pause")
        rec["pause_http"] = st
        d = subprocess.Popen([*self.numa, *(TIME + [tf] if TIME else []), self.od, "snapshot",
                              "--socket", osock, "--output-dir", out, "--compression", self.eng["codec"],
                              "--chunk-size", "2097152", "--workers", str(self.a.workers),
                              "--classify", self.eng["classify"], "--dsa-depth", "32"],
                             env=self.penv, stdout=open(f"{self.out}/sd{sb['slot']}.{sb['gen']}.log", "wb"),
                             stderr=subprocess.STDOUT)
        with self.lock:
            self.pids.add(d.pid)
        for _ in range(400):
            if os.path.exists(osock):
                break
            time.sleep(0.025)
        t0 = time.time()
        r = subprocess.run([self.remote, "--api-socket", sb["sock"], "send-migration",
                            f"destination_url=unix:{osock},memory_mode=memfds,preserve_source=on"],
                           capture_output=True, text=True)
        try:
            d.wait(timeout=300)
        except subprocess.TimeoutExpired:
            d.kill()
            d.wait()
        with self.lock:
            self.pids.discard(d.pid)
        rec["wall"] = time.time() - t0
        rec["rc"] = d.returncode
        try:
            e, u, sy = open(tf).read().split()[-3:]
            rec["cpu"] = float(u) + float(sy)
        except (OSError, ValueError):
            rec["cpu"] = None
        size = 0
        for dp, _, fs in os.walk(out):
            for f in fs:
                try:
                    size += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        rec["bytes"] = size
        if r.returncode != 0 or d.returncode != 0:
            rec["failed"] = (r.stderr.strip().splitlines() or ["daemon rc %s" % d.returncode])[-1][:160]
        shutil.rmtree(out, ignore_errors=True)
        return rec

    def teardown(self, sb):
        rec = sb["rec"]
        pid = sb["vmm"].pid
        tot, vcpu = proc_cpu(pid)
        rec["vmm_cpu"] = tot
        rec["vmm_vcpu_cpu"] = vcpu
        if sb.get("daemon"):
            dp = sb["daemon"]["proc"]
            rec["daemon_cpu"] = proc_cpu(dp.pid)[0]
            kill_pid(dp.pid, signal.SIGTERM, wait=5.0)
            kill_pid(dp.pid)
            shutil.rmtree(sb["daemon"]["root"], ignore_errors=True)
        kill_pid(pid)
        with self.lock:
            self.pids.discard(pid)
            if sb.get("daemon"):
                self.pids.discard(sb["daemon"]["proc"].pid)
        for f in (sb["disk"], sb["sock"], sb["rdata"]):
            try:
                os.unlink(f)
            except OSError:
                pass
        rec["t_torn_down"] = time.time()

    # ---- console
    def drain(self, sb, now):
        try:
            with open(sb["console"], "rb") as f:
                f.seek(sb["off"])
                data = f.read()
        except OSError:
            return
        cut = data.rfind(b"\n") + 1
        if not cut:
            return
        sb["off"] += cut
        data = data[:cut]
        for m in WL2_RE.finditer(data):
            sb["wl2"][int(m.group(2))] = (int(m.group(3)), int(m.group(4) or 0), int(m.group(5) or 0))
        for m in STEP_RE.finditer(data):
            k, wl, seed, secs = int(m.group(1)), m.group(2).decode(), int(m.group(3)), float(m.group(4))
            rc, ob, eb = sb["wl2"].get(seed, (None, None, None))
            if sb["t_first_turn"] is None:
                sb["t_first_turn"] = now
            sb["last_step"] = (k, seed, now)
            with self.lock:
                self.steps.append([sb["slot"], sb["gen"], sb["session"], k, seed, now, secs, rc, ob, eb])
            sb["turns"] += 1
            sb["events"].put(("step", k, now, seed))
        if sb["t_done"] is None and b"AGENT_DIFF_POINT" in data:
            sb["t_done"] = now
            sb["events"].put(("done", 0, now, None))

    def drainer(self):
        while not self.stop.is_set():
            with self.lock:
                cur = list(self.live.values())
            for sb in cur:
                if sb.get("vmm") is not None and sb["rec"].get("t_restored"):
                    self.drain(sb, time.time())
            self.stop.wait(DRAIN_SECS)

    def sampler(self):
        cpus = node_cpus(self.node)
        b0, t0 = cpu_busy(cpus)
        while not self.stop.is_set():
            self.stop.wait(1.0)
            b1, t1 = cpu_busy(cpus)
            busy = (b1 - b0) / max(1, t1 - t0)
            b0, t0 = b1, t1
            try:
                used = shutil.disk_usage(self.store)
                store_mib = (used.total - used.free) // 2**20
            except OSError:
                store_mib = None
            with self.lock:
                live = len(self.live)
                self.samples.append([round(time.time() - self.t_start, 1), round(busy, 4), node_free_mib(self.node),
                                     hp_free(self.node), store_mib, live])

    # ---- one slot's thread: sessions back to back until the cell ends
    def slot_thread(self, slot, t_first):
        a = self.a
        while time.time() < t_first and not self.stop.is_set():
            time.sleep(max(0.0, min(0.05, t_first - time.time())))
        gen = 0
        fails = 0
        while not self.stop.is_set() and time.time() < self.t_end:
            gen += 1
            session = self.pick_session()
            sb = {"slot": slot, "gen": gen, "session": session, "man": self.templates[session],
                  "disk": f"{self.sbdisks}/s{slot}.g{gen}.ext4", "console": f"{self.out}/s{slot}.{gen}.console",
                  "chlog": f"{self.out}/ch{slot}.{gen}.log", "sock": f"{self.sockdir}/s{slot}.g{gen}.api",
                  "rdata": f"{self.sockdir}/s{slot}.g{gen}.rdata", "off": 0, "wl2": {}, "turns": 0,
                  "last_step": None, "t_first_turn": None, "t_done": None, "events": queue.Queue(),
                  "rec": {"slot": slot, "gen": gen, "session": session, "engine": a.engine}}
            rec = sb["rec"]
            with self.lock:
                self.live[slot] = sb
            try:
                self.run_session(sb)
            except Exception as e:  # one sandbox must not take the cell down
                rec["failed"] = f"exception {e!r}"
            finally:
                try:
                    if sb.get("vmm") is not None:
                        self.teardown(sb)
                except Exception as e:
                    rec.setdefault("failed", f"teardown {e!r}")
                rec["turns"] = sb["turns"]
                rec["t_first_turn"] = sb["t_first_turn"]
                rec["t_done"] = sb["t_done"]
                with self.lock:
                    if not rec.get("skipped"):
                        self.sandboxes.append(rec)
                    self.live.pop(slot, None)
            if rec.get("skipped"):
                break
            if "failed" in rec:
                fails += 1
                log(f"  slot {slot} gen {gen} ({session}) failed: {rec['failed']}")
                if fails > a.max_fails:
                    log(f"  slot {slot} retired after {fails} failures")
                    break
            else:
                fails = 0
            if a.no_recycle:
                break

    def run_session(self, sb):
        a, rec = self.a, sb["rec"]
        rec["t_launch"] = time.time()
        self.restore_sem.acquire()
        try:
            if time.time() >= self.t_end or self.stop.is_set():
                rec["skipped"] = True
                return
            sb["vmm"] = self.launch_vmm(sb)
            if not self.restore(sb):
                return
        finally:
            self.restore_sem.release()
        if self.eng["daemon"]:
            if not self.start_daemon(sb):
                return
            if a.ckpt_every >= 0:
                c = self.checkpoint(sb, "base", 0, rec["t_restored"])
                with self.lock:
                    self.ckpts.append(c)
                if "failed" in c:
                    rec["failed"] = "base checkpoint: " + c["failed"]
                    return
        t_last = time.time()
        nudged = 0
        while True:
            try:
                kind, k, t, turn = sb["events"].get(timeout=0.5)
            except queue.Empty:
                if sb["vmm"].poll() is not None:
                    rec["failed"] = "VMM died"
                    return
                if self.stop.is_set() or time.time() >= self.t_end + a.grace:
                    rec["cut"] = True
                    return
                if time.time() - t_last > STALL_S:
                    # No step for STALL_S: most likely the guest was left paused.  One resume
                    # nudge (harmless when it is running: 405/409/500), then fail the session.
                    if nudged:
                        rec["failed"] = f"stalled: no step for {STALL_S:.0f} s after a resume nudge"
                        return
                    st = ch_api(sb["sock"], "vm.resume")
                    rec["stall_nudge"] = st
                    log(f"  slot {sb['slot']} gen {sb['gen']} ({sb['session']}): no step for {STALL_S:.0f} s, "
                        f"resume nudge -> {st}")
                    nudged += 1
                    t_last = time.time()
                continue
            t_last = time.time()
            if kind == "step":
                if self.eng["daemon"] and a.ckpt_every > 0 and k % a.ckpt_every == 0:
                    c = self.checkpoint(sb, "step", k, t, turn)
                    with self.lock:
                        self.ckpts.append(c)
            elif kind == "done":
                if self.eng["daemon"] and not a.no_save:
                    rec["save"] = self.save(sb)
                return

    # ---- the cell
    def run(self):
        a = self.a
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.sockdir, exist_ok=True)
        os.makedirs(self.sbdisks, exist_ok=True)
        os.makedirs(self.sbstore, exist_ok=True)
        why = self.preflight()
        if why:
            self.refused = why
            print(f"CELL_REFUSED {a.label}: {why}", flush=True)
            json.dump({"label": a.label, "refused": why}, open(f"{self.out}/refused.json", "w"))
            return 3
        offloading = set_offloading(self.eng["offload"]) if self.kernel_offload else {}
        counters0 = provider_counters()
        mean_secs = sum(self.templates[s]["session_secs"] for s in self.sessions) / len(self.sessions)
        if a.launch == "burst":
            stagger = 0.0
            max_restores = a.max_restores or a.density
        else:
            stagger = a.stagger if a.stagger is not None else max(0.5, mean_secs / a.density)
            max_restores = a.max_restores or 8
        self.restore_sem = threading.Semaphore(max_restores)
        self.t_start = time.time()
        self.t_end = self.t_start + a.duration
        cfg = {"label": a.label, "density": a.density, "engine": a.engine, "launch": a.launch, "stagger_s": stagger,
               "max_restores": max_restores, "regime": a.regime, "duration_s": a.duration, "ckpt_every": a.ckpt_every,
               "save": self.eng["daemon"] and not a.no_save, "recycle": not a.no_recycle, "sessions": self.sessions,
               "mean_session_secs": mean_secs, "socket": self.node, "cpus": self.env.get("VMD_CPUS"),
               "guest_mem": self.env.get("GUEST_MEM"), "guest_vcpus": self.env.get("GUEST_VCPUS"),
               "workers": a.workers, "submitters": a.submitters, "kernel_offload": self.kernel_offload,
               "uffd_copy": self.uffd_copy, "template_fmt": self.eng["fmt"], "offloading": offloading,
               "think_cap": a.think_cap}
        try:
            cfg["bin_versions"] = open(f"{self.bin}/VERSIONS").read().strip().splitlines()
        except OSError:
            pass
        log(f"cell {a.label}: N={a.density} engine={a.engine} launch={a.launch} stagger={stagger:.2f}s "
            f"restores<={max_restores} regime={a.regime} duration={a.duration}s sessions={len(self.sessions)} "
            f"(mean {mean_secs:.0f} s) uffd_copy={self.uffd_copy}")
        threads = [threading.Thread(target=self.drainer, daemon=True),
                   threading.Thread(target=self.sampler, daemon=True)]
        for t in threads:
            t.start()
        slots = []
        for i in range(a.density):
            t = threading.Thread(target=self.slot_thread, args=(i, self.t_start + i * stagger), daemon=True)
            t.start()
            slots.append(t)
        last = 0
        while any(t.is_alive() for t in slots):
            time.sleep(1.0)
            now = time.time()
            if now - last >= 30:
                last = now
                with self.lock:
                    done = len([s for s in self.sandboxes if "failed" not in s])
                    failed = len([s for s in self.sandboxes if "failed" in s])
                    nsteps = len(self.steps)
                    live = len(self.live)
                log(f"  t+{now - self.t_start:.0f}s: live {live}, sessions done {done}, failed {failed}, turns {nsteps}")
            if now > self.t_end + a.grace + 60:
                log("  grace over: stopping the remaining sandboxes")
                self.stop.set()
                break
        self.stop.set()
        for t in slots:
            t.join(timeout=60)
        t_stop = time.time()
        counters1 = provider_counters()
        if self.kernel_offload:
            set_offloading(0)
        res = {"config": cfg,
               "host": {"hostname": socket.gethostname(), "kernel": os.uname().release, "t_start": self.t_start,
                        "t_end": self.t_end, "t_stop": t_stop, "hp_free_start": None,
                        "providers_delta": {k: counters1.get(k, 0) - counters0.get(k, 0) for k in counters1},
                        "samples": self.samples},
               "sandboxes": self.sandboxes, "steps": self.steps, "ckpts": self.ckpts}
        json.dump(res, open(f"{self.out}/result.json", "w"))
        self.report(res)
        return 0

    def report(self, res):
        a = self.a
        sbs = res["sandboxes"]
        ok = [s for s in sbs if "failed" not in s]
        done = [s for s in ok if s.get("t_done")]
        failed = [s for s in sbs if "failed" in s]
        secs = [st[6] for st in res["steps"]]
        rest = [s["restore"]["wall"] for s in sbs if s.get("restore") and s["restore"].get("rc") == 0]
        rcpu = [s["restore"]["cpu"] for s in sbs if s.get("restore") and s["restore"].get("cpu") is not None]
        ttft = [s["t_first_turn"] - s["t_launch"] for s in ok if s.get("t_first_turn") and s.get("t_launch")]
        diffs = [c for c in res["ckpts"] if c["kind"] == "step" and "failed" not in c]
        bases = [c for c in res["ckpts"] if c["kind"] == "base" and "failed" not in c]
        cfail = [c for c in res["ckpts"] if "failed" in c]
        saves = [s["save"] for s in sbs if s.get("save") and "failed" not in s["save"]]
        hours = (res["host"]["t_stop"] - res["host"]["t_start"]) / 3600
        print(f"== {a.label}: N={a.density} {a.engine} {a.launch}: sessions done {len(done)} "
              f"({len(done) / hours:.0f}/h), failed {len(failed)}, turns {len(secs)}, cut {len([s for s in sbs if s.get('cut')])}")
        print(f"   turn secs p50/p95/p99/max {fmt_ms(pct(secs, 50))}/{fmt_ms(pct(secs, 95))}/{fmt_ms(pct(secs, 99))}/"
              f"{fmt_ms(max(secs) if secs else None)} ms; time to first turn p50/p99 "
              f"{fmt_ms(pct(ttft, 50))}/{fmt_ms(pct(ttft, 99))} ms")
        print(f"   restore wall p50/p99 {fmt_ms(pct(rest, 50))}/{fmt_ms(pct(rest, 99))} ms, cpu p50 "
              f"{round(pct(rcpu, 50), 3) if rcpu else '-'} core-s; base ckpt p50 {fmt_ms(pct([c['wall'] for c in bases], 50))} ms")
        if diffs:
            ov = len([c for c in diffs if c.get("overlap")])
            print(f"   diff ckpt pause p50/p99 {fmt_ms(pct([c['pause_wall'] for c in diffs], 50))}/"
                  f"{fmt_ms(pct([c['pause_wall'] for c in diffs], 99))} ms, cpu_ms p50 "
                  f"{pct([c.get('cpu_ms') for c in diffs], 50)}, overlapped the think {ov}/{len(diffs)}, "
                  f"failed {len(cfail)}")
        if saves:
            sc = pct([s['cpu'] for s in saves], 50)
            print(f"   save wall p50 {fmt_ms(pct([s['wall'] for s in saves], 50))} ms, cpu p50 "
                  f"{round(sc, 3) if sc is not None else '-'} core-s, {pct([s['bytes'] for s in saves], 50) / 2**20:.0f} MiB")
        vm = [s["vmm_cpu"] for s in done if s.get("vmm_cpu") is not None]
        dm = [s["daemon_cpu"] for s in done if s.get("daemon_cpu") is not None]
        if done:
            vc = pct([s['vmm_vcpu_cpu'] for s in done if s.get('vmm_vcpu_cpu') is not None], 50)
            print(f"   per session: vmm cpu p50 {round(pct(vm, 50), 2) if vm else '-'} core-s (vcpu "
                  f"{round(vc, 2) if vc is not None else '-'}), daemon cpu p50 {round(pct(dm, 50), 2) if dm else '-'} core-s")
        pd = res["host"]["providers_delta"]
        if pd:
            keys = [k for k in ("uffd_copies_done", "uffd_copies_failed", "uffd_copies_refused", "dcbm_folios_cleared",
                                "dcbm_admit_failures") if k in pd]
            print("   providers: " + ", ".join(f"{k}={pd[k]}" for k in keys))
        if failed:
            for s in failed[:5]:
                print(f"   failed: slot {s['slot']} gen {s['gen']} {s['session']}: {s['failed']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", "-n", type=int, required=True, help="number of sandbox slots")
    ap.add_argument("--engine", choices=sorted(ENGINES), required=True)
    ap.add_argument("--launch", choices=["burst", "staggered"], default="staggered")
    ap.add_argument("--stagger", type=float, default=None, help="seconds between slot starts (staggered; default mean session / N)")
    ap.add_argument("--max-restores", type=int, default=0, help="restores in flight (default: N for burst, 8 for staggered)")
    ap.add_argument("--regime", default="x0.1", help="template think regime: x0.1 | real | x<scale>")
    ap.add_argument("--think-cap", type=float, default=30.0, help="the templates' agent_think_cap (for overlap accounting)")
    ap.add_argument("--duration", type=int, default=600, help="seconds of launching sessions")
    ap.add_argument("--grace", type=int, default=120, help="seconds after the duration for in-flight sessions")
    ap.add_argument("--ckpt-every", type=int, default=1, help="diff checkpoint after every k-th turn (0 = base only, -1 = none)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-recycle", action="store_true", help="one session per slot")
    ap.add_argument("--max-fails", type=int, default=2, help="consecutive failures before a slot is retired")
    ap.add_argument("--sessions", default="", help="comma-separated session names (default: every template of the regime)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--submitters", type=int, default=2)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.label is None:
        a.label = f"n{a.density}_{a.engine}_{a.launch}_{a.regime.replace('.', '')}"
    home = os.environ.get("VMD_HOME") or f"{os.environ.get('AGENTSYSPERF_HOME', os.path.expanduser('~/.agentsysperf'))}/vm_density"
    envp = f"{home}/host.env"
    if not os.path.exists(envp):
        sys.exit(f"no {envp}: run host_setup.sh first")
    env = read_env(envp)
    lockf = open(f"{home}/.cell.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another cell is running on this host (lock held)")
    cell = Cell(a, env)

    def cleanup(*_):
        cell.stop.set()
        with cell.lock:
            pids = list(cell.pids)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        shutil.rmtree(cell.sbdisks, ignore_errors=True)
        shutil.rmtree(cell.sbstore, ignore_errors=True)
        shutil.rmtree(cell.sockdir, ignore_errors=True)
        if cell.kernel_offload and cell.eng["offload"]:
            set_offloading(0)

    atexit.register(cleanup)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(s) != signal.SIG_IGN:      # keep nohup's immunity
            signal.signal(s, lambda *_: sys.exit(130))
    sys.exit(cell.run())


if __name__ == "__main__":
    main()
