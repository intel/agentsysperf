#!/usr/bin/env python3
"""Analyse a microVM density run: scaling curves, knees, burst vs staggered,
accelerator gain.

    analysis.py $VMD_HOME/results/screen [--print] [--plots] [--out DIR]

Reads every <run>/<cell>/result.json written by vmfleet.py and writes

  all_results.json      one record per cell in the scaling experiment's shape
                        (config, agents[] with turn_durations and throughput)
                        plus a `vm` block: sessions per hour, turn latency
                        (all turns and the steady state after the ramp),
                        time to first turn, restore, checkpoint, save and host
                        CPU per session, provider counters, node CPU busy,
                        memory and store peaks
  analysis_report.json  curves per engine and launch pattern over density
                        (median over reps), the knee of each curve, burst vs
                        staggered at equal N, each accelerator arm against cpu
                        at equal N, findings
  plots/*.png           with --plots (matplotlib)

A cell's steady state starts when the last slot has run its first turn, so a
burst cell's first wave and a staggered cell's ramp are reported separately
(ramp: restore p99 and time to first turn p99 of the first generation). The
per-turn slowdown compares each (session, turn) with the same engine and
launch pattern at N=1, so it is independent of which sessions a cell
happened to run. A curve's knee is the first density where the per-turn
slowdown exceeds 1.3 at the median or 2.0 at the 95th percentile, or the
per-slot session rate falls under 70 % of N=1.
"""
import argparse
import glob
import json
import os
import statistics
import sys

ENGINES = ["none", "cpu", "iaa", "dsa", "dsaiaa"]
LAUNCHES = ["burst", "staggered"]
KNEE_SLOWDOWN = 1.3
KNEE_SLOWDOWN_P95 = 2.0
KNEE_RATE = 0.7


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def r(x, n=1):
    return None if x is None else round(x, n)


def ms(x):
    return None if x is None else round(x * 1000.0, 1)


# ---- one cell
def summarize(path):
    res = json.load(open(path))
    cfg, host = res["config"], res["host"]
    sbs, steps, ckpts = res["sandboxes"], res["steps"], res["ckpts"]
    n = cfg["density"]
    t0 = host["t_start"]
    hours = (host["t_stop"] - t0) / 3600.0
    ok = [s for s in sbs if "failed" not in s]
    done = [s for s in ok if s.get("t_done")]
    failed = [s for s in sbs if "failed" in s]
    cut = [s for s in sbs if s.get("cut")]

    # the ramp ends when every slot has run its first turn (first generation)
    first = {}
    for s in sbs:
        if s.get("t_first_turn") and s["slot"] not in first:
            first[s["slot"]] = s["t_first_turn"]
    ramp_end = max(first.values()) if first else t0
    gen1 = [s for s in sbs if s["gen"] == 1]

    # steps: [slot, gen, session, k, turn, t_host, secs, rc, out, err]
    secs_all = [st[6] for st in steps]
    secs_steady = [st[6] for st in steps if st[5] >= ramp_end]
    per_turn = {}
    for st in steps:
        per_turn.setdefault(f"{st[2]}/{st[4]}", []).append(st[6])

    def rest(xs):
        return [s["restore"] for s in xs if s.get("restore") and s["restore"].get("rc") == 0]

    ttft = [s["t_first_turn"] - s["t_launch"] for s in ok if s.get("t_first_turn") and s.get("t_launch")]
    ttft1 = [s["t_first_turn"] - s["t_launch"] for s in gen1 if s.get("t_first_turn") and s.get("t_launch")]
    bases = [c for c in ckpts if c["kind"] == "base" and "failed" not in c]
    diffs = [c for c in ckpts if c["kind"] == "step" and "failed" not in c]
    cfail = [c for c in ckpts if "failed" in c]
    saves = [s["save"] for s in sbs if s.get("save") and "failed" not in s["save"]]
    vm_cpu = [s["vmm_cpu"] for s in done if s.get("vmm_cpu") is not None]
    vc_cpu = [s["vmm_vcpu_cpu"] for s in done if s.get("vmm_vcpu_cpu") is not None]
    dm_cpu = [s["daemon_cpu"] for s in done if s.get("daemon_cpu") is not None]
    tot_cpu = [(s.get("vmm_cpu") or 0) + (s.get("daemon_cpu") or 0) for s in done if s.get("vmm_cpu") is not None]
    sess_secs = [s["t_done"] - s["t_launch"] for s in done if s.get("t_launch")]

    # samples: [t, node cpu busy, node free MiB, hp free, store MiB, live]
    samples = host.get("samples", [])
    steady_busy = [x[1] for x in samples if t0 + x[0] >= ramp_end and x[5] > 0]
    busy = [x[1] for x in samples if x[5] > 0]
    pd = host.get("providers_delta", {})
    fallbacks = sum(pd.get(k, 0) for k in ("uffd_copies_failed", "uffd_copies_refused", "uffd_admit_failures",
                                           "dcbm_admit_failures", "dcbm_clear_failures"))
    if diffs:
        redo = sum(c.get("dsa_cpu_redo", 0) or 0 for c in diffs)
    else:
        redo = 0

    # the scaling experiment's shape: one agent per slot
    agents = []
    for slot in range(n):
        st_s = [st for st in steps if st[0] == slot]
        t_slot = [s for s in sbs if s["slot"] == slot]
        span = (max((s.get("t_torn_down") or s.get("t_done") or t0) for s in t_slot)
                - min(s.get("t_launch", t0) for s in t_slot)) if t_slot else 0.0
        agents.append({"agent_id": slot, "phase_mix": cfg["regime"], "cpuset": [], "turns_completed": len(st_s),
                       "total_duration_s": round(span, 3),
                       "throughput_turns_per_s": round(len(st_s) / span, 4) if span > 0 else 0.0,
                       "mean_turn_ms": ms(sum(st[6] for st in st_s) / len(st_s)) if st_s else 0.0,
                       "turn_durations": [{"total_s": st[6], "turn": st[4], "session": st[2]} for st in st_s],
                       "sessions_done": len([s for s in t_slot if s.get("t_done")]),
                       "error": "; ".join(s["failed"] for s in t_slot if "failed" in s) or None})
    rates = [a["throughput_turns_per_s"] for a in agents]
    vm = {
        "sessions_done": len(done), "sessions_failed": len(failed), "sessions_cut": len(cut),
        "sessions_per_hour": r(len(done) / hours, 1) if hours > 0 else None,
        "sessions_per_hour_per_slot": r(len(done) / hours / n, 2) if hours > 0 else None,
        "session_secs_p50": r(pct(sess_secs, 50), 1),
        "turns": len(secs_all), "turns_steady": len(secs_steady),
        "turn_ms": {"p50": ms(pct(secs_all, 50)), "p95": ms(pct(secs_all, 95)), "p99": ms(pct(secs_all, 99)),
                    "max": ms(max(secs_all)) if secs_all else None,
                    "mean": ms(sum(secs_all) / len(secs_all)) if secs_all else None},
        "turn_steady_ms": {"p50": ms(pct(secs_steady, 50)), "p95": ms(pct(secs_steady, 95)),
                           "p99": ms(pct(secs_steady, 99))},
        "per_turn_ms": {k: ms(med(v)) for k, v in per_turn.items()},
        "ramp_s": r(ramp_end - t0, 1),
        "ttft_ms": {"p50": ms(pct(ttft, 50)), "p99": ms(pct(ttft, 99))},
        "ramp": {"restore_p99_ms": ms(pct([x["wall"] for x in rest(gen1)], 99)), "ttft_p99_ms": ms(pct(ttft1, 99)),
                 "restores": len(rest(gen1))},
        "restore": {"n": len(rest(sbs)), "wall_p50_ms": ms(pct([x["wall"] for x in rest(sbs)], 50)),
                    "wall_p99_ms": ms(pct([x["wall"] for x in rest(sbs)], 99)),
                    "cpu_core_s_p50": r(pct([x.get("cpu") for x in rest(sbs)], 50), 3),
                    "failed": len([s for s in sbs if s.get("restore") and s["restore"].get("rc") not in (0, None)])},
        "base_ckpt": {"n": len(bases), "wall_p50_ms": ms(pct([c["wall"] for c in bases], 50)),
                      "cpu_ms_p50": r(pct([c.get("cpu_ms") for c in bases], 50), 1)},
        "diff_ckpt": {"n": len(diffs), "failed": len(cfail),
                      "pause_p50_ms": ms(pct([c["pause_wall"] for c in diffs], 50)),
                      "pause_p99_ms": ms(pct([c["pause_wall"] for c in diffs], 99)),
                      "cpu_ms_p50": r(pct([c.get("cpu_ms") for c in diffs], 50), 2),
                      "overlap_frac": r(len([c for c in diffs if c.get("overlap")]) / len(diffs), 3) if diffs else None,
                      "dirty_pages_p50": r(pct([c.get("dirty_pages") for c in diffs], 50), 0),
                      "changed_pages_p50": r(pct([c.get("changed_pages") for c in diffs], 50), 0),
                      "bytes_p50": r(pct([c.get("output_bytes") for c in diffs], 50), 0),
                      "dsa_ops": sum(c.get("dsa_ops", 0) or 0 for c in diffs), "dsa_cpu_redo": redo},
        "save": {"n": len(saves), "wall_p50_ms": ms(pct([s["wall"] for s in saves], 50)),
                 "cpu_core_s_p50": r(pct([s.get("cpu") for s in saves], 50), 3),
                 "mib_p50": r(pct([s["bytes"] for s in saves], 50) / 2**20, 0) if saves else None},
        "core_s_per_session": {"vmm": r(pct(vm_cpu, 50), 3), "vcpu": r(pct(vc_cpu, 50), 3),
                               "daemon": r(pct(dm_cpu, 50), 3), "total": r(pct(tot_cpu, 50), 3)},
        "node_cpu_busy": {"mean": r(sum(busy) / len(busy), 4) if busy else None,
                          "steady_mean": r(sum(steady_busy) / len(steady_busy), 4) if steady_busy else None},
        "node_free_min_mib": min(x[2] for x in samples) if samples else None,
        "store_max_mib": max(x[4] for x in samples) if samples else None,
        "providers": pd, "fallbacks": fallbacks,
    }
    return {"label": cfg["label"], "cell_dir": os.path.dirname(path),
            "config": {"density": n, "socket": cfg.get("socket"), "launch": cfg["launch"], "stagger_s": cfg.get("stagger_s"),
                       "max_restores": cfg.get("max_restores"), "engine": cfg["engine"], "regime": cfg["regime"],
                       "kernel_offload": cfg.get("kernel_offload"), "uffd_copy": cfg.get("uffd_copy"),
                       "ckpt_every": cfg.get("ckpt_every"), "duration_s": cfg.get("duration_s"),
                       "guest_mem": cfg.get("guest_mem"), "guest_vcpus": cfg.get("guest_vcpus"),
                       "rep": _rep(cfg["label"]), "placement": "single_socket", "phase_mix": cfg["regime"]},
            "mean_throughput_turns_per_s": round(sum(rates) / len(rates), 4) if rates else 0.0,
            "aggregate_throughput_turns_per_s": round(sum(rates), 4),
            "total_duration_s": round(host["t_stop"] - t0, 1),
            "agents": agents, "error": None, "vm": vm,
            "host": {"hostname": host.get("hostname"), "kernel": host.get("kernel"), "t_start": t0,
                     "bin_versions": cfg.get("bin_versions")}}


def _rep(label):
    tail = label.rsplit("_r", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 1


# ---- curves
def point(cells):
    """One curve point: the median over the reps of a (engine, launch, density)."""
    def m(f):
        return med([f(c) for c in cells])
    vm = lambda c: c["vm"]  # noqa: E731
    p = {"density": cells[0]["config"]["density"], "reps": len(cells), "labels": [c["label"] for c in cells],
         "sessions_per_hour": m(lambda c: vm(c)["sessions_per_hour"]),
         "sessions_per_hour_per_slot": m(lambda c: vm(c)["sessions_per_hour_per_slot"]),
         "turn_p50_ms": m(lambda c: vm(c)["turn_ms"]["p50"]), "turn_p95_ms": m(lambda c: vm(c)["turn_ms"]["p95"]),
         "turn_p99_ms": m(lambda c: vm(c)["turn_ms"]["p99"]),
         "steady_p50_ms": m(lambda c: vm(c)["turn_steady_ms"]["p50"]),
         "steady_p99_ms": m(lambda c: vm(c)["turn_steady_ms"]["p99"]),
         "ttft_p50_ms": m(lambda c: vm(c)["ttft_ms"]["p50"]), "ttft_p99_ms": m(lambda c: vm(c)["ttft_ms"]["p99"]),
         "ramp_restore_p99_ms": m(lambda c: vm(c)["ramp"]["restore_p99_ms"]),
         "ramp_ttft_p99_ms": m(lambda c: vm(c)["ramp"]["ttft_p99_ms"]),
         "restore_p50_ms": m(lambda c: vm(c)["restore"]["wall_p50_ms"]),
         "restore_p99_ms": m(lambda c: vm(c)["restore"]["wall_p99_ms"]),
         "restore_cpu_core_s": m(lambda c: vm(c)["restore"]["cpu_core_s_p50"]),
         "base_ckpt_p50_ms": m(lambda c: vm(c)["base_ckpt"]["wall_p50_ms"]),
         "ckpt_pause_p50_ms": m(lambda c: vm(c)["diff_ckpt"]["pause_p50_ms"]),
         "ckpt_pause_p99_ms": m(lambda c: vm(c)["diff_ckpt"]["pause_p99_ms"]),
         "ckpt_cpu_ms": m(lambda c: vm(c)["diff_ckpt"]["cpu_ms_p50"]),
         "ckpt_overlap_frac": m(lambda c: vm(c)["diff_ckpt"]["overlap_frac"]),
         "save_p50_ms": m(lambda c: vm(c)["save"]["wall_p50_ms"]),
         "save_cpu_core_s": m(lambda c: vm(c)["save"]["cpu_core_s_p50"]),
         "core_s_per_session": m(lambda c: vm(c)["core_s_per_session"]["total"]),
         "vmm_core_s_per_session": m(lambda c: vm(c)["core_s_per_session"]["vmm"]),
         "daemon_core_s_per_session": m(lambda c: vm(c)["core_s_per_session"]["daemon"]),
         "node_cpu_busy": m(lambda c: vm(c)["node_cpu_busy"]["steady_mean"]),
         "node_cpu_busy_pct": m(lambda c: None if vm(c)["node_cpu_busy"]["steady_mean"] is None
                                 else vm(c)["node_cpu_busy"]["steady_mean"] * 100),
         "node_free_min_mib": m(lambda c: vm(c)["node_free_min_mib"]),
         "store_max_mib": m(lambda c: vm(c)["store_max_mib"]),
         "fallbacks": sum(vm(c)["fallbacks"] for c in cells),
         "sessions_failed": sum(vm(c)["sessions_failed"] for c in cells),
         "ckpts_failed": sum(vm(c)["diff_ckpt"]["failed"] for c in cells)}
    # per-(session, turn) medians pooled over the reps, for the slowdown against N=1
    pooled = {}
    for c in cells:
        for k, v in vm(c)["per_turn_ms"].items():
            pooled.setdefault(k, []).append(v)
    p["_per_turn"] = {k: med(v) for k, v in pooled.items()}
    return p


def slowdown(p, base):
    ratios = [p["_per_turn"][k] / base["_per_turn"][k] for k in p["_per_turn"]
              if k in base["_per_turn"] and base["_per_turn"][k] and p["_per_turn"][k] is not None]
    return {"slowdown_p50": r(pct(ratios, 50), 3), "slowdown_p95": r(pct(ratios, 95), 3), "slowdown_turns": len(ratios)}


def build_curves(cells):
    groups = {}
    for c in cells:
        k = (c["config"]["engine"], c["config"]["launch"], c["config"]["regime"])
        groups.setdefault(k, {}).setdefault(c["config"]["density"], []).append(c)
    curves = {}
    for (engine, launch, regime), by_n in sorted(groups.items()):
        pts = [point(by_n[n]) for n in sorted(by_n)]
        base = pts[0] if pts and pts[0]["density"] == 1 else None
        for p in pts:
            if base:
                p.update(slowdown(p, base))
                b = base["sessions_per_hour_per_slot"]
                p["normalized_throughput"] = r(p["sessions_per_hour_per_slot"] / b, 3) if b else None
                p["p99_vs_n1"] = r(p["turn_p99_ms"] / base["turn_p99_ms"], 3) if base["turn_p99_ms"] else None
        for p in pts:
            p.pop("_per_turn", None)
        key = f"{engine}_{launch}" + ("" if regime == "x0.1" else f"_{regime}")
        curves[key] = pts
    return curves


def knees(curves):
    out = []
    for key, pts in curves.items():
        if len(pts) < 2 or pts[0]["density"] != 1:
            continue
        for p in pts[1:]:
            reasons = []
            if p.get("slowdown_p50") is not None and p["slowdown_p50"] > KNEE_SLOWDOWN:
                reasons.append(f"per-turn p50 {p['slowdown_p50']:.2f}x N=1")
            if p.get("normalized_throughput") is not None and p["normalized_throughput"] < KNEE_RATE:
                reasons.append(f"sessions/h per slot {p['normalized_throughput']:.0%} of N=1")
            if p.get("slowdown_p95") is not None and p["slowdown_p95"] > KNEE_SLOWDOWN_P95:
                reasons.append(f"per-turn p95 {p['slowdown_p95']:.2f}x N=1")
            if reasons:
                out.append({"curve": key, "density": p["density"], "reasons": reasons,
                            "slowdown_p50": p.get("slowdown_p50"), "normalized_throughput": p.get("normalized_throughput"),
                            "turn_p99_ms": p["turn_p99_ms"]})
                break
    return out


def by_density(curves, key):
    return {p["density"]: p for p in curves.get(key, [])}


def launch_comparison(curves):
    """burst against staggered at equal engine and N."""
    out = []
    keys = {k.replace("_burst", "").replace("_staggered", "") for k in curves}
    for base in sorted(keys):
        eng, _, regime = base.partition("_")      # keys look like engine_launch[_regime]
        kb = f"{eng}_burst" + (f"_{regime}" if regime else "")
        ks = f"{eng}_staggered" + (f"_{regime}" if regime else "")
        b, s = by_density(curves, kb), by_density(curves, ks)
        for n in sorted(set(b) & set(s)):
            pb, ps = b[n], s[n]
            def ratio(k):
                return r(pb[k] / ps[k], 3) if pb.get(k) and ps.get(k) else None
            out.append({"engine": eng, "regime": regime or "x0.1", "density": n,
                        "burst_over_staggered": {"turn_p99": ratio("turn_p99_ms"), "turn_p50": ratio("turn_p50_ms"),
                                                 "ttft_p99": ratio("ttft_p99_ms"), "restore_p99": ratio("restore_p99_ms"),
                                                 "ckpt_pause_p99": ratio("ckpt_pause_p99_ms"),
                                                 "sessions_per_hour": ratio("sessions_per_hour"),
                                                 "core_s_per_session": ratio("core_s_per_session")},
                        "burst": {"ramp_restore_p99_ms": pb["ramp_restore_p99_ms"], "ramp_ttft_p99_ms": pb["ramp_ttft_p99_ms"],
                                  "fallbacks": pb["fallbacks"]},
                        "staggered": {"stagger_s": None, "fallbacks": ps["fallbacks"]}})
    return out


def accelerator_gain(curves):
    """each accelerator arm against cpu at equal launch and N (ratios: arm / cpu)."""
    out = []
    for launch in LAUNCHES:
        for regime in ("", "_real"):
            cpu = by_density(curves, f"cpu_{launch}{regime}")
            if not cpu:
                continue
            for eng in ("none", "iaa", "dsa", "dsaiaa"):
                arm = by_density(curves, f"{eng}_{launch}{regime}")
                for n in sorted(set(cpu) & set(arm)):
                    a, c = arm[n], cpu[n]
                    def ratio(k):
                        return r(a[k] / c[k], 3) if a.get(k) and c.get(k) else None
                    out.append({"engine": eng, "launch": launch, "regime": regime.strip("_") or "x0.1", "density": n,
                                "vs_cpu": {"core_s_per_session": ratio("core_s_per_session"),
                                           "daemon_core_s": ratio("daemon_core_s_per_session"),
                                           "restore_cpu": ratio("restore_cpu_core_s"), "restore_p50": ratio("restore_p50_ms"),
                                           "ckpt_cpu_ms": ratio("ckpt_cpu_ms"), "ckpt_pause_p50": ratio("ckpt_pause_p50_ms"),
                                           "save_cpu": ratio("save_cpu_core_s"), "turn_p50": ratio("turn_p50_ms"),
                                           "turn_p99": ratio("turn_p99_ms"), "sessions_per_hour": ratio("sessions_per_hour")}})
    return out


def findings(curves, kn, lc, ag):
    out = []
    for k in kn:
        out.append(f"[{k['curve']}] knee at N={k['density']}: " + "; ".join(k["reasons"]))
    for key, pts in curves.items():
        if pts and pts[0]["density"] == 1 and len(pts) > 1:
            viable = [p for p in pts if (p.get("normalized_throughput") or 0) >= 0.5 and p["sessions_per_hour"]]
            if viable:
                top = max(viable, key=lambda p: p["density"])
                out.append(f"[{key}] largest density with per-slot session rate >= 50 % of N=1: {top['density']} "
                           f"({top['normalized_throughput']:.0%}, {top['sessions_per_hour']:.0f} sessions/h)")
        best = max(pts, key=lambda p: p["sessions_per_hour"] or 0)
        if best["sessions_per_hour"]:
            out.append(f"[{key}] peak {best['sessions_per_hour']:.0f} sessions/h at N={best['density']}")
    worst = [x for x in lc if x["burst_over_staggered"]["turn_p99"]]
    if worst:
        w = max(worst, key=lambda x: x["burst_over_staggered"]["turn_p99"])
        out.append(f"burst vs staggered: largest turn p99 penalty {w['burst_over_staggered']['turn_p99']:.2f}x "
                   f"for {w['engine']} at N={w['density']}")
    for eng in ("dsaiaa", "dsa", "iaa"):
        xs = [x for x in ag if x["engine"] == eng and x["vs_cpu"]["core_s_per_session"]]
        if xs:
            top = max(xs, key=lambda x: x["density"])
            out.append(f"{eng} vs cpu at N={top['density']} ({top['launch']}): host core-s per session "
                       f"{top['vs_cpu']['core_s_per_session']:.2f}x, restore cpu {top['vs_cpu']['restore_cpu']}x, "
                       f"ckpt cpu {top['vs_cpu']['ckpt_cpu_ms']}x, turn p99 {top['vs_cpu']['turn_p99']}x")
    return out


# ---- output
def print_curves(curves, kn):
    cols = [("N", "density", "{:>3}"), ("reps", "reps", "{:>4}"), ("sess/h", "sessions_per_hour", "{:>7.0f}"),
            ("norm", "normalized_throughput", "{:>5.2f}"), ("turn p50", "turn_p50_ms", "{:>8.0f}"),
            ("p99", "turn_p99_ms", "{:>7.0f}"), ("slow p50", "slowdown_p50", "{:>8.2f}"),
            ("ttft p99", "ttft_p99_ms", "{:>8.0f}"), ("rest p50", "restore_p50_ms", "{:>8.0f}"),
            ("p99", "restore_p99_ms", "{:>7.0f}"), ("rcpu", "restore_cpu_core_s", "{:>5.2f}"),
            ("ckpt p50", "ckpt_pause_p50_ms", "{:>8.1f}"), ("p99", "ckpt_pause_p99_ms", "{:>7.0f}"),
            ("ckcpu", "ckpt_cpu_ms", "{:>5.1f}"), ("save", "save_p50_ms", "{:>5.0f}"),
            ("core-s", "core_s_per_session", "{:>6.2f}"), ("busy%", "node_cpu_busy_pct", "{:>5.1f}"),
            ("fb", "fallbacks", "{:>4}"), ("fail", "sessions_failed", "{:>4}")]
    knee_at = {k["curve"]: k["density"] for k in kn}
    for key, pts in curves.items():
        print(f"\n== {key}" + (f"  (knee at N={knee_at[key]})" if key in knee_at else ""))
        print(" ".join(f"{h:>{len(f.format(0))}}" for h, _, f in cols))
        for p in pts:
            cells = []
            for h, k, f in cols:
                v = p.get(k)
                w = len(f.format(0))
                cells.append(f.format(v) if v is not None else f"{'-':>{w}}")
            print(" ".join(cells))


def plots(curves, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("no matplotlib: plots skipped", file=sys.stderr)
        return []
    os.makedirs(outdir, exist_ok=True)
    metrics = [("sessions_per_hour", "sessions per hour", "sessions_per_hour"),
               ("turn_p50_ms", "turn p50 (ms)", "turn_p50"), ("turn_p99_ms", "turn p99 (ms)", "turn_p99"),
               ("slowdown_p50", "per-turn p50 slowdown vs N=1", "slowdown"),
               ("ttft_p99_ms", "time to first turn p99 (ms)", "ttft_p99"),
               ("restore_p99_ms", "restore wall p99 (ms)", "restore_p99"),
               ("ckpt_pause_p99_ms", "checkpoint pause p99 (ms)", "ckpt_pause_p99"),
               ("core_s_per_session", "host core-s per session (VMM + daemon)", "core_s"),
               ("node_cpu_busy_pct", "socket CPU busy % (steady state)", "node_busy")]
    colors = {"none": "0.5", "cpu": "tab:blue", "iaa": "tab:green", "dsa": "tab:orange", "dsaiaa": "tab:red"}
    files = []
    for k, title, name in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        drawn = 0
        for key, pts in curves.items():
            xs = [p["density"] for p in pts if p.get(k) is not None]
            ys = [p[k] for p in pts if p.get(k) is not None]
            if not xs:
                continue
            eng = key.split("_")[0]
            ls = "-" if "_burst" in key else "--"
            ax.plot(xs, ys, ls, marker="o", ms=4, color=colors.get(eng, "k"), label=key)
            drawn += 1
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xlabel("sandboxes on the socket (N)")
        ax.set_ylabel(title)
        ax.set_title(title + " vs density (solid burst, dashed staggered)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        f = f"{outdir}/{name}.png"
        fig.savefig(f, dpi=130, bbox_inches="tight")
        plt.close(fig)
        files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a run_experiment.py output directory (or any directory of cell directories)")
    ap.add_argument("--out", default=None, help="where to write (default: the run directory)")
    ap.add_argument("--print", action="store_true", help="print the curves")
    ap.add_argument("--plots", action="store_true", help="write plots/*.png")
    a = ap.parse_args()
    out = a.out or a.run_dir
    paths = sorted(glob.glob(f"{a.run_dir}/*/result.json"))
    if not paths:
        sys.exit(f"no */result.json under {a.run_dir}")
    cells = []
    for p in paths:
        try:
            cells.append(summarize(p))
        except (KeyError, ValueError, ZeroDivisionError) as e:
            print(f"skipping {p}: {e!r}", file=sys.stderr)
    curves = build_curves(cells)
    kn = knees(curves)
    lc = launch_comparison(curves)
    ag = accelerator_gain(curves)
    fd = findings(curves, kn, lc, ag)
    os.makedirs(out, exist_ok=True)
    json.dump(cells, open(f"{out}/all_results.json", "w"), indent=1)
    report = {"summary": {"cells": len(cells), "curves": len(curves), "knees": len(kn),
                          "hosts": sorted({c["host"]["hostname"] for c in cells}),
                          "kernels": sorted({c["host"]["kernel"] for c in cells if c["host"]["kernel"]})},
              "scaling_curves": curves, "knees": kn, "launch_comparison": lc, "accelerator_gain": ag, "findings": fd}
    json.dump(report, open(f"{out}/analysis_report.json", "w"), indent=1)
    print(f"{len(cells)} cells, {len(curves)} curves -> {out}/all_results.json, analysis_report.json")
    if a.print:
        print_curves(curves, kn)
        print()
    for f in fd:
        print("  " + f)
    if a.plots:
        for f in plots(curves, f"{out}/plots"):
            print("  " + f)


if __name__ == "__main__":
    main()
