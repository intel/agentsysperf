# microVM density experiment

How per-sandbox performance and host cost scale with the number of
Cloud Hypervisor agent sandboxes on one socket, with and without the DSA and
IAA accelerators, under two launch patterns: every sandbox started at once
(burst) and starts spread out in time (staggered).

Each sandbox replays a recorded Terminal-Bench 2 session (the seven sessions
shipped in Cloud Hypervisor's `benchmark/agent/traj`) through the full
lifecycle: warm restore from the session's template, the recorded turns with
a checkpoint after every turn, a full save at the end, teardown, and the slot
recycled with the next session. The replay is deterministic and has no LLM in
the loop, so the only variable across cells is the sandbox count, the
accelerator arm and the launch pattern.

| arm | checkpoint classify and diff compare | codec | restore populate |
|---|---|---|---|
| none | no daemon | - | CPU |
| cpu | CPU | lz4 | CPU |
| dsa | DSA | lz4 | uffd_dma and dcbm in the kernel, DTO memfill |
| iaa | CPU | IAA static Huffman (QPL) | CPU |
| dsaiaa | DSA | IAA | uffd_dma and dcbm |

The DSA restore side needs a kernel with the `uffd_dma` and `dcbm` providers;
without them the dsa arms fall back to the CPU and the results are labelled
`kernel_offload=0`.

## Host setup

```bash
VMD_HOME=$AGENTSYSPERF_HOME/vm_density VMD_SOCKET=0 \
GUEST_KERNEL=/path/to/vmlinuz GUEST_INITRD=/path/to/initrd \
bash experiments/vm_density/host_setup.sh
```

`host_setup.sh` runs these stages (each can be run alone, all are idempotent):

| stage | what it does |
|---|---|
| fetch | clones Cloud Hypervisor (`bravindr/cloud-hypervisor`, branch `dsa-integration`, pinned revision), Intel QPL v1.9.0 and Intel DTO v1.2 into `$VMD_HOME` |
| qpl | builds the static QPL library |
| build | `cargo build --release` of the VMM, `ch-remote` and `offload_daemon` with the `qpl` and `dto` features (DTO is built by the daemon's build script); binaries and `libdto_explicit.so` land in `$VMD_HOME/bin`, provenance in `bin/VERSIONS` |
| guest | stages the guest kernel and initrd (any bzImage with virtio-blk) |
| accel | configures the socket's DSA devices with `KERNEL_WQS` kernel dmaengine queues (for the providers) plus one shared user queue each, the socket's IAA devices with one shared user queue each, and disables the other sockets' devices. Needs the providers unloaded; if they cannot be unloaded the layout is deferred to after a reboot |
| providers | loads `dcbm` and `uffd_dma` with offloading off (the runner turns it on for the dsa arms), unprivileged userfaultfd, performance governor, NUMA balancing off |
| pool | frees the socket's idle 1 GiB huge page pool (`FREE_1G_POOL=0` keeps it) so the 2 MiB pool, which the runner sizes per cell, can grow to the full density; a boot-time 1 GiB pool returns at the next reboot |
| store | mounts a tmpfs checkpoint store bound to the socket's memory node |
| disks | creates and mounts an XFS loop file with reflink for the guest disks, so every sandbox's writable copy of a task image is a metadata operation rather than a copy |
| verify | writes `$VMD_HOME/host.env` (paths, CPU list, `DTO_WQ_LIST`, IAA queues, store, disks, guest geometry, `KERNEL_OFFLOAD`) for the runner |

The accel, providers, pool, store and disks stages use `sudo`. The accel stage switches
the providers' offloading off and unloads them before touching the queues (an
active provider pins its module, and unbinding a queue a provider holds has
hung hosts); nothing may be using DMA at that point. A host whose boot scripts
apply their own queue layout needs `host_setup.sh accel providers verify`
after every reboot.

Validated on gnr-qual1 (Xeon 6787P, socket 0: dsa0/2/4/6 and iax1/3/5/7,
kernel 7.2.0-mmoff-v1+): the lifecycle smoke in Cloud Hypervisor's
`benchmark/agent/lifecycle_cadence.sh` run against `$VMD_HOME/bin` gives full
checkpoints of 54 ms and per-turn DSA diffs of 4-12 ms with no fallbacks, and
restores of both chains boot the guest.

## Images and templates

```bash
bash experiments/vm_density/images.sh            # rootfs, templates, then a restore check per codec
bash experiments/vm_density/images.sh rootfs regex-chess_30b
bash experiments/vm_density/images.sh check circuit-fibsqrt_7b_x0.1 qpl
```

| stage | what it does |
|---|---|
| rootfs | one ext4 root filesystem per recorded session under `$VMD_DISKS/rootfs`: the Terminal-Bench task's own Docker image (the one the trajectory names; pulled from the registry the Harbor task declares, else built from its Dockerfile) plus the guest init, the recorded command of every turn and the recorded think time of every turn, built with Cloud Hypervisor's `benchmark/agent/mkrootfs_traj.sh` |
| templates | for every session and think regime (`real`, `x0.1`): boot a guest of the configured geometry from a reflink copy of its rootfs to the base point, pause it, and take one warm checkpoint per codec (`lz4` with CPU classification for the cpu and dsa arms, IAA static Huffman with DSA classification for the iaa and dsaiaa arms) into `$VMD_HOME/templates/<session>_<regime>/<codec>`, with a `manifest.json` (disk, console, geometry, kernel command line, per-codec size and time) |
| check | restores a template into a fresh VMM the way the runner does and waits for the guest's first turn |

A snapshot carries the template's disk and console paths, so a restored VMM
runs in its own user and mount namespace with the sandbox's reflink copy and
console file bound over those paths. This needs unprivileged user namespaces
(on Ubuntu 24.04 the AppArmor restriction allows plain `unshare -Urm`).

## Running a cell

```bash
python3 experiments/vm_density/vmfleet.py --density 16 --engine dsaiaa --launch burst --duration 600
python3 experiments/vm_density/vmfleet.py --density 16 --engine cpu --launch staggered --duration 600
```

`vmfleet.py` runs one cell: N sandbox slots on the socket, each restored from
a session's warm template, driven through the recorded turns with a diff
checkpoint after every turn and a full save at the end, then recycled with
the next session until `--duration` is up (in-flight sessions get `--grace`
seconds to finish). One thread per slot keeps a sandbox's checkpoints in
order while sandboxes overlap freely; a drainer thread parses every console
into per-slot events.

| option | meaning |
|---|---|
| `--launch burst` | every slot starts at once, restores in flight unbounded: the first wave of restores, turns and checkpoints collides |
| `--launch staggered` | starts spaced by `--stagger` (default: mean session length / N, at least 0.5 s), at most `--max-restores` (8) restores in flight |
| `--engine` | `none` (no daemon), `cpu`, `dsa`, `iaa`, `dsaiaa` as in the arms table; the dsa engines restore with UFFDIO_COPY and turn the providers' offloading on for the cell when the kernel has them |
| `--regime` | which templates: `x0.1` (recorded think times scaled by 0.1) or `real` |
| `--ckpt-every k` | diff after every k-th turn (0 = base checkpoint only, -1 = none); `--no-save`, `--no-recycle` |

Pause and resume go straight over the VMM's API socket; send-migration goes
through `ch-remote`; the resident daemon's `Checkpoint N` log line is the
completion signal. The cell refuses to start (exit 3, `refused.json`) when the
huge page pool, the store or the accelerator queues cannot hold it, and one
cell runs at a time per host (`$VMD_HOME/.cell.lock`).

Output under `$VMD_HOME/results/<label>/`: `result.json` with the cell
config, host samples (socket CPU busy, node free memory, huge pages, store
use, live slots, once a second), provider counter deltas, one record per
sandbox incarnation (restore wall and CPU, base checkpoint, turns, save,
VMM CPU with and without the vCPU threads, daemon CPU), every turn (guest
seconds, rc, output bytes) and every checkpoint (pause wall, after-step
delay, daemon cpu_ms, dirty and changed pages, bytes, DSA ops, whether the
pause outlasted the recorded think time), plus the per-sandbox console,
VMM, daemon and restore logs. A summary is printed at the end.

Smoke on gnr-qual1 (x0.1 regime, 2 slots, 60 s cells): dsaiaa burst restores
in 100 ms at 0.04 core-s with 920 UFFDIO_COPYs on DSA, diff checkpoints
pause 9 ms (3.6 ms of daemon CPU), saves 65 ms; cpu staggered restores in
188 ms at 0.26 core-s, diffs pause 13 ms (7.6 ms CPU), saves 115 ms; a slot
with no daemon runs the session at 0.96 core-s of VMM time.

## Running the matrix

```bash
python3 experiments/vm_density/run_experiment.py --quick      # 8 short cells, about 40 min
python3 experiments/vm_density/run_experiment.py --screen     # phase 1, 64 cells, about 14 h
python3 experiments/vm_density/run_experiment.py --confirm --densities 1,16,36   # phase 2, 3 reps
python3 experiments/vm_density/run_experiment.py --densities 16 --engines cpu,dsaiaa --regime real
```

`run_experiment.py` runs the cells of a matrix one after another with
`vmfleet.py`. `--screen` is the screening ladder: `none`, `cpu` and `dsaiaa`
at every density (1 2 4 8 16 24 32 36) under both launch patterns, then
`dsa` and `iaa` at 1, 8, 24 and 36. `--confirm` repeats every arm and
pattern three times at the densities given (N=1, the knee of the screening
curves and the cap). `--full` is every arm at every density and pattern. Any
of `--densities`, `--engines`, `--launch`, `--regime`, `--reps` and
`--duration` overrides the mode, and `--cell-args` passes extra options to
the controller. Cells are ordered rep, engine (`none cpu iaa dsa dsaiaa`),
density, launch pattern, so the arms without kernel copy offload finish
first.

Before each cell the driver sizes the socket's 2 MiB huge page pool to the
guests of the cell (one spare guest's worth on top, compaction and a cache
drop if the first attempt falls short) and checks that the node has room
beside the pool for the VMMs, the resident daemons and their checkpoints
in the node-bound store (about 1 GiB per sandbox: the raw reference image,
the base checkpoint and the save), refusing the cell otherwise
(`--force-memory` overrides), then reads the templates back into the page
cache because growing the pool evicts them; after the cell the pool must
come back whole and nothing of the cell may be left running. The driver and the
controller run on another socket than the measured one. A cell with a
`result.json` is skipped, so an interrupted run resumes with the same
command (`--rerun` repeats them); refused and failed cells are retried.
`--dry-run` lists the cells and the time they need.

Output under `$VMD_HOME/results/<mode>/`: one directory per cell (the
controller's `result.json` and logs plus `cell.log`), `run.json` (arguments,
`host.env`, binary versions) and `cells.json`, one record per cell with its
status, exit code, pool size, timing and printed summary, rewritten after
every cell.

## Analysis

```bash
python3 experiments/vm_density/analysis.py $VMD_HOME/results/screen --print --plots
```

`analysis.py` reads every cell's `result.json` under a run directory and
writes `all_results.json` (one record per cell in the scaling experiment's
shape, `config` and `agents[]` with turn durations and throughput, plus a
`vm` block with sessions per hour, turn latency over all turns and after the
ramp, time to first turn, restore, checkpoint, save and host CPU per
session, provider counters, memory and store peaks), `analysis_report.json`
(curves per engine and launch pattern over density, the median over reps;
the knee of each curve; burst against staggered at equal N; each
accelerator arm against cpu at equal N; findings) and, with `--plots`, one
PNG per metric with solid lines for burst and dashed for staggered.

A cell's steady state starts when its last slot has run its first turn, so a
burst cell's first wave and a staggered cell's ramp are reported separately
(restore p99 and time to first turn p99 of the first generation). The
per-turn slowdown compares each recorded (session, turn) with the same arm
and pattern at N=1, so the session mix a cell happened to run does not
enter it. A curve's knee is the first density where that slowdown exceeds
1.3 at the median or 2.0 at the 95th percentile, or the per-slot session
rate falls under 70 % of N=1.
