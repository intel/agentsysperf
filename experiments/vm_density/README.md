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
| store | mounts a tmpfs checkpoint store bound to the socket's memory node |
| disks | creates and mounts an XFS loop file with reflink for the guest disks, so every sandbox's writable copy of a task image is a metadata operation rather than a copy |
| verify | writes `$VMD_HOME/host.env` (paths, CPU list, `DTO_WQ_LIST`, IAA queues, store, disks, guest geometry, `KERNEL_OFFLOAD`) for the runner |

The accel, providers and store stages use `sudo`. The accel stage switches
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
