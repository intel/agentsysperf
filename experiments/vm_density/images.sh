#!/usr/bin/env bash
# Guest images and warm templates for the microVM density experiment.
#
#   images.sh [rootfs|templates|check|all] [session ...]
#
# rootfs     one ext4 root filesystem per recorded session: the Terminal-Bench
#            task's own Docker image (pulled from the registry the Harbor task
#            declares, else built from its Dockerfile) plus the guest init, the
#            recorded command of every turn and the recorded think time of every
#            turn, built by Cloud Hypervisor's benchmark/agent/mkrootfs_traj.sh.
# templates  for every session and think regime: boot the guest to its base
#            point and take one warm checkpoint per codec (lz4 for the cpu and
#            dsa arms, IAA static Huffman for the iaa and dsaiaa arms).
# check      restore a template into a fresh VMM the way the runner does and
#            wait for the guest's first turn (check [template [format]]).
#
# Reads $VMD_HOME/host.env (host_setup.sh). Environment: SESSIONS (default: every
# trajectory in the Cloud Hypervisor checkout), REGIMES ("real x0.1"), FORMATS
# ("lz4 qpl"), THINK_CAP (30 s), FIRST_WAIT (6 s), CHUNK (2 MiB), WORKERS (8),
# HARBOR_TASKS (~/.cache/harbor/tasks), FORCE=1 to rebuild what exists.
set -euo pipefail

VMD_HOME=${VMD_HOME:-${AGENTSYSPERF_HOME:-$HOME/.agentsysperf}/vm_density}
[ -f "$VMD_HOME/host.env" ] || { echo "no $VMD_HOME/host.env: run host_setup.sh first" >&2; exit 1; }
set -a; . "$VMD_HOME/host.env"; set +a
AGENT=$CH_SRC/benchmark/agent
TRAJ=$AGENT/traj
ROOTFS=$VMD_DISKS/rootfs
TMPL_DISKS=$VMD_DISKS/tmpl
TMPL=$VMD_HOME/templates
WORK=$VMD_HOME/work
HARBOR_TASKS=${HARBOR_TASKS:-$HOME/.cache/harbor/tasks}
REGIMES=${REGIMES:-"real x0.1"}
FORMATS=${FORMATS:-"lz4 qpl"}
THINK_CAP=${THINK_CAP:-30}
FIRST_WAIT=${FIRST_WAIT:-6}
CHUNK=${CHUNK:-2097152}
WORKERS=${WORKERS:-8}
FORCE=${FORCE:-0}
NODE=$VMD_SOCKET
CH=$VMD_BIN/cloud-hypervisor; REM=$VMD_BIN/ch-remote; OD=$VMD_BIN/offload_daemon
export DTO_WQ_LIST DTO_IS_NUMA_AWARE=1 DTO_LOG_LEVEL=1 LD_LIBRARY_PATH=$VMD_BIN

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }
sessions() { if [ $# -gt 0 ]; then echo "$@"; else ls "$TRAJ"/*.traj.json | sed 's#.*/##; s#\.traj\.json$##'; fi; }
traj_field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$TRAJ/$1.traj.json" "$2"; }
turn_range() { python3 -c 'import json,sys; ts=[x["turn"] for x in json.load(open(sys.argv[1]))["turns"]]; print(min(ts), max(ts))' "$TRAJ/$1.traj.json"; }
manifest_field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
wait_marker() { local f=$1 m=$2 to=$3 i; for i in $(seq $((to * 5))); do grep -aq "$m" "$f" 2>/dev/null && return 0; sleep 0.2; done; return 1; }
hp_free() { cat "/sys/devices/system/node/node$NODE/hugepages/hugepages-2048kB/free_hugepages"; }
hp_needed() { echo $(( $(numfmt --from=iec "${GUEST_MEM}") / 2097152 )); }
stop_vmm() { kill -9 "$1" 2>/dev/null || true; wait "$1" 2>/dev/null || true; }

# ---------------------------------------------------------------- rootfs
ensure_image() { # session -> the docker image tag the trajectory was recorded in
    local s=$1 image task tdir declared
    image=$(traj_field "$s" image); [ -n "$image" ] || die "$s: trajectory has no image field"
    if docker image inspect "$image" >/dev/null 2>&1; then echo "$image"; return; fi
    task=$(traj_field "$s" task); task=${task#terminal-bench/}
    tdir=$(ls -d "$HARBOR_TASKS"/packages/terminal-bench/"$task"/*/ 2>/dev/null | head -1)
    [ -n "$tdir" ] || die "$s: image $image is not present and task $task is not in $HARBOR_TASKS (an agentsysperf run downloads the tasks)"
    declared=$(sed -n 's/^docker_image *= *"\(.*\)"/\1/p' "$tdir/task.toml" | head -1)
    if [ -n "$declared" ] && docker pull -q "$declared" >/dev/null 2>&1; then
        docker tag "$declared" "$image"
        log "rootfs: $s: pulled $declared as $image"
    else
        log "rootfs: $s: building $image from $tdir/environment"
        docker build -q -t "$image" "$tdir/environment" >/dev/null || die "$s: docker build failed"
    fi
    echo "$image"
}
render_think() { # session image: write /traj/<session>/<turn>.think (recorded latency, seconds) into the ext4 image; no mount, no root
    local s=$1 img=$2 tk n=0 f
    tk=$(mktemp -d)
    python3 - "$TRAJ/$s.traj.json" "$tk" <<'PY'
import json, os, sys
t = json.load(open(sys.argv[1])); d = sys.argv[2]
for x in t["turns"]:
    ms = x.get("latency_ms")
    if ms is not None:
        open(os.path.join(d, "%d.think" % x["turn"]), "w").write("%.3f\n" % (ms / 1000.0))
PY
    for f in "$tk"/*.think; do
        debugfs -w -R "rm /traj/$s/$(basename "$f")" "$img" >/dev/null 2>&1 || true
        debugfs -w -R "write $f /traj/$s/$(basename "$f")" "$img" >/dev/null 2>&1 || die "$s: debugfs could not write $(basename "$f")"
        n=$((n + 1))
    done
    rm -rf "$tk"
    echo "$n"
}
stage_rootfs() {
    mkdir -p "$ROOTFS" "$WORK"
    local s image out nk nc
    for s in $(sessions "$@"); do
        out=$ROOTFS/rootfs_traj_$s.ext4
        if [ -f "$out" ] && [ "$FORCE" != 1 ]; then log "rootfs: $s: exists (FORCE=1 rebuilds)"; continue; fi
        image=$(ensure_image "$s")
        ROOTFS_DIR=$ROOTFS WORK_DIR=$WORK bash "$AGENT/mkrootfs_traj.sh" "$s" "$image" > "$WORK/$s.mkrootfs.log" 2>&1 \
            || die "$s: mkrootfs_traj.sh failed, see $WORK/$s.mkrootfs.log"
        nk=$(render_think "$s" "$out")
        nc=$(debugfs -R "ls -l /traj/$s" "$out" 2>/dev/null | grep -c '\.cmd$' || true)
        log "rootfs: $s: $image -> $(du -m "$out" | cut -f1) MiB, $nc command files, $nk think files"
    done
}

# ---------------------------------------------------------------- templates
stage_templates() {
    mkdir -p "$TMPL" "$TMPL_DISKS"
    local s reg name dir disk con sock lo hi think cmdline chpid osock dpid fmt codec cls t0 t1 rc ms formats
    for s in $(sessions "$@"); do
        [ -f "$ROOTFS/rootfs_traj_$s.ext4" ] || die "$s: no rootfs (run the rootfs stage)"
        read -r lo hi <<<"$(turn_range "$s")"
        for reg in $REGIMES; do
            name=${s}_$reg; dir=$TMPL/$name; disk=$TMPL_DISKS/$name.ext4; con=$dir/console.log
            if [ -f "$dir/manifest.json" ] && [ "$FORCE" != 1 ]; then log "templates: $name: exists"; continue; fi
            [ "$(hp_free)" -ge "$(hp_needed)" ] || die "templates: node $NODE has $(hp_free) free 2 MiB pages, a $GUEST_MEM guest needs $(hp_needed)"
            rm -rf "$dir"; mkdir -p "$dir"
            cp --reflink=always "$ROOTFS/rootfs_traj_$s.ext4" "$disk"
            case $reg in
                real) think="agent_think=real agent_think_scale=1.0" ;;
                x*)   think="agent_think=real agent_think_scale=${reg#x}" ;;
                *)    think="agent_think=$reg" ;;
            esac
            cmdline="console=ttyS0 root=/dev/vda rw init=/agent_init agent_wl=idle agent_warm= agent_steps=traj-$s:$lo-$hi $think agent_think_cap=$THINK_CAP agent_exit=idle agent_first_wait=$FIRST_WAIT"
            sock=$dir/api.sock; rm -f "$sock"; : > "$con"
            numactl -N"$NODE" -m"$NODE" "$CH" --api-socket "$sock" --kernel "$GUEST_KERNEL" --initramfs "$GUEST_INITRD" \
                --disk "path=$disk,image_type=raw" --cmdline "$cmdline" --cpus "boot=$GUEST_VCPUS" \
                --memory "size=$GUEST_MEM,shared=on,hugepages=on,hugepage_size=2M,prefault=on" \
                --serial "file=$con" --console off > "$dir/ch.log" 2>&1 &
            chpid=$!
            wait_marker "$con" AGENT_BASE_POINT 120 || { stop_vmm $chpid; die "templates: $name: no AGENT_BASE_POINT (see $dir/ch.log and $con)"; }
            "$REM" --api-socket "$sock" pause >/dev/null || { stop_vmm $chpid; die "templates: $name: pause failed"; }
            formats=""
            for fmt in $FORMATS; do
                case $fmt in
                    lz4)  codec=lz4; cls=cpu ;;
                    zstd) codec=zstd; cls=cpu ;;
                    qpl)  codec=qpl-hardware-static-async; cls=dsa ;;
                    *)    stop_vmm $chpid; die "templates: unknown format $fmt" ;;
                esac
                osock=$dir/snap.sock; rm -f "$osock"
                "$OD" snapshot --socket "$osock" --output-dir "$dir/$fmt" --compression "$codec" --chunk-size "$CHUNK" \
                    --workers "$WORKERS" --classify "$cls" > "$dir/daemon_$fmt.log" 2>&1 &
                dpid=$!
                for _ in $(seq 200); do [ -S "$osock" ] && break; sleep 0.05; done
                t0=$(date +%s%N)
                "$REM" --api-socket "$sock" send-migration "destination_url=unix:$osock,memory_mode=memfds,preserve_source=on" > "$dir/remote_$fmt.log" 2>&1 || true
                rc=0; wait $dpid || rc=$?
                t1=$(date +%s%N); ms=$(( (t1 - t0) / 1000000 ))
                [ $rc = 0 ] || { stop_vmm $chpid; die "templates: $name/$fmt: snapshot rc=$rc (see $dir/daemon_$fmt.log)"; }
                log "templates: $name/$fmt: $ms ms, $(du -sm "$dir/$fmt" | cut -f1) MiB"
                formats="$formats\"$fmt\": {\"dir\": \"$dir/$fmt\", \"codec\": \"$codec\", \"classify\": \"$cls\", \"snapshot_ms\": $ms, \"mib\": $(du -sm "$dir/$fmt" | cut -f1)}, "
            done
            stop_vmm $chpid
            rm -f "$sock" "$osock"
            cat > "$dir/manifest.json" <<JSON
{"session": "$s", "regime": "$reg", "name": "$name", "turns": [$lo, $hi],
 "disk": "$disk", "console": "$con", "mem": "$GUEST_MEM", "vcpus": $GUEST_VCPUS,
 "cmdline": "$cmdline",
 "formats": {${formats%, }}}
JSON
        done
    done
}

# ---------------------------------------------------------------- check
stage_check() {
    local name=${1:-} fmt=${2:-lz4} dir tdisk tcon w sdisk sock rdata chpid rpid t0 t1 rc
    [ -n "$name" ] || name=$(ls "$TMPL" | grep -m1 'x0.1$' || ls "$TMPL" | head -1)
    dir=$TMPL/$name; [ -f "$dir/manifest.json" ] || die "check: no template $name under $TMPL"
    [ -d "$dir/$fmt" ] || die "check: template $name has no $fmt format"
    tdisk=$(manifest_field "$dir/manifest.json" disk); tcon=$(manifest_field "$dir/manifest.json" console)
    [ "$(hp_free)" -ge "$(hp_needed)" ] || die "check: node $NODE has $(hp_free) free 2 MiB pages, the guest needs $(hp_needed)"
    w=$WORK/check_$name.$fmt; rm -rf "$w"; mkdir -p "$w" "$VMD_DISKS/sandboxes"
    # the sandbox's disk is a reflink of the template's, so it must stay on the XFS volume
    sdisk=$VMD_DISKS/sandboxes/check_$name.$fmt.ext4; rm -f "$sdisk"
    cp --reflink=always "$tdisk" "$sdisk"; : > "$w/console.log"
    sock=$w/api.sock; rdata=$w/rdata.sock; rm -f "$sock" "$rdata"
    # The snapshot carries the template's disk and console paths, so the VMM runs
    # in its own user+mount namespace with this sandbox's copies bound over them.
    unshare -Urm --propagation private sh -c "mount --bind '$sdisk' '$tdisk' && mount --bind '$w/console.log' '$tcon' && exec numactl -N$NODE -m$NODE '$CH' --api-socket '$sock'" > "$w/ch.log" 2>&1 &
    chpid=$!
    for _ in $(seq 200); do [ -S "$sock" ] && break; sleep 0.05; done
    [ -S "$sock" ] || { stop_vmm $chpid; die "check: VMM api socket never appeared (see $w/ch.log)"; }
    "$REM" --api-socket "$sock" receive-migration "receiver_url=unix:$rdata" > "$w/recv.log" 2>&1 &
    rpid=$!
    for _ in $(seq 200); do [ -S "$rdata" ] && break; sleep 0.05; done
    [ -S "$rdata" ] || { stop_vmm $chpid; die "check: receive-migration socket never appeared (see $w/recv.log)"; }
    t0=$(date +%s%N)
    rc=0
    numactl -N"$NODE" -m"$NODE" "$OD" restore --socket "$rdata" --input-dir "$dir/$fmt" --resume --hugetlb \
        --workers "$WORKERS" --dsa-depth 32 ${RESTORE_ARGS:-} > "$w/restore.log" 2>&1 || rc=$?
    t1=$(date +%s%N)
    wait $rpid || true
    [ $rc = 0 ] || { stop_vmm $chpid; die "check: restore rc=$rc (see $w/restore.log)"; }
    if wait_marker "$w/console.log" "AGENT_STEP_DONE 1 " 120; then
        log "check: $name/$fmt: restored in $(( (t1 - t0) / 1000000 )) ms; $(grep -a 'AGENT_STEP_DONE 1 ' "$w/console.log" | head -1)"
    else
        stop_vmm $chpid; die "check: $name/$fmt: the guest did not complete turn 1 (see $w/console.log)"
    fi
    stop_vmm $chpid
    rm -f "$sock" "$rdata" "$sdisk"
}

stage=${1:-all}; [ $# -gt 0 ] && shift
case $stage in
    rootfs|templates) "stage_$stage" "$@" ;;
    check) stage_check "$@" ;;
    all) stage_rootfs "$@"; stage_templates "$@"
         for f in $FORMATS; do stage_check "" "$f"; done ;;
    *) die "usage: images.sh [rootfs|templates|check|all] [session ...]" ;;
esac
log "IMAGES_DONE $stage"
