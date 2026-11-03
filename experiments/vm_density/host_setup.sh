#!/usr/bin/env bash
# Host setup for the microVM density experiment (experiments/vm_density).
#
# Builds Cloud Hypervisor's dsa-integration branch from GitHub at a pinned
# revision (the VMM, ch-remote and the offload daemon with its QPL and DTO
# features) together with Intel QPL and DTO at pinned tags, stages a guest
# kernel, configures the DSA and IAA work queues of ONE socket (kernel dmaengine
# queues for the uffd_dma and dcbm providers plus one shared user queue per DSA
# device, one shared user queue per IAA device; the other socket's devices are
# disabled so every accelerator the cell can reach is local), loads the
# providers, mounts a checkpoint store bound to that socket's memory node, and
# writes host.env for the runner.
#
#   host_setup.sh [stage ...]
#   stages: fetch qpl build guest accel providers store verify   (default: all, in that order)
#
# Environment:
#   VMD_HOME        workspace (default $AGENTSYSPERF_HOME/vm_density or ~/.agentsysperf/vm_density)
#   VMD_SOCKET      socket = NUMA node whose CPUs, memory and accelerators the cells use (0)
#   CH_REPO CH_BRANCH CH_REV   cloud-hypervisor source (bravindr/cloud-hypervisor dsa-integration, pinned sha)
#   QPL_REPO QPL_TAG DTO_REPO DTO_TAG   intel/qpl v1.9.0, intel/DTO v1.2
#   KERNEL_WQS      kernel dmaengine work queues per DSA device (3); the user queue is wq<dev>.<KERNEL_WQS>
#   STORE_GIB       size of the tmpfs checkpoint store (40), VMD_STORE its mount point
#   GUEST_KERNEL GUEST_INITRD   a bzImage + initrd with virtio-blk (default: the host's /boot pair if readable)
#   CARGO           cargo binary (default: cargo in PATH, else ~/.cargo/bin/cargo)
# The accel, providers and store stages use sudo.
set -euo pipefail

VMD_HOME=${VMD_HOME:-${AGENTSYSPERF_HOME:-$HOME/.agentsysperf}/vm_density}
VMD_SOCKET=${VMD_SOCKET:-0}
VMD_STORE=${VMD_STORE:-/mnt/vmdensity-store}
CH_REPO=${CH_REPO:-https://github.com/bravindr/cloud-hypervisor.git}
CH_BRANCH=${CH_BRANCH:-dsa-integration}
CH_REV=${CH_REV:-23fac59806f9356bb8000c5b4211cbee13a39df6}
QPL_REPO=${QPL_REPO:-https://github.com/intel/qpl.git}
QPL_TAG=${QPL_TAG:-v1.9.0}
DTO_REPO=${DTO_REPO:-https://github.com/intel/DTO.git}
DTO_TAG=${DTO_TAG:-v1.2}
KERNEL_WQS=${KERNEL_WQS:-3}
STORE_GIB=${STORE_GIB:-40}
GUEST_KERNEL=${GUEST_KERNEL:-}
GUEST_INITRD=${GUEST_INITRD:-}
CARGO=${CARGO:-$(command -v cargo 2>/dev/null || echo "$HOME/.cargo/bin/cargo")}

CH_SRC=$VMD_HOME/cloud-hypervisor
QPL_SRC=$VMD_HOME/qpl
DTO_SRC=$VMD_HOME/DTO
BIN=$VMD_HOME/bin
ACCEL_PENDING=0

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing tool: $1"; }

node_cpus() { cat "/sys/devices/system/node/node$VMD_SOCKET/cpulist"; }
# accelerator devices (dsa|iax) on this socket / on the other sockets, sorted by number
socket_devs() { local d; for d in /sys/bus/dsa/devices/$1*; do if [ -e "$d/numa_node" ] && [ "$(cat "$d/numa_node")" = "$VMD_SOCKET" ]; then basename "$d"; fi; done | sort -V; }
other_devs()  { local d; for d in /sys/bus/dsa/devices/$1*; do if [ -e "$d/numa_node" ] && [ "$(cat "$d/numa_node")" != "$VMD_SOCKET" ]; then basename "$d"; fi; done | sort -V; }
wq_attr() { cat "/sys/bus/dsa/devices/$1/$2" 2>/dev/null || true; }
# user work queues of this socket's dsa|iax devices that have a device node (whatever the layout)
socket_user_wqs() {
    local d id w
    for d in $(socket_devs "$1"); do
        id=${d#dsa}; id=${id#iax}
        for w in /sys/bus/dsa/devices/$d/wq$id.*; do
            if [ -e "$w/type" ] && [ "$(cat "$w/type")" = user ] && [ -e "/dev/$1/$(basename "$w")" ]; then basename "$w"; fi
        done
    done | sort -V
}

# ---------------------------------------------------------------- fetch
clone_at() { # dir repo ref [recursive]
    local dir=$1 repo=$2 ref=$3 rec=${4:-}
    if [ ! -d "$dir/.git" ]; then
        log "fetch: cloning $repo"
        git clone -q ${rec:+--recursive} "$repo" "$dir"
    fi
    git -C "$dir" fetch -q origin "$ref" || git -C "$dir" fetch -q origin
    git -C "$dir" checkout -q --detach "$ref"
    [ -n "$rec" ] && git -C "$dir" submodule update -q --init --recursive
    log "fetch: $dir at $(git -C "$dir" describe --tags --always)"
}
stage_fetch() {
    need git
    mkdir -p "$VMD_HOME"
    if [ ! -d "$CH_SRC/.git" ]; then
        log "fetch: cloning $CH_REPO ($CH_BRANCH)"
        git clone -q --branch "$CH_BRANCH" "$CH_REPO" "$CH_SRC"
    fi
    git -C "$CH_SRC" fetch -q origin "$CH_BRANCH"
    git -C "$CH_SRC" merge-base --is-ancestor "$CH_REV" FETCH_HEAD \
        || die "CH_REV $CH_REV is not on $CH_BRANCH of $CH_REPO"
    git -C "$CH_SRC" checkout -q --detach "$CH_REV"
    log "fetch: cloud-hypervisor at $(git -C "$CH_SRC" log -1 --format='%h %s' | cut -c1-80)"
    clone_at "$QPL_SRC" "$QPL_REPO" "$QPL_TAG" recursive
    clone_at "$DTO_SRC" "$DTO_REPO" "$DTO_TAG"
}

# ---------------------------------------------------------------- qpl
qpl_lib() { find "$QPL_SRC/build" -name libqpl.a 2>/dev/null | head -1; }
stage_qpl() {
    need cmake; need nasm
    local b=$QPL_SRC/build gen=()
    if [ -f "$b/.built-$QPL_TAG" ] && [ -n "$(qpl_lib)" ]; then log "qpl: already built ($QPL_TAG)"; return; fi
    command -v ninja >/dev/null 2>&1 && gen=(-G Ninja)
    log "qpl: building $QPL_TAG"
    cmake -S "$QPL_SRC" -B "$b" "${gen[@]}" -DCMAKE_BUILD_TYPE=Release \
        -DQPL_BUILD_TESTS=OFF -DQPL_BUILD_EXAMPLES=OFF -DQPL_LIBRARY_TYPE=STATIC > "$b.cmake.log" 2>&1 \
        || die "qpl: cmake configure failed, see $b.cmake.log"
    cmake --build "$b" --target qpl --parallel > "$b.build.log" 2>&1 || die "qpl: build failed, see $b.build.log"
    [ -n "$(qpl_lib)" ] || die "qpl: libqpl.a not produced"
    touch "$b/.built-$QPL_TAG"
    log "qpl: $(qpl_lib)"
}

# ---------------------------------------------------------------- build
stage_build() {
    need cmake
    [ -x "$CARGO" ] || die "cargo not found (CARGO=$CARGO); install rustup or set CARGO"
    local lib; lib=$(qpl_lib); [ -n "$lib" ] || die "build: QPL not built (run the qpl stage)"
    export QPL_INCLUDE_DIR=$QPL_SRC/include QPL_LIB_DIR=$(dirname "$lib") DTO_SRC_DIR=$DTO_SRC
    export CARGO_TARGET_DIR=$VMD_HOME/target
    log "build: cargo build (cloud-hypervisor, ch-remote, offload_daemon +qpl +dto) in $CARGO_TARGET_DIR"
    (cd "$CH_SRC" && "$CARGO" build --release -p cloud-hypervisor --bins -p offload_daemon \
        --features offload_daemon/qpl,offload_daemon/dto > "$VMD_HOME/cargo-build.log" 2>&1) \
        || die "build: cargo failed, see $VMD_HOME/cargo-build.log"
    mkdir -p "$BIN"
    install -m 755 "$CARGO_TARGET_DIR"/release/{cloud-hypervisor,ch-remote,offload_daemon} "$BIN/"
    local so; so=$(find "$CARGO_TARGET_DIR/release/build" -path '*offload_daemon-*/out/dto/libdto_explicit.so' | head -1)
    [ -n "$so" ] || die "build: libdto_explicit.so not found under $CARGO_TARGET_DIR"
    install -m 644 "$so" "$BIN/"
    {
        echo "cloud-hypervisor $CH_REPO $CH_BRANCH $(git -C "$CH_SRC" rev-parse HEAD)"
        echo "qpl $QPL_REPO $QPL_TAG $(git -C "$QPL_SRC" rev-parse HEAD)"
        echo "dto $DTO_REPO $DTO_TAG $(git -C "$DTO_SRC" rev-parse HEAD)"
        echo "rustc $("$(dirname "$CARGO")/rustc" --version 2>/dev/null || rustc --version)"
        echo "built $(date -Is) on $(hostname) $(uname -r)"
    } > "$BIN/VERSIONS"
    log "build: $("$BIN/cloud-hypervisor" --version | head -1); daemon $(ls -l "$BIN/offload_daemon" | awk '{print $5}') bytes"
}

# ---------------------------------------------------------------- guest
stage_guest() {
    mkdir -p "$VMD_HOME/guest"
    local k=$GUEST_KERNEL i=$GUEST_INITRD
    if [ -z "$k" ]; then
        k=/boot/vmlinuz-$(uname -r); i=/boot/initrd.img-$(uname -r)
        [ -r "$k" ] && [ -r "$i" ] || die "guest: set GUEST_KERNEL and GUEST_INITRD (a bzImage + initrd with virtio-blk); $k is not readable"
    fi
    [ -r "$k" ] && [ -r "$i" ] || die "guest: cannot read $k / $i"
    install -m 644 "$k" "$VMD_HOME/guest/vmlinuz"
    install -m 644 "$i" "$VMD_HOME/guest/initrd"
    echo "$k $i" > "$VMD_HOME/guest/SOURCE"
    log "guest: kernel $(basename "$k"), initrd $(basename "$i")"
}

# ---------------------------------------------------------------- accel
accel_ok() {
    local d id q
    for d in $(socket_devs dsa); do
        id=${d#dsa}
        [ "$(wq_attr "wq$id.$KERNEL_WQS" name)" = vmdensity-dsa ] && [ -e "/dev/dsa/wq$id.$KERNEL_WQS" ] || return 1
        for q in $(seq 0 $((KERNEL_WQS - 1))); do
            [ "$(wq_attr "wq$id.$q" type)" = kernel ] && [ "$(wq_attr "wq$id.$q" state)" = enabled ] || return 1
        done
    done
    for d in $(socket_devs iax); do
        id=${d#iax}
        [ "$(wq_attr "wq$id.0" name)" = vmdensity-iaa ] && [ -e "/dev/iax/wq$id.0" ] || return 1
    done
    for d in $(other_devs dsa) $(other_devs iax); do
        [ "$(wq_attr "$d" state)" = disabled ] || return 1
    done
    return 0
}
unload_providers() {
    # The WQ layout can only change while no provider holds dmaengine channels:
    # unbinding a claimed queue has hung hosts before. Switch offloading off (an
    # active provider pins its module), unload them (nothing may be using DMA:
    # no VMM, no daemon, no open /dev/uffd_dma) and let the providers stage reload them.
    local m removed=()
    [ -e /sys/module/dcbm/parameters/offloading ] && echo 0 | sudo tee /sys/module/dcbm/parameters/offloading >/dev/null
    [ -e /sys/kernel/uffd_dma/offloading ] && echo 0 | sudo tee /sys/kernel/uffd_dma/offloading >/dev/null
    for m in uffd_dma dcbm mm_offload_dma; do
        lsmod | grep -q "^$m " || continue
        if sudo rmmod "$m" 2>/dev/null; then removed+=("$m"); else
            log "accel: cannot unload $m (refcnt $(cat /sys/module/$m/refcnt 2>/dev/null)); reloading ${removed[*]:-nothing}"
            local r; for r in "${removed[@]}"; do sudo modprobe "$r" 2>/dev/null || true; done
            return 1
        fi
    done
    [ ${#removed[@]} -gt 0 ] && log "accel: unloaded providers ${removed[*]} for the layout change"
    return 0
}
disable_dev() {
    local d=$1 id=${1#dsa}; id=${id#iax}; local q
    for q in $(seq 0 15); do sudo accel-config disable-wq "$d/wq$id.$q" >/dev/null 2>&1 || true; done
    sudo accel-config disable-device "$d" >/dev/null 2>&1 || true
}
config_dsa() {
    local d=$1 id=${1#dsa} total ksz usz q e
    disable_dev "$d"
    total=$(cat "/sys/bus/dsa/devices/$d/max_work_queues_size")
    ksz=$((total / (KERNEL_WQS + 1))); usz=$((total - KERNEL_WQS * ksz))
    for e in /sys/bus/dsa/devices/$d/engine$id.*; do
        sudo accel-config config-engine "$d/$(basename "$e")" --group-id=0 >/dev/null || die "accel: config-engine $d failed"
    done
    for q in $(seq 0 $((KERNEL_WQS - 1))); do
        sudo accel-config config-wq "$d/wq$id.$q" --group-id=0 --mode=dedicated --type=kernel \
            --name=vmdensity-k --driver-name=dmaengine --priority=10 --block-on-fault=0 \
            --wq-size="$ksz" --max-transfer-size=1073741824 --max-batch-size=1024 >/dev/null || die "accel: config-wq $d/wq$id.$q failed"
    done
    sudo accel-config config-wq "$d/wq$id.$KERNEL_WQS" --group-id=0 --mode=shared --type=user \
        --name=vmdensity-dsa --driver-name=user --priority=10 --block-on-fault=1 \
        --threshold=$((usz - 8)) --wq-size="$usz" --max-transfer-size=2147483648 --max-batch-size=1024 >/dev/null || die "accel: config-wq $d/wq$id.$KERNEL_WQS failed"
    sudo accel-config enable-device "$d" >/dev/null 2>&1 || die "accel: enable-device $d failed"
    for q in $(seq 0 "$KERNEL_WQS"); do
        sudo accel-config enable-wq "$d/wq$id.$q" >/dev/null 2>&1 || die "accel: enable-wq $d/wq$id.$q failed"
    done
    log "accel: $d: $KERNEL_WQS kernel WQs x $ksz + user wq$id.$KERNEL_WQS ($usz, shared, bof)"
}
config_iax() {
    local d=$1 id=${1#iax} sz e
    disable_dev "$d"
    sz=$(cat "/sys/bus/dsa/devices/$d/max_work_queues_size")
    for e in /sys/bus/dsa/devices/$d/engine$id.*; do
        sudo accel-config config-engine "$d/$(basename "$e")" --group-id=0 >/dev/null || die "accel: config-engine $d failed"
    done
    sudo accel-config config-wq "$d/wq$id.0" --group-id=0 --mode=shared --type=user \
        --name=vmdensity-iaa --driver-name=user --priority=10 --block-on-fault=1 \
        --threshold=$((sz - 8)) --wq-size="$sz" --max-transfer-size=1073741824 >/dev/null || die "accel: config-wq $d/wq$id.0 failed"
    sudo accel-config enable-device "$d" >/dev/null 2>&1 || die "accel: enable-device $d failed"
    sudo accel-config enable-wq "$d/wq$id.0" >/dev/null 2>&1 || die "accel: enable-wq $d/wq$id.0 failed"
    log "accel: $d: user wq$id.0 ($sz, shared, bof, 1 GiB transfers)"
}
stage_accel() {
    need accel-config
    local dsas iaxs d
    dsas=$(socket_devs dsa); iaxs=$(socket_devs iax)
    [ -n "$dsas" ] || die "accel: no DSA device on socket $VMD_SOCKET"
    [ -n "$iaxs" ] || log "accel: WARNING no IAA device on socket $VMD_SOCKET (iaa arms unavailable)"
    if accel_ok; then
        log "accel: socket $VMD_SOCKET layout already in place ($(echo $dsas) / $(echo $iaxs))"
    else
        if ! unload_providers; then
            ACCEL_PENDING=1
            log "accel: layout change deferred: reboot, run 'host_setup.sh accel providers' before anything loads dcbm or uffd_dma, then continue"
            return 0
        fi
        lsmod | grep -q '^iaa_crypto ' && sudo rmmod iaa_crypto
        for d in $(other_devs dsa) $(other_devs iax); do disable_dev "$d"; done
        for d in $dsas; do config_dsa "$d"; done
        for d in $iaxs; do config_iax "$d"; done
        log "accel: other sockets' devices disabled: $(echo $(other_devs dsa) $(other_devs iax))"
    fi
    sudo chmod 666 /dev/dsa/wq* /dev/iax/wq* 2>/dev/null || true
    log "accel: user WQs $(ls /dev/dsa 2>/dev/null | tr '\n' ' ')/ $(ls /dev/iax 2>/dev/null | tr '\n' ' '); dmaengine channels $(ls /sys/class/dma 2>/dev/null | wc -l)"
}

# ---------------------------------------------------------------- providers
stage_providers() {
    local m
    for m in dcbm uffd_dma; do
        lsmod | grep -q "^$m " || sudo modprobe "$m" 2>/dev/null \
            || log "providers: WARNING $m did not load (no mm_offload kernel: DSA restore arms fall back to the CPU)"
    done
    [ -e /sys/module/dcbm/parameters/offloading ] && echo 0 | sudo tee /sys/module/dcbm/parameters/offloading >/dev/null
    [ -e /sys/kernel/uffd_dma/offloading ] && echo 0 | sudo tee /sys/kernel/uffd_dma/offloading >/dev/null
    [ -e /dev/uffd_dma ] && sudo chmod 666 /dev/uffd_dma
    sudo sysctl -q vm.unprivileged_userfaultfd=1
    echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null
    local g; for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -e "$g" ] && echo performance | sudo tee "$g" >/dev/null
    done
    log "providers: $(lsmod | grep -E '^(dcbm|uffd_dma|mm_offload_dma) ' | awk '{print $1}' | tr '\n' ' ')(offloading off until a dsa arm turns it on); governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
}

# ---------------------------------------------------------------- store
stage_store() {
    sudo mkdir -p "$VMD_STORE"
    if mountpoint -q "$VMD_STORE"; then
        log "store: $VMD_STORE already mounted ($(df -h "$VMD_STORE" | awk 'NR==2{print $2}'))"
    else
        sudo mount -t tmpfs -o "size=${STORE_GIB}G,mpol=bind:$VMD_SOCKET,mode=1777" vmdensity-store "$VMD_STORE"
        log "store: tmpfs ${STORE_GIB} GiB bound to node $VMD_SOCKET at $VMD_STORE"
    fi
}

# ---------------------------------------------------------------- verify + host.env
kernel_offload() { [ -e /dev/uffd_dma ] && lsmod | grep -q '^dcbm ' && echo 1 || echo 0; }
write_env() {
    local dto iax
    dto=$(socket_user_wqs dsa); iax=$(socket_user_wqs iax)
    cat > "$VMD_HOME/host.env" <<ENV
# written by host_setup.sh $(date -Is); sourced by the runner
VMD_HOME='$VMD_HOME'
VMD_SOCKET=$VMD_SOCKET
VMD_CPUS='$(node_cpus)'
VMD_BIN='$BIN'
CH_SRC='$CH_SRC'
GUEST_KERNEL='$VMD_HOME/guest/vmlinuz'
GUEST_INITRD='$VMD_HOME/guest/initrd'
VMD_STORE='$VMD_STORE'
DTO_WQ_LIST='$(echo $dto | tr ' ' ';')'
IAX_WQS='$(echo $iax | tr ' ' ';')'
KERNEL_OFFLOAD=$(kernel_offload)
ACCEL_PENDING=$ACCEL_PENDING
LD_LIBRARY_PATH='$BIN'
ENV
}
stage_verify() {
    write_env
    log "verify: host.env written"
    [ -x "$BIN/cloud-hypervisor" ] && log "verify: $("$BIN/cloud-hypervisor" --version | head -1)" || log "verify: binaries not built"
    [ -x "$BIN/offload_daemon" ] && LD_LIBRARY_PATH=$BIN "$BIN/offload_daemon" serve --help >/dev/null && log "verify: offload_daemon serve ok"
    [ -f "$BIN/VERSIONS" ] && sed 's/^/verify:   /' "$BIN/VERSIONS"
    log "verify: socket $VMD_SOCKET cpus $(node_cpus), free $(awk '/MemFree/ {printf "%d", $4/1048576}' "/sys/devices/system/node/node$VMD_SOCKET/meminfo") GiB, 2 MiB pool $(cat "/sys/devices/system/node/node$VMD_SOCKET/hugepages/hugepages-2048kB/nr_hugepages") pages"
    sed 's/^/verify:   /' "$VMD_HOME/host.env" | grep -E 'DTO_WQ_LIST|IAX_WQS|KERNEL_OFFLOAD|ACCEL_PENDING'
    [ "$ACCEL_PENDING" = 1 ] && log "verify: WARNING WQ layout pending (see accel stage)"
    log "verify: done"
}

stages=("$@"); [ ${#stages[@]} -gt 0 ] || stages=(fetch qpl build guest accel providers store verify)
for s in "${stages[@]}"; do
    case $s in
        fetch|qpl|build|guest|accel|providers|store|verify) log "== $s"; "stage_$s" ;;
        all) for t in fetch qpl build guest accel providers store verify; do log "== $t"; "stage_$t"; done ;;
        *) die "unknown stage $s" ;;
    esac
done
log "HOST_SETUP_DONE"
