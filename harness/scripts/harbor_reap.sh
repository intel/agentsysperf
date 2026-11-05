#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Task-scoped lookup and teardown of harbor trial containers. Sourced by the
# harness drivers; defines functions only.
#
# The drivers used to filter on `label=com.docker.compose.project` with no
# value, i.e. EVERY compose container on the host, and force-remove them. On a
# shared host that deletes other users' containers mid-run.
#
# harbor names each trial's compose project
#     <task>__<7-char trial id>__env      (container <project>-main-1)
# where <task> is the task directory's name cut to 32 characters with trailing
# '-'/'_' stripped, then lowercased with anything outside [a-z0-9_-] turned
# into '-' (harbor TrialConfig.generate_trial_name and
# _sanitize_docker_compose_project_name). Matching the whole project name for
# the tasks a driver runs cannot reach another task's or another tool's
# containers. run_density_test_cwf.sh scopes its own teardown the same way.

# <task dir> -> the compose-project prefix harbor gives that task's trials.
harbor_project_prefix() {
    local name
    name="$(basename "${1%/}")"
    name="${name:0:32}"
    while [[ "$name" == *[-_] ]]; do name="${name%?}"; done
    name="${name,,}"
    name="${name//[^a-z0-9_-]/-}"
    [[ "$name" =~ ^[a-z0-9] ]] || name="0$name"
    printf '%s\n' "$name"
}

# [-a] <task dir>... -> ids of those tasks' harbor trial containers, one per
# line. Running containers only, unless -a (as for `docker ps -a`).
harbor_container_ids() {
    local all="" t id project prefixes=()
    if [[ "${1:-}" == "-a" ]]; then all="-a"; shift; fi
    for t in "$@"; do prefixes+=("$(harbor_project_prefix "$t")"); done
    (( ${#prefixes[@]} )) || return 0
    local re
    re="^($(IFS='|'; printf '%s' "${prefixes[*]}"))__[a-z0-9]{7}__env$"
    # $all is unquoted on purpose: empty means no flag. A docker failure
    # yields no ids rather than tripping a caller's `set -e -o pipefail`.
    # shellcheck disable=SC2086
    { docker ps $all --filter "label=com.docker.compose.project" \
            --format '{{.ID}} {{.Label "com.docker.compose.project"}}' 2>/dev/null || true; } |
        while read -r id project; do
            if [[ "$project" =~ $re ]]; then printf '%s\n' "$id"; fi
        done
}

# <task dir>... -> force-remove those tasks' harbor trial containers, if any.
reap_harbor_containers() {
    harbor_container_ids -a "$@" | xargs -r docker rm -f >/dev/null 2>&1 || true
}
