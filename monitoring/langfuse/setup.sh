#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Start Langfuse for AgentSysPerf and create initial API keys.
#
# Usage:
#   cd /path/to/agentsysperf
#   bash monitoring/langfuse/setup.sh
#
# After running, export the printed env vars to enable tracing:
#   export AGENTSYSPERF_LANGFUSE=1
#   export LANGFUSE_HOST=http://localhost:7862
#   export LANGFUSE_PUBLIC_KEY=pk-lf-...
#   export LANGFUSE_SECRET_KEY=sk-lf-...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# docker-compose publishes 7862 on this host, so localhost is correct for a
# local run. Set $LANGFUSE_HOST when Langfuse serves a different address --
# same variable the tracing client reads, so one export covers both.
LANGFUSE_URL="${LANGFUSE_HOST:-http://localhost:7862}"

echo "=== Starting Langfuse on ${LANGFUSE_URL} ==="
docker compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d

echo ""
echo "Waiting for Langfuse to be ready..."
for i in $(seq 1 30); do
    if curl -sf "${LANGFUSE_URL}/api/public/health" >/dev/null 2>&1; then
        echo "Langfuse is ready!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "WARNING: Langfuse did not become healthy in 30s. Check logs:"
        echo "  docker compose -f ${SCRIPT_DIR}/docker-compose.yml logs langfuse-web"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Langfuse is running ==="
echo ""
echo "  UI:  ${LANGFUSE_URL}"
echo ""
echo "To connect AgentSysPerf, create a project + API keys in the Langfuse UI, then:"
echo ""
echo "  export AGENTSYSPERF_LANGFUSE=1"
echo "  export LANGFUSE_HOST=${LANGFUSE_URL}"
echo "  export LANGFUSE_PUBLIC_KEY=pk-lf-<from-langfuse-ui>"
echo "  export LANGFUSE_SECRET_KEY=sk-lf-<from-langfuse-ui>"
echo ""
echo "First-time setup: open ${LANGFUSE_URL} → create account → Settings → API Keys"
