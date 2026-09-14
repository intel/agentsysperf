#!/usr/bin/env bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Launch the AgentSysPerf observability stack WITHOUT the docker compose plugin.
# Same three containers, ports, volumes, and a shared network as
# docker-compose.yml. Use `docker compose up -d` instead if you have the plugin.
#
#   ./monitoring/up.sh         bring the stack up
#   ./monitoring/up.sh down    tear it down
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NET=agentsysperf-mon

down() {
  echo "Stopping AgentSysPerf monitoring stack..."
  docker rm -f agentsysperf-prometheus agentsysperf-pushgateway agentsysperf-grafana >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
  echo "Done."
}

if [[ "${1:-up}" == "down" ]]; then
  down
  exit 0
fi

# Fresh start (idempotent).
down >/dev/null 2>&1 || true
docker network create "$NET" >/dev/null

echo "Starting pushgateway (:9091)..."
docker run -d --name agentsysperf-pushgateway --network "$NET" \
  -p 9091:9091 --restart unless-stopped \
  prom/pushgateway:latest >/dev/null

echo "Starting prometheus (:9090)..."
docker run -d --name agentsysperf-prometheus --network "$NET" \
  -p 9090:9090 --restart unless-stopped \
  -v "$HERE/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.retention.time=30d >/dev/null

echo "Starting grafana (:3000)..."
docker run -d --name agentsysperf-grafana --network "$NET" \
  -p 3000:3000 --restart unless-stopped \
  -e GF_AUTH_ANONYMOUS_ENABLED=true \
  -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
  -e GF_AUTH_DISABLE_LOGIN_FORM=true \
  -v "$HERE/grafana/provisioning:/etc/grafana/provisioning:ro" \
  -v "$HERE/grafana/dashboards:/var/lib/grafana/dashboards:ro" \
  grafana/grafana:latest >/dev/null

echo
echo "Stack up:"
echo "  Prometheus  -> http://localhost:9090"
echo "  Pushgateway -> http://localhost:9091"
echo "  Grafana     -> http://localhost:3000  (AgentSysPerf Benchmark Dashboard)"
