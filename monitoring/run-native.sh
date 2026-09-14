#!/usr/bin/env bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# Run the AgentSysPerf observability stack WITHOUT Docker -- three static
# userspace binaries (no root, no daemon). For hosts where Docker/registry
# access is unavailable. Binaries live under $VENDOR (downloaded separately).
#
#   ./monitoring/run-native.sh         start prometheus + pushgateway + grafana
#   ./monitoring/run-native.sh down     stop them
#
# Ports: prometheus :9090  pushgateway :9091  grafana :3000
set -uo pipefail

VENDOR="${AGENTSYSPERF_MON_VENDOR:-$HOME/agentsysperf-monitoring}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$VENDOR/run"          # runtime: pids, data, native configs
PROM_DIR="$VENDOR/prometheus-2.53.2.linux-amd64"
PGW_DIR="$VENDOR/pushgateway-1.9.0.linux-amd64"
GRAF_DIR="$VENDOR/grafana-v11.2.0"

down() {
  echo "Stopping native monitoring stack..."
  for svc in grafana prometheus pushgateway; do
    if [[ -f "$RUN/$svc.pid" ]]; then
      kill "$(cat "$RUN/$svc.pid")" 2>/dev/null || true
      rm -f "$RUN/$svc.pid"
    fi
  done
  echo "Done."
}

if [[ "${1:-up}" == "down" ]]; then down; exit 0; fi

down >/dev/null 2>&1 || true
mkdir -p "$RUN"/{prom-data,graf-data,graf-logs,graf-plugins,provisioning/datasources,provisioning/dashboards,dashboards}

# --- native Prometheus config: scrape pushgateway on localhost ---
cat > "$RUN/prometheus.yml" <<'EOF'
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: pushgateway
    honor_labels: true
    static_configs:
      - targets: ['localhost:9091']
  - job_name: prometheus
    static_configs:
      - targets: ['localhost:9090']
EOF

# --- native Grafana datasource: localhost url, name "Prometheus" ---
cat > "$RUN/provisioning/datasources/prometheus.yml" <<'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: true
EOF

# --- native Grafana dashboard provider -> load JSON from run/dashboards ---
cat > "$RUN/provisioning/dashboards/agentsysperf.yml" <<EOF
apiVersion: 1
providers:
  - name: AgentSysPerf
    orgId: 1
    folder: ''
    type: file
    allowUiUpdates: true
    options:
      path: $RUN/dashboards
EOF

# copy the generated dashboard (raw provisioning form) into place
cp "$HERE/grafana/dashboards/agentsysperf_dashboard.json" "$RUN/dashboards/" 2>/dev/null \
  || echo "WARN: dashboard JSON missing -- run scripts/push_to_prometheus.py --dashboard-only"

echo "Starting pushgateway (:9091)..."
nohup "$PGW_DIR/pushgateway" --web.listen-address=":9091" \
  > "$RUN/pushgateway.log" 2>&1 &
echo $! > "$RUN/pushgateway.pid"

echo "Starting prometheus (:9090)..."
nohup "$PROM_DIR/prometheus" \
  --config.file="$RUN/prometheus.yml" \
  --storage.tsdb.path="$RUN/prom-data" \
  --web.listen-address=":9090" \
  > "$RUN/prometheus.log" 2>&1 &
echo $! > "$RUN/prometheus.pid"

echo "Starting grafana (:3000)..."
# Grafana reads paths via env; no root, anonymous admin for zero-login viewing.
GF_PATHS_DATA="$RUN/graf-data" \
GF_PATHS_LOGS="$RUN/graf-logs" \
GF_PATHS_PLUGINS="$RUN/graf-plugins" \
GF_PATHS_PROVISIONING="$RUN/provisioning" \
GF_SERVER_HTTP_PORT=3000 \
GF_AUTH_ANONYMOUS_ENABLED=true \
GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
GF_AUTH_DISABLE_LOGIN_FORM=true \
GF_ANALYTICS_REPORTING_ENABLED=false \
GF_ANALYTICS_CHECK_FOR_UPDATES=false \
  nohup "$GRAF_DIR/bin/grafana" server --homepath "$GRAF_DIR" \
  > "$RUN/grafana.log" 2>&1 &
echo $! > "$RUN/grafana.pid"

sleep 1
echo
echo "Stack starting (give Grafana ~10s):"
echo "  Prometheus  -> http://localhost:9090"
echo "  Pushgateway -> http://localhost:9091"
echo "  Grafana     -> http://localhost:3000  (AgentSysPerf Benchmark Dashboard)"
echo "Logs: $RUN/*.log"
