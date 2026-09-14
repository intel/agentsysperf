# AgentSysPerf Observability Stack

Push AgentSysPerf benchmark metrics to **Prometheus** and view them in **Grafana**.

```
benchmark run → measurement_records.json
   → PrometheusExporter (push) → Pushgateway :9091
   → Prometheus :9090 (scrapes gateway)
   → Grafana :3000  "AgentSysPerf Benchmark Dashboard"
```

The exporter writes the Prometheus text exposition format by hand — **no
OpenTelemetry, no prometheus_client dependency**. Observability overhead is
negligible (serialize-after-span + one HTTP POST). Measurement overhead is
unchanged (L1 < 1%, L3 perf-gated, PerfSpect/TMA heaviest).

## Two ways to run

### A. Docker (preferred, needs Docker + registry access)

```bash
poetry run python scripts/push_to_prometheus.py --dashboard-only   # emit dashboard JSON
docker compose -f monitoring/docker-compose.yml up -d              # or: ./monitoring/up.sh
```

`up.sh` is a fallback that launches the same containers via `docker run` when
the compose plugin is unavailable.

### B. Native binaries (no Docker, no root)

For hosts without Docker/registry access (e.g. behind a corporate proxy where
the daemon can't pull images). Downloads three static userspace binaries.

```bash
# one-time: fetch binaries into ~/agentsysperf-monitoring (uses your shell proxy)
mkdir -p ~/agentsysperf-monitoring/dl && cd ~/agentsysperf-monitoring/dl
curl -sSL -o prometheus.tgz  https://github.com/prometheus/prometheus/releases/download/v2.53.2/prometheus-2.53.2.linux-amd64.tar.gz
curl -sSL -o pushgateway.tgz https://github.com/prometheus/pushgateway/releases/download/v1.9.0/pushgateway-1.9.0.linux-amd64.tar.gz
curl -sSL -o grafana.tgz     https://dl.grafana.com/oss/release/grafana-11.2.0.linux-amd64.tar.gz
cd .. && tar xzf dl/prometheus.tgz && tar xzf dl/pushgateway.tgz && tar xzf dl/grafana.tgz

# start / stop (run from the repo root)
./monitoring/run-native.sh          # start
./monitoring/run-native.sh down     # stop
```

Override the binary location with `AGENTSYSPERF_MON_VENDOR=/path ./monitoring/run-native.sh`.

## Push data and view

```bash
# generate a quick L1 dataset (synthetic, no API key) ...
poetry run python run_simple_benchmark.py
# ... then push it + (re)emit the dashboard
poetry run python scripts/push_to_prometheus.py \
  --records /tmp/agentsysperf_results/measurement_records.json --run-id simple_demo
```

Open **http://localhost:3000** → *AgentSysPerf Benchmark Dashboard* → pick `run_id`.

Endpoints: Prometheus `:9090` · Pushgateway `:9091` · Grafana `:3000`
(anonymous admin, no login).

## Which panels light up

Panels reflect the **measurement layers present in the data**:

| Panel group | Needs layer | Synthetic L1 run | Full perf run |
|---|---|---|---|
| Duration, CPU%, Memory/RSS | L1 | ✅ | ✅ |
| IPC, Cache Miss % | L3 (`perf`) | — | ✅ |
| TMA L1 Breakdown | PerfSpect | — | ✅ |

Empty IPC/cache/TMA panels on an L1-only run are expected, not a bug — those
metrics simply weren't collected. Run with the `l3_perf` / `perfspect`
measurement plugins (perf_event_paranoid permitting) to populate them.

## Notes

- `scripts/push_to_prometheus.py` writes the dashboard JSON in Grafana's
  **file-provisioning** form (bare dashboard dict). `GrafanaDashboardGenerator.save()`
  writes the **HTTP API** form (`{"dashboard": …}`) — different envelope, used
  when POSTing to `/api/dashboards/db`.
- The native path keeps all runtime state under `~/agentsysperf-monitoring/run/`
  (logs, TSDB, Grafana db). Delete it to reset.
