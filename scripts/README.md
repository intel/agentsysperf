# scripts/

Utility and operational scripts. These are NOT part of the installed package — they support development, migration, and field testing.

| Script | Purpose |
|--------|---------|
| `field_test_rc1.sh` | Interactive guided field-test workflow (setup → benchmark → dashboard) |
| `field_test_verify.py` | Automated 6-check verification suite for RC validation |
| `migrate_to_unified_store.py` | Migrate scattered per-directory DBs into the canonical store |
| `verify_backfill_parity.py` | Verify migration parity (store >= legacy per run) |
| `push_to_prometheus.py` | Push benchmark metrics to Prometheus Pushgateway |
| `run_tb2_dashboard_demo.py` | One-shot: run TB2 + populate dashboard (demo prep) |
| `list_tb2_tasks.py` | List available Terminal-Bench 2 tasks |
| `test_migrate.py` | Unit tests for the migration script |

## Usage

These scripts are run directly (not via `agentsysperf` CLI):

```bash
# Field testing
bash scripts/field_test_rc1.sh
python scripts/field_test_verify.py

# Migration (from pre-unified scattered DBs)
python scripts/migrate_to_unified_store.py --source-root /path/to/old/runs --apply
python scripts/verify_backfill_parity.py
```
