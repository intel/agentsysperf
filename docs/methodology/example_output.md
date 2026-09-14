# AgentSysPerf — Example Output (Mockup)

What a comparison run looks like after `agentsysperf compare`.

---

## Single-variant dashboard view

```
                            AgentSysPerf | legal_review | variant=node_local_design
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                              Metric ┃     avg  ┃     min  ┃     max  ┃     p99  ┃     p90  ┃     p50  ┃     std  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ TIME — TASK LEVEL                   │          │          │          │          │          │          │          │
│ task_latency_e2e_ms                 │   1,247  │     680  │   4,920  │   3,810  │   2,180  │   1,070  │     724  │
│ first_step_ttft_ms                  │      45  │      12  │     410  │     280  │      95  │      32  │      52  │
│                                     │          │          │          │          │          │          │          │
│ TIME — STEP LEVEL                   │          │          │          │          │          │          │          │
│ step_latency_ms{classify}           │       3  │       1  │      18  │      14  │       6  │       3  │       2  │
│ step_latency_ms{summarize}          │     680  │     320  │   2,100  │   1,800  │     980  │     590  │     290  │
│ step_latency_ms{rag_retrieve}       │      62  │      35  │     245  │     180  │      88  │      54  │      35  │
│ step_latency_ms{rerank}             │      18  │       8  │      85  │      67  │      28  │      15  │      12  │
│ step_latency_ms{propose_revisions}  │     420  │     180  │   2,300  │   1,650  │     720  │     310  │     290  │
│ step_latency_ms{format}             │       4  │       1  │      22  │      17  │       7  │       3  │       3  │
│                                     │          │          │          │          │          │          │          │
│ TIME — ATTRIBUTION                  │          │          │          │          │          │          │          │
│ framework_overhead_ms               │      28  │       8  │     120  │      94  │      47  │      21  │      18  │
│ gateway_latency_ms                  │       3  │       1  │      18  │      14  │       6  │       2  │       2  │
│ routing_decision_ms                 │       7  │       2  │      45  │      32  │      14  │       5  │       6  │
│ backend_latency_ms                  │   1,150  │     620  │   4,700  │   3,580  │   2,020  │     980  │     680  │
│ tool_latency_ms                     │      59  │      32  │     230  │     170  │      82  │      51  │      33  │
│                                     │          │          │          │          │          │          │          │
│ QUALITY                             │          │          │          │          │          │          │          │
│ task_success_rate                   │   0.940  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │
│ intent_classification_accuracy      │   0.985  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │
│ tool_call_precision                 │   0.920  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │
│ routing_accuracy                    │   0.910  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │
│                                     │          │          │          │          │          │          │          │
│ RESOURCES & COST                    │          │          │          │          │          │          │          │
│ tokens_in_per_task                  │   3,200  │   1,100  │   9,800  │   8,200  │   5,100  │   2,800  │   1,540  │
│ tokens_out_per_task                 │     680  │     250  │   1,950  │   1,640  │   1,020  │     580  │     320  │
│ cost_per_task_usd                   │  0.0142  │  0.0048  │  0.0420  │  0.0350  │  0.0218  │  0.0124  │  0.0066  │
│ cpu_core_seconds_per_task           │    8.40  │    2.10  │   32.10  │   24.50  │   13.80  │    6.90  │    4.20  │
│                                     │          │          │          │          │          │          │          │
│ ROUTING                             │          │          │          │          │          │          │          │
│ wss_classification_accuracy         │   0.940  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │
│ kv_cache_hit_rate                   │   0.380  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │
│ cross_pool_handoffs_per_task        │    2.40  │       0  │       6  │       5  │       4  │       2  │    1.10  │
│ cold_route_rate                     │   0.180  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │
│ fallback_rate                       │   0.020  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │     N/A  │
└─────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

Benchmark Duration: 387.4 sec | Total Tasks: 200 (3 runs of 200 + 20 warmup) | Failures: 12 (6.0%)
CSV Export: ./agentsysperf_results/node_local_design/profile.csv
JSON Export: ./agentsysperf_results/node_local_design/profile.json
OTEL Traces: http://<jaeger-host>/search?service=agentsysperf
```

---

## Comparison report (the killer feature)

```
                                        AgentSysPerf — Comparison Report
                                        Workload: legal_review | 200 tasks per variant × 3 runs

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                   Metric ┃   baseline_centralized ┃     node_local_design  ┃        gpu_baseline    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ task_latency_e2e_ms.p99                  │   4,820 ± 180          │   3,810 ± 145   ★      │     980 ± 60    ★★     │
│ task_latency_e2e_ms.p50                  │   1,420 ± 35           │   1,070 ± 28    ★      │     340 ± 15    ★★     │
│ task_success_rate                        │   0.945 ± 0.012        │   0.940 ± 0.014        │   0.951 ± 0.011        │
│ cost_per_task_usd                        │   0.0118 ± 0.0008      │   0.0142 ± 0.0010      │   0.0710 ± 0.0042  ↓   │
│ routing_accuracy                         │   0.895 ± 0.018        │   0.910 ± 0.015        │   N/A                  │
│ kv_cache_hit_rate                        │   0.420 ± 0.025        │   0.380 ± 0.022        │   0.650 ± 0.030 ★      │
│ gateway_latency_ms.p99                   │   45 ± 5               │   14 ± 3        ★      │   42 ± 4               │
│ cross_pool_handoffs_per_task             │   2.8 ± 0.3            │   2.4 ± 0.2            │   N/A                  │
└──────────────────────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┘

Legend: ★ = best (p<0.05)  ★★ = best by >2× margin  ↓ = worst on this metric (cost)

Statistical Notes:
  - Pairwise t-tests with Bonferroni correction
  - 95% confidence intervals shown (± half-width)
  - Each cell is mean across 3 runs of 200 tasks (n=600)

Bottleneck Attribution (variant: baseline_centralized):
  ┌─────────────────────────────┬─────────────┐
  │ Component                   │ % of p99    │
  ├─────────────────────────────┼─────────────┤
  │ backend (vLLM inference)    │      78%    │
  │ tool calls (RAG)            │      11%    │
  │ gateway (Envoy)             │       6%    │
  │ framework (LangGraph)       │       3%    │
  │ routing decision            │       2%    │
  └─────────────────────────────┴─────────────┘

Recommendation: node_local_design wins on p99 latency (21% improvement) and routing accuracy.
GPU baseline is 3.9× faster but 5x more expensive per task. For latency-critical
workloads (p99 < 1.5s), gpu_baseline is justified. For cost-sensitive workloads,
node_local_design beats baseline_centralized.

Full HTML report: ./agentsysperf_results/comparison_centralized_vs_node_local.html
```

---

## What this output enables

This format makes architecture decisions **defensible with data**. Instead of:

> "We think the node-local design is better."

teams can say:

> "On 200 legal-review tasks (3 runs each, p<0.05 significance), node-local
> design reduced p99 latency by 21% and improved routing accuracy by 1.5pp,
> while costing 20% more per task. We recommend it for workloads where p99
> matters more than cost; otherwise stick with centralized."

That's the level of rigor AIPerf brought to inference servers, applied to the
agentic stack.
