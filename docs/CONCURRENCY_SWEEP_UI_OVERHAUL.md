# Concurrency Sweep page — UI overhaul plan

From a UI-design + CPU-architect review workflow (2026-06-12). Fixes three pain
points: weak comparison, thin detail, buried bottleneck/recommendations — plus
synthetic-sweep labeling and per-task clarity.

**Decisions (user):** explicit `View: Single | Compare` toggle · persist
`data_source` ('synthetic'|'measured') in sweep metadata · saturation panel keeps
2 plots by default with extra proof-signals behind an expander · implement
**P1–P5 now** (P6 calibration-honesty, P7 variance-bands deferred).

**Invariants:** keep `scaling_views.py` figure builders pure (dict rows in,
go.Figure out — no Streamlit/DB), extend existing builders only via
optional/defaulted args (don't break current callers/tests), Streamlit+Plotly
only (no new deps).

## P1 — Bottleneck/recs prominence + explicit empty-states  (low effort)
- demo_app: colored callout band (BOTTLENECK_STYLE color) under the verdict,
  bottleneck label + plain-language meaning; recommendations promoted OUT of the
  bottom expander to an always-visible block right under it.
- demo_app: at-knee evidence strip — st.metric row: CPU peak@knee, CPU avg@knee,
  runqueue@knee, agents/physical-core@knee.
- demo_app: replace the silent `if _cs_per_task:` skip with an explicit empty-state
  ("synthetic sweeps have no per-task breakdown"); no-knee fallback text when
  `_cs_knee` is None (verdict no_knee_within_swept_range).
- demo_app: synthetic badge (uses P? data_source; until then heuristic).

## P2 — Genuine paired Compare layout + delta table  (low effort, no new queries)
- demo_app: top-level `st.radio('View', ['Single sweep','Compare two sweeps'])`.
- Use the secondary verdict ALREADY queried (`_cmp_v`, currently discarded):
  mirrored `st.columns(2)` card-stacks (A | B) of verdict/bottleneck/knee, both
  sweeps' recommendations side by side.
- Delta table (st.dataframe): rows = knee_density, knee_concurrency,
  throughput@knee, efficiency@knee, p95@knee, bottleneck, confidence — A, B, Δ.
- One-line auto diff sentence: "Knee moved ±X% (A→B agents/core); bottleneck
  A→B."

## P3 — Throughput-per-core efficiency + agents/physical-core  (med)
- scaling.py: add `efficiency_at_knee` (= knee throughput / vcpu_basis) and
  `p95_corroborated` to evidence.knee.
- scaling_views: NEW `efficiency_figure(points, vcpu_basis, knee=, bottleneck=)`
  — throughput-per-core vs density, knee marked. Dynamic density-axis title
  across builders.
- demo_app: render efficiency_figure beneath the headline throughput plot.

## P4 — Saturation-signature proof panel  (med; extras behind expander)
- scaling_views: NEW `saturation_signature_figure(points, logical_cpus,
  bottleneck=, knee=)` via plotly make_subplots — ctx_sw_per_s, iowait_pct_avg,
  mem_avail_mb_min + runqueue reference lines at logical_cpus and 1.5×logical_cpus.
- demo_app: keep the default 2-plot headline (throughput + cpu_runqueue); put the
  signature panel behind a "secondary saturation signals" expander.
- Import threshold constants (RUNQUEUE_OVERSUB_RATIO, etc.) from analyzers.scaling
  for the reference lines (one-way scaling_views→scaling import).

## P5 — Compare-capable overlays  (med)
- scaling_views: NEW `cpu_runqueue_compare_figure(sweeps)` and
  `efficiency_compare_figure(sweeps, vcpu_bases=)` — same {label, points, knee,
  logical_cpus} list shape as throughput_compare_figure. Fix the knee-vline
  legend gap (label the vlines per sweep).
- demo_app: in Compare mode render the CPU/runqueue + efficiency overlays; when
  both sweeps have per_task, show the SAME selected task side by side (two
  per_task_profile_figure in columns).

## Deferred
- P6 — calibration honesty (threshold disclosure, confidence basis, p95 band).
- P7 — replicate-variance bands (opt-in) + elapsed/completed_trials surfacing.

## New figure builders (all pure)
- `efficiency_figure(points, vcpu_basis, knee=None, bottleneck=None) -> Figure`
- `saturation_signature_figure(points, logical_cpus, bottleneck=None, knee=None) -> Figure`
- `cpu_runqueue_compare_figure(sweeps) -> Figure`
- `efficiency_compare_figure(sweeps, vcpu_bases=None) -> Figure`
- existing builders gain optional args only (back-compatible).

## Provenance
- store_sweep_metadata: persist `data_source` ('synthetic' for dry-run, else
  'measured'); harbor_sweep passes it from the dry_run bool. Migration: existing
  rows default to a heuristic (per_task empty + iowait==0 + replay_fixture None)
  at read time; new rows authoritative.
