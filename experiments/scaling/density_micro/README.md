This directory preserves the original density experiment workloads that used
generic micro-benchmarks (array traversal, simple shell commands) rather than
TB2-representative CPU signatures.

Files:
- workloads.py: COMPUTE_HEAVY / IO_HEAVY / BALANCED profiles (array stride walks)
- config.py: PhaseMix enum and experiment configuration (snapshot)
- agent_worker.py: Worker that ran the generic workloads (snapshot)

The parent directory now uses workloads derived from harness/scripts/synthetic_tasks.py
which faithfully reproduce the CPU signatures of real TB2 tasks (compile_like,
ml_train_like, raytrace_like, etc.).
