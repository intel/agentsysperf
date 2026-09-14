#!/bin/bash
#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

# AgentSysPerf 0.1.0-RC1 Field Testing Script
# Runs on CWF (Xeon) hardware to validate RC1 workflows

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================================"
echo "  AgentSysPerf 0.1.0-RC1 Field Testing on CWF (Xeon)"
echo "================================================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: SETUP
# ─────────────────────────────────────────────────────────────────────────────

echo "[Phase 1] Environment Setup"
echo "───────────────────────────────────────────────────────────────────────────"

if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists"
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3.12 -m venv .venv
    source .venv/bin/activate
fi

echo "Installing agentsysperf..."
pip install -q -e .

echo "Verifying installation..."
agentsysperf list > /dev/null && echo "✓ agentsysperf CLI ready" || echo "✗ Installation failed"

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: POPULATE WITH SYNTHETIC CPU (Offline)
# ─────────────────────────────────────────────────────────────────────────────

echo "[Phase 2] Populate Demo App with Synthetic CPU Baseline"
echo "───────────────────────────────────────────────────────────────────────────"

read -p "Run synthetic_cpu benchmark (9 tasks, ~3m)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running: agentsysperf run --benchmark synthetic_cpu --num-tasks 9"
    agentsysperf run --benchmark synthetic_cpu --num-tasks 9

    echo ""
    echo "✓ Synthetic CPU complete. Running database check..."
    agentsysperf db ls | head -5
else
    echo "Skipped synthetic_cpu"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: POPULATE WITH SCALING SWEEP (Offline, no API key)
# ─────────────────────────────────────────────────────────────────────────────

echo "[Phase 3] Populate Demo App with Concurrency Sweep (Offline)"
echo "───────────────────────────────────────────────────────────────────────────"

read -p "Run scaling sweep (6 density points, ~2m, no API key needed)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running: agentsysperf sweep run --dry-run"
    agentsysperf sweep run --dry-run

    echo ""
    echo "✓ Scaling sweep complete."
else
    echo "Skipped scaling sweep"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: POPULATE WITH TERMINAL-BENCH (Requires LLM)
# ─────────────────────────────────────────────────────────────────────────────

echo "[Phase 4] Populate Demo App with Terminal-Bench (Agent Performance)"
echo "───────────────────────────────────────────────────────────────────────────"

read -p "Run Terminal-Bench benchmark (requires LLM)? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Terminal-Bench requires an LLM backend. Choose one:"
    echo "  1) OpenAI (gpt-4o-mini) — via OPENAI_API_KEY"
    echo "  2) Local vLLM server — http://localhost:8000"
    echo "  3) Replay from fixture (free, deterministic)"
    echo "  4) Skip Terminal-Bench"
    echo ""
    read -p "Choice (1-4): " choice

    case $choice in
        1)
            echo ""
            read -p "Enter your OPENAI_API_KEY (or press Enter to skip): " -s api_key
            echo ""

            if [ -n "$api_key" ]; then
                export OPENAI_API_KEY="$api_key"
                echo "Testing API key..."

                read -p "Number of tasks (default 2): " num_tasks
                num_tasks=${num_tasks:-2}

                read -p "Model (gpt-4o-mini|gpt-4o|...): " model
                model=${model:-gpt-4o-mini}

                echo "Running: agentsysperf run -b terminal-bench --num-tasks $num_tasks --model $model"
                agentsysperf run -b terminal-bench --num-tasks "$num_tasks" --model "$model"
                echo "✓ Terminal-Bench with OpenAI complete"
            else
                echo "Skipped (no API key provided)"
            fi
            ;;

        2)
            echo ""
            echo "Checking for local vLLM server at http://localhost:8000..."
            if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
                echo "✓ vLLM server detected"

                read -p "Number of tasks (default 2): " num_tasks
                num_tasks=${num_tasks:-2}

                echo "Running: agentsysperf run -b terminal-bench --num-tasks $num_tasks --model local-vllm"
                agentsysperf run -b terminal-bench --num-tasks "$num_tasks" --model "http://localhost:8000/v1" 2>&1 | head -50
                echo "✓ Terminal-Bench with local vLLM complete"
            else
                echo "✗ vLLM server not found at http://localhost:8000"
                echo "  Start with: vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --port 8000"
            fi
            ;;

        3)
            echo ""
            echo "Checking for replay fixture..."
            if [ -f "/path/to/your/fixture.jsonl" ]; then
                echo "✓ Fixture found: /path/to/your/fixture.jsonl"

                read -p "Number of tasks (default 2): " num_tasks
                num_tasks=${num_tasks:-2}

                echo "Running: agentsysperf run -b terminal-bench --num-tasks $num_tasks --replay <fixture>"
                agentsysperf run -b terminal-bench --num-tasks "$num_tasks" \
                  --replay /path/to/your/fixture.jsonl
                echo "✓ Terminal-Bench with replay complete"
            else
                echo "✗ Fixture not found"
                echo "  Generate one with: agentsysperf run -b terminal-bench -n 2 --record /tmp/fixture.jsonl"
                echo "  (requires OPENAI_API_KEY + Docker/Harbor)"
            fi
            ;;

        4)
            echo "Skipped Terminal-Bench"
            ;;

        *)
            echo "Invalid choice"
            ;;
    esac
else
    echo "Skipped Terminal-Bench"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: LAUNCH DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

echo "[Phase 5] Launch Demo App Dashboard"
echo "───────────────────────────────────────────────────────────────────────────"

echo ""
echo "To view the dashboard:"
echo ""
echo "  Option A (Local browser):"
echo "    streamlit run demo_app.py --server.port 7860 --server.address 0.0.0.0"
echo ""
echo "  Option B (SSH tunnel from your laptop):"
echo "    ssh -fN -L 7860:localhost:7860 <user>@$(hostname -f)"
echo "    Then browse: http://localhost:7860"
echo ""
echo "  Option C (Direct on lab network, if ACL allows):"
echo "    http://$(hostname -f):7860"
echo ""

read -p "Start dashboard now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting Streamlit on port 7860..."
    streamlit run demo_app.py --server.port 7860 --server.address 0.0.0.0
else
    echo "Skipped dashboard"
fi

echo ""

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

echo "[Summary] Field Testing Complete"
echo "───────────────────────────────────────────────────────────────────────────"

echo ""
echo "Database summary:"
agentsysperf db ls | head -10

echo ""
echo "Analyzer verdicts:"
agentsysperf db show $(agentsysperf db ls | head -1 | awk '{print $1}') 2>/dev/null | grep -A 20 "analyzers:" || echo "(no analyzers)"

echo ""
echo "================================================================================"
echo "  ✓ Field testing complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review dashboard tabs (Hardware Baselines, Scaling, Agent Performance, etc.)"
echo "  2. Check for any issues or unexpected results"
echo "  3. Report findings to team"
echo "  4. If all good, proceed with team review + merge"
echo ""
