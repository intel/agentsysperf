#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""ReACT agent loop for Terminal-Bench, backed by LiteLLM.

Ported from AgentOptimizer's ``react_terminal_loop.py``. Differences:

- ``litellm.completion()`` replaces AgentFlow's ``ExecutionEngine``
- No ``KVCacheSessionManager`` (LiteLLM exposes no KV-cache hooks for
  hosted endpoints; cache reuse is provider-internal)
- No budget enforcement (Phase B keeps it simple; add later if needed)
- Synchronous: hosted LLM latency dominates; async buys little here
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import litellm

# Drop per-model-unsupported params instead of erroring: model families differ
# on what they accept (e.g. gpt-5 rejects temperature=0.0 and tool_choice; Qwen
# accepts both). drop_params lets one agent loop drive any of them — litellm
# silently omits params a given model can't take.
litellm.drop_params = True

from src.agent_loops.react_parser import (
    _HARMONY_MARKER,
    _strip_harmony_markers,
    parse_model_response,
)
from src.trace.langfuse_setup import completion_metadata, enable_langfuse
from src.benchmarks.terminal_bench.commands import (
    CommandCallRecord,
    CommandMetrics,
    classify_command,
)
from src.benchmarks.terminal_bench.environment import (
    EnvironmentBackend,
    SyncEnvironmentWrapper,
)
from src.replay.fixture import FIXTURE_MISS_MARKER
from src.runner import RunContext, track_span

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    turn: int
    action: str = "__thinking__"
    command: str = ""
    command_tier: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_ms: float = 0.0
    command_ms: float = 0.0
    exit_code: Optional[int] = None
    stop_reason: str = ""
    error: Optional[str] = None
    # Absolute wall-clock timestamps (epoch nanoseconds) for the LLM call and
    # the shell command, so StepTrace can carry real start/end times and the
    # report can render a per-turn timeline. 0 = not recorded.
    llm_start_ns: int = 0
    llm_end_ns: int = 0
    cmd_start_ns: int = 0
    cmd_end_ns: int = 0
    # Pipeline phase this turn's command was tagged as: "act" for ordinary shell
    # work, "retrieve" for semantic / index-backed retrieval. retrieval_signals
    # records WHY (e.g. ["vector_index"]) so the attribution is auditable rather
    # than taken on faith — an unexplainable phase label is not evidence.
    phase: str = ""
    retrieval_signals: List[str] = field(default_factory=list)


@dataclass
class TrialResult:
    task_id: str
    passed: bool = False
    reward: float = 0.0
    submitted: bool = False

    num_turns: int = 0
    num_commands: int = 0
    total_generation_ms: float = 0.0
    total_command_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    wall_clock_ms: float = 0.0

    turns: List[TurnRecord] = field(default_factory=list)
    command_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "reward": self.reward,
            "submitted": self.submitted,
            "num_turns": self.num_turns,
            "num_commands": self.num_commands,
            "total_generation_ms": round(self.total_generation_ms, 1),
            "total_command_ms": round(self.total_command_ms, 1),
            "wall_clock_ms": round(self.wall_clock_ms, 1),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "command_metrics": self.command_metrics,
            "error": self.error,
        }


_SYSTEM_PROMPT = """\
You are an expert system administrator and software engineer.
You have full shell access to a Linux environment.
Your goal is to complete the given task by executing shell commands.

## Available Tools

- shell: Execute a shell command. Usage: Action Input: {"command": "..."}
- submit: Signal that the task is complete. Usage: Action Input: {}

## Response Format

On every turn respond with exactly:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <json object>

Example:
Thought: I need to check what files are in the workspace.
Action: shell
Action Input: {"command": "ls -la /workspace"}

When done:
Thought: The task is complete because ...
Action: submit
Action Input: {}

## Rules

- Always Thought, then Action, then Action Input.
- Action Input MUST be valid JSON on a single line.
- One tool per turn.
- Inspect before modifying.
- Use non-interactive flags (-y, --yes).
"""

_USER_FIRST_TURN = """\
## Task

{instruction}

Begin by understanding what needs to be done, then execute the necessary commands.
"""

_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Signal that the task is complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class LiteLLMTerminalAgentLoop:
    """ReACT agent loop driving Terminal-Bench tasks via LiteLLM.

    Parameters
    ----------
    model:
        LiteLLM model identifier (e.g. ``"gpt-4o-mini"``,
        ``"openai/gpt-4o-mini"``, ``"anthropic/claude-haiku-4-5"``).
    env:
        Environment backend the agent operates in.
    max_turns:
        Maximum number of agent turns before forced stop.
    temperature:
        Sampling temperature for the LLM.
    max_tokens:
        Max completion tokens per LLM call.
    command_timeout:
        Default timeout in seconds for shell commands.
    use_native_tools:
        Use OpenAI-style tool calling (vs ReACT text). Auto-falls
        back to text on parse failure.
    """

    def __init__(
        self,
        model: str,
        env: EnvironmentBackend,
        *,
        max_turns: int = 200,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        command_timeout: int = 120,
        use_native_tools: bool = True,
        run_context: Optional[RunContext] = None,
    ) -> None:
        self.model = model
        self.env = env
        self._sync_env = SyncEnvironmentWrapper(env)
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.command_timeout = command_timeout
        self._use_native_tools = use_native_tools
        self.command_metrics = CommandMetrics()
        self._run_context = run_context
        # Opt-in Langfuse tracing (config-only; no-op unless AGENTSYSPERF_LANGFUSE=1
        # and creds are set). Registered once here, instruments every completion.
        enable_langfuse()

    def solve(self, task_id: str, instruction: str) -> TrialResult:
        result = TrialResult(task_id=task_id)
        t0 = time.time()
        self.command_metrics = CommandMetrics()
        conversation = self._build_initial_conversation(instruction)
        consecutive_parse_fails = 0

        try:
            for turn_idx in range(self.max_turns):
                record = self._run_turn(turn_idx, conversation, parent_span_id=task_id)
                result.turns.append(record)
                if turn_idx == 1:
                    print(" ...", end="", flush=True)
                elif turn_idx > 0 and turn_idx % 5 == 0:
                    print(f" t{turn_idx}", end="", flush=True)
                result.num_turns += 1
                result.total_generation_ms += record.generation_ms
                result.total_prompt_tokens += record.prompt_tokens
                result.total_completion_tokens += record.completion_tokens

                # A replay fixture miss is permanent — the fixture cannot gain
                # the missing turn while the run is in flight. Continuing just
                # replays the same failure for every remaining turn (observed:
                # turns 0..199 on one miss), which buries the real cause and
                # wastes the run. Abort the trial and surface the trial_key,
                # which is the only thing that identifies the missing entry.
                if record.error and FIXTURE_MISS_MARKER in record.error:
                    result.error = record.error
                    logger.error(
                        "Aborting %s at turn %d: %s", task_id, turn_idx, record.error
                    )
                    break

                if record.action not in ("__thinking__",):
                    result.num_commands += 1
                    result.total_command_ms += record.command_ms
                    consecutive_parse_fails = 0

                if record.action == "submit":
                    result.submitted = True
                    break

                if record.action == "__thinking__":
                    consecutive_parse_fails += 1
                    if self._use_native_tools and consecutive_parse_fails >= 2:
                        logger.info("Falling back to ReACT text parsing.")
                        self._use_native_tools = False
                        consecutive_parse_fails = 0
        except Exception as e:
            logger.exception("Fatal error in agent loop for %s", task_id)
            result.error = str(e)

        result.wall_clock_ms = (time.time() - t0) * 1000.0
        result.command_metrics = self.command_metrics.to_dict()

        # Build step-level trace rows from this task's turns and hand them to
        # the RunContext for persistence. Best-effort: tracing must never break
        # a benchmark run.
        if self._run_context is not None:
            try:
                from src.trace.builder import turns_to_steptraces

                steps = turns_to_steptraces(
                    run_id=self._run_context.run_id,
                    task_id=task_id,
                    turns=result.turns,
                    model=self.model,
                    task_duration_s=result.wall_clock_ms / 1000.0,
                )
                self._run_context.add_step_traces(steps)
            except Exception:
                logger.warning("Failed to build step traces for %s", task_id,
                               exc_info=True)

        return result

    def _run_turn(
        self,
        turn_idx: int,
        conversation: List[Dict[str, str]],
        parent_span_id: str = "unknown",
    ) -> TurnRecord:
        record = TurnRecord(turn=turn_idx)

        # Sub-span for LLM inference (Phase 03: Reason)
        gen_t0 = time.time()
        record.llm_start_ns = time.time_ns()  # epoch ns, for the StepTrace timeline
        span_id = f"{parent_span_id}/turn_{turn_idx}_llm"

        ctx = self._run_context
        if ctx is not None:
            span_ctx = track_span(
                ctx,
                span_id=span_id,
                kind="inference",
                node_id=f"llm_call_{turn_idx}",
                phase="reason",
            )
            span_ctx.__enter__()
        else:
            span_ctx = None

        # Guarantee EXACTLY ONE span close via a single exc-info path: the
        # finally always closes the span (with real exc info on failure, or
        # (None,None,None) on success), so an unexpected exception can never
        # leak a half-open span into a centralized stop() flush. The Langfuse
        # completion_metadata() call is left untouched in place.
        _exc_info: tuple = (None, None, None)
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": conversation,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self._use_native_tools:
                kwargs["tools"] = _TOOL_DEFINITIONS
                kwargs["tool_choice"] = "auto"

            # Attach Langfuse trace metadata (run_id/task_id) when tracing is on;
            # returns {} otherwise so this is a harmless splat.
            run_id = getattr(ctx, "run_id", "") if ctx is not None else ""
            kwargs.update(
                completion_metadata(
                    run_id=run_id, task_id=parent_span_id, turn_idx=turn_idx
                )
            )

            response = litellm.completion(**kwargs)
        except Exception as e:
            record.error = f"Generation failed: {e}"
            logger.warning("LiteLLM call failed at turn %d: %s", turn_idx, e)
            _exc_info = (type(e), e, e.__traceback__)
        finally:
            if span_ctx is not None:
                span_ctx.__exit__(*_exc_info)
        if record.error is not None:
            return record

        record.generation_ms = (time.time() - gen_t0) * 1000.0
        record.llm_end_ns = time.time_ns()

        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        if usage is not None:
            record.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            record.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        record.stop_reason = getattr(choice, "finish_reason", "") or ""

        action: Optional[str] = None
        args: Dict[str, Any] = {}
        reply_text = (getattr(message, "content", None) or "").strip()
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            tc = tool_calls[0]
            action = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}
            conversation.append({
                "role": "assistant",
                "content": (
                    f"Thought: (native tool call)\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(args)}"
                ),
            })
        elif reply_text:
            thought, action, args = parse_model_response(reply_text)
            if _HARMONY_MARKER in reply_text:
                clean = (
                    f"Thought: {thought or ''}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(args)}"
                ) if action is not None else _strip_harmony_markers(reply_text)
                conversation.append({"role": "assistant", "content": clean})
            else:
                conversation.append({"role": "assistant", "content": reply_text})

        if action is None:
            record.error = "No action parsed from model response"
            if not tool_calls and reply_text:
                conversation.append({
                    "role": "user",
                    "content": (
                        "Observation: Your response did not include a valid "
                        "Action. Respond with:\nThought: ...\nAction: shell\n"
                        'Action Input: {"command": "..."}'
                    ),
                })
            return record

        if action == "submit":
            record.action = "submit"
            conversation.append({
                "role": "user",
                "content": "Observation: Task submitted.",
            })
            return record

        record.action = "shell"
        command = args.get("command", "")
        if not command:
            record.error = "No command provided in Action Input"
            conversation.append({
                "role": "user",
                "content": (
                    "Observation: Error — no 'command' key in Action Input. "
                    'Use: Action Input: {"command": "your command"}'
                ),
            })
            return record

        record.command = command[:500]
        tier = classify_command(command)
        record.command_tier = tier.value if hasattr(tier, "value") else str(tier)

        # Sub-span for command execution: Phase 04 (Act) for ordinary shell
        # work, or Phase 02 (Retrieve) when the command is semantic /
        # index-backed retrieval — vector search, embedding generation,
        # BM25/TF-IDF, reranking. Retrieve is a first-class pipeline phase with
        # its own hardware signature and its own optimizations
        # (PHASE_SOLUTIONS["retrieve"]), so folding it into `act` hid it.
        # Filesystem inspection (grep/find/cat) stays `act` — see
        # benchmarks/terminal_bench/retrieval.py for the boundary and why a
        # false positive is worse than a miss here.
        from src.benchmarks.terminal_bench.retrieval import retrieval_signals
        _signals = retrieval_signals(command)
        _phase = "retrieve" if _signals else "act"
        record.phase = _phase
        record.retrieval_signals = _signals

        cmd_span_id = f"{parent_span_id}/turn_{turn_idx}_cmd"
        ctx = self._run_context
        if ctx is not None:
            cmd_span_ctx = track_span(
                ctx,
                span_id=cmd_span_id,
                kind="retrieval" if _phase == "retrieve" else "execution",
                node_id=f"cmd_{turn_idx}",
                phase=_phase,
            )
            cmd_span_ctx.__enter__()
        else:
            cmd_span_ctx = None

        cmd_t0 = time.time()
        record.cmd_start_ns = time.time_ns()
        env_result = None
        exec_error: Optional[Exception] = None
        try:
            env_result = self._sync_env.exec(command, timeout_sec=self.command_timeout)
        except Exception as e:  # env not started / backend failure
            exec_error = e
        finally:
            if cmd_span_ctx is not None:
                cmd_span_ctx.__exit__(None, None, None)

        # Fail closed: if the environment could not execute the command, record
        # an error turn and feed the agent an observation — NEVER fall through
        # to host execution. (See docs/INCIDENT_host_exec_venv_pollution.md.)
        if exec_error is not None:
            record.command_ms = (time.time() - cmd_t0) * 1000.0
            record.cmd_end_ns = time.time_ns()
            record.exit_code = -1
            record.error = f"Environment exec failed: {type(exec_error).__name__}: {exec_error}"
            logger.warning("Command exec failed in env at turn %d: %s", turn_idx, exec_error)
            conversation.append({
                "role": "user",
                "content": (
                    f"Observation: command could not be executed in the "
                    f"environment ({type(exec_error).__name__}). The environment "
                    f"may be unavailable."
                ),
            })
            return record

        cmd_ms = (time.time() - cmd_t0) * 1000.0
        record.command_ms = cmd_ms
        record.cmd_end_ns = time.time_ns()
        record.exit_code = env_result.return_code

        self.command_metrics.calls.append(
            CommandCallRecord(
                command=command[:200],
                resource_tier=tier,
                wall_clock_ms=cmd_ms,
                exit_code=env_result.return_code,
                output_length=len(env_result.output),
            )
        )

        output = env_result.output
        if len(output) > 6000:
            output = output[:6000] + "\n... [output truncated]"

        if env_result.success:
            observation = f"Observation: {output}" if output else "Observation: (no output)"
        else:
            observation = (
                f"Observation: Command exited with code {env_result.return_code}.\n"
                f"{output}"
            )

        conversation.append({"role": "user", "content": observation})
        return record

    @staticmethod
    def _build_initial_conversation(instruction: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_FIRST_TURN.format(instruction=instruction)},
        ]


__all__ = [
    "LiteLLMTerminalAgentLoop",
    "TurnRecord",
    "TrialResult",
]
