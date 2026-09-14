#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""ReACT response parser, ported from AgentOptimizer's react_swe_loop.

Standalone module — no AgentFlow runtime dependencies. Handles both
plain ReACT format (``Thought:``/``Action:``/``Action Input:``) and
gpt-oss Harmony channel markers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_HARMONY_MARKER = "<|channel|>"

_HARMONY_CHANNEL_RE = re.compile(
    r"<\|channel\|>(?P<channel>\w+)"
    r"(?:\s+to=(?P<recipient>[^\s<]+))?"
    r"(?:\s+\w+|\s*<\|constrain\|>\w+)*"
    r"<\|message\|>(?P<content>.*?)(?:<\|end\|>|<\|call\|>|<\|return\|>|$)",
    re.DOTALL,
)


@dataclass
class _HarmonyParsed:
    analysis: Optional[str] = None
    final: Optional[str] = None
    tool_calls: List[tuple] = field(default_factory=list)


def _parse_harmony_channels(text: str) -> Optional[_HarmonyParsed]:
    if _HARMONY_MARKER not in text:
        return None

    result = _HarmonyParsed()
    for m in _HARMONY_CHANNEL_RE.finditer(text):
        channel = m.group("channel")
        content = m.group("content").strip()
        recipient = m.group("recipient")

        if recipient:
            tool_name = recipient.split(".")[-1] if "." in recipient else recipient
            result.tool_calls.append((tool_name, content))
        elif channel == "analysis":
            result.analysis = content
        elif channel == "final":
            result.final = content
    return result


def _strip_harmony_markers(text: str) -> str:
    harmony = _parse_harmony_channels(text)
    if harmony is None:
        return text

    parts: list[str] = []
    if harmony.analysis:
        parts.append(harmony.analysis)
    if harmony.final:
        parts.append(harmony.final)
    for _name, args_json in harmony.tool_calls:
        parts.append(args_json)
    return "\n".join(parts) if parts else text


_THOUGHT_RE = re.compile(r"Thought:\s*(?P<thought>.+?)(?=\nAction:)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(?P<action>\S+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(?P<input>\{.*\})", re.DOTALL)


def parse_react_response(
    text: str,
) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Parse plain ReACT format. Returns (thought, action, args)."""
    thought = None
    action = None
    args: Dict[str, Any] = {}

    m = _THOUGHT_RE.search(text)
    if m:
        thought = m.group("thought").strip()

    m = _ACTION_RE.search(text)
    if m:
        action = m.group("action").strip()

    m = _ACTION_INPUT_RE.search(text)
    if m:
        raw = m.group("input").strip()
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            brace_depth = 0
            for i, ch in enumerate(raw):
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        try:
                            args = json.loads(raw[: i + 1])
                        except json.JSONDecodeError:
                            pass
                        break
    return thought, action, args


def parse_model_response(
    text: str,
) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Parse a model response handling both Harmony and plain ReACT.

    Returns ``(thought, action_name, action_args)``.
    """
    harmony = _parse_harmony_channels(text)
    if harmony is None:
        return parse_react_response(text)

    if harmony.tool_calls:
        tool_name, args_json = harmony.tool_calls[0]
        thought = harmony.analysis or harmony.final
        args: Dict[str, Any] = {}
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            pass
        return thought, tool_name, args

    final_text = harmony.final or ""
    thought, action, args = parse_react_response(final_text)
    if thought is None and harmony.analysis:
        thought = harmony.analysis

    if action is None:
        combined = ""
        if harmony.analysis:
            combined += harmony.analysis + "\n"
        if harmony.final:
            combined += harmony.final
        if combined != final_text:
            _, action, args = parse_react_response(combined)

    return thought, action, args


__all__ = [
    "parse_model_response",
    "parse_react_response",
    "_strip_harmony_markers",
    "_HARMONY_MARKER",
]
