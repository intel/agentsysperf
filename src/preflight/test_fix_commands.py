#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Tests for the `preflight --fix` command contract.

`--fix` runs commands under sudo. One of them interpolates a runtime value
(``platform.release()`` into the linux-headers package name), so running the set
through a shell by default would make that an injection vector. These tests pin
the invariant that only commands genuinely needing shell metacharacters declare
``fix_needs_shell``.
"""

from __future__ import annotations

import shlex

from src.preflight.system_check import SystemCheck

# Characters that give the shell semantics an argv list cannot express.
_SHELL_METACHARACTERS = set("|&;<>()$`")


def _all_fix_commands():
    checks = SystemCheck().run().checks
    return [c for c in checks if c.fix_command]


def test_fix_needs_shell_matches_actual_metacharacter_use():
    """A command needs the shell iff it contains shell metacharacters.

    Both directions matter. A command flagged as needing a shell that doesn't
    pays the injection risk for nothing; one that needs it but isn't flagged
    breaks at runtime.
    """
    for c in _all_fix_commands():
        has_meta = bool(set(c.fix_command) & _SHELL_METACHARACTERS)
        assert c.fix_needs_shell == has_meta, (
            f"{c.name}: fix_needs_shell={c.fix_needs_shell} but "
            f"command {c.fix_command!r} "
            f"{'contains' if has_meta else 'contains no'} shell metacharacters"
        )


def test_non_shell_fix_commands_survive_shlex_split():
    """The argv path must not mangle any command that uses it."""
    for c in _all_fix_commands():
        if c.fix_needs_shell:
            continue
        argv = shlex.split(c.fix_command)
        assert argv, f"{c.name}: {c.fix_command!r} split to nothing"
        # Round-trips without introducing quoting the shell would have removed.
        assert shlex.split(shlex.join(argv)) == argv, f"{c.name}: unstable split"


def test_interpolated_fix_command_does_not_take_the_shell_path():
    """The linux-headers fix embeds platform.release(); it must stay argv-only.

    This is the specific command that made shell=True a security concern rather
    than a style question.
    """
    headers = [c for c in _all_fix_commands() if "linux-headers-" in c.fix_command]
    for c in headers:
        assert not c.fix_needs_shell, (
            f"{c.name} interpolates a runtime value and must not run via a shell"
        )
