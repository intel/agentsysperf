#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Proxy-port admission for run-streams.

``_check_proxy_ports`` gates a uvicorn subprocess per worker, and uvicorn binds its
listener with ``SO_REUSEADDR`` (asyncio's ``create_server`` default on POSIX).
A probe that binds *without* it is stricter than the server it gates and
rejects ports the proxy would have started on — which is what made consecutive
``run-streams`` invocations fail with "proxy port 4001 is unavailable" while
nothing was listening.

These tests pin both halves of the contract: a drained port must be admitted,
and a port held by a live listener must still be refused.
"""

from __future__ import annotations

import socket
import time

import pytest

from src.streams.orchestrator import _check_proxy_ports


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _time_wait_port() -> int:
    """A port a closed listener left in TIME_WAIT, as an exited proxy does."""
    port = _free_port()
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    cli = socket.create_connection(("127.0.0.1", port))
    acc, _ = srv.accept()
    # Close the local (server) side first so the local port enters TIME_WAIT,
    # exactly as it does when a proxy subprocess exits.
    srv.close()
    acc.close()
    cli.close()
    time.sleep(0.3)
    return port


def test_port_in_time_wait_is_admitted():
    """The regression: a just-exited proxy leaves TIME_WAIT behind."""
    _check_proxy_ports(_time_wait_port(), 1)  # must not raise


def test_port_with_live_listener_is_refused():
    """The check must still fail closed: SO_REUSEADDR is not SO_REUSEPORT."""
    port = _free_port()
    live = socket.socket()
    live.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    live.bind(("127.0.0.1", port))
    live.listen(1)
    try:
        with pytest.raises(RuntimeError, match="unavailable"):
            _check_proxy_ports(port, 1)
    finally:
        live.close()


def test_a_free_port_is_admitted():
    _check_proxy_ports(_free_port(), 1)


def test_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="invalid"):
        _check_proxy_ports(65535, 4)
    with pytest.raises(ValueError, match="invalid"):
        _check_proxy_ports(0, 1)


def test_every_port_in_the_range_is_probed():
    """A live listener anywhere in the span must fail the whole range."""
    base = _free_port()
    # Probe a 4-wide range whose *third* port is occupied.
    live = socket.socket()
    live.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        live.bind(("127.0.0.1", base + 2))
        live.listen(1)
    except OSError:  # pragma: no cover - the ephemeral neighbour was taken
        live.close()
        pytest.skip("could not reserve base+2")
    try:
        with pytest.raises(RuntimeError, match=f"port {base + 2} is unavailable"):
            _check_proxy_ports(base, 4)
    finally:
        live.close()
