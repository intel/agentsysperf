#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""ReplayProxy — start/stop the record/replay proxy as a subprocess.

The proxy is launched as its own process (``python -m src.replay.proxy``)
rather than in-process because uvicorn owns the event loop and the proxy must
outlive any single agent call. The manager:

  * launches the proxy in the requested mode,
  * health-checks ``/healthz`` so a misconfigured fixture fails loudly at
    setup, not mid-sweep,
  * exposes the ``OPENAI_API_BASE`` / ``OPENAI_BASE_URL`` env the agent (or
    Harbor ``--ae`` flags) must point at,
  * tears the proxy down on context exit.

Usage::

    with ReplayProxy(mode="replay", fixture=Path("fixture.jsonl")) as proxy:
        env = {**os.environ, **proxy.env}        # OPENAI_API_BASE -> the proxy
        subprocess.run(harbor_cmd, env=env)      # agent's LLM calls are replayed
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import httpx

log = logging.getLogger(__name__)


class ReplayProxy:
    """Context manager around the record/replay proxy subprocess."""

    def __init__(
        self,
        *,
        mode: str = "off",
        fixture: Optional[Path] = None,
        upstream: str = "http://127.0.0.1:4000",
        upstream_key: Optional[str] = None,
        port: int = 4001,
        host: str = "127.0.0.1",
        advertise_host: Optional[str] = None,
        inject_latency: bool = False,
        strict_miss: bool = True,
        startup_timeout_s: float = 15.0,
    ) -> None:
        if mode not in ("off", "replay", "record"):
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "replay" and fixture is None:
            raise ValueError("replay mode requires a fixture path")
        if mode == "replay" and not Path(fixture).exists():
            raise FileNotFoundError(f"fixture not found: {fixture}")
        self.mode = mode
        self.fixture = Path(fixture) if fixture else None
        self.upstream = upstream
        # Real upstream credential for record mode (the proxy re-signs the
        # Authorization header; the agent only ever sees the dummy key).
        self.upstream_key = upstream_key
        self.port = port
        self.host = host
        # The address agents DIAL (vs `host`, the bind address). For an agent
        # running inside a Docker container, 127.0.0.1 is the CONTAINER's
        # loopback — it must reach the host proxy via the docker bridge gateway
        # (e.g. 172.17.0.1). Defaults to `host` for the in-process/host case.
        self.advertise_host = advertise_host or host
        self.inject_latency = inject_latency
        self.strict_miss = strict_miss
        self.startup_timeout_s = startup_timeout_s
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL agents should target — uses
        advertise_host so containerized agents dial the bridge gateway, not
        their own loopback."""
        return f"http://{self.advertise_host}:{self.port}/v1"

    @property
    def env(self) -> Dict[str, str]:
        """Env vars that redirect an OpenAI/LiteLLM client to this proxy.

        Merge into the agent's environment. The dummy key satisfies clients
        that require one; the proxy never validates it.
        """
        return {
            "OPENAI_API_KEY": "sk-not-needed-local-only",
            "OPENAI_API_BASE": self.base_url,
            "OPENAI_BASE_URL": self.base_url,
        }

    def start(self) -> "ReplayProxy":
        if self._proc is not None:
            raise RuntimeError("proxy already started")

        cmd = [
            sys.executable, "-m", "src.replay.proxy",
            "--mode", self.mode,
            "--port", str(self.port),
            "--host", self.host,
        ]
        if self.mode == "replay":
            cmd += ["--fixture", str(self.fixture)]
            if self.inject_latency:
                cmd += ["--inject-latency"]
            if not self.strict_miss:
                cmd += ["--no-strict"]
        elif self.mode == "record":
            cmd += ["--upstream", self.upstream]
            if self.fixture:
                cmd += ["--fixture", str(self.fixture)]

        # Pass the real upstream credential to the proxy explicitly (record
        # mode re-signs Authorization with it). Deterministic — does not rely on
        # fork-time env timing.
        proc_env = dict(os.environ)
        if self.upstream_key:
            proc_env["AGENTSYSPERF_UPSTREAM_KEY"] = self.upstream_key

        log.info("Starting replay proxy: mode=%s port=%d", self.mode, self.port)
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=proc_env,
        )
        if not self._wait_healthy():
            self.stop()
            raise RuntimeError(
                f"replay proxy failed health check on port {self.port}"
            )
        return self

    def _wait_healthy(self) -> bool:
        deadline = time.monotonic() + self.startup_timeout_s
        # Probe via loopback — 0.0.0.0 is a valid BIND address but not a valid
        # GET target. The proxy listens on all interfaces when host=0.0.0.0, so
        # 127.0.0.1 reaches it from the host.
        # a comparison, not a bind
        _probe_host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host  # nosec B104
        url = f"http://{_probe_host}:{self.port}/healthz"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                # Proxy died during startup — surface its stderr.
                err = (self._proc.stderr.read().decode(errors="replace")
                       if self._proc.stderr else "")
                log.error("replay proxy exited during startup: %s", err[-500:])
                return False
            try:
                # trust_env=False: never route a localhost health check through
                # a corporate HTTP_PROXY (EMR sets one), which 403s 127.0.0.1.
                r = httpx.get(url, timeout=2.0, trust_env=False)
                if r.status_code == 200:
                    h = r.json()
                    log.info(
                        "replay proxy healthy: mode=%s trials=%s entries=%s",
                        h.get("mode"), h.get("fixture_trials"), h.get("fixture_entries"),
                    )
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        return False

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def __enter__(self) -> "ReplayProxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


__all__ = ["ReplayProxy"]
