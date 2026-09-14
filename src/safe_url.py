#=========================== begin_copyright_notice ============================
#
# Copyright (C) 2026 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
#============================ end_copyright_notice =============================

"""Scheme gate for outbound HTTP.

`urllib.request.urlopen` is a *general* URL opener, not an HTTP client: handed
`file:///etc/passwd` it happily reads a local file, and `ftp://` /
`data:` open their own transports. Everywhere we call it, the target is a
service endpoint assembled from configuration (`--server-url`, a Pushgateway
address, a proxy base). Configuration is not attacker input in normal use, but
"normal use" is exactly the assumption that rots: a base URL that reaches the
opener from a YAML sweep spec, an env var, or a future `--url` flag turns a
metrics push into an arbitrary local-file read.

One line at each call site removes the whole class, and costs nothing for the
http/https URLs we actually use.
"""

from __future__ import annotations

from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def require_http_url(url: str) -> str:
    """Return `url` unchanged if it is http(s); raise ValueError otherwise.

    Accepts either a string or anything with a `.full_url` (a
    `urllib.request.Request`), so it can wrap both urlopen call shapes.
    """
    raw = getattr(url, "full_url", url)
    scheme = urlparse(str(raw)).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"refusing to open non-HTTP URL (scheme {scheme!r}): {raw!r}"
        )
    return url
