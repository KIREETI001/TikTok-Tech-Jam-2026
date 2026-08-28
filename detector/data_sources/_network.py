"""Shared network-retry helper for streamed data sources.

huggingface_hub's own retry (via ``http_backoff``) handles brief blips --
a handful of attempts within about a minute. That wasn't enough for a real
incident hit during a multi-hour training run: a transient network/DNS
disruption (``getaddrinfo failed`` / a stale socket) outlasted it and
crashed the entire run partway through epoch 1, losing everything. A
single dropped connection shouldn't cost hours of training, so shard
fetches in dragon.py/sid_set_stream.py retry through this helper with a
much more patient budget before giving up for good.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_network_call(
    fn: Callable[[], T],
    *,
    description: str,
    attempts: int = 8,
    delay_seconds: float = 60.0,
) -> T:
    """Call ``fn()``, retrying on any exception up to ``attempts`` times
    with a fixed delay between tries (default: 8 attempts, 60s apart --
    ~7 minutes of patience total). Re-raises the last exception if every
    attempt fails.
    """

    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: network/DNS/socket errors vary
            last_exc = exc
            if attempt == attempts:
                break
            print(
                f"  [retry] {description} failed (attempt {attempt}/{attempts}): "
                f"{exc!r} -- retrying in {delay_seconds:.0f}s"
            )
            time.sleep(delay_seconds)
    assert last_exc is not None
    raise last_exc
