"""Graceful parent-only SIGTERM lifecycle guards for build worker pools (#881).

Contract
--------
- In main-thread CLI and library execution where ``signal.SIGTERM`` is currently
  set to ``signal.SIG_DFL``, :func:`graceful_sigterm_scope` installs a temporary
  handler that converts ``SIGTERM`` into :class:`ParentTerminated` (a
  :class:`SystemExit` subclass with POSIX exit code ``128 + signum``).
- Converting ``SIGTERM`` into a :class:`BaseException` lets Python unwind all
  ``finally`` blocks and context managers so owned worker children are promptly
  terminated, owned scratch directories are removed, and transactional
  generation publication never advertises incomplete output or disturbs the
  previous published generation.
- Caller signal policies are preserved: if ``signal.SIGTERM`` already has a
  custom Python handler or ``signal.SIG_IGN``, or if execution is running on a
  non-main thread (or a platform without ``SIGTERM``), the scope is a no-op and
  never overwrites the caller's handler.

Limits
------
- Uncatchable termination (``SIGKILL``, kernel OOM-killer, abrupt container or
  host power loss) bypasses user-space Python handlers and cannot guarantee
  child termination or scratch removal. Even then, Hitlist's atomic generation
  publication ensures the previous published generation remains intact and no
  partial worker output is advertised as complete.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Iterator


class ParentTerminated(SystemExit):
    """Raised when a parent-only ``SIGTERM`` interrupts a managed worker scope."""

    def __init__(self, signum: int | None = None) -> None:
        resolved = int(getattr(signal, "SIGTERM", 15) if signum is None else signum)
        self.signum = resolved
        super().__init__(128 + resolved)


@contextlib.contextmanager
def graceful_sigterm_scope() -> Iterator[bool]:
    """Convert default parent-only ``SIGTERM`` handling into :class:`ParentTerminated`.

    Yields ``True`` when this scope installed the temporary handler and ``False``
    when the scope was a no-op (non-main thread, missing ``SIGTERM``, nested
    scope, or caller-installed custom / ``SIG_IGN`` handler).
    """
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is None or threading.current_thread() is not threading.main_thread():
        yield False
        return

    previous = signal.getsignal(sigterm)
    if previous is not signal.SIG_DFL:
        yield False
        return

    def _handle_sigterm(signum: int, _frame: object) -> None:
        # Ignore duplicate SIGTERMs while owned workers and scratch unwind;
        # SIGKILL remains unconditionally available.
        with contextlib.suppress(OSError, ValueError):
            signal.signal(sigterm, signal.SIG_IGN)
        raise ParentTerminated(signum)

    signal.signal(sigterm, _handle_sigterm)
    try:
        yield True
    finally:
        with contextlib.suppress(OSError, ValueError):
            signal.signal(sigterm, previous)


def terminate_process_pool(pool: object, *, timeout: float = 5.0) -> None:
    """Promptly terminate owned ``ProcessPoolExecutor`` workers and shut down the pool."""
    processes = list((getattr(pool, "_processes", None) or {}).values())
    for process in processes:
        with contextlib.suppress(OSError, ValueError):
            if process.is_alive():
                process.terminate()

    deadline = time.monotonic() + max(0.0, float(timeout))
    for process in processes:
        with contextlib.suppress(OSError, ValueError):
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)

    shutdown = getattr(pool, "shutdown", None)
    if callable(shutdown):
        with contextlib.suppress(Exception):
            shutdown(wait=True, cancel_futures=True)
