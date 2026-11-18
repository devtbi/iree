# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Log sink setup for application entry points, not for library imports."""

import os
import re
import sys
import threading

from loguru import logger

__all__ = ["configure_application_logging"]

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>"
    " | <level>{level: <8}</level>"
    " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
    " - <level>{message}</level>"
)

# The stderr sink that Loguru installs by itself when it is imported. It is the
# first sink, so it has id 0, and this is how Loguru shows it.
_PRECONFIGURED_SINK_ID = 0
_PRECONFIGURED_SINK = re.compile(r"\(id=0, level=\d+, sink=<stderr>\)")

# Calls are serialized: replacing the sink of the previous call is one step.
_lock = threading.Lock()

# The sink that the last call of configure_application_logging added.
_sink_id: int | None = None


def _has_preconfigured_sink() -> bool:
    """Whether the stderr sink that Loguru installed by itself is still there.

    That sink has id 0, and Loguru never uses an id twice. A stderr sink that
    the application added as its first one, with Loguru's own turned off,
    cannot be told apart from it.
    """
    return _PRECONFIGURED_SINK.search(repr(logger)) is not None


def _remove_sink(sink_id: int) -> None:
    try:
        logger.remove(sink_id)
    except ValueError:
        pass  # Already removed.


def configure_application_logging(
    log_output: str | os.PathLike[str] | None = None,
    level: str | int = "INFO",
    format: str = _DEFAULT_FORMAT,
) -> int | None:
    """Send log messages to stderr or to a rotating file.

    For the entry point of an application. Libraries must not call it: where
    their messages go is up to the application. It replaces the stderr sink
    that Loguru installs by itself and the sink of an earlier call, and keeps
    the sinks that the application added.

    Args:
        log_output: Optional log file path. Files rotate at 10 MB. If omitted,
            log messages are written to stderr.
        level: Minimum severity to write, defaulting to INFO.
        format: Loguru message format, defaulting to timestamps, severity,
            source location and message.

    Returns:
        The id of the new sink, which ``logger.remove()`` accepts until the
        next call replaces the sink, or ``None`` if there is nothing to write
        to: no ``log_output`` and no stderr.

    Raises:
        ValueError: If Loguru rejects ``level`` or ``format``. The sinks that
            were in place are kept.
    """
    global _sink_id
    with _lock:
        # Add the new sink first: if that fails, the sinks in place are kept.
        new_sink_id: int | None = None
        if log_output is not None:
            # Loguru fills in braces in a file name, as in "{time}.log".
            path = os.fspath(log_output).replace("{", "{{").replace("}", "}}")
            # No color codes in files, whatever the environment asks for.
            new_sink_id = logger.add(
                path, format=format, level=level, rotation="10 MB", colorize=False
            )
        elif sys.stderr is not None:
            # Loguru decides about colors itself: it knows about terminals and
            # the environment variables that turn colors on and off.
            new_sink_id = logger.add(sys.stderr, format=format, level=level)

        if _has_preconfigured_sink():
            _remove_sink(_PRECONFIGURED_SINK_ID)
        if _sink_id is not None:
            _remove_sink(_sink_id)
        _sink_id = new_sink_id
        return new_sink_id
