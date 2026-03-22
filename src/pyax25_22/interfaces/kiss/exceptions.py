# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/exceptions.py

Exception hierarchy for the KISS interface family.

All exceptions here are subclasses of KISSError, which is itself a subclass of
the library-wide KISSError already defined in pyax25_22.core.exceptions.
"""

from pyax25_22.core.exceptions import KISSError


class KISSTransportError(KISSError):
    """Raised when the underlying transport (serial or TCP) fails.

    This is the base class for all transport-layer KISS errors.
    """


class KISSSerialError(KISSTransportError):
    """Raised when a serial port operation fails.

    Examples:
        - The serial device does not exist or cannot be opened.
        - A write to the serial port is interrupted.
    """


class KISSTCPError(KISSTransportError):
    """Raised when a TCP socket operation fails.

    Examples:
        - Cannot connect to the remote KISS server.
        - The TCP connection drops mid-receive.
    """


class KISSFrameError(KISSError):
    """Raised when a received KISS frame is malformed.

    Examples:
        - A frame body has zero bytes (missing the command byte).
        - The payload contains an illegal escape sequence.
    """


class KISSChecksumError(KISSError):
    """Raised when XOR or CRC checksum verification fails.

    Used by XKISS (XOR checksum mode) and SMACK (CRC-16 mode).
    """


class KISSQueueFullError(KISSError):
    """Raised (or logged as a warning) when a receive queue overflows.

    In XKISS polling mode, incoming frames are buffered until the host polls.
    If the buffer exceeds max_queue_size, this error is raised (or the oldest
    frame is silently dropped, depending on the configuration).
    """
