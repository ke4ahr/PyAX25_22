# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/__init__.py

Public re-exports for the pyax25_22 KISS interface package.

Hierarchy:
    KISSBase          -- abstract framing, transport-agnostic
    KISSSerial        -- KISS over serial port
    KISSTCP           -- KISS over TCP socket
    XKISS             -- XKISS (multi-drop) over serial
    XKISSTCP          -- XKISS over TCP
    SMACK             -- SMACK (CRC-16) over serial
    SMACKTCP          -- SMACK over TCP
"""

from .base import KISSBase
from .serial import KISSSerial
from .tcp import KISSTCP
from .xkiss import XKISSMixin, XKISS, XKISSTCP
from .smack import SMACKMixin, SMACK, SMACKTCP
from .async_tcp import AsyncKISSTCP
from .constants import (
    FEND, FESC, TFEND, TFESC,
    CMD_DATA, CMD_TXDELAY, CMD_PERSIST, CMD_SLOTTIME,
    CMD_TXTAIL, CMD_FULLDUP, CMD_HARDWARE, CMD_EXIT,
    CMD_POLL, CMD_DATA_ACK,
    SMACK_FLAG, SMACK_POLY, SMACK_INIT, SMACK_CRC_SIZE,
    PORT_MASK, CMD_MASK,
    DEFAULT_BAUDRATE, DEFAULT_POLL_INTERVAL, DEFAULT_MAX_QUEUE,
)
from .exceptions import (
    KISSTransportError,
    KISSSerialError,
    KISSTCPError,
    KISSFrameError,
    KISSChecksumError,
    KISSQueueFullError,
)

# Backward-compatible alias used by legacy tests and code
KISSInterface = KISSSerial

__all__ = [
    # Base classes
    "KISSBase",
    "KISSSerial",
    "KISSTCP",
    # XKISS
    "XKISSMixin",
    "XKISS",
    "XKISSTCP",
    # SMACK
    "SMACKMixin",
    "SMACK",
    "SMACKTCP",
    # Async
    "AsyncKISSTCP",
    # Constants
    "FEND", "FESC", "TFEND", "TFESC",
    "CMD_DATA", "CMD_TXDELAY", "CMD_PERSIST", "CMD_SLOTTIME",
    "CMD_TXTAIL", "CMD_FULLDUP", "CMD_HARDWARE", "CMD_EXIT",
    "CMD_POLL", "CMD_DATA_ACK",
    "SMACK_FLAG", "SMACK_POLY", "SMACK_INIT", "SMACK_CRC_SIZE",
    "PORT_MASK", "CMD_MASK",
    "DEFAULT_BAUDRATE", "DEFAULT_POLL_INTERVAL", "DEFAULT_MAX_QUEUE",
    # Exceptions
    "KISSTransportError",
    "KISSSerialError",
    "KISSTCPError",
    "KISSFrameError",
    "KISSChecksumError",
    "KISSQueueFullError",
    # Legacy alias
    "KISSInterface",
]
