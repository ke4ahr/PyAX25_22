# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/agw/__init__.py

Public re-exports for the pyax25_22 AGW Packet Engine interface package.
"""

from .client import AGWPEClient, AGWPEFrame
from .serial import AGWSerial
from .exceptions import (
    AGWConnectionError,
    AGWFrameError,
    AGWLoginError,
    AGWTimeoutError,
)
from .constants import (
    AGWPE_DEFAULT_PORT,
    AGWPE_HEADER_SIZE,
    KIND_VERSION,
    KIND_REGISTER,
    KIND_UNREGISTER,
    KIND_PORT_INFO,
    KIND_EXTENDED_VER,
    KIND_MEMORY_USAGE,
    KIND_ENABLE_MON,
    KIND_RAW_MON,
    KIND_RAW_SEND,
    KIND_UNPROTO,
    KIND_UNPROTO_VIA,
    KIND_UNPROTO_DATA,
    KIND_CONNECT,
    KIND_CONNECT_INC,
    KIND_DISC,
    KIND_CONN_DATA,
    KIND_OUTSTANDING,
    KIND_HEARD,
    KIND_LOGIN,
    KIND_PARAMETER,
)

__all__ = [
    "AGWPEClient",
    "AGWPEFrame",
    "AGWSerial",
    "AGWConnectionError",
    "AGWFrameError",
    "AGWLoginError",
    "AGWTimeoutError",
    "AGWPE_DEFAULT_PORT",
    "AGWPE_HEADER_SIZE",
    "KIND_VERSION",
    "KIND_REGISTER",
    "KIND_UNREGISTER",
    "KIND_PORT_INFO",
    "KIND_EXTENDED_VER",
    "KIND_MEMORY_USAGE",
    "KIND_ENABLE_MON",
    "KIND_RAW_MON",
    "KIND_RAW_SEND",
    "KIND_UNPROTO",
    "KIND_UNPROTO_VIA",
    "KIND_UNPROTO_DATA",
    "KIND_CONNECT",
    "KIND_CONNECT_INC",
    "KIND_DISC",
    "KIND_CONN_DATA",
    "KIND_OUTSTANDING",
    "KIND_HEARD",
    "KIND_LOGIN",
    "KIND_PARAMETER",
]
