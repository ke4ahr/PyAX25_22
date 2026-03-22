# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/agw/exceptions.py

Exceptions for the AGW Packet Engine (AGWPE) interface.
"""

from pyax25_22.core.exceptions import AGWPEError


class AGWConnectionError(AGWPEError):
    """Raised when the TCP connection to AGWPE cannot be established or drops.

    Examples:
        - AGWPE server is not running.
        - All retry attempts exhausted.
        - Connection dropped during receive.
    """


class AGWFrameError(AGWPEError):
    """Raised when an AGWPE frame is malformed or has invalid fields.

    Examples:
        - Header shorter than 36 bytes.
        - Declared data length exceeds a safe limit.
        - Unknown or unsupported frame type.
    """


class AGWLoginError(AGWPEError):
    """Raised when AGWPE login fails.

    Note: Not all AGWPE implementations require or support login.
    """


class AGWTimeoutError(AGWPEError):
    """Raised when an AGWPE operation times out."""
