# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.exceptions.py

Comprehensive exception hierarchy for PyAX25_22.

All exceptions inherit from AX25Error to allow broad catching while
providing specific subclasses for detailed error handling and logging.

The hierarchy is organized by error domain:
- Frame-level errors (parsing, encoding, validation)
- Connection state and protocol errors
- Transport/interface errors
- Configuration and negotiation errors

Every exception logs at ERROR level on instantiation for traceability.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AX25Error(Exception):
    """
    Base exception for all AX.25-related errors in PyAX25_22.

    All library-specific exceptions inherit from this class.
    """

    def __init__(self, message: str, *, frame_data: Optional[bytes] = None) -> None:
        """
        Initialize with message and optional raw frame data.

        Args:
            message: Human-readable error description
            frame_data: Raw frame bytes that caused the error (for debugging)
        """
        super().__init__(message)
        self.frame_data = frame_data
        logger.error(f"{self.__class__.__name__}: {message}")
        if frame_data is not None:
            logger.debug(f"Associated frame data: {frame_data.hex()}")


# Frame-level errors
class FrameError(AX25Error):
    """Base class for errors during frame parsing or encoding."""


class InvalidAddressError(FrameError):
    """Raised when an AX.25 address field is malformed or invalid."""
    pass


class InvalidControlFieldError(FrameError):
    """Raised when the control field is invalid or unsupported."""
    pass


class FCSError(FrameError):
    """Raised when FCS (CRC) validation fails."""
    pass


class BitStuffingError(FrameError):
    """Raised when bit stuffing/destuffing rules are violated."""
    pass


class SegmentationError(FrameError):
    """Raised during segmentation/reassembly failures."""
    pass


# Connection and protocol errors
class ConnectionError(AX25Error):
    """Base class for connection-related errors."""
    pass


class ConnectionStateError(ConnectionError):
    """
    Raised when an operation is attempted in an invalid connection state.

    Examples: Sending data while disconnected, or receiving unexpected frame.
    """
    pass


class TimeoutError(ConnectionError):
    """
    Raised when a timer expires (T1 acknowledgment, T3 idle probe).
    """
    pass


class ProtocolViolationError(ConnectionError):
    """Raised for general AX.25 v2.2 protocol violations (e.g., FRMR conditions)."""
    pass


# Transport and interface errors
class TransportError(AX25Error):
    """Base class for errors in transport interfaces (KISS, AGWPE)."""
    pass


class KISSError(TransportError):
    """Raised for KISS protocol violations or serial I/O errors."""
    pass


class AGWPEError(TransportError):
    """Raised for AGWPE TCP/IP protocol violations or connection issues."""
    pass


# Configuration and negotiation
class ConfigurationError(AX25Error):
    """
    Raised when configuration parameters violate AX.25 v2.2 limits.

    Used during AX25Config creation and parameter validation.
    """
    pass


class NegotiationError(ConnectionError):
    """
    Raised during XID parameter negotiation failure.

    Examples: Modulo mismatch, unsupported features.
    """
    pass


# Optional: Future expansion
class ResourceExhaustionError(AX25Error):
    """Raised when internal resources (buffers, windows) are exhausted."""
    pass


# Summary of exception hierarchy:
#
# AX25Error
# ├── FrameError
# │   ├── InvalidAddressError
# │   ├── InvalidControlFieldError
# │   ├── FCSError
# │   ├── BitStuffingError
# │   └── SegmentationError
# ├── ConnectionError
# │   ├── ConnectionStateError
# │   ├── TimeoutError
# │   ├── ProtocolViolationError
# │   └── NegotiationError
# ├── TransportError
# │   ├── KISSError
# │   └── AGWPEError
# ├── ConfigurationError
# └── ResourceExhaustionError
