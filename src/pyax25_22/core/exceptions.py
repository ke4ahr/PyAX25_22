# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.exceptions -- Error types for the PyAX25_22 library.

This file defines all the errors that PyAX25_22 can raise.
Every error in the library comes from AX25Error, so you can catch
all library errors with a single except AX25Error clause.

Errors are grouped by what caused them:
  - Frame errors: something is wrong with how a frame is built or read
  - Connection errors: something went wrong with the radio link
  - Transport errors: something went wrong with the serial port or network
  - Configuration errors: the settings you gave are not allowed

Every error logs itself when it is created, so you always get a
record of what went wrong and when.

Hierarchy::

    AX25Error
    +-- FrameError
    |   +-- InvalidAddressError
    |   +-- InvalidControlFieldError
    |   +-- FCSError
    |   +-- BitStuffingError
    |   +-- SegmentationError
    +-- ConnectionError
    |   +-- ConnectionStateError
    |   +-- TimeoutError
    |   +-- ProtocolViolationError
    |   +-- NegotiationError
    +-- TransportError
    |   +-- KISSError
    |   +-- AGWPEError
    +-- ConfigurationError
    +-- ResourceExhaustionError
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class AX25Error(Exception):
    """Base error for everything in PyAX25_22.

    All other errors in this library inherit from this one.
    Catching AX25Error will catch any library error.

    Attributes:
        frame_data: The raw bytes of the frame that caused the error,
            or None if there is no frame involved.

    Example::

        try:
            frame = AX25Frame.decode(raw_bytes)
        except AX25Error as err:
            print(f"Something went wrong: {err}")
    """

    def __init__(
        self,
        message: str,
        *,
        frame_data: Optional[bytes] = None,
    ) -> None:
        """Set up the error with a message and optional frame bytes.

        This logs the error right away so it appears in log files
        even if the caller does not handle it immediately.

        Args:
            message: A plain-English description of what went wrong.
            frame_data: The raw frame bytes that caused the problem,
                used for debugging. Leave as None if not available.

        Raises:
            Nothing -- this is the error, not the raiser.
        """
        super().__init__(message)
        self.frame_data: Optional[bytes] = frame_data
        logger.error("%s: %s", self.__class__.__name__, message)
        if frame_data is not None:
            logger.debug(
                "%s: associated frame bytes: %s",
                self.__class__.__name__,
                frame_data.hex(),
            )


# ---------------------------------------------------------------------------
# Frame-level errors
# ---------------------------------------------------------------------------

class FrameError(AX25Error):
    """Something went wrong while building or reading an AX.25 frame.

    Use more specific subclasses when possible:
      - InvalidAddressError: the callsign or SSID is bad
      - FCSError: the checksum does not match
      - BitStuffingError: the bit pattern is wrong
    """


class InvalidAddressError(FrameError):
    """An AX.25 address (callsign + SSID) is not valid.

    This happens when:
      - The callsign is longer than 6 characters
      - The callsign contains characters that are not letters or digits
      - The SSID is not in the range 0 to 15

    Example::

        # Raises InvalidAddressError because SSID 20 is out of range
        addr = AX25Address("KE4AHR", ssid=20)
    """


class InvalidControlFieldError(FrameError):
    """The control field of a frame is not valid or not supported.

    The control field tells us what kind of frame this is (I, S, or U).
    This error means the value found in the frame does not match any
    known frame type.
    """


class FCSError(FrameError):
    """The frame check sequence (CRC) is wrong.

    Every AX.25 frame has a 2-byte checksum at the end called the FCS.
    This error means the checksum calculated from the frame data does not
    match the checksum stored in the frame. This usually means the frame
    was damaged during transmission.

    Example::

        # Raises FCSError when the last 2 bytes do not match
        frame = AX25Frame.decode(corrupted_bytes)
    """


class BitStuffingError(FrameError):
    """The bit stuffing pattern in a frame is wrong.

    AX.25 uses HDLC bit stuffing: after five 1-bits in a row, a 0-bit
    is always inserted. On the receive side that extra 0-bit is removed.
    This error means the bit pattern in the frame breaks that rule, which
    usually means the frame is corrupted.
    """


class SegmentationError(FrameError):
    """A frame segmentation or reassembly step failed.

    Large messages can be split into smaller pieces (segmented) and put
    back together (reassembled). This error means that process went wrong.
    For example, a piece arrived out of order, or a piece is missing.
    """


# ---------------------------------------------------------------------------
# Connection and protocol errors
# ---------------------------------------------------------------------------

class ConnectionError(AX25Error):
    """Something went wrong with the AX.25 radio connection.

    This is the base class for all connection problems. Use a more
    specific subclass when possible.
    """


class ConnectionStateError(ConnectionError):
    """An action was requested at the wrong time.

    AX.25 connections follow a set of rules about what can happen when.
    For example, you cannot send data before the connection is established.
    This error means you (or the remote station) did something that is not
    allowed in the current state.

    Example::

        # Raises ConnectionStateError -- not connected yet
        conn.send_data(b"hello")
    """


class TimeoutError(ConnectionError):
    """A timer ran out before the expected response arrived.

    AX.25 uses timers to detect problems:
      - T1: waiting for an acknowledgment from the other station
      - T3: checking that an idle connection is still alive

    This error means a timer expired and the connection could not continue.
    """


class ProtocolViolationError(ConnectionError):
    """The other station did something that breaks the AX.25 rules.

    AX.25 v2.2 defines exactly how stations must behave. This error
    means the other station sent something unexpected or illegal
    according to the specification. This can also be raised when a
    FRMR (Frame Reject) frame is received.
    """


class NegotiationError(ConnectionError):
    """XID parameter negotiation failed.

    Before some connections are made, the two stations can exchange XID
    frames to agree on settings like the window size and modulo mode.
    This error means they could not agree on settings that both support.

    Example::

        # Raises NegotiationError if modulo values do not match
        config = negotiate_config(local_config, remote_params)
    """


# ---------------------------------------------------------------------------
# Transport and interface errors
# ---------------------------------------------------------------------------

class TransportError(AX25Error):
    """Something went wrong with the hardware or network connection.

    This is the base class for errors from the serial port (KISS) or
    the TCP connection (AGWPE). Use a more specific subclass when possible.
    """


class KISSError(TransportError):
    """Something went wrong with the KISS serial interface.

    KISS (Keep It Simple, Stupid) is a simple protocol for talking to
    a TNC (Terminal Node Controller) over a serial port. This error
    means something went wrong at that level -- for example:
      - The serial port could not be opened
      - The serial port stopped responding
      - A KISS frame had an invalid format

    Example::

        kiss = KISSInterface("/dev/ttyUSB0")
        kiss.connect()   # Raises KISSError if port not found
    """


class AGWPEError(TransportError):
    """Something went wrong with the AGWPE TCP/IP connection.

    AGWPE is a protocol for talking to a TNC over a TCP network
    connection (like a local network or the internet). This error
    means something went wrong at that level -- for example:
      - Could not connect to the AGWPE server
      - The network connection was dropped
      - An AGWPE frame had an invalid format

    Example::

        agwpe = AGWPEInterface()
        agwpe.connect()  # Raises AGWPEError if server is not running
    """


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(AX25Error):
    """A configuration value is outside the allowed range.

    AX.25 v2.2 defines allowed ranges for all settings. This error
    means a value you provided (or a default) is not allowed.

    Common causes:
      - modulo is not 8 or 128
      - window_size is out of range for the chosen modulo
      - t1_timeout is negative

    Example::

        # Raises ConfigurationError -- modulo must be 8 or 128
        config = AX25Config(modulo=16)
    """


# ---------------------------------------------------------------------------
# Resource errors
# ---------------------------------------------------------------------------

class ResourceExhaustionError(AX25Error):
    """An internal resource (like a buffer or window) ran out of space.

    This error means the library cannot process any more frames right now
    because it has run out of space. For example, the transmit window is
    full or a receive buffer is overflowing.
    """
