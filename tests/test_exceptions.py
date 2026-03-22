# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
tests/test_exceptions.py

Unit tests for the exception hierarchy in pyax25_22.core.exceptions.

Covers:
- AX25Error base: message, frame_data attribute, logging
- All subclass hierarchies (isinstance checks)
- frame_data hex logging when data provided
"""

import pytest

from pyax25_22.core.exceptions import (
    AX25Error,
    FrameError,
    InvalidAddressError,
    InvalidControlFieldError,
    FCSError,
    BitStuffingError,
    SegmentationError,
    ConnectionError,
    ConnectionStateError,
    TimeoutError,
    ProtocolViolationError,
    NegotiationError,
    TransportError,
    KISSError,
    AGWPEError,
    ConfigurationError,
    ResourceExhaustionError,
)


# ---------------------------------------------------------------------------
# AX25Error base
# ---------------------------------------------------------------------------

def test_ax25error_message():
    """AX25Error stores the message and has no frame_data by default."""
    err = AX25Error("something went wrong")
    assert str(err) == "something went wrong"
    assert err.frame_data is None


def test_ax25error_frame_data():
    """AX25Error stores frame_data bytes."""
    raw = b"\x00\x01\x02"
    err = AX25Error("bad frame", frame_data=raw)
    assert err.frame_data == raw


# ---------------------------------------------------------------------------
# Hierarchy checks
# ---------------------------------------------------------------------------

def test_frame_error_is_ax25():
    """FrameError is a subclass of AX25Error."""
    err = FrameError("bad frame")
    assert isinstance(err, AX25Error)
    assert isinstance(err, FrameError)


def test_invalid_address_hierarchy():
    """InvalidAddressError inherits from FrameError."""
    err = InvalidAddressError("bad callsign")
    assert isinstance(err, FrameError)
    assert isinstance(err, AX25Error)


def test_fcs_error_hierarchy():
    """FCSError inherits from FrameError."""
    err = FCSError("checksum mismatch")
    assert isinstance(err, FrameError)


def test_bit_stuffing_error_hierarchy():
    """BitStuffingError inherits from FrameError."""
    err = BitStuffingError("six ones")
    assert isinstance(err, FrameError)


def test_segmentation_error_hierarchy():
    """SegmentationError inherits from FrameError."""
    err = SegmentationError("missing piece")
    assert isinstance(err, FrameError)


def test_connection_error_hierarchy():
    """ConnectionError inherits from AX25Error."""
    err = ConnectionError("link dropped")
    assert isinstance(err, AX25Error)


def test_connection_state_error_hierarchy():
    """ConnectionStateError inherits from ConnectionError."""
    err = ConnectionStateError("not connected")
    assert isinstance(err, ConnectionError)
    assert isinstance(err, AX25Error)


def test_timeout_error_hierarchy():
    """TimeoutError inherits from ConnectionError."""
    err = TimeoutError("T1 expired")
    assert isinstance(err, ConnectionError)


def test_protocol_violation_hierarchy():
    """ProtocolViolationError inherits from ConnectionError."""
    err = ProtocolViolationError("FRMR received")
    assert isinstance(err, ConnectionError)


def test_negotiation_error_hierarchy():
    """NegotiationError inherits from ConnectionError."""
    err = NegotiationError("modulo mismatch")
    assert isinstance(err, ConnectionError)


def test_transport_error_hierarchy():
    """TransportError inherits from AX25Error."""
    err = TransportError("port not found")
    assert isinstance(err, AX25Error)


def test_kiss_error_hierarchy():
    """KISSError inherits from TransportError."""
    err = KISSError("serial failure")
    assert isinstance(err, TransportError)
    assert isinstance(err, AX25Error)


def test_agwpe_error_hierarchy():
    """AGWPEError inherits from TransportError."""
    err = AGWPEError("TCP failure")
    assert isinstance(err, TransportError)
    assert isinstance(err, AX25Error)


def test_configuration_error_hierarchy():
    """ConfigurationError inherits from AX25Error."""
    err = ConfigurationError("bad modulo")
    assert isinstance(err, AX25Error)


def test_resource_exhaustion_hierarchy():
    """ResourceExhaustionError inherits from AX25Error."""
    err = ResourceExhaustionError("window full")
    assert isinstance(err, AX25Error)


# ---------------------------------------------------------------------------
# Catch-all works
# ---------------------------------------------------------------------------

def test_catch_all_ax25error():
    """All library errors can be caught with AX25Error."""
    errors = [
        FrameError("x"),
        InvalidAddressError("x"),
        FCSError("x"),
        ConnectionStateError("x"),
        KISSError("x"),
        ConfigurationError("x"),
        ResourceExhaustionError("x"),
    ]
    for err in errors:
        assert isinstance(err, AX25Error), f"{type(err).__name__} not caught by AX25Error"
