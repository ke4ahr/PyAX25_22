# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
tests/test_validation.py

Unit tests for validate_frame_structure and full_validation.

Covers:
- I-frame: must have PID, info within N1
- S-frame: must not have PID or info
- UI-frame: must have PID
- Other U-frames: must not have PID
- Digipeater count limit (max 8)
"""

import pytest

from pyax25_22.core.framing import AX25Frame, AX25Address
from pyax25_22.core.validation import validate_frame_structure, full_validation
from pyax25_22.core.config import AX25Config, DEFAULT_CONFIG_MOD8
from pyax25_22.core.exceptions import (
    InvalidAddressError,
    InvalidControlFieldError,
    FrameError,
)


def make_frame(control, pid=None, info=b"", digis=None, config=None):
    """Helper to create a test frame."""
    dest = AX25Address("W1AW")
    src = AX25Address("KE4AHR")
    return AX25Frame(
        destination=dest,
        source=src,
        digipeaters=digis or [],
        control=control,
        pid=pid,
        info=info,
        config=config or DEFAULT_CONFIG_MOD8,
    )


# ---------------------------------------------------------------------------
# I-frame tests (bit 0 = 0)
# ---------------------------------------------------------------------------

def test_i_frame_valid():
    """Valid I-frame passes validation."""
    frame = make_frame(control=0x00, pid=0xF0, info=b"hello")
    validate_frame_structure(frame)  # Should not raise


def test_i_frame_no_pid_raises():
    """I-frame without PID raises InvalidControlFieldError."""
    frame = make_frame(control=0x00, pid=None, info=b"hello")
    with pytest.raises(InvalidControlFieldError):
        validate_frame_structure(frame)


def test_i_frame_info_too_long_raises():
    """I-frame with info longer than N1 raises FrameError."""
    cfg = AX25Config(max_frame=10)
    frame = make_frame(control=0x00, pid=0xF0, info=b"X" * 11, config=cfg)
    with pytest.raises(FrameError):
        validate_frame_structure(frame)


def test_i_frame_info_exactly_n1():
    """I-frame with info exactly N1 bytes passes."""
    cfg = AX25Config(max_frame=10)
    frame = make_frame(control=0x00, pid=0xF0, info=b"X" * 10, config=cfg)
    validate_frame_structure(frame)


# ---------------------------------------------------------------------------
# S-frame tests (bits 1-0 = 01)
# ---------------------------------------------------------------------------

def test_s_frame_rr_valid():
    """RR supervisory frame with no PID/info passes."""
    frame = make_frame(control=0x01)  # RR
    validate_frame_structure(frame)


def test_s_frame_with_pid_raises():
    """S-frame with PID raises InvalidControlFieldError."""
    frame = make_frame(control=0x01, pid=0xF0)
    with pytest.raises(InvalidControlFieldError):
        validate_frame_structure(frame)


def test_s_frame_with_info_raises():
    """S-frame with info field raises FrameError."""
    frame = make_frame(control=0x01, info=b"bad")
    with pytest.raises(FrameError):
        validate_frame_structure(frame)


# ---------------------------------------------------------------------------
# UI-frame tests (control 0x03)
# ---------------------------------------------------------------------------

def test_ui_frame_valid():
    """UI frame with PID passes."""
    frame = make_frame(control=0x03, pid=0xF0, info=b"APRS")
    validate_frame_structure(frame)


def test_ui_frame_no_pid_raises():
    """UI frame without PID raises InvalidControlFieldError."""
    frame = make_frame(control=0x03, pid=None, info=b"data")
    with pytest.raises(InvalidControlFieldError):
        validate_frame_structure(frame)


# ---------------------------------------------------------------------------
# Other U-frame tests
# ---------------------------------------------------------------------------

def test_sabm_no_pid_valid():
    """SABM U-frame with no PID passes."""
    frame = make_frame(control=0x2F)  # SABM with P=1
    validate_frame_structure(frame)


def test_sabm_with_pid_raises():
    """SABM with PID raises InvalidControlFieldError."""
    frame = make_frame(control=0x2F, pid=0xF0)
    with pytest.raises(InvalidControlFieldError):
        validate_frame_structure(frame)


# ---------------------------------------------------------------------------
# Digipeater count
# ---------------------------------------------------------------------------

def test_too_many_digipeaters_raises():
    """More than 8 digipeaters raises InvalidAddressError."""
    digis = [AX25Address("DIGI%d" % i) for i in range(9)]
    frame = make_frame(control=0x03, pid=0xF0, digis=digis)
    with pytest.raises(InvalidAddressError):
        validate_frame_structure(frame)


def test_eight_digipeaters_valid():
    """Exactly 8 digipeaters passes."""
    digis = [AX25Address("RELAY%d" % i) for i in range(8)]
    frame = make_frame(control=0x03, pid=0xF0, digis=digis)
    validate_frame_structure(frame)


# ---------------------------------------------------------------------------
# full_validation
# ---------------------------------------------------------------------------

def test_full_validation_delegates():
    """full_validation calls the same checks as validate_frame_structure."""
    frame = make_frame(control=0x03, pid=0xF0, info=b"test")
    full_validation(frame)   # Should not raise

    bad_frame = make_frame(control=0x01, pid=0xF0)  # S-frame with PID
    with pytest.raises(InvalidControlFieldError):
        full_validation(bad_frame)
