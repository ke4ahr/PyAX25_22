# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
tests/test_kiss_base.py

Unit tests for KISSBase framing (transport-agnostic layer).

Covers:
- _stuff(): frame construction with FEND/FESC escaping
- _destuff(): escape removal from received bytes
- _process_byte(): state machine for frame assembly
- _dispatch(): callback invocation
- send(): validates command, calls write() with stuffed frame
"""

import pytest
from unittest.mock import MagicMock

from pyax25_22.interfaces.kiss.base import KISSBase
from pyax25_22.interfaces.kiss.constants import (
    FEND, FESC, TFEND, TFESC, CMD_DATA, CMD_TXDELAY, CMD_EXIT,
)
from pyax25_22.interfaces.kiss.exceptions import KISSFrameError


# ---------------------------------------------------------------------------
# Concrete subclass for testing (implements abstract write())
# ---------------------------------------------------------------------------

class MockKISS(KISSBase):
    """Testable KISSBase subclass with in-memory write buffer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.written: list = []

    def write(self, data: bytes) -> None:
        self.written.append(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tnc():
    return MockKISS()


@pytest.fixture
def callback_tnc():
    received = []
    def on_frame(cmd, payload):
        received.append((cmd, payload))
    t = MockKISS(on_frame=on_frame)
    t.received = received
    return t


# ---------------------------------------------------------------------------
# _stuff() tests
# ---------------------------------------------------------------------------

def test_stuff_basic(tnc):
    """Basic payload with no special bytes is wrapped in FENDs."""
    result = tnc._stuff(b"\x01\x02\x03", CMD_DATA)
    assert result[0] == FEND
    assert result[-1] == FEND
    assert result[1] == CMD_DATA
    assert result[2:5] == b"\x01\x02\x03"


def test_stuff_escapes_fend(tnc):
    """FEND byte in payload is replaced with FESC TFEND."""
    result = tnc._stuff(bytes([FEND]), CMD_DATA)
    body = result[2:-1]  # strip leading FEND+cmd and trailing FEND
    assert body == bytes([FESC, TFEND])


def test_stuff_escapes_fesc(tnc):
    """FESC byte in payload is replaced with FESC TFESC."""
    result = tnc._stuff(bytes([FESC]), CMD_DATA)
    body = result[2:-1]
    assert body == bytes([FESC, TFESC])


def test_stuff_mixed_payload(tnc):
    """Mixed payload with FEND, FESC, and normal bytes."""
    payload = bytes([0x01, FEND, 0x02, FESC, 0x03])
    result = tnc._stuff(payload, CMD_DATA)
    body = result[2:-1]
    expected = bytes([
        0x01,
        FESC, TFEND,   # FEND -> FESC TFEND
        0x02,
        FESC, TFESC,   # FESC -> FESC TFESC
        0x03,
    ])
    assert body == expected


def test_stuff_empty_payload(tnc):
    """Empty payload produces FEND + cmd + FEND (3 bytes)."""
    result = tnc._stuff(b"", CMD_DATA)
    assert result == bytes([FEND, CMD_DATA, FEND])


# ---------------------------------------------------------------------------
# _destuff() tests
# ---------------------------------------------------------------------------

def test_destuff_basic(tnc):
    """Normal bytes pass through unchanged."""
    data = bytearray([0x01, 0x02, 0x03])
    assert tnc._destuff(data) == b"\x01\x02\x03"


def test_destuff_fesc_tfend(tnc):
    """FESC TFEND -> FEND."""
    data = bytearray([FESC, TFEND])
    assert tnc._destuff(data) == bytes([FEND])


def test_destuff_fesc_tfesc(tnc):
    """FESC TFESC -> FESC."""
    data = bytearray([FESC, TFESC])
    assert tnc._destuff(data) == bytes([FESC])


def test_destuff_mixed(tnc):
    """Mixed escapes and normal bytes."""
    data = bytearray([0x01, FESC, TFEND, 0x02, FESC, TFESC])
    result = tnc._destuff(data)
    assert result == bytes([0x01, FEND, 0x02, FESC])


def test_destuff_truncated_escape_raises(tnc):
    """FESC at end of data with no following byte raises KISSFrameError."""
    data = bytearray([0x01, FESC])  # truncated
    with pytest.raises(KISSFrameError):
        tnc._destuff(data)


def test_destuff_roundtrip(tnc):
    """stuff -> destuff round-trip recovers original payload."""
    original = bytes([0x00, FEND, 0x01, FESC, 0x02, 0xFF])
    stuffed = tnc._stuff(original, CMD_DATA)
    # Strip FEND+cmd+FEND framing
    body = bytearray(stuffed[2:-1])
    recovered = tnc._destuff(body)
    assert recovered == original


# ---------------------------------------------------------------------------
# _process_byte() / frame assembly tests
# ---------------------------------------------------------------------------

def test_process_byte_assembles_frame(callback_tnc):
    """Feeding a complete frame triggers the on_frame callback."""
    payload = b"\xAA\xBB\xCC"
    frame = bytes([FEND, CMD_DATA]) + payload + bytes([FEND])
    for b in frame:
        callback_tnc._process_byte(b)
    assert len(callback_tnc.received) == 1
    cmd, data = callback_tnc.received[0]
    assert cmd == CMD_DATA
    assert data == payload


def test_process_byte_ignores_empty_frame(callback_tnc):
    """FEND FEND (empty frame body) does not trigger callback."""
    for b in bytes([FEND, FEND]):
        callback_tnc._process_byte(b)
    assert len(callback_tnc.received) == 0


def test_process_byte_multiple_frames(callback_tnc):
    """Two back-to-back frames both trigger the callback."""
    frame1 = bytes([FEND, CMD_DATA, 0x11, FEND])
    frame2 = bytes([FEND, CMD_TXDELAY, 0x22, FEND])
    for b in frame1 + frame2:
        callback_tnc._process_byte(b)
    assert len(callback_tnc.received) == 2
    assert callback_tnc.received[0] == (CMD_DATA, b"\x11")
    assert callback_tnc.received[1] == (CMD_TXDELAY, b"\x22")


def test_process_byte_destuffs_on_receipt(callback_tnc):
    """FESC sequences in received frames are resolved before callback."""
    # Frame containing a FEND byte in payload (escaped as FESC TFEND)
    inner = bytes([FESC, TFEND])  # represents FEND
    frame = bytes([FEND, CMD_DATA]) + inner + bytes([FEND])
    for b in frame:
        callback_tnc._process_byte(b)
    assert callback_tnc.received[0][1] == bytes([FEND])


# ---------------------------------------------------------------------------
# send() tests
# ---------------------------------------------------------------------------

def test_send_calls_write(tnc):
    """send() calls write() exactly once with a properly framed packet."""
    tnc.send(b"\x01\x02", CMD_DATA)
    assert len(tnc.written) == 1
    pkt = tnc.written[0]
    assert pkt[0] == FEND
    assert pkt[-1] == FEND
    assert pkt[1] == CMD_DATA


def test_send_invalid_cmd_raises(tnc):
    """send() raises ValueError for an invalid KISS command."""
    with pytest.raises(ValueError):
        tnc.send(b"\x00", cmd=0x07)   # 0x07 not in valid_cmds


def test_send_exit_cmd_is_valid(tnc):
    """CMD_EXIT (0xFF) is a valid KISS command."""
    tnc.send(b"", cmd=CMD_EXIT)
    assert len(tnc.written) == 1


def test_no_callback_does_not_raise(tnc):
    """Received frame with no callback set does not raise."""
    frame = bytes([FEND, CMD_DATA, 0x42, FEND])
    for b in frame:
        tnc._process_byte(b)   # Should not raise


def test_callback_exception_is_caught(callback_tnc):
    """Exception in on_frame callback is caught and does not propagate."""
    callback_tnc.on_frame = lambda cmd, data: 1 / 0  # ZeroDivisionError
    frame = bytes([FEND, CMD_DATA, 0x01, FEND])
    for b in frame:
        callback_tnc._process_byte(b)   # Should not raise
