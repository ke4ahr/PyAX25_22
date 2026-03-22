# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_xkiss.py

Unit tests for XKISSMixin (multi-drop, polling, checksum, DIGIPEAT) and
SMACKMixin (CRC-16 auto-switch).

Uses a concrete MockXKISS that injects bytes directly (no serial port needed).

Covers:
- Port decoding from command byte high nibble
- XOR checksum verification
- DIGIPEAT: enable/disable/set/get per port
- DIGIPEAT: frame relay when first un-repeated digipeater matches our_calls
- SMACK: CRC-16 computation
- SMACK: auto-switch on first valid CRC frame
- SMACK: corrupt CRC drops frame
"""

import pytest
from unittest.mock import MagicMock

from pyax25_22.interfaces.kiss.xkiss import XKISSMixin, _digipeat_frame
from pyax25_22.interfaces.kiss.smack import SMACKMixin
from pyax25_22.interfaces.kiss.base import KISSBase
from pyax25_22.interfaces.kiss.constants import (
    FEND, FESC, TFEND, TFESC, CMD_DATA, CMD_POLL,
    PORT_MASK, CMD_MASK, SMACK_FLAG,
)
from pyax25_22.interfaces.kiss.exceptions import KISSChecksumError


# ---------------------------------------------------------------------------
# Concrete mock transport that drives both XKISSMixin and SMACKMixin
# ---------------------------------------------------------------------------

class MockTransport(KISSBase):
    """In-memory write buffer, no I/O."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.written: list = []

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def inject(self, cmd: int, payload: bytes) -> None:
        """Simulate receiving a raw KISS frame (cmd + payload, no escaping)."""
        frame = bytes([FEND, cmd]) + payload + bytes([FEND])
        for b in frame:
            self._process_byte(b)


class MockXKISS(XKISSMixin, MockTransport):
    """XKISSMixin over MockTransport for testing."""
    pass


class MockSMACK(SMACKMixin, MockXKISS):
    """SMACKMixin over MockXKISS for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_smack_raw()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def xk():
    received = []
    def on_xframe(addr, port, data):
        received.append((addr, port, data))
    inst = MockXKISS(on_xframe=on_xframe)
    inst.received = received
    return inst


@pytest.fixture
def xk_checksum():
    received = []
    def on_xframe(addr, port, data):
        received.append((addr, port, data))
    inst = MockXKISS(checksum_mode=True, on_xframe=on_xframe)
    inst.received = received
    return inst


@pytest.fixture
def smack():
    received = []
    def on_smack_frame(addr, port, data):
        received.append((addr, port, data))
    inst = MockSMACK(on_smack_frame=on_smack_frame)
    inst.received = received
    return inst


# ---------------------------------------------------------------------------
# Port decoding
# ---------------------------------------------------------------------------

def test_port_decoded_from_high_nibble(xk):
    """High nibble of command byte is decoded as port number."""
    port = 3
    cmd = (port << 4) | CMD_DATA
    xk.inject(cmd, b"\xAA\xBB")
    assert len(xk.received) == 1
    _, recv_port, data = xk.received[0]
    assert recv_port == port
    assert data == b"\xAA\xBB"


def test_port_zero(xk):
    """Port 0 is the default (low command byte)."""
    xk.inject(CMD_DATA, b"\x01\x02")
    assert xk.received[0][1] == 0


def test_port_fifteen(xk):
    """Port 15 (high nibble = 0xF) is decoded correctly."""
    cmd = (15 << 4) | CMD_DATA
    xk.inject(cmd, b"\xFF")
    assert xk.received[0][1] == 15


# ---------------------------------------------------------------------------
# XOR checksum mode
# ---------------------------------------------------------------------------

def test_xor_checksum_valid(xk_checksum):
    """Valid XOR checksum -- frame passes to callback."""
    payload = b"\x01\x02\x03"
    cmd = CMD_DATA
    checksum = cmd
    for b in payload:
        checksum ^= b
    # Append checksum byte to payload
    xk_checksum.inject(cmd, payload + bytes([checksum & 0xFF]))
    assert len(xk_checksum.received) == 1
    # Checksum byte is stripped
    assert xk_checksum.received[0][2] == payload


def test_xor_checksum_invalid_drops_frame(xk_checksum):
    """Invalid XOR checksum -- frame is dropped silently."""
    payload = b"\x01\x02\x03\xFF"  # 0xFF is wrong checksum
    xk_checksum.inject(CMD_DATA, payload)
    assert len(xk_checksum.received) == 0


def test_xor_checksum_too_short_drops_frame(xk_checksum):
    """Frame too short for checksum (empty payload) -- dropped."""
    xk_checksum.inject(CMD_DATA, b"")
    assert len(xk_checksum.received) == 0


# ---------------------------------------------------------------------------
# DIGIPEAT feature
# ---------------------------------------------------------------------------

def test_enable_disable_digipeat():
    """enable_digipeat and disable_digipeat toggle the port set."""
    xk = MockXKISS()
    assert not xk.get_digipeat(0)
    xk.enable_digipeat(0)
    assert xk.get_digipeat(0)
    xk.disable_digipeat(0)
    assert not xk.get_digipeat(0)


def test_set_digipeat_on_off():
    """set_digipeat(port, True/False) is equivalent to enable/disable."""
    xk = MockXKISS()
    xk.set_digipeat(2, True)
    assert xk.get_digipeat(2)
    xk.set_digipeat(2, False)
    assert not xk.get_digipeat(2)


def test_enable_digipeat_out_of_range():
    """enable_digipeat raises ValueError for port > 15."""
    xk = MockXKISS()
    with pytest.raises(ValueError):
        xk.enable_digipeat(16)


def test_disable_digipeat_out_of_range():
    """disable_digipeat raises ValueError for port < 0."""
    xk = MockXKISS()
    with pytest.raises(ValueError):
        xk.disable_digipeat(-1)


def test_digipeat_on_port_no_our_calls(xk):
    """DIGIPEAT enabled but our_calls empty -- no relay."""
    xk.enable_digipeat(0)
    # Build a minimal AX.25 address field: dest(7) + src(7) + digi(7, H-bit clear)
    dest = bytes([0x00] * 6 + [0x00])    # dest, not last
    src  = bytes([0x00] * 6 + [0x00])    # src, not last
    digi = bytes([0x00] * 6 + [0x01])    # digi, last, H-bit clear
    xk.inject(CMD_DATA, dest + src + digi)
    # No relay sent because our_calls is empty
    sent_data_frames = [
        p for p in xk.written if len(p) > 2 and p[1] == CMD_DATA
    ]
    assert len(sent_data_frames) == 0


# ---------------------------------------------------------------------------
# _digipeat_frame() helper
# ---------------------------------------------------------------------------

def test_digipeat_frame_too_short():
    """Payload shorter than 14 bytes returns None (can't parse addresses)."""
    assert _digipeat_frame(b"\x00" * 10, {"KE4AHR"}) is None


def test_digipeat_frame_no_digipeater():
    """Frame with no digipeater path (last bit set on src) returns None."""
    # dest (7 bytes, not last), src (7 bytes, last bit set, no more)
    dest = bytes([ord('A') << 1] * 6 + [0x00])  # not last
    src  = bytes([ord('K') << 1] * 6 + [0x01])  # last (no digipeaters)
    frame = dest + src
    assert _digipeat_frame(frame, {"KE4AHR"}) is None


def test_digipeat_frame_already_relayed():
    """Frame with H-bit set on first digipeater returns None."""
    dest = bytes([0x00] * 6 + [0x00])          # not last
    src  = bytes([0x00] * 6 + [0x00])          # not last
    digi = bytes([0x00] * 6 + [0x81])          # last + H-bit set
    frame = dest + src + digi
    assert _digipeat_frame(frame, {"KE4AHR"}) is None


# ---------------------------------------------------------------------------
# SMACK CRC-16
# ---------------------------------------------------------------------------

def test_smack_crc16_known_value(smack):
    """SMACK CRC-16 of empty string is 0x0000 (poly 0x8005, init 0)."""
    crc = smack._crc16(b"")
    assert crc == b"\x00\x00"


def test_smack_crc16_is_two_bytes(smack):
    """CRC result is always 2 bytes."""
    crc = smack._crc16(b"Hello")
    assert len(crc) == 2


def test_smack_crc16_different_data(smack):
    """Different data produces different CRC."""
    crc1 = smack._crc16(b"\x01")
    crc2 = smack._crc16(b"\x02")
    assert crc1 != crc2


# ---------------------------------------------------------------------------
# SMACK auto-switch
# ---------------------------------------------------------------------------

def test_smack_starts_not_enabled(smack):
    """SMACK is not enabled at startup."""
    assert not smack.smack_enabled


def test_smack_send_plain_before_switch(smack):
    """Before auto-switch, send() transmits without CRC."""
    smack.send(b"\x01\x02")
    assert len(smack.written) == 1
    # The packet should NOT have SMACK_FLAG set in the command byte
    pkt = smack.written[0]
    cmd_byte = pkt[1]
    assert not (cmd_byte & SMACK_FLAG)


def test_smack_corrupt_crc_drops_frame(smack):
    """Frame with SMACK flag but wrong CRC is dropped."""
    cmd = CMD_DATA | SMACK_FLAG
    payload = b"\x01\x02\xFF\xFF"  # bad CRC bytes at end
    smack.inject(cmd, payload)
    assert len(smack.received) == 0


def test_smack_valid_crc_delivers_frame(smack):
    """Frame with SMACK flag and correct CRC is delivered."""
    cmd = CMD_DATA | SMACK_FLAG
    body = b"\xAA\xBB"
    crc = smack._crc16(bytes([cmd]) + body)
    smack.inject(cmd, body + crc)
    assert len(smack.received) == 1
    assert smack.received[0][2] == body


def test_smack_enables_after_valid_crc(smack):
    """After receiving a valid CRC frame, smack_enabled becomes True."""
    cmd = CMD_DATA | SMACK_FLAG
    body = b"\xDE\xAD"
    crc = smack._crc16(bytes([cmd]) + body)
    smack.inject(cmd, body + crc)
    assert smack.smack_enabled


def test_smack_reset_disables(smack):
    """reset_smack() clears the enabled flag."""
    smack._smack_enabled = True
    smack.reset_smack()
    assert not smack.smack_enabled
