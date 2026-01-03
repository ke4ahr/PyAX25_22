# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_framing.py

Comprehensive unit tests for framing module.

Covers:
- Address encoding/decoding and validation
- FCS calculation and verification
- Bit stuffing/destuffing
- Full frame round-trip (encode/decode)
- Error cases (invalid address, FCS mismatch, short frames)
"""

import pytest

from pyax25-22.core.framing import (
    AX25Frame,
    AX25Address,
    fcs_calc,
    verify_fcs,
)
from pyax25-22.core.exceptions import (
    InvalidAddressError,
    FCSError,
    FrameError,
)
from pyax25-22.core.config import DEFAULT_CONFIG_MOD8


@pytest.fixture
def basic_addresses():
    """Fixture for basic source and destination addresses."""
    dest = AX25Address("DEST", ssid=1)
    src = AX25Address("SRC", ssid=2)
    return dest, src


def test_address_validation():
    """Test AX25Address validation rules."""
    # Valid cases
    AX25Address("AB1CDE", ssid=0)
    AX25Address("KE4AHR", ssid=15)
    AX25Address("A", ssid=7)  # Minimum length

    # Invalid callsign length
    with pytest.raises(InvalidAddressError):
        AX25Address("", ssid=0)  # Empty

    with pytest.raises(InvalidAddressError):
        AX25Address("TOOLONG", ssid=0)  # 7 chars

    # Invalid SSID range
    with pytest.raises(InvalidAddressError):
        AX25Address("TEST", ssid=-1)

    with pytest.raises(InvalidAddressError):
        AX25Address("TEST", ssid=16)


def test_address_encoding(basic_addresses):
    """Test AX25Address encoding."""
    dest, src = basic_addresses

    # Basic encoding
    assert dest.encode(last=False) == bytes([ord('D')<<1, ord('E')<<1, ord('S')<<1, ord('T')<<1, ord(' ')<<1, ord(' ')<<1, 0x60 | (1<<1)])
    assert src.encode(last=True) == bytes([ord('S')<<1, ord('R')<<1, ord('C')<<1, ord(' ')<<1, ord(' ')<<1, ord(' ')<<1, 0x61 | (2<<1)])

    # With C-bit and H-bit
    addr = AX25Address("TEST", ssid=3, c_bit=True, h_bit=True)
    encoded = addr.encode(last=True)
    assert encoded[-1] & 0xE0 == 0xE0  # C and H bits set


def test_address_decoding():
    """Test AX25Address decoding."""
    # Basic decode
    data = bytes([ord('D')<<1, ord('E')<<1, ord('S')<<1, ord('T')<<1, ord(' ')<<1, ord(' ')<<1, 0x61 | (1<<1)])
    addr, is_last = AX25Address.decode(data)
    assert addr.callsign == "DEST"
    assert addr.ssid == 1
    assert is_last == True

    # With bits
    data_with_bits = bytes([ord('T')<<1, ord('E')<<1, ord('S')<<1, ord('T')<<1, ord(' ')<<1, ord(' ')<<1, 0xE0 | (3<<1)])
    addr, _ = AX25Address.decode(data_with_bits)
    assert addr.c_bit == True
    assert addr.h_bit == True

    # Invalid length
    with pytest.raises(InvalidAddressError):
        AX25Address.decode(bytes(6))


def test_fcs_calculation():
    """Test FCS calculation and verification."""
    test_data = b'ABCDEF'
    addr_payload = b'DEST  \x63SRC   \x03\xF0' + test_data
    fcs = fcs_calc(addr_payload)
    assert isinstance(fcs, int)
    assert 0 <= fcs <= 0xFFFF

    # Verify
    assert verify_fcs(addr_payload, fcs)

    # Invalid
    assert not verify_fcs(addr_payload + b'\x00', fcs)


def test_bit_stuffing():
    """Test bit stuffing and destuffing round-trip."""
    test_data = bytes.fromhex("7E 7E 7E")  # Multiple flags - should stuff
    stuffed = AX25Frame._bit_stuff(test_data)
    destuffed = AX25Frame._bit_destuff(stuffed)
    assert destuffed == test_data

    # 5 ones
    five_ones = bytes([0x1F])  # 00011111
    stuffed = AX25Frame._bit_stuff(five_ones)
    assert len(stuffed) > len(five_ones)  # Extra 0 inserted

    # Error in destuff (invalid sequence)
    invalid_stuffed = bytes([0xFF])  # 11111111 - invalid (6 ones)
    with pytest.raises(BitStuffingError):  # Assume we add this in exceptions
        AX25Frame._bit_destuff(invalid_stuffed)


def test_frame_roundtrip(basic_addresses):
    """Test full frame encode/decode cycle."""
    dest, src = basic_addresses

    # Basic UI frame
    frame = AX25Frame(
        destination=dest,
        source=src,
        control=0x03,  # UI
        pid=0xF0,
        info=b"Hello PyAX25_22",
    )
    encoded = frame.encode()
    assert encoded[0] == 0x7E
    assert encoded[-1] == 0x7E

    decoded = AX25Frame.decode(encoded)
    assert decoded.destination.callsign == "DEST"
    assert decoded.destination.ssid == 1
    assert decoded.source.callsign == "SRC"
    assert decoded.source.ssid == 2
    assert decoded.control == 0x03
    assert decoded.pid == 0xF0
    assert decoded.info == b"Hello PyAX25_22"

    # With digipeaters
    digi1 = AX25Address("DIGI1", ssid=0, h_bit=True)
    frame.digipeaters = [digi1]
    encoded_digi = frame.encode()
    decoded_digi = AX25Frame.decode(encoded_digi)
    assert len(decoded_digi.digipeaters) == 1
    assert decoded_digi.digipeaters[0].h_bit == True

    # Invalid FCS
    bad_encoded = encoded[:-3] + b'\x00\x00' + encoded[-1:]
    with pytest.raises(FCSError):
        AX25Frame.decode(bad_encoded)

    # Too short
    with pytest.raises(FrameError):
        AX25Frame.decode(bytes(10))


def test_i_frame():
    """Test I-frame specific encoding."""
    # Mod 8 I-frame N(S)=3, N(R)=5, P=1
    control = (3 << 1) | (5 << 5) | 0x10
    frame = AX25Frame(control=control, pid=0xF0, config=DEFAULT_CONFIG_MOD8)
    encoded = frame.encode()

    decoded = AX25Frame.decode(encoded)
    assert decoded.control == control


def test_extended_mod128():
    """Test modulo 128 extended control field."""
    config_mod128 = AX25Config(modulo=128)
    # Extended I-frame N(S)=100, N(R)=120, P=1
    control = (100 << 1) | (120 << 9) | 0x100  # 16-bit control
    frame = AX25Frame(control=control, pid=0xF0, config=config_mod128)
    encoded = frame.encode()

    decoded = AX25Frame.decode(encoded)
    assert decoded.control == control
