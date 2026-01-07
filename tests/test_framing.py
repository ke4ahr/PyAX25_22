# tests/test_framing.py
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_framing.py - VERSION 0.5.31

Fixed test data and removed bit stuffing tests (handled by hardware).
"""

import pytest

from pyax25_22.core.framing import (
    AX25Frame,
    AX25Address,
    fcs_calc,
    verify_fcs,
)
from pyax25_22.core.exceptions import (
    InvalidAddressError,
    FCSError,
    FrameError,
)
from pyax25_22.core.config import DEFAULT_CONFIG_MOD8


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
    AX25Address("A", ssid=7)

    # Invalid callsign length
    with pytest.raises(InvalidAddressError):
        AX25Address("", ssid=0)

    with pytest.raises(InvalidAddressError):
        AX25Address("TOOLONG", ssid=0)

    # Invalid SSID range
    with pytest.raises(InvalidAddressError):
        AX25Address("TEST", ssid=-1)

    with pytest.raises(InvalidAddressError):
        AX25Address("TEST", ssid=16)


def test_address_encoding(basic_addresses):
    """Test AX25Address encoding."""
    dest, src = basic_addresses

    # With reserved bit set (0x80)
    assert dest.encode(last=False)[-1] == 0x82  # SSID=1, reserved=1, ext=0
    assert src.encode(last=True)[-1] == 0x85     # SSID=2, reserved=1, ext=1

    # With C-bit and H-bit
    addr = AX25Address("TEST", ssid=3, c_bit=True, h_bit=True)
    encoded = addr.encode(last=True)
    assert encoded[-1] == 0xE7  # All bits set correctly


def test_address_decoding():
    """Test AX25Address decoding."""
    # Basic decode - DEST-1
    data = b'\x88\x8a\xa6\xa8@@\x82'
    addr, is_last = AX25Address.decode(data)
    assert addr.callsign == "DEST"
    assert addr.ssid == 1
    assert addr.c_bit == False
    assert addr.h_bit == False
    assert is_last == False

    # TEST-3 with C=1, H=1 - CORRECTED encoding
    data = b'\xa8\x8a\xa6\xa8@@\xE7'  # TEST encoded correctly
    addr, is_last = AX25Address.decode(data)
    assert addr.callsign == "TEST"
    assert addr.ssid == 3
    assert addr.c_bit == True
    assert addr.h_bit == True
    assert is_last == True

    # Invalid length
    with pytest.raises(InvalidAddressError):
        AX25Address.decode(bytes(6))


def test_fcs_calculation():
    """Test FCS calculation and verification."""
    test_data = b'ABCDEF'
    fcs = fcs_calc(test_data)
    assert isinstance(fcs, int)
    assert 0 <= fcs <= 0xFFFF

    # Verify
    assert verify_fcs(test_data, fcs)

    # Invalid
    assert not verify_fcs(test_data + b'\x00', fcs)


def test_bit_stuffing():
    """
    Test frame round-trip (bit stuffing handled by hardware).
    
    NOTE: Bit stuffing is NOT done in software. It's handled by the
    HDLC hardware (TNC). Software frames are transmitted as-is.
    """
    dest = AX25Address("TEST")
    src = AX25Address("TEST")

    # Frame with flag bytes in info field - NO stuffing applied in software
    frame = AX25Frame(
        destination=dest,
        source=src,
        control=0x03,
        pid=0xF0,
        info=bytes([0x7E, 0x7E, 0x7E]),
        config=DEFAULT_CONFIG_MOD8
    )

    encoded = frame.encode()
    decoded = AX25Frame.decode(encoded, DEFAULT_CONFIG_MOD8)

    # Should round-trip correctly
    assert decoded.info == bytes([0x7E, 0x7E, 0x7E])
    assert decoded.destination.callsign == "TEST"
    assert decoded.source.callsign == "TEST"


def test_frame_roundtrip(basic_addresses):
    """Test full frame encode/decode cycle."""
    dest, src = basic_addresses

    frame = AX25Frame(
        destination=dest,
        source=src,
        control=0x03,
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
    control = (3 << 1) | (5 << 5) | 0x10
    dest = AX25Address("DEST")
    src = AX25Address("SRC")
    frame = AX25Frame(
        destination=dest,
        source=src,
        control=control,
        pid=0xF0,
        config=DEFAULT_CONFIG_MOD8
    )
    encoded = frame.encode()

    decoded = AX25Frame.decode(encoded)
    assert decoded.control == control


def test_extended_mod128():
    """Test modulo 128 extended control field."""
    from pyax25_22.core.config import AX25Config
    config_mod128 = AX25Config(modulo=128)
    control = (100 << 1) | (120 << 9) | 0x100
    dest = AX25Address("DEST")
    src = AX25Address("SRC")
    frame = AX25Frame(
        destination=dest,
        source=src,
        control=control,
        pid=0xF0,
        config=config_mod128
    )
    encoded = frame.encode()

    decoded = AX25Frame.decode(encoded, config=config_mod128)
    assert decoded.control == control
