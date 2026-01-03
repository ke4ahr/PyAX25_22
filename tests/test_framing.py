# tests/test_framing.py
"""
Comprehensive Tests for AX.25 Framing

Covers:
- Address encoding/decoding
- Frame construction/parsing
- FCS calculation
- Bit stuffing/destuffing

License: LGPLv3.0
Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""

import pytest
import struct
from pyax25_22.core.framing import (
    AX25Frame,
    AX25Address,
    fcs_calc,
    bit_stuff,
    bit_destuff,
    FrameType,
    PID,
    FLAG
)

@pytest.fixture
def valid_callsigns():
    return [
        ("TEST", 0),
        ("LONGER", 15),
        ("SHORT", 7),
        ("A", 1)
    ]

class TestAX25Address:
    def test_valid_address_encoding(self, valid_callsigns):
        for call, ssid in valid_callsigns:
            addr = AX25Address(call, ssid)
            encoded = addr.encoded()
            assert len(encoded) == 7
            assert encoded[:6] == bytes(call.ljust(6), 'ascii').translate(bytes.maketrans(b' ', b'\x00'))
            assert (encoded[6] >> 1) & 0x0F == ssid

    def test_address_decoding(self):
        encoded = b'TEST  \x60'  # TEST-0
        addr, last = AX25Address.from_bytes(encoded)
        assert addr.callsign.strip() == "TEST"
        assert addr.ssid == 0
        assert not last

    def test_invalid_ssid(self):
        with pytest.raises(ValueError):
            AX25Address("TEST", -1)
        with pytest.raises(ValueError):
            AX25Address("TEST", 16)

    def test_last_address_bit(self):
        addr = AX25Address("TEST", 0)
        encoded = addr.encoded(last=True)
        assert encoded[6] & 0x01 == 0x01

class TestAX25Frame:
    @pytest.fixture
    def sample_ui_frame(self):
        return AX25Frame(
            dest=AX25Address("DEST", 1),
            src=AX25Address("SRC", 2),
            pid=PID.NO_LAYER3
        )

    def test_ui_frame_encoding(self, sample_ui_frame):
        payload = b"Hello AX.25!"
        frame = sample_ui_frame.encode_ui(payload)
        
        assert frame.startswith(bytes([FLAG]))
        assert frame.endswith(bytes([FLAG]))
        
        stripped = frame[1:-1]
        destuffed = bit_destuff(stripped)
        
        # Verify addresses
        assert destuffed[:7] == AX25Address("DEST", 1).encoded()
        assert destuffed[7:14] == AX25Address("SRC", 2).encoded(last=True)
        
        # Verify control/PID
        assert destuffed[14] == 0x03  # UI frame
        assert destuffed[15] == PID.NO_LAYER3
        
        # Verify payload
        assert destuffed[16:-2] == payload
        
        # Verify FCS
        calculated_fcs = fcs_calc(destuffed[:-2])
        stored_fcs = struct.unpack('<H', destuffed[-2:])[0]
        assert calculated_fcs == stored_fcs

    def test_i_frame_encoding(self):
        frame = AX25Frame(
            dest=AX25Address("DEST", 1),
            src=AX25Address("SRC", 2)
        )
        encoded = frame.encode_i(b"data", ns=3, nr=5, poll=True)
        destuffed = bit_destuff(encoded[1:-1])
        
        control = destuffed[14]
        assert (control >> 1) & 0x07 == 3  # NS
        assert (control >> 5) & 0x07 == 5  # NR
        assert (control >> 4) & 0x01 == 1  # Poll

    def test_sabm_frame(self):
        frame = AX25Frame(
            dest=AX25Address("DEST", 1),
            src=AX25Address("SRC", 2)
        )
        encoded = frame.encode_sabm(poll=True)
        destuffed = bit_destuff(encoded[1:-1])
        assert destuffed[14] == 0x2F | (1 << 4)

    def test_invalid_frame_parsing(self):
        # Too short
        with pytest.raises(ValueError):
            AX25Frame.from_bytes(bytes([FLAG, FLAG]))
            
        # Bad FCS
        bad_frame = AX25Frame(
            dest=AX25Address("A", 0),
            src=AX25Address("B", 0)
        ).encode_ui(b"test")[:-2] + b'\x00\x00'
        with pytest.raises(ValueError):
            AX25Frame.from_bytes(bad_frame)

class TestFCS:
    def test_known_fcs_values(self):
        # Test vectors from AX.25 spec
        assert fcs_calc(b"\x01") == 0x0E1F
        assert fcs_calc(b"\x03\xF0") == 0x8E72
        assert fcs_calc(b"C") == 0x14E1

class TestBitStuffing:
    @pytest.mark.parametrize("data,expected", [
        (b"\x7E", b"\xDB\xDC"),
        (b"\xDB", b"\xDB\xDD"),
        (b"Hello\x7E", b"Hello\xDB\xDC"),
    ])
    def test_stuffing(self, data, expected):
        assert bit_stuff(data) == expected
        
    def test_roundtrip(self):
        orig = b"\x01\x02\x7E\xDB\x03"
        stuffed = bit_stuff(orig)
        destuffed = bit_destuff(stuffed)
        assert destuffed == orig

    def test_invalid_stuffing(self):
        # 6 1s in a row (0xFC is 11111100)
        with pytest.raises(ValueError):
            bit_destuff(b"\xFC")

if __name__ == "__main__":
    pytest.main([__file__])
