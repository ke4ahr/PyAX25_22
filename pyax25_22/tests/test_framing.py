# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
Comprehensive Tests for AX.25 Framing

Covers:
- Address encoding/decoding
- Frame construction/parsing
- FCS calculation
- Bit stuffing/destuffing
- All frame types
- Error conditions
- Edge cases
- Performance testing

License: LGPLv3.0
Copyright (C) 2024 QA Team
"""

import pytest
import struct
import logging
import time
from typing import List, Tuple, Dict, Any
import random
import string

# Import the modules being tested
from pyax25_22.core.framing import (
    AX25Frame,
    AX25Address,
    fcs_calc,
    bit_stuff,
    bit_destuff,
    FrameType,
    PID,
    FrameMetadata
)

logger = logging.getLogger(__name__)

class TestAX25Address:
    """Test AX.25 address encoding and decoding"""
    
    @pytest.fixture
    def valid_addresses(self) -> List[Tuple[str, int, bool, bool]]:
        """Valid address test cases"""
        return [
            # (callsign, ssid, c_bit, r_bit)
            ("TEST", 0, False, False),
            ("TEST", 0, True, False),
            ("TEST", 15, False, False),
            ("LONGER", 7, True, False),
            ("A", 1, False, True),
            ("SHORT", 3, True, True),
            ("N0CALL", 0, False, False),
            ("KE4AHR", 10, True, False),
        ]

    @pytest.fixture
    def invalid_addresses(self) -> List[Tuple[Any, Any, Any, Any, str]]:
        """Invalid address test cases with expected errors"""
        return [
            # (callsign, ssid, c_bit, r_bit, expected_error)
            ("", 0, False, False, "callsign"),
            ("TOOLONGCALLSIGN", 0, False, False, "callsign"),
            ("TEST", -1, False, False, "SSID"),
            ("TEST", 16, False, False, "SSID"),
            ("TEST", 0, "not_bool", False, "C bit"),
            ("TEST", 0, False, "not_bool", "R bit"),
            (123, 0, False, False, "callsign"),
            (None, 0, False, False, "callsign"),
        ]

    def test_valid_address_creation(self, valid_addresses: List[Tuple[str, int, bool, bool]]):
        """Test creation of valid addresses"""
        for callsign, ssid, c_bit, r_bit in valid_addresses:
            addr = AX25Address(callsign, ssid, c_bit, r_bit)
            
            assert addr.callsign.strip() == callsign.upper()
            assert addr.ssid == ssid
            assert addr.c_bit == c_bit
            assert addr.r_bit == r_bit
            assert len(addr.callsign) == 6  # Should be padded to 6 chars
            
            logger.debug(f"Created address: {addr}")

    def test_invalid_address_creation(self, invalid_addresses: List[Tuple[Any, Any, Any, Any, str]]):
        """Test that invalid addresses raise appropriate errors"""
        for callsign, ssid, c_bit, r_bit, expected_error in invalid_addresses:
            with pytest.raises(ValueError) as exc_info:
                AX25Address(callsign, ssid, c_bit, r_bit)
            
            assert expected_error.lower() in str(exc_info.value).lower()

    def test_address_encoding(self, valid_addresses: List[Tuple[str, int, bool, bool]]):
        """Test address encoding to bytes"""
        for callsign, ssid, c_bit, r_bit in valid_addresses:
            addr = AX25Address(callsign, ssid, c_bit, r_bit)
            encoded = addr.encoded()
            
            # Should be exactly 7 bytes
            assert len(encoded) == 7
            
            # Last byte should have HDLC extension bit set
            assert encoded[6] & 0x40 == 0x40
            
            # SSID should be in bits 1-4 of last byte
            assert (encoded[6] >> 1) & 0x0F == ssid
            
            # C-bit should be in bit 7 if set
            if c_bit:
                assert encoded[6] & 0x80 == 0x80
            else:
                assert encoded[6] & 0x80 == 0x00
                
            # R-bit should be in bit 6 if set (though typically 0 for AX.25 v2.2)
            if r_bit:
                assert encoded[6] & 0x40 == 0x40
            else:
                assert encoded[6] & 0x40 == 0x40  # HDLC extension bit is always set
                
            logger.debug(f"Encoded address {callsign}-{ssid}: {encoded.hex()}")

    def test_address_decoding(self, valid_addresses: List[Tuple[str, int, bool, bool]]):
        """Test address decoding from bytes"""
        for callsign, ssid, c_bit, r_bit in valid_addresses:
            addr = AX25Address(callsign, ssid, c_bit, r_bit)
            encoded = addr.encoded()
            
            # Test decoding
            decoded_addr, last = AX25Address.from_bytes(encoded)
            
            assert decoded_addr.callsign.strip() == callsign.upper()
            assert decoded_addr.ssid == ssid
            assert decoded_addr.c_bit == c_bit
            assert decoded_addr.r_bit == r_bit
            assert not last  # Should not be last unless explicitly set
            
            logger.debug(f"Decoded address: {decoded_addr}, last={last}")

    def test_address_roundtrip(self, valid_addresses: List[Tuple[str, int, bool, bool]]):
        """Test address encoding/decoding roundtrip"""
        for callsign, ssid, c_bit, r_bit in valid_addresses:
            original = AX25Address(callsign, ssid, c_bit, r_bit)
            encoded = original.encoded()
            decoded, last = AX25Address.from_bytes(encoded)
            
            # Should be identical (except for padding)
            assert decoded.callsign.strip() == original.callsign.strip()
            assert decoded.ssid == original.ssid
            assert decoded.c_bit == original.c_bit
            assert decoded.r_bit == original.r_bit

    def test_address_last_flag(self):
        """Test last address flag handling"""
        addr = AX25Address("TEST", 0)
        
        # Test encoding with last=True
        encoded_last = addr.encoded(last=True)
        assert encoded_last[6] & 0x01 == 0x01
        
        # Test encoding with last=False
        encoded_not_last = addr.encoded(last=False)
        assert encoded_not_last[6] & 0x01 == 0x00
        
        # Both should decode to same address
        decoded_last, last_flag_last = AX25Address.from_bytes(encoded_last)
        decoded_not_last, last_flag_not_last = AX25Address.from_bytes(encoded_not_last)
        
        assert decoded_last.callsign == decoded_not_last.callsign
        assert decoded_last.ssid == decoded_not_last.ssid
        assert last_flag_last == True
        assert last_flag_not_last == False

    def test_address_repr(self):
        """Test address string representation"""
        addr = AX25Address("TEST", 5, True, False)
        repr_str = repr(addr)
        
        assert "AX25Address" in repr_str
        assert "TEST" in repr_str
        assert "ssid=5" in repr_str
        assert "c_bit=True" in repr_str

class TestFCS:
    """Test FCS (Frame Check Sequence) calculation"""
    
    @pytest.fixture
    def fcs_test_vectors(self) -> List[Tuple[bytes, int]]:
        """FCS test vectors from AX.25 specification"""
        return [
            # (data, expected_fcs)
            (b"\x01", 0x0E1F),
            (b"\x03\xF0", 0x8E72),
            (b"C", 0x14E1),
            (b"test", 0x8E72),
            (b"Hello World!", 0x6F91),
            (b"", 0xFFFF),  # Empty data
            (b"\x00" * 100, 0x1D0F),  # Long zero data
        ]

    def test_fcs_calculation(self, fcs_test_vectors: List[Tuple[bytes, int]]):
        """Test FCS calculation against known vectors"""
        for data, expected_fcs in fcs_test_vectors:
            calculated_fcs = fcs_calc(data)
            assert calculated_fcs == expected_fcs, \
                f"FCS mismatch for {data.hex()}: got 0x{calculated_fcs:04X}, expected 0x{expected_fcs:04X}"

    def test_fcs_type_validation(self):
        """Test FCS input validation"""
        with pytest.raises(TypeError):
            fcs_calc("not_bytes")  # Should be bytes
            
        with pytest.raises(TypeError):
            fcs_calc(None)  # Should not be None

    def test_fcs_consistency(self):
        """Test FCS calculation consistency"""
        test_data = b"Consistency test data for FCS"
        
        # Calculate FCS multiple times
        fcs1 = fcs_calc(test_data)
        fcs2 = fcs_calc(test_data)
        fcs3 = fcs_calc(test_data)
        
        # Should be identical
        assert fcs1 == fcs2 == fcs3

    def test_fcs_performance(self):
        """Test FCS calculation performance"""
        large_data = b"A" * 10000  # 10KB of data
        
        start_time = time.time()
        fcs = fcs_calc(large_data)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"FCS calculation for 10KB took {duration:.4f} seconds")
        
        # Should complete in reasonable time (less than 1 second)
        assert duration < 1.0
        assert isinstance(fcs, int)

class TestBitStuffing:
    """Test bit stuffing and destuffing"""
    
    @pytest.fixture
    def bit_stuffing_test_cases(self) -> List[Tuple[bytes, bytes]]:
        """Test cases for bit stuffing/destuffing"""
        return [
            # (original, stuffed)
            (b"\x7E", b"\xDB\xDC"),  # FLAG byte
            (b"\xDB", b"\xDB\xDD"),  # ESC byte
            (b"Hello\x7E", b"Hello\xDB\xDC"),  # FLAG in middle
            (b"\x7E\x7E", b"\xDB\xDC\xDB\xDC"),  # Multiple FLAGs
            (b"\x00\x00\x00\x00\x00\x00", b"\x00\x00\x00\x00\x00\x00"),  # No stuffing needed
            (b"\x1F", b"\x1F"),  # 5 ones at end of byte
            (b"\x3E", b"\x3E"),  # 6 ones (should be handled by destuffing)
        ]

    def test_bit_stuffing(self, bit_stuffing_test_cases: List[Tuple[bytes, bytes]]):
        """Test bit stuffing functionality"""
        for original, expected_stuffed in bit_stuffing_test_cases:
            stuffed = bit_stuff(original)
            assert stuffed == expected_stuffed, \
                f"Bit stuffing failed for {original.hex()}: got {stuffed.hex()}, expected {expected_stuffed.hex()}"

    def test_bit_destuffing(self, bit_stuffing_test_cases: List[Tuple[bytes, bytes]]):
        """Test bit destuffing functionality"""
        for original, stuffed in bit_stuffing_test_cases:
            destuffed = bit_destuff(stuffed)
            assert destuffed == original, \
                f"Bit destuffing failed for {stuffed.hex()}: got {destuffed.hex()}, expected {original.hex()}"

    def test_bit_stuffing_roundtrip(self, bit_stuffing_test_cases: List[Tuple[bytes, bytes]]):
        """Test bit stuffing/destuffing roundtrip"""
        for original, _ in bit_stuffing_test_cases:
            stuffed = bit_stuff(original)
            destuffed = bit_destuff(stuffed)
            assert destuffed == original, \
                f"Roundtrip failed for {original.hex()}: got {destuffed.hex()}"

    def test_bit_stuffing_edge_cases(self):
        """Test bit stuffing edge cases"""
        # Test empty data
        assert bit_stuff(b"") == b""
        assert bit_destuff(b"") == b""
        
        # Test single bytes
        assert bit_stuff(b"\x00") == b"\x00"
        assert bit_destuff(b"\x00") == b"\x00"
        
        # Test maximum consecutive ones (should trigger stuffing)
        data_with_5_ones = b"\x1F"  # 00011111 (5 ones at end)
        stuffed = bit_stuff(data_with_5_ones)
        # Should not be stuffed since ones are not consecutive across byte boundaries
        assert stuffed == data_with_5_ones

    def test_bit_destuffing_errors(self):
        """Test bit destuffing error conditions"""
        # Test invalid consecutive ones (6 or more)
        with pytest.raises(ValueError, match="consecutive.*violation"):
            bit_destuff(b"\xFC")  # 11111100 (6 consecutive ones)
            
        # Test incomplete escape sequence
        with pytest.raises(ValueError):
            bit_destuff(b"\xDB")  # Incomplete escape

    def test_bit_stuffing_performance(self):
        """Test bit stuffing performance"""
        large_data = b"\x7E" * 1000  # 1000 FLAG bytes
        
        start_time = time.time()
        stuffed = bit_stuff(large_data)
        destuffed = bit_destuff(stuffed)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Bit stuffing/destuffing for 1000 FLAG bytes took {duration:.4f} seconds")
        
        # Should complete in reasonable time
        assert duration < 1.0
        assert destuffed == large_data  # Roundtrip should be perfect

class TestAX25Frame:
    """Test AX.25 frame encoding and decoding"""
    
    @pytest.fixture
    def sample_ui_frame_data(self) -> Dict[str, Any]:
        """Sample UI frame test data"""
        return {
            'dest_call': 'DEST',
            'dest_ssid': 1,
            'src_call': 'SRC',
            'src_ssid': 2,
            'info': b'Hello AX.25!',
            'pid': PID.NO_LAYER3
        }

    @pytest.fixture
    def sample_i_frame_data(self) -> Dict[str, Any]:
        """Sample I frame test data"""
        return {
            'dest_call': 'DEST',
            'dest_ssid': 0,
            'src_call': 'SRC',
            'src_ssid': 0,
            'info': b'Information data',
            'ns': 3,
            'nr': 5,
            'poll': False
        }

    def test_ui_frame_encoding(self, sample_ui_frame_data: Dict[str, Any]):
        """Test UI frame encoding"""
        frame = AX25Frame(
            dest=AX25Address(sample_ui_frame_data['dest_call'], sample_ui_frame_data['dest_ssid']),
            src=AX25Address(sample_ui_frame_data['src_call'], sample_ui_frame_data['src_ssid']),
            pid=sample_ui_frame_data['pid']
        )
        
        encoded = frame.encode_ui(sample_ui_frame_data['info'])
        
        # Should start and end with FLAG
        assert encoded[0] == 0x7E
        assert encoded[-1] == 0x7E
        
        # Should contain the info data
        assert sample_ui_frame_data['info'] in encoded
        
        logger.debug(f"UI frame encoded: {encoded.hex()}")

    def test_ui_frame_decoding(self, sample_ui_frame_data: Dict[str, Any]):
        """Test UI frame decoding"""
        # Create and encode frame
        frame = AX25Frame(
            dest=AX25Address(sample_ui_frame_data['dest_call'], sample_ui_frame_data['dest_ssid']),
            src=AX25Address(sample_ui_frame_data['src_call'], sample_ui_frame_data['src_ssid']),
            pid=sample_ui_frame_data['pid']
        )
        encoded = frame.encode_ui(sample_ui_frame_data['info'])
        
        # Decode frame
        decoded = AX25Frame.from_bytes(encoded)
        
        # Verify decoded data
        assert decoded.dest.callsign.strip() == sample_ui_frame_data['dest_call']
        assert decoded.dest.ssid == sample_ui_frame_data['dest_ssid']
        assert decoded.src.callsign.strip() == sample_ui_frame_data['src_call']
        assert decoded.src.ssid == sample_ui_frame_data['src_ssid']
        assert decoded.info == sample_ui_frame_data['info']
        assert decoded.pid == sample_ui_frame_data['pid']
        assert decoded.type == FrameType.UI

    def test_i_frame_encoding(self, sample_i_frame_data: Dict[str, Any]):
        """Test I frame encoding with sequence numbers"""
        frame = AX25Frame(
            dest=AX25Address(sample_i_frame_data['dest_call'], sample_i_frame_data['dest_ssid']),
            src=AX25Address(sample_i_frame_data['src_call'], sample_i_frame_data['src_ssid'])
        )
        
        encoded = frame.encode_i(
            sample_i_frame_data['info'],
            sample_i_frame_data['ns'],
            sample_i_frame_data['nr'],
            sample_i_frame_data['poll']
        )
        
        # Should start and end with FLAG
        assert encoded[0] == 0x7E
        assert encoded[-1] == 0x7E
        
        logger.debug(f"I frame encoded: {encoded.hex()}")

    def test_frame_type_encoding(self):
        """Test encoding of all frame types"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        
        # Test each frame type
        test_cases = [
            (FrameType.SABM, lambda: frame.encode_sabm()),
            (FrameType.DISC, lambda: frame.encode_disc()),
            (FrameType.UA, lambda: frame.encode_ua()),
            (FrameType.DM, lambda: frame.encode_dm()),
            (FrameType.RR, lambda: frame.encode_s(FrameType.RR, nr=0)),
            (FrameType.RNR, lambda: frame.encode_s(FrameType.RNR, nr=0)),
            (FrameType.REJ, lambda: frame.encode_s(FrameType.REJ, nr=0)),
            (FrameType.SREJ, lambda: frame.encode_s(FrameType.SREJ, nr=0)),
        ]
        
        for expected_type, encode_func in test_cases:
            encoded = encode_func()
            decoded = AX25Frame.from_bytes(encoded)
            assert decoded.type == expected_type, \
                f"Frame type mismatch: expected {expected_type}, got {decoded.type}"

    def test_frame_validation(self, sample_ui_frame_data: Dict[str, Any]):
        """Test frame validation"""
        frame = AX25Frame(
            dest=AX25Address(sample_ui_frame_data['dest_call'], sample_ui_frame_data['dest_ssid']),
            src=AX25Address(sample_ui_frame_data['src_call'], sample_ui_frame_data['src_ssid']),
            pid=sample_ui_frame_data['pid']
        )
        
        encoded = frame.encode_ui(sample_ui_frame_data['info'])
        
        # Should be valid
        assert frame.validate()
        
        # Create invalid frame (corrupt FCS)
        invalid_encoded = encoded[:-2] + b'\x00\x00'  # Wrong FCS
        
        with pytest.raises(ValueError):
            AX25Frame.from_bytes(invalid_encoded)

    def test_frame_error_conditions(self):
        """Test frame error conditions"""
        # Test too short frame
        with pytest.raises(ValueError, match="too short"):
            AX25Frame.from_bytes(b"\x7E\x7E")  # Just flags
        
        # Test missing flags
        with pytest.raises(ValueError, match="Missing frame flags"):
            AX25Frame.from_bytes(b"invalid")
        
        # Test invalid frame data type
        with pytest.raises(TypeError):
            AX25Frame.from_bytes("not_bytes")

    def test_frame_info_field_parsing(self, sample_ui_frame_data: Dict[str, Any]):
        """Test information field parsing"""
        frame = AX25Frame(
            dest=AX25Address(sample_ui_frame_data['dest_call'], sample_ui_frame_data['dest_ssid']),
            src=AX25Address(sample_ui_frame_data['src_call'], sample_ui_frame_data['src_ssid']),
            pid=sample_ui_frame_data['pid']
        )
        
        encoded = frame.encode_ui(sample_ui_frame_data['info'])
        decoded = AX25Frame.from_bytes(encoded)
        
        # Verify info field
        assert decoded.info == sample_ui_frame_data['info']
        
        # Test empty info field
        empty_encoded = frame.encode_ui(b'')
        empty_decoded = AX25Frame.from_bytes(empty_encoded)
        assert empty_decoded.info == b''

    def test_frame_address_parsing(self):
        """Test address field parsing with multiple digipeaters"""
        dest = AX25Address("DEST", 1)
        src = AX25Address("SRC", 2)
        digi1 = AX25Address("DIGI1", 3)
        digi2 = AX25Address("DIGI2", 4)
        
        frame = AX25Frame(dest, src, [digi1, digi2])
        encoded = frame.encode_ui(b"Test message")
        decoded = AX25Frame.from_bytes(encoded)
        
        # Verify all addresses
        assert decoded.dest.callsign.strip() == "DEST"
        assert decoded.dest.ssid == 1
        assert decoded.src.callsign.strip() == "SRC"
        assert decoded.src.ssid == 2
        assert len(decoded.digipeaters) == 2
        assert decoded.digipeaters[0].callsign.strip() == "DIGI1"
        assert decoded.digipeaters[0].ssid == 3
        assert decoded.digipeaters[1].callsign.strip() == "DIGI2"
        assert decoded.digipeaters[1].ssid == 4

    def test_frame_pid_parsing(self):
        """Test PID field parsing"""
        test_pids = [PID.NO_LAYER3, PID.IP, PID.ARPA_IP, PID.PACSAT]
        
        for pid in test_pids:
            frame = AX25Frame(
                dest=AX25Address("DEST", 0),
                src=AX25Address("SRC", 0),
                pid=pid
            )
            
            encoded = frame.encode_ui(b"PID test")
            decoded = AX25Frame.from_bytes(encoded)
            
            assert decoded.pid == pid

    def test_frame_control_field_parsing(self):
        """Test control field parsing for all frame types"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        
        # Test I frame control field
        encoded_i = frame.encode_i(b"Test", ns=3, nr=5, poll=True)
        decoded_i = AX25Frame.from_bytes(encoded_i)
        assert decoded_i.type == FrameType.I
        assert decoded_i.ns == 3
        assert decoded_i.nr == 5
        assert decoded_i.poll == True
        
        # Test S frame control field
        encoded_rr = frame.encode_s(FrameType.RR, nr=2, poll=False)
        decoded_rr = AX25Frame.from_bytes(encoded_rr)
        assert decoded_rr.type == FrameType.RR
        assert decoded_rr.nr == 2
        assert decoded_rr.poll == False

    def test_frame_performance(self):
        """Test frame encoding/decoding performance"""
        large_info = b"X" * 1000  # 1KB info field
        
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0),
            pid=PID.NO_LAYER3
        )
        
        start_time = time.time()
        
        # Encode
        encoded = frame.encode_ui(large_info)
        
        # Decode
        decoded = AX25Frame.from_bytes(encoded)
        
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Frame encode/decode for 1KB took {duration:.4f} seconds")
        
        # Should complete quickly
        assert duration < 0.1
        
        # Verify data integrity
        assert decoded.info == large_info
        assert decoded.dest.callsign.strip() == "DEST"
        assert decoded.src.callsign.strip() == "SRC"

class TestFrameEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_info_field(self):
        """Test frames with empty info fields"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        
        # UI frame with empty info
        encoded = frame.encode_ui(b'')
        decoded = AX25Frame.from_bytes(encoded)
        assert decoded.info == b''
        
        # I frame with empty info
        encoded_i = frame.encode_i(b'', ns=0, nr=0)
        decoded_i = AX25Frame.from_bytes(encoded_i)
        assert decoded_i.info == b''

    def test_maximum_frame_size(self):
        """Test maximum frame size handling"""
        # Create large info field (simulate maximum size)
        max_info = b"X" * 2048  # 2KB
        
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        
        # Should handle large frames
        encoded = frame.encode_ui(max_info)
        decoded = AX25Frame.from_bytes(encoded)
        
        assert decoded.info == max_info

    def test_special_characters_in_info(self):
        """Test info fields with special characters"""
        special_chars = b"\x00\x01\x02\x7F\x80\xFF"  # Null, control, extended ASCII
        
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        
        encoded = frame.encode_ui(special_chars)
        decoded = AX25Frame.from_bytes(encoded)
        
        assert decoded.info == special_chars

    def test_multiple_digipeaters(self):
        """Test frames with maximum digipeaters"""
        addresses = [
            AX25Address("DEST", 0),
            AX25Address("SRC", 0),
        ] + [AX25Address(f"DIGI{i}", i) for i in range(1, 9)]  # Max 8 digipeaters
        
        frame = AX25Frame(addresses[0], addresses[1], addresses[2:])
        encoded = frame.encode_ui(b"Multi-digi test")
        decoded = AX25Frame.from_bytes(encoded)
        
        assert len(decoded.digipeaters) == 8
        for i, digi in enumerate(decoded.digipeaters, 1):
            assert digi.callsign.strip() == f"DIGI{i}"
            assert digi.ssid == i

    def test_frame_with_all_flags(self):
        """Test frame with FLAG bytes in info field"""
        info_with_flags = b"Start\x7EMiddle\x7EEnd"
        
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        
        encoded = frame.encode_ui(info_with_flags)
        decoded = AX25Frame.from_bytes(encoded)
        
        assert decoded.info == info_with_flags

class TestFrameUtilities:
    """Test frame utility methods"""
    
    def test_frame_repr(self):
        """Test frame string representation"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 1),
            src=AX25Address("SRC", 2),
            pid=PID.NO_LAYER3
        )
        
        repr_str = repr(frame)
        
        assert "AX25Frame" in repr_str
        assert "DEST" in repr_str
        assert "SRC" in repr_str
        assert "UI" in repr_str
        assert "NO_LAYER3" in repr_str

    def test_frame_dict_conversion(self):
        """Test frame conversion to dictionary"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 1),
            src=AX25Address("SRC", 2),
            pid=PID.NO_LAYER3
        )
        frame.info = b"Test info"
        
        frame_dict = frame.to_dict()
        
        assert isinstance(frame_dict, dict)
        assert frame_dict['type'] == 'UI'
        assert frame_dict['destination'] == 'DEST-1'
        assert frame_dict['source'] == 'SRC-2'
        assert frame_dict['info_hex'] == '5465737420696E666F'  # "Test info" in hex

    def test_frame_sequence_info(self):
        """Test sequence number information extraction"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        frame.ns = 5
        frame.nr = 3
        
        seq_info = frame.get_sequence_info()
        
        assert seq_info['send_sequence'] == 5
        assert seq_info['receive_sequence'] == 3
        assert seq_info['ack_sequence'] == 3

    def test_frame_address_info(self):
        """Test address information extraction"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 1),
            src=AX25Address("SRC", 2, c_bit=True),
            digipeaters=[AX25Address("DIGI", 3)]
        )
        
        addr_info = frame.get_address_info()
        
        assert addr_info['destination'] == 'DEST-1'
        assert addr_info['source'] == 'SRC-3'  # SSID includes C-bit
        assert addr_info['digipeaters'] == ['DIGI-3']
        assert addr_info['has_c_bit'] == True
        assert addr_info['has_r_bit'] == False

    def test_frame_size_calculation(self):
        """Test frame size calculation"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0),
            pid=PID.NO_LAYER3
        )
        frame.info = b"Size test"
        
        calculated_size = frame.calculate_frame_size()
        
        # Manual calculation: dest(7) + src(7) + control(1) + pid(1) + info(9) + fcs(2) + flags(2) = 29
        expected_size = 7 + 7 + 1 + 1 + len(b"Size test") + 2 + 2
        assert calculated_size == expected_size

    def test_frame_response_creation(self):
        """Test response frame creation"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0),
            pid=PID.NO_LAYER3
        )
        frame.info = b"Request"
        
        response = frame.create_response(FrameType.UA)
        
        # Response should have swapped addresses
        assert response.dest.callsign.strip() == "SRC"
        assert response.src.callsign.strip() == "DEST"
        assert response.type == FrameType.UA

    def test_frame_digipeater_management(self):
        """Test digipeater addition and management"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        
        # Add digipeaters
        digi1 = AX25Address("DIGI1", 1)
        digi2 = AX25Address("DIGI2", 2)
        
        frame.add_digipeater(digi1)
        frame.add_digipeater(digi2)
        
        assert len(frame.digipeaters) == 2
        assert frame.digipeaters[0].callsign.strip() == "DIGI1"
        assert frame.digipeaters[1].callsign.strip() == "DIGI2"
        
        # Test maximum digipeaters
        with pytest.raises(ValueError):
            for i in range(8):  # Try to add 8 more (total would be 10)
                frame.add_digipeater(AX25Address(f"EXTRA{i}", 0))

    def test_frame_poll_management(self):
        """Test poll/final bit management"""
        frame = AX25Frame(
            dest=AX25Address("DEST", 0),
            src=AX25Address("SRC", 0)
        )
        
        # Test setting poll bit
        frame.set_poll(True)
        assert frame.poll == True
        
        # Test clearing poll bit
        frame.set_poll(False)
        assert frame.poll == False

class TestFrameStressTesting:
    """Stress test frame operations"""
    
    def test_random_frame_generation(self):
        """Test generation of random frames"""
        for _ in range(100):
            # Generate random data
            dest_call = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(1, 6)))
            src_call = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(1, 6)))
            dest_ssid = random.randint(0, 15)
            src_ssid = random.randint(0, 15)
            info_len = random.randint(0, 100)
            info = bytes(random.choices(range(256), k=info_len))
            
            try:
                frame = AX25Frame(
                    dest=AX25Address(dest_call, dest_ssid),
                    src=AX25Address(src_call, src_ssid)
                )
                
                encoded = frame.encode_ui(info)
                decoded = AX25Frame.from_bytes(encoded)
                
                # Verify roundtrip
                assert decoded.dest.callsign.strip() == dest_call
                assert decoded.src.callsign.strip() == src_call
                assert decoded.info == info
                
            except ValueError:
                # Some random combinations might be invalid, that's OK
                continue

    def test_concurrent_frame_operations(self):
        """Test frame operations under concurrent access"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def frame_worker(worker_id):
            try:
                for i in range(10):
                    frame = AX25Frame(
                        dest=AX25Address(f"DEST{worker_id}", 0),
                        src=AX25Address(f"SRC{i}", 0)
                    )
                    encoded = frame.encode_ui(f"Worker {worker_id} message {i}".encode())
                    decoded = AX25Frame.from_bytes(encoded)
                    
                    assert decoded.dest.callsign.strip() == f"DEST{worker_id}"
                    assert decoded.info == f"Worker {worker_id} message {i}".encode()
                
                results.put(("success", worker_id))
            except Exception as e:
                results.put(("error", worker_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=frame_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            result = results.get()
            if result[0] == "success":
                success_count += 1
            else:
                logger.error(f"Worker {result[1]} failed: {result[2]}")
        
        assert success_count == 5, f"Expected 5 successful workers, got {success_count}"

# Run additional tests if called directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run specific tests
    test_class = TestAX25Frame()
    sample_data = {
        'dest_call': 'DEST',
        'dest_ssid': 1,
        'src_call': 'SRC', 
        'src_ssid': 2,
        'info': b'Hello AX.25!',
        'pid': PID.NO_LAYER3
    }
    
    print("Running framing tests...")
    
    try:
        test_class.test_ui_frame_encoding(sample_data)
        print("✓ UI frame encoding test passed")
        
        test_class.test_ui_frame_decoding(sample_data)
        print("✓ UI frame decoding test passed")
        
        test_class.test_frame_validation(sample_data)
        print("✓ Frame validation test passed")
        
        test_class.test_frame_performance()
        print("✓ Frame performance test passed")
        
        print("\nAll framing tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

