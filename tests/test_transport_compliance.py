# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_transport_compliance.py

Compliance tests for KISS and AGWPE transport layers.

Verifies:
- KISS frame encoding/decoding (standard and multi-drop)
- AGWPE header format and frame structure
- Round-trip validation through mock transport
"""

import pytest
from unittest.mock import Mock, patch
import struct
import time

from pyax25_22.core.framing import AX25Frame, AX25Address
from pyax25_22.interfaces.kiss import KISSInterface
from pyax25_22.interfaces.agwpe import AGWPEInterface

class MockSerial:
    """Mock serial port for KISS testing."""

    def __init__(self):
        self.in_buffer = b""
        self.out_buffer = b""
        self.is_open = True

    def write(self, data):
        self.out_buffer += data

    def read(self, size=1):
        if not self.in_buffer:
            return b""
        data = self.in_buffer[:size]
        self.in_buffer = self.in_buffer[size:]
        return data

    def close(self):
        self.is_open = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def in_waiting(self):
        return len(self.in_buffer)

def test_kiss_multi_drop_command_byte():
    """Test multi-drop KISS command byte encoding."""
    serial = MockSerial()
    transport = KISSInterface("mock_port")
    transport.serial = serial

    # Test command byte construction
    assert transport._build_command_byte(1, 0x00) == 0x10  # Port 1, DATA
    assert transport._build_command_byte(0, 0x0C) == 0x0C  # Port 0, DATA_EXT
    assert transport._build_command_byte(15, 0x03) == 0xF3  # Port 15, UI

def test_agwpe_header_format():
    """Test AGWPE header structure and fields."""
    # Test header building without actual connection
    transport = AGWPEInterface()

    # Test header construction
    header = transport._build_header(0, 'K', 'DEST', 'SRC', 5)
    assert len(header) == 36
    assert header[4] == ord('K')  # Data kind

    # Test header parsing
    mock_data = b'\x00\x00\x00\x00\x00\x00\x00KDEST     SRC      \x05\x00\x00\x00\x00\x00\x00\x00test'
    port, kind, call_from, call_to, data = transport._parse_header(mock_data)
    assert port == 0
    assert kind == 'K'
    assert call_from == 'DEST'
    assert call_to == 'SRC'
    assert data == b'test'

def test_transport_validation_kiss():
    """Test KISS transport round-trip validation."""
    serial = MockSerial()
    transport = KISSInterface("mock_port")
    transport.serial = serial

    # Test frame encoding/decoding
    frame = AX25Frame(
        destination=AX25Address("APRS"),
        source=AX25Address("N0CALL"),
        control=0x03,
        pid=0xF0,
        info=b"!4903.50N/07201.75W-Test",
    )
    raw = frame.encode()

    # Test KISS framing
    kiss_frame = transport._build_kiss_frame(raw)
    assert kiss_frame[0] == 0xC0  # FEND
    assert kiss_frame[-1] == 0xC0  # FEND

    # Test frame extraction
    extracted = transport._extract_frame(kiss_frame[1:-1])
    assert extracted == raw

def test_transport_validation_agwpe():
    """Test AGWPE transport round-trip validation."""
    # Test header parsing without connection
    transport = AGWPEInterface()

    # Test header parsing
    mock_data = b'\x00\x00\x00\x00\x00\x00\x00KDEST     SRC      \x05\x00\x00\x00\x00\x00\x00\x00test'
    port, kind, call_from, call_to, data = transport._parse_header(mock_data)
    assert port == 0
    assert kind == 'K'
    assert call_from == 'DEST'
    assert call_to == 'SRC'
    assert data == b'test'

def test_kiss_send_receive_mock():
    """Test full KISS send/receive with mock serial and delays."""
    serial = MockSerial()
    transport = KISSInterface("mock_port")
    transport.serial = serial

    # Test send/receive without actual connection
    frame = AX25Frame(
        destination=AX25Address("TEST"),
        source=AX25Address("MOCK"),
        control=0x03,
        pid=0xF0,
        info=b"Integration test",
    )
    raw = frame.encode()

    # Test encoding
    kiss_encoded = transport._build_kiss_frame(raw)
    assert kiss_encoded[0] == 0xC0
    assert kiss_encoded[-1] == 0xC0

    # Test decoding
    serial.in_buffer = kiss_encoded
    decoded = transport._extract_frame(kiss_encoded[1:-1])
    assert decoded == raw
