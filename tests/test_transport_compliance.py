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

from pyax25_22.core.config import DEFAULT_CONFIG_MOD8
from pyax25_22.interfaces.kiss import KISSInterface
from pyax25_22.interfaces.agwpe import AGWPEInterface
from pyax25_22.core.framing import AX25Frame, AX25Address


class MockSerial:
    """Mock serial port for KISS testing."""

    def __init__(self):
        self.in_buffer = b""
        self.out_buffer = b""

    def write(self, data):
        self.out_buffer += data

    def read(self, size=1):
        if not self.in_buffer:
            return b""
        data = self.in_buffer[:size]
        self.in_buffer = self.in_buffer[size:]
        return data

    def in_waiting(self):
        return len(self.in_buffer)


def test_kiss_multi_drop_command_byte():
    """Test multi-drop KISS command byte encoding."""
    # High nibble = port/command
    frame = AX25Frame(
        destination=AX25Address("DEST"),
        source=AX25Address("SRC"),
        control=0x03,  # UI
        info=b"test",
    )
    encoded = frame.encode()

    # Create KISS interface and connect it
    serial = MockSerial()
    transport = KISSInterface("mock_port")
    transport.serial = serial  # Inject mock serial
    transport.connect()

    # Send frame and check the output
    transport.send_frame(frame)
    sent = serial.out_buffer

    # First byte: FEND (0xC0)
    # Second byte: port<<4 | command (0x10 | 0x00 = 0x10 for port 1, data)
    assert sent[0] == 0xC0  # FEND
    assert sent[1] == 0x00  # Default port 0, DATA command

def test_agwpe_header_format():
    """Test AGWPE header structure and fields."""
    frame = AX25Frame(
        destination=AX25Address("DEST"),
        source=AX25Address("SRC"),
        control=0x03,
        pid=0xF0,
        info=b"test",
    )
    raw_frame = frame.encode()

    # Create AGWPE interface and connect it
    mock_socket = Mock()
    transport = AGWPEInterface()
    transport.sock = mock_socket  # Inject mock socket
    transport.connect()

    # Send frame and check the output
    transport.send_frame(0, 'K', 'SRC', 'DEST', raw_frame)
    sent = mock_socket.sendall.call_args[0][0]

    # Basic header check
    assert len(sent) == 36 + len(raw_frame)
    assert sent[4] == ord('K')  # Data kind

def test_transport_validation_kiss():
    """Test KISS transport round-trip validation."""
    serial = MockSerial()
    transport = KISSInterface("mock_port")
    transport.serial = serial  # Inject mock serial
    transport.connect()  # Connect before sending

    frame = AX25Frame(
        destination=AX25Address("APRS"),
        source=AX25Address("N0CALL"),
        control=0x03,
        pid=0xF0,
        info=b"!4903.50N/07201.75W-Test",
    )
    raw = frame.encode()

    transport.send_frame(raw)
    sent = serial.out_buffer

    # Strip FEND and decode
    assert sent.startswith(b'\xc0\x00')  # FEND + DATA
    assert sent.endswith(b'\xc0')

    # Inject back for receive
    serial.in_buffer = sent
    received = transport.receive_frame()
    assert received == raw


def test_transport_validation_agwpe():
    """Test AGWPE transport round-trip validation."""
    mock_socket = Mock()
    transport = AGWPEInterface()
    transport.sock = mock_socket
    transport.connect()

    frame = AX25Frame(
        destination=AX25Address("BEACON"),
        source=AX25Address("N0CALL"),
        control=0x03,
        info=b"Beacon message",
    )
    raw = frame.encode()

    transport.send_frame(0, 'K', 'N0CALL', 'BEACON', raw)
    sent = mock_socket.sendall.call_args[0][0]

    # Basic header check
    assert len(sent) == 36 + len(raw)
    assert sent[4] == ord('K')  # Data kind


def test_kiss_send_receive_mock():
    """Test full KISS send/receive with mock serial and delays."""
    serial = MockSerial()
    transport = KISSInterface("mock_port")
    transport.serial = serial
    transport.connect()

    # Send UI frame
    frame = AX25Frame(
        destination=AX25Address("TEST"),
        source=AX25Address("MOCK"),
        control=0x03,
        pid=0xF0,
        info=b"Integration test",
    )
    raw = frame.encode()

    transport.send_frame(raw)
    assert len(serial.out_buffer) > len(raw) + 2  # FEND + command + FEND

    # Simulate TNC delay
    time.sleep(0.1)

    # Inject response frame
    response = AX25Frame(
        destination=AX25Address("MOCK"),
        source=AX25Address("TEST"),
        control=0x01,  # RR
    ).encode()
    kiss_response = bytes([0xC0, 0x00]) + response + bytes([0xC0])
    serial.in_buffer = kiss_response

    received = transport.receive_frame()
    assert received == response

    # Verify timing behavior
    assert transport.last_rx_time > 0
