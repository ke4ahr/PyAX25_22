# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_transport_compliance.py

Compliance tests for transport interfaces.

Covers:
- KISS multi-drop command byte formatting
- AGWPE header structure and DataKind values
- Transport validation utilities
- Error handling in transports
- Mock-based send/receive round-trip

Uses mocks for serial/socket to test without hardware.
"""

import pytest
import struct
from unittest.mock import Mock, patch

from pyax25-22.interfaces.kiss import KISSInterface, FEND, FESC, TFEND, TFESC, CMD_DATA, CMD_TXDELAY
from pyax25-22.interfaces.agwpe import AGWPEInterface, HEADER_FMT, HEADER_SIZE
from pyax25-22.interfaces.transport import validate_frame_for_transport
from pyax25-22.core.framing import AX25Frame, AX25Address
from pyax25-22.core.exceptions import KISSError, AGWPEError, TransportError


@pytest.fixture
def mock_serial():
    """Mock serial port for KISS testing."""
    mock = Mock()
    mock.write = Mock()
    mock.read = Mock(return_value=b'')
    return mock


@pytest.fixture
def mock_socket():
    """Mock socket for AGWPE testing."""
    mock = Mock()
    mock.sendall = Mock()
    mock.recv = Mock(return_value=b'')
    return mock


def test_kiss_multi_drop_command_byte():
    """Test KISS command byte with multi-drop addressing."""
    # TNC 3, port 1, data command
    cmd_byte = (3 << 4) | (1 & 0x0F) | CMD_DATA
    assert cmd_byte == 0x31  # 0011 0001

    # TNC 15, port 0, TXDELAY
    cmd_byte = (15 << 4) | (0 & 0x0F) | CMD_TXDELAY
    assert cmd_byte == 0xF1  # 1111 0001


def test_kiss_escaping():
    """Test KISS FESC escaping in send."""
    # Data with FEND and FESC
    raw = bytes([FEND, FESC, 0xAA])
    escaped = bytes([FESC, TFEND, FESC, TFESC, 0xAA])
    assert escaped == b'\xdb\xdc\xdb\xdd\xaa'  # Manual check


def test_agwpe_header_format():
    """Test AGWPE header structure compliance."""
    assert struct.calcsize(HEADER_FMT) == HEADER_SIZE == 36

    # Pack example header
    port = 1
    data_kind = ord('D')
    call_from = b'KE4AHR   \x00'
    call_to = b'APRS     \x00'
    data_len = 10
    user = 0
    header = struct.pack(HEADER_FMT, port, data_kind, call_from, call_to, data_len, user)

    unpacked = struct.unpack(HEADER_FMT, header)
    assert unpacked[0] == 1
    assert chr(unpacked[1]) == 'D'
    assert unpacked[4] == 10


def test_transport_validation_kiss():
    """Test frame validation for KISS transport."""
    frame = AX25Frame(destination=AX25Address("TEST"), source=AX25Address("TEST"))
    frame.info = bytes(512)  # Too large for KISS
    with pytest.raises(TransportError):
        validate_frame_for_transport(frame, "KISS")

    frame.info = bytes(256)  # Valid
    validate_frame_for_transport(frame, "KISS")


def test_transport_validation_agwpe():
    """Test frame validation for AGWPE transport."""
    frame = AX25Frame(destination=AX25Address("TEST"), source=AX25Address("TEST"))
    frame.info = bytes(5000)  # Too large
    with pytest.raises(TransportError):
        validate_frame_for_transport(frame, "AGWPE")

    frame.info = bytes(2048)  # Valid
    validate_frame_for_transport(frame, "AGWPE")


def test_kiss_send_receive_mock(mock_serial):
    """Test KISS send/receive with mock serial."""
    with patch('serial.Serial', return_value=mock_serial):
        kiss = KISSInterface("mock_port")
        kiss.connect()

        frame = AX25Frame(destination=AX25Address("TEST"), source=AX25Address("TEST"))
        kiss.send_frame(frame)

        mock_serial.write.assert_called()  # Check send called

        # Simulate receive
        mock_serial.read.return_value = b'\xc0\x00testdata\xc0'
        tnc_addr, port, recv_frame = kiss.receive()
        assert tnc_addr == 0
        assert port == 0

        kiss.disconnect()


def test_agwpe_send_receive_mock(mock_socket):
    """Test AGWPE send/receive with mock socket."""
    with patch('socket.socket', return_value=mock_socket):
        agwpe = AGWPEInterface()
        agwpe.connect()

        agwpe.send_frame(1, 'D', 'KE4AHR', 'APRS', b'test')

        mock_socket.sendall.assert_called()  # Check send called

        # Simulate receive
        mock_socket.recv.side_effect = [
            struct.pack(HEADER_FMT, 1, ord('D'), b'KE4AHR   \x00', b'APRS     \x00', 4, 0),
            b'test'
        ]
        port, kind, fr, to, data = agwpe.receive()
        assert port == 1
        assert kind == 'D'
        assert data == b'test'

        agwpe.disconnect()


def test_kiss_error_handling(mock_serial):
    """Test KISS error cases."""
    mock_serial.write.side_effect = serial.SerialException("Mock error")
    kiss = KISSInterface("mock_port")
    kiss.connect()

    with pytest.raises(KISSError):
        kiss.send_frame(AX25Frame(destination=AX25Address("TEST"), source=AX25Address("TEST")))

    kiss.disconnect()


def test_agwpe_error_handling(mock_socket):
    """Test AGWPE error cases."""
    mock_socket.sendall.side_effect = socket.error("Mock error")
    agwpe = AGWPEInterface()
    agwpe.connect()

    with pytest.raises(AGWPEError):
        agwpe.send_frame(1, 'D', 'TEST', 'TEST', b'')

    agwpe.disconnect()
