# tests/test_kiss.py
"""
Comprehensive KISS Interface Tests

Covers:
- Frame encoding/decoding
- Serial/TCP transports
- Multi-drop functionality
- Error conditions

License: LGPLv3.0
Copyright (C) 2024 QA Team
"""

import pytest
import logging
from unittest.mock import Mock, patch
from pyax25_22.interfaces.kiss import (
    KISSInterface,
    SerialKISSInterface,
    TCPKISSInterface,
    KISSCommand,
    KISSProtocolError,
    TransportError
)
from pyax25_22.interfaces.kiss_tcp import FLAG as KISS_FLAG

@pytest.fixture
def mock_serial():
    return Mock()

@pytest.fixture
def mock_socket():
    return Mock()

class TestKISSProtocol:
    @pytest.mark.parametrize("data,expected", [
        (b"\x00Hello", b"\xc0\x00Hello\xc0"),
        (b"\xc0\xdb", b"\xc0\x00\xdb\xdc\xdb\xdd\xc0"),
        (b"\x01\x02\x03", b"\xc0\x01\x02\x03\xc0"),
    ])
    def test_frame_encoding(self, data, expected):
        kiss = KISSInterface()
        encoded = kiss._build_frame(bytes([data[0]]), data[1:])
        assert encoded == expected

    def test_command_encoding(self):
        kiss = KISSInterface(tnc_address=0x0A)
        cmd_byte = kiss._encode_command(KISSCommand.POLL)
        assert cmd_byte == 0xAE  # 0xA0 (TNC 10) | 0x0E (POLL)

    def test_invalid_command(self):
        kiss = KISSInterface()
        with pytest.raises(KISSProtocolError):
            kiss.send_frame(b"", cmd=0xFF)  # Invalid command

class TestSerialKISSInterface:
    def test_serial_send(self, mock_serial):
        kiss = SerialKISSInterface("/dev/ttyUSB0")
        kiss._serial = mock_serial
        kiss.send_frame(b"test")
        mock_serial.write.assert_called_once_with(b"\xc0\x00test\xc0")

    def test_serial_receive(self, mock_serial):
        kiss = SerialKISSInterface("/dev/ttyUSB0")
        kiss._serial = mock_serial
        callback = Mock()
        kiss.register_rx_callback(callback)
        
        # Simulate received frame
        mock_serial.read.side_effect = [
            b"\xc0\x00test\xc0",
            b""  # EOF
        ]
        kiss.start()
        time.sleep(0.1)
        kiss.stop()
        
        callback.assert_called_once_with(b"test", 0x00)

class TestTCPKISSInterface:
    def test_tcp_send(self, mock_socket):
        kiss = TCPKISSInterface("localhost", 8001)
        kiss._socket = mock_socket
        kiss.send_frame(b"test")
        mock_socket.sendall.assert_called_once_with(b"\xc0\x00test\xc0")

    def test_tcp_reconnect(self, mock_socket):
        kiss = TCPKISSInterface("localhost", 8001, reconnect_interval=0.1)
        
        # First connection fails
        with patch('socket.socket.connect') as mock_connect:
            mock_connect.side_effect = ConnectionRefusedError
            assert not kiss._ensure_connected()
            
        # Then succeeds
        with patch('socket.socket.connect'):
            kiss._socket = mock_socket
            assert kiss._ensure_connected()

class TestMultiDrop:
    def test_poll_command(self):
        kiss = KISSInterface(tnc_address=1)
        callback = Mock()
        kiss.register_poll_callback(callback)
        
        # Simulate receiving poll command from TNC 2
        frame = kiss._build_frame(bytes([0x2E]), b"")  # TNC 2, POLL
        kiss._process_frame(frame)
        callback.assert_called_once_with(2)

    def test_multi_addressing(self):
        kiss = KISSInterface(tnc_address=5)
        
        # Send from TNC 3
        frame = kiss._build_frame(bytes([0x30]), b"data")  # TNC 3, DATA
        kiss._process_frame(frame[1:-1])  # Strip flags
        
        # Verify our KISS interface ignores frames not for us (unless broadcast)
        # Implementation would need rx_callback to verify

class TestErrorConditions:
    def test_serial_error(self, mock_serial):
        kiss = SerialKISSInterface("/dev/ttyUSB0")
        kiss._serial = mock_serial
        mock_serial.read.side_effect = serial.SerialException("Test error")
        
        with pytest.raises(TransportError):
            kiss.start()
            time.sleep(0.1)
            kiss.stop()

    def test_invalid_tnc_address(self):
        with pytest.raises(ValueError):
            KISSInterface(tnc_address=16)

if __name__ == "__main__":
    pytest.main([__file__])
