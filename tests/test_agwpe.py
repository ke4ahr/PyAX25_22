# tests/test_agwpe.py
"""
Comprehensive AGWPE Interface Tests

Covers:
- Header parsing/construction
- Connection/registration
- Frame transmission
- Error conditions

License: LGPLv3.0
Copyright (C) 2025-2026 Kris Kirby, KE4AHR
"""

import pytest
import struct
import logging
from unittest.mock import Mock, patch
from pyax25_22.interfaces.agwpe import (
    AGWPEClient,
    AGWHeader,
    AGWFrameType,
    AGWProtocolError,
    TransportError
)

@pytest.fixture
def agw_client():
    return AGWPEClient(host="localhost", callsign="TEST")

class TestAGWPEHeader:
    def test_header_packing(self):
        header = AGWHeader(
            port=1,
            data_kind=AGWFrameType.DATA,
            pid=0xF0,
            call_from=b"FROMCALL",
            call_to=b"TOCALL ",
            data_len=10
        )
        packed = header.pack()
        
        assert len(packed) == 36
        unpacked = AGWHeader.unpack(packed)
        assert unpacked.port == 1
        assert unpacked.data_kind == AGWFrameType.DATA
        assert unpacked.call_from.strip() == b"FROMCALL"

    def test_header_short(self):
        with pytest.raises(AGWProtocolError):
            AGWHeader.unpack(b'\x00' * 35)

class TestAGWConnect:
    def test_successful_connection(self, agw_client):
        mock_socket = Mock()
        mock_socket.recv.return_value = b'X' + b'\x00'*35  # Success response
        
        with patch('socket.socket', return_value=mock_socket):
            agw_client.connect()
            
        # Verify registration frame sent
        sent_data = mock_socket.sendall.call_args[0][0]
        assert len(sent_data) == 36
        assert sent_data[4] == AGWFrameType.REGISTER

    def test_connection_failure(self, agw_client):
        with patch('socket.socket') as mock_socket:
            mock_socket.return_value.connect.side_effect = ConnectionRefusedError
            with pytest.raises(TransportError):
                agw_client.connect()

class TestAGWSend:
    def test_send_data_frame(self, agw_client):
        mock_socket = Mock()
        agw_client._socket = mock_socket
        
        agw_client.send_frame(b"payload", AGWFrameType.DATA, dest="DEST")
        
        sent_data = mock_socket.sendall.call_args[0][0]
        assert len(sent_data) == 36 + 6  # Header + payload
        assert sent_data[4] == AGWFrameType.DATA
        assert b"DEST" in sent_data

class TestAGWReceive:
    def test_receive_frame(self, agw_client):
        mock_socket = Mock()
        agw_client._socket = mock_socket
        callback = Mock()
        agw_client.register_frame_callback(AGWFrameType.DATA, callback)
        
        # Simulate received DATA frame header
        header = struct.pack(
            AGWPEClient.HEADER_FORMAT,
            0,  # Cookie
            b'',  # Reserved
            0,  # Port
            AGWFrameType.DATA,  # Type
            0, 0, 0, 0, 0,  # Reserved
            b"FROMCALL",
            b"TOCALL  ",
            5  # Data length
        )
        mock_socket.recv.side_effect = [
            header,
            b"hello"  # Payload
        ]
        
        agw_client._receive_loop()
        callback.assert_called_once_with(b"hello", "FROMCALL", "TOCALL")

class TestErrorHandling:
    def test_invalid_registration(self, agw_client):
        mock_socket = Mock()
        mock_socket.recv.return_value = b'\x00'*36  # Bad response
        
        with patch('socket.socket', return_value=mock_socket):
            with pytest.raises(AGWProtocolError):
                agw_client.connect()

    def test_send_disconnected(self, agw_client):
        with pytest.raises(TransportError):
            agw_client.send_frame(b"test")

if __name__ == "__main__":
    pytest.main([__file__])
