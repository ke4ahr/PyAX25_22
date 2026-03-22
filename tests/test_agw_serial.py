# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
tests/test_agw_serial.py

Unit tests for the AGWSerial bridge (interfaces/agw/serial.py).

Covers:
- AX.25 address encode/decode helpers
- UI frame construction
- AX.25 address field parsing
- _AGWClientConn.send_frame builds correct AGWPE header
- AGWSerial._handle_client_frame dispatches correctly (with mock KISS)
- AGWSerial._on_kiss_frame parses and forwards to monitoring clients
- Version and port info responses
"""

import socket
import struct
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call

from pyax25_22.interfaces.agw.serial import (
    _encode_ax25_addr,
    _decode_ax25_addr,
    _build_ui_frame,
    _parse_ax25_addresses,
    _AGWClientConn,
    AGWSerial,
)
from pyax25_22.interfaces.agw.constants import (
    AGWPE_HEADER_SIZE,
    KIND_VERSION,
    KIND_REGISTER,
    KIND_UNREGISTER,
    KIND_PORT_INFO,
    KIND_PORT_CAPS,
    KIND_ENABLE_MON,
    KIND_RAW_MON,
    KIND_RAW_SEND,
    KIND_UNPROTO_DATA,
)
from pyax25_22.interfaces.agw.client import AGWPEFrame


# ---------------------------------------------------------------------------
# AX.25 address helpers
# ---------------------------------------------------------------------------

class TestEncodeAX25Addr(unittest.TestCase):

    def test_encode_basic_callsign(self):
        """Encode KE4AHR produces correct shifted bytes."""
        encoded = _encode_ax25_addr("KE4AHR")
        assert len(encoded) == 7
        # First 6 bytes: each char shifted left 1
        expected_chars = [ord(c) << 1 for c in "KE4AHR"]
        assert list(encoded[:6]) == expected_chars

    def test_encode_last_flag_set(self):
        """last=True sets bit 0 of the SSID byte."""
        encoded_last = _encode_ax25_addr("KE4AHR", last=True)
        encoded_nolast = _encode_ax25_addr("KE4AHR", last=False)
        assert encoded_last[6] & 0x01 == 1
        assert encoded_nolast[6] & 0x01 == 0

    def test_encode_ssid(self):
        """SSID is encoded in bits 1-4 of byte 7."""
        for ssid in range(16):
            encoded = _encode_ax25_addr("W1AW", ssid=ssid)
            decoded_ssid = (encoded[6] >> 1) & 0x0F
            assert decoded_ssid == ssid, f"SSID mismatch for {ssid}"

    def test_encode_short_callsign_padded(self):
        """Short callsigns are space-padded to 6 chars."""
        encoded = _encode_ax25_addr("KA1B")
        # Bytes 4 and 5 should be space (0x20 << 1 = 0x40)
        assert encoded[4] == ord(" ") << 1
        assert encoded[5] == ord(" ") << 1


class TestDecodeAX25Addr(unittest.TestCase):

    def test_roundtrip_callsign(self):
        """Encode then decode recovers the original callsign."""
        for call in ["KE4AHR", "W1AW", "KA1BCG", "VK2ABC"]:
            encoded = _encode_ax25_addr(call)
            decoded, ssid, last = _decode_ax25_addr(encoded)
            assert decoded == call.upper().rstrip(), f"Mismatch for {call}"
            assert ssid == 0

    def test_roundtrip_ssid(self):
        """Encode then decode recovers the SSID."""
        encoded = _encode_ax25_addr("KE4AHR", ssid=7, last=True)
        decoded, ssid, last = _decode_ax25_addr(encoded)
        assert ssid == 7
        assert last is True

    def test_decode_with_offset(self):
        """offset parameter reads from the correct position."""
        buf = bytes(7) + _encode_ax25_addr("W1AW", ssid=3, last=True)
        call, ssid, last = _decode_ax25_addr(buf, offset=7)
        assert call == "W1AW"
        assert ssid == 3
        assert last is True


# ---------------------------------------------------------------------------
# UI frame construction
# ---------------------------------------------------------------------------

class TestBuildUIFrame(unittest.TestCase):

    def test_minimum_frame_length(self):
        """UI frame with empty info has 14 + 2 bytes (dest+src+ctrl+pid)."""
        frame = _build_ui_frame("APRS", "KE4AHR", pid=0xF0, info=b"")
        assert len(frame) == 14 + 2   # two 7-byte addrs + ctrl + pid

    def test_control_byte_is_03(self):
        """Control byte is always 0x03 (UI)."""
        frame = _build_ui_frame("APRS", "KE4AHR", pid=0xF0, info=b"test")
        ctrl = frame[14]
        assert ctrl == 0x03

    def test_pid_byte_correct(self):
        """PID byte matches what was given."""
        for pid in [0xF0, 0xCF, 0x00]:
            frame = _build_ui_frame("APRS", "KE4AHR", pid=pid, info=b"x")
            assert frame[15] == pid, f"PID mismatch for 0x{pid:02X}"

    def test_info_appended(self):
        """Info bytes are appended after PID."""
        info = b"Hello, world!"
        frame = _build_ui_frame("APRS", "KE4AHR", pid=0xF0, info=info)
        assert frame[16:] == info

    def test_dest_last_bit_clear_when_digis_absent(self):
        """Destination address end-of-address bit is NOT set (it's first)."""
        frame = _build_ui_frame("APRS", "KE4AHR", pid=0xF0, info=b"")
        assert frame[6] & 0x01 == 0   # dest last bit clear

    def test_src_last_bit_set_when_no_digis(self):
        """Source address end-of-address bit is set when there are no digi."""
        frame = _build_ui_frame("APRS", "KE4AHR", pid=0xF0, info=b"")
        assert frame[13] & 0x01 == 1   # src last bit set


# ---------------------------------------------------------------------------
# AX.25 address field parsing
# ---------------------------------------------------------------------------

class TestParseAX25Addresses(unittest.TestCase):

    def _make_frame(self, dest, src, ctrl=0x03, pid=0xF0, info=b""):
        return _build_ui_frame(dest, src, pid=pid, info=info)

    def test_parse_dest_and_src(self):
        """Parsed dest and src match what was encoded."""
        frame = self._make_frame("APRS", "KE4AHR", info=b"test")
        dest, src, digis, hdr_len = _parse_ax25_addresses(frame)
        assert dest == "APRS"
        assert src == "KE4AHR"
        assert digis == []
        assert hdr_len == 14

    def test_parse_too_short(self):
        """Frames shorter than 14 bytes return None for dest."""
        dest, src, digis, hdr_len = _parse_ax25_addresses(b"\x00" * 10)
        assert dest is None

    def test_roundtrip_via_encode_decode(self):
        """Encode a frame, parse it back, recover callsigns."""
        frame = self._make_frame("CQ", "KA1B", info=b"de KA1B")
        dest, src, digis, hdr_len = _parse_ax25_addresses(frame)
        assert dest == "CQ"
        assert src == "KA1B"
        assert hdr_len == 14


# ---------------------------------------------------------------------------
# _AGWClientConn
# ---------------------------------------------------------------------------

class TestAGWClientConn(unittest.TestCase):

    def _make_conn(self):
        """Return a _AGWClientConn with a mock socket."""
        mock_sock = MagicMock()
        conn = _AGWClientConn(mock_sock, ("127.0.0.1", 12345))
        return conn, mock_sock

    def test_send_frame_header_format(self):
        """send_frame sends a 36-byte header with correct fields."""
        conn, mock_sock = self._make_conn()
        conn.send_frame(
            data_kind=KIND_VERSION,
            port=0,
            call_from="KE4AHR",
            call_to="APRS",
            data=b"hello",
        )
        mock_sock.sendall.assert_called_once()
        packet = mock_sock.sendall.call_args[0][0]
        # Header
        assert len(packet) == AGWPE_HEADER_SIZE + 5
        assert packet[0:1] == KIND_VERSION
        assert struct.unpack("<I", packet[28:32])[0] == 5
        assert packet[8:18] == b"KE4AHR    "
        assert packet[18:28] == b"APRS      "

    def test_send_frame_returns_false_on_error(self):
        """send_frame returns False if socket raises OSError."""
        conn, mock_sock = self._make_conn()
        mock_sock.sendall.side_effect = OSError("broken pipe")
        result = conn.send_frame(data_kind=KIND_VERSION)
        assert result is False

    def test_send_frame_returns_true_on_success(self):
        """send_frame returns True on success."""
        conn, mock_sock = self._make_conn()
        result = conn.send_frame(data_kind=KIND_VERSION, data=b"test")
        assert result is True


# ---------------------------------------------------------------------------
# AGWSerial frame handling (with mocked KISSSerial)
# ---------------------------------------------------------------------------

class TestAGWSerialFrameHandling(unittest.TestCase):
    """Test _handle_client_frame without starting real threads or serial."""

    def _make_bridge(self):
        """Return AGWSerial with a mock KISS and a mock client connection."""
        with patch("pyax25_22.interfaces.agw.serial.KISSSerial"):
            bridge = AGWSerial("/dev/null", 9600)
        bridge._kiss = MagicMock()
        mock_sock = MagicMock()
        conn = _AGWClientConn(mock_sock, ("127.0.0.1", 9999))
        return bridge, conn, mock_sock

    def _make_frame(self, kind, call_from="KE4AHR", call_to="APRS", data=b""):
        f = AGWPEFrame()
        f.data_kind = kind
        f.call_from = call_from
        f.call_to = call_to
        f.data = data
        f.data_len = len(data)
        f.port = 0
        return f

    # ------ VERSION ------

    def test_version_request_sends_response(self):
        """'R' frame produces a version response back to client."""
        bridge, conn, sock = self._make_bridge()
        bridge._handle_client_frame(conn, self._make_frame(KIND_VERSION))
        sock.sendall.assert_called_once()
        packet = sock.sendall.call_args[0][0]
        assert packet[0:1] == KIND_VERSION

    # ------ PORT INFO ------

    def test_port_info_sends_g_response(self):
        """'G' request sends a 'g' port caps response."""
        bridge, conn, sock = self._make_bridge()
        bridge._handle_client_frame(conn, self._make_frame(KIND_PORT_INFO))
        sock.sendall.assert_called_once()
        packet = sock.sendall.call_args[0][0]
        assert packet[0:1] == KIND_PORT_CAPS

    # ------ CALLSIGN REGISTRATION ------

    def test_register_callsign_sends_ack(self):
        """'X' registration stores callsign and sends ACK."""
        bridge, conn, sock = self._make_bridge()
        bridge._handle_client_frame(
            conn, self._make_frame(KIND_REGISTER, call_from="KE4AHR")
        )
        assert "KE4AHR" in conn.registered_calls
        sock.sendall.assert_called_once()
        packet = sock.sendall.call_args[0][0]
        assert packet[0:1] == KIND_REGISTER

    def test_unregister_callsign_removes_it(self):
        """'x' removes a previously registered callsign."""
        bridge, conn, sock = self._make_bridge()
        conn.registered_calls.add("KE4AHR")
        bridge._handle_client_frame(
            conn, self._make_frame(KIND_UNREGISTER, call_from="KE4AHR")
        )
        assert "KE4AHR" not in conn.registered_calls

    # ------ MONITORING ------

    def test_enable_monitor_sets_flag(self):
        """'M' sets the monitoring flag on the client."""
        bridge, conn, sock = self._make_bridge()
        assert not conn.monitoring
        bridge._handle_client_frame(conn, self._make_frame(KIND_ENABLE_MON))
        assert conn.monitoring

    # ------ RAW SEND ------

    def test_raw_send_calls_kiss(self):
        """'K' raw send passes data directly to KISSSerial.send()."""
        bridge, conn, sock = self._make_bridge()
        ax25 = b"\x82\x84\xA6\xA8" + bytes(10)
        bridge._handle_client_frame(
            conn, self._make_frame(KIND_RAW_SEND, data=ax25)
        )
        bridge._kiss.send.assert_called_once_with(ax25, cmd=0x00)

    def test_raw_send_empty_data_no_kiss_call(self):
        """'K' with empty data does not call KISS."""
        bridge, conn, sock = self._make_bridge()
        bridge._handle_client_frame(
            conn, self._make_frame(KIND_RAW_SEND, data=b"")
        )
        bridge._kiss.send.assert_not_called()

    # ------ UNPROTO UI SEND ------

    def test_ui_send_builds_ax25_frame(self):
        """'D' sends a valid AX.25 UI frame via KISS."""
        bridge, conn, sock = self._make_bridge()
        # data[0] = PID 0xF0, data[1:] = info
        ui_data = bytes([0xF0]) + b"Hello"
        bridge._handle_client_frame(
            conn, self._make_frame(
                KIND_UNPROTO_DATA, call_from="KE4AHR",
                call_to="APRS", data=ui_data
            )
        )
        bridge._kiss.send.assert_called_once()
        ax25_sent = bridge._kiss.send.call_args[0][0]
        # Minimum: 2 addrs (14 bytes) + ctrl + pid + info
        assert len(ax25_sent) >= 16
        assert ax25_sent[14] == 0x03   # control
        assert ax25_sent[15] == 0xF0   # PID


# ---------------------------------------------------------------------------
# AGWSerial TNC -> client forwarding
# ---------------------------------------------------------------------------

class TestAGWSerialForwarding(unittest.TestCase):
    """Test _on_kiss_frame forwarding to connected clients."""

    def _make_bridge_with_client(self, monitoring=True):
        with patch("pyax25_22.interfaces.agw.serial.KISSSerial"):
            bridge = AGWSerial("/dev/null", 9600)
        bridge._kiss = MagicMock()
        mock_sock = MagicMock()
        conn = _AGWClientConn(mock_sock, ("127.0.0.1", 9999))
        conn.monitoring = monitoring
        bridge._clients[mock_sock] = conn
        return bridge, conn, mock_sock

    def _make_ui_ax25(self, dest="APRS", src="KE4AHR", info=b"test"):
        return _build_ui_frame(dest, src, pid=0xF0, info=info)

    def test_monitoring_client_receives_raw_frame(self):
        """Monitoring clients receive 'K' frame for every KISS data frame."""
        bridge, conn, sock = self._make_bridge_with_client(monitoring=True)
        ax25 = self._make_ui_ax25()
        bridge._on_kiss_frame(0x00, ax25)
        sock.sendall.assert_called()
        # At least one call should be a 'K' frame
        packets = [c[0][0] for c in sock.sendall.call_args_list]
        kinds = [p[0:1] for p in packets]
        assert KIND_RAW_MON in kinds

    def test_ui_frame_also_forwarded_as_d(self):
        """UI frames also produce a 'D' frame for clients."""
        bridge, conn, sock = self._make_bridge_with_client(monitoring=True)
        ax25 = self._make_ui_ax25(info=b"Hello APRS")
        bridge._on_kiss_frame(0x00, ax25)
        packets = [c[0][0] for c in sock.sendall.call_args_list]
        kinds = [p[0:1] for p in packets]
        assert KIND_UNPROTO_DATA in kinds

    def test_non_monitoring_client_skips_k(self):
        """Non-monitoring clients do not receive 'K' frames."""
        bridge, conn, sock = self._make_bridge_with_client(monitoring=False)
        ax25 = self._make_ui_ax25()
        bridge._on_kiss_frame(0x00, ax25)
        packets = [c[0][0] for c in sock.sendall.call_args_list]
        kinds = [p[0:1] for p in packets]
        assert KIND_RAW_MON not in kinds

    def test_non_data_cmd_ignored(self):
        """Non-DATA KISS command bytes (e.g. TXDELAY) are not forwarded."""
        bridge, conn, sock = self._make_bridge_with_client(monitoring=True)
        bridge._on_kiss_frame(0x01, b"txdelay")   # CMD_TXDELAY, not data
        sock.sendall.assert_not_called()

    def test_too_short_frame_ignored(self):
        """KISS frames shorter than 15 bytes are silently dropped."""
        bridge, conn, sock = self._make_bridge_with_client(monitoring=True)
        bridge._on_kiss_frame(0x00, b"\x00" * 10)
        sock.sendall.assert_not_called()

    def test_dead_client_removed(self):
        """A client whose socket raises OSError is removed from client list."""
        bridge, conn, sock = self._make_bridge_with_client(monitoring=True)
        sock.sendall.side_effect = OSError("broken pipe")
        ax25 = self._make_ui_ax25()
        bridge._on_kiss_frame(0x00, ax25)
        assert sock not in bridge._clients


if __name__ == "__main__":
    unittest.main()
