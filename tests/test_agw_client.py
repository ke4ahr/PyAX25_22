# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
tests/test_agw_client.py

Unit tests for AGWPEClient (interfaces/agw/client.py).

Uses a mock socket so no real network connection is needed.

Covers:
- AGWPEFrame attributes
- _send_frame() header construction
- Dispatch callbacks: on_frame, on_connected_data, on_outstanding, on_heard
- _parse_heard(): heard station list decoding
- _parse_memory(): memory usage decoding
- close(): clears connected flag
- send_ui(), send_connect(), send_disconnect(), send_connected_data()
"""

import struct
import pytest
from unittest.mock import MagicMock, patch

from pyax25_22.interfaces.agw.client import AGWPEClient, AGWPEFrame, _MAX_DATA_LEN
from pyax25_22.interfaces.agw.constants import (
    AGWPE_HEADER_SIZE, CALLSIGN_WIDTH,
    KIND_UNPROTO_DATA, KIND_CONN_DATA, KIND_OUTSTANDING, KIND_HEARD,
    KIND_EXTENDED_VER, KIND_MEMORY_USAGE, KIND_CONNECT_INC, KIND_DISC,
    KIND_REGISTER,
)
from pyax25_22.interfaces.agw.exceptions import AGWConnectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_header(
    data_kind: bytes,
    port: int = 0,
    call_from: str = "",
    call_to: str = "",
    data_len: int = 0,
) -> bytes:
    """Build a 36-byte AGWPE header for test injection."""
    hdr = bytearray(AGWPE_HEADER_SIZE)
    hdr[0:1] = data_kind
    struct.pack_into("<I", hdr, 4, port)
    hdr[8:18] = call_from.upper().ljust(CALLSIGN_WIDTH)[:CALLSIGN_WIDTH].encode("ascii")
    hdr[18:28] = call_to.upper().ljust(CALLSIGN_WIDTH)[:CALLSIGN_WIDTH].encode("ascii")
    struct.pack_into("<I", hdr, 28, data_len)
    return bytes(hdr)


def make_mock_client() -> AGWPEClient:
    """Return a pre-connected AGWPEClient with a mock socket."""
    client = AGWPEClient("127.0.0.1", 8000, "KE4AHR")
    client.sock = MagicMock()
    client.sock.sendall = MagicMock()
    client.connected = True
    return client


# ---------------------------------------------------------------------------
# AGWPEFrame tests
# ---------------------------------------------------------------------------

def test_agwpe_frame_defaults():
    """AGWPEFrame has sensible defaults."""
    f = AGWPEFrame()
    assert f.data_kind == b""
    assert f.port == 0
    assert f.call_from == ""
    assert f.call_to == ""
    assert f.data_len == 0
    assert f.data == b""


def test_agwpe_frame_repr():
    """AGWPEFrame repr includes kind, port, call_from, call_to."""
    f = AGWPEFrame()
    f.data_kind = b"D"
    f.port = 0
    f.call_from = "KE4AHR"
    f.call_to = "APRS"
    text = repr(f)
    assert "KE4AHR" in text
    assert "APRS" in text


# ---------------------------------------------------------------------------
# _send_frame() tests
# ---------------------------------------------------------------------------

def test_send_frame_not_connected_raises():
    """_send_frame() raises AGWConnectionError when not connected."""
    client = AGWPEClient()
    client.connected = False
    with pytest.raises(AGWConnectionError):
        client._send_frame(KIND_UNPROTO_DATA)


def test_send_frame_builds_correct_header():
    """_send_frame() sends exactly AGWPE_HEADER_SIZE + data_len bytes."""
    client = make_mock_client()
    data = b"\x01\x02\x03"
    client._send_frame(KIND_UNPROTO_DATA, port=0, call_from="KE4AHR",
                       call_to="APRS", data=data)
    assert client.sock.sendall.called
    sent = client.sock.sendall.call_args[0][0]
    assert len(sent) == AGWPE_HEADER_SIZE + len(data)


def test_send_frame_kind_in_header():
    """The data_kind byte appears at offset 0 in the header."""
    client = make_mock_client()
    client._send_frame(KIND_REGISTER, call_from="KE4AHR")
    sent = client.sock.sendall.call_args[0][0]
    assert sent[0:1] == KIND_REGISTER


def test_send_frame_port_in_header():
    """Port number is encoded as little-endian uint32 at offset 4."""
    client = make_mock_client()
    client._send_frame(KIND_UNPROTO_DATA, port=3)
    sent = client.sock.sendall.call_args[0][0]
    assert struct.unpack("<I", sent[4:8])[0] == 3


# ---------------------------------------------------------------------------
# High-level send methods
# ---------------------------------------------------------------------------

def test_send_ui_includes_pid():
    """send_ui() prepends the PID byte to info in data field."""
    client = make_mock_client()
    client.send_ui(port=0, dest="APRS", src="KE4AHR", pid=0xF0, info=b"Hello")
    sent = client.sock.sendall.call_args[0][0]
    payload = sent[AGWPE_HEADER_SIZE:]
    assert payload[0] == 0xF0
    assert payload[1:] == b"Hello"


def test_send_connect():
    """send_connect() uses KIND_CONNECT data_kind."""
    client = make_mock_client()
    client.send_connect(port=0, dest="W1AW")
    sent = client.sock.sendall.call_args[0][0]
    assert sent[0:1] == b"C"


def test_send_disconnect():
    """send_disconnect() uses KIND_DISC data_kind."""
    client = make_mock_client()
    client.send_disconnect(port=0, dest="W1AW")
    sent = client.sock.sendall.call_args[0][0]
    assert sent[0:1] == b"d"


def test_send_connected_data():
    """send_connected_data() uses KIND_CONN_DATA and our callsign."""
    client = make_mock_client()
    client.send_connected_data(port=0, dest="W1AW", data=b"Info")
    sent = client.sock.sendall.call_args[0][0]
    assert sent[0:1] == KIND_CONN_DATA
    payload = sent[AGWPE_HEADER_SIZE:]
    assert payload == b"Info"


# ---------------------------------------------------------------------------
# _dispatch() callback tests
# ---------------------------------------------------------------------------

def test_dispatch_on_frame_called():
    """on_frame callback is invoked for unproto data frame."""
    client = AGWPEClient()
    frames_received = []
    client.on_frame = lambda f: frames_received.append(f)

    frame = AGWPEFrame()
    frame.data_kind = KIND_UNPROTO_DATA
    client._dispatch(frame)
    assert len(frames_received) == 1


def test_dispatch_on_connected_data_called():
    """on_connected_data callback receives port, call_from, data.

    In AGWPE, 'd' (lowercase) carries connected session data.
    'D' (uppercase) is unproto data and is dispatched to on_frame.
    """
    client = AGWPEClient()
    received = []
    client.on_connected_data = lambda port, call, data: received.append((port, call, data))

    frame = AGWPEFrame()
    frame.data_kind = b"d"   # lowercase 'd' = connected data in AGWPE
    frame.port = 0
    frame.call_from = "W1AW"
    frame.data = b"test payload"
    client._dispatch(frame)
    assert len(received) == 1
    assert received[0] == (0, "W1AW", b"test payload")


def test_dispatch_on_outstanding_called():
    """on_outstanding callback receives port and frame count."""
    client = AGWPEClient()
    received = []
    client.on_outstanding = lambda port, count: received.append((port, count))

    frame = AGWPEFrame()
    frame.data_kind = KIND_OUTSTANDING
    frame.port = 1
    frame.data = struct.pack("<I", 42)
    client._dispatch(frame)
    assert received[0] == (1, 42)


def test_dispatch_on_connect_called():
    """on_connect callback is called for incoming connection notification."""
    client = AGWPEClient()
    received = []
    client.on_connect = lambda port, call: received.append((port, call))

    frame = AGWPEFrame()
    frame.data_kind = KIND_CONNECT_INC
    frame.port = 0
    frame.call_from = "VK2TDS"
    client._dispatch(frame)
    assert received[0] == (0, "VK2TDS")


def test_dispatch_on_disconnect_called():
    """on_disconnect callback is called for disconnect notification."""
    client = AGWPEClient()
    received = []
    client.on_disconnect = lambda port, call: received.append((port, call))

    frame = AGWPEFrame()
    frame.data_kind = KIND_DISC
    frame.port = 0
    frame.call_from = "VK2TDS"
    client._dispatch(frame)
    assert received[0] == (0, "VK2TDS")


def test_dispatch_extended_version():
    """on_extended_version callback receives version string."""
    client = AGWPEClient()
    received = []
    client.on_extended_version = lambda v: received.append(v)

    frame = AGWPEFrame()
    frame.data_kind = KIND_EXTENDED_VER
    frame.data = b"2.0.26\x00"
    client._dispatch(frame)
    assert "2.0.26" in received[0]


def test_dispatch_memory_usage():
    """on_memory_usage callback receives dict with free_kb and used_kb."""
    client = AGWPEClient()
    received = []
    client.on_memory_usage = lambda m: received.append(m)

    frame = AGWPEFrame()
    frame.data_kind = KIND_MEMORY_USAGE
    frame.data = struct.pack("<II", 512 * 1024, 256 * 1024)  # 512 KB free, 256 KB used
    client._dispatch(frame)
    assert received[0]["free_kb"] == 512
    assert received[0]["used_kb"] == 256


# ---------------------------------------------------------------------------
# _parse_heard()
# ---------------------------------------------------------------------------

def test_parse_heard_empty():
    """Empty data returns empty list."""
    client = AGWPEClient()
    assert client._parse_heard(b"") == []


def test_parse_heard_one_entry():
    """One 14-byte entry is parsed correctly."""
    client = AGWPEClient()
    call = b"KE4AHR    "  # 10 bytes
    ts = struct.pack("<I", 1700000000)
    data = call + ts
    result = client._parse_heard(data)
    assert len(result) == 1
    assert result[0]["callsign"] == "KE4AHR"
    assert result[0]["last_heard"] == 1700000000


# ---------------------------------------------------------------------------
# _parse_memory()
# ---------------------------------------------------------------------------

def test_parse_memory_too_short():
    """Short payload returns zeros."""
    client = AGWPEClient()
    result = client._parse_memory(b"\x00\x01\x02")  # < 8 bytes
    assert result == {"free_kb": 0, "used_kb": 0}


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

def test_close_clears_connected():
    """close() sets connected to False."""
    client = make_mock_client()
    client.close()
    assert not client.connected
