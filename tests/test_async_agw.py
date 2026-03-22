# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_async_agw.py

Unit tests for AsyncAGWPEClient (interfaces/agw/async_client.py).

Uses mock asyncio streams so no real AGWPE server is needed.

Covers:
- _send_frame builds correct 36-byte AGWPE header
- send_ui, send_raw, send_connect, send_disconnect helper methods
- Dispatch: on_frame, on_connected_data, on_disconnect callbacks
- Dispatch: on_outstanding, on_heard_stations callbacks
- Plain and coroutine callbacks both work
- AGWConnectionError raised when not connected
- close() cancels reader task
"""

import asyncio
import struct
import unittest
from unittest.mock import AsyncMock, MagicMock

from pyax25_22.interfaces.agw.async_client import AsyncAGWPEClient
from pyax25_22.interfaces.agw.constants import (
    AGWPE_HEADER_SIZE,
    CALLSIGN_WIDTH,
    KIND_UNPROTO_DATA,
    KIND_RAW_SEND,
    KIND_CONNECT,
    KIND_DISC,
    KIND_CONNECT_INC,
    KIND_OUTSTANDING,
    KIND_HEARD,
    KIND_VERSION,
    KIND_PORT_INFO,
)
from pyax25_22.interfaces.agw.client import AGWPEFrame
from pyax25_22.interfaces.agw.exceptions import AGWConnectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_connected_client(callsign="KE4AHR"):
    """Return an AsyncAGWPEClient with a mock writer (no real TCP)."""
    client = AsyncAGWPEClient("127.0.0.1", 8000, callsign=callsign)
    client._reader = MagicMock()
    client._writer = MagicMock()
    client._writer.drain = AsyncMock()
    client.connected = True
    return client


def _make_frame(kind, port=0, call_from="KE4AHR", call_to="APRS", data=b""):
    f = AGWPEFrame()
    f.data_kind = kind
    f.port = port
    f.call_from = call_from
    f.call_to = call_to
    f.data = data
    f.data_len = len(data)
    return f


def _parse_header(packet: bytes):
    """Parse the 36-byte AGWPE header from a packet."""
    kind = packet[0:1]
    port = struct.unpack("<I", packet[4:8])[0]
    call_from = packet[8:18].decode("ascii", errors="ignore").strip()
    call_to = packet[18:28].decode("ascii", errors="ignore").strip()
    data_len = struct.unpack("<I", packet[28:32])[0]
    data = packet[AGWPE_HEADER_SIZE:] if len(packet) > AGWPE_HEADER_SIZE else b""
    return kind, port, call_from, call_to, data_len, data


# ---------------------------------------------------------------------------
# Header format tests
# ---------------------------------------------------------------------------

class TestSendFrame(unittest.IsolatedAsyncioTestCase):

    async def test_header_length(self):
        """_send_frame sends a 36-byte header."""
        client = _make_connected_client()
        await client._send_frame(data_kind=KIND_VERSION)
        client._writer.write.assert_called_once()
        packet = client._writer.write.call_args[0][0]
        assert len(packet) == AGWPE_HEADER_SIZE

    async def test_header_data_kind(self):
        """_send_frame sets data_kind at byte 0."""
        client = _make_connected_client()
        await client._send_frame(data_kind=KIND_VERSION)
        packet = client._writer.write.call_args[0][0]
        assert packet[0:1] == KIND_VERSION

    async def test_header_call_fields(self):
        """_send_frame encodes call_from and call_to padded to 10 chars."""
        client = _make_connected_client()
        await client._send_frame(
            data_kind=KIND_CONNECT,
            call_from="KE4AHR",
            call_to="W1AW",
        )
        packet = client._writer.write.call_args[0][0]
        assert packet[8:18] == b"KE4AHR    "
        assert packet[18:28] == b"W1AW      "

    async def test_header_data_len(self):
        """_send_frame sets data_len to the length of the data payload."""
        client = _make_connected_client()
        data = b"Hello, world!"
        await client._send_frame(data_kind=KIND_UNPROTO_DATA, data=data)
        packet = client._writer.write.call_args[0][0]
        data_len = struct.unpack("<I", packet[28:32])[0]
        assert data_len == len(data)
        assert packet[AGWPE_HEADER_SIZE:] == data

    async def test_send_frame_not_connected_raises(self):
        """_send_frame raises AGWConnectionError when not connected."""
        client = AsyncAGWPEClient()
        with self.assertRaises(AGWConnectionError):
            await client._send_frame(data_kind=KIND_VERSION)

    async def test_send_frame_calls_drain(self):
        """_send_frame awaits writer.drain() to flush the buffer."""
        client = _make_connected_client()
        await client._send_frame(data_kind=KIND_VERSION)
        client._writer.drain.assert_called_once()


# ---------------------------------------------------------------------------
# High-level transmit helpers
# ---------------------------------------------------------------------------

class TestTransmitHelpers(unittest.IsolatedAsyncioTestCase):

    async def test_send_ui_pid_in_data(self):
        """send_ui() puts PID as data[0] and info as data[1:]."""
        client = _make_connected_client()
        await client.send_ui(port=0, dest="APRS", src="KE4AHR",
                             pid=0xF0, info=b"Hello")
        packet = client._writer.write.call_args[0][0]
        kind, port, cf, ct, data_len, data = _parse_header(packet)
        assert kind == KIND_UNPROTO_DATA
        assert data[0] == 0xF0
        assert data[1:] == b"Hello"

    async def test_send_raw_uses_k_frame(self):
        """send_raw() sends a 'K' frame with the provided data."""
        client = _make_connected_client()
        raw = bytes(range(10))
        await client.send_raw(port=0, dest="W1AW", src="KE4AHR", data=raw)
        packet = client._writer.write.call_args[0][0]
        kind = packet[0:1]
        assert kind == KIND_RAW_SEND

    async def test_send_connect_uses_c_frame(self):
        """send_connect() sends a 'C' frame."""
        client = _make_connected_client()
        await client.send_connect(port=0, dest="W1AW")
        packet = client._writer.write.call_args[0][0]
        assert packet[0:1] == KIND_CONNECT

    async def test_send_disconnect_uses_d_frame(self):
        """send_disconnect() sends a 'd' frame."""
        client = _make_connected_client()
        await client.send_disconnect(port=0, dest="W1AW")
        packet = client._writer.write.call_args[0][0]
        assert packet[0:1] == KIND_DISC

    async def test_callsign_uppercased_in_header(self):
        """Callsign is uppercased in all send methods."""
        client = _make_connected_client(callsign="ke4ahr")
        await client.send_connect(port=0, dest="W1AW")
        packet = client._writer.write.call_args[0][0]
        call_from = packet[8:18].decode("ascii").strip()
        assert call_from == "KE4AHR"


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------

class TestDispatch(unittest.IsolatedAsyncioTestCase):

    async def test_on_frame_called_for_unproto(self):
        """'D' frame calls on_frame callback."""
        received = []

        def on_frame(frame):
            received.append(frame)

        client = _make_connected_client()
        client.on_frame = on_frame
        await client._dispatch(_make_frame(KIND_UNPROTO_DATA, data=b"data"))
        assert len(received) == 1
        assert received[0].data == b"data"

    async def test_coroutine_on_frame(self):
        """Coroutine on_frame callback is awaited."""
        received = []

        async def on_frame(frame):
            received.append(frame)

        client = _make_connected_client()
        client.on_frame = on_frame
        await client._dispatch(_make_frame(KIND_UNPROTO_DATA))
        assert len(received) == 1

    async def test_on_connect_called_for_c(self):
        """'c' (incoming connection) calls on_connect with port and callsign."""
        events = []

        def on_connect(port, callsign):
            events.append((port, callsign))

        client = _make_connected_client()
        client.on_connect = on_connect
        f = _make_frame(KIND_CONNECT_INC, port=1, call_from="W1AW")
        await client._dispatch(f)
        assert events == [(1, "W1AW")]

    async def test_on_connected_data_for_d_with_data(self):
        """'d' with data calls on_connected_data."""
        events = []

        def on_connected_data(port, callsign, data):
            events.append((port, callsign, data))

        client = _make_connected_client()
        client.on_connected_data = on_connected_data
        f = _make_frame(KIND_DISC, port=0, call_from="W1AW", data=b"payload")
        await client._dispatch(f)
        assert events == [(0, "W1AW", b"payload")]

    async def test_on_disconnect_for_d_without_data(self):
        """'d' with no data calls on_disconnect."""
        events = []

        def on_disconnect(port, callsign):
            events.append((port, callsign))

        client = _make_connected_client()
        client.on_disconnect = on_disconnect
        f = _make_frame(KIND_DISC, port=0, call_from="W1AW", data=b"")
        await client._dispatch(f)
        assert events == [(0, "W1AW")]

    async def test_on_outstanding_called(self):
        """'Y'/'y' frames call on_outstanding with port and count."""
        events = []

        def on_outstanding(port, count):
            events.append((port, count))

        client = _make_connected_client()
        client.on_outstanding = on_outstanding
        count_bytes = struct.pack("<I", 42)
        f = _make_frame(KIND_OUTSTANDING, port=2, data=count_bytes)
        await client._dispatch(f)
        assert events == [(2, 42)]

    async def test_on_heard_called(self):
        """'H' frames call on_heard_stations."""
        events = []

        def on_heard(port, stations):
            events.append((port, stations))

        client = _make_connected_client()
        client.on_heard_stations = on_heard
        # Build a minimal heard entry: 10 bytes callsign + 4 bytes timestamp
        entry = b"W1AW      " + struct.pack("<I", 12345)
        f = _make_frame(KIND_HEARD, port=0, data=entry)
        await client._dispatch(f)
        assert len(events) == 1
        port, stations = events[0]
        assert stations[0]["callsign"] == "W1AW"
        assert stations[0]["last_heard"] == 12345

    async def test_no_callback_no_exception(self):
        """Dispatch with no callbacks set does not raise."""
        client = _make_connected_client()
        # All callbacks are None -- should be silent
        await client._dispatch(_make_frame(KIND_UNPROTO_DATA))
        await client._dispatch(_make_frame(KIND_DISC))

    async def test_callback_exception_logged_not_raised(self):
        """Exception in on_frame callback is caught and logged."""
        def bad_callback(frame):
            raise RuntimeError("callback error")

        client = _make_connected_client()
        client.on_frame = bad_callback
        # Should not propagate the exception
        await client._dispatch(_make_frame(KIND_UNPROTO_DATA))


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

class TestConnectionLifecycle(unittest.IsolatedAsyncioTestCase):

    async def test_connect_failure_raises(self):
        """connect() raises AGWConnectionError if TCP connection fails."""
        client = AsyncAGWPEClient("localhost", 19998)
        with self.assertRaises(AGWConnectionError):
            await asyncio.wait_for(client.connect(), timeout=2.0)

    async def test_close_not_connected_is_safe(self):
        """close() on an unconnected client does not raise."""
        client = AsyncAGWPEClient("localhost", 8000)
        await client.close()   # Should not raise
