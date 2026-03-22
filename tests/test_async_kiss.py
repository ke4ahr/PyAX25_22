# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_async_kiss.py

Unit tests for AsyncKISSTCP (interfaces/kiss/async_tcp.py).

Uses asyncio.TestCase (Python 3.8+) and mock streams so no real TCP server
is needed.

Covers:
- KISS frame construction (_stuff / _destuff)
- Per-byte frame assembly (_process_byte)
- send() builds correct KISS frame and writes to the stream
- on_frame plain callback invoked for received frames
- on_frame coroutine callback invoked for received frames
- asyncio.Queue delivery
- close() cancels the reader task
- KISSTCPError raised when not connected
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

from pyax25_22.interfaces.kiss.async_tcp import AsyncKISSTCP
from pyax25_22.interfaces.kiss.constants import FEND, FESC, TFEND, TFESC, CMD_DATA
from pyax25_22.interfaces.kiss.exceptions import KISSTCPError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_connected_tnc(host="localhost", port=8001, on_frame=None, queue=None):
    """Return an AsyncKISSTCP with mocked reader/writer (no real TCP)."""
    tnc = AsyncKISSTCP(host, port, on_frame=on_frame, queue=queue)
    tnc._reader = MagicMock()
    tnc._writer = MagicMock()
    tnc._writer.drain = AsyncMock()
    tnc._running = True
    return tnc


def _kiss_frame(payload: bytes, cmd: int = CMD_DATA) -> bytes:
    """Build a reference KISS frame for comparison."""
    frame = bytearray([FEND, cmd])
    for b in payload:
        if b == FEND:
            frame.extend([FESC, TFEND])
        elif b == FESC:
            frame.extend([FESC, TFESC])
        else:
            frame.append(b)
    frame.append(FEND)
    return bytes(frame)


# ---------------------------------------------------------------------------
# Frame construction tests (sync, no async needed)
# ---------------------------------------------------------------------------

class TestStuff(unittest.TestCase):

    def test_basic_payload(self):
        """_stuff wraps payload with FEND delimiters."""
        payload = b"\x01\x02\x03"
        frame = AsyncKISSTCP._stuff(payload, CMD_DATA)
        assert frame[0] == FEND
        assert frame[-1] == FEND
        assert frame[1] == CMD_DATA
        assert frame[2:-1] == payload

    def test_fend_escaped(self):
        """FEND bytes in payload become FESC TFEND."""
        payload = bytes([FEND])
        frame = AsyncKISSTCP._stuff(payload, CMD_DATA)
        body = frame[2:-1]
        assert body == bytes([FESC, TFEND])

    def test_fesc_escaped(self):
        """FESC bytes in payload become FESC TFESC."""
        payload = bytes([FESC])
        frame = AsyncKISSTCP._stuff(payload, CMD_DATA)
        body = frame[2:-1]
        assert body == bytes([FESC, TFESC])

    def test_empty_payload(self):
        """Empty payload produces a 3-byte frame: FEND cmd FEND."""
        frame = AsyncKISSTCP._stuff(b"", CMD_DATA)
        assert frame == bytes([FEND, CMD_DATA, FEND])


class TestDestuff(unittest.TestCase):

    def test_plain_bytes(self):
        """Bytes without escapes pass through unchanged."""
        data = bytearray(b"\x01\x02\x03")
        assert AsyncKISSTCP._destuff(data) == b"\x01\x02\x03"

    def test_fesc_tfend(self):
        """FESC TFEND decodes to FEND."""
        data = bytearray([FESC, TFEND])
        assert AsyncKISSTCP._destuff(data) == bytes([FEND])

    def test_fesc_tfesc(self):
        """FESC TFESC decodes to FESC."""
        data = bytearray([FESC, TFESC])
        assert AsyncKISSTCP._destuff(data) == bytes([FESC])

    def test_roundtrip(self):
        """stuff then destuff recovers original payload."""
        payload = bytes([0, FEND, FESC, 255, 0x55])
        frame = AsyncKISSTCP._stuff(payload, CMD_DATA)
        body = bytearray(frame[2:-1])
        assert AsyncKISSTCP._destuff(body) == payload


class TestProcessByte(unittest.TestCase):

    def _feed(self, tnc, data):
        """Feed bytes and return list of (cmd, payload) results."""
        results = []
        for b in data:
            r = tnc._process_byte(b)
            if r is not None:
                results.append(r)
        return results

    def setUp(self):
        self.tnc = _make_connected_tnc()

    def test_complete_frame(self):
        """A complete KISS frame produces one result."""
        payload = b"Hello"
        frame = _kiss_frame(payload)
        results = self._feed(self.tnc, frame)
        assert len(results) == 1
        cmd, got = results[0]
        assert cmd == CMD_DATA
        assert got == payload

    def test_partial_frame_no_result(self):
        """Feeding partial frame bytes yields no result yet."""
        frame = _kiss_frame(b"Test")
        partial = frame[:-1]
        results = self._feed(self.tnc, partial)
        assert results == []

    def test_two_frames(self):
        """Two consecutive frames each produce one result."""
        f1 = _kiss_frame(b"Frame1")
        f2 = _kiss_frame(b"Frame2")
        results = self._feed(self.tnc, f1 + f2)
        assert len(results) == 2
        assert results[0][1] == b"Frame1"
        assert results[1][1] == b"Frame2"

    def test_leading_fend_ignored(self):
        """Leading FEND bytes (common preamble) are silently ignored."""
        payload = b"data"
        frame = bytes([FEND, FEND]) + _kiss_frame(payload)
        results = self._feed(self.tnc, frame)
        assert len(results) == 1
        assert results[0][1] == payload


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------

class TestAsyncKISSTCPSend(unittest.IsolatedAsyncioTestCase):

    async def test_send_writes_kiss_frame(self):
        """send() writes a correctly structured KISS frame to the writer."""
        tnc = _make_connected_tnc()
        payload = b"AX25DATA"
        await tnc.send(payload, cmd=CMD_DATA)
        tnc._writer.write.assert_called_once()
        written = tnc._writer.write.call_args[0][0]
        assert written == _kiss_frame(payload, CMD_DATA)

    async def test_send_calls_drain(self):
        """send() calls writer.drain() after writing."""
        tnc = _make_connected_tnc()
        await tnc.send(b"test")
        tnc._writer.drain.assert_called_once()

    async def test_send_raises_when_not_connected(self):
        """send() raises KISSTCPError if writer is None."""
        tnc = AsyncKISSTCP("localhost", 8001)
        with self.assertRaises(KISSTCPError):
            await tnc.send(b"data")

    async def test_send_raises_on_oserror(self):
        """send() raises KISSTCPError if writer.write raises OSError."""
        tnc = _make_connected_tnc()
        tnc._writer.write.side_effect = OSError("broken pipe")
        with self.assertRaises(KISSTCPError):
            await tnc.send(b"data")


class TestAsyncKISSTCPReceive(unittest.IsolatedAsyncioTestCase):

    async def test_plain_callback_invoked(self):
        """Plain function on_frame is called for received frames."""
        received = []

        def on_frame(cmd, payload):
            received.append((cmd, payload))

        tnc = _make_connected_tnc(on_frame=on_frame)
        payload = b"test"
        frame = _kiss_frame(payload)
        for b in frame:
            r = tnc._process_byte(b)
            if r:
                await tnc._dispatch(*r)

        assert len(received) == 1
        assert received[0] == (CMD_DATA, payload)

    async def test_coroutine_callback_invoked(self):
        """Coroutine on_frame is awaited for received frames."""
        received = []

        async def on_frame(cmd, payload):
            received.append((cmd, payload))

        tnc = _make_connected_tnc(on_frame=on_frame)
        payload = b"async test"
        frame = _kiss_frame(payload)
        for b in frame:
            r = tnc._process_byte(b)
            if r:
                await tnc._dispatch(*r)

        assert len(received) == 1
        assert received[0][1] == payload

    async def test_queue_delivery(self):
        """Frames are put into the asyncio.Queue when queue= is set."""
        q = asyncio.Queue()
        tnc = _make_connected_tnc(queue=q)
        payload = b"queued frame"
        frame = _kiss_frame(payload)
        for b in frame:
            r = tnc._process_byte(b)
            if r:
                await tnc._dispatch(*r)

        assert not q.empty()
        cmd, got = await q.get()
        assert got == payload

    async def test_both_callback_and_queue(self):
        """Frames go to both on_frame callback and queue."""
        q = asyncio.Queue()
        received = []

        def on_frame(cmd, payload):
            received.append((cmd, payload))

        tnc = _make_connected_tnc(on_frame=on_frame, queue=q)
        frame = _kiss_frame(b"both")
        for b in frame:
            r = tnc._process_byte(b)
            if r:
                await tnc._dispatch(*r)

        assert len(received) == 1
        assert not q.empty()


class TestAsyncKISSTCPConnect(unittest.IsolatedAsyncioTestCase):

    async def test_connect_failure_raises(self):
        """connect() raises KISSTCPError if the TCP connection fails."""
        tnc = AsyncKISSTCP("localhost", 19999)
        with self.assertRaises(KISSTCPError):
            await asyncio.wait_for(tnc.connect(), timeout=2.0)

    async def test_close_not_connected_is_safe(self):
        """close() on an unconnected tnc does not raise."""
        tnc = AsyncKISSTCP("localhost", 8001)
        await tnc.close()   # Should not raise
