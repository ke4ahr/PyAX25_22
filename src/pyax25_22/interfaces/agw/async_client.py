# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/agw/async_client.py

AsyncAGWPEClient -- asyncio-native AGWPE TCP client.

Drop-in async replacement for AGWPEClient.  Uses ``asyncio.open_connection()``
instead of blocking sockets and threads.  Callbacks may be plain functions or
coroutine functions.

Usage::

    import asyncio
    from pyax25_22.interfaces.agw.async_client import AsyncAGWPEClient

    async def main():
        client = AsyncAGWPEClient("localhost", 8000, callsign="KE4AHR")

        async def on_frame(frame):
            print(f"Got: {frame}")

        client.on_frame = on_frame
        await client.connect()
        await client.send_ui(port=0, dest="APRS", src="KE4AHR",
                             pid=0xF0, info=b"Hello")
        await asyncio.sleep(5)
        await client.close()

    asyncio.run(main())

Reconnection:
    Call await client.connect() again after a connection is lost.  A future
    version will add auto-reconnect with exponential backoff.
"""

import asyncio
import logging
import struct
from typing import Awaitable, Callable, Dict, List, Optional, Union

from .constants import (
    AGWPE_DEFAULT_PORT,
    AGWPE_HEADER_SIZE,
    CALLSIGN_WIDTH,
    KIND_REGISTER,
    KIND_UNREGISTER,
    KIND_VERSION,
    KIND_PORT_INFO,
    KIND_PORT_CAPS,
    KIND_EXTENDED_VER,
    KIND_MEMORY_USAGE,
    KIND_ENABLE_MON,
    KIND_RAW_MON,
    KIND_RAW_SEND,
    KIND_UNPROTO,
    KIND_UNPROTO_VIA,
    KIND_UNPROTO_DATA,
    KIND_CONNECT,
    KIND_CONNECT_INC,
    KIND_DISC,
    KIND_CONN_DATA,
    KIND_OUTSTANDING,
    KIND_OUTSTANDING_R,
    KIND_HEARD,
    KIND_LOGIN,
    KIND_PARAMETER,
)
from .exceptions import AGWConnectionError, AGWFrameError
from .client import AGWPEFrame

logger = logging.getLogger(__name__)

_MAX_DATA_LEN = 65536

# Type alias for callbacks -- plain or coroutine
_Callback = Union[Callable[..., None], Callable[..., Awaitable[None]]]


async def _call(fn: Optional[_Callback], *args) -> None:
    """Call fn(*args) whether it is a plain function or a coroutine."""
    if fn is None:
        return
    result = fn(*args)
    if asyncio.iscoroutine(result):
        await result


class AsyncAGWPEClient:
    """asyncio-native AGWPE TCP client.

    Connects to an AGWPE-compatible server using asyncio streams.  All I/O is
    non-blocking.  Callbacks can be plain functions or coroutine functions.

    Args:
        host: AGWPE server hostname or IP (default: ``"127.0.0.1"``).
        port: AGWPE server TCP port (default: 8000).
        callsign: Station callsign to register on connect.

    Attributes:
        on_frame: Callback for unproto / monitored frames (kind D/K/U/V).
        on_connected_data: Callback for connected-mode data (kind d + data).
        on_outstanding: Callback for outstanding frame count (kind Y/y).
        on_heard_stations: Callback for heard-stations list (kind H).
        on_extended_version: Callback for extended version string (kind v).
        on_memory_usage: Callback for memory usage dict (kind m).
        on_connect: Callback for incoming connection notifications (kind c).
        on_disconnect: Callback for disconnect notifications (kind d, no data).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = AGWPE_DEFAULT_PORT,
        callsign: str = "NOCALL",
    ) -> None:
        self.host = host
        self.port = port
        self.callsign = callsign.upper().strip()

        # Public callbacks (may be plain functions or coroutines)
        self.on_frame: Optional[_Callback] = None
        self.on_connected_data: Optional[_Callback] = None
        self.on_outstanding: Optional[_Callback] = None
        self.on_heard_stations: Optional[_Callback] = None
        self.on_extended_version: Optional[_Callback] = None
        self.on_memory_usage: Optional[_Callback] = None
        self.on_connect: Optional[_Callback] = None
        self.on_disconnect: Optional[_Callback] = None

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._write_lock = asyncio.Lock()
        self.connected = False

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    async def connect(self, connect_timeout: float = 10.0) -> bool:
        """Connect to the AGWPE server and register the callsign.

        Args:
            connect_timeout: TCP connect timeout in seconds.

        Returns:
            True on success.

        Raises:
            AGWConnectionError: If the connection fails.
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise AGWConnectionError(
                f"AsyncAGWPEClient connect failed ({self.host}:{self.port}): {exc}"
            ) from exc

        self.connected = True

        # Register callsign using 'X' frame
        await self._send_frame(data_kind=KIND_REGISTER, call_from=self.callsign)

        self._reader_task = asyncio.ensure_future(self._receive_loop())
        logger.info(
            "AsyncAGWPEClient: connected to %s:%d as %s",
            self.host, self.port, self.callsign,
        )
        return True

    async def close(self) -> None:
        """Disconnect from the AGWPE server."""
        self.connected = False
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except OSError:
                pass
            self._writer = None
            self._reader = None

        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        logger.info(
            "AsyncAGWPEClient: disconnected from %s:%d", self.host, self.port
        )

    # -----------------------------------------------------------------------
    # Low-level frame transmitter
    # -----------------------------------------------------------------------

    async def _send_frame(
        self,
        data_kind: bytes,
        port: int = 0,
        call_from: str = "",
        call_to: str = "",
        data: bytes = b"",
    ) -> None:
        """Build and send a 36-byte AGWPE header + data.

        Raises:
            AGWConnectionError: If not connected or the write fails.
        """
        if not self.connected or self._writer is None:
            raise AGWConnectionError("AsyncAGWPEClient: not connected")

        cf = call_from.upper().ljust(CALLSIGN_WIDTH)[:CALLSIGN_WIDTH].encode(
            "ascii", errors="replace"
        )
        ct = call_to.upper().ljust(CALLSIGN_WIDTH)[:CALLSIGN_WIDTH].encode(
            "ascii", errors="replace"
        )

        header = bytearray(AGWPE_HEADER_SIZE)
        header[0:1] = data_kind
        struct.pack_into("<I", header, 4, port)
        header[8:18] = cf
        header[18:28] = ct
        struct.pack_into("<I", header, 28, len(data))

        packet = bytes(header) + data

        async with self._write_lock:
            try:
                self._writer.write(packet)
                await self._writer.drain()
                logger.debug(
                    "AsyncAGWPE send: kind=%s port=%d from=%s to=%s data_len=%d",
                    data_kind.decode("ascii", errors="?"),
                    port, call_from, call_to, len(data),
                )
            except OSError as exc:
                self.connected = False
                raise AGWConnectionError(
                    f"AsyncAGWPE write failed: {exc}"
                ) from exc

    # -----------------------------------------------------------------------
    # High-level transmit methods
    # -----------------------------------------------------------------------

    async def send_ui(
        self,
        port: int,
        dest: str,
        src: str,
        pid: int,
        info: bytes = b"",
    ) -> None:
        """Send an unproto UI frame.

        Args:
            port: TNC port number.
            dest: Destination callsign.
            src: Source callsign.
            pid: Protocol identifier byte (e.g., 0xF0 for no layer 3).
            info: Information field bytes.
        """
        await self._send_frame(
            data_kind=KIND_UNPROTO_DATA,
            port=port,
            call_from=src,
            call_to=dest,
            data=bytes([pid]) + info,
        )

    async def send_raw(
        self, port: int, dest: str, src: str, data: bytes
    ) -> None:
        """Send a raw AX.25 frame using the 'K' frame type."""
        await self._send_frame(
            data_kind=KIND_RAW_SEND,
            port=port,
            call_from=src,
            call_to=dest,
            data=data,
        )

    async def send_connect(self, port: int, dest: str) -> None:
        """Initiate a connected-mode AX.25 connection."""
        await self._send_frame(
            data_kind=KIND_CONNECT,
            port=port,
            call_from=self.callsign,
            call_to=dest,
        )

    async def send_disconnect(self, port: int, dest: str) -> None:
        """Disconnect from a connected-mode session."""
        await self._send_frame(
            data_kind=KIND_DISC,
            port=port,
            call_from=self.callsign,
            call_to=dest,
        )

    async def send_connected_data(
        self, port: int, dest: str, data: bytes
    ) -> None:
        """Send data on an established connected-mode circuit."""
        await self._send_frame(
            data_kind=KIND_CONN_DATA,
            port=port,
            call_from=self.callsign,
            call_to=dest,
            data=data,
        )

    async def enable_monitor(self, port: int = 0) -> None:
        """Enable frame monitoring (receive all frames as 'K' callbacks)."""
        await self._send_frame(data_kind=KIND_ENABLE_MON, port=port)

    async def request_outstanding(self, port: int = 0) -> None:
        """Query the number of outstanding (unacknowledged) frames."""
        await self._send_frame(data_kind=KIND_OUTSTANDING, port=port)

    async def request_heard_stations(self, port: int = 0) -> None:
        """Request the list of recently heard stations."""
        await self._send_frame(data_kind=KIND_HEARD, port=port)

    async def send_login(self, username: str, password: str) -> None:
        """Send AGWPE login credentials ('T' frame)."""
        payload = f"{username}\x00{password}\x00".encode("ascii", errors="replace")
        await self._send_frame(data_kind=KIND_LOGIN, data=payload)

    async def set_parameter(self, port: int, param_id: int, value: int) -> None:
        """Set a TNC parameter via AGWPE."""
        payload = struct.pack("<BI", param_id, value)
        await self._send_frame(data_kind=KIND_PARAMETER, port=port, data=payload)

    async def request_version(self) -> None:
        """Request the AGWPE version string ('R' frame)."""
        await self._send_frame(data_kind=KIND_VERSION)

    async def request_port_info(self) -> None:
        """Request information about available TNC ports ('G' frame)."""
        await self._send_frame(data_kind=KIND_PORT_INFO)

    # -----------------------------------------------------------------------
    # Receiver
    # -----------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Background coroutine: read and parse AGWPE frames."""
        logger.debug("AsyncAGWPE reader started (%s:%d)", self.host, self.port)
        buf = b""

        try:
            while self.connected and self._reader is not None:
                try:
                    chunk = await self._reader.read(4096)
                except asyncio.CancelledError:
                    break
                except OSError as exc:
                    if self.connected:
                        logger.error(
                            "AsyncAGWPE recv error (%s:%d): %s",
                            self.host, self.port, exc,
                        )
                    break

                if not chunk:
                    logger.warning(
                        "AsyncAGWPE: server %s:%d closed connection",
                        self.host, self.port,
                    )
                    break

                buf += chunk

                while len(buf) >= AGWPE_HEADER_SIZE:
                    data_kind = buf[0:1]
                    port = struct.unpack("<I", buf[4:8])[0]
                    call_from = buf[8:18].decode("ascii", errors="ignore").strip()
                    call_to = buf[18:28].decode("ascii", errors="ignore").strip()
                    data_len = struct.unpack("<I", buf[28:32])[0]

                    if data_len > _MAX_DATA_LEN:
                        logger.error(
                            "AsyncAGWPE: data_len=%d exceeds limit -- dropping",
                            data_len,
                        )
                        self.connected = False
                        return

                    if len(buf) < AGWPE_HEADER_SIZE + data_len:
                        break

                    payload = buf[AGWPE_HEADER_SIZE: AGWPE_HEADER_SIZE + data_len]
                    buf = buf[AGWPE_HEADER_SIZE + data_len:]

                    frame = AGWPEFrame()
                    frame.data_kind = data_kind
                    frame.port = port
                    frame.call_from = call_from
                    frame.call_to = call_to
                    frame.data_len = data_len
                    frame.data = payload

                    logger.debug("AsyncAGWPE rx: %r", frame)
                    await self._dispatch(frame)

        finally:
            self.connected = False
            logger.debug("AsyncAGWPE reader stopped (%s:%d)", self.host, self.port)

    async def _dispatch(self, frame: AGWPEFrame) -> None:
        """Route a received AGWPE frame to the appropriate callback.

        Args:
            frame: A fully decoded AGWPEFrame.
        """
        dk = frame.data_kind

        try:
            if dk in (KIND_UNPROTO_DATA, KIND_RAW_MON, KIND_UNPROTO,
                       KIND_UNPROTO_VIA):
                await _call(self.on_frame, frame)

            elif dk == KIND_CONNECT_INC:
                await _call(self.on_connect, frame.port, frame.call_from)

            elif dk == KIND_DISC:
                if frame.data:
                    await _call(
                        self.on_connected_data,
                        frame.port, frame.call_from, frame.data,
                    )
                    if self.on_connected_data is None:
                        await _call(self.on_frame, frame)
                else:
                    await _call(self.on_disconnect, frame.port, frame.call_from)

            elif dk in (KIND_OUTSTANDING, KIND_OUTSTANDING_R):
                if frame.data and len(frame.data) >= 4:
                    count = struct.unpack("<I", frame.data[:4])[0]
                    await _call(self.on_outstanding, frame.port, count)

            elif dk == KIND_HEARD:
                heard_list = self._parse_heard(frame.data)
                await _call(self.on_heard_stations, frame.port, heard_list)

            elif dk == KIND_EXTENDED_VER:
                version_str = frame.data.decode("ascii", errors="ignore").strip()
                await _call(self.on_extended_version, version_str)

            elif dk == KIND_MEMORY_USAGE:
                mem = self._parse_memory(frame.data)
                await _call(self.on_memory_usage, mem)

            elif dk == KIND_PORT_CAPS:
                logger.debug("AsyncAGWPE: port caps (port=%d)", frame.port)

            else:
                logger.debug(
                    "AsyncAGWPE: unhandled frame kind %r (port=%d)",
                    dk, frame.port,
                )

        except Exception:
            logger.exception(
                "AsyncAGWPE: error in dispatch for kind %r", dk
            )

    # -----------------------------------------------------------------------
    # Parsers (shared with sync client)
    # -----------------------------------------------------------------------

    def _parse_heard(self, data: bytes) -> List[Dict]:
        """Parse an 'H' heard-stations payload."""
        heard = []
        entry_size = 14
        for i in range(len(data) // entry_size):
            offset = i * entry_size
            call = data[offset: offset + 10].decode("ascii", errors="ignore").strip()
            ts = struct.unpack("<I", data[offset + 10: offset + 14])[0]
            if call:
                heard.append({"callsign": call, "last_heard": ts})
        return heard

    def _parse_memory(self, data: bytes) -> Dict[str, int]:
        """Parse an 'm' memory-usage payload."""
        if len(data) >= 8:
            free_mem = struct.unpack("<I", data[0:4])[0]
            used_mem = struct.unpack("<I", data[4:8])[0]
            return {"free_kb": free_mem // 1024, "used_kb": used_mem // 1024}
        return {"free_kb": 0, "used_kb": 0}
