# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/async_tcp.py

AsyncKISSTCP -- asyncio-native KISS over TCP.

Drop-in async replacement for KISSTCP.  Uses ``asyncio.open_connection()``
for non-blocking I/O instead of threads.  The on_frame callback may be
either a plain function or a coroutine function -- both are supported.

Usage::

    import asyncio
    from pyax25_22.interfaces.kiss.async_tcp import AsyncKISSTCP

    async def main():
        async def on_frame(cmd, payload):
            print(f"Frame: cmd=0x{cmd:02X} len={len(payload)}")

        tnc = AsyncKISSTCP("localhost", 8001, on_frame=on_frame)
        await tnc.connect()
        await tnc.send(ax25_bytes)
        await asyncio.sleep(10)
        await tnc.close()

    asyncio.run(main())

Frame delivery:
    Received frames are passed to on_frame(cmd: int, payload: bytes).
    If on_frame is a coroutine function it is scheduled with
    ``asyncio.ensure_future()`` so it runs concurrently with the reader.
    Plain functions are called directly (they must not block).

    Alternatively, set queue= to receive frames via an asyncio.Queue:
        q = asyncio.Queue()
        tnc = AsyncKISSTCP("localhost", 8001, queue=q)
        await tnc.connect()
        cmd, payload = await q.get()
"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Tuple, Union

from .constants import FEND, FESC, TFEND, TFESC, CMD_DATA, CMD_EXIT
from .exceptions import KISSTCPError, KISSFrameError

logger = logging.getLogger(__name__)

_READ_BUFFER_SIZE = 4096

# Type alias for on_frame -- plain function or coroutine
FrameCallback = Union[
    Callable[[int, bytes], None],
    Callable[[int, bytes], Awaitable[None]],
]


class AsyncKISSTCP:
    """asyncio-native KISS over TCP.

    Connects to a KISS TCP server and provides non-blocking frame send/receive
    using asyncio streams.

    Args:
        host: KISS server hostname or IP.
        port: KISS server TCP port (Dire Wolf default: 8001).
        on_frame: Optional callback ``(cmd, payload) -> None`` or async
            coroutine function.  Called for each received KISS data frame.
        queue: Optional ``asyncio.Queue`` to receive ``(cmd, payload)`` tuples.
            Can be used alongside or instead of on_frame.
        connect_timeout: TCP connect timeout in seconds (default: 10.0).
        keepalive: Enable TCP keepalive (default: True).

    Raises:
        KISSTCPError: If the TCP connection cannot be established.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_frame: Optional[FrameCallback] = None,
        queue: Optional[asyncio.Queue] = None,
        connect_timeout: float = 10.0,
        keepalive: bool = True,
    ) -> None:
        self._host = host
        self._port = port
        self._on_frame = on_frame
        self._queue = queue
        self._connect_timeout = connect_timeout
        self._keepalive = keepalive

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._running = False

        # KISS frame assembler state (same logic as KISSBase._process_byte)
        self._buf: bytearray = bytearray()
        self._write_lock = asyncio.Lock()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the TCP connection and start the background reader coroutine.

        Raises:
            KISSTCPError: If the connection fails.
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise KISSTCPError(
                f"AsyncKISSTCP connect failed ({self._host}:{self._port}): {exc}"
            ) from exc

        if self._keepalive:
            sock = self._writer.get_extra_info("socket")
            if sock is not None:
                import socket as _socket
                sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_KEEPALIVE, 1)
                if hasattr(_socket, "TCP_KEEPIDLE"):
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPIDLE, 60)
                if hasattr(_socket, "TCP_KEEPINTVL"):
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPINTVL, 10)
                if hasattr(_socket, "TCP_KEEPCNT"):
                    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_KEEPCNT, 5)

        self._running = True
        self._reader_task = asyncio.ensure_future(self._receive_loop())
        logger.info("AsyncKISSTCP: connected to %s:%d", self._host, self._port)

    async def close(self) -> None:
        """Close the TCP connection and stop the reader coroutine."""
        self._running = False
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

        logger.info("AsyncKISSTCP: closed connection to %s:%d", self._host, self._port)

    # -----------------------------------------------------------------------
    # Send
    # -----------------------------------------------------------------------

    async def send(self, payload: bytes, cmd: int = CMD_DATA) -> None:
        """Encode payload as a KISS frame and send it.

        Args:
            payload: Raw AX.25 frame bytes.
            cmd: KISS command byte (default 0x00 = data frame).

        Raises:
            KISSTCPError: If not connected or the write fails.
        """
        if self._writer is None:
            raise KISSTCPError("AsyncKISSTCP: not connected")

        frame = self._stuff(payload, cmd)
        async with self._write_lock:
            try:
                self._writer.write(frame)
                await self._writer.drain()
                logger.debug(
                    "AsyncKISSTCP send: cmd=0x%02X payload=%d frame=%d to %s:%d",
                    cmd, len(payload), len(frame), self._host, self._port,
                )
            except OSError as exc:
                raise KISSTCPError(f"AsyncKISSTCP write failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # KISS frame construction (mirrors KISSBase._stuff)
    # -----------------------------------------------------------------------

    @staticmethod
    def _stuff(payload: bytes, cmd: int) -> bytes:
        """Build a complete KISS frame (FEND + cmd + escaped payload + FEND)."""
        frame = bytearray([FEND, cmd])
        for byte in payload:
            if byte == FEND:
                frame.extend([FESC, TFEND])
            elif byte == FESC:
                frame.extend([FESC, TFESC])
            else:
                frame.append(byte)
        frame.append(FEND)
        return bytes(frame)

    # -----------------------------------------------------------------------
    # KISS frame disassembly (mirrors KISSBase._destuff / _process_byte)
    # -----------------------------------------------------------------------

    def _process_byte(self, byte: int) -> Optional[Tuple[int, bytes]]:
        """Feed one byte into the KISS assembler.

        Returns:
            ``(cmd, payload)`` when a complete frame is assembled, else None.
        """
        if byte == FEND:
            if len(self._buf) >= 1:
                cmd = self._buf[0]
                try:
                    payload = self._destuff(self._buf[1:])
                except KISSFrameError as exc:
                    logger.warning("AsyncKISSTCP destuff error: %s", exc)
                    self._buf = bytearray()
                    return None
                self._buf = bytearray()
                return cmd, payload
            self._buf = bytearray()
        else:
            self._buf.append(byte)
        return None

    @staticmethod
    def _destuff(data: bytearray) -> bytes:
        """Remove KISS escape sequences from a frame body."""
        out = bytearray()
        i = 0
        while i < len(data):
            b = data[i]
            if b == FESC:
                i += 1
                if i >= len(data):
                    raise KISSFrameError("Truncated FESC at end of frame")
                nxt = data[i]
                if nxt == TFEND:
                    out.append(FEND)
                elif nxt == TFESC:
                    out.append(FESC)
                else:
                    out.append(nxt)
            else:
                out.append(b)
            i += 1
        return bytes(out)

    # -----------------------------------------------------------------------
    # Reader coroutine
    # -----------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Background coroutine: read bytes and dispatch complete KISS frames."""
        logger.debug("AsyncKISSTCP reader started (%s:%d)", self._host, self._port)
        try:
            while self._running and self._reader is not None:
                try:
                    data = await self._reader.read(_READ_BUFFER_SIZE)
                except asyncio.CancelledError:
                    break
                except OSError as exc:
                    if self._running:
                        logger.error(
                            "AsyncKISSTCP recv error (%s:%d): %s",
                            self._host, self._port, exc,
                        )
                    break

                if not data:
                    logger.warning(
                        "AsyncKISSTCP: server %s:%d closed the connection",
                        self._host, self._port,
                    )
                    break

                for byte in data:
                    result = self._process_byte(byte)
                    if result is not None:
                        cmd, payload = result
                        await self._dispatch(cmd, payload)

        finally:
            self._running = False
            logger.debug(
                "AsyncKISSTCP reader stopped (%s:%d)", self._host, self._port
            )

    async def _dispatch(self, cmd: int, payload: bytes) -> None:
        """Deliver a decoded frame to on_frame and/or the queue.

        Args:
            cmd: KISS command byte.
            payload: Destuffed frame payload.
        """
        if self._queue is not None:
            await self._queue.put((cmd, payload))

        if self._on_frame is not None:
            try:
                result = self._on_frame(cmd, payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "AsyncKISSTCP: on_frame raised an exception "
                    "(cmd=0x%02X len=%d)", cmd, len(payload),
                )
