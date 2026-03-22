# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/tcp.py

KISSTCP -- KISS over a TCP socket (e.g., Dire Wolf TCP KISS server).

Extends KISSBase with:
- TCP socket management (IPv4/IPv6)
- Optional TCP keepalive (SO_KEEPALIVE)
- Blocking write with lock
- Background reader thread (daemon)

KISS-over-TCP is used with software TNCs such as Dire Wolf, soundmodem, or
pat (winlink) that expose a KISS interface on a TCP port (default: 8001).
"""

import logging
import socket
import threading
import time
from typing import Callable, Optional

from .base import KISSBase
from .exceptions import KISSTCPError

logger = logging.getLogger(__name__)

_DEFAULT_TCP_TIMEOUT = 10.0    # connect timeout in seconds
_READ_BUFFER_SIZE = 4096       # bytes per recv() call


class KISSTCP(KISSBase):
    """KISS protocol over a TCP socket.

    Opens a TCP connection to a KISS server, starts a background reader
    thread, and implements write() by calling socket.sendall().

    Args:
        host: Hostname or IP address of the KISS TCP server.
        port: TCP port number (common default: 8001 for Dire Wolf).
        on_frame: Optional callback ``(cmd: int, payload: bytes) -> None``
            called for each received KISS frame.
        connect_timeout: How long to wait for the TCP connection to be
            established, in seconds (default: 10.0).
        keepalive: If True, enables TCP keepalive on the socket so the OS
            detects dead connections (default: True).

    Raises:
        KISSTCPError: If the TCP connection cannot be established.

    Example:
        def on_frame(cmd, data):
            print(f"Got KISS frame: cmd=0x{cmd:02X}")

        tnc = KISSTCP("localhost", 8001, on_frame=on_frame)
        tnc.send(ax25_bytes)
        tnc.close()
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_frame: Optional[Callable[[int, bytes], None]] = None,
        connect_timeout: float = _DEFAULT_TCP_TIMEOUT,
        keepalive: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(on_frame=on_frame, **kwargs)

        self._host = host
        self._port = port
        self._write_lock = threading.Lock()
        self._running = True
        self._sock: Optional[socket.socket] = None

        self._connect(connect_timeout, keepalive)

        self._thread = threading.Thread(
            target=self._receive_loop,
            name=f"KISSTCP-{host}:{port}",
            daemon=True,
        )
        self._thread.start()

    # -----------------------------------------------------------------------
    # Connection setup
    # -----------------------------------------------------------------------

    def _connect(self, timeout: float, keepalive: bool) -> None:
        """Open the TCP connection.

        Args:
            timeout: Connect timeout in seconds.
            keepalive: If True, enable SO_KEEPALIVE and platform-specific
                keepalive parameters.

        Raises:
            KISSTCPError: If the connect fails.
        """
        try:
            sock = socket.create_connection(
                (self._host, self._port), timeout=timeout
            )
        except OSError as exc:
            logger.critical(
                "KISSTCP: cannot connect to %s:%d -- %s",
                self._host, self._port, exc,
            )
            raise KISSTCPError(
                f"TCP connect failed ({self._host}:{self._port}): {exc}"
            ) from exc

        if keepalive:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            if hasattr(socket, "TCP_KEEPINTVL"):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            if hasattr(socket, "TCP_KEEPCNT"):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

        # Switch to blocking with a per-recv timeout for the reader thread
        sock.settimeout(2.0)
        self._sock = sock
        logger.info("KISSTCP: connected to %s:%d", self._host, self._port)

    # -----------------------------------------------------------------------
    # KISSBase abstract method implementation
    # -----------------------------------------------------------------------

    def write(self, data: bytes) -> None:
        """Send raw KISS frame bytes over the TCP socket.

        Uses socket.sendall() to guarantee all bytes are sent.  A write lock
        prevents interleaving from concurrent callers.

        Args:
            data: Encoded KISS frame bytes (including FEND delimiters).

        Raises:
            KISSTCPError: If the socket is not connected or the send fails.
        """
        if self._sock is None:
            raise KISSTCPError("KISSTCP: not connected")
        with self._write_lock:
            try:
                self._sock.sendall(data)
                logger.debug(
                    "KISSTCP write: %d bytes to %s:%d",
                    len(data), self._host, self._port,
                )
            except OSError as exc:
                logger.error(
                    "KISSTCP write error to %s:%d: %s",
                    self._host, self._port, exc,
                )
                raise KISSTCPError(f"TCP write failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # Reader thread
    # -----------------------------------------------------------------------

    def _receive_loop(self) -> None:
        """Background thread: receive bytes from TCP and process KISS frames.

        Reads chunks from the socket and passes each byte to _process_byte()
        (inherited from KISSBase).  On connection close or error, the loop
        exits and _running is cleared.
        """
        logger.debug(
            "KISSTCP reader thread started (%s:%d)",
            self._host, self._port,
        )
        while self._running and self._sock is not None:
            try:
                data = self._sock.recv(_READ_BUFFER_SIZE)
                if not data:
                    logger.warning(
                        "KISSTCP: server %s:%d closed the connection",
                        self._host, self._port,
                    )
                    self._running = False
                    break
                for byte in data:
                    self._process_byte(byte)
            except socket.timeout:
                continue    # Normal: check _running and loop
            except OSError as exc:
                if self._running:
                    logger.error(
                        "KISSTCP receive error (%s:%d): %s",
                        self._host, self._port, exc,
                    )
                self._running = False
                break
            except Exception:
                logger.exception(
                    "KISSTCP: unexpected error in reader thread (%s:%d)",
                    self._host, self._port,
                )
                self._running = False
                break

        logger.debug(
            "KISSTCP reader thread stopped (%s:%d)", self._host, self._port
        )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Close the TCP socket and stop the reader thread.

        The reader thread will exit on its next recv() timeout.
        """
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
                logger.info(
                    "KISSTCP: closed connection to %s:%d",
                    self._host, self._port,
                )
            except OSError as exc:
                logger.warning("KISSTCP close error: %s", exc)
            finally:
                self._sock = None
        super().close()

    def __del__(self) -> None:
        """Destructor: attempt a graceful close on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
