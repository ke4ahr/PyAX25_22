# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.interfaces.agwpe -- AGWPE TCP/IP API client.

AGWPE (AGW Packet Engine) is a Windows program by George Rossopoulos
(SV2AGW) that talks to a TNC and provides a TCP socket API so other
programs can send and receive AX.25 frames over the network.

Think of AGWPE as a bridge:

  Your program  <--- TCP socket --->  AGWPE server  <--- serial --->  TNC  <--- radio

All communication over the TCP socket uses a 36-byte binary header
followed by optional data. The header says what kind of message it is
(data, connection, version query, etc.) and who it is from and to.

This file implements a synchronous AGWPE client with a background
reader thread. It also exposes the AGWPE DataKind constants so callers
can check what kind of frame they received.

Known bug (documented, not fixed here): some implementations send
``b'R'`` for callsign registration but the spec says it should be
``b'X'``. This implementation uses the correct ``X`` command.

Compliant with the AGWPE Socket Interface specification (SV2AGW, 2000).
"""

from __future__ import annotations

import queue
import socket
import struct
import threading
import logging
from typing import Callable, Dict, Optional, Tuple

from pyax25_22.core.exceptions import AGWPEError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AGWPE header format
# ---------------------------------------------------------------------------

#: Struct format string for the 36-byte AGWPE frame header.
#:   Port      -- 4 bytes (LOWORD = port index 0+, HIWORD reserved)
#:   DataKind  -- 4 bytes (LOWORD = ASCII command byte, HIWORD = extra)
#:   CallFrom  -- 10 bytes (NULL-terminated callsign, max 9 chars + NUL)
#:   CallTo    -- 10 bytes (NULL-terminated callsign, max 9 chars + NUL)
#:   DataLen   -- 4 bytes (number of data bytes that follow the header)
#:   USER      -- 4 bytes (reserved / undefined, always 0)
HEADER_FMT: str = "<II10s10sII"
HEADER_SIZE: int = struct.calcsize(HEADER_FMT)

# ---------------------------------------------------------------------------
# DataKind constants (AGWPE frame type codes, ASCII values)
# ---------------------------------------------------------------------------

#: DataKind 'D': Connected-mode data received from remote station.
DATAKIND_CONNECTED_DATA: str = "D"

#: DataKind 'U': Unproto (UI) frame received -- monitoring data.
DATAKIND_UNPROTO_MONITOR: str = "U"

#: DataKind 'T': Transmitted frame (TX monitor data).
DATAKIND_TX_MONITOR: str = "T"

#: DataKind 'S': Monitor header only (no data body).
DATAKIND_MONITOR_HEADER: str = "S"

#: DataKind 'I': Monitor header + full frame data.
DATAKIND_MONITOR_FULL: str = "I"

#: DataKind 'c': A new connected-mode session was established.
DATAKIND_NEW_CONNECTION: str = "c"

#: DataKind 'd': A connected-mode session was disconnected or timed out.
DATAKIND_DISCONNECT: str = "d"

#: DataKind 'H': Heard list response (one callsign per reply).
DATAKIND_HEARD_LIST: str = "H"

#: DataKind 'X': Callsign registration reply (success or failure).
DATAKIND_REGISTRATION: str = "X"

#: DataKind 'Y': Outstanding frames in queue for a connection.
DATAKIND_OUTSTANDING: str = "Y"

#: DataKind 'g': Radio port capabilities response.
DATAKIND_PORT_CAPABILITIES: str = "g"

#: DataKind 'R': AGWPE version information response.
DATAKIND_VERSION: str = "R"

#: DataKind 'k': Raw AX.25 frame monitor data.
DATAKIND_RAW_FRAMES: str = "k"

#: DataKind 'm': Monitoring control toggle.
DATAKIND_MONITORING: str = "m"


# ---------------------------------------------------------------------------
# AGWPE-specific exceptions
# ---------------------------------------------------------------------------

class AGWPEConnectionError(AGWPEError):
    """Failed to connect to or communicate with the AGWPE server.

    Raised when the TCP connection to the AGWPE server cannot be
    established or is lost, or when a send fails.
    """


class AGWPEFrameError(AGWPEError):
    """An AGWPE frame could not be parsed.

    Raised when the header is the wrong size, the data is truncated,
    or the connection closes while we are in the middle of reading
    a frame.
    """


# ---------------------------------------------------------------------------
# AGWPE client
# ---------------------------------------------------------------------------

class AGWPEInterface:
    """Synchronous AGWPE client with a background reader thread.

    Connects to an AGWPE server over TCP, sends commands (register
    callsign, enable monitoring, send unproto, etc.), and receives
    incoming frames in a background thread.

    Attributes:
        host: The AGWPE server hostname or IP address.
        port: The TCP port to connect to (default 8000).
        timeout: Socket timeout in seconds.

    Raises:
        AGWPEConnectionError: If the TCP connection fails.
        AGWPEFrameError: If a received frame header is malformed.

    Example::

        client = AGWPEInterface(host="192.168.1.10")
        client.connect()
        client.register_callsign("KE4AHR-1")
        client.enable_monitoring()
        while True:
            port, kind, frm, to, data = client.receive(timeout=10.0)
            print(f"[{kind}] {frm} -> {to}: {data[:32]!r}")
        client.disconnect()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout: float = 5.0,
    ) -> None:
        """Set up the AGWPE client. Does not connect yet.

        Args:
            host: The hostname or IP address of the AGWPE server.
                Default is ``"127.0.0.1"`` (localhost).
            port: The TCP port of the AGWPE server. Default is 8000.
            timeout: Socket read timeout in seconds. Lower values make
                the reader thread more responsive but use more CPU.
                Default is 5.0.
        """
        self.host = host
        self.port = port
        self.timeout = timeout

        self.sock: Optional[socket.socket] = None
        self._recv_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._callbacks: Dict[str, Callable] = {}

        logger.info(
            "AGWPEInterface initialized: host=%s port=%d timeout=%.1fs",
            host, port, timeout,
        )

    # -----------------------------------------------------------------------
    # Connect / disconnect
    # -----------------------------------------------------------------------

    def connect(self) -> None:
        """Open the TCP connection to the AGWPE server and start reading.

        Raises:
            AGWPEConnectionError: If the TCP connection cannot be made.

        Example::

            client.connect()
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            self._running = True
            self._thread = threading.Thread(
                target=self._reader_thread, daemon=True, name="AGWPEReader"
            )
            self._thread.start()
            logger.info("Connected to AGWPE server at %s:%d", self.host, self.port)
        except socket.error as exc:
            logger.error(
                "Failed to connect to AGWPE at %s:%d: %s",
                self.host, self.port, exc,
            )
            raise AGWPEConnectionError(
                f"Failed to connect to {self.host}:{self.port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Close the TCP connection and stop the reader thread.

        Safe to call even if already disconnected.

        Example::

            client.disconnect()
        """
        self._running = False

        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except socket.error:
                pass
            try:
                self.sock.close()
            except socket.error as exc:
                logger.warning("Error closing socket: %s", exc)
            finally:
                self.sock = None

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("AGWPE reader thread did not stop cleanly")
            self._thread = None

        logger.info("Disconnected from AGWPE server")

    # -----------------------------------------------------------------------
    # Command methods
    # -----------------------------------------------------------------------

    def register_callsign(self, callsign: str) -> None:
        """Register a callsign with the AGWPE server (command 'X').

        AGWPE will reply with an 'X' DataKind frame indicating success
        or failure. Listen for it with ``receive()`` or a callback.

        Args:
            callsign: The callsign to register (e.g. ``"KE4AHR-1"``).
                Maximum 9 characters.

        Raises:
            AGWPEConnectionError: If not connected or the send fails.

        Example::

            client.register_callsign("KE4AHR-1")
        """
        self._send_frame(0, "X", callsign, "")
        logger.info("Sent callsign registration for %s", callsign)

    def enable_monitoring(self) -> None:
        """Enable monitoring of all received frames ('m' command).

        After enabling, the server will send 'U', 'I', 'S', and 'T'
        frames for every packet it receives, not just those addressed
        to registered callsigns.

        Raises:
            AGWPEConnectionError: If not connected or the send fails.
        """
        self._send_frame(0, "m", "", "")
        logger.info("Monitoring enabled")

    def disable_monitoring(self) -> None:
        """Disable monitoring ('m' command toggled off).

        Raises:
            AGWPEConnectionError: If not connected or the send fails.
        """
        self._send_frame(0, "m", "", "")
        logger.info("Monitoring disabled")

    def query_outstanding_frames(
        self,
        port: int = 0,
        callsign: str = "",
    ) -> None:
        """Ask how many frames are queued for transmission ('y' command).

        The server replies with a 'Y' DataKind frame containing the count
        in the DataLen field (not as data bytes).

        Args:
            port: Radio port to query (0-based index).
            callsign: Optional callsign to query a specific connection.

        Raises:
            AGWPEConnectionError: If not connected or the send fails.
        """
        self._send_frame(port, "y", callsign, callsign)
        logger.debug("Queried outstanding frames for port=%d callsign=%s", port, callsign)

    def query_port_capabilities(self, port: int = 0) -> None:
        """Ask for the capabilities of a radio port ('g' command).

        The server replies with a 'g' DataKind frame.

        Args:
            port: The radio port to query (0-based index).

        Raises:
            AGWPEConnectionError: If not connected or the send fails.
        """
        self._send_frame(port, "g", "", "")
        logger.debug("Queried capabilities for port=%d", port)

    def query_version(self) -> None:
        """Ask for the AGWPE software version ('R' command).

        The server replies with an 'R' DataKind frame.

        Raises:
            AGWPEConnectionError: If not connected or the send fails.
        """
        self._send_frame(0, "R", "", "")
        logger.debug("Queried AGWPE version")

    def enable_raw_frames(self) -> None:
        """Enable reception of raw AX.25 frames ('k' command).

        After enabling, the server sends a 'k' DataKind frame for every
        raw frame it receives, including frames that AGWPE would normally
        not pass up (e.g., frames addressed to other stations).

        Raises:
            AGWPEConnectionError: If not connected or the send fails.
        """
        self._send_frame(0, "k", "", "")
        logger.info("Raw frame monitoring enabled")

    # -----------------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------------

    def register_callback(
        self,
        data_kind: str,
        callback: Callable[[int, str, str, bytes], None],
    ) -> None:
        """Register a function to call when a specific DataKind is received.

        The callback is called from the reader thread. It must be
        thread-safe and should return quickly.

        Args:
            data_kind: Single ASCII character (e.g. ``"D"`` for connected
                data, ``"U"`` for unproto, ``"X"`` for registration reply).
            callback: A callable with signature
                ``(port: int, call_from: str, call_to: str, data: bytes)``.

        Example::

            def on_data(port, frm, to, data):
                print(f"Data from {frm}: {data!r}")

            client.register_callback("D", on_data)
        """
        if len(data_kind) != 1:
            raise ValueError(
                f"data_kind must be a single ASCII character, got {data_kind!r}"
            )
        self._callbacks[data_kind] = callback
        logger.debug("Registered callback for DataKind '%s'", data_kind)

    # -----------------------------------------------------------------------
    # Frame sending
    # -----------------------------------------------------------------------

    def _send_frame(
        self,
        port: int,
        data_kind: str,
        call_from: str,
        call_to: str,
        data: bytes = b"",
    ) -> None:
        """Build and send one AGWPE frame over the TCP socket.

        Packs the 36-byte header followed by the data bytes and sends
        the whole thing in one ``sendall()`` call.

        Args:
            port: Radio port index (0-based).
            data_kind: Single ASCII character command code.
            call_from: Source callsign (up to 9 characters). Padded
                with NUL bytes to exactly 10 bytes.
            call_to: Destination callsign (up to 9 characters).
            data: Optional data payload bytes. Default is empty.

        Raises:
            AGWPEConnectionError: If not connected or the send fails.
        """
        if self.sock is None:
            raise AGWPEConnectionError("Not connected -- call connect() first")

        call_from_b = call_from.encode("ascii")[:9].ljust(10, b"\x00")
        call_to_b = call_to.encode("ascii")[:9].ljust(10, b"\x00")

        header = struct.pack(
            HEADER_FMT,
            port,
            ord(data_kind),
            call_from_b,
            call_to_b,
            len(data),
            0,   # USER field -- always 0
        )
        full_frame = header + data

        try:
            self.sock.sendall(full_frame)
            logger.debug(
                "_send_frame: port=%d kind='%s' from=%s to=%s data=%d bytes",
                port, data_kind, call_from, call_to, len(data),
            )
        except socket.error as exc:
            logger.error("_send_frame: socket send failed: %s", exc)
            raise AGWPEConnectionError(f"AGWPE send failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # Frame receiving
    # -----------------------------------------------------------------------

    def receive(
        self,
        timeout: Optional[float] = None,
    ) -> Tuple[int, str, str, str, bytes]:
        """Wait for and return the next received AGWPE frame.

        Blocks until a frame is available in the queue or the timeout
        expires. Frames arrive in the order they were received.

        Args:
            timeout: How many seconds to wait. None waits forever.
                0 does a non-blocking check.

        Returns:
            A 5-tuple of:
            - port (int): The radio port index.
            - data_kind (str): Single character DataKind code.
            - call_from (str): Source callsign.
            - call_to (str): Destination callsign.
            - data (bytes): The data payload (may be empty).

        Raises:
            AGWPEConnectionError: If the timeout expires before a frame
                arrives.

        Example::

            port, kind, frm, to, data = client.receive(timeout=5.0)
        """
        try:
            return self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise AGWPEConnectionError("receive: timeout -- no frame received")

    # -----------------------------------------------------------------------
    # Reader thread (internal)
    # -----------------------------------------------------------------------

    def _reader_thread(self) -> None:
        """Background thread that reads AGWPE frames from the TCP socket.

        Continuously reads 36-byte headers, then the data body, and puts
        each frame into the receive queue. Calls registered callbacks.
        Stops when ``self._running`` becomes False or the connection drops.
        """
        logger.debug("AGWPE reader thread started")

        while self._running:
            try:
                header_data = self._recv_exact(HEADER_SIZE)
                if not header_data:
                    break

                port, dk_int, call_from_b, call_to_b, data_len, user = struct.unpack(
                    HEADER_FMT, header_data
                )
                data_kind = chr(dk_int & 0xFF)
                call_from = call_from_b.rstrip(b"\x00").decode("ascii", errors="ignore")
                call_to = call_to_b.rstrip(b"\x00").decode("ascii", errors="ignore")

                data = self._recv_exact(data_len) if data_len > 0 else b""

                logger.debug(
                    "_reader_thread: port=%d kind='%s' from=%s to=%s data=%d bytes",
                    port, data_kind, call_from, call_to, len(data),
                )

                # Dispatch to registered callback
                if data_kind in self._callbacks:
                    try:
                        self._callbacks[data_kind](port, call_from, call_to, data)
                    except Exception as exc:
                        logger.error(
                            "_reader_thread: callback for '%s' raised: %s",
                            data_kind, exc,
                        )

                # Put in receive queue for receive()
                self._recv_queue.put((port, data_kind, call_from, call_to, data))

            except AGWPEFrameError as exc:
                logger.warning("_reader_thread: frame error: %s", exc)

            except socket.timeout:
                continue   # Normal -- just retry

            except socket.error as exc:
                if self._running:
                    logger.error("_reader_thread: socket read error: %s", exc)
                break

            except Exception as exc:
                if self._running:
                    logger.error("_reader_thread: unexpected error: %s", exc)
                break

        logger.info("AGWPE reader thread stopped")

    def _recv_exact(self, size: int) -> bytes:
        """Read exactly ``size`` bytes from the socket, blocking as needed.

        Args:
            size: The exact number of bytes to read.

        Returns:
            Exactly ``size`` bytes.

        Raises:
            AGWPEFrameError: If the connection closes before all bytes
                are received.
        """
        if size == 0:
            return b""

        data = b""
        while len(data) < size:
            try:
                chunk = self.sock.recv(size - len(data))
            except socket.timeout:
                continue

            if not chunk:
                raise AGWPEFrameError(
                    f"Connection closed while reading: needed {size} bytes, "
                    f"got {len(data)}"
                )
            data += chunk

        return data
