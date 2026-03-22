# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/agw/client.py

AGWPEClient -- full AGWPE TCP/IP API client.

Merged from PyAGW3 with the following improvements:
- Correct callsign registration using 'X' frame (PyAGW3 used 'R' by mistake).
- Exponential backoff reconnect (from PyAGW3).
- TCP keepalive (SO_KEEPALIVE + TCP_KEEPIDLE/KEEPINTVL/KEEPCNT).
- All frame kinds documented and dispatched.
- Comprehensive logging and error handling.
- Thread-safe with RLock.

AGWPE Header format (36 bytes, little-endian):
    Offset  Size  Field
    0       1     data_kind (ASCII frame type byte)
    1       3     reserved (zero)
    4       4     port (uint32 LE)
    8       10    call_from (space-padded ASCII)
    18      10    call_to (space-padded ASCII)
    28      4     data_len (uint32 LE)
    32      4     reserved (zero)
"""

import logging
import random
import socket
import struct
import threading
import time
from typing import Callable, Dict, List, Optional

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

logger = logging.getLogger(__name__)

# Safety limit: reject frames claiming more than 64 KiB of data
_MAX_DATA_LEN = 65536


class AGWPEFrame:
    """Represents a single decoded AGWPE frame.

    Attributes:
        data_kind: Single-byte frame type (e.g., ``b'D'`` for unproto data).
        port: TNC port number (0-based).
        call_from: Source callsign string (stripped of padding).
        call_to: Destination callsign string (stripped of padding).
        data_len: Length of the data field in bytes.
        data: Raw data payload bytes.
    """

    def __init__(self) -> None:
        self.data_kind: bytes = b""
        self.port: int = 0
        self.call_from: str = ""
        self.call_to: str = ""
        self.data_len: int = 0
        self.data: bytes = b""

    def __repr__(self) -> str:
        return (
            f"AGWPEFrame(kind={self.data_kind!r} port={self.port} "
            f"from={self.call_from!r} to={self.call_to!r} "
            f"data_len={self.data_len})"
        )


class AGWPEClient:
    """Full AGWPE TCP/IP API client.

    Connects to an AGWPE-compatible server (e.g., Direwolf, AGWPE, UISS),
    registers a callsign, and dispatches received frames to registered
    callbacks.

    Args:
        host: Hostname or IP address of the AGWPE server (default: localhost).
        port: TCP port (default: 8000).
        callsign: Station callsign to register with AGWPE.

    Example:
        client = AGWPEClient("localhost", 8000, callsign="KE4AHR")
        client.on_frame = lambda f: print(f)
        client.connect()
        client.send_ui(port=0, dest="APRS", src="KE4AHR", pid=0xF0,
                       info=b"Hello")
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

        self.sock: Optional[socket.socket] = None
        self.connected = False

        # User-settable callbacks (set before calling connect())
        self.on_frame: Optional[Callable[[AGWPEFrame], None]] = None
        self.on_connected_data: Optional[Callable[[int, str, bytes], None]] = None
        self.on_outstanding: Optional[Callable[[int, int], None]] = None
        self.on_heard_stations: Optional[Callable[[int, List[Dict]], None]] = None
        self.on_extended_version: Optional[Callable[[str], None]] = None
        self.on_memory_usage: Optional[Callable[[Dict[str, int]], None]] = None
        self.on_connect: Optional[Callable[[int, str], None]] = None
        self.on_disconnect: Optional[Callable[[int, str], None]] = None

        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._rx_buffer = b""

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    def connect(
        self, max_retries: int = 10, base_delay: float = 1.0
    ) -> bool:
        """Connect to AGWPE with exponential backoff retry.

        On success, registers the callsign with the server using an 'X' frame
        and starts the background reader thread.

        Args:
            max_retries: Maximum number of connection attempts before giving up.
            base_delay: Base delay (seconds) for the exponential backoff.

        Returns:
            True if the connection was established, False otherwise.

        Raises:
            AGWConnectionError: If max_retries is exhausted.
        """
        attempt = 0
        while attempt <= max_retries:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                # TCP keepalive
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if hasattr(socket, "TCP_KEEPIDLE"):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
                if hasattr(socket, "TCP_KEEPINTVL"):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                if hasattr(socket, "TCP_KEEPCNT"):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)

                sock.connect((self.host, self.port))
                sock.settimeout(2.0)
                self.sock = sock
                self.connected = True

                # Register callsign using 'X' (not 'R' -- 'R' is version info)
                self._send_frame(
                    data_kind=KIND_REGISTER,
                    call_from=self.callsign,
                )

                self._thread = threading.Thread(
                    target=self._receive_loop,
                    name=f"AGWPE-{self.host}:{self.port}",
                    daemon=True,
                )
                self._thread.start()

                logger.info(
                    "AGWPE: connected to %s:%d as %s (attempt %d)",
                    self.host, self.port, self.callsign, attempt + 1,
                )
                return True

            except OSError as exc:
                if self.sock:
                    self.sock.close()
                    self.sock = None
                self.connected = False
                attempt += 1

                if attempt > max_retries:
                    logger.error(
                        "AGWPE: connection failed after %d retries: %s",
                        max_retries, exc,
                    )
                    raise AGWConnectionError(
                        f"AGWPE connect failed after {max_retries} retries: {exc}"
                    ) from exc

                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(
                    0, base_delay
                )
                logger.warning(
                    "AGWPE: attempt %d failed (%s) -- retry in %.2f s",
                    attempt, exc, delay,
                )
                time.sleep(delay)

        return False

    def close(self) -> None:
        """Disconnect from AGWPE and stop the reader thread."""
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        logger.info("AGWPE: disconnected from %s:%d", self.host, self.port)

    # -----------------------------------------------------------------------
    # Low-level frame transmitter
    # -----------------------------------------------------------------------

    def _send_frame(
        self,
        data_kind: bytes,
        port: int = 0,
        call_from: str = "",
        call_to: str = "",
        data: bytes = b"",
    ) -> None:
        """Build and send a raw 36-byte AGWPE header + data.

        Args:
            data_kind: Single-byte frame type (e.g., KIND_REGISTER = b'X').
            port: TNC port number.
            call_from: Source callsign (max 10 chars, space-padded).
            call_to: Destination callsign (max 10 chars, space-padded).
            data: Frame data payload.

        Raises:
            AGWConnectionError: If not connected.
        """
        if not self.connected or not self.sock:
            raise AGWConnectionError("AGWPE: not connected")

        cf_bytes = call_from.upper().ljust(CALLSIGN_WIDTH)[:CALLSIGN_WIDTH].encode("ascii", errors="replace")
        ct_bytes = call_to.upper().ljust(CALLSIGN_WIDTH)[:CALLSIGN_WIDTH].encode("ascii", errors="replace")

        header = bytearray(AGWPE_HEADER_SIZE)
        header[0:1] = data_kind
        struct.pack_into("<I", header, 4, port)
        header[8:18] = cf_bytes
        header[18:28] = ct_bytes
        struct.pack_into("<I", header, 28, len(data))

        packet = bytes(header) + data

        with self._lock:
            try:
                self.sock.sendall(packet)
                logger.debug(
                    "AGWPE send: kind=%s port=%d from=%s to=%s data_len=%d",
                    data_kind.decode("ascii", errors="?"),
                    port, call_from, call_to, len(data),
                )
            except OSError as exc:
                logger.error("AGWPE send failed: %s", exc)
                self.connected = False
                raise AGWConnectionError(f"AGWPE send failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # High-level transmit methods
    # -----------------------------------------------------------------------

    def send_ui(
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
            dest: Destination callsign (e.g., ``"APRS"``).
            src: Source callsign.
            pid: Protocol identifier byte (e.g., ``0xF0`` for no layer 3).
            info: Information field bytes.
        """
        self._send_frame(
            data_kind=KIND_UNPROTO_DATA,
            port=port,
            call_from=src,
            call_to=dest,
            data=bytes([pid]) + info,
        )

    def send_raw(
        self,
        port: int,
        dest: str,
        src: str,
        data: bytes,
    ) -> None:
        """Send a raw (unformatted) frame using the 'K' frame type.

        Args:
            port: TNC port number.
            dest: Destination callsign.
            src: Source callsign.
            data: Raw frame data.
        """
        self._send_frame(
            data_kind=KIND_RAW_SEND,
            port=port,
            call_from=src,
            call_to=dest,
            data=data,
        )

    def send_connect(self, port: int, dest: str) -> None:
        """Initiate a connected-mode (AX.25) connection.

        Args:
            port: TNC port number.
            dest: Remote station callsign.
        """
        self._send_frame(
            data_kind=KIND_CONNECT,
            port=port,
            call_from=self.callsign,
            call_to=dest,
        )

    def send_disconnect(self, port: int, dest: str) -> None:
        """Disconnect from a connected-mode session.

        Args:
            port: TNC port number.
            dest: Remote station callsign.
        """
        self._send_frame(
            data_kind=KIND_DISC,
            port=port,
            call_from=self.callsign,
            call_to=dest,
        )

    def send_connected_data(
        self, port: int, dest: str, data: bytes
    ) -> None:
        """Send data on an established connected-mode circuit.

        Args:
            port: TNC port number.
            dest: Remote station callsign.
            data: Data bytes to send.
        """
        self._send_frame(
            data_kind=KIND_CONN_DATA,
            port=port,
            call_from=self.callsign,
            call_to=dest,
            data=data,
        )

    def enable_monitor(self, port: int = 0) -> None:
        """Enable frame monitoring on a port (receive all frames).

        Args:
            port: TNC port to monitor.
        """
        self._send_frame(data_kind=KIND_ENABLE_MON, port=port)

    def request_outstanding(self, port: int = 0) -> None:
        """Query the number of outstanding (unacknowledged) frames.

        Args:
            port: TNC port to query.
        """
        self._send_frame(data_kind=KIND_OUTSTANDING, port=port)

    def request_heard_stations(self, port: int = 0) -> None:
        """Request the list of recently heard stations.

        Args:
            port: TNC port to query.
        """
        self._send_frame(data_kind=KIND_HEARD, port=port)

    def send_login(self, username: str, password: str) -> None:
        """Send AGWPE login credentials ('T' frame).

        Note: Login is not required by all AGWPE implementations.

        Args:
            username: Login username.
            password: Login password.
        """
        payload = f"{username}\x00{password}\x00".encode("ascii", errors="replace")
        self._send_frame(data_kind=KIND_LOGIN, data=payload)

    def set_parameter(self, port: int, param_id: int, value: int) -> None:
        """Set a TNC parameter via AGWPE.

        Args:
            port: TNC port number.
            param_id: Parameter identifier byte.
            value: Parameter value (32-bit unsigned).
        """
        payload = struct.pack("<BI", param_id, value)
        self._send_frame(data_kind=KIND_PARAMETER, port=port, data=payload)

    def request_version(self) -> None:
        """Request the AGWPE version string ('R' frame)."""
        self._send_frame(data_kind=KIND_VERSION)

    def request_extended_version(self) -> None:
        """Request extended version information ('v' frame)."""
        self._send_frame(data_kind=KIND_EXTENDED_VER)

    def request_memory_usage(self) -> None:
        """Request memory usage information ('m' frame)."""
        self._send_frame(data_kind=KIND_MEMORY_USAGE)

    def request_port_info(self) -> None:
        """Request information about available TNC ports ('G' frame)."""
        self._send_frame(data_kind=KIND_PORT_INFO)

    # -----------------------------------------------------------------------
    # Receiver
    # -----------------------------------------------------------------------

    def _receive_loop(self) -> None:
        """Background thread: receive and parse AGWPE frames.

        Accumulates raw TCP bytes in a buffer, parses complete 36-byte headers
        + data fields, and dispatches each completed frame to the appropriate
        callback.
        """
        logger.debug("AGWPE reader thread started (%s:%d)", self.host, self.port)
        buf = b""

        while self.connected and self.sock:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    logger.warning(
                        "AGWPE: server %s:%d closed connection",
                        self.host, self.port,
                    )
                    self.connected = False
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
                            "AGWPE: received data_len=%d exceeds safety limit -- "
                            "dropping connection",
                            data_len,
                        )
                        self.connected = False
                        break

                    if len(buf) < AGWPE_HEADER_SIZE + data_len:
                        break   # Wait for more data

                    payload = buf[AGWPE_HEADER_SIZE: AGWPE_HEADER_SIZE + data_len]
                    buf = buf[AGWPE_HEADER_SIZE + data_len:]

                    frame = AGWPEFrame()
                    frame.data_kind = data_kind
                    frame.port = port
                    frame.call_from = call_from
                    frame.call_to = call_to
                    frame.data_len = data_len
                    frame.data = payload

                    logger.debug("AGWPE rx: %r", frame)
                    self._dispatch(frame)

            except socket.timeout:
                continue
            except OSError as exc:
                if self.connected:
                    logger.error(
                        "AGWPE receive error (%s:%d): %s", self.host, self.port, exc
                    )
                self.connected = False
                break
            except Exception:
                logger.exception(
                    "AGWPE: unexpected error in reader thread (%s:%d)",
                    self.host, self.port,
                )
                self.connected = False
                break

        logger.debug("AGWPE reader thread stopped (%s:%d)", self.host, self.port)

    def _dispatch(self, frame: AGWPEFrame) -> None:
        """Route a received AGWPE frame to the appropriate callback.

        Args:
            frame: A fully decoded AGWPEFrame.
        """
        dk = frame.data_kind

        try:
            if dk in (KIND_UNPROTO_DATA, KIND_RAW_MON, KIND_UNPROTO,
                       KIND_UNPROTO_VIA):
                # 'D' uppercase = unproto/monitored data
                if self.on_frame:
                    self.on_frame(frame)

            elif dk == KIND_CONNECT_INC:
                # 'c' lowercase = incoming connection notification
                if self.on_connect:
                    self.on_connect(frame.port, frame.call_from)

            elif dk == KIND_DISC:
                # 'd' lowercase = connected data (if data present) or disconnect
                if frame.data:
                    if self.on_connected_data:
                        self.on_connected_data(
                            frame.port, frame.call_from, frame.data
                        )
                    elif self.on_frame:
                        self.on_frame(frame)
                else:
                    if self.on_disconnect:
                        self.on_disconnect(frame.port, frame.call_from)

            elif dk in (KIND_OUTSTANDING, KIND_OUTSTANDING_R):
                if frame.data and len(frame.data) >= 4:
                    count = struct.unpack("<I", frame.data[:4])[0]
                    if self.on_outstanding:
                        self.on_outstanding(frame.port, count)

            elif dk == KIND_HEARD:
                heard_list = self._parse_heard(frame.data)
                if self.on_heard_stations:
                    self.on_heard_stations(frame.port, heard_list)

            elif dk == KIND_EXTENDED_VER:
                version_str = frame.data.decode("ascii", errors="ignore").strip()
                if self.on_extended_version:
                    self.on_extended_version(version_str)

            elif dk == KIND_MEMORY_USAGE:
                mem = self._parse_memory(frame.data)
                if self.on_memory_usage:
                    self.on_memory_usage(mem)

            elif dk == KIND_PORT_CAPS:
                logger.debug(
                    "AGWPE: port caps received (port=%d)", frame.port
                )

            else:
                logger.debug(
                    "AGWPE: unhandled frame kind %r (port=%d)",
                    dk, frame.port,
                )

        except Exception:
            logger.exception(
                "AGWPE: error in dispatch callback for kind %r", dk
            )

    # -----------------------------------------------------------------------
    # Parsers for complex response frames
    # -----------------------------------------------------------------------

    def _parse_heard(self, data: bytes) -> List[Dict]:
        """Parse an 'H' heard-stations response.

        Args:
            data: Raw payload from an 'H' frame.

        Returns:
            List of dicts with keys ``callsign`` and ``last_heard`` (timestamp).
        """
        heard = []
        entry_size = 14
        for i in range(len(data) // entry_size):
            offset = i * entry_size
            call = data[offset:offset + 10].decode("ascii", errors="ignore").strip()
            ts = struct.unpack("<I", data[offset + 10: offset + 14])[0]
            if call:
                heard.append({"callsign": call, "last_heard": ts})
        return heard

    def _parse_memory(self, data: bytes) -> Dict[str, int]:
        """Parse an 'm' memory-usage response.

        Args:
            data: Raw payload from an 'm' frame.

        Returns:
            Dict with keys ``free_kb`` and ``used_kb``.
        """
        if len(data) >= 8:
            free_mem = struct.unpack("<I", data[0:4])[0]
            used_mem = struct.unpack("<I", data[4:8])[0]
            return {"free_kb": free_mem // 1024, "used_kb": used_mem // 1024}
        return {"free_kb": 0, "used_kb": 0}
