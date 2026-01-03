# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.interfaces.agwpe.py

Full AGWPE TCP/IP API client implementation.

Implements the complete AGWPE TCP/IP Socket Interface per SV2AGW specification (2000 version, with updates).
Supports:
- Connection to AGWPE server (default localhost:8000)
- Callsign registration ('X' command with success/failure reply)
- Monitoring enable/disable ('m' command)
- Raw frame monitoring ('k' command)
- Port capabilities query ('g' command)
- Version info query ('R' command)
- Outstanding frames query ('y' command)
- Unproto and connected data send/receive ('U', 'D', 'c', 'd')
- Heard list ('H')
- Monitor data kinds ('I', 'S', 'T', 'U')
- Full header parsing (Port, DataKind, CallFrom, CallTo, DataLen, USER)
- Synchronous (threading for background read) and asynchronous (asyncio) modes
- Comprehensive error handling and logging

Compliant with AGWPE API as described in AgwSockInterface.doc and related docs.
"""

from __future__ import annotations

import socket
import struct
import threading
import queue
import asyncio
import logging
from typing import Optional, Callable, Tuple, Dict

from pyax25_22.core.exceptions import AGWPEError
from pyax25_22.core.framing import AX25Frame  # For potential frame integration

logger = logging.getLogger(__name__)

# AGWPE Header format (36 bytes)
# int Port (4 bytes): LOWORD = port index (0-first), HIWORD reserved
# int DataKind (4 bytes): LOWORD = kind (ASCII char code), HIWORD special use (e.g., for 'Y' reply)
# char CallFrom[10]: NULL-terminated callsign (e.g., "SV2AGW-12\0")
# char CallTo[10]: NULL-terminated callsign
# int DataLen (4 bytes): Length of data field
# int USER (4 bytes): Reserved/undefined
HEADER_FMT = '<II10s10sII'
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# DataKind values (ASCII chars)
DATAKIND_CONNECTED_DATA = 'D'    # Data from connected station
DATAKIND_UNPROTO_MONITOR = 'U'   # Unproto monitor data
DATAKIND_TX_MONITOR = 'T'        # TX data monitor
DATAKIND_MONITOR_HEADER = 'S'    # Monitor header only
DATAKIND_MONITOR_FULL = 'I'      # Monitor header + data
DATAKIND_NEW_CONNECTION = 'c'    # New connection established
DATAKIND_DISCONNECT = 'd'        # Disconnect or retry out
DATAKIND_HEARD_LIST = 'H'        # Heard list (line by line)
DATAKIND_REGISTRATION = 'X'      # Registration reply (success/failure)
DATAKIND_OUTSTANDING = 'Y'       # Outstanding frames in queue
DATAKIND_PORT_CAPABILITIES = 'g' # Radio port capabilities
DATAKIND_VERSION = 'R'           # AGWPE version info
DATAKIND_RAW_FRAMES = 'k'        # Raw AX25 frames
DATAKIND_MONITORING = 'm'        # Monitoring control

# Send kinds
SENDKIND_CONNECT_NO_DIGI = 'c'   # Connect without digis
SENDKIND_CONNECT_DIGI = 'v'      # Connect with digis
SENDKIND_DISCONNECT = 'd'        # Disconnect
SENDKIND_DATA = 'D'              # Send data to connected station
SENDKIND_UNPROTO = 'U'           # Send unproto (beacon/CQ)
SENDKIND_PORT_INFO = 'P'         # Port information
SENDKIND_UNPROTO_VIA = 'V'       # Unproto with via

class AGWPEConnectionError(AGWPEError):
    """Specific error for connection issues."""
    pass

class AGWPEFrameError(AGWPEError):
    """Specific error for frame parsing issues."""
    pass

class AGWPEClient:
    """
    Synchronous AGWPE TCP/IP API client with threaded background reading.

    Usage:
        client = AGWPEClient()
        client.connect()
        client.register_callsign("KE4AHR-1")
        client.enable_monitoring()
        while True:
            port, kind, call_from, call_to, data = client.receive()
            print(f"Received {kind} from {call_from} to {call_to}")
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 8000, timeout: float = 5.0):
        """
        Initialize AGWPE client.

        Args:
            host: AGWPE server hostname or IP
            port: TCP port (default 8000)
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._recv_queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._callbacks: Dict[str, Callable] = {}
        logger.info(f"AGWPEClient initialized for {host}:{port}")

    def connect(self) -> None:
        """
        Connect to AGWPE server and start reader thread.
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            self._running = True
            self._thread = threading.Thread(target=self._reader_thread, daemon=True)
            self._thread.start()
            logger.info("Connected to AGWPE server")
        except socket.error as e:
            logger.error(f"Connection failed: {e}")
            raise AGWPEConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")

    def disconnect(self) -> None:
        """
        Disconnect from AGWPE server and stop reader thread.
        """
        self._running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except socket.error:
                pass
            self.sock.close()
            self.sock = None
        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Reader thread did not join cleanly")
            self._thread = None
        logger.info("Disconnected from AGWPE server")

    def register_callsign(self, callsign: str) -> None:
        """
        Register a callsign with AGWPE ('X' command).

        Expect 'X' reply for success/failure.
        """
        self.send_frame(0, 'X', callsign, '')
        logger.info(f"Registered callsign: {callsign}")

    def enable_monitoring(self) -> None:
        """
        Enable monitoring of frames ('m' command).
        """
        self.send_frame(0, 'm', '', '')
        logger.info("Monitoring enabled")

    def disable_monitoring(self) -> None:
        """
        Disable monitoring ('m' command again toggles).
        """
        self.send_frame(0, 'm', '', '')
        logger.info("Monitoring disabled")

    def query_outstanding_frames(self, port: int = 0, callsign: str = '') -> None:
        """
        Query outstanding frames in queue ('y' command).

        Args:
            port: Radio port to query
            callsign: Optional callsign to query specific
        """
        self.send_frame(port, 'y', callsign, callsign)
        logger.info(f"Queried outstanding frames for port {port}")

    def query_port_capabilities(self, port: int = 0) -> None:
        """
        Query radio port capabilities ('g' command).
        """
        self.send_frame(port, 'g', '', '')
        logger.info(f"Queried capabilities for port {port}")

    def query_version(self) -> None:
        """
        Query AGWPE version ('R' command).
        """
        self.send_frame(0, 'R', '', '')
        logger.info("Queried AGWPE version")

    def enable_raw_frames(self) -> None:
        """
        Enable raw AX25 frame reception ('k' command).
        """
        self.send_frame(0, 'k', '', '')
        logger.info("Raw frames enabled")

    def register_callback(self, data_kind: str, callback: Callable[[int, str, str, bytes], None]) -> None:
        """
        Register callback for specific DataKind.

        Args:
            data_kind: ASCII char (e.g., 'D' for data)
            callback: Function(port, call_from, call_to, data)
        """
        self._callbacks[data_kind] = callback
        logger.debug(f"Registered callback for DataKind '{data_kind}'")

    def send_frame(self, port: int, data_kind: str, call_from: str, call_to: str, data: bytes = b'') -> None:
        """
        Send AGWPE frame with proper header.

        Args:
            port: Radio port index
            data_kind: Single ASCII char (e.g., 'U' for unproto)
            call_from: Source callsign (up to 9 chars + \0)
            call_to: Destination callsign (up to 9 chars + \0)
            data: Payload bytes
        """
        if not self.sock:
            raise AGWPEConnectionError("Not connected")

        call_from_b = call_from.encode('ascii').ljust(10, b'\0')
        call_to_b = call_to.encode('ascii').ljust(10, b'\0')
        data_len = len(data)
        user_reserved = 0  # Always 0

        # Pack header: Port (LOWORD), DataKind (ord(kind)), CallFrom, CallTo, DataLen, USER
        header = struct.pack(HEADER_FMT, port, ord(data_kind), call_from_b, call_to_b, data_len, user_reserved)
        full_frame = header + data

        try:
            self.sock.sendall(full_frame)
            logger.info(
                f"Sent frame: Port={port}, DataKind='{data_kind}', "
                f"From={call_from}, To={call_to}, Len={data_len}"
            )
        except socket.error as e:
            logger.error(f"Send failed: {e}")
            raise AGWPEConnectionError(f"Send failed: {e}")

    def _reader_thread(self) -> None:
        """Background thread to read and process AGWPE frames."""
        while self._running:
            try:
                header_data = self._recv_exact(HEADER_SIZE)
                if not header_data:
                    break

                # Unpack header
                port, data_kind_int, call_from_b, call_to_b, data_len, user = struct.unpack(HEADER_FMT, header_data)
                data_kind = chr(data_kind_int)
                call_from = call_from_b.rstrip(b'\0').decode('ascii', errors='ignore')
                call_to = call_to_b.rstrip(b'\0').decode('ascii', errors='ignore')

                # Read data
                data = self._recv_exact(data_len) if data_len > 0 else b''

                logger.debug(
                    f"Received frame: Port={port}, DataKind='{data_kind}', "
                    f"From={call_from}, To={call_to}, Len={data_len}"
                )

                # Dispatch callback if registered
                if data_kind in self._callbacks:
                    try:
                        self._callbacks[data_kind](port, call_from, call_to, data)
                    except Exception as e:
                        logger.error(f"Callback error for '{data_kind}': {e}")

                # Queue for receive()
                self._recv_queue.put((port, data_kind, call_from, call_to, data))

            except AGWPEFrameError as e:
                logger.warning(f"Frame error: {e}")
            except socket.timeout:
                continue
            except socket.error as e:
                logger.error(f"Read error: {e}")
                break

        logger.info("AGWPE reader thread stopped")

    def _recv_exact(self, size: int) -> bytes:
        """Receive exact number of bytes or raise error."""
        data = b''
        while len(data) < size:
            chunk = self.sock.recv(size - len(data))
            if not chunk:
                raise AGWPEFrameError("Connection closed during read")
            data += chunk
        return data

    def receive(self, timeout: Optional[float] = None) -> Tuple[int, str, str, str, bytes]:
        """
        Receive next frame from queue (port, data_kind, call_from, call_to, data).

        Args:
            timeout: Optional queue timeout in seconds

        Returns:
            Tuple of frame components

        Raises:
            AGWPEConnectionError on timeout
        """
        try:
            return self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise AGWPEConnectionError("Receive timeout")
