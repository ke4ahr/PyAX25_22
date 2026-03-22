# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/agw/serial.py

AGWSerial -- AGWPE protocol server bridging to a serial KISS TNC.

Acts as an AGWPE-compatible TCP server.  Applications (UISS, WinPack, etc.)
connect to this server using the AGWPE protocol.  Received AGWPE frames are
translated to KISS and forwarded to the serial TNC.  Frames received from the
TNC (via KISS) are translated back to AGWPE and forwarded to all connected
clients.

Supported AGWPE frame types:
    R  -- Version info response (to client 'R' requests)
    G  -- Port info response
    X  -- Callsign registration (accepted, no TNC action)
    x  -- Callsign unregistration
    M  -- Enable monitoring (receive all frames via 'K' callbacks)
    K  -- Raw AX.25 frame transmit (passed directly to KISS)
    k  -- Raw AX.25 frame receive (forwarded to monitoring clients)
    D  -- Unproto UI data send/receive (builds AX.25 UI frame)

Connected-mode frames (C, c, d, I) require AX.25 Layer 2 state machine
integration with core.AX25Connection -- not yet implemented in this module.

Wire format (AGWPE header, 36 bytes):
    Offset  Size  Description
    0       1     data_kind (ASCII frame type)
    1       3     reserved (zero)
    4       4     port number (little-endian uint32)
    8       10    call_from (space-padded ASCII)
    18      10    call_to (space-padded ASCII)
    28      4     data_len (little-endian uint32)
    32      4     reserved (zero)
"""

import logging
import select
import socket
import struct
import threading
from typing import Dict, List, Optional, Set

from ..kiss.serial import KISSSerial
from ..kiss.constants import CMD_DATA
from .constants import (
    AGWPE_DEFAULT_PORT,
    AGWPE_HEADER_SIZE,
    CALLSIGN_WIDTH,
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
from .exceptions import AGWConnectionError, AGWFrameError
from .client import AGWPEFrame

logger = logging.getLogger(__name__)

# Safety limit on incoming data_len
_MAX_DATA_LEN = 65536

# AGWPE version string returned in 'R' responses
_VERSION_STRING = "2.0 (PyAX25_22 AGWSerial bridge)"

# -----------------------------------------------------------------------
# AX.25 address helpers
# -----------------------------------------------------------------------

def _encode_ax25_addr(callsign: str, ssid: int = 0, last: bool = False) -> bytes:
    """Encode an AX.25 7-byte address field.

    Each of the 6 callsign characters is shifted left 1 bit.  The seventh byte
    holds the SSID (bits 1-4), the H-bit (bit 7), and the end-of-address bit
    (bit 0).

    Args:
        callsign: Up to 6-character callsign (letters and digits).
        ssid: Secondary Station ID 0-15 (default 0).
        last: True if this is the last address in the address field.

    Returns:
        7-byte AX.25 address encoding.
    """
    padded = callsign.upper().ljust(6)[:6]
    addr_bytes = bytes([ord(c) << 1 for c in padded])
    ssid_byte = ((ssid & 0x0F) << 1) | 0x60  # bits 5-6 always 1 per AX.25 spec
    if last:
        ssid_byte |= 0x01
    return addr_bytes + bytes([ssid_byte])


def _decode_ax25_addr(data: bytes, offset: int = 0):
    """Decode one 7-byte AX.25 address field.

    Args:
        data: Byte buffer.
        offset: Starting byte offset in data.

    Returns:
        Tuple (callsign: str, ssid: int, is_last: bool).
    """
    call_bytes = data[offset: offset + 6]
    ssid_byte = data[offset + 6]
    call = "".join(chr(b >> 1) for b in call_bytes).rstrip()
    ssid = (ssid_byte >> 1) & 0x0F
    is_last = bool(ssid_byte & 0x01)
    return call, ssid, is_last


def _build_ui_frame(
    dest: str,
    src: str,
    pid: int,
    info: bytes,
    digipeaters: Optional[List[str]] = None,
) -> bytes:
    """Build an AX.25 UI frame.

    UI frame structure:
        [DEST 7 bytes][SRC 7 bytes][DIGI... 7 bytes each][CTRL 0x03][PID 1 byte][INFO ...]

    Args:
        dest: Destination callsign.
        src: Source callsign.
        pid: Protocol identifier (0xF0 for no layer 3, 0xCF for NET/ROM, etc.)
        info: Information field bytes.
        digipeaters: Optional list of digipeater callsigns.

    Returns:
        Raw AX.25 UI frame bytes.
    """
    digipeaters = digipeaters or []
    last_in_chain = not bool(digipeaters)
    frame = bytearray()
    frame += _encode_ax25_addr(dest, last=False)
    frame += _encode_ax25_addr(src, last=last_in_chain)
    for i, digi in enumerate(digipeaters):
        is_last = (i == len(digipeaters) - 1)
        frame += _encode_ax25_addr(digi, last=is_last)
    frame.append(0x03)   # Control: UI frame
    frame.append(pid & 0xFF)
    frame += info
    return bytes(frame)


def _parse_ax25_addresses(frame: bytes):
    """Parse the AX.25 address field from the start of a frame.

    Args:
        frame: Raw AX.25 frame bytes (starting with destination address).

    Returns:
        Tuple (dest, src, digipeaters, header_len) where:
            dest -- destination callsign string (with SSID if nonzero)
            src -- source callsign string (with SSID if nonzero)
            digipeaters -- list of digipeater callsign strings
            header_len -- total byte length of the address field
    """
    if len(frame) < 14:
        return None, None, [], 0

    def fmt_call(call, ssid):
        return call if ssid == 0 else f"{call}-{ssid}"

    dest_call, dest_ssid, _ = _decode_ax25_addr(frame, 0)
    src_call, src_ssid, src_last = _decode_ax25_addr(frame, 7)
    dest = fmt_call(dest_call, dest_ssid)
    src = fmt_call(src_call, src_ssid)
    digipeaters = []

    offset = 14
    if not src_last:
        while offset + 7 <= len(frame):
            digi_call, digi_ssid, digi_last = _decode_ax25_addr(frame, offset)
            digipeaters.append(fmt_call(digi_call, digi_ssid))
            offset += 7
            if digi_last:
                break

    return dest, src, digipeaters, offset


# -----------------------------------------------------------------------
# Per-client connection state
# -----------------------------------------------------------------------

class _AGWClientConn:
    """State for one connected AGWPE client."""

    def __init__(self, sock: socket.socket, addr) -> None:
        self.sock = sock
        self.addr = addr
        self.buf = b""
        self.monitoring = False          # 'M' frame received -- wants all frames
        self.registered_calls: Set[str] = set()
        self._lock = threading.Lock()

    def send_frame(
        self,
        data_kind: bytes,
        port: int = 0,
        call_from: str = "",
        call_to: str = "",
        data: bytes = b"",
    ) -> bool:
        """Build and send one AGWPE frame to this client.

        Returns:
            True if sent successfully, False if the socket is broken.
        """
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
        with self._lock:
            try:
                self.sock.sendall(packet)
                return True
            except OSError as exc:
                logger.debug(
                    "AGWSerial: send to client %s failed: %s", self.addr, exc
                )
                return False

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# -----------------------------------------------------------------------
# Bridge server
# -----------------------------------------------------------------------

class AGWSerial:
    """AGWPE protocol server bridging to a serial KISS TNC.

    Listens for AGWPE TCP client connections and translates between the AGWPE
    API and KISS frames sent/received over a serial TNC.

    Args:
        serial_port: Serial port device (e.g., ``"/dev/ttyUSB0"``).
        baud_rate: Serial baud rate (default 9600).
        agw_host: Interface to listen on (default ``"127.0.0.1"``).
        agw_port: TCP port to listen on (default 8000).
        port_name: Human-readable name for TNC port 0 (used in 'G' responses).

    Example::

        bridge = AGWSerial("/dev/ttyUSB0", 9600)
        bridge.start()
        # Applications can now connect on localhost:8000
        input("Press Enter to stop...")
        bridge.stop()
    """

    def __init__(
        self,
        serial_port: str,
        baud_rate: int = 9600,
        agw_host: str = "127.0.0.1",
        agw_port: int = AGWPE_DEFAULT_PORT,
        port_name: str = "KISS Port 0",
    ) -> None:
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.agw_host = agw_host
        self.agw_port = agw_port
        self.port_name = port_name

        self._kiss: Optional[KISSSerial] = None
        self._server_sock: Optional[socket.socket] = None
        self._clients: Dict[socket.socket, _AGWClientConn] = {}
        self._clients_lock = threading.Lock()

        self._accept_thread: Optional[threading.Thread] = None
        self._running = False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Open the serial TNC and start the AGWPE server.

        Raises:
            OSError: If the serial port or TCP socket cannot be opened.
        """
        # Open KISS serial TNC
        self._kiss = KISSSerial(
            port=self.serial_port,
            baud=self.baud_rate,
            on_frame=self._on_kiss_frame,
        )
        logger.info(
            "AGWSerial: TNC opened on %s at %d baud",
            self.serial_port, self.baud_rate,
        )

        # Open AGW server socket
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.agw_host, self.agw_port))
        self._server_sock.listen(8)
        self._server_sock.settimeout(1.0)
        logger.info(
            "AGWSerial: AGWPE server listening on %s:%d",
            self.agw_host, self.agw_port,
        )

        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="AGWSerial-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def stop(self) -> None:
        """Stop the AGWPE server and close the serial TNC."""
        self._running = False

        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        with self._clients_lock:
            for conn in list(self._clients.values()):
                conn.close()
            self._clients.clear()

        if self._kiss:
            self._kiss.close()
            self._kiss = None

        logger.info("AGWSerial: stopped")

    # -----------------------------------------------------------------------
    # TCP server accept loop
    # -----------------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Accept new AGWPE client connections and spawn per-client threads."""
        logger.debug("AGWSerial: accept loop started")
        while self._running:
            try:
                client_sock, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            client_sock.settimeout(1.0)
            conn = _AGWClientConn(client_sock, addr)
            with self._clients_lock:
                self._clients[client_sock] = conn

            t = threading.Thread(
                target=self._client_loop,
                args=(conn,),
                name=f"AGWSerial-client-{addr}",
                daemon=True,
            )
            t.start()
            logger.info("AGWSerial: client connected from %s", addr)

        logger.debug("AGWSerial: accept loop stopped")

    # -----------------------------------------------------------------------
    # Per-client receive loop
    # -----------------------------------------------------------------------

    def _client_loop(self, conn: _AGWClientConn) -> None:
        """Read and process AGWPE frames from one connected client."""
        while self._running:
            try:
                chunk = conn.sock.recv(4096)
                if not chunk:
                    break
                conn.buf += chunk

                while len(conn.buf) >= AGWPE_HEADER_SIZE:
                    data_kind = conn.buf[0:1]
                    port = struct.unpack("<I", conn.buf[4:8])[0]
                    call_from = conn.buf[8:18].decode("ascii", errors="ignore").strip()
                    call_to = conn.buf[18:28].decode("ascii", errors="ignore").strip()
                    data_len = struct.unpack("<I", conn.buf[28:32])[0]

                    if data_len > _MAX_DATA_LEN:
                        logger.error(
                            "AGWSerial: client %s sent data_len=%d -- dropping",
                            conn.addr, data_len,
                        )
                        break

                    if len(conn.buf) < AGWPE_HEADER_SIZE + data_len:
                        break   # Wait for the rest

                    payload = conn.buf[AGWPE_HEADER_SIZE: AGWPE_HEADER_SIZE + data_len]
                    conn.buf = conn.buf[AGWPE_HEADER_SIZE + data_len:]

                    frame = AGWPEFrame()
                    frame.data_kind = data_kind
                    frame.port = port
                    frame.call_from = call_from
                    frame.call_to = call_to
                    frame.data_len = data_len
                    frame.data = payload

                    self._handle_client_frame(conn, frame)

            except socket.timeout:
                continue
            except OSError:
                break

        logger.info("AGWSerial: client %s disconnected", conn.addr)
        with self._clients_lock:
            self._clients.pop(conn.sock, None)
        conn.close()

    # -----------------------------------------------------------------------
    # Client -> TNC (AGW frame handling)
    # -----------------------------------------------------------------------

    def _handle_client_frame(
        self, conn: _AGWClientConn, frame: AGWPEFrame
    ) -> None:
        """Dispatch one frame received from an AGWPE client.

        Args:
            conn: The client connection that sent the frame.
            frame: Decoded AGWPE frame.
        """
        dk = frame.data_kind
        logger.debug(
            "AGWSerial rx from %s: kind=%s port=%d from=%s to=%s data_len=%d",
            conn.addr, dk, frame.port, frame.call_from, frame.call_to,
            frame.data_len,
        )

        if dk == KIND_VERSION:
            # Client requests version info -- respond immediately
            self._send_version(conn)

        elif dk == KIND_PORT_INFO:
            # Client requests port list -- respond with our one KISS port
            self._send_port_info(conn)

        elif dk == KIND_REGISTER:
            # Callsign registration: accept it, no TNC action
            conn.registered_calls.add(frame.call_from.upper())
            logger.info(
                "AGWSerial: client %s registered callsign %s",
                conn.addr, frame.call_from,
            )
            # ACK with an 'X' response (registered = 1)
            conn.send_frame(
                data_kind=KIND_REGISTER,
                call_from=frame.call_from,
                data=b"\x01",
            )

        elif dk == KIND_UNREGISTER:
            conn.registered_calls.discard(frame.call_from.upper())
            logger.info(
                "AGWSerial: client %s unregistered callsign %s",
                conn.addr, frame.call_from,
            )

        elif dk == KIND_ENABLE_MON:
            # Enable monitoring: all received KISS frames are forwarded as 'K'
            conn.monitoring = True
            logger.info("AGWSerial: client %s enabled monitoring", conn.addr)

        elif dk == KIND_RAW_SEND:
            # Raw send: data is a complete AX.25 frame, pass to KISS directly
            if self._kiss and frame.data:
                try:
                    self._kiss.send(frame.data, cmd=CMD_DATA)
                    logger.debug(
                        "AGWSerial: raw send %d bytes to TNC (port %d)",
                        len(frame.data), frame.port,
                    )
                except OSError as exc:
                    logger.error("AGWSerial: KISS write failed: %s", exc)

        elif dk == KIND_UNPROTO_DATA:
            # UI frame: data[0] = PID, data[1:] = info field
            if self._kiss and frame.data:
                pid = frame.data[0] if frame.data else 0xF0
                info = frame.data[1:] if len(frame.data) > 1 else b""
                ax25 = _build_ui_frame(
                    dest=frame.call_to,
                    src=frame.call_from,
                    pid=pid,
                    info=info,
                )
                try:
                    self._kiss.send(ax25, cmd=CMD_DATA)
                    logger.debug(
                        "AGWSerial: UI frame %s -> %s (%d bytes) sent to TNC",
                        frame.call_from, frame.call_to, len(ax25),
                    )
                except OSError as exc:
                    logger.error("AGWSerial: KISS write failed: %s", exc)

        else:
            logger.debug(
                "AGWSerial: unhandled frame kind %s from client %s",
                dk, conn.addr,
            )

    # -----------------------------------------------------------------------
    # TNC -> Clients (KISS frame received from serial)
    # -----------------------------------------------------------------------

    def _on_kiss_frame(self, cmd: int, payload: bytes) -> None:
        """Called by KISSSerial for each received KISS data frame.

        Translates the received AX.25 frame bytes into an appropriate AGWPE
        frame type and forwards it to all interested clients.

        Args:
            cmd: KISS command byte (low nibble; 0x00 for data).
            payload: Destuffed AX.25 frame bytes.
        """
        if (cmd & 0x0F) != CMD_DATA:
            # Non-data KISS frame (TNC parameter notification) -- ignore
            return

        if len(payload) < 15:
            logger.debug("AGWSerial: KISS frame too short (%d bytes)", len(payload))
            return

        # Parse the AX.25 address field
        dest, src, digipeaters, hdr_len = _parse_ax25_addresses(payload)
        if dest is None:
            return

        # Extract control and PID bytes if present
        ctrl = payload[hdr_len] if hdr_len < len(payload) else 0
        is_ui = (ctrl == 0x03)
        pid = payload[hdr_len + 1] if (is_ui and hdr_len + 1 < len(payload)) else 0xF0
        info = payload[hdr_len + 2:] if (is_ui and hdr_len + 2 <= len(payload)) else b""

        logger.debug(
            "AGWSerial: KISS rx: %s -> %s ctrl=0x%02X pid=0x%02X info_len=%d",
            src, dest, ctrl, pid, len(info),
        )

        self._forward_to_clients(
            payload=payload,
            dest=dest,
            src=src,
            is_ui=is_ui,
            pid=pid,
            info=info,
        )

    def _forward_to_clients(
        self,
        payload: bytes,
        dest: str,
        src: str,
        is_ui: bool,
        pid: int,
        info: bytes,
    ) -> None:
        """Forward a received AX.25 frame to all interested AGWPE clients.

        Args:
            payload: Complete raw AX.25 frame bytes.
            dest: Decoded destination callsign.
            src: Decoded source callsign.
            is_ui: True if this is a UI (unproto) frame.
            pid: Protocol ID byte.
            info: Information field bytes.
        """
        dead = []
        with self._clients_lock:
            clients = list(self._clients.values())

        for conn in clients:
            if conn.monitoring:
                # Send raw frame as 'K' (monitoring)
                ok = conn.send_frame(
                    data_kind=KIND_RAW_MON,
                    call_from=src,
                    call_to=dest,
                    data=payload,
                )
                if not ok:
                    dead.append(conn)
                    continue

            if is_ui:
                # Also send as a 'D' frame so the client can process the data
                agw_data = bytes([pid]) + info
                conn.send_frame(
                    data_kind=KIND_UNPROTO_DATA,
                    call_from=src,
                    call_to=dest,
                    data=agw_data,
                )

        if dead:
            with self._clients_lock:
                for conn in dead:
                    self._clients.pop(conn.sock, None)
                    conn.close()

    # -----------------------------------------------------------------------
    # Standard response helpers
    # -----------------------------------------------------------------------

    def _send_version(self, conn: _AGWClientConn) -> None:
        """Send an AGWPE 'R' version response to a client."""
        data = _VERSION_STRING.encode("ascii") + b"\x00"
        conn.send_frame(data_kind=KIND_VERSION, data=data)

    def _send_port_info(self, conn: _AGWClientConn) -> None:
        """Send an AGWPE 'G' port capabilities response to a client.

        Advertises one port corresponding to our serial KISS TNC.
        The AGWPE 'G' response format is a NUL-terminated ASCII string
        listing each port's info, separated by semicolons.

        Format: "Portcount;PortName[;...]\x00"
        """
        port_info = f"1;{self.port_name} {self.serial_port}\x00"
        conn.send_frame(
            data_kind=KIND_PORT_CAPS,
            data=port_info.encode("ascii"),
        )
