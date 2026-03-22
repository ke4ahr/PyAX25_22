# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/xkiss.py

XKISS -- Extended KISS (XKISS / BPQKISS / Multi-Drop KISS).

Extends KISSBase with:
- Multi-drop addressing: high nibble of command byte = TNC port (0-15)
- Active polling: background thread sends 0x0E POLL frames to the TNC
- Passive polling: buffers incoming data and flushes it when a poll arrives
- Optional XOR checksum (Kantronics / BPQ32 CHECKSUM mode)
- Per-port receive queues with configurable maximum size
- DIGIPEAT per port: optionally re-transmits frames whose digipeater
  addresses have not yet been satisfied

Design
------
XKISSMixin provides the XKISS logic without binding to a specific transport.
XKISS and XKISSTCP are concrete ready-to-use classes that combine XKISSMixin
with KISSSerial (serial) and KISSTCP (TCP) respectively.

References
----------
- G8BPQ Multi-Drop KISS documentation
- Karl Medcalf WK5M XKISS notes
- BPQ32 source code
"""

import logging
import threading
import time
from collections import deque
from typing import Callable, Dict, Iterable, Optional, Set

from .base import KISSBase
from .serial import KISSSerial
from .tcp import KISSTCP
from .constants import (
    CMD_DATA, CMD_POLL, PORT_MASK, CMD_MASK, DEFAULT_POLL_INTERVAL,
    DEFAULT_MAX_QUEUE,
)
from .exceptions import KISSChecksumError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional AX.25 digipeating helper
# ---------------------------------------------------------------------------

def _digipeat_frame(payload: bytes, our_calls: Set[str]) -> Optional[bytes]:
    """Attempt to digipeat (relay) an AX.25 frame.

    Parses the AX.25 address field to find the first un-repeated digipeater
    address.  If that address matches one of our callsigns (or is a generic
    alias like RELAY, WIDE1-1), sets its H-bit and returns the modified frame.

    Args:
        payload: Raw AX.25 frame bytes (starting with destination address).
        our_calls: Set of callsign strings (upper-case, no SSID) that this
            station will answer to for digipeating.

    Returns:
        Modified frame bytes with the next digipeater H-bit set, or None if
        this frame does not need digipeating.
    """
    # AX.25 address field: 7 bytes per address; at minimum dest (7) + src (7)
    if len(payload) < 14:
        return None

    # Walk address fields.  Each address is 7 bytes, last byte bit 0 = end flag.
    offset = 0
    while offset + 7 <= len(payload):
        is_last = bool(payload[offset + 6] & 0x01)
        if offset >= 14:
            # This is a digipeater address
            # Bit 7 of octet 6 is the H-bit (has-been-repeated)
            h_bit = bool(payload[offset + 6] & 0x80)
            if not h_bit:
                # First un-repeated digipeater -- check if it's us
                call_bytes = payload[offset:offset + 6]
                callsign = "".join(
                    chr(b >> 1) for b in call_bytes
                ).rstrip()
                ssid = (payload[offset + 6] >> 1) & 0x0F

                # Generic aliases always get repeated regardless of callsign
                generic_aliases = {
                    "RELAY", "WIDE", "TRACE", "GATE",
                    "WIDE1", "WIDE2", "WIDE3", "WIDE4",
                    "WIDE1-1", "WIDE2-1", "WIDE2-2",
                }
                call_with_ssid = callsign if ssid == 0 else f"{callsign}-{ssid}"

                if callsign in our_calls or call_with_ssid in our_calls \
                        or callsign in generic_aliases:
                    # Set the H-bit
                    frame = bytearray(payload)
                    frame[offset + 6] |= 0x80
                    logger.debug(
                        "digipeat: relaying frame via %s", call_with_ssid
                    )
                    return bytes(frame)
                else:
                    return None  # Not for us
        if is_last:
            break
        offset += 7

    return None  # No un-repeated digipeaters or address field too short


# ---------------------------------------------------------------------------
# XKISSMixin
# ---------------------------------------------------------------------------

class XKISSMixin(KISSBase):
    """XKISS logic as a mixin.

    Adds multi-drop port addressing, optional XOR checksum, active/passive
    polling, and per-port DIGIPEAT to any KISSBase transport subclass.

    This class is not meant to be instantiated directly.  Use XKISS
    (serial) or XKISSTCP (TCP) instead.

    Args:
        address: This station's TNC address (0-15) in a multi-drop topology.
        polling_mode: If True, enables both active and passive polling.
            Active polling: a background thread sends CMD_POLL frames
            periodically so the TNC releases buffered data.
            Passive polling: incoming data frames are queued and only
            forwarded to on_xframe after a poll is received.
        poll_interval: Seconds between active poll frames (default: 0.1 s).
        checksum_mode: If True, appends/verifies a 1-byte XOR checksum to
            all data frames (Kantronics extension).
        max_queue_size: Maximum number of frames buffered per RX port queue
            in passive polling mode.
        digipeat_ports: Set of port numbers (0-15) on which received frames
            should be automatically digipeated if the frame has an
            un-satisfied digipeater address matching our_calls.
        our_calls: Set of callsign strings (upper-case, SSID optional) that
            identify this station for digipeating purposes.
        on_xframe: Optional callback ``(address, port, payload) -> None``
            called for each received data frame after checksum verification
            and queue management.
        on_frame: Passed to KISSBase (used internally; overridden by XKISS).
    """

    def __init__(
        self,
        address: int = 0,
        polling_mode: bool = False,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        checksum_mode: bool = False,
        max_queue_size: int = DEFAULT_MAX_QUEUE,
        digipeat_ports: Optional[Iterable[int]] = None,
        our_calls: Optional[Iterable[str]] = None,
        on_xframe: Optional[Callable[[int, int, bytes], None]] = None,
        **kwargs,
    ) -> None:
        if not 0 <= address <= 15:
            raise ValueError("address must be 0-15")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")

        # Install our internal frame handler as the KISSBase callback
        super().__init__(on_frame=self._xkiss_on_frame, **kwargs)

        self.address = address & 0x0F
        self.polling_mode = polling_mode
        self.poll_interval = poll_interval
        self.checksum_mode = checksum_mode
        self.max_queue_size = max_queue_size
        self.on_xframe = on_xframe

        self.digipeat_ports: Set[int] = (
            set(digipeat_ports) if digipeat_ports else set()
        )
        self.our_calls: Set[str] = (
            {c.upper() for c in our_calls} if our_calls else set()
        )

        # Per-port RX queues (passive polling)
        self._rx_queues: Dict[int, deque] = {
            p: deque(maxlen=max_queue_size) for p in range(16)
        }

        self._poll_thread: Optional[threading.Thread] = None
        if self.polling_mode:
            self._start_active_poller()

        logger.info(
            "XKISS: address=%d polling=%s checksum=%s poll_interval=%.2fs "
            "digipeat_ports=%s",
            self.address, polling_mode, checksum_mode, poll_interval,
            sorted(self.digipeat_ports),
        )

    # -----------------------------------------------------------------------
    # Send (overrides KISSBase.send)
    # -----------------------------------------------------------------------

    def send(
        self, payload: bytes, port: int = 0, cmd: int = CMD_DATA
    ) -> None:
        """Send an XKISS data frame on the specified port.

        Builds the full command byte as ``(port << 4) | cmd``, optionally
        appends an XOR checksum byte, then calls write() with the fully
        KISS-escaped frame.

        This method fixes a bug in the original PyXKISS XKISS.send() which
        omitted KISS escaping of payload bytes.

        Args:
            payload: AX.25 frame bytes (or other data) to transmit.
            port: TNC port number, 0-15 (high nibble of command byte).
            cmd: KISS command (low nibble, default: CMD_DATA = 0x00).

        Raises:
            KISSTransportError: If the write fails.
        """
        full_cmd = ((port & 0x0F) << 4) | (cmd & 0x0F)

        if self.checksum_mode:
            checksum = full_cmd
            for b in payload:
                checksum ^= b
            payload = payload + bytes([checksum & 0xFF])
            logger.debug(
                "XKISS send: XOR checksum=0x%02X appended", checksum & 0xFF
            )

        frame = self._stuff(payload, full_cmd)
        logger.debug(
            "XKISS send: port=%d cmd=0x%02X payload_len=%d",
            port, full_cmd, len(payload),
        )
        self.write(frame)

    def poll(self, port: int = 0) -> None:
        """Send a manual POLL frame to request buffered data from the TNC.

        Args:
            port: TNC port to poll (default: 0).
        """
        self.send(b"", port=port, cmd=CMD_POLL)
        logger.debug("XKISS: sent POLL on port %d", port)

    # -----------------------------------------------------------------------
    # Internal frame handler (registered with KISSBase)
    # -----------------------------------------------------------------------

    def _xkiss_on_frame(self, cmd: int, payload: bytes) -> None:
        """Handle a decoded KISS frame received from KISSBase.

        Steps:
        1. Verify optional XOR checksum and strip it.
        2. Decode port (high nibble) and real command (low nibble).
        3. If CMD_POLL: flush the port's receive queue.
        4. If CMD_DATA + polling_mode: queue the frame.
        5. Otherwise: attempt digipeating, then call on_xframe.

        Args:
            cmd: Raw command byte from the KISS frame.
            payload: Destuffed frame body.
        """
        try:
            # 1. XOR checksum verification
            if self.checksum_mode:
                if len(payload) < 1:
                    logger.warning("XKISS: frame too short for checksum")
                    return
                expected = 0
                for b in bytes([cmd]) + payload[:-1]:
                    expected ^= b
                received = payload[-1]
                if received != (expected & 0xFF):
                    raise KISSChecksumError(
                        f"XOR checksum mismatch: "
                        f"expected=0x{expected & 0xFF:02X} "
                        f"received=0x{received:02X}"
                    )
                payload = payload[:-1]

            # 2. Decode port and real command
            port = (cmd & PORT_MASK) >> 4
            real_cmd = cmd & CMD_MASK

            logger.debug(
                "XKISS rx: port=%d real_cmd=0x%02X payload_len=%d",
                port, real_cmd, len(payload),
            )

            # 3. Passive polling: POLL received from TNC
            if real_cmd == CMD_POLL and self.polling_mode:
                logger.info("XKISS: POLL on port %d -- flushing queue", port)
                self._flush_queue(port)
                return

            # 4. Passive polling: buffer data frames
            if real_cmd == CMD_DATA and self.polling_mode:
                q = self._rx_queues[port]
                if len(q) >= self.max_queue_size:
                    q.popleft()
                    logger.warning(
                        "XKISS: RX queue overflow on port %d -- oldest frame dropped",
                        port,
                    )
                q.append(payload)
                logger.debug(
                    "XKISS: queued frame on port %d (queue depth=%d)",
                    port, len(q),
                )
                return

            # 5. DIGIPEAT (if enabled for this port)
            if real_cmd == CMD_DATA and port in self.digipeat_ports:
                relay = _digipeat_frame(payload, self.our_calls)
                if relay is not None:
                    logger.info(
                        "XKISS: digipeating %d-byte frame on port %d",
                        len(relay), port,
                    )
                    self.send(relay, port=port, cmd=CMD_DATA)

            # 6. Deliver to user callback
            if self.on_xframe:
                self.on_xframe(self.address, port, payload)

        except KISSChecksumError as exc:
            logger.error("XKISS checksum error: %s", exc)
        except Exception:
            logger.exception(
                "XKISS: unexpected error processing frame "
                "(cmd=0x%02X payload_len=%d)",
                cmd, len(payload),
            )

    # -----------------------------------------------------------------------
    # Passive polling helpers
    # -----------------------------------------------------------------------

    def _flush_queue(self, port: int) -> None:
        """Flush all buffered frames for a port after a POLL is received.

        Each buffered frame is re-transmitted via send(), then delivered to
        on_xframe.

        Args:
            port: The port whose receive queue should be flushed.
        """
        q = self._rx_queues.get(port, deque())
        flushed = 0
        while q:
            payload = q.popleft()
            try:
                self.send(payload, port=port, cmd=CMD_DATA)
                if self.on_xframe:
                    self.on_xframe(self.address, port, payload)
                flushed += 1
            except Exception:
                logger.exception(
                    "XKISS: error flushing queued frame on port %d", port
                )
                break
        if flushed:
            logger.info(
                "XKISS: flushed %d frames from port %d queue", flushed, port
            )

    # -----------------------------------------------------------------------
    # Active polling
    # -----------------------------------------------------------------------

    def _start_active_poller(self) -> None:
        """Start the background thread that sends periodic POLL frames."""
        def _poller():
            while True:
                time.sleep(self.poll_interval)
                try:
                    self.poll()
                except Exception:
                    logger.exception("XKISS active poller: send failed")

        self._poll_thread = threading.Thread(
            target=_poller,
            name="XKISS-ActivePoller",
            daemon=True,
        )
        self._poll_thread.start()
        logger.debug(
            "XKISS: active poller started (interval=%.2f s)",
            self.poll_interval,
        )

    # -----------------------------------------------------------------------
    # DIGIPEAT management
    # -----------------------------------------------------------------------

    def enable_digipeat(self, port: int) -> None:
        """Enable digipeating on the given port.

        Args:
            port: TNC port number (0-15).

        Raises:
            ValueError: If port is out of range.
        """
        if not 0 <= port <= 15:
            raise ValueError(f"port must be 0-15, got {port}")
        self.digipeat_ports.add(port)
        logger.info("XKISS: DIGIPEAT enabled on port %d", port)

    def disable_digipeat(self, port: int) -> None:
        """Disable digipeating on the given port.

        Args:
            port: TNC port number (0-15).

        Raises:
            ValueError: If port is out of range.
        """
        if not 0 <= port <= 15:
            raise ValueError(f"port must be 0-15, got {port}")
        self.digipeat_ports.discard(port)
        logger.info("XKISS: DIGIPEAT disabled on port %d", port)

    def set_digipeat(self, port: int, enabled: bool) -> None:
        """Set DIGIPEAT=ON or DIGIPEAT=OFF for a specific port.

        Args:
            port: TNC port number (0-15).
            enabled: True to enable digipeating, False to disable.
        """
        if enabled:
            self.enable_digipeat(port)
        else:
            self.disable_digipeat(port)

    def get_digipeat(self, port: int) -> bool:
        """Return the current DIGIPEAT setting for a port.

        Args:
            port: TNC port number (0-15).

        Returns:
            True if digipeating is enabled on this port, False otherwise.
        """
        return port in self.digipeat_ports

    # -----------------------------------------------------------------------
    # XOR checksum
    # -----------------------------------------------------------------------

    def _compute_xor(self, data: bytes) -> int:
        """Compute a 1-byte XOR checksum over data bytes.

        Args:
            data: Bytes to checksum (typically cmd byte + payload).

        Returns:
            XOR of all bytes, masked to 0-255.
        """
        result = 0
        for b in data:
            result ^= b
        return result & 0xFF


# ---------------------------------------------------------------------------
# Concrete XKISS classes (serial and TCP)
# ---------------------------------------------------------------------------

class XKISS(XKISSMixin, KISSSerial):
    """XKISS over a serial port (most common use case).

    Combines XKISSMixin logic with KISSSerial transport.

    Args:
        device: Serial device path (e.g., ``/dev/ttyUSB0``).
        baudrate: Serial baud rate.
        address: This station's TNC address (0-15).
        polling_mode: Enable active/passive polling.
        poll_interval: Seconds between active polls.
        checksum_mode: Enable XOR checksum mode.
        max_queue_size: Max frames buffered per port queue.
        digipeat_ports: Set of ports where DIGIPEAT=ON.
        our_calls: Callsigns this station answers to for digipeating.
        on_xframe: Callback ``(address, port, payload) -> None``.

    Example:
        xk = XKISS(
            "/dev/ttyUSB0",
            address=0,
            on_xframe=lambda addr, port, data: print(data.hex()),
        )
    """

    def __init__(
        self,
        device: str,
        baudrate: int = 9600,
        address: int = 0,
        polling_mode: bool = False,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        checksum_mode: bool = False,
        max_queue_size: int = DEFAULT_MAX_QUEUE,
        digipeat_ports: Optional[Iterable[int]] = None,
        our_calls: Optional[Iterable[str]] = None,
        on_xframe: Optional[Callable[[int, int, bytes], None]] = None,
    ) -> None:
        super().__init__(
            # XKISSMixin params
            address=address,
            polling_mode=polling_mode,
            poll_interval=poll_interval,
            checksum_mode=checksum_mode,
            max_queue_size=max_queue_size,
            digipeat_ports=digipeat_ports,
            our_calls=our_calls,
            on_xframe=on_xframe,
            # KISSSerial params
            device=device,
            baudrate=baudrate,
            # KISSBase.on_frame is set internally by XKISSMixin
        )


class XKISSTCP(XKISSMixin, KISSTCP):
    """XKISS over a TCP socket.

    Combines XKISSMixin logic with KISSTCP transport.

    Args:
        host: Hostname or IP of the KISS TCP server.
        port: TCP port number.
        address: This station's TNC address (0-15).
        polling_mode: Enable active/passive polling.
        poll_interval: Seconds between active polls.
        checksum_mode: Enable XOR checksum mode.
        max_queue_size: Max frames buffered per port queue.
        digipeat_ports: Set of ports where DIGIPEAT=ON.
        our_calls: Callsigns this station answers to for digipeating.
        on_xframe: Callback ``(address, port, payload) -> None``.

    Example:
        xk = XKISSTCP(
            "localhost", 8001,
            on_xframe=lambda addr, port, data: print(data.hex()),
        )
    """

    def __init__(
        self,
        host: str,
        port: int,
        address: int = 0,
        polling_mode: bool = False,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        checksum_mode: bool = False,
        max_queue_size: int = DEFAULT_MAX_QUEUE,
        digipeat_ports: Optional[Iterable[int]] = None,
        our_calls: Optional[Iterable[str]] = None,
        on_xframe: Optional[Callable[[int, int, bytes], None]] = None,
        connect_timeout: float = 10.0,
        keepalive: bool = True,
    ) -> None:
        super().__init__(
            # XKISSMixin params
            address=address,
            polling_mode=polling_mode,
            poll_interval=poll_interval,
            checksum_mode=checksum_mode,
            max_queue_size=max_queue_size,
            digipeat_ports=digipeat_ports,
            our_calls=our_calls,
            on_xframe=on_xframe,
            # KISSTCP params
            host=host,
            port=port,
            connect_timeout=connect_timeout,
            keepalive=keepalive,
        )
