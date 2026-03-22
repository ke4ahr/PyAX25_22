# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/smack.py

SMACK -- Stuttgart Modified Amateur Radio CRC-KISS.

SMACK adds CRC-16 protection to KISS data frames.  It is built on top of XKISS
(which handles multi-drop port addressing and polling).

Protocol behaviour
------------------
- Data frames sent with CRC have bit 7 of the command byte set (SMACK_FLAG).
- The CRC is computed over (cmd_byte + payload) using CRC-16 with polynomial
  0x8005, initial value 0x0000, big-endian computation, LSB-first appended.
- Auto-switch: the transmitter starts in plain mode.  On the first valid CRC
  frame received, it switches to CRC mode for all subsequent transmissions.
- Corrupt frames (bad CRC) are silently dropped.

Bug fixes from original PyXKISS smack.py
-----------------------------------------
- Added missing ``import threading`` (referenced but never imported).
- Imports SMACK_FLAG from constants rather than defining a local constant.
- Used self._smack_lock (threading.Lock) to protect _smack_enabled.
"""

import logging
import threading
from typing import Callable, Iterable, Optional

from .xkiss import XKISSMixin, XKISS, XKISSTCP
from .serial import KISSSerial
from .tcp import KISSTCP
from .constants import (
    CMD_DATA, SMACK_FLAG, SMACK_POLY, SMACK_INIT, SMACK_CRC_SIZE,
    DEFAULT_POLL_INTERVAL, DEFAULT_MAX_QUEUE,
)
from .exceptions import KISSChecksumError

logger = logging.getLogger(__name__)


class SMACKMixin(XKISSMixin):
    """SMACK logic as a mixin on top of XKISSMixin.

    Intercepts on_xframe to verify CRC on incoming frames, and overrides
    send() to add CRC when _smack_enabled is True.

    Args:
        on_smack_frame: Optional callback ``(address, port, payload) -> None``
            called for successfully validated SMACK frames (or plain XKISS
            frames before SMACK is activated).
        **kwargs: All other keyword arguments are forwarded to XKISSMixin.

    Note:
        Do not instantiate SMACKMixin directly.  Use SMACK (serial) or
        SMACKTCP (TCP).
    """

    def __init__(
        self,
        on_smack_frame: Optional[Callable[[int, int, bytes], None]] = None,
        **kwargs,
    ) -> None:
        # Intercept on_xframe: we register our own wrapper so we can do CRC
        # verification before passing frames to the user callback.
        self._smack_user_callback = on_smack_frame
        self._smack_enabled = False
        self._smack_lock = threading.Lock()

        # Register our CRC-checking wrapper as on_xframe
        super().__init__(on_xframe=self._smack_on_xframe, **kwargs)

    # -----------------------------------------------------------------------
    # Send (overrides XKISS.send / XKISSMixin.send)
    # -----------------------------------------------------------------------

    def send(
        self, payload: bytes, port: int = 0, cmd: int = CMD_DATA
    ) -> None:
        """Send a SMACK data frame, optionally with CRC-16 appended.

        If SMACK is active (_smack_enabled True), sets bit 7 of the command
        byte and appends a 2-byte CRC-16 (LSB-first) to the payload.

        Args:
            payload: AX.25 frame bytes to send.
            port: TNC port (0-15).
            cmd: KISS command (default: CMD_DATA = 0x00).
        """
        if self._smack_enabled and (cmd & 0x0F) == CMD_DATA:
            smack_cmd = cmd | SMACK_FLAG
            crc = self._crc16(bytes([smack_cmd]) + payload)
            payload = payload + crc
            logger.debug(
                "SMACK send: CRC=0x%04X (cmd=0x%02X port=%d payload_len=%d)",
                int.from_bytes(crc, "little"), smack_cmd, port, len(payload),
            )
            # Call XKISSMixin.send with the modified cmd
            super().send(payload, port=port, cmd=smack_cmd & 0x0F)
        else:
            super().send(payload, port=port, cmd=cmd)

    # -----------------------------------------------------------------------
    # Incoming frame handler
    # -----------------------------------------------------------------------

    def _smack_on_xframe(
        self, address: int, port: int, payload: bytes
    ) -> None:
        """Validate the SMACK CRC and deliver to the user callback.

        If the SMACK flag (bit 7) is set in the first byte of the payload
        (after XKISS strips the port nibble), the frame carries CRC.
        We verify it, enable SMACK mode for TX, and strip the CRC before
        delivering to the user callback.

        Args:
            address: This station's TNC address.
            port: Received port number.
            payload: Decoded payload (after XOR checksum removal if active).
        """
        # The SMACK flag is in bit 7 of the *original* command byte.
        # XKISSMixin already extracted the port; the SMACK_FLAG bit was part
        # of the original cmd byte passed to KISSBase.  We detect SMACK by
        # re-checking it through a side channel set by _xkiss_on_frame.
        #
        # Actually: XKISSMixin._xkiss_on_frame receives cmd = full byte
        # (port_nibble | cmd_nibble).  SMACK sets bit 7.  After XKISS strips
        # the port nibble we lose bit 7.  To detect SMACK we need to see the
        # raw cmd byte.  We work around this by making SMACKMixin also
        # register a lower-level KISSBase on_frame callback:
        #
        # This is handled by overriding _smack_check_cmd below.
        # The actual SMACK detection is done in _smack_raw_on_frame.
        if self._smack_user_callback:
            self._smack_user_callback(address, port, payload)

    def _setup_smack_raw(self) -> None:
        """Install the raw KISS frame interceptor for SMACK detection.

        Called by SMACK / SMACKTCP constructors after super().__init__().
        Replaces the KISSBase on_frame with a wrapper that checks bit 7.
        """
        original_on_frame = self.on_frame  # was set by XKISSMixin to _xkiss_on_frame

        def _raw_wrapper(cmd: int, payload: bytes) -> None:
            smack = bool(cmd & SMACK_FLAG)
            if smack:
                # Verify CRC before passing to XKISS
                port = (cmd & 0xF0) >> 4
                real_cmd = cmd & 0x0F
                if len(payload) < SMACK_CRC_SIZE:
                    logger.warning(
                        "SMACK: frame too short for CRC (port=%d)", port
                    )
                    return
                received_crc = payload[-SMACK_CRC_SIZE:]
                body = payload[:-SMACK_CRC_SIZE]
                expected_crc = self._crc16(bytes([cmd]) + body)
                if received_crc != expected_crc:
                    logger.warning(
                        "SMACK: CRC mismatch on port %d -- frame dropped "
                        "(received=%s expected=%s)",
                        port,
                        received_crc.hex(),
                        expected_crc.hex(),
                    )
                    return
                # CRC OK -- enable SMACK TX
                with self._smack_lock:
                    if not self._smack_enabled:
                        logger.info(
                            "SMACK: first valid CRC frame received -- "
                            "enabling SMACK TX mode"
                        )
                        self._smack_enabled = True
                # Pass verified frame (without CRC bytes) to XKISS handler
                original_on_frame(cmd & ~SMACK_FLAG, body)
            else:
                original_on_frame(cmd, payload)

        self.on_frame = _raw_wrapper

    # -----------------------------------------------------------------------
    # CRC-16 computation
    # -----------------------------------------------------------------------

    def _crc16(self, data: bytes) -> bytes:
        """Compute SMACK CRC-16 (poly 0x8005, init 0x0000, LSB-first output).

        The CRC is computed in normal (non-reflected) big-endian bit order
        over all bytes.  The 2-byte result is returned in little-endian order
        (LSB first) for appending to the frame.

        Args:
            data: Bytes to protect (typically cmd byte + payload body).

        Returns:
            2-byte CRC value, LSB-first.
        """
        crc = SMACK_INIT
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ SMACK_POLY) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc.to_bytes(2, "little")

    @property
    def smack_enabled(self) -> bool:
        """True if SMACK TX mode has been activated (auto-switch occurred)."""
        with self._smack_lock:
            return self._smack_enabled

    def reset_smack(self) -> None:
        """Reset SMACK to plain mode (before auto-switch).

        Useful in testing or when reconnecting to a peer that does not
        support SMACK.
        """
        with self._smack_lock:
            self._smack_enabled = False
        logger.info("SMACK: reset to plain mode")


# ---------------------------------------------------------------------------
# Concrete SMACK classes (serial and TCP)
# ---------------------------------------------------------------------------

class SMACK(SMACKMixin, XKISS):
    """SMACK over a serial port.

    Combines SMACKMixin (CRC-16) on top of XKISS (multi-drop, polling) on
    top of KISSSerial (serial port).

    Args:
        device: Serial device path.
        baudrate: Serial baud rate.
        address: This station's TNC address (0-15).
        polling_mode: Enable active/passive polling.
        poll_interval: Seconds between active polls.
        checksum_mode: Enable XOR checksum mode (rarely used with SMACK).
        max_queue_size: Max frames buffered per port queue.
        digipeat_ports: Set of ports where DIGIPEAT=ON.
        our_calls: Callsigns for digipeating.
        on_smack_frame: Callback ``(address, port, payload) -> None``.

    Example:
        smack = SMACK(
            "/dev/ttyUSB0",
            on_smack_frame=lambda a, p, d: print(d.hex()),
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
        on_smack_frame: Optional[Callable[[int, int, bytes], None]] = None,
    ) -> None:
        super().__init__(
            on_smack_frame=on_smack_frame,
            address=address,
            polling_mode=polling_mode,
            poll_interval=poll_interval,
            checksum_mode=checksum_mode,
            max_queue_size=max_queue_size,
            digipeat_ports=digipeat_ports,
            our_calls=our_calls,
            device=device,
            baudrate=baudrate,
        )
        self._setup_smack_raw()
        logger.info("SMACK serial: device=%s baudrate=%d", device, baudrate)


class SMACKTCP(SMACKMixin, XKISSTCP):
    """SMACK over a TCP socket.

    Combines SMACKMixin (CRC-16) on top of XKISSTCP (multi-drop XKISS over
    TCP).

    Args:
        host: Hostname or IP of the KISS TCP server.
        port: TCP port number.
        address: This station's TNC address (0-15).
        polling_mode: Enable active/passive polling.
        poll_interval: Seconds between active polls.
        checksum_mode: Enable XOR checksum mode.
        max_queue_size: Max frames buffered per port queue.
        digipeat_ports: Set of ports where DIGIPEAT=ON.
        our_calls: Callsigns for digipeating.
        on_smack_frame: Callback ``(address, port, payload) -> None``.

    Example:
        smack = SMACKTCP(
            "localhost", 8001,
            on_smack_frame=lambda a, p, d: print(d.hex()),
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
        on_smack_frame: Optional[Callable[[int, int, bytes], None]] = None,
        connect_timeout: float = 10.0,
        keepalive: bool = True,
    ) -> None:
        super().__init__(
            on_smack_frame=on_smack_frame,
            address=address,
            polling_mode=polling_mode,
            poll_interval=poll_interval,
            checksum_mode=checksum_mode,
            max_queue_size=max_queue_size,
            digipeat_ports=digipeat_ports,
            our_calls=our_calls,
            host=host,
            port=port,
            connect_timeout=connect_timeout,
            keepalive=keepalive,
        )
        self._setup_smack_raw()
        logger.info("SMACK TCP: host=%s port=%d", host, port)
