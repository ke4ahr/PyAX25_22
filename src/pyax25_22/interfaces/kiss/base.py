# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/base.py

KISSBase -- transport-agnostic KISS framing layer.

Handles:
- KISS frame construction (stuffing: FEND/FESC escaping)
- KISS frame deconstruction (destuffing: FESC removal)
- Per-byte receive state machine (_process_byte)
- Dispatching received frames to the on_frame callback

Does NOT handle I/O.  Concrete subclasses (KISSSerial, KISSTCP) implement
the write() method and start a reader thread that feeds bytes into
_process_byte().

Design note:
    KISSBase uses the Template Method pattern.  send() builds a frame and
    calls self.write(frame).  write() is declared as abstract here and
    implemented by each concrete transport subclass.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

from .constants import FEND, FESC, TFEND, TFESC, CMD_DATA, CMD_EXIT
from .exceptions import KISSFrameError

logger = logging.getLogger(__name__)

# All valid standard KISS command codes (low nibble values)
_VALID_CMDS = frozenset([
    CMD_DATA,   # 0x00 -- data frame
    0x01,       # TXDELAY
    0x02,       # PERSIST
    0x03,       # SLOTTIME
    0x04,       # TXTAIL
    0x05,       # FULLDUP
    0x06,       # HARDWARE
    CMD_EXIT,   # 0xFF -- exit KISS mode
])


class KISSBase(ABC):
    """Abstract base class for all KISS protocol implementations.

    Think of KISSBase as the "brain" of KISS -- it knows how to wrap data in
    KISS frames and how to unwrap them, but it has no idea whether the frames
    travel over a serial wire, a TCP socket, or anything else.  The concrete
    subclass supplies the I/O plumbing through write().

    Args:
        on_frame: Optional callback called with (cmd, payload) whenever a
            complete, destuffed KISS frame is received.  cmd is the raw
            command byte (including port nibble for XKISS).  payload is
            the frame data with all FESC sequences removed.
    """

    def __init__(
        self,
        on_frame: Optional[Callable[[int, bytes], None]] = None,
        **kwargs,
    ) -> None:
        """Initialize KISSBase.

        Args:
            on_frame: Callback(cmd: int, payload: bytes) for received frames.
                If None, received frames are logged at DEBUG level and discarded.
            **kwargs: Forwarded to super().__init__ for cooperative multiple
                inheritance compatibility.
        """
        super().__init__(**kwargs)
        self.on_frame = on_frame
        self._buf: bytearray = bytearray()   # accumulates bytes between FENDs

    # -----------------------------------------------------------------------
    # Abstract interface (subclasses must implement)
    # -----------------------------------------------------------------------

    @abstractmethod
    def write(self, data: bytes) -> None:
        """Write raw bytes to the underlying transport.

        This is the only I/O method required by KISSBase.  Concrete
        subclasses send the bytes over serial, TCP, or any other medium.

        Args:
            data: The fully-encoded KISS frame bytes to transmit (including
                FEND delimiters and escaping).

        Raises:
            KISSTransportError: (or a subclass) if the write fails.
        """

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def send(self, payload: bytes, cmd: int = CMD_DATA) -> None:
        """Encode payload as a KISS frame and transmit it.

        The command byte cmd is placed at the start of the frame.  For
        standard KISS, cmd is one of CMD_DATA .. CMD_EXIT.  For XKISS, the
        high nibble encodes the port number -- use the XKISS.send() override.

        The payload bytes are KISS-escaped (FEND -> FESC TFEND, FESC -> FESC
        TFESC) before transmission.

        Args:
            payload: Raw bytes to send inside the KISS frame.
            cmd: KISS command byte (default: 0x00 = CMD_DATA).

        Raises:
            ValueError: If cmd is not a recognised KISS command.
            KISSTransportError: If the write fails.

        Example:
            kiss.send(ax25_frame_bytes)
            kiss.send(bytes([200]), cmd=CMD_TXDELAY)
        """
        low_nibble = cmd & 0x0F
        # Allow XKISS port bits in high nibble; validate low nibble only
        if low_nibble not in _VALID_CMDS and cmd != CMD_EXIT:
            raise ValueError(
                f"Invalid KISS command low nibble: 0x{low_nibble:02X}"
            )

        frame = self._stuff(payload, cmd)
        logger.debug(
            "KISS send: cmd=0x%02X payload_len=%d frame_len=%d",
            cmd, len(payload), len(frame),
        )
        self.write(frame)

    def close(self) -> None:
        """Close the transport and stop any reader threads.

        Subclasses should override this to close the serial port / socket and
        signal any background threads to stop.  Always call super().close()
        at the end of the override so cooperative inheritance works.
        """
        logger.info("%s closed", type(self).__name__)

    # -----------------------------------------------------------------------
    # Frame construction (stuffing / escaping)
    # -----------------------------------------------------------------------

    def _stuff(self, payload: bytes, cmd: int) -> bytes:
        """Build a complete KISS frame from a payload and command byte.

        Inserts FEND markers at the start and end.  Any FEND (0xC0) or FESC
        (0xDB) byte inside the payload is replaced with a two-byte escape
        sequence so the receiver can find frame boundaries unambiguously.

        Args:
            payload: Raw data bytes (not yet escaped).
            cmd: Command byte to place at the start of the frame body.

        Returns:
            Fully framed bytes: FEND + cmd + escaped_payload + FEND.
        """
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
    # Frame disassembly (destuffing / receiving)
    # -----------------------------------------------------------------------

    def _process_byte(self, byte: int) -> None:
        """Feed one received byte into the KISS frame assembler.

        This is the per-byte state machine.  Call it for each byte received
        from the transport.  When a complete frame is detected (FEND seen
        after at least one body byte), the frame is destuffed and dispatched
        to on_frame.

        Args:
            byte: A single byte value (0-255) from the transport.
        """
        if byte == FEND:
            if len(self._buf) >= 1:
                cmd = self._buf[0]
                try:
                    payload = self._destuff(self._buf[1:])
                except KISSFrameError as exc:
                    logger.warning("KISS destuff error: %s", exc)
                else:
                    self._dispatch(cmd, payload)
            self._buf = bytearray()
        else:
            self._buf.append(byte)

    def _destuff(self, data: bytearray) -> bytes:
        """Remove KISS transparency escaping from frame body bytes.

        After a FESC byte, the next byte is interpreted as follows:
            TFEND (0xDC) -> FEND (0xC0)
            TFESC (0xDD) -> FESC (0xDB)
            anything else -> logged as a warning, byte kept as-is

        Args:
            data: Frame body bytes (everything after the command byte, before
                the closing FEND), still containing escape sequences.

        Returns:
            Destuffed payload as bytes.

        Raises:
            KISSFrameError: If a FESC is found at the very end of data with
                no following byte (truncated escape sequence).
        """
        out = bytearray()
        i = 0
        while i < len(data):
            b = data[i]
            if b == FESC:
                i += 1
                if i >= len(data):
                    raise KISSFrameError("Truncated FESC escape at end of frame")
                nxt = data[i]
                if nxt == TFEND:
                    out.append(FEND)
                elif nxt == TFESC:
                    out.append(FESC)
                else:
                    logger.warning(
                        "KISS destuff: unexpected byte 0x%02X after FESC -- kept raw",
                        nxt,
                    )
                    out.append(nxt)
            else:
                out.append(b)
            i += 1
        return bytes(out)

    def _dispatch(self, cmd: int, payload: bytes) -> None:
        """Deliver a decoded KISS frame to the application callback.

        Args:
            cmd: The raw command byte from the KISS frame (may include port
                nibble for XKISS).
            payload: The destuffed frame body (bytes after the command byte).
        """
        if self.on_frame:
            try:
                self.on_frame(cmd, payload)
            except Exception:
                logger.exception(
                    "KISS on_frame callback raised an exception "
                    "(cmd=0x%02X, payload_len=%d)",
                    cmd, len(payload),
                )
        else:
            logger.debug(
                "KISS frame received (no callback): cmd=0x%02X len=%d",
                cmd, len(payload),
            )
