# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
pyax25_22.interfaces.kiss -- KISS serial TNC interface.

KISS (Keep It Simple, Stupid) is a protocol from 1986 that lets a
computer talk to a TNC (Terminal Node Controller) over a serial port.
The TNC handles the radio: it listens for a quiet channel, presses
the transmit button, and converts between serial data and radio signals.
The computer just sends and receives AX.25 frame bytes through the serial
port wrapped in a simple envelope.

The KISS envelope has:
  - A FEND byte (0xC0) at the start and end of each frame.
  - A command byte right after the first FEND.
  - The actual frame data in between (with 0xC0 and 0xDB bytes escaped).

This file also supports Multi-Drop KISS (XKISS), a G8BPQ/WK5M extension
that uses the high nibble of the command byte to address up to 16 TNCs
on one serial bus.

Bugs fixed from the original:
  - Added missing ``Dict`` import.
  - Fixed ``set_parameter()`` to build a proper raw KISS frame instead
    of calling ``AX25Frame()`` with no arguments.
  - Fixed ``_reader_thread`` to use a stored config instead of the
    undefined ``self.config``.

Compliant with TAPR KISS Specification and Multi-Drop KISS (G8BPQ/WK5M).
"""

from __future__ import annotations

import queue
import threading
import logging
from typing import Callable, Dict, Optional, Tuple

import serial

from pyax25_22.core.config import AX25Config, DEFAULT_CONFIG_MOD8
from pyax25_22.core.framing import AX25Frame
from pyax25_22.core.exceptions import KISSError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KISS protocol constants
# ---------------------------------------------------------------------------

#: Frame End -- marks the start and end of each KISS frame.
FEND: int = 0xC0

#: Frame Escape -- precedes an escaped byte inside the frame data.
FESC: int = 0xDB

#: Transposed FEND -- FEND inside data is replaced by FESC + TFEND.
TFEND: int = 0xDC

#: Transposed FESC -- FESC inside data is replaced by FESC + TFESC.
TFESC: int = 0xDD

#: Command code 0: Data frame (the TNC should transmit this as an AX.25 frame).
CMD_DATA: int = 0x00

#: Command code 1: Set TX delay (time for the radio to start transmitting).
CMD_TXDELAY: int = 0x01

#: Command code 2: Set P-persistence (how aggressively to grab the channel).
CMD_PERSISTENCE: int = 0x02

#: Command code 3: Set slot time (how long to wait between channel checks).
CMD_SLOTTIME: int = 0x03

#: Command code 4: Set TX tail (how long to keep transmitting after last byte).
CMD_TXTAIL: int = 0x04

#: Command code 5: Set full duplex on (1) or off (0).
CMD_FULLDUPLEX: int = 0x05

#: Command code 6: Send hardware-specific commands (TNC-dependent).
CMD_SETHARDWARE: int = 0x06

#: Command code 0x0C: Extended data transmit (Multi-Drop KISS).
CMD_DATA_EXT: int = 0x0C

#: Command code 0x0E: Poll frame (Multi-Drop KISS -- host polls TNC).
CMD_POLL: int = 0x0E

#: Global exit command: tells all TNCs to leave KISS mode.
CMD_EXIT_KISS: int = 0xFF


# ---------------------------------------------------------------------------
# KISS interface class
# ---------------------------------------------------------------------------

class KISSInterface:
    """A synchronous KISS interface to a TNC over a serial port.

    Opens a serial port, wraps outgoing AX.25 frames in KISS format, and
    unwraps incoming KISS frames in a background reader thread.

    Also supports Multi-Drop KISS (XKISS): the high nibble of the command
    byte addresses one of up to 16 TNCs on the same serial bus.

    Attributes:
        port_path: The serial device path (e.g. ``/dev/ttyUSB0``).
        baudrate: The serial baud rate.
        tnc_address: The TNC address (0-15) for Multi-Drop KISS.
        timeout: Serial read timeout in seconds.
        frame_config: The AX.25 config used to decode received frames.

    Raises:
        KISSError: If the serial port cannot be opened, or if a send
            or receive operation fails.

    Example::

        kiss = KISSInterface("/dev/ttyUSB0", baudrate=9600)
        kiss.connect()
        kiss.send_frame(my_frame)
        tnc, port, frame = kiss.receive(timeout=5.0)
        kiss.disconnect()
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        tnc_address: int = 0,
        timeout: float = 1.0,
        frame_config: Optional[AX25Config] = None,
    ) -> None:
        """Set up a KISSInterface for the given serial port.

        Does not open the serial port. Call ``connect()`` to open it.

        Args:
            port: Serial device path (e.g. ``/dev/ttyUSB0`` on Linux or
                ``COM3`` on Windows).
            baudrate: Serial baud rate. Common TNC rates: 9600, 19200,
                38400, 57600. Default is 9600.
            tnc_address: TNC address for Multi-Drop KISS (0-15). For a
                single TNC on the bus, use 0. Default is 0.
            timeout: How many seconds to wait for serial data before
                the read loop yields. Shorter values are more responsive
                but use more CPU. Default is 1.0.
            frame_config: AX.25 configuration used to decode received
                frames. Defaults to the standard modulo-8 configuration.
        """
        self.port_path = port
        self.baudrate = baudrate
        self.tnc_address = tnc_address & 0x0F   # Clamp to 0-15
        self.timeout = timeout
        self.frame_config = frame_config or DEFAULT_CONFIG_MOD8

        self.serial: Optional[serial.Serial] = None
        self._recv_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._callbacks: Dict[int, Callable] = {}

        logger.info(
            "KISSInterface initialized: %s @ %d baud, TNC addr=%d",
            port, baudrate, self.tnc_address,
        )

    # -----------------------------------------------------------------------
    # Connect / disconnect
    # -----------------------------------------------------------------------

    def connect(self) -> None:
        """Open the serial port and start the background reader thread.

        Raises:
            KISSError: If the serial port cannot be opened (for example,
                the device does not exist or is already in use).

        Example::

            kiss.connect()
        """
        try:
            self.serial = serial.Serial(
                port=self.port_path,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self._running = True
            self._thread = threading.Thread(
                target=self._reader_thread, daemon=True, name="KISSReader"
            )
            self._thread.start()
            logger.info(
                "KISS connected: %s @ %d baud", self.port_path, self.baudrate
            )
        except serial.SerialException as exc:
            logger.error("Failed to open serial port %s: %s", self.port_path, exc)
            raise KISSError(f"Failed to open {self.port_path}: {exc}") from exc

    def disconnect(self) -> None:
        """Close the serial port and stop the reader thread.

        Safe to call even if already disconnected.

        Example::

            kiss.disconnect()
        """
        self._running = False

        if self.serial is not None:
            try:
                self.serial.close()
            except Exception as exc:
                logger.warning("Error closing serial port: %s", exc)
            finally:
                self.serial = None

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "KISS reader thread did not stop cleanly within 5 seconds"
                )
            self._thread = None

        logger.info("KISS disconnected")

    # -----------------------------------------------------------------------
    # Sending
    # -----------------------------------------------------------------------

    def send_frame(self, frame: AX25Frame, cmd: int = CMD_DATA) -> None:
        """Encode an AX.25 frame in KISS format and write it to the serial port.

        Builds the KISS envelope:
          FEND | cmd_byte | escaped_frame_data | FEND

        The command byte is built from the TNC address (high nibble) and
        the command code (low nibble).

        Args:
            frame: The AX25Frame to transmit. It will be encoded to bytes
                and then KISS-escaped.
            cmd: Low-nibble command code. Default is CMD_DATA (0x00) which
                tells the TNC to transmit this frame over the air.

        Raises:
            KISSError: If the interface is not connected, or if the serial
                write fails.

        Example::

            kiss.send_frame(my_frame)
            kiss.send_frame(my_frame, cmd=CMD_DATA_EXT)
        """
        if self.serial is None:
            raise KISSError("send_frame: not connected -- call connect() first")

        raw = frame.encode()
        cmd_byte = (self.tnc_address << 4) | (cmd & 0x0F)

        kiss_frame = bytearray([FEND, cmd_byte])
        for b in raw:
            if b == FEND:
                kiss_frame += bytes([FESC, TFEND])
            elif b == FESC:
                kiss_frame += bytes([FESC, TFESC])
            else:
                kiss_frame.append(b)
        kiss_frame.append(FEND)

        try:
            self.serial.write(kiss_frame)
            logger.info(
                "send_frame: cmd=0x%02X data=%d bytes (KISS frame=%d bytes)",
                cmd_byte, len(raw), len(kiss_frame),
            )
        except serial.SerialException as exc:
            logger.error("send_frame: serial write failed: %s", exc)
            raise KISSError(f"Serial send failed: {exc}") from exc

    def set_parameter(self, cmd: int, value: int) -> None:
        """Send a KISS parameter command to the TNC.

        Parameter commands set TNC operating values like TX delay and
        P-persistence. Each command takes one byte of data (the value).

        Unlike send_frame(), this does not wrap an AX.25 frame -- it
        sends a raw KISS command frame.

        Args:
            cmd: The command low nibble. Use one of:
                ``CMD_TXDELAY``, ``CMD_PERSISTENCE``, ``CMD_SLOTTIME``,
                ``CMD_TXTAIL``, ``CMD_FULLDUPLEX``.
            value: The parameter value (0-255).

        Raises:
            KISSError: If the interface is not connected or the write fails.

        Example::

            kiss.set_parameter(CMD_TXDELAY, 50)     # 500ms TX delay
            kiss.set_parameter(CMD_PERSISTENCE, 63) # 25% P-persistence
        """
        if self.serial is None:
            raise KISSError("set_parameter: not connected -- call connect() first")

        cmd_byte = (self.tnc_address << 4) | (cmd & 0x0F)
        kiss_frame = bytes([FEND, cmd_byte, value & 0xFF, FEND])

        try:
            self.serial.write(kiss_frame)
            logger.debug(
                "set_parameter: cmd=0x%02X value=%d", cmd_byte, value
            )
        except serial.SerialException as exc:
            logger.error("set_parameter: serial write failed: %s", exc)
            raise KISSError(f"Parameter set failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # Receiving
    # -----------------------------------------------------------------------

    def receive(
        self,
        timeout: Optional[float] = None,
    ) -> Tuple[int, int, AX25Frame]:
        """Wait for and return the next received frame.

        Blocks until a frame is available or the timeout expires.

        Args:
            timeout: How many seconds to wait. None waits forever.
                0 does a non-blocking check.

        Returns:
            A 3-tuple of:
            - tnc_addr (int): The TNC address (0-15) that sent the frame.
            - port (int): The KISS port (0 for standard KISS).
            - frame (AX25Frame): The decoded AX.25 frame.

        Raises:
            KISSError: If the timeout expires before a frame arrives.

        Example::

            tnc, port, frame = kiss.receive(timeout=10.0)
            print(f"From {frame.source.callsign}: {frame.info}")
        """
        try:
            return self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise KISSError("receive: timeout -- no frame received")

    def register_callback(self, cmd: int, callback: Callable) -> None:
        """Register a function to call when a specific KISS command is received.

        Args:
            cmd: The command low nibble to watch for (e.g. ``CMD_DATA``).
            callback: A callable that accepts (tnc_addr, port, frame).
                Called from the reader thread -- must be thread-safe.

        Example::

            def on_data(tnc_addr, port, frame):
                print(f"Received data from TNC {tnc_addr}")

            kiss.register_callback(CMD_DATA, on_data)
        """
        self._callbacks[cmd] = callback
        logger.debug("Registered callback for KISS command 0x%02X", cmd)

    # -----------------------------------------------------------------------
    # Reader thread (internal)
    # -----------------------------------------------------------------------

    def _reader_thread(self) -> None:
        """Background thread that reads bytes from serial and assembles frames.

        Runs until ``self._running`` is False. Handles FEND frame
        boundaries, FESC escape sequences, and dispatches complete frames
        to the receive queue and registered callbacks.
        """
        buffer = bytearray()
        in_escape = False

        logger.debug("KISS reader thread started")

        while self._running:
            try:
                data = self.serial.read(512)
                if not data:
                    continue

                for byte in data:
                    if in_escape:
                        if byte == TFEND:
                            buffer.append(FEND)
                        elif byte == TFESC:
                            buffer.append(FESC)
                        else:
                            logger.warning(
                                "_reader_thread: invalid escape byte 0x%02X -- "
                                "discarding frame", byte,
                            )
                            buffer.clear()
                        in_escape = False

                    elif byte == FEND:
                        # End of frame (or beginning of next frame)
                        if len(buffer) >= 2:
                            self._process_received_frame(bytes(buffer))
                        buffer.clear()

                    elif byte == FESC:
                        in_escape = True

                    else:
                        buffer.append(byte)

            except serial.SerialException as exc:
                if self._running:
                    logger.error("_reader_thread: serial read error: %s", exc)
                break
            except Exception as exc:
                if self._running:
                    logger.error("_reader_thread: unexpected error: %s", exc)

        logger.info("KISS reader thread stopped")

    def _process_received_frame(self, raw: bytes) -> None:
        """Decode and dispatch one received KISS frame.

        Called by the reader thread when a complete frame is assembled.
        Parses the command byte, decodes the AX.25 frame, and puts it
        in the receive queue.

        Args:
            raw: The raw bytes between two FEND markers, including the
                command byte as the first byte.
        """
        if len(raw) < 2:
            logger.debug("_process_received_frame: too short (%d bytes)", len(raw))
            return

        cmd_byte = raw[0]
        tnc_addr = cmd_byte >> 4
        cmd_low = cmd_byte & 0x0F
        frame_data = raw[1:]

        logger.debug(
            "_process_received_frame: cmd=0x%02X TNC=%d cmd_low=0x%X data=%d bytes",
            cmd_byte, tnc_addr, cmd_low, len(frame_data),
        )

        try:
            frame = AX25Frame.decode(frame_data, config=self.frame_config)
        except Exception as exc:
            logger.error(
                "_process_received_frame: could not decode AX.25 frame: %s", exc
            )
            return

        port = 0   # Standard KISS does not have a separate port field here

        self._recv_queue.put((tnc_addr, port, frame))
        logger.debug(
            "_process_received_frame: queued frame from %s to %s",
            frame.source.callsign, frame.destination.callsign,
        )

        # Fire registered callback if any
        if cmd_low in self._callbacks:
            try:
                self._callbacks[cmd_low](tnc_addr, port, frame)
            except Exception as exc:
                logger.error(
                    "_process_received_frame: callback for cmd 0x%X raised: %s",
                    cmd_low, exc,
                )
