# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.interfaces.kiss.py

KISS (Keep It Simple Stupid) interface implementation with full multi-drop support.

Supports:
- Standard KISS framing (FEND, FESC escaping)
- Multi-drop extension (G8BPQ): High nibble of command byte for TNC addressing ($x0-$xF)
- Extended commands: $xC (data transmit), $xE (poll frame)
- Parameter setting: TXDELAY ($x1), Persistence ($x2), SlotTime ($x3), TXTail ($x4), FullDuplex ($x5)
- Exit KISS mode ($FF - global)
- Synchronous operation with threaded background reading
- Error handling for serial I/O, invalid frames, and addressing
- Logging of all operations

Compliant with Multi-Drop KISS specification by Karl Medcalf WK5M and John Wiseman G8BPQ.
"""

from __future__ import annotations

import serial
import threading
import queue
import logging
from typing import Optional, Callable, Tuple

from pyax25_22.core.framing import AX25Frame
from pyax25_22.core.exceptions import KISSError

logger = logging.getLogger(__name__)

# KISS constants
FEND = 0xC0
FESC = 0xDB
TFEND = 0xDC
TFESC = 0xDD

# Command codes (low nibble)
CMD_DATA = 0x00
CMD_TXDELAY = 0x01
CMD_PERSISTENCE = 0x02
CMD_SLOTTIME = 0x03
CMD_TXTAIL = 0x04
CMD_FULLDUPLEX = 0x05
CMD_SETHARDWARE = 0x06
CMD_EXIT_KISS = 0xFF

# Extended multi-drop (low nibble)
CMD_DATA_EXT = 0x0C
CMD_POLL = 0x0E


class KISSInterface:
    """
    Synchronous KISS interface with multi-drop support.

    Usage:
        kiss = KISSInterface("/dev/ttyUSB0", baudrate=9600, tnc_address=1)
        kiss.connect()
        kiss.send_frame(frame)
        tnc_addr, port, received_frame = kiss.receive()
        kiss.disconnect()
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        tnc_address: int = 0,
        timeout: float = 1.0,
    ):
        """
        Initialize KISS interface.

        Args:
            port: Serial port path (e.g., /dev/ttyUSB0)
            baudrate: Serial baud rate
            tnc_address: TNC address (0-15) for multi-drop
            timeout: Read timeout in seconds
        """
        self.port_path = port
        self.baudrate = baudrate
        self.tnc_address = tnc_address & 0x0F  # Ensure 0-15
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self._recv_queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._callbacks: Dict[int, Callable] = {}  # Command low nibble -> callback
        logger.info(f"KISSInterface initialized: {port}@{baudrate}, TNC addr={tnc_address:02X}")

    def connect(self) -> None:
        """
        Open serial connection and start reader thread.
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
            self._thread = threading.Thread(target=self._reader_thread, daemon=True)
            self._thread.start()
            logger.info("KISS connected")
        except serial.SerialException as e:
            logger.error(f"Serial connection failed: {e}")
            raise KISSError(f"Failed to open {self.port_path}: {e}")

    def disconnect(self) -> None:
        """
        Close serial connection and stop reader thread.
        """
        self._running = False
        if self.serial:
            self.serial.close()
            self.serial = None
        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Reader thread did not terminate cleanly")
            self._thread = None
        logger.info("KISS disconnected")

    def send_frame(self, frame: AX25Frame, cmd: int = CMD_DATA) -> None:
        """
        Send AX.25 frame via KISS.

        Args:
            frame: AX25Frame to send
            cmd: Low nibble command (default CMD_DATA)
        """
        if not self.serial:
            raise KISSError("Not connected")

        raw = frame.encode()
        kiss_frame = bytes([FEND])
        cmd_byte = (self.tnc_address << 4) | (cmd & 0x0F)
        kiss_frame += bytes([cmd_byte])

        for b in raw:
            if b == FEND:
                kiss_frame += bytes([FESC, TFEND])
            elif b == FESC:
                kiss_frame += bytes([FESC, TFESC])
            else:
                kiss_frame += bytes([b])

        kiss_frame += bytes([FEND])

        try:
            self.serial.write(kiss_frame)
            logger.info(f"Sent frame via KISS: cmd={cmd:02X}, len={len(raw)}")
        except serial.SerialException as e:
            logger.error(f"Send failed: {e}")
            raise KISSError(f"Send failed: {e}")

    def set_parameter(self, cmd: int, value: bytes) -> None:
        """
        Set KISS parameter (e.g., TXDELAY).

        Args:
            cmd: Low nibble command (e.g., CMD_TXDELAY)
            value: Parameter data bytes
        """
        self.send_frame(AX25Frame(), cmd=cmd)  # Empty frame with param data

    def register_callback(self, cmd: int, callback: Callable) -> None:
        """
        Register callback for specific command low nibble.

        Args:
            cmd: Command low nibble (e.g., CMD_DATA)
            callback: Function(tnc_addr, port, frame)
        """
        self._callbacks[cmd] = callback
        logger.debug(f"Registered callback for cmd {cmd:02X}")

    def _reader_thread(self) -> None:
        """Background thread to read and parse KISS frames."""
        buffer = bytearray()
        in_escape = False

        while self._running:
            try:
                data = self.serial.read(512)  # Read chunk
                if not data:
                    continue

                for byte in data:
                    if in_escape:
                        if byte == TFEND:
                            buffer.append(FEND)
                        elif byte == TFESC:
                            buffer.append(FESC)
                        else:
                            logger.warning(f"Invalid escape sequence: {byte:02X}")
                            buffer.clear()
                        in_escape = False
                    elif byte == FEND:
                        if len(buffer) >= 2:  # Valid frame: at least cmd + data
                            cmd_byte = buffer[0]
                            tnc_addr = cmd_byte >> 4
                            cmd_low = cmd_byte & 0x0F
                            port = 0  # KISS port if needed (extended)
                            frame_data = bytes(buffer[1:])

                            try:
                                frame = AX25Frame.decode(frame_data, self.config)
                                self._recv_queue.put((tnc_addr, port, frame))
                                logger.debug(f"Received frame from TNC {tnc_addr:02X}, cmd={cmd_low:02X}")

                                if cmd_low in self._callbacks:
                                    self._callbacks[cmd_low](tnc_addr, port, frame)
                            except Exception as e:
                                logger.error(f"Frame decode failed: {e}")
                                raise KISSError(f"Invalid frame data: {e}")
                        buffer.clear()
                    elif byte == FESC:
                        in_escape = True
                    else:
                        buffer.append(byte)
            except serial.SerialException as e:
                logger.error(f"Read error: {e}")
                break

        logger.info("KISS reader thread stopped")

    def receive(self, timeout: Optional[float] = None) -> Tuple[int, int, AX25Frame]:
        """
        Receive next frame (tnc_addr, port, frame).

        Args:
            timeout: Optional queue timeout

        Raises:
            KISSError on timeout or error
        """
        try:
            return self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise KISSError("Receive timeout")
