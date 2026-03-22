# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/serial.py

KISSSerial -- KISS over a serial port (RS-232 / USB-serial TNC).

Extends KISSBase with:
- pyserial serial port management
- Background reader thread (daemon)
- Blocking write with lock

This is the most common KISS transport -- a hardware TNC connected via a
serial cable or a USB-to-serial adapter.
"""

import logging
import threading
import time
from typing import Callable, Optional

try:
    import serial
    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False

from .base import KISSBase
from .constants import DEFAULT_BAUDRATE
from .exceptions import KISSSerialError

logger = logging.getLogger(__name__)


class KISSSerial(KISSBase):
    """KISS protocol over a serial port.

    Opens a pyserial connection to the TNC, starts a background reader
    thread, and implements write() by calling serial.write().

    Args:
        device: OS path to the serial device (e.g., ``/dev/ttyS0`` or
            ``COM3``).
        baudrate: Serial baud rate (default: 9600).  Must match TNC setting.
        on_frame: Optional callback ``(cmd: int, payload: bytes) -> None``
            called for each received KISS frame.
        read_timeout: Per-byte serial read timeout in seconds (default: 1.0).
            Controls how often the reader thread wakes up to check
            ``_running``.

    Raises:
        ImportError: If pyserial is not installed.
        KISSSerialError: If the serial port cannot be opened.

    Example:
        def on_frame(cmd, data):
            print(f"Got frame: cmd=0x{cmd:02X} len={len(data)}")

        tnc = KISSSerial("/dev/ttyUSB0", baudrate=9600, on_frame=on_frame)
        tnc.send(ax25_bytes)
        tnc.close()
    """

    def __init__(
        self,
        device: str,
        baudrate: int = DEFAULT_BAUDRATE,
        on_frame: Optional[Callable[[int, bytes], None]] = None,
        read_timeout: float = 1.0,
        **kwargs,
    ) -> None:
        if not _HAS_SERIAL:
            raise ImportError(
                "pyserial is required for KISSSerial. "
                "Install it with: pip install pyserial"
            )

        super().__init__(on_frame=on_frame, **kwargs)

        self._device = device
        self._baudrate = baudrate
        self._read_timeout = read_timeout
        self._write_lock = threading.Lock()
        self._running = True

        try:
            self._serial = serial.Serial(
                device, baudrate, timeout=read_timeout
            )
        except serial.SerialException as exc:
            logger.critical(
                "KISSSerial: cannot open %s @ %d baud: %s",
                device, baudrate, exc,
            )
            raise KISSSerialError(f"Serial open failed: {exc}") from exc

        self._thread = threading.Thread(
            target=self._receive_loop,
            name=f"KISSSerial-{device}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "KISSSerial: opened %s @ %d baud", device, baudrate
        )

    # -----------------------------------------------------------------------
    # KISSBase abstract method implementation
    # -----------------------------------------------------------------------

    def write(self, data: bytes) -> None:
        """Write raw KISS frame bytes to the serial port.

        Acquires a write lock so that concurrent calls from multiple threads
        (e.g., T1 timeout retransmission + normal send) do not interleave.

        Args:
            data: Encoded KISS frame bytes (including FEND delimiters).

        Raises:
            KISSSerialError: If the serial write fails.
        """
        with self._write_lock:
            try:
                self._serial.write(data)
                logger.debug(
                    "KISSSerial write: %d bytes to %s",
                    len(data), self._device,
                )
            except serial.SerialException as exc:
                logger.error(
                    "KISSSerial write error on %s: %s", self._device, exc
                )
                raise KISSSerialError(f"Serial write failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # Reader thread
    # -----------------------------------------------------------------------

    def _receive_loop(self) -> None:
        """Background thread: read bytes from serial and process KISS frames.

        Reads up to 1024 bytes at a time (blocking with read_timeout).  Each
        received byte is passed to _process_byte() inherited from KISSBase.

        The loop runs until _running is set to False by close().
        """
        logger.debug(
            "KISSSerial reader thread started for %s", self._device
        )
        while self._running:
            try:
                data = self._serial.read(1024)
                if not data:
                    continue
                for byte in data:
                    self._process_byte(byte)
            except serial.SerialException as exc:
                if self._running:
                    logger.error(
                        "KISSSerial read error on %s: %s -- retrying in 1 s",
                        self._device, exc,
                    )
                    time.sleep(1)
            except Exception:
                logger.exception(
                    "KISSSerial: unexpected error in reader thread for %s",
                    self._device,
                )
                break
        logger.debug(
            "KISSSerial reader thread stopped for %s", self._device
        )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Close the serial port and stop the reader thread.

        Signals the reader thread to stop by clearing _running, then closes
        the serial port.  The reader thread will exit on its next read timeout.
        """
        self._running = False
        if self._serial.is_open:
            try:
                self._serial.close()
                logger.info("KISSSerial: closed %s", self._device)
            except serial.SerialException as exc:
                logger.warning(
                    "KISSSerial: error closing %s: %s", self._device, exc
                )
        super().close()

    def __del__(self) -> None:
        """Destructor: attempt a graceful close on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
