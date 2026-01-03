# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
examples/kiss_monitor.py

Real-time monitor for KISS traffic with full multi-drop support.

This example demonstrates:
- Connecting to a KISS TNC (serial port)
- Using multi-drop addressing (TNC address in high nibble)
- Registering callbacks for data frames
- Receiving and decoding frames in real time
- Graceful shutdown on interrupt

Run with:
    python examples/kiss_monitor.py

Adjust the serial port and TNC address as needed for your hardware.
"""

import signal
import sys
import logging

from pyax25-22.interfaces.kiss import KISSInterface
from pyax25-22.utils.logging import get_logger

# Configure logging for the example
logger = get_logger("kiss_monitor")
logging.getLogger("pyax25-22").setLevel(logging.INFO)

# Default configuration - modify for your setup
SERIAL_PORT = "/dev/ttyUSB0"      # Common Linux path; Windows: "COM3"
BAUDRATE = 9600
TNC_ADDRESS = 0                   # 0-15; 0 is default for most TNCs


def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown."""
    logger.info("Interrupt received - shutting down...")
    sys.exit(0)


def on_data_frame(tnc_addr: int, port: int, frame) -> None:
    """
    Callback for received data frames (CMD_DATA = 0x00).

    Args:
        tnc_addr: TNC address from multi-drop high nibble
        port: Port index (usually 0)
        frame: Decoded AX25Frame
    """
    try:
        info_text = frame.info.decode('ascii', errors='replace').strip()
    except Exception:
        info_text = f"<binary data: {len(frame.info)} bytes>"

    print(f"[{tnc_addr:02X}:{port}] {frame.source.callsign}-{frame.source.ssid} → "
          f"{frame.destination.callsign}-{frame.destination.ssid}: {info_text}")


def main() -> None:
    """Main monitoring loop."""
    print("PyAX25_22 KISS Monitor")
    print(f"Connecting to {SERIAL_PORT} @ {BAUDRATE} baud, TNC address {TNC_ADDRESS:02X}")
    print("Press Ctrl+C to stop\n")

    # Create KISS interface
    kiss = KISSInterface(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        tnc_address=TNC_ADDRESS,
    )

    # Register callback for data frames
    kiss.register_callback(0x00, on_data_frame)  # CMD_DATA

    # Handle interrupt
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Connect and start monitoring
        kiss.connect()
        logger.info("Monitoring started - waiting for frames...")

        # Main loop - receive handles blocking with optional timeout
        while True:
            try:
                tnc_addr, port, frame = kiss.receive(timeout=1.0)
                # Primary handling via callback, fallback print
                print(f"Frame from TNC {tnc_addr:02X} port {port}")
            except Exception as e:
                # Timeout is normal - continue loop
                if "timeout" not in str(e).lower():
                    logger.error(f"Receive error: {e}")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        kiss.disconnect()
        print("Monitor stopped.")


if __name__ == "__main__":
    main()
