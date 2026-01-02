```python
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
examples/agwpe_monitor.py

Real-time monitor for AGWPE traffic.

This example demonstrates:
- Connecting to an AGWPE server (default localhost:8000)
- Registering callbacks for monitored frames ('M' DataKind)
- Enabling monitoring mode
- Receiving and decoding frames in real time
- Graceful shutdown on interrupt

Run with:
    python examples/agwpe_monitor.py

Adjust host/port as needed for your AGWPE setup.
"""

import signal
import sys
import logging

from pyax25_22.interfaces.agwpe import AGWPEInterface
from pyax25_22.utils.logging import get_logger

# Configure logging for the example
logger = get_logger("agwpe_monitor")
logging.getLogger("pyax25_22").setLevel(logging.INFO)

# Default configuration - modify for your setup
AGWPE_HOST = "127.0.0.1"
AGWPE_PORT = 8000


def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown."""
    logger.info("Interrupt received - shutting down...")
    sys.exit(0)


def on_monitored_frame(port: int, fr: str, to: str, data: bytes) -> None:
    """
    Callback for monitored frames ('M' DataKind).

    Args:
        port: Radio port index
        fr: From callsign
        to: To callsign
        data: Frame data
    """
    try:
        data_text = data.decode('ascii', errors='replace').strip()
    except Exception:
        data_text = f"<binary data: {len(data)} bytes>"

    print(f"[{port}] {fr} → {to}: {data_text}")


def main() -> None:
    """Main monitoring loop."""
    print("PyAX25_22 AGWPE Monitor")
    print(f"Connecting to {AGWPE_HOST}:{AGWPE_PORT}")
    print("Press Ctrl+C to stop\n")

    # Create AGWPE interface
    agwpe = AGWPEInterface(host=AGWPE_HOST, port=AGWPE_PORT)

    # Register callback for monitored frames
    agwpe.register_callback('M', on_monitored_frame)

    # Handle interrupt
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Connect and enable monitoring
        agwpe.connect()
        agwpe.enable_monitoring()
        logger.info("Monitoring started - waiting for frames...")

        # Main loop - receive handles blocking with optional timeout
        while True:
            try:
                port, kind, fr, to, data = agwpe.receive(timeout=1.0)
                # Primary handling via callback, fallback print for other kinds
                if kind != 'M':
                    print(f"Raw [{port}] {kind}: {fr} → {to} ({len(data)} bytes)")
            except Exception as e:
                # Timeout is normal - continue loop
                if "timeout" not in str(e).lower():
                    logger.error(f"Receive error: {e}")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        agwpe.disconnect()
        print("Monitor stopped.")


if __name__ == "__main__":
    main()
