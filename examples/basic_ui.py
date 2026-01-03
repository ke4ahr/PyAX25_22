# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
examples/basic_ui.py

Simple demonstration of creating and encoding an AX.25 UI (Unconnected Information) frame.

This example shows the minimal usage of the PyAX25_22 core framing module to:
- Create source and destination addresses with SSIDs
- Build a UI frame (common for beacons, APRS, PACSAT broadcasts)
- Set PID = 0xF0 (no Layer 3 protocol)
- Encode the complete frame with flags, bit stuffing, and FCS
- Display the raw encoded frame in hexadecimal

Run this script directly to see the output.
"""

from pyax25_22.core.framing import AX25Frame, AX25Address

def main() -> None:
    """
    Main function demonstrating basic UI frame creation.
    """
    # Define destination and source addresses (common APRS/beacon style)
    destination = AX25Address(callsign="APRS", ssid=0)      # Wide-area digipeater
    source = AX25Address(callsign="KE4AHR", ssid=1)         # Your station with SSID 1

    # Create a UI frame with example payload
    ui_frame = AX25Frame(
        destination=destination,
        source=source,
        control=0x03,          # UI frame (unconnected)
        pid=0xF0,              # No Layer 3 protocol
        info=b"PyAX25_22 v0.1.0 beacon - 73 de KE4AHR"
    )

    # Encode the full frame (includes flags, bit stuffing, FCS)
    encoded_frame = ui_frame.encode()

    # Display results
    print("Basic UI Frame Example")
    print("=" * 50)
    print(f"Source:      {source.callsign}-{source.ssid}")
    print(f"Destination: {destination.callsign}-{destination.ssid}")
    print(f"Info field:  {ui_frame.info.decode('ascii', errors='replace')}")
    print(f"Frame length: {len(encoded_frame)} bytes")
    print(f"Encoded frame (hex):")
    print(encoded_frame.hex().upper())

    # Optional: Show frame breakdown
    print("\nFrame breakdown:")
    print(f"  Start flag:  {encoded_frame[0]:02X}")
    print(f"  Address field: {encoded_frame[1:15].hex().upper()}")
    print(f"  Control:     {encoded_frame[15]:02X} (UI)")
    print(f"  PID:         {encoded_frame[16]:02X} (No L3)")
    print(f"  Info length: {len(ui_frame.info)} bytes")
    print(f"  FCS:         {encoded_frame[-3:-1].hex().upper()}")
    print(f"  End flag:    {encoded_frame[-1]:02X}")

if __name__ == "__main__":
    main()
