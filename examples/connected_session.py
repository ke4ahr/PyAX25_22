# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
examples/connected_session.py

Demonstration of a full connected-mode AX.25 session.

This example shows how to:
- Create local and remote addresses
- Initialize a connection object in initiating mode
- Send SABM/SABME to start connection
- Simulate receiving UA response from peer
- Send information frames with data
- Handle disconnection gracefully
- Use the state machine and flow control components

Run this script to see a simulated connected session lifecycle.
"""

from pyax25_22.core.framing import AX25Address
from pyax25_22.core.connected import AX25Connection
from pyax25_22.core.config import DEFAULT_CONFIG_MOD8

def main() -> None:
    """
    Main function demonstrating a complete connected session.
    """
    # Define addresses
    local = AX25Address("KE4AHR", ssid=1)    # Your station
    remote = AX25Address("PACKET", ssid=0)  # Remote node/BBS

    # Create connection that will initiate
    conn = AX25Connection(
        local_addr=local,
        remote_addr=remote,
        config=DEFAULT_CONFIG_MOD8,
        initiate=True,
    )

    print("=== AX.25 Connected Session Example ===")
    print(f"Local:  {local.callsign}-{local.ssid}")
    print(f"Remote: {remote.callsign}-{remote.ssid}")
    print(f"Modulo: {conn.config.modulo}")
    print()

    # Step 1: Send connection request
    print("1. Sending connection request (SABM/SABME)...")
    sabm_frame = conn.connect()
    print(f"   Encoded SABM frame: {sabm_frame.encode().hex().upper()}")
    print(f"   Current state: {conn.sm.state.name}")
    print()

    # Step 2: Simulate receiving UA response from peer
    print("2. Simulating peer response (UA)...")
    # In real code, this would come from transport.receive()
    # Here we create a mock UA frame
    from pyax25_22.core.framing import AX25Frame
    ua_control = 0x63  # UA with F=1
    ua_frame = AX25Frame(
        destination=local,
        source=remote,
        control=ua_control,
        config=conn.config,
    )
    conn.process_frame(ua_frame)
    print(f"   Connection established! State: {conn.sm.state.name}")
    print()

    # Step 3: Send some data
    print("3. Sending information frames...")
    conn.send_data(b"Hello from PyAX25_22!")
    conn.send_data(b"This is a connected mode test.")
    conn.send_data(b"73 de KE4AHR")
    print("   Data queued for transmission")
    print(f"   Outstanding frames: {len(conn.flow.outstanding_seqs)}")
    print()

    # Step 4: Simulate acknowledgment
    print("4. Simulating peer acknowledgment (RR)...")
    rr_control = 0x01 | (conn.v_r << 5)  # RR with current N(R)
    rr_frame = AX25Frame(
        destination=local,
        source=remote,
        control=rr_control,
        config=conn.config,
    )
    conn.process_frame(rr_frame)
    print("   Acknowledgments processed")
    print()

    # Step 5: Disconnect
    print("5. Initiating disconnection...")
    disc_frame = conn.disconnect()
    print(f"   Sent DISC frame: {disc_frame.encode().hex().upper()}")
    print(f"   Final state: {conn.sm.state.name}")
    print()

    print("=== Session complete ===")
    print("This demonstrates full connected-mode lifecycle:")
    print("   • Connection establishment")
    print("   • Information transfer")
    print("   • Flow control")
    print("   • Graceful disconnection")

if __name__ == "__main__":
    main()
