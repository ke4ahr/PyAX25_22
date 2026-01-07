# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_integration.py

Integration tests for full Layer 2 operation.

Covers:
- Complete connection lifecycle (SABM → UA → I-frames → DISC → DM)
- Timer interactions (T1 timeout, retry)
- Flow control integration
- Modulo 8 and 128 behavior
"""

import pytest
import pytest_asyncio
import time

from pyax25_22.core.framing import AX25Frame, AX25Address
from pyax25_22.core.statemachine import AX25StateMachine, AX25State
from pyax25_22.core.connected import AX25Connection
from pyax25_22.core.config import AX25Config, DEFAULT_CONFIG_MOD8, DEFAULT_CONFIG_MOD128


@pytest.fixture
def mock_connection_mod8():
    """Mock connection with modulo 8."""
    local = AX25Address("TEST")
    remote = AX25Address("DEST")
    conn = AX25Connection(
        local_addr=local,
        remote_addr=remote,
        config=DEFAULT_CONFIG_MOD8,
        initiate=True,
    )
    # Replace transport with mock
    conn.transport = MockTransport()
    return conn


@pytest.fixture
def mock_connection_mod128():
    """Mock connection with modulo 128."""
    config = AX25Config(modulo=128, window_size=63)
    local = AX25Address("TEST")
    remote = AX25Address("DEST")
    conn = AX25Connection(
        local_addr=local,
        remote_addr=remote,
        config=config,
        initiate=True,
    )
    conn.transport = MockTransport()
    return conn


class MockTransport:
    """Simple mock transport for integration testing."""

    def __init__(self):
        self.sent_frames = []
        self.received_frames = []

    def send_frame(self, frame: bytes):
        self.sent_frames.append(frame)

    def receive_frame(self):
        if self.received_frames:
            return self.received_frames.pop(0)
        return None

    def inject_frame(self, frame: bytes):
        self.received_frames.append(frame)

@pytest.mark.asyncio
async def test_full_connected_lifecycle(mock_connection_mod8):
    """Test complete connection lifecycle."""
    conn = mock_connection_mod8

    # Initiate connection
    await conn.connect()
    assert conn.state == AX25State.AWAITING_CONNECTION

    # Simulate UA response
    ua_frame = AX25Frame(
        destination=AX25Address("TEST"),
        source=AX25Address("DEST"),
        control=0x63,  # UA
    ).encode()
    conn.transport.inject_frame(ua_frame)
    await conn._process_incoming()

    assert conn.state == AX25State.CONNECTED

    # Send data
    await conn.send(b"Hello")
    assert len(conn.transport.sent_frames) > 0

    # Simulate ACK
    rr_frame = AX25Frame(
        destination=AX25Address("TEST"),
        source=AX25Address("DEST"),
        control=0x01,  # RR, N(R)=0
    ).encode()
    conn.transport.inject_frame(rr_frame)
    await conn._process_incoming()

    # Disconnect
    await conn.disconnect()
    assert conn.state == AX25State.AWAITING_RELEASE

    # Simulate UA
    ua_disc = AX25Frame(
        destination=AX25Address("TEST"),
        source=AX25Address("DEST"),
        control=0x63,
    ).encode()
    conn.transport.inject_frame(ua_disc)
    await conn._process_incoming()

    assert conn.state == AX25State.DISCONNECTED


@pytest.mark.asyncio
async def test_async_timer_t1(mock_connection_mod8):
    """Test T1 timeout and retry behavior."""
    conn = mock_connection_mod8
    # Create a new config with the desired timeout
    new_config = AX25Config(
    modulo=conn.config.modulo,
    max_frame=conn.config.max_frame,
    window_size=conn.config.window_size,
    t1_timeout=0.5,  # Short for testing
    t3_timeout=conn.config.t3_timeout,
    retry_count=conn.config.retry_count,
    tx_delay=conn.config.tx_delay,
    tx_tail=conn.config.tx_tail,
    persistence=conn.config.persistence,
    slot_time=conn.config.slot_time
    )
    conn.config = new_config

    await conn.connect()
    assert conn.state == AX25State.AWAITING_CONNECTION

    # Wait for T1 timeout (no UA received)
    await asyncio.sleep(1.0)
    await conn._process_timers()

    assert conn.state == AX25State.DISCONNECTED
    assert len(conn.transport.sent_frames) >= conn.config.retry_count  # Retries sent


@pytest.mark.asyncio
async def test_flow_control_integration(mock_connection_mod8):
    """Test flow control with peer busy."""
    conn = mock_connection_mod8

    await conn.connect()
    # Simulate UA
    conn.transport.inject_frame(
        AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("DEST"),
            control=0x63,
        ).encode()
    )
    await conn._process_incoming()
    assert conn.state == AX25State.CONNECTED

    # Simulate peer busy (RNR)
    conn.transport.inject_frame(
        AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("DEST"),
            control=0x85,  # RNR, N(R)=0
        ).encode()
    )
    await conn._process_incoming()

    assert conn.peer_busy

    # Should not send new I-frames while busy
    initial_sent = len(conn.transport.sent_frames)
    await conn.send(b"Blocked data")
    assert len(conn.transport.sent_frames) == initial_sent  # Enqueue only

    # Simulate peer ready (RR)
    conn.transport.inject_frame(
        AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("DEST"),
            control=0x01,  # RR
        ).encode()
    )
    await conn._process_incoming()

    assert not conn.peer_busy
    # Data should now be sent
    assert len(conn.transport.sent_frames) > initial_sent


@pytest.mark.asyncio
async def test_mod128_lifecycle(mock_connection_mod128):
    """Test connection with modulo 128."""
    conn = mock_connection_mod128

    await conn.connect()
    # Simulate SABME response with UA
    conn.transport.inject_frame(
        AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("DEST"),
            control=0x6F,  # UA (extended)
        ).encode()
    )
    await conn._process_incoming()

    assert conn.state == AX25State.CONNECTED
    assert conn.modulo == 128

    await conn.disconnect()
    assert conn.state == AX25State.AWAITING_RELEASE

    # Simulate UA
    conn.transport.inject_frame(
        AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("DEST"),
            control=0x6F,
        ).encode()
    )
    await conn._process_incoming()

    assert conn.state == AX25State.DISCONNECTED
