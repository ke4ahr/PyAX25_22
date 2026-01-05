# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_integration.py

End-to-end integration tests for PyAX25_22.

Covers:
- Full connected session lifecycle (SABM → UA → data → RR → DISC → UA)
- KISS transport round-trip with mock serial
- AGWPE transport round-trip with mock socket
- Multi-drop KISS addressing
- Async timer integration
- Error propagation across layers

Uses mocks for serial/socket to avoid hardware dependency.
"""

import pytest
import asyncio
import struct
from unittest.mock import Mock, MagicMock, patch

from pyax25_22.core.connected import AX25Connection
from pyax25_22.core.framing import AX25Address, AX25Frame
from pyax25_22.core.statemachine import AX25State
from pyax25_22.core.config import DEFAULT_CONFIG_MOD8
from pyax25_22.interfaces.kiss import KISSInterface
from pyax25_22.interfaces.agwpe import AGWPEInterface, HEADER_FMT, HEADER_SIZE
from pyax25_22.core.exceptions import KISSError, AGWPEError


def test_full_connected_lifecycle():
    """Test complete connected session from SABM to DISC."""
    local = AX25Address("KE4AHR", ssid=1)
    remote = AX25Address("NODE", ssid=0)

    # Initiating side
    initiator = AX25Connection(local, remote, initiate=True)
    sabm_frame = initiator.connect()
    assert initiator.sm.state == AX25State.AWAITING_CONNECTION

    # Receiving side
    receiver = AX25Connection(remote, local, initiate=False)
    receiver.process_frame(sabm_frame)
    assert receiver.sm.state == AX25State.CONNECTED

    # Send UA response
    ua_control = 0x63  # UA F=1
    ua_frame = AX25Frame(
        destination=local,
        source=remote,
        control=ua_control,
        config=DEFAULT_CONFIG_MOD8,
    )
    initiator.process_frame(ua_frame)
    assert initiator.sm.state == AX25State.CONNECTED

    # Send data
    initiator.send_data(b"Hello from PyAX25_22!")
    assert len(initiator.flow.outstanding_seqs) == 1

    # Peer acknowledges
    rr_control = 0x01 | (1 << 5)  # RR N(R)=1
    rr_frame = AX25Frame(
        destination=local,
        source=remote,
        control=rr_control,
        config=DEFAULT_CONFIG_MOD8,
    )
    initiator.process_frame(rr_frame)
    assert len(initiator.flow.outstanding_seqs) == 0

    # Disconnect
    disc_frame = initiator.disconnect()
    receiver.process_frame(disc_frame)
    assert receiver.sm.state == AX25State.DISCONNECTED

    # Final UA
    final_ua = receiver._send_ua()  # Internal helper
    initiator.process_frame(final_ua)
    assert initiator.sm.state == AX25State.DISCONNECTED


@pytest.mark.asyncio
async def test_async_timer_t1():
    """Test async T1 timeout handling."""
    from pyax25_22.core.timers import AX25Timers

    config = AX25Config(t1_timeout=0.1)  # Fast timeout
    timers = AX25Timers(config)

    timeout_called = False

    async def on_timeout():
        nonlocal timeout_called
        timeout_called = True

    await timers.start_t1_async(on_timeout)
    await asyncio.sleep(0.2)  # Wait for timeout
    assert timeout_called


@pytest.fixture
def mock_serial():
    """Mock serial port for KISS testing."""
    mock = MagicMock()
    mock.write = Mock()
    mock.read = Mock()
    return mock


@pytest.fixture
def mock_socket():
    """Mock socket for AGWPE testing."""
    mock = MagicMock()
    mock.sendall = Mock()
    mock.recv = Mock()
    return mock


@pytest.mark.usefixtures("mock_serial")
def test_kiss_integration_mock(mock_serial):
    """Test KISS send/receive with mock serial."""
    with patch('serial.Serial', return_value=mock_serial):
        kiss = KISSInterface("mock", tnc_address=1)
        kiss.connect()

        frame = AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("KE4AHR"),
            control=0x03,
            info=b"test"
        )
        kiss.send_frame(frame)

        mock_serial.write.assert_called()

        # Simulate receive
        encoded = frame.encode()
        kiss_frame = bytes([0xC0, 0x10]) + encoded + bytes([0xC0])  # TNC 1, port 0
        mock_serial.read.return_value = kiss_frame

        tnc_addr, port, recv_frame = kiss.receive()
        assert tnc_addr == 1
        assert port == 0
        assert recv_frame.info == b"test"

        kiss.disconnect()


@pytest.mark.usefixtures("mock_socket")
def test_agwpe_integration_mock(mock_socket):
    """Test AGWPE send/receive with mock socket."""
    with patch('socket.socket', return_value=mock_socket):
        agwpe = AGWPEInterface()
        agwpe.connect()

        agwpe.send_frame(1, 'D', 'KE4AHR', 'NODE', b'test')

        mock_socket.sendall.assert_called()

        # Simulate receive
        header = struct.pack(HEADER_FMT, 1, ord('D'), b'KE4AHR   \\x00', b'NODE     \\x00', 4, 0)
        mock_socket.recv.side_effect = [header, b'test']

        port, kind, fr, to, data = agwpe.receive()
        assert port == 1
        assert kind == 'D'
        assert data == b'test'

        agwpe.disconnect()


def test_multi_drop_kiss():
    """Test multi-drop addressing in KISS."""
    mock_serial = MagicMock()
    mock_serial.read.return_value = b'\\xc0\\x21testdata\\xc0'  # TNC 2, port 1

    with patch('serial.Serial', return_value=mock_serial):
        kiss = KISSInterface("mock", tnc_address=2)
        kiss.connect()

        tnc_addr, port, _ = kiss.receive()
        assert tnc_addr == 2
        assert port == 1


def test_error_propagation():
    """Test error propagation from transport to core."""
    mock_serial = MagicMock()
    mock_serial.write.side_effect = OSError("Mock I/O error")

    with patch('serial.Serial', return_value=mock_serial):
        kiss = KISSInterface("mock")
        kiss.connect()

        frame = AX25Frame(destination=AX25Address("TEST"), source=AX25Address("TEST"))
        with pytest.raises(KISSError):
            kiss.send_frame(frame)