# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_flow_control.py

Comprehensive unit tests for flow control module.

Covers:
- Window availability and limits
- Enqueue and acknowledgment
- Peer/local busy states
- REJ and SREJ generation/handling
- Reset
- Integration with state machine
- Error cases (window overflow, invalid states)
"""

import pytest

from pyax25_22.core.flow_control import AX25FlowControl
from pyax25_22.core.statemachine import AX25StateMachine
from pyax25_22.core.framing import AX25Frame
from pyax25_22.core.exceptions import FrameError
from pyax25_22.core.config import AX25Config, DEFAULT_CONFIG_MOD8


@pytest.fixture
def flow_mod8():
    """Fixture for mod 8 flow control."""
    sm = AX25StateMachine()
    config = DEFAULT_CONFIG_MOD8
    return AX25FlowControl(sm, config)


@pytest.fixture
def flow_mod128():
    """Fixture for mod 128 flow control."""
    sm = AX25StateMachine(config=AX25Config(modulo=128, window_size=63))
    return AX25FlowControl(sm, sm.config)


def test_initial_state(flow_mod8):
    """Test initial flow control state."""
    assert flow_mod8.window_available == 7
    assert flow_mod8.can_send_i_frame()
    assert not flow_mod8.local_busy
    assert not flow_mod8.peer_busy
    assert not flow_mod8.rej_sent
    assert not flow_mod8.srej_sent


def test_window_management(flow_mod8):
    """Test basic enqueue and acknowledgment."""
    for i in range(7):
        flow_mod8.enqueue_i_frame(i)
    assert flow_mod8.window_available == 0
    assert not flow_mod8.can_send_i_frame()

    with pytest.raises(FrameError):
        flow_mod8.enqueue_i_frame(7)  # Overflow

    flow_mod8.acknowledge_up_to(4)
    assert flow_mod8.window_available == 4
    assert flow_mod8.can_send_i_frame()


def test_busy_states(flow_mod8):
    """Test local and peer busy handling."""
    assert flow_mod8.can_send_i_frame()

    # Set peer busy
    flow_mod8.set_peer_busy()
    assert not flow_mod8.can_send_i_frame()
    flow_mod8.clear_peer_busy()
    assert flow_mod8.can_send_i_frame()

    # Set local busy
    flow_mod8.set_local_busy()
    assert flow_mod8.local_busy
    flow_mod8.clear_local_busy()
    assert not flow_mod8.local_busy


def test_reject_generation(flow_mod8):
    """Test REJ frame generation."""
    rej_frame = flow_mod8.send_reject(5)
    assert rej_frame is not None
    assert rej_frame.control & 0x0F == 0x09  # REJ base
    assert (rej_frame.control >> 5) & 0x07 == 5  # N(R)=5
    assert flow_mod8.rej_sent == True

    # No duplicate REJ
    assert flow_mod8.send_reject(5) is None


def test_srej_generation(flow_mod8):
    """Test SREJ frame generation."""
    srej_frame = flow_mod8.send_selective_reject(3)
    assert srej_frame is not None
    assert srej_frame.control & 0x0F == 0x0D  # SREJ base
    assert (srej_frame.control >> 5) & 0x07 == 3  # N(R)=3
    assert flow_mod8.srej_sent == True
    assert 3 in flow_mod8.srej_requested

    # No duplicate SREJ
    assert flow_mod8.send_selective_reject(3) is None


def test_rr_rnr_generation(flow_mod8):
    """Test RR and RNR frame generation."""
    # Assume sm.v_r = 4
    flow_mod8.sm.v_r = 4

    rr_frame = flow_mod8.send_rr(pf_bit=True)
    assert rr_frame.control & 0x0F == 0x01  # RR base
    assert (rr_frame.control >> 5) & 0x07 == 4  # N(R)
    assert rr_frame.control & 0x10 == 0x10  # P/F

    rnr_frame = flow_mod8.send_rnr(pf_bit=False)
    assert rnr_frame.control & 0x0F == 0x05  # RNR base
    assert (rnr_frame.control >> 5) & 0x07 == 4
    assert rnr_frame.control & 0x10 == 0x00


def test_reset(flow_mod8):
    """Test reset clears all state."""
    flow_mod8.enqueue_i_frame(1)
    flow_mod8.set_peer_busy()
    flow_mod8.set_local_busy()
    flow_mod8.rej_sent = True
    flow_mod8.srej_sent = True
    flow_mod8.srej_requested = [2, 3]

    flow_mod8.reset()

    assert len(flow_mod8.outstanding_seqs) == 0
    assert len(flow_mod8.srej_requested) == 0
    assert not flow_mod8.local_busy
    assert not flow_mod8.peer_busy
    assert not flow_mod8.rej_sent
    assert not flow_mod8.srej_sent


def test_mod128_window(flow_mod128):
    """Test large window in mod 128."""
    for i in range(63):
        flow_mod128.enqueue_i_frame(i)
    assert flow_mod128.window_available == 0

    flow_mod128.acknowledge_up_to(30)
    assert flow_mod128.window_available == 30


def test_busy_prevents_enqueue(flow_mod8):
    """Test enqueue blocks when peer busy."""
    flow_mod8.set_peer_busy()
    with pytest.raises(FrameError):
        flow_mod8.enqueue_i_frame(0)
