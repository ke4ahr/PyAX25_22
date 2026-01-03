# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_statemachine.py

Comprehensive unit tests for AX.25 state machine.

Covers:
- All valid state transitions per v2.2 SDL
- Invalid transition error handling
- Sequence variable resets
- Modulo 8 and 128 differences
- Full event coverage (connect_request, SABM/UA, DISC/DM, timeouts, etc.)
"""

import pytest

from pyax25_22.core.statemachine import AX25StateMachine, AX25State
from pyax25_22.core.exceptions import ConnectionStateError
from pyax25_22.core.config import AX25Config, DEFAULT_CONFIG_MOD8, DEFAULT_CONFIG_MOD128


@pytest.fixture
def sm_mod8():
    """Fixture for modulo 8 state machine."""
    return AX25StateMachine(config=DEFAULT_CONFIG_MOD8)


@pytest.fixture
def sm_mod128():
    """Fixture for modulo 128 state machine."""
    return AX25StateMachine(config=AX25Config(modulo=128))


def test_initial_state(sm_mod8):
    """Test initial disconnected state."""
    assert sm_mod8.state == AX25State.DISCONNECTED
    assert sm_mod8.v_s == 0
    assert sm_mod8.v_r == 0
    assert sm_mod8.v_a == 0
    assert not sm_mod8.peer_busy
    assert not sm_mod8.reject_sent
    assert not sm_mod8.srej_sent
    assert not sm_mod8.layer3_initiated


def test_connect_request_without_layer3(sm_mod8):
    """Test connect_request without layer3_initiated."""
    with pytest.raises(ConnectionStateError):
        sm_mod8.transition("connect_request")


def test_connect_request(sm_mod8):
    """Test successful connect_request transition."""
    sm_mod8.layer3_initiated = True
    sm_mod8.transition("connect_request")
    assert sm_mod8.state == AX25State.AWAITING_CONNECTION
    assert sm_mod8.v_s == 0
    assert sm_mod8.v_r == 0
    assert sm_mod8.v_a == 0


def test_sabm_received_from_disconnected(sm_mod8):
    """Test SABM reception from disconnected."""
    sm_mod8.transition("SABM_received")
    assert sm_mod8.state == AX25State.CONNECTED


def test_sabme_received_mod128(sm_mod128):
    """Test SABME for mod128."""
    sm_mod128.transition("SABME_received")
    assert sm_mod128.state == AX25State.CONNECTED


def test_disc_from_disconnected(sm_mod8):
    """Test DISC from disconnected (remains disconnected)."""
    sm_mod8.transition("DISC_received")
    assert sm_mod8.state == AX25State.DISCONNECTED


def test_ua_from_awaiting_connection(sm_mod8):
    """Test UA response in awaiting connection."""
    sm_mod8.layer3_initiated = True
    sm_mod8.transition("connect_request")
    sm_mod8.transition("UA_received")
    assert sm_mod8.state == AX25State.DISCONNECTED


def test_timeout_from_awaiting_connection(sm_mod8):
    """Test T1 timeout in awaiting connection."""
    sm_mod8.layer3_initiated = True
    sm_mod8.transition("connect_request")
    sm_mod8.transition("T1_timeout")
    assert sm_mod8.state == AX25State.DISCONNECTED


def test_disconnect_request_from_connected(sm_mod8):
    """Test disconnect from connected."""
    sm_mod8.transition("SABM_received")
    sm_mod8.transition("disconnect_request")
    assert sm_mod8.state == AX25State.AWAITING_RELEASE


def test_disc_from_connected(sm_mod8):
    """Test DISC reception in connected."""
    sm_mod8.transition("SABM_received")
    sm_mod8.transition("DISC_received")
    assert sm_mod8.state == AX25State.DISCONNECTED


def test_rnr_in_connected(sm_mod8):
    """Test RNR supervisory in connected."""
    sm_mod8.transition("SABM_received")
    assert not sm_mod8.peer_busy
    sm_mod8.transition("RNR_received")
    assert sm_mod8.peer_busy


def test_rr_in_connected(sm_mod8):
    """Test RR supervisory in connected."""
    sm_mod8.transition("SABM_received")
    sm_mod8.peer_busy = True
    sm_mod8.transition("RR_received")
    assert not sm_mod8.peer_busy


def test_rej_in_connected(sm_mod8):
    """Test REJ supervisory in connected."""
    sm_mod8.transition("SABM_received")
    assert not sm_mod8.reject_sent
    sm_mod8.transition("REJ_received")
    assert sm_mod8.reject_sent


def test_srej_in_connected(sm_mod8):
    """Test SREJ supervisory in connected."""
    sm_mod8.transition("SABM_received")
    assert not sm_mod8.srej_sent
    sm_mod8.transition("SREJ_received")
    assert sm_mod8.srej_sent


def test_t1_timeout_from_connected(sm_mod8):
    """Test T1 timeout to timer recovery."""
    sm_mod8.transition("SABM_received")
    sm_mod8.transition("T1_timeout")
    assert sm_mod8.state == AX25State.TIMER_RECOVERY


def test_ack_response_from_timer_recovery(sm_mod8):
    """Test acknowledgment in timer recovery."""
    sm_mod8.transition("SABM_received")
    sm_mod8.transition("T1_timeout")  # To recovery
    sm_mod8.reject_sent = True
    sm_mod8.srej_sent = True
    sm_mod8.transition("RR_response")
    assert sm_mod8.state == AX25State.CONNECTED
    assert not sm_mod8.reject_sent
    assert not sm_mod8.srej_sent


def test_t1_timeout_from_timer_recovery(sm_mod8):
    """Test repeated T1 in recovery."""
    sm_mod8.transition("SABM_received")
    sm_mod8.transition("T1_timeout")  # To recovery
    sm_mod8.transition("T1_timeout")  # Stays, increments retry (handled elsewhere)


def test_ua_from_awaiting_release(sm_mod8):
    """Test UA in awaiting release."""
    sm_mod8.transition("SABM_received")
    sm_mod8.transition("disconnect_request")
    sm_mod8.transition("UA_received")
    assert sm_mod8.state == AX25State.DISCONNECTED


def test_t1_timeout_from_awaiting_release(sm_mod8):
    """Test T1 in awaiting release."""
    sm_mod8.transition("SABM_received")
    sm_mod8.transition("disconnect_request")
    sm_mod8.transition("T1_timeout")
    assert sm_mod8.state == AX25State.DISCONNECTED


def test_xid_from_awaiting_xid(sm_mod8):
    """Test XID response in awaiting XID."""
    sm_mod8.state = AX25State.AWAITING_XID
    sm_mod8.transition("XID_response")
    assert sm_mod8.state == AX25State.CONNECTED


def test_t1_timeout_from_awaiting_xid(sm_mod8):
    """Test T1 in awaiting XID."""
    sm_mod8.state = AX25State.AWAITING_XID
    sm_mod8.transition("T1_timeout")
    assert sm_mod8.state == AX25State.DISCONNECTED


def test_invalid_event_raises_error(sm_mod8):
    """Test invalid events in various states."""
    with pytest.raises(ConnectionStateError):
        sm_mod8.transition("invalid_event")

    sm_mod8.transition("SABM_received")
    with pytest.raises(ConnectionStateError):
        sm_mod8.transition("connect_request")  # Invalid in connected


def test_sequence_increment_mod8(sm_mod8):
    """Test sequence wrap-around mod 8."""
    assert sm_mod8.increment_seq(7) == 0
    assert sm_mod8.increment_seq(0) == 1


def test_sequence_increment_mod128(sm_mod128):
    """Test sequence wrap-around mod 128."""
    assert sm_mod128.increment_seq(127) == 0
    assert sm_mod128.increment_seq(100) == 101


def test_modulo_mask(sm_mod8, sm_mod128):
    """Test modulo mask property."""
    assert sm_mod8.modulo_mask == 0x07
    assert sm_mod128.modulo_mask == 0x7F
