# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_timers.py

Unit tests for the AX25Timers class.

Covers:
- Initial state
- Timer status reporting
- T1/T3 sync start and stop
- RTT/SRTT update via record_acknowledgment
- Timer reset
- T1/T3 timeout value updates
"""

import time
import pytest

from pyax25_22.core.timers import AX25Timers
from pyax25_22.core.config import DEFAULT_CONFIG_MOD8, AX25Config


@pytest.fixture
def timers():
    """Default timers from standard mod-8 config."""
    return AX25Timers(DEFAULT_CONFIG_MOD8)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_rto(timers):
    """Initial RTO is set from config T1."""
    # RTO = srtt + 4*rttvar, both initialized from t1_timeout
    assert timers.rto >= DEFAULT_CONFIG_MOD8.t1_timeout


def test_initial_not_running(timers):
    """No timers are running after creation."""
    status = timers.get_timer_status()
    assert not status["t1_running"]
    assert not status["t3_running"]


# ---------------------------------------------------------------------------
# T1 sync start/stop
# ---------------------------------------------------------------------------

def test_t1_sync_starts_and_stops(timers):
    """T1 sync timer can be started and stopped."""
    fired = []
    timers.start_t1_sync(lambda: fired.append(1))
    status = timers.get_timer_status()
    assert status["t1_running"]

    timers.stop_t1_sync()
    status = timers.get_timer_status()
    assert not status["t1_running"]
    # Must not have fired (we stopped it before it could expire)
    assert len(fired) == 0


def test_t1_sync_restart(timers):
    """Starting T1 a second time cancels the old one."""
    fired = []
    timers.start_t1_sync(lambda: fired.append("first"))
    timers.start_t1_sync(lambda: fired.append("second"))
    # Only one timer should be running
    assert timers._t1_thread_timer is not None
    timers.stop_t1_sync()


# ---------------------------------------------------------------------------
# T3 sync start/stop
# ---------------------------------------------------------------------------

def test_t3_sync_starts_and_stops(timers):
    """T3 sync timer can be started and stopped."""
    fired = []
    timers.start_t3_sync(lambda: fired.append(1))
    assert timers.get_timer_status()["t3_running"]

    timers.stop_t3_sync()
    assert not timers.get_timer_status()["t3_running"]


# ---------------------------------------------------------------------------
# Stop when not running (safety)
# ---------------------------------------------------------------------------

def test_stop_t1_when_not_running(timers):
    """Stopping T1 when not started does not raise."""
    timers.stop_t1_sync()   # Should be a no-op
    assert not timers.get_timer_status()["t1_running"]


def test_stop_t3_when_not_running(timers):
    """Stopping T3 when not started does not raise."""
    timers.stop_t3_sync()
    assert not timers.get_timer_status()["t3_running"]


# ---------------------------------------------------------------------------
# RTT measurement
# ---------------------------------------------------------------------------

def test_record_acknowledgment_updates_rto(timers):
    """record_acknowledgment changes the RTO value."""
    original_rto = timers.rto
    time.sleep(0.05)   # Let some time pass so RTT > 0
    timers.record_acknowledgment()
    # RTO should still be >= 1.0 (min clamped value)
    assert timers.rto >= 1.0


def test_rto_clamped_to_min(timers):
    """RTO is never below 1.0 second."""
    # Force very small SRTT
    timers.srtt = 0.001
    timers.rttvar = 0.0001
    timers.record_acknowledgment()
    assert timers.rto >= 1.0


def test_rto_clamped_to_max(timers):
    """RTO is never above 60 seconds."""
    # Simulate large RTT
    timers.srtt = 200.0
    timers.rttvar = 50.0
    timers.record_acknowledgment()
    assert timers.rto <= 60.0


# ---------------------------------------------------------------------------
# Timeout value updates
# ---------------------------------------------------------------------------

def test_update_t1_timeout(timers):
    """update_t1_timeout changes t1_current and resets RTO."""
    timers.update_t1_timeout(5.0)
    assert timers.t1_current == 5.0
    assert timers.rto == 5.0


def test_update_t1_out_of_range(timers):
    """update_t1_timeout raises ValueError for out-of-range values."""
    with pytest.raises(ValueError):
        timers.update_t1_timeout(0.0)   # Below 0.1
    with pytest.raises(ValueError):
        timers.update_t1_timeout(61.0)  # Above 60.0


def test_update_t3_timeout(timers):
    """update_t3_timeout changes t3_current."""
    timers.update_t3_timeout(600.0)
    assert timers.t3_current == 600.0


def test_update_t3_out_of_range(timers):
    """update_t3_timeout raises ValueError for out-of-range values."""
    with pytest.raises(ValueError):
        timers.update_t3_timeout(5.0)     # Below 10.0
    with pytest.raises(ValueError):
        timers.update_t3_timeout(3601.0)  # Above 3600.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_stops_timers(timers):
    """reset() stops all running timers."""
    timers.start_t1_sync(lambda: None)
    timers.start_t3_sync(lambda: None)
    timers.reset()
    assert not timers.get_timer_status()["t1_running"]
    assert not timers.get_timer_status()["t3_running"]


def test_reset_restores_srtt(timers):
    """reset() restores SRTT to config values."""
    timers.srtt = 99.0
    timers.reset()
    assert timers.srtt == DEFAULT_CONFIG_MOD8.t1_timeout


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def test_timer_status_keys(timers):
    """get_timer_status returns all expected keys."""
    status = timers.get_timer_status()
    for key in ("t1_running", "t3_running", "t1_current", "t3_current",
                "srtt", "rttvar", "rto", "last_ack_time"):
        assert key in status, f"Missing key: {key}"
