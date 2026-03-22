# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
tests/test_config.py

Unit tests for the AX25Config class and pre-built configuration objects.

Covers:
- Valid configuration creation
- Field range validation
- ConfigurationError for out-of-range values
- Modulo-window consistency
- Pre-built config objects
"""

import pytest

from pyax25_22.core.config import (
    AX25Config,
    DEFAULT_CONFIG_MOD8,
    DEFAULT_CONFIG_MOD128,
    CONFIG_APRS,
    CONFIG_PACSAT_BROADCAST,
)
from pyax25_22.core.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Valid configurations
# ---------------------------------------------------------------------------

def test_default_mod8():
    """DEFAULT_CONFIG_MOD8 is valid and has the right values."""
    cfg = DEFAULT_CONFIG_MOD8
    assert cfg.modulo == 8
    assert cfg.window_size == 7
    assert cfg.max_frame == 256
    assert cfg.t1_timeout == 10.0
    assert cfg.t3_timeout == 300.0
    assert cfg.retry_count == 10


def test_default_mod128():
    """DEFAULT_CONFIG_MOD128 is valid and has the right values."""
    cfg = DEFAULT_CONFIG_MOD128
    assert cfg.modulo == 128
    assert 1 <= cfg.window_size <= 127


def test_config_aprs():
    """CONFIG_APRS is valid."""
    assert CONFIG_APRS.modulo == 8
    assert CONFIG_APRS.window_size >= 1


def test_config_pacsat():
    """CONFIG_PACSAT_BROADCAST has T1=0 and retry_count=0."""
    assert CONFIG_PACSAT_BROADCAST.t1_timeout == 0.0
    assert CONFIG_PACSAT_BROADCAST.retry_count == 0


def test_create_mod8_min_window():
    """Minimum window_size=1 is valid for modulo 8."""
    cfg = AX25Config(modulo=8, window_size=1)
    assert cfg.window_size == 1


def test_create_mod8_max_window():
    """Maximum window_size=7 is valid for modulo 8."""
    cfg = AX25Config(modulo=8, window_size=7)
    assert cfg.window_size == 7


def test_create_mod128_min_window():
    """Minimum window_size=1 is valid for modulo 128."""
    cfg = AX25Config(modulo=128, window_size=1)
    assert cfg.window_size == 1


def test_create_mod128_max_window():
    """Maximum window_size=127 is valid for modulo 128."""
    cfg = AX25Config(modulo=128, window_size=127)
    assert cfg.window_size == 127


def test_frozen():
    """AX25Config is immutable -- assignment raises AttributeError."""
    cfg = DEFAULT_CONFIG_MOD8
    with pytest.raises(AttributeError):
        cfg.modulo = 128   # type: ignore[misc]


# ---------------------------------------------------------------------------
# Invalid configurations
# ---------------------------------------------------------------------------

def test_bad_modulo():
    """Modulo values other than 8 or 128 raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(modulo=16)
    with pytest.raises(ConfigurationError):
        AX25Config(modulo=0)


def test_window_too_large_mod8():
    """window_size > 7 with modulo 8 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(modulo=8, window_size=8)


def test_window_too_small():
    """window_size < 1 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(modulo=8, window_size=0)


def test_window_too_large_mod128():
    """window_size > 127 with modulo 128 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(modulo=128, window_size=128)


def test_max_frame_too_small():
    """max_frame < 1 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(max_frame=0)


def test_max_frame_too_large():
    """max_frame > 4096 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(max_frame=4097)


def test_t1_negative():
    """Negative t1_timeout raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(t1_timeout=-1.0)


def test_t1_too_large():
    """t1_timeout > 300.0 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(t1_timeout=301.0)


def test_t3_too_small():
    """t3_timeout < 10.0 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(t3_timeout=9.9)


def test_t3_too_large():
    """t3_timeout > 3600.0 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(t3_timeout=3601.0)


def test_retry_count_negative():
    """Negative retry_count raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(retry_count=-1)


def test_retry_count_too_large():
    """retry_count > 255 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(retry_count=256)


def test_persistence_negative():
    """Negative persistence raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(persistence=-1)


def test_persistence_too_large():
    """persistence > 255 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(persistence=256)


def test_slot_time_too_small():
    """slot_time < 0.01 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(slot_time=0.001)


def test_slot_time_too_large():
    """slot_time > 1.0 raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        AX25Config(slot_time=1.1)
