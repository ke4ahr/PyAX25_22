# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.config.py

Configuration management for AX.25 v2.2 protocol parameters.

Provides immutable, validated configuration objects with defaults compliant
with the AX.25 v2.2 specification (July 1998).

All parameters are validated on creation to prevent invalid runtime states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import logging

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AX25Config:
    """
    Immutable AX.25 configuration object.

    All values are validated against AX.25 v2.2 specification limits.
    """

    # Modulo operation: 8 or 128
    modulo: Literal[8, 128] = 8

    # N1 - Maximum I-field length in bytes
    max_frame: int = 256

    # k - Maximum number of outstanding I-frames (window size)
    window_size: int = 7

    # T1 - Acknowledgment timer in seconds
    t1_timeout: float = 10.0

    # T3 - Idle channel probe timer in seconds
    t3_timeout: float = 300.0

    # N2 - Maximum number of retries
    retry_count: int = 10

    # TXDELAY - Transmitter keyup delay in seconds
    tx_delay: float = 0.3

    # TXTTAIL - Transmitter keydown tail in seconds
    tx_tail: float = 0.05

    # Persistence and slot time for CSMA (optional, for future use)
    persistence: int = 63      # p = 63/256 ≈ 25%
    slot_time: float = 0.1     # Slot time in seconds

    def __post_init__(self) -> None:
        """
        Validate all parameters against AX.25 v2.2 limits.

        Raises ConfigurationError if any parameter is invalid.
        """
        # Modulo validation
        if self.modulo not in (8, 128):
            raise ConfigurationError(f"modulo must be 8 or 128, got {self.modulo}")

        # Window size limits
        if self.modulo == 8 and not (1 <= self.window_size <= 7):
            raise ConfigurationError(
                f"window_size must be 1-7 for modulo 8, got {self.window_size}"
            )
        if self.modulo == 128 and not (1 <= self.window_size <= 127):
            raise ConfigurationError(
                f"window_size must be 1-127 for modulo 128, got {self.window_size}"
            )

        # Frame size limits (practical range)
        if not (1 <= self.max_frame <= 4096):
            raise ConfigurationError(
                f"max_frame must be 1-4096 bytes, got {self.max_frame}"
            )

        # Timer bounds
        if not (0.1 <= self.t1_timeout <= 300.0):
            raise ConfigurationError(
                f"t1_timeout must be 0.1-300.0 seconds, got {self.t1_timeout}"
            )
        if not (10.0 <= self.t3_timeout <= 3600.0):
            raise ConfigurationError(
                f"t3_timeout must be 10.0-3600.0 seconds, got {self.t3_timeout}"
            )

        # Retry count
        if not (1 <= self.retry_count <= 255):
            raise ConfigurationError(
                f"retry_count must be 1-255, got {self.retry_count}"
            )

        # CSMA parameters
        if not (0 <= self.persistence <= 255):
            raise ConfigurationError(
                f"persistence must be 0-255, got {self.persistence}"
            )
        if not (0.01 <= self.slot_time <= 1.0):
            raise ConfigurationError(
                f"slot_time must be 0.01-1.0 seconds, got {self.slot_time}"
            )

        logger.info(
            f"AX25Config validated: modulo={self.modulo}, k={self.window_size}, "
            f"N1={self.max_frame}, T1={self.t1_timeout}s, N2={self.retry_count}"
        )


# Standard predefined configurations

DEFAULT_CONFIG_MOD8 = AX25Config(
    modulo=8,
    max_frame=256,
    window_size=7,
    t1_timeout=10.0,
    t3_timeout=300.0,
    retry_count=10,
)

DEFAULT_CONFIG_MOD128 = AX25Config(
    modulo=128,
    max_frame=2048,
    window_size=63,
    t1_timeout=15.0,
    t3_timeout=300.0,
    retry_count=20,
)

# Common configurations for specific use cases

CONFIG_APRS = AX25Config(
    modulo=8,
    max_frame=256,
    window_size=4,
    t1_timeout=5.0,
    retry_count=5,
    tx_delay=0.3,
)

CONFIG_PACSAT_BROADCAST = AX25Config(
    modulo=8,
    max_frame=256,
    window_size=1,  # Unconnected UI frames
    t1_timeout=10.0,  # No retransmission; was 0.0, causes errors on import.
    retry_count=1,    # was set to zero. I get it but this is nuts.
    tx_delay=0.3,   # gonna add this here so it stops complaining abt PACSAT.
)
