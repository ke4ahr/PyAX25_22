# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.config -- Settings for the AX.25 protocol.

This file holds the AX25Config class, which stores all the settings
that control how AX.25 connections behave.

Think of AX25Config like a recipe card. It tells the library:
  - How big can packets be?
  - How many packets can be in flight at once?
  - How long to wait before retrying?
  - How many times to retry?

Once created, the settings cannot be changed. This is on purpose --
it prevents bugs caused by changing settings while a connection is open.

Several ready-made setting objects are provided at the bottom of this
file for common situations like APRS monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import logging

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AX25Config:
    """A set of settings that controls AX.25 protocol behavior.

    All settings are checked when this object is created. If any
    setting is out of range, a ConfigurationError is raised right away
    instead of causing strange problems later.

    Once created, none of the settings can be changed. Make a new
    AX25Config object if you need different settings.

    Attributes:
        modulo: How sequence numbers wrap around. Must be 8 or 128.
            Use 8 for most connections. Use 128 for faster connections
            that can have many packets in flight at once.
        max_frame: The largest allowed size (in bytes) for the
            information part of a data frame. Called N1 in the spec.
            Range: 1 to 4096 bytes. Default: 256.
        window_size: How many unacknowledged frames can be "in flight"
            at the same time. Called k in the spec.
            For modulo 8: 1 to 7. For modulo 128: 1 to 127.
            Default: 7 (modulo 8).
        t1_timeout: How many seconds to wait for an acknowledgment
            before retrying. Called T1 in the spec.
            Range: 0.0 to 300.0 seconds. Default: 10.0.
        t3_timeout: How many seconds an idle connection can sit quiet
            before we send a probe to see if the other side is still
            there. Called T3 in the spec.
            Range: 10.0 to 3600.0 seconds. Default: 300.0.
        retry_count: How many times to retry before giving up and
            declaring the connection broken. Called N2 in the spec.
            Range: 0 to 255. Default: 10.
        tx_delay: How many seconds to wait after keying the transmitter
            before sending data. Gives the radio time to start
            transmitting. Called TXDELAY. Default: 0.3.
        tx_tail: How many seconds to keep the transmitter on after the
            last byte is sent. Called TXTAIL. Default: 0.05.
        persistence: Controls how aggressively to grab the channel when
            there are multiple stations waiting. Range: 0 to 255.
            Higher = more aggressive. Default: 63 (about 25%).
        slot_time: How many seconds to wait between checking if the
            channel is free. Called SLOTTIME. Range: 0.01 to 1.0.
            Default: 0.1 seconds.

    Raises:
        ConfigurationError: If any setting value is not allowed.

    Example::

        # Standard modulo-8 connection
        cfg = AX25Config(modulo=8, window_size=4, t1_timeout=15.0)

        # High-throughput modulo-128 connection
        cfg = AX25Config(modulo=128, window_size=63, t1_timeout=15.0)
    """

    # Sequence number modulo (N1 in spec)
    modulo: Literal[8, 128] = 8

    # Maximum information field size in bytes (N1)
    max_frame: int = 256

    # Window size -- max outstanding I-frames (k)
    window_size: int = 7

    # T1 -- acknowledgment timer in seconds
    t1_timeout: float = 10.0

    # T3 -- idle channel probe timer in seconds
    t3_timeout: float = 300.0

    # N2 -- maximum retransmission count
    retry_count: int = 10

    # TXDELAY in seconds
    tx_delay: float = 0.3

    # TXTAIL in seconds
    tx_tail: float = 0.05

    # P-persistence (0-255) for CSMA
    persistence: int = 63

    # Slot time in seconds for CSMA
    slot_time: float = 0.1

    def __post_init__(self) -> None:
        """Check all settings when the object is first created.

        This runs automatically after __init__. It checks every setting
        against the rules in the AX.25 v2.2 specification. If anything
        is wrong, a ConfigurationError is raised immediately.

        Raises:
            ConfigurationError: If any setting is out of range or
                inconsistent with other settings.
        """
        logger.debug(
            "Validating AX25Config: modulo=%d, k=%d, N1=%d, "
            "T1=%.1fs, T3=%.1fs, N2=%d",
            self.modulo, self.window_size, self.max_frame,
            self.t1_timeout, self.t3_timeout, self.retry_count,
        )

        # Modulo must be exactly 8 or 128
        if self.modulo not in (8, 128):
            raise ConfigurationError(
                f"modulo must be 8 or 128, got {self.modulo}"
            )

        # Window size limits depend on which modulo is chosen
        if self.modulo == 8 and not (1 <= self.window_size <= 7):
            raise ConfigurationError(
                f"window_size must be 1-7 for modulo 8, got {self.window_size}"
            )
        if self.modulo == 128 and not (1 <= self.window_size <= 127):
            raise ConfigurationError(
                f"window_size must be 1-127 for modulo 128, got {self.window_size}"
            )

        # Frame size must be at least 1 byte and at most 4096 bytes
        if not (1 <= self.max_frame <= 4096):
            raise ConfigurationError(
                f"max_frame must be 1-4096 bytes, got {self.max_frame}"
            )

        # T1 must be 0 (disabled) or a positive number up to 5 minutes
        if not (0.0 <= self.t1_timeout <= 300.0):
            raise ConfigurationError(
                f"t1_timeout must be 0.0-300.0 seconds, got {self.t1_timeout}"
            )

        # T3 must be at least 10 seconds and at most 1 hour
        if not (10.0 <= self.t3_timeout <= 3600.0):
            raise ConfigurationError(
                f"t3_timeout must be 10.0-3600.0 seconds, got {self.t3_timeout}"
            )

        # Retry count must fit in one byte (0-255)
        if not (0 <= self.retry_count <= 255):
            raise ConfigurationError(
                f"retry_count must be 0-255, got {self.retry_count}"
            )

        # Persistence must fit in one byte (0-255)
        if not (0 <= self.persistence <= 255):
            raise ConfigurationError(
                f"persistence must be 0-255, got {self.persistence}"
            )

        # Slot time must be a reasonable fraction of a second
        if not (0.01 <= self.slot_time <= 1.0):
            raise ConfigurationError(
                f"slot_time must be 0.01-1.0 seconds, got {self.slot_time}"
            )

        logger.info(
            "AX25Config ready: modulo=%d, k=%d, N1=%d bytes, "
            "T1=%.1fs, T3=%.1fs, N2=%d retries",
            self.modulo, self.window_size, self.max_frame,
            self.t1_timeout, self.t3_timeout, self.retry_count,
        )


# ---------------------------------------------------------------------------
# Ready-made configuration objects for common uses
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_MOD8 = AX25Config(
    modulo=8,
    max_frame=256,
    window_size=7,
    t1_timeout=10.0,
    t3_timeout=300.0,
    retry_count=10,
)
"""Standard AX.25 modulo-8 configuration.

Good for most amateur packet radio connections where you do not need
high throughput. Works with all AX.25-capable TNCs.
"""

DEFAULT_CONFIG_MOD128 = AX25Config(
    modulo=128,
    max_frame=256,
    window_size=63,
    t1_timeout=15.0,
    t3_timeout=300.0,
    retry_count=20,
)
"""Extended AX.25 modulo-128 configuration.

Use this for high-throughput links where both sides support the
extended (modulo-128) mode. Allows up to 127 frames in flight at once.
Both sides must negotiate this mode using XID frames before connecting.
"""

CONFIG_APRS = AX25Config(
    modulo=8,
    max_frame=256,
    window_size=4,
    t1_timeout=5.0,
    t3_timeout=300.0,
    retry_count=5,
    tx_delay=0.3,
)
"""Settings tuned for APRS monitoring and beaconing.

Uses a smaller window and shorter timeout so beacons do not hold up
other traffic for long. A good starting point for APRS iGates and
APRS tracking applications.
"""

CONFIG_PACSAT_BROADCAST = AX25Config(
    modulo=8,
    max_frame=256,
    window_size=1,
    t1_timeout=0.0,
    t3_timeout=300.0,
    retry_count=0,
    tx_delay=0.3,
)
"""Settings for PACSAT-style one-way broadcast (no acknowledgments).

PACSAT satellites broadcast files to the ground without waiting for
acknowledgments. T1 is set to 0 (no retry timer) and retry_count is
0 because there is no one to acknowledge the frames.
"""
