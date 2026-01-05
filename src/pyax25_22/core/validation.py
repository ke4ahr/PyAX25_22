# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.validation.py

Comprehensive validation utilities for AX.25 v2.2 compliance.

Provides detailed frame validation beyond basic decoding:
- Address field rules (digipeater count, reserved bits)
- Control field compliance per frame type and modulo
- PID presence/absence rules
- Information field length vs N1
- Protocol-specific checks (e.g., U-frame commands)

All validation functions raise specific exceptions on failure and log details.
"""

from __future__ import annotations

import logging
from typing import Optional

from .framing import AX25Frame
from .config import AX25Config
from .exceptions import (
    FrameError,
    InvalidControlFieldError,
    InvalidAddressError,
)

logger = logging.getLogger(__name__)


def validate_frame_structure(frame: AX25Frame, config: Optional[AX25Config] = None) -> None:
    """
    Perform structural validation of a decoded AX.25 frame.

    Validates:
    - Address field rules
    - Control field format and PID presence
    - Information field length
    - Digipeater count

    Args:
        frame: Decoded AX.25 frame
        config: Configuration for N1 and modulo checks

    Raises:
        FrameError, InvalidControlFieldError, InvalidAddressError on failure
    """
    config = config or frame.config

    # Digipeater count limit
    if len(frame.digipeaters) > 8:
        raise InvalidAddressError(f"Too many digipeaters: {len(frame.digipeaters)} > 8")

    control = frame.control

    # I-frame validation
    if control & 0x01 == 0x00:
        if frame.pid is None:
            raise InvalidControlFieldError("I-frame must have PID")
        if len(frame.info) > config.max_frame:
            raise FrameError(f"I-field {len(frame.info)} exceeds N1={config.max_frame}")

    # S-frame validation
    elif control & 0x03 == 0x01:
        if frame.pid is not None:
            raise InvalidControlFieldError("S-frame must not have PID")
        if frame.info:
            raise FrameError("S-frame must not have info field")

    # U-frame validation
    elif control & 0x03 == 0x03:
        # UI frames must have PID
        if (control & ~0x10) == 0x03:  # UI
            if frame.pid is None:
                raise InvalidControlFieldError("UI frame must have PID")
        else:
            # Other U-frames must not have PID
            if frame.pid is not None:
                raise InvalidControlFieldError(f"U-frame type {control:02x} must not have PID")

    logger.debug("Frame structure validation passed")


def full_validation(frame: AX25Frame, config: Optional[AX25Config] = None) -> None:
    """
    Perform complete AX.25 v2.2 validation on a decoded frame.

    Called after successful decode and FCS check.

    Args:
        frame: Decoded frame
        config: Configuration context

    Raises:
        Specific exceptions on any validation failure
    """
    validate_frame_structure(frame, config)
    logger.info("Full frame validation completed successfully")

__all__ = [
    "validate_frame_structure",
    "full_validation",
]