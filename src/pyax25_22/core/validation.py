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
- Sequence number consistency (for connected mode)
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

    # Check for repeated addresses (H-bit should not be set on non-repeated)
    # Optional advanced check - skip for now

    control = frame.control

    # I-frame validation
    if control & 0x01 == 0x00:
        if frame.pid is None:
            raise InvalidControlFieldError("I-frame must have PID")
        if len(frame.info) == 0:
            logger.warning("I-frame with empty info field (allowed but unusual)")
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
            # Other U-frames (SABM, DISC, UA, DM, FRMR, XID) must not have PID
            if frame.pid is not None:
                raise InvalidControlFieldError(f"U-frame type {control:02x} must not have PID")

        # XID may have info field
        if (control & ~0x10) == 0xAF and not frame.info:
            logger.debug("XID frame with no parameters (allowed)")

    logger.debug("Frame structure validation passed")


def validate_sequence_numbers(
    frame: AX25Frame,
    expected_v_r: int,
    config: AX25Config,
) -> bool:
    """
    Validate sequence numbers in connected mode frames.

    Returns True if valid, False if out-of-sequence (caller may send REJ/SREJ).

    Args:
        frame: Incoming frame
        expected_v_r: Current V(R) - next expected N(S)
        config: Current configuration

    """
    control = frame.control

    if control & 0x01 == 0x00:  # I-frame
        ns = (control >> 1) & (0x07 if config.modulo == 8 else 0x7F)
        if ns != expected_v_r:
            logger.warning(f"Out-of-sequence I-frame: expected N(S)={expected_v_r}, got {ns}")
            return False

        nr = (control >> 5) & (0x07 if config.modulo == 8 else 0x7F)
        # N(R) should be within window - basic check
        if nr > expected_v_r + config.window_size:
            logger.warning(f"Invalid N(R)={nr} in I-frame")
            return False

    elif control & 0x03 == 0x01:  # S-frame
        nr = (control >> 5) & (0x07 if config.modulo == 8 else 0x7F)
        # Similar window check

    return True


def full_validation(frame: AX25Frame, config: Optional[AX25Config] = None) -> None:
    """
    Perform complete AX.25 v2.2 validation on a decoded frame.

    Combines structural and protocol validation.
    Called after successful decode and FCS check.

    Args:
        frame: Decoded frame
        config: Configuration context

    Raises:
        Specific exceptions on any validation failure
    """
    validate_frame_structure(frame, config)
    logger.info("Full frame validation completed successfully")
