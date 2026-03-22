# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.validation -- Deep AX.25 frame validity checks.

After a frame is decoded and its FCS is confirmed, this module checks
that the frame makes sense according to the AX.25 v2.2 rules:

  - Are there too many digipeaters?
  - Does the control field match one of the known frame types?
  - Does the PID appear where it should and only where it should?
  - Is the information field short enough to fit within N1?
  - Is the frame type allowed in the current state?

These checks go beyond what the basic decoder does. They are the
equivalent of checking not just that a letter arrived, but that the
letter is addressed correctly and the contents make sense.

Each function raises a specific exception with a clear message so the
caller knows exactly what went wrong and can log or handle it correctly.

Compliant with AX.25 v2.2 Section 4 (Data Link Layer).
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


def validate_frame_structure(
    frame: AX25Frame,
    config: Optional[AX25Config] = None,
) -> None:
    """Check the structure of a decoded AX.25 frame for compliance.

    Validates the following rules from AX.25 v2.2:
      1. The number of digipeaters must not exceed 8.
      2. I-frames must have a PID and their info field must fit in N1.
      3. S-frames (RR, RNR, REJ, SREJ) must not have a PID or info.
      4. UI-frames must have a PID.
      5. Other U-frames must not have a PID.

    This is meant to be called after ``AX25Frame.decode()`` succeeds.
    It does not re-check the FCS (that is done by decode).

    Args:
        frame: A fully decoded AX25Frame to check.
        config: Optional configuration to use for the N1 check. If not
            given, ``frame.config`` is used.

    Raises:
        InvalidAddressError: If there are more than 8 digipeaters.
        InvalidControlFieldError: If the PID is present or absent when
            it should not be.
        FrameError: If the information field is longer than N1, or if
            an S-frame has an information field.

    Example::

        frame = AX25Frame.decode(raw_bytes)
        validate_frame_structure(frame)
        # Passes silently if everything is OK
    """
    cfg = config or frame.config

    logger.debug(
        "validate_frame_structure: control=0x%02X pid=%s info=%d bytes digis=%d",
        frame.control, frame.pid, len(frame.info), len(frame.digipeaters),
    )

    # Rule 1: AX.25 allows at most 8 digipeaters in the address field
    if len(frame.digipeaters) > 8:
        raise InvalidAddressError(
            f"Frame has {len(frame.digipeaters)} digipeaters; "
            f"AX.25 v2.2 allows at most 8"
        )

    control = frame.control

    # --- I-frame (bit 0 = 0) ---
    if control & 0x01 == 0x00:
        if frame.pid is None:
            raise InvalidControlFieldError(
                "I-frame (control=0x{:02X}) must have a PID byte".format(control)
            )
        if len(frame.info) > cfg.max_frame:
            raise FrameError(
                f"I-frame information field is {len(frame.info)} bytes, "
                f"which exceeds N1={cfg.max_frame}"
            )
        logger.debug(
            "validate_frame_structure: I-frame OK (pid=0x%02X info=%d N1=%d)",
            frame.pid, len(frame.info), cfg.max_frame,
        )

    # --- S-frame (bits 1-0 = 01) ---
    elif control & 0x03 == 0x01:
        if frame.pid is not None:
            raise InvalidControlFieldError(
                "S-frame (control=0x{:02X}) must not have a PID byte "
                "(got pid=0x{:02X})".format(control, frame.pid)
            )
        if frame.info:
            raise FrameError(
                f"S-frame (control=0x{control:02X}) must not have an "
                f"information field (got {len(frame.info)} bytes)"
            )
        logger.debug(
            "validate_frame_structure: S-frame OK (type=%s)",
            {0x01: "RR", 0x05: "RNR", 0x09: "REJ", 0x0D: "SREJ"}.get(
                control & 0x0F, f"0x{control & 0x0F:02X}"
            ),
        )

    # --- U-frame (bits 1-0 = 11) ---
    elif control & 0x03 == 0x03:
        # UI frame (control byte without P/F is 0x03)
        is_ui = (control & ~0x10) == 0x03

        if is_ui:
            if frame.pid is None:
                raise InvalidControlFieldError(
                    "UI frame (control=0x{:02X}) must have a PID byte".format(control)
                )
            if len(frame.info) > cfg.max_frame:
                raise FrameError(
                    f"UI frame information field is {len(frame.info)} bytes, "
                    f"which exceeds N1={cfg.max_frame}"
                )
            logger.debug(
                "validate_frame_structure: UI-frame OK (pid=0x%02X info=%d)",
                frame.pid, len(frame.info),
            )
        else:
            # All other U-frames (SABM, SABME, DISC, UA, DM, XID, FRMR, TEST)
            # must not carry a PID
            if frame.pid is not None:
                raise InvalidControlFieldError(
                    "U-frame (control=0x{:02X}) must not have a PID byte "
                    "(got pid=0x{:02X})".format(control, frame.pid)
                )
            logger.debug(
                "validate_frame_structure: U-frame OK (control=0x%02X)", control
            )

    else:
        raise InvalidControlFieldError(
            f"Control field 0x{control:02X} does not match any known frame type "
            f"(I, S, or U)"
        )

    logger.debug("validate_frame_structure: validation passed")


def full_validation(
    frame: AX25Frame,
    config: Optional[AX25Config] = None,
) -> None:
    """Run all validation checks on a decoded AX.25 frame.

    This is a convenience function that calls all available validators
    in order. Use this when you want the strictest possible compliance
    checking after decoding a frame.

    Currently calls:
      - ``validate_frame_structure``

    More checks may be added in future versions.

    Args:
        frame: A fully decoded AX25Frame to validate.
        config: Optional configuration context.

    Raises:
        The same exceptions as ``validate_frame_structure``.

    Example::

        frame = AX25Frame.decode(raw_bytes)
        full_validation(frame, config=my_config)
        # Process the frame only after this succeeds
    """
    logger.debug("full_validation: starting validation for frame control=0x%02X", frame.control)
    validate_frame_structure(frame, config)
    logger.info("full_validation: all checks passed")


__all__ = [
    "validate_frame_structure",
    "full_validation",
]
