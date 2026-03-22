# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/fx25/encoder.py

FX.25 encoder: wraps an AX.25 frame with Reed-Solomon FEC.

Wire format produced:
    [PREAMBLE bytes]        -- optional 0x00 preamble bytes
    [CONNECT TAG: 8 bytes]  -- identifies RS parameters, provides sync
    [AX.25 FRAME DATA]      -- original AX.25 frame bytes
    [PAD BYTES]             -- 0x00 fill to reach block data_bytes size
    [CHECK BYTES]           -- RS check symbols (ncheck bytes)

Usage::

    encoder = FX25Encoder()
    fx25_bytes = encoder.encode(ax25_frame_bytes)

The encoder automatically selects the smallest FX.25 tag that fits the
AX.25 frame.  Use tag_id= to force a specific tag.
"""

import logging
from typing import Optional

from .constants import (
    FX25_TAGS, DEFAULT_TAG, PREAMBLE_BYTE, ctag_to_wire, choose_tag, tag_info,
)
from .rs import rs_encode

logger = logging.getLogger(__name__)


class FX25Encoder:
    """Encode AX.25 frame bytes as an FX.25 block.

    Args:
        preamble_len: Number of 0x00 preamble bytes to prepend (default 0).
        default_tag: Default tag ID to use when not specified in encode().
    """

    def __init__(
        self,
        preamble_len: int = 0,
        default_tag: int = DEFAULT_TAG,
    ) -> None:
        self.preamble_len = preamble_len
        self.default_tag = default_tag

    def encode(
        self,
        ax25_frame: bytes,
        tag_id: Optional[int] = None,
    ) -> bytes:
        """Encode an AX.25 frame as an FX.25 block.

        Args:
            ax25_frame: Raw AX.25 frame bytes (without KISS framing).
            tag_id: FX.25 tag ID (0x01-0x0A) to use.  If None, the smallest
                tag that fits the frame is chosen automatically.

        Returns:
            Complete FX.25 block: preamble + connect_tag + data + pad + check.

        Raises:
            ValueError: If the frame is too large for any FX.25 tag (> 239 bytes).
        """
        if tag_id is None:
            tag_id = choose_tag(len(ax25_frame))
        else:
            check_bytes, data_bytes, _ = tag_info(tag_id)
            if len(ax25_frame) > data_bytes:
                raise ValueError(
                    f"AX.25 frame ({len(ax25_frame)} bytes) too large for "
                    f"tag 0x{tag_id:02X} (max {data_bytes} bytes)"
                )

        check_bytes, data_bytes, total_bytes = tag_info(tag_id)

        # Pad frame to data_bytes with trailing zeros
        pad_len = data_bytes - len(ax25_frame)
        padded = ax25_frame + bytes(pad_len)

        # RS encode: padded data + check symbols
        codeword = rs_encode(padded, check_bytes)

        # Wire format
        preamble = bytes([PREAMBLE_BYTE] * self.preamble_len)
        connect_tag = ctag_to_wire(tag_id)

        result = preamble + connect_tag + codeword
        logger.debug(
            "FX25 encode: tag=0x%02X ax25=%d pad=%d check=%d total=%d",
            tag_id, len(ax25_frame), pad_len, check_bytes, len(result),
        )
        return result
