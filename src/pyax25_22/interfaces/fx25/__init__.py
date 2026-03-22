# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/fx25/__init__.py

FX.25 -- Forward Error Correction for AX.25 (Reed-Solomon FEC).

FX.25 wraps an AX.25 frame with a Reed-Solomon FEC block, allowing receivers
to correct bit errors.  See TAPR FX.25 specification (fx25.pdf) for details.

Modules:
    constants.py  -- Correlation tag bytes and RS parameters
    rs.py         -- Reed-Solomon GF(2^8) encoder/decoder (pure Python)
    encoder.py    -- FX25Encoder: wraps AX.25 frame bytes with FEC
    decoder.py    -- FX25Decoder: strips wrapper, corrects errors, returns AX.25 frame
"""

from .encoder import FX25Encoder
from .decoder import FX25Decoder
from .constants import (
    FX25_TAGS,
    DEFAULT_TAG,
    MAX_AX25_FRAME,
    CTAG_SIZE,
    tag_info,
    ctag_to_wire,
    wire_to_tag_id,
    choose_tag,
)
from .rs import rs_encode, rs_decode

__all__ = [
    "FX25Encoder",
    "FX25Decoder",
    "FX25_TAGS",
    "DEFAULT_TAG",
    "MAX_AX25_FRAME",
    "CTAG_SIZE",
    "tag_info",
    "ctag_to_wire",
    "wire_to_tag_id",
    "choose_tag",
    "rs_encode",
    "rs_decode",
]
