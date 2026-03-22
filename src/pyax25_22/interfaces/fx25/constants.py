# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
interfaces/fx25/constants.py

Constants for the FX.25 Forward Error Correction protocol.

FX.25 wraps an AX.25 frame with a Reed-Solomon FEC block, allowing receivers
to automatically correct bit errors.

Frame structure:
    [PREAMBLE bytes]        -- 0x00 padding before tag (optional)
    [CONNECT TAG: 8 bytes]  -- identifies RS parameters + provides synchronisation
    [AX.25 FRAME DATA]      -- original encoded AX.25 frame bytes
    [PAD BYTES]             -- 0x00 fill to reach block size
    [CHECK BYTES]           -- RS check (parity) symbols

Reference: TAPR FX.25 Specification, v0.04, 2006.
Implemented to match Dire Wolf's FX.25 implementation for interoperability.
"""

# ---------------------------------------------------------------------------
# GF(2^8) field parameters for the Reed-Solomon code
# ---------------------------------------------------------------------------

# Generator polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1 = 0x11D
# Same as used by CCSDS and Dire Wolf.
GF_POLY = 0x11D

# First consecutive root of the RS generator polynomial (alpha^FCR)
FCR = 1

# Primitive element of GF(2^8): alpha = 2
PRIM = 2

# ---------------------------------------------------------------------------
# Connect tag table
#
# Each entry: (check_bytes, data_bytes, total_bytes, ctag_32bit)
#
# The 8-byte connect tag on the wire is:
#     ctag_32bit as 4 bytes (little-endian)
#     ctag_32bit as 4 bytes (little-endian) XOR 0xFF each byte
#
# ctag_32bit values from the TAPR FX.25 specification.
# ---------------------------------------------------------------------------

#: Mapping from tag_id (0x01-0x0A) to (check_bytes, data_bytes, total_bytes, ctag_32)
FX25_TAGS = {
    0x01: (16,  239, 255, 0xB74DB7DF),   # RS(255,239,8)   -- up to 8 errors correctable
    0x02: (32,  223, 255, 0x26FF60A6),   # RS(255,223,16)  -- up to 16 errors
    0x03: (64,  191, 255, 0xC7DC0508),   # RS(255,191,32)  -- up to 32 errors
    0x04: (16,  128, 144, 0x6EB8DBB4),   # RS(144,128,8)   -- medium frame, 8 errors
    0x05: (32,  128, 160, 0x8F9BBE1A),   # RS(160,128,16)  -- medium frame, 16 errors
    0x06: (64,  128, 192, 0x1E29694B),   # RS(192,128,32)  -- medium frame, 32 errors
    0x07: (16,   64,  80, 0xFF0A0CE5),   # RS(80,64,8)     -- small frame, 8 errors
    0x08: (32,   64,  96, 0x1122EE17),   # RS(96,64,16)    -- small frame, 16 errors
    0x09: (16,   32,  48, 0xF001DBB9),   # RS(48,32,8)     -- tiny frame, 8 errors
    0x0A: (32,   32,  64, 0xB0228847),   # RS(64,32,16)    -- tiny frame, 16 errors
}

# Reverse lookup: ctag_32 value -> tag_id
_CTAG_TO_ID = {ctag: tid for tid, (_, _, _, ctag) in FX25_TAGS.items()}

# Connect tag wire size (bytes)
CTAG_SIZE = 8

# Preamble byte value
PREAMBLE_BYTE = 0x00

# Default tag: RS(255,239) -- most widely supported
DEFAULT_TAG = 0x01

# Maximum AX.25 frame bytes that can fit in any FX.25 block (tag 0x03, 191 data bytes)
MAX_AX25_FRAME = 239   # tag 0x01 supports up to 239 bytes of AX.25 data


def tag_info(tag_id: int):
    """Return (check_bytes, data_bytes, total_bytes) for a tag.

    Args:
        tag_id: FX.25 tag ID (0x01-0x0A).

    Returns:
        Tuple (check_bytes, data_bytes, total_bytes).

    Raises:
        KeyError: If tag_id is not in the table.
    """
    check, data, total, _ = FX25_TAGS[tag_id]
    return check, data, total


def ctag_to_wire(tag_id: int) -> bytes:
    """Build the 8-byte connect tag wire representation.

    The wire format is: ctag_32 (4 bytes LE) + ctag_32 XOR 0xFF (4 bytes LE).

    Args:
        tag_id: FX.25 tag ID (0x01-0x0A).

    Returns:
        8 bytes to transmit before the FEC block.

    Raises:
        KeyError: If tag_id is unknown.
    """
    _, _, _, ctag = FX25_TAGS[tag_id]
    low = ctag.to_bytes(4, "little")
    high = bytes(b ^ 0xFF for b in low)
    return low + high


def wire_to_tag_id(wire: bytes) -> int:
    """Decode the tag ID from 8 received bytes.

    Checks both the main and complemented halves for consistency.
    The Hamming-distance-4 property of the tag patterns allows detection
    of tag errors even with 1-3 corrupted bits.

    Args:
        wire: Exactly 8 bytes from the received stream.

    Returns:
        Tag ID (0x01-0x0A).

    Raises:
        ValueError: If wire is not 8 bytes, or no valid tag is recognised.
    """
    if len(wire) != CTAG_SIZE:
        raise ValueError(f"Connect tag must be {CTAG_SIZE} bytes, got {len(wire)}")

    low = wire[:4]
    high = wire[4:]

    # Verify complement relationship
    if bytes(b ^ 0xFF for b in low) != high:
        raise ValueError(
            "Connect tag complement check failed -- tag may be corrupted"
        )

    ctag = int.from_bytes(low, "little")
    if ctag not in _CTAG_TO_ID:
        raise ValueError(f"Unknown connect tag value: 0x{ctag:08X}")

    return _CTAG_TO_ID[ctag]


def choose_tag(ax25_len: int) -> int:
    """Choose the smallest FX.25 tag that can fit the given AX.25 frame length.

    Always uses 16 check bytes (the minimum -- up to 8 correctable errors).
    Returns the tag with the smallest block size that still fits the frame.

    Args:
        ax25_len: Length of the AX.25 frame in bytes.

    Returns:
        Tag ID (0x01-0x0A).

    Raises:
        ValueError: If the frame is too large for any tag (> 239 bytes).
    """
    # Tags sorted by data capacity (smallest first for efficiency)
    for tag_id in (0x09, 0x0A, 0x07, 0x08, 0x04, 0x05, 0x06, 0x01, 0x02, 0x03):
        _, data, _, _ = FX25_TAGS[tag_id]
        if ax25_len <= data:
            return tag_id
    raise ValueError(
        f"AX.25 frame too large for FX.25: {ax25_len} bytes "
        f"(max {MAX_AX25_FRAME})"
    )
