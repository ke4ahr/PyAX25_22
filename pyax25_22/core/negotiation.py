# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.negotiation.py

XID Frame handling and parameter negotiation per AX.25 v2.2 Section 4.3.4.

Supports negotiation of:
- Modulo (8/128)
- Window size (k)
- Max frame size (N1)
- Selective Reject support

This module provides utilities to build and parse XID frames, as well as
negotiate final configuration parameters between peers.
"""

from __future__ import annotations

import struct
from typing import Dict
import logging

from .framing import AX25Frame
from .config import AX25Config
from .exceptions import NegotiationError

logger = logging.getLogger(__name__)

# XID Parameter Identifiers per AX.25 v2.2 spec
XID_MODULO = 0x01    # Modulo parameter (1 byte: 8 or 128)
XID_WINDOW = 0x02    # Window size k (1 byte)
XID_N1 = 0x03        # Max I-field length N1 (2 bytes)
XID_SREJ = 0x08      # Selective Reject support (1 byte: 0/1)

# Additional XID IDs for future expansion (optional per spec)
XID_RETRY = 0x04     # N2 retry count (1 byte)
XID_T1 = 0x05        # T1 timer multiplier (1 byte)
XID_T2 = 0x06        # T2 timer (1 byte)
XID_T3 = 0x07        # T3 timer (2 bytes)


def build_xid_frame(config: AX25Config) -> bytes:
    """
    Build the information field for an XID frame.

    Encodes supported parameters in Type-Length-Value (TLV) format
    as per AX.25 v2.2 Section 4.3.4.

    Args:
        config: Current local configuration to advertise

    Returns:
        Bytes for the XID information field

    Raises:
        ConfigurationError if config is invalid
    """
    params = bytearray()

    # Modulo (mandatory)
    params += struct.pack('BB', XID_MODULO, 1)
    params += struct.pack('B', config.modulo)

    # Window size k (mandatory)
    params += struct.pack('BB', XID_WINDOW, 1)
    params += struct.pack('B', config.window_size)

    # N1 max frame size (mandatory)
    params += struct.pack('BB', XID_N1, 2)
    params += struct.pack('<H', config.max_frame)

    # SREJ support (optional, but we advertise it)
    params += struct.pack('BB', XID_SREJ, 1)
    params += struct.pack('B', 1)  # 1 = supported

    # Retry count N2 (optional)
    params += struct.pack('BB', XID_RETRY, 1)
    params += struct.pack('B', config.retry_count)

    logger.info(f"Built XID frame with {len(params)} bytes of parameters")
    return bytes(params)


def parse_xid_frame(info: bytes) -> Dict[int, int]:
    """
    Parse XID information field into parameter dictionary.

    Decodes TLV parameters and validates lengths.

    Args:
        info: XID information field bytes

    Returns:
        Dict of param_id: value

    Raises:
        NegotiationError if parsing fails (invalid length, truncation)
    """
    params: Dict[int, int] = {}
    offset = 0

    while offset + 2 <= len(info):
        param_id = info[offset]
        length = info[offset + 1]
        offset += 2

        if offset + length > len(info):
            raise NegotiationError(
                f"Truncated XID parameter {param_id}: expected {length} bytes, got {len(info) - offset}"
            )

        # Decode value based on expected length
        if length == 1:
            value = struct.unpack('B', info[offset:offset+length])[0]
        elif length == 2:
            value = struct.unpack('<H', info[offset:offset+length])[0]
        else:
            raise NegotiationError(f"Unsupported parameter length {length} for ID {param_id}")

        params[param_id] = value
        offset += length

    if offset != len(info):
        raise NegotiationError("Extra bytes after last parameter")

    logger.info(f"Parsed XID frame with {len(params)} parameters: {params}")
    return params


def negotiate_config(local: AX25Config, remote_params: Dict[int, int]) -> AX25Config:
    """
    Negotiate final configuration from local config and remote XID parameters.

    Rules per AX.25 v2.2:
    - Modulo must match (no fallback)
    - Take minimum of window size (k)
    - Take minimum of max frame (N1)
    - SREJ only if both support
    - Other params take remote if provided, else local

    Args:
        local: Local configuration
        remote_params: Parsed parameters from peer XID

    Returns:
        Negotiated AX25Config

    Raises:
        NegotiationError if incompatible (e.g., modulo mismatch)
    """
    # Start with local defaults
    negotiated_values = {
        'modulo': local.modulo,
        'window_size': local.window_size,
        'max_frame': local.max_frame,
        'retry_count': local.retry_count,
        # Add more as needed
    }

    # Modulo - must match exactly
    if XID_MODULO in remote_params:
        remote_mod = remote_params[XID_MODULO]
        if remote_mod != local.modulo:
            raise NegotiationError(
                f"Modulo mismatch: local={local.modulo}, remote={remote_mod}"
            )

    # Window size k - take minimum
    if XID_WINDOW in remote_params:
        negotiated_values['window_size'] = min(
            local.window_size, remote_params[XID_WINDOW]
        )

    # N1 max frame - take minimum
    if XID_N1 in remote_params:
        negotiated_values['max_frame'] = min(
            local.max_frame, remote_params[XID_N1]
        )

    # Retry count N2 - take remote if provided
    if XID_RETRY in remote_params:
        negotiated_values['retry_count'] = remote_params[XID_RETRY]

    # SREJ - only if remote supports (we always do)
    srej_supported = XID_SREJ in remote_params and remote_params[XID_SREJ] == 1
    if not srej_supported:
        logger.warning("SREJ not supported by peer")

    # Create new config
    negotiated = AX25Config(**negotiated_values)

    logger.info(f"Negotiated final config: {negotiated}")
    return negotiated
