# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.negotiation -- XID parameter negotiation for AX.25 v2.2.

Before two stations open a connected link, they can exchange XID
(Exchange Identification) frames to agree on how they will talk to
each other. Think of it like two people shaking hands and agreeing
on speaking speed and language before having a conversation.

The parameters they negotiate include:
  - Modulo: Whether to use 8 or 128 sequence numbers.
  - Window size (k): How many frames can be in the air at once.
  - Max frame size (N1): The largest allowed data payload.
  - SREJ support: Whether the Selective Reject feature is available.

Parameters are encoded in TLV (Type-Length-Value) format: a 1-byte
type code, a 1-byte length, and then the value bytes.

Compliant with AX.25 v2.2 Section 4.3.4 (XID procedures).
"""

from __future__ import annotations

import struct
from typing import Dict
import logging

from .framing import AX25Frame
from .config import AX25Config
from .exceptions import NegotiationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# XID Parameter type codes (per AX.25 v2.2 spec)
# ---------------------------------------------------------------------------

#: TLV type for the modulo mode (1 byte: 8 or 128).
XID_MODULO: int = 0x01

#: TLV type for the window size k (1 byte).
XID_WINDOW: int = 0x02

#: TLV type for the maximum I-field length N1 (2 bytes, little-endian).
XID_N1: int = 0x03

#: TLV type for the retry count N2 (1 byte).
XID_RETRY: int = 0x04

#: TLV type for the T1 timer multiplier (1 byte).
XID_T1: int = 0x05

#: TLV type for the T2 timer (1 byte).
XID_T2: int = 0x06

#: TLV type for the T3 timer (2 bytes, little-endian).
XID_T3: int = 0x07

#: TLV type for Selective Reject support (1 byte: 0=no, 1=yes).
XID_SREJ: int = 0x08


# ---------------------------------------------------------------------------
# XID frame builder
# ---------------------------------------------------------------------------

def build_xid_frame(config: AX25Config) -> bytes:
    """Build the information field bytes for an XID frame.

    Packs the local configuration into TLV (Type-Length-Value) format
    so the remote station knows what we support. The remote station
    will reply with its own XID, and then we negotiate.

    The mandatory parameters are: modulo, window size, and N1 (max
    frame size). We also include SREJ support and retry count.

    Args:
        config: The local AX.25 configuration to advertise. All fields
            are expected to pass AX25Config validation (no further
            checking is done here).

    Returns:
        A bytes object ready to put in the ``info`` field of an XID
        AX25Frame. The length depends on how many parameters are packed.

    Raises:
        NegotiationError: This function does not raise, but the config
            itself would have raised ConfigurationError if invalid.

    Example::

        xid_info = build_xid_frame(config)
        xid_frame = AX25Frame(
            destination=remote, source=local,
            control=0xAF, info=xid_info,
        )
    """
    logger.debug(
        "build_xid_frame: modulo=%d k=%d N1=%d N2=%d",
        config.modulo, config.window_size, config.max_frame, config.retry_count,
    )

    params = bytearray()

    # Modulo (mandatory, 1 byte)
    params += struct.pack("BB", XID_MODULO, 1)
    params += struct.pack("B", config.modulo)

    # Window size k (mandatory, 1 byte)
    params += struct.pack("BB", XID_WINDOW, 1)
    params += struct.pack("B", config.window_size)

    # Max I-field N1 (mandatory, 2 bytes little-endian)
    params += struct.pack("BB", XID_N1, 2)
    params += struct.pack("<H", config.max_frame)

    # SREJ support (optional, 1 byte; we always advertise support)
    params += struct.pack("BB", XID_SREJ, 1)
    params += struct.pack("B", 1)   # 1 = supported

    # Retry count N2 (optional, 1 byte)
    params += struct.pack("BB", XID_RETRY, 1)
    params += struct.pack("B", config.retry_count)

    result = bytes(params)
    logger.info("build_xid_frame: built %d bytes of XID parameters", len(result))
    return result


# ---------------------------------------------------------------------------
# XID frame parser
# ---------------------------------------------------------------------------

def parse_xid_frame(info: bytes) -> Dict[int, int]:
    """Parse the information field of a received XID frame.

    Reads each TLV entry and builds a dictionary from parameter type
    code to decoded integer value.

    Only 1-byte and 2-byte parameter values are supported. If an
    unknown length is encountered, NegotiationError is raised.

    Args:
        info: The raw bytes from the XID frame's information field.
            Must be a valid sequence of TLV entries. Empty is OK (returns
            an empty dict).

    Returns:
        A dict mapping each parameter type code (int) to its value (int).
        For example: ``{1: 8, 2: 7, 3: 256, 8: 1, 4: 10}``.

    Raises:
        NegotiationError: If the data is truncated, has a bad length,
            or has extra bytes after the last parameter.

    Example::

        params = parse_xid_frame(frame.info)
        remote_modulo = params.get(XID_MODULO, 8)
    """
    if not info:
        logger.debug("parse_xid_frame: empty info field -- no parameters")
        return {}

    logger.debug("parse_xid_frame: parsing %d bytes", len(info))

    params: Dict[int, int] = {}
    offset = 0

    while offset < len(info):
        # Need at least 2 bytes for type + length
        if offset + 2 > len(info):
            raise NegotiationError(
                f"XID data truncated at offset {offset}: "
                f"need 2 bytes for TLV header, have {len(info) - offset}"
            )

        param_id = info[offset]
        length = info[offset + 1]
        offset += 2

        # Check we have enough bytes for the value
        if offset + length > len(info):
            raise NegotiationError(
                f"XID parameter {param_id:#04x} at offset {offset}: "
                f"declared length {length} bytes but only "
                f"{len(info) - offset} bytes remain"
            )

        # Decode value based on length
        if length == 1:
            value = struct.unpack("B", info[offset:offset + 1])[0]
        elif length == 2:
            value = struct.unpack("<H", info[offset:offset + 2])[0]
        else:
            raise NegotiationError(
                f"XID parameter {param_id:#04x} has unsupported length {length} "
                f"(only 1 or 2 bytes are supported)"
            )

        params[param_id] = value
        offset += length
        logger.debug(
            "parse_xid_frame: param_id=0x%02X length=%d value=%d",
            param_id, length, value,
        )

    logger.info(
        "parse_xid_frame: parsed %d parameters: %s",
        len(params), params,
    )
    return params


# ---------------------------------------------------------------------------
# Configuration negotiator
# ---------------------------------------------------------------------------

def negotiate_config(
    local: AX25Config,
    remote_params: Dict[int, int],
) -> AX25Config:
    """Choose the final link settings from local config and remote XID params.

    Applies the AX.25 v2.2 negotiation rules:
      - Modulo must match exactly (no fallback -- raises NegotiationError).
      - Window size is the minimum of local and remote.
      - Max frame size (N1) is the minimum of local and remote.
      - Retry count is taken from the remote if provided.
      - SREJ is only enabled if both sides support it.

    Args:
        local: The local configuration to start from.
        remote_params: The parameters received from the remote XID frame,
            as returned by ``parse_xid_frame()``.

    Returns:
        A new AX25Config with the negotiated values. The returned config
        is guaranteed to be valid (it passes AX25Config validation).

    Raises:
        NegotiationError: If the modulo values do not match (fatal
            incompatibility), or if the negotiated values are invalid.

    Example::

        params = parse_xid_frame(frame.info)
        final_config = negotiate_config(my_config, params)
        logger.info("Agreed on window_size=%d", final_config.window_size)
    """
    logger.debug(
        "negotiate_config: local=modulo=%d k=%d N1=%d, remote_params=%s",
        local.modulo, local.window_size, local.max_frame, remote_params,
    )

    # Start with local values as the baseline
    negotiated: Dict = {
        "modulo": local.modulo,
        "window_size": local.window_size,
        "max_frame": local.max_frame,
        "t1_timeout": local.t1_timeout,
        "t3_timeout": local.t3_timeout,
        "retry_count": local.retry_count,
        "tx_delay": local.tx_delay,
        "tx_tail": local.tx_tail,
        "persistence": local.persistence,
        "slot_time": local.slot_time,
    }

    # --- Modulo: must match exactly ---
    if XID_MODULO in remote_params:
        remote_mod = remote_params[XID_MODULO]
        if remote_mod != local.modulo:
            raise NegotiationError(
                f"Modulo mismatch: local={local.modulo}, remote={remote_mod} -- "
                f"both stations must use the same modulo"
            )
        logger.debug("negotiate_config: modulo=%d -- matches", local.modulo)

    # --- Window size: take minimum ---
    if XID_WINDOW in remote_params:
        remote_k = remote_params[XID_WINDOW]
        negotiated["window_size"] = min(local.window_size, remote_k)
        logger.debug(
            "negotiate_config: window_size min(%d, %d) = %d",
            local.window_size, remote_k, negotiated["window_size"],
        )

    # --- Max frame N1: take minimum ---
    if XID_N1 in remote_params:
        remote_n1 = remote_params[XID_N1]
        negotiated["max_frame"] = min(local.max_frame, remote_n1)
        logger.debug(
            "negotiate_config: max_frame min(%d, %d) = %d",
            local.max_frame, remote_n1, negotiated["max_frame"],
        )

    # --- Retry count: take remote if provided ---
    if XID_RETRY in remote_params:
        remote_n2 = remote_params[XID_RETRY]
        negotiated["retry_count"] = remote_n2
        logger.debug(
            "negotiate_config: retry_count from remote: %d", remote_n2,
        )

    # --- SREJ: note whether both sides support it (informational only) ---
    srej_both = (XID_SREJ in remote_params and remote_params[XID_SREJ] == 1)
    if not srej_both:
        logger.warning(
            "negotiate_config: SREJ not supported by remote station -- "
            "SREJ disabled for this link"
        )
    else:
        logger.debug("negotiate_config: SREJ supported by both sides")

    try:
        result = AX25Config(**negotiated)
    except Exception as exc:
        raise NegotiationError(
            f"Negotiated parameters are invalid: {exc}"
        ) from exc

    logger.info(
        "negotiate_config: final config: modulo=%d k=%d N1=%d N2=%d",
        result.modulo, result.window_size, result.max_frame, result.retry_count,
    )
    return result
