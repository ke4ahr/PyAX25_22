# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
tests/test_negotiation.py

Unit tests for XID frame building, parsing, and parameter negotiation.

Covers:
- build_xid_frame round-trip with parse_xid_frame
- Modulo match and mismatch
- Window size and N1 minimums
- Retry count from remote
- Error cases: truncated TLV, extra bytes, unknown length
"""

import pytest

from pyax25_22.core.negotiation import (
    build_xid_frame,
    parse_xid_frame,
    negotiate_config,
    XID_MODULO,
    XID_WINDOW,
    XID_N1,
    XID_RETRY,
    XID_SREJ,
)
from pyax25_22.core.config import AX25Config, DEFAULT_CONFIG_MOD8
from pyax25_22.core.exceptions import NegotiationError


@pytest.fixture
def local_mod8():
    """Standard modulo-8 config."""
    return AX25Config(modulo=8, window_size=7, max_frame=256, retry_count=10)


@pytest.fixture
def local_mod128():
    """Modulo-128 config with small window."""
    return AX25Config(modulo=128, window_size=32, max_frame=256, retry_count=5)


# ---------------------------------------------------------------------------
# build_xid_frame / parse_xid_frame round-trip
# ---------------------------------------------------------------------------

def test_build_parse_roundtrip(local_mod8):
    """build then parse returns same parameters."""
    xid_bytes = build_xid_frame(local_mod8)
    params = parse_xid_frame(xid_bytes)

    assert params[XID_MODULO] == local_mod8.modulo
    assert params[XID_WINDOW] == local_mod8.window_size
    assert params[XID_N1] == local_mod8.max_frame
    assert params[XID_RETRY] == local_mod8.retry_count
    assert params[XID_SREJ] == 1


def test_build_parse_mod128(local_mod128):
    """Round-trip for modulo-128 config."""
    xid_bytes = build_xid_frame(local_mod128)
    params = parse_xid_frame(xid_bytes)
    assert params[XID_MODULO] == 128
    assert params[XID_WINDOW] == local_mod128.window_size


def test_parse_empty_bytes():
    """Empty info field returns empty dict."""
    params = parse_xid_frame(b"")
    assert params == {}


def test_parse_truncated_raises():
    """Truncated TLV raises NegotiationError."""
    with pytest.raises(NegotiationError):
        parse_xid_frame(b"\x01\x02\x08")  # Type=1, Length=2, only 1 value byte


def test_parse_unsupported_length_raises():
    """3-byte TLV value raises NegotiationError."""
    bad = b"\x01\x03\x00\x00\x00"  # Type=1, Length=3, value 3 bytes
    with pytest.raises(NegotiationError):
        parse_xid_frame(bad)


# ---------------------------------------------------------------------------
# negotiate_config
# ---------------------------------------------------------------------------

def test_negotiate_modulo_match(local_mod8):
    """Negotiation succeeds when modulo matches."""
    remote_params = {XID_MODULO: 8, XID_WINDOW: 4, XID_N1: 128}
    result = negotiate_config(local_mod8, remote_params)
    assert result.modulo == 8
    assert result.window_size == 4   # min(7, 4)
    assert result.max_frame == 128   # min(256, 128)


def test_negotiate_modulo_mismatch(local_mod8):
    """Modulo mismatch raises NegotiationError."""
    remote_params = {XID_MODULO: 128}
    with pytest.raises(NegotiationError):
        negotiate_config(local_mod8, remote_params)


def test_negotiate_window_min(local_mod8):
    """Negotiated window is minimum of local and remote."""
    remote_params = {XID_WINDOW: 3}
    result = negotiate_config(local_mod8, remote_params)
    assert result.window_size == 3

    remote_params2 = {XID_WINDOW: 10}   # larger than local
    result2 = negotiate_config(local_mod8, remote_params2)
    assert result2.window_size == 7  # local is smaller


def test_negotiate_n1_min(local_mod8):
    """Negotiated N1 is minimum of local and remote."""
    remote_params = {XID_N1: 100}
    result = negotiate_config(local_mod8, remote_params)
    assert result.max_frame == 100


def test_negotiate_retry_from_remote(local_mod8):
    """Remote retry count is used when provided."""
    remote_params = {XID_RETRY: 5}
    result = negotiate_config(local_mod8, remote_params)
    assert result.retry_count == 5


def test_negotiate_no_remote_params(local_mod8):
    """Empty remote params -- local config is returned unchanged."""
    result = negotiate_config(local_mod8, {})
    assert result.modulo == local_mod8.modulo
    assert result.window_size == local_mod8.window_size
    assert result.max_frame == local_mod8.max_frame
