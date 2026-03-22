# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
tests/test_fx25.py

Unit tests for the FX.25 Reed-Solomon FEC implementation.

Covers:
- GF(2^8) field arithmetic (gf_mul, gf_div, gf_pow, gf_inv)
- RS encode/decode round-trip (no errors)
- RS decode: corrects 1, t, and t+1 errors
- FX.25 constants: ctag_to_wire / wire_to_tag_id round-trip
- FX25Encoder: produces correct block size and wire format
- FX25Decoder: recovers frame from encoded stream
- FX25Decoder: corrects errors in stream
- FX25Decoder: rejects uncorrectable block
"""

import pytest

from pyax25_22.interfaces.fx25.rs import (
    gf_mul, gf_div, gf_pow, gf_inv,
    gf_poly_mul, gf_poly_eval,
    rs_encode, rs_decode,
)
from pyax25_22.interfaces.fx25.constants import (
    FX25_TAGS, CTAG_SIZE, ctag_to_wire, wire_to_tag_id, choose_tag, tag_info,
)
from pyax25_22.interfaces.fx25.encoder import FX25Encoder
from pyax25_22.interfaces.fx25.decoder import FX25Decoder


# ---------------------------------------------------------------------------
# GF(2^8) arithmetic
# ---------------------------------------------------------------------------

def test_gf_mul_zero():
    """Multiplying by zero returns zero."""
    assert gf_mul(0, 127) == 0
    assert gf_mul(99, 0) == 0


def test_gf_mul_one():
    """Multiplying by 1 (alpha^0) returns the other operand."""
    assert gf_mul(1, 42) == 42
    assert gf_mul(42, 1) == 42


def test_gf_mul_commutative():
    """Multiplication is commutative in GF(2^8)."""
    assert gf_mul(73, 137) == gf_mul(137, 73)


def test_gf_mul_associative():
    """Multiplication is associative in GF(2^8)."""
    a, b, c = 3, 7, 11
    assert gf_mul(gf_mul(a, b), c) == gf_mul(a, gf_mul(b, c))


def test_gf_div_by_one():
    """Dividing by 1 returns the dividend."""
    assert gf_div(99, 1) == 99


def test_gf_div_inverse_of_mul():
    """a * b / b == a for all non-zero b."""
    a, b = 42, 73
    assert gf_div(gf_mul(a, b), b) == a


def test_gf_div_zero_dividend():
    """0 / b == 0 for any non-zero b."""
    assert gf_div(0, 5) == 0


def test_gf_div_by_zero_raises():
    """Dividing by zero raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        gf_div(5, 0)


def test_gf_pow_zero_exponent():
    """x^0 == 1 for any non-zero x."""
    assert gf_pow(7, 0) == 1


def test_gf_pow_one_exponent():
    """x^1 == x."""
    assert gf_pow(42, 1) == 42


def test_gf_pow_zero_base():
    """0^n == 0 for any n."""
    assert gf_pow(0, 5) == 0


def test_gf_inv_times_original_is_one():
    """x * x^-1 == 1 for any non-zero x."""
    for x in [1, 2, 3, 42, 127, 255]:
        assert gf_mul(x, gf_inv(x)) == 1


def test_gf_inv_zero_raises():
    """Inverse of zero raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        gf_inv(0)


# ---------------------------------------------------------------------------
# RS encode / decode round-trip (no errors)
# ---------------------------------------------------------------------------

def test_rs_encode_decode_roundtrip_ncheck16():
    """RS encode + decode with 16 check bytes, no errors."""
    data = bytes(range(32))
    codeword = rs_encode(data, 16)
    assert len(codeword) == len(data) + 16
    recovered, nerr = rs_decode(codeword, 16)
    assert recovered == data
    assert nerr == 0


def test_rs_encode_decode_roundtrip_ncheck32():
    """RS encode + decode with 32 check bytes, no errors."""
    data = b"Hello FX.25 world!" + bytes(10)
    codeword = rs_encode(data, 32)
    recovered, nerr = rs_decode(codeword, 32)
    assert recovered == data
    assert nerr == 0


def test_rs_encode_data_too_long_raises():
    """rs_encode raises ValueError if data > 255 - ncheck bytes."""
    with pytest.raises(ValueError):
        rs_encode(bytes(250), 16)  # 250 > 255-16=239


def test_rs_decode_too_short_raises():
    """rs_decode raises ValueError if codeword shorter than ncheck."""
    with pytest.raises(ValueError):
        rs_decode(bytes(8), 16)  # k = 8-16 <= 0


# ---------------------------------------------------------------------------
# RS error correction
# ---------------------------------------------------------------------------

def test_rs_corrects_single_error():
    """RS decode corrects a single symbol error."""
    data = bytes(range(30))
    ncheck = 16
    codeword = bytearray(rs_encode(data, ncheck))
    codeword[5] ^= 0xAB    # corrupt byte 5
    recovered, nerr = rs_decode(bytes(codeword), ncheck)
    assert recovered == data
    assert nerr == 1


def test_rs_corrects_max_errors():
    """RS decode corrects exactly t = ncheck/2 symbol errors."""
    data = bytes(range(30))
    ncheck = 16
    t = ncheck // 2          # 8 correctable errors
    codeword = bytearray(rs_encode(data, ncheck))
    for i in range(t):
        codeword[i] ^= (0x11 * (i + 1)) & 0xFF
    recovered, nerr = rs_decode(bytes(codeword), ncheck)
    assert recovered == data
    assert nerr == t


def test_rs_too_many_errors_returns_none():
    """RS decode returns None when errors exceed correction capability."""
    data = bytes(range(30))
    ncheck = 16
    t_plus_one = ncheck // 2 + 1   # 9 errors -- beyond capability
    codeword = bytearray(rs_encode(data, ncheck))
    for i in range(t_plus_one):
        codeword[i] ^= (0x55 * (i + 1)) & 0xFF
    recovered, nerr = rs_decode(bytes(codeword), ncheck)
    assert recovered is None
    assert nerr == -1


def test_rs_no_errors_returns_zero():
    """rs_decode returns 0 error count when codeword is clean."""
    data = bytes(50)
    codeword = rs_encode(data, 16)
    _, nerr = rs_decode(codeword, 16)
    assert nerr == 0


# ---------------------------------------------------------------------------
# FX.25 constants
# ---------------------------------------------------------------------------

def test_ctag_roundtrip_all_tags():
    """ctag_to_wire / wire_to_tag_id round-trip for all defined tags."""
    for tag_id in FX25_TAGS:
        wire = ctag_to_wire(tag_id)
        assert len(wire) == CTAG_SIZE
        assert wire_to_tag_id(wire) == tag_id


def test_ctag_complement_check():
    """wire_to_tag_id raises if complement bytes are wrong."""
    wire = bytearray(ctag_to_wire(0x01))
    wire[4] ^= 0xFF  # Break complement
    with pytest.raises(ValueError):
        wire_to_tag_id(bytes(wire))


def test_ctag_wrong_length_raises():
    """wire_to_tag_id raises ValueError for non-8-byte input."""
    with pytest.raises(ValueError):
        wire_to_tag_id(b"short")


def test_choose_tag_small_frame():
    """choose_tag picks tag 0x09 (tiny, 32 data bytes) for a 20-byte frame."""
    tag = choose_tag(20)
    _, data, _ = tag_info(tag)
    assert data >= 20


def test_choose_tag_full_frame():
    """choose_tag picks tag 0x01 for a 239-byte frame."""
    tag = choose_tag(239)
    assert tag == 0x01


def test_choose_tag_too_large_raises():
    """choose_tag raises ValueError for frames larger than 239 bytes."""
    with pytest.raises(ValueError):
        choose_tag(240)


# ---------------------------------------------------------------------------
# FX25Encoder
# ---------------------------------------------------------------------------

def test_encoder_block_size():
    """Encoded block has correct size: preamble + 8 (tag) + total_bytes."""
    encoder = FX25Encoder(preamble_len=0)
    ax25 = bytes(20)
    result = encoder.encode(ax25)
    # Tag 0x09: total=48
    assert len(result) == CTAG_SIZE + 48


def test_encoder_preamble():
    """Preamble bytes are prepended as 0x00."""
    encoder = FX25Encoder(preamble_len=4)
    result = encoder.encode(bytes(20))
    assert result[:4] == bytes(4)


def test_encoder_connect_tag_valid():
    """Encoded block starts with a valid FX.25 connect tag (after preamble)."""
    encoder = FX25Encoder(preamble_len=0)
    result = encoder.encode(bytes(20))
    tag_id = wire_to_tag_id(result[:CTAG_SIZE])
    assert tag_id in FX25_TAGS


def test_encoder_forced_tag():
    """Forcing tag_id=0x02 produces the expected block size."""
    encoder = FX25Encoder()
    ax25 = bytes(100)
    result = encoder.encode(ax25, tag_id=0x02)
    _, _, total = tag_info(0x02)
    assert len(result) == CTAG_SIZE + total


def test_encoder_forced_tag_too_small_raises():
    """Encoder raises ValueError if frame is too large for forced tag."""
    encoder = FX25Encoder()
    with pytest.raises(ValueError):
        encoder.encode(bytes(200), tag_id=0x09)   # tag 0x09 holds only 32 bytes


# ---------------------------------------------------------------------------
# FX25Decoder
# ---------------------------------------------------------------------------

def _make_fx25(ax25: bytes, tag_id: int = None) -> bytes:
    """Helper: encode ax25 bytes as FX.25."""
    enc = FX25Encoder(preamble_len=0)
    return enc.encode(ax25, tag_id=tag_id)


def test_decoder_roundtrip():
    """Decoder recovers original AX.25 bytes from encoded stream."""
    ax25 = b"KE4AHR" + bytes(10)
    fx25 = _make_fx25(ax25)

    received = []
    dec = FX25Decoder(on_frame=lambda d, e: received.append((d, e)))
    dec.feed(fx25)

    assert len(received) == 1
    recovered, nerr = received[0]
    # Decoder returns full data field (padded to data_bytes); AX.25 is prefix
    assert recovered[:len(ax25)] == ax25
    assert nerr == 0


def test_decoder_with_preamble():
    """Decoder finds tag even with preamble bytes."""
    ax25 = bytes(10)
    enc = FX25Encoder(preamble_len=8)
    fx25 = enc.encode(ax25)

    received = []
    dec = FX25Decoder(on_frame=lambda d, e: received.append((d, e)))
    dec.feed(fx25)
    assert len(received) == 1


def test_decoder_corrects_errors():
    """Decoder corrects a single byte error and calls on_frame."""
    ax25 = bytes(range(20))
    fx25 = bytearray(_make_fx25(ax25))
    # Corrupt one byte in the data section (after the 8-byte tag)
    fx25[CTAG_SIZE + 3] ^= 0xAA

    received = []
    dec = FX25Decoder(on_frame=lambda d, e: received.append((d, e)))
    dec.feed(bytes(fx25))

    assert len(received) == 1
    recovered, nerr = received[0]
    assert recovered[:len(ax25)] == ax25
    assert nerr == 1


def test_decoder_uncorrectable_calls_on_error():
    """Decoder calls on_error for an uncorrectable block."""
    ax25 = bytes(20)
    fx25 = bytearray(_make_fx25(ax25, tag_id=0x09))  # 16 check bytes, t=8
    # Corrupt 9 bytes -- beyond correction capability
    for i in range(9):
        fx25[CTAG_SIZE + i] ^= (0x33 * (i + 1)) & 0xFF

    errors = []
    received = []
    dec = FX25Decoder(
        on_frame=lambda d, e: received.append((d, e)),
        on_error=lambda r: errors.append(r),
    )
    dec.feed(bytes(fx25))
    assert len(received) == 0
    assert len(errors) == 1


def test_decoder_multiple_frames():
    """Decoder handles two consecutive FX.25 blocks."""
    ax25_a = b"Frame A" + bytes(10)
    ax25_b = b"Frame B" + bytes(10)
    stream = _make_fx25(ax25_a) + _make_fx25(ax25_b)

    received = []
    dec = FX25Decoder(on_frame=lambda d, e: received.append((d, e)))
    dec.feed(stream)
    assert len(received) == 2


def test_decoder_reset():
    """reset() returns decoder to hunt state."""
    ax25 = bytes(10)
    fx25 = _make_fx25(ax25)

    received = []
    dec = FX25Decoder(on_frame=lambda d, e: received.append((d, e)))
    # Feed half the stream, then reset
    dec.feed(fx25[:len(fx25) // 2])
    dec.reset()
    # Now feed a fresh complete block -- should decode OK
    dec.feed(fx25)
    assert len(received) == 1
