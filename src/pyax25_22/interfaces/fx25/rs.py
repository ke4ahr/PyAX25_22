# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
interfaces/fx25/rs.py

Reed-Solomon GF(2^8) encoder and decoder -- pure Python implementation.

This module implements the Reed-Solomon FEC code used by FX.25, following
the TAPR FX.25 specification and compatible with Dire Wolf's implementation.

Theory
------
Reed-Solomon codes work over a finite field (Galois Field).  FX.25 uses
GF(2^8) with the primitive polynomial 0x11D (x^8 + x^4 + x^3 + x^2 + 1).

An RS(n, k) code takes k data symbols and produces n - k check symbols.
Up to (n - k) / 2 symbol errors can be corrected.

For FX.25:
    n = total block size (e.g., 255 for tags 01-03)
    k = data symbols (e.g., 239 for tag 01)
    check symbols = n - k (e.g., 16 for tag 01)

Encoding algorithm:
    1. Treat the k data bytes as coefficients of a polynomial D(x).
    2. Compute D(x) * x^(n-k) mod G(x), where G(x) is the generator poly.
    3. The n-k remainder bytes are the check symbols, appended to D(x).

Decoding algorithm (Berlekamp-Massey):
    1. Compute n-k syndromes S_i = received_poly(alpha^(FCR+i)).
    2. If all zero, no errors -- return data symbols.
    3. Berlekamp-Massey: find error locator polynomial Lambda(x).
    4. Chien search: evaluate Lambda(x) at each field element to find error
       positions.
    5. Forney algorithm: compute error magnitudes.
    6. Correct errors in the received codeword.

Implementation notes:
    - Log/antilog tables speed up field arithmetic.
    - The implementation is based on classic texts (Wicker, Blahut) and
      open-source RS libraries (ezpwd, libfec, reedsolo).
"""

import logging
from typing import List, Optional, Tuple

from .constants import GF_POLY, FCR, PRIM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GF(2^8) Field tables
# ---------------------------------------------------------------------------

_EXP = [0] * 512   # Antilog table: _EXP[i] = alpha^i; doubled for wrap-around
_LOG = [0] * 256   # Log table: _LOG[v] = i such that alpha^i = v; _LOG[0] = -inf


def _build_tables() -> None:
    """Build log and antilog tables for GF(2^8).

    Uses the primitive polynomial GF_POLY to generate all 255 non-zero field
    elements as powers of alpha (the primitive element, alpha = PRIM = 2).
    """
    x = 1
    for i in range(255):
        _EXP[i] = x
        _EXP[i + 255] = x   # Doubled for modular arithmetic convenience
        _LOG[x] = i
        x = x << 1
        if x & 0x100:
            x ^= GF_POLY     # Reduce modulo the field polynomial


_build_tables()


def gf_mul(a: int, b: int) -> int:
    """Multiply two elements of GF(2^8).

    Args:
        a, b: Field elements (0-255).

    Returns:
        Product a * b in GF(2^8).
    """
    if a == 0 or b == 0:
        return 0
    return _EXP[(_LOG[a] + _LOG[b]) % 255]


def gf_div(a: int, b: int) -> int:
    """Divide a by b in GF(2^8).

    Args:
        a: Dividend (0-255).
        b: Divisor (1-255, must not be zero).

    Returns:
        Quotient a / b in GF(2^8).

    Raises:
        ZeroDivisionError: If b == 0.
    """
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b] + 255) % 255]


def gf_pow(x: int, power: int) -> int:
    """Compute x^power in GF(2^8).

    Args:
        x: Base element (0-255).
        power: Exponent (non-negative integer).

    Returns:
        x^power in GF(2^8).
    """
    if x == 0:
        return 0
    return _EXP[(_LOG[x] * power) % 255]


def gf_inv(x: int) -> int:
    """Compute the multiplicative inverse of x in GF(2^8).

    Args:
        x: Field element (1-255).

    Returns:
        x^(-1) such that x * x^(-1) = 1 in GF(2^8).

    Raises:
        ZeroDivisionError: If x == 0 (zero has no inverse).
    """
    if x == 0:
        raise ZeroDivisionError("GF inverse of zero is undefined")
    return _EXP[255 - _LOG[x]]


def gf_poly_mul(p: List[int], q: List[int]) -> List[int]:
    """Multiply two polynomials over GF(2^8).

    Coefficients are listed from highest degree to lowest degree
    (big-endian / standard mathematical convention).

    Args:
        p, q: Polynomials as lists of GF(2^8) coefficients.

    Returns:
        Product polynomial.
    """
    result = [0] * (len(p) + len(q) - 1)
    for i, coef_p in enumerate(p):
        for j, coef_q in enumerate(q):
            result[i + j] ^= gf_mul(coef_p, coef_q)
    return result


def gf_poly_eval(poly: List[int], x: int) -> int:
    """Evaluate a polynomial at a given field element using Horner's method.

    Args:
        poly: Polynomial coefficients (highest degree first).
        x: Field element to evaluate at (0-255).

    Returns:
        Value of poly(x) in GF(2^8).
    """
    result = 0
    for coef in poly:
        result = gf_mul(result, x) ^ coef
    return result


# ---------------------------------------------------------------------------
# Generator polynomial
# ---------------------------------------------------------------------------

def _make_generator(ncheck: int) -> List[int]:
    """Build the RS generator polynomial for ncheck check symbols.

    G(x) = prod_{i=FCR}^{FCR+ncheck-1} (x - alpha^i)

    For FX.25, FCR = 1 (alpha^1), so:
        G(x) = (x - alpha)(x - alpha^2)...(x - alpha^ncheck)

    Args:
        ncheck: Number of check symbols (must be even).

    Returns:
        Generator polynomial coefficients (highest degree first).
        Length is ncheck + 1.
    """
    g = [1]
    for i in range(FCR, FCR + ncheck):
        root = gf_pow(PRIM, i)
        g = gf_poly_mul(g, [1, root])
    return g


# Cache generator polynomials (they are fixed for a given ncheck)
_GEN_CACHE: dict = {}


def _get_generator(ncheck: int) -> List[int]:
    """Return cached RS generator polynomial for ncheck check symbols."""
    if ncheck not in _GEN_CACHE:
        _GEN_CACHE[ncheck] = _make_generator(ncheck)
    return _GEN_CACHE[ncheck]


# ---------------------------------------------------------------------------
# RS Encoder
# ---------------------------------------------------------------------------

def rs_encode(data: bytes, ncheck: int) -> bytes:
    """Encode data bytes using Reed-Solomon, appending check symbols.

    Computes ncheck check symbols and returns data + check_symbols.

    Args:
        data: Message bytes to protect (length must be <= 255 - ncheck).
        ncheck: Number of check symbols to append.

    Returns:
        Original data bytes followed by ncheck check symbol bytes.

    Raises:
        ValueError: If data is longer than 255 - ncheck bytes.
    """
    if len(data) > 255 - ncheck:
        raise ValueError(
            f"RS encode: data length {len(data)} exceeds max {255 - ncheck} "
            f"for ncheck={ncheck}"
        )

    gen = _get_generator(ncheck)
    # Multiply message by x^ncheck (shift left by ncheck positions)
    msg_ex = list(data) + [0] * ncheck
    # Divide msg_ex by generator polynomial using synthetic division
    for i in range(len(data)):
        coef = msg_ex[i]
        if coef != 0:
            for j in range(1, len(gen)):
                msg_ex[i + j] ^= gf_mul(gen[j], coef)

    # The last ncheck bytes are the check symbols
    check = bytes(msg_ex[len(data):])
    logger.debug(
        "RS encode: %d data bytes -> %d check bytes", len(data), ncheck
    )
    return bytes(data) + check


# ---------------------------------------------------------------------------
# RS Decoder
# ---------------------------------------------------------------------------

def rs_decode(
    codeword: bytes, ncheck: int
) -> Tuple[Optional[bytes], int]:
    """Decode a received RS codeword, correcting errors if possible.

    Args:
        codeword: Received bytes (data + check), length must equal data+ncheck.
        ncheck: Number of check symbols (same as used during encoding).

    Returns:
        Tuple (corrected_data, num_errors_corrected).
        corrected_data is the k data bytes, or None if uncorrectable.
        num_errors_corrected is 0 if no errors, -1 if uncorrectable.

    Raises:
        ValueError: If codeword length is invalid.
    """
    n = len(codeword)
    k = n - ncheck
    if k <= 0:
        raise ValueError(
            f"RS decode: codeword length {n} too short for ncheck={ncheck}"
        )

    # Step 1: Compute syndromes
    syndromes = [
        gf_poly_eval(list(codeword), gf_pow(PRIM, FCR + i))
        for i in range(ncheck)
    ]

    if all(s == 0 for s in syndromes):
        logger.debug("RS decode: no errors detected")
        return bytes(codeword[:k]), 0

    # Step 2: Berlekamp-Massey -- find error locator polynomial
    locator = _berlekamp_massey(syndromes, ncheck)
    if locator is None:
        logger.warning("RS decode: too many errors -- uncorrectable")
        return None, -1

    # Step 3: Chien search -- find error positions
    positions = _chien_search(locator, n)
    if positions is None or len(positions) != len(locator) - 1:
        logger.warning("RS decode: chien search failed -- too many errors")
        return None, -1

    # Step 4: Forney algorithm -- compute error magnitudes
    magnitudes = _forney(syndromes, locator, positions, n, ncheck)
    if magnitudes is None:
        logger.warning("RS decode: Forney algorithm failed")
        return None, -1

    # Step 5: Correct errors
    # Chien search positions are power indices i where X_k = alpha^i.
    # The codeword is big-endian: byte k holds coefficient of x^(n-1-k),
    # so X_k = alpha^(n-1-k), meaning byte_pos = n - 1 - i.
    corrected = bytearray(codeword)
    for pos, mag in zip(positions, magnitudes):
        byte_pos = n - 1 - pos
        corrected[byte_pos] ^= mag

    # Verify correction
    check_syndromes = [
        gf_poly_eval(list(corrected), gf_pow(PRIM, FCR + i))
        for i in range(ncheck)
    ]
    if not all(s == 0 for s in check_syndromes):
        logger.warning("RS decode: correction failed syndrome check")
        return None, -1

    num_errors = len(positions)
    logger.debug("RS decode: corrected %d errors", num_errors)
    return bytes(corrected[:k]), num_errors


def _berlekamp_massey(syndromes: List[int], ncheck: int) -> Optional[List[int]]:
    """Berlekamp-Massey algorithm to find the error locator polynomial.

    Uses little-endian representation internally (C[i] = coefficient of x^i),
    which matches the LFSR discrepancy formula.  Returns big-endian for use
    with gf_poly_eval (highest degree first).

    Args:
        syndromes: List of ncheck syndrome values.
        ncheck: Number of check symbols.

    Returns:
        Error locator polynomial Lambda(x) as big-endian list (degree 0
        term last), or None if the errors exceed the correction capability.
    """
    # Little-endian: C[0] = Lambda_0 = 1 always, C[i] = Lambda_i (coeff of x^i)
    C = [1]   # Error locator poly (current estimate)
    B = [1]   # Previous estimate
    L = 0     # Current LFSR length
    b = 1     # Discrepancy from previous step

    for n, s in enumerate(syndromes):
        # Compute discrepancy: delta = sum_{i=0}^{L} C[i] * S_{n-i}
        delta = s
        for i in range(1, L + 1):
            if i < len(C):
                delta ^= gf_mul(C[i], syndromes[n - i])

        if delta == 0:
            B = [0] + B   # x * B(x) in little-endian: prepend 0
            continue

        T = C[:]
        # C = C - (delta/b) * x * B
        coef = gf_div(delta, b)
        xB = [0] + B       # x * B(x) in little-endian: prepend 0
        # Extend with zeros appended at the HIGH-degree end (little-endian)
        while len(C) < len(xB):
            C.append(0)
        while len(xB) < len(C):
            xB.append(0)

        C = [c ^ gf_mul(coef, xb) for c, xb in zip(C, xB)]

        if 2 * L <= n:
            L = n + 1 - L
            B = T
            b = delta
        else:
            B = [0] + B   # x * B(x) in little-endian

    # Check: correctable errors?
    num_errors = len(C) - 1
    if num_errors > ncheck // 2:
        return None

    # Return big-endian (highest degree first) for gf_poly_eval
    return list(reversed(C))


def _chien_search(locator: List[int], n: int) -> Optional[List[int]]:
    """Find roots of the error locator polynomial using Chien search.

    Evaluates Lambda at every element of GF(2^8) and collects the
    elements where Lambda(element) = 0.  The error positions are the
    inverses of the roots.

    Args:
        locator: Error locator polynomial.
        n: Codeword length.

    Returns:
        List of error byte positions (0-based), or None on failure.
    """
    positions = []
    for i in range(n):
        # Evaluate Lambda at alpha^-i = alpha^(255-i)
        if gf_poly_eval(locator, gf_pow(PRIM, (255 - i) % 255)) == 0:
            positions.append(i)
    return positions if positions else None


def _forney(
    syndromes: List[int],
    locator: List[int],
    positions: List[int],
    n: int,
    ncheck: int,
) -> Optional[List[int]]:
    """Forney algorithm: compute error magnitudes.

    Args:
        syndromes: Syndrome values.
        locator: Error locator polynomial Lambda(x).
        positions: Error byte positions (from Chien search).
        n: Codeword length.
        ncheck: Number of check symbols.

    Returns:
        List of error magnitudes (one per position), or None on failure.
    """
    # Error evaluator polynomial: Omega(x) = S(x) * Lambda(x) mod x^ncheck
    # S(x) in big-endian: [S_{ncheck-1}, ..., S_1, S_0]
    # mod x^ncheck keeps degrees 0..ncheck-1 = the last ncheck coefficients (big-endian)
    S_big = list(reversed(syndromes))
    omega_full = gf_poly_mul(S_big, locator)
    # In big-endian, mod x^ncheck = keep last ncheck elements (lowest-degree terms)
    omega = omega_full[-ncheck:] if len(omega_full) >= ncheck else omega_full

    # Degree of locator polynomial
    d = len(locator) - 1

    magnitudes = []
    for pos in positions:
        # pos is the Chien search index i where Lambda(alpha^(255-i)) == 0.
        # X_k = alpha^i (error locator number), X_k^{-1} = alpha^(255-i).
        xi = gf_pow(PRIM, pos)        # X_k = alpha^i
        xi_inv = gf_inv(xi)           # X_k^{-1} = alpha^(255-i)

        # Omega(X_k^{-1})
        omega_val = gf_poly_eval(omega, xi_inv)

        # Lambda'(X_k^{-1}): formal derivative of Lambda evaluated at xi_inv.
        # For big-endian Lambda = [L_d, ..., L_1, L_0]:
        #   Lambda'(x) = sum_{j odd, j=1..d} L_j * x^{j-1}
        #   In big-endian, L_j = locator[d - j]
        lambda_prime = 0
        for j in range(1, d + 1, 2):
            lambda_prime ^= gf_mul(locator[d - j], gf_pow(xi_inv, j - 1))

        if lambda_prime == 0:
            logger.warning("Forney: Lambda'(xi_inv) == 0 -- division by zero")
            return None

        # Error magnitude: e = X_k^{1-FCR} * Omega(X_k^{-1}) / Lambda'(X_k^{-1})
        # With FCR=1: X_k^{1-1} = X_k^0 = 1, so magnitude = Omega / Lambda'
        magnitude = gf_mul(
            gf_pow(xi, 1 - FCR),
            gf_div(omega_val, lambda_prime),
        )
        magnitudes.append(magnitude)

    return magnitudes
