# src/pyax25_22/core/framing.py
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.framing.py

Complete AX.25 v2.2 frame encoding and decoding implementation.

VERSION: 0.5.31
CRITICAL FIX: Removed bit stuffing from software encoding/decoding.
Bit stuffing is handled by HDLC hardware (TNC), not in software frames.

Previous versions incorrectly applied bit stuffing, causing FCS errors.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

from .config import AX25Config, DEFAULT_CONFIG_MOD8
from .exceptions import (
    InvalidAddressError,
    FCSError,
    FrameError,
)

logger = logging.getLogger(__name__)

# AX.25 constants
FLAG = 0x7E
FCS_INIT = 0xFFFF
FCS_POLY = 0x8408


def fcs_calc(data: bytes) -> int:
    """Calculate AX.25 FCS (CRC-16/CCITT-FALSE)."""
    fcs = FCS_INIT
    for byte in data:
        fcs ^= byte
        for _ in range(8):
            if fcs & 1:
                fcs = (fcs >> 1) ^ FCS_POLY
            else:
                fcs >>= 1
    return ~fcs & 0xFFFF


def verify_fcs(data: bytes, received_fcs: int) -> bool:
    """Verify received FCS against calculated value."""
    return fcs_calc(data) == received_fcs


@dataclass
class AX25Address:
    """AX.25 address field with callsign, SSID, and control bits."""

    callsign: str
    ssid: int = 0
    c_bit: bool = False
    h_bit: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize address."""
        if not (0 <= self.ssid <= 15):
            raise InvalidAddressError(f"SSID {self.ssid} out of range (0-15)")

        callsign_clean = self.callsign.upper().strip().replace("-", "")
        if not (1 <= len(callsign_clean) <= 6):
            raise InvalidAddressError(f"Callsign '{self.callsign}' length invalid")

        # Shift callsign characters left by 1
        self._call_bytes = bytes((ord(c) << 1) for c in callsign_clean.ljust(6, " "))

    def encode(self, last: bool = False) -> bytes:
        """Encode 7-byte address field."""
        ssid_byte = 0x80  # Bit 7 = 1 (reserved)
        ssid_byte |= (self.ssid << 1) & 0x1E
        ssid_byte |= 0x40 if self.c_bit else 0x00
        ssid_byte |= 0x20 if self.h_bit else 0x00
        ssid_byte |= 0x01 if last else 0x00
        return self._call_bytes + bytes([ssid_byte])

    @classmethod
    def decode(cls, data: bytes) -> Tuple["AX25Address", bool]:
        """Decode address field from 7 bytes."""
        if len(data) < 7:
            raise InvalidAddressError("Address field too short")

        call_bytes = data[:6]
        ssid_byte = data[6]

        callsign_chars = []
        for b in call_bytes:
            char_code = b >> 1  # Shift right to decode
            if char_code == 0x20:  # Space padding
                break
            callsign_chars.append(chr(char_code))
        callsign = "".join(callsign_chars)

        addr = cls(
            callsign=callsign,
            ssid=(ssid_byte >> 1) & 0x0F,
            c_bit=bool(ssid_byte & 0x40),
            h_bit=bool(ssid_byte & 0x20),
        )

        is_last = bool(ssid_byte & 0x01)
        return addr, is_last


@dataclass
class AX25Frame:
    """Complete AX.25 frame with full v2.2 support."""

    destination: AX25Address
    source: AX25Address
    digipeaters: List[AX25Address] = field(default_factory=list)
    control: int = 0
    pid: Optional[int] = None
    info: bytes = b""
    config: AX25Config = DEFAULT_CONFIG_MOD8

    def encode(self) -> bytes:
        """
        Encode complete frame with flags and FCS.
        
        IMPORTANT: NO bit stuffing applied here. Bit stuffing is handled
        by HDLC hardware (TNC), not in software frame encoding.
        """
        # Address field
        addr_field = self.destination.encode(last=False)
        addr_field += self.source.encode(last=(len(self.digipeaters) == 0))

        for i, digi in enumerate(self.digipeaters):
            last = i == len(self.digipeaters) - 1
            addr_field += digi.encode(last=last)

        # Control + PID + Info
        payload = bytes([self.control & 0xFF])
        if self.config.modulo == 128 and (self.control & 0x01 == 0):
            payload += bytes([(self.control >> 8) & 0xFF])
        if self.pid is not None:
            payload += bytes([self.pid])
        payload += self.info

        # FCS over address + payload
        fcs = fcs_calc(addr_field + payload)
        frame_body = addr_field + payload + struct.pack("<H", fcs)

        # Return frame with flags (NO BIT STUFFING)
        return bytes([FLAG]) + frame_body + bytes([FLAG])

    @classmethod
    def decode(cls, raw: bytes, config: AX25Config = DEFAULT_CONFIG_MOD8) -> "AX25Frame":
        """
        Decode raw frame bytes (including flags).
        
        IMPORTANT: NO bit destuffing applied. Bit stuffing is handled by
        HDLC hardware (TNC), not in software.
        """
        if raw[0] != FLAG or raw[-1] != FLAG:
            raise FrameError("Missing start/end flag")

        # Extract frame body (NO DESTUFFING)
        frame_body = raw[1:-1]

        if len(frame_body) < 16:
            raise FrameError("Frame too short")

        # Parse addresses
        offset = 0
        dest, _ = AX25Address.decode(frame_body[offset:offset+7])
        offset += 7
        src, last = AX25Address.decode(frame_body[offset:offset+7])
        offset += 7

        digipeaters = []
        while not last and offset + 7 <= len(frame_body):
            digi, last = AX25Address.decode(frame_body[offset:offset+7])
            digipeaters.append(digi)
            offset += 7

        # Control field
        control = frame_body[offset]
        offset += 1
        if config.modulo == 128 and (control & 0x01 == 0):
            if offset >= len(frame_body):
                raise FrameError("Truncated extended control field")
            control |= frame_body[offset] << 8
            offset += 1

        # PID field
        pid = None
        if (control & 0x01 == 0) or ((control & 0xFF) == 0x03):
            if offset >= len(frame_body):
                raise FrameError("Truncated PID field")
            pid = frame_body[offset]
            offset += 1

        # Info and FCS
        info = frame_body[offset:-2]
        received_fcs = struct.unpack("<H", frame_body[-2:])[0]

        # Verify FCS
        frame_without_fcs = frame_body[:-2]
        if not verify_fcs(frame_without_fcs, received_fcs):
            raise FCSError("Invalid FCS")

        return cls(
            destination=dest,
            source=src,
            digipeaters=digipeaters,
            control=control,
            pid=pid,
            info=info,
            config=config,
        )
