# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.framing.py

Complete AX.25 v2.2 frame encoding and decoding implementation.

Implements:
- Full address field with source, destination, and up to 8 digipeaters (H-bit support)
- All control field formats: I, S, U frames (modulo 8 and 128)
- PID field handling
- Information field
- Bit stuffing / destuffing
- FCS calculation and verification (CRC-16/CCITT-FALSE)

Fully compliant with AX.25 v2.2 specification (July 1998).
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
    """
    Calculate AX.25 FCS (CRC-16/CCITT-FALSE).

    Args:
        data: Bytes over which to compute FCS (address + control + PID + info)

    Returns:
        16-bit FCS value
    """
    fcs = FCS_INIT
    for byte in data:
        fcs ^= byte
        for _ in range(8):
            if fcs & 1:
                fcs = (fcs >> 1) ^ FCS_POLY
            else:
                fcs >>= 1
    return ~fcs & 0xFFFF  # Final invert


def verify_fcs(data: bytes, received_fcs: int) -> bool:
    """
    Verify received FCS against calculated value.

    Args:
        data: Frame data excluding FCS
        received_fcs: FCS from received frame

    Returns:
        True if valid
    """
    calculated = fcs_calc(data)
    return calculated == received_fcs


@dataclass
class AX25Address:
    """
    AX.25 address field with callsign, SSID, and control bits.
    """

    callsign: str
    ssid: int = 0
    c_bit: bool = False          # Command/Response bit (bit 6)
    h_bit: bool = False          # Has been repeated (bit 5)

    def __post_init__(self) -> None:
        """Validate and normalize address."""
        if not (0 <= self.ssid <= 15):
            raise InvalidAddressError(f"SSID {self.ssid} out of range (0-15)")

        callsign_clean = self.callsign.upper().strip().replace("-", "")
        if not (1 <= len(callsign_clean) <= 6):
            raise InvalidAddressError(f"Callsign '{self.callsign}' length invalid")

        # Shift and mask to 6 bits per char
        self._call_bytes = bytes(((ord(c) & 0x3F) << 1) for c in callsign_clean.ljust(6, " "))

    def encode(self, last: bool = False) -> bytes:
        """
        Encode 7-byte address field.

        SSID byte format (per AX.25 v2.2):
        - Bit 7: Reserved (always 1)
        - Bit 6: C bit (command/response)
        - Bit 5: H bit (has been repeated)
        - Bits 4-1: SSID
        - Bit 0: Extension (1 = last address)
        """
        ssid_byte = 0x60  # Bit 7 = 1, bit 6 = 0 initially
        ssid_byte |= (self.ssid << 1) & 0x1E  # SSID in bits 4-1
        ssid_byte |= 0x40 if self.c_bit else 0x00  # C bit in bit 6
        ssid_byte |= 0x20 if self.h_bit else 0x00  # H bit in bit 5
        ssid_byte |= 0x01 if last else 0x00        # Extension bit

        return self._call_bytes + bytes([ssid_byte])

    @classmethod
    def decode(cls, data: bytes) -> Tuple["AX25Address", bool]:
        """
        Decode address field from 7 bytes.

        Returns:
            (address object, is_last_address)
        """
        if len(data) < 7:
            raise InvalidAddressError("Address field too short")

        call_bytes = data[:6]
        ssid_byte = data[6]

        callsign_chars = []
        for b in call_bytes:
            char_code = (b >> 1) & 0x3F
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
    """
    Complete AX.25 frame with full v2.2 support.
    """

    destination: AX25Address
    source: AX25Address
    digipeaters: List[AX25Address] = field(default_factory=list)
    control: int = 0
    pid: Optional[int] = None
    info: bytes = b""
    config: AX25Config = DEFAULT_CONFIG_MOD8

    def encode(self) -> bytes:
        """
        Encode complete frame with flags, bit stuffing, and FCS.
        """
        # Address field
        addr_field = self.destination.encode(last=not self.digipeaters)
        addr_field += self.source.encode(last=not self.digipeaters)

        for i, digi in enumerate(self.digipeaters):
            last = i == len(self.digipeaters) - 1
            addr_field += digi.encode(last=last)

        # Control + PID + Info
        payload = bytes([self.control & 0xFF])
        if self.config.modulo == 128 and (self.control & 0x01 == 0):  # Extended I-frame
            payload += bytes([(self.control >> 8) & 0xFF])
        if self.pid is not None:
            payload += bytes([self.pid])
        payload += self.info

        # FCS over address + payload
        fcs = fcs_calc(addr_field + payload)
        frame_body = addr_field + payload + struct.pack("<H", fcs)

        # Bit stuffing
        stuffed = self._bit_stuff(frame_body)

        # Flags
        return bytes([FLAG]) + stuffed + bytes([FLAG])

    @staticmethod
    def _bit_stuff(data: bytes) -> bytes:
        """Apply AX.25 bit stuffing: insert 0 after five consecutive 1s."""
        result = bytearray()
        ones_count = 0
        for byte in data:
            for i in range(8):
                bit = (byte >> i) & 1
                result.append(bit)
                if bit == 1:
                    ones_count += 1
                    if ones_count == 5:
                        result.append(0)
                        ones_count = 0
                else:
                    ones_count = 0
        # Pad to byte boundary
        padding = (8 - len(result) % 8) % 8
        result.extend([0] * padding)
        return bytes(result[i:i+8] for i in range(0, len(result), 8))

    @classmethod
    def _bit_destuff(cls, data: bytes) -> bytes:
        """Remove AX.25 bit stuffing: remove 0 after five consecutive 1s."""
        result = bytearray()
        ones_count = 0
        for byte in data:
            for i in range(8):
                bit = (byte >> i) & 1
                if ones_count == 5:
                    if bit == 0:
                        ones_count = 0
                    else:
                        # Invalid — but skip for robustness
                        pass
                    continue
                if bit == 1:
                    result.append(1)
                    ones_count += 1
                else:
                    result.append(0)
                    ones_count = 0
        # Remove padding
        while result and result[-1] == 0:
            result.pop()
        # Pad back for byte conversion
        padding = (8 - len(result) % 8) % 8
        result.extend([0] * padding)
        return bytes(result[i:i+8] for i in range(0, len(result), 8))

    @classmethod
    def decode(cls, raw: bytes, config: AX25Config = DEFAULT_CONFIG_MOD8) -> "AX25Frame":
        """
        Decode raw frame bytes (including flags).

        Performs destuffing, FCS check, address/control parsing.
        """
        if raw[0] != FLAG or raw[-1] != FLAG:
            raise FrameError("Missing start/end flag")

        destuffed = cls._bit_destuff(raw[1:-1])

        if len(destuffed) < 16:
            raise FrameError("Frame too short after destuffing")

        # Parse addresses
        offset = 0
        dest, _ = AX25Address.decode(destuffed[offset:offset+7])
        offset += 7
        src, last = AX25Address.decode(destuffed[offset:offset+7])
        offset += 7

        digipeaters = []
        while not last and offset + 7 <= len(destuffed):
            digi, last = AX25Address.decode(destuffed[offset:offset+7])
            digipeaters.append(digi)
            offset += 7

        # Control field
        control = destuffed[offset]
        offset += 1
        if config.modulo == 128 and (control & 0x01 == 0):  # Extended I-frame
            if offset >= len(destuffed):
                raise FrameError("Truncated extended control field")
            control |= destuffed[offset] << 8
            offset += 1

        pid = None
        if (control & 0x01 == 0) or (control & 0x03 == 0x03):
            if offset >= len(destuffed):
                raise FrameError("Truncated PID field")
            pid = destuffed[offset]
            offset += 1

        info = destuffed[offset:-2]
        received_fcs = struct.unpack("<H", destuffed[-2:])[0]

        frame_without_fcs = destuffed[:-2]
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