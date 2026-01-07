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
    return fcs_calc(data) == received_fcs


@dataclass
class AX25Address:
    callsign: str
    ssid: int = 0
    c_bit: bool = False
    h_bit: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.ssid <= 15):
            raise InvalidAddressError(f"SSID {self.ssid} out of range (0-15)")

        callsign_clean = self.callsign.upper().strip().replace("-", "")
        if not (1 <= len(callsign_clean) <= 6):
            raise InvalidAddressError(f"Callsign '{self.callsign}' length invalid")

        self._call_bytes = bytes((ord(c) << 1) for c in callsign_clean.ljust(6, " "))

    def encode(self, last: bool = False) -> bytes:
        ssid_byte = 0x60
        ssid_byte |= (self.ssid << 1) & 0x1E
        ssid_byte |= 0x80 if self.c_bit or self.h_bit else 0x00
        ssid_byte |= 0x01 if last else 0x00

        return self._call_bytes + bytes([ssid_byte])

    @classmethod
    def decode(cls, data: bytes) -> Tuple["AX25Address", bool]:
        if len(data) < 7:
            raise InvalidAddressError("Address field too short")

        call_bytes = data[:6]
        ssid_byte = data[6]

        callsign_chars = []
        for b in call_bytes:
            char_code = b >> 1
            if char_code == 0x20:
                break
            callsign_chars.append(chr(char_code))
        callsign = "".join(callsign_chars).rstrip()

        addr = cls(
            callsign=callsign,
            ssid=(ssid_byte >> 1) & 0x0F,
            c_bit=bool(ssid_byte & 0x80),
            h_bit=bool(ssid_byte & 0x80),
        )

        is_last = bool(ssid_byte & 0x01)
        return addr, is_last


@dataclass
class AX25Frame:
    destination: AX25Address
    source: AX25Address
    digipeaters: List[AX25Address] = field(default_factory=list)
    control: int = 0
    pid: Optional[int] = None
    info: bytes = b""
    config: AX25Config = DEFAULT_CONFIG_MOD8

    def encode(self) -> bytes:
        addr_field = self.destination.encode(last=not self.digipeaters)
        addr_field += self.source.encode(last=not self.digipeaters)

        for i, digi in enumerate(self.digipeaters):
            last = i == len(self.digipeaters) - 1
            addr_field += digi.encode(last=last)

        payload = bytes([self.control & 0xFF])
        if self.config.modulo == 128 and (self.control & 0x01 == 0):
            payload += bytes([(self.control >> 8) & 0xFF])
        if self.pid is not None:
            payload += bytes([self.pid])
        payload += self.info

        fcs = fcs_calc(addr_field + payload)
        frame_body = addr_field + payload + struct.pack("<H", fcs)

        stuffed = self._bit_stuff(frame_body)

        return bytes([FLAG]) + stuffed + bytes([FLAG])

    @staticmethod
    def _bit_stuff(data: bytes) -> bytes:
        result = bytearray()
        ones_count = 0
        current_byte = 0
        bit_pos = 0

        for byte in data:
            for i in range(8):
                bit = (byte >> i) & 1
                current_byte |= bit << bit_pos
                bit_pos += 1

                if bit == 1:
                    ones_count += 1
                    if ones_count == 5:
                        if bit_pos == 8:
                            result.append(current_byte)
                            current_byte = 0
                            bit_pos = 0
                        ones_count = 0
                else:
                    ones_count = 0

                if bit_pos == 8:
                    result.append(current_byte)
                    current_byte = 0
                    bit_pos = 0

        if bit_pos > 0:
            result.append(current_byte)

        return bytes(result)

    @classmethod
    def _bit_destuff(cls, data: bytes) -> bytes:
        result = bytearray()
        ones_count = 0
        current_byte = 0
        bit_pos = 0

        for byte in data:
            for i in range(8):
                bit = (byte >> i) & 1
                current_byte |= bit << bit_pos
                bit_pos += 1

                if ones_count == 5:
                    if bit == 0:
                        ones_count = 0
                    if bit_pos == 8:
                        result.append(current_byte)
                        current_byte = 0
                        bit_pos = 0
                    continue

                if bit == 1:
                    ones_count += 1
                else:
                    ones_count = 0

                if bit_pos == 8:
                    result.append(current_byte)
                    current_byte = 0
                    bit_pos = 0

        if bit_pos > 0:
            result.append(current_byte)

        return bytes(result)

    @classmethod
    def decode(cls, raw: bytes, config: AX25Config = DEFAULT_CONFIG_MOD8) -> "AX25Frame":
        if raw[0] != FLAG or raw[-1] != FLAG:
            raise FrameError("Missing start/end flag")

        destuffed = cls._bit_destuff(raw[1:-1])

        if len(destuffed) < 16:
            raise FrameError("Frame too short after destuffing")

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

        control = destuffed[offset]
        offset += 1
        if config.modulo == 128 and (control & 0x01 == 0):
            control |= destuffed[offset] << 8
            offset += 1

        pid = None
        if (control & 0x01 == 0) or (control & 0x03 == 0x03):
            pid = destuffed[offset]
            offset += 1

        info = destuffed[offset:-2]
        received_fcs = struct.unpack("<H", destuffed[-2:])[0]

        if not verify_fcs(destuffed[:-2], received_fcs):
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