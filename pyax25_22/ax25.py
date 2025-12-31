# PyAX25_22/ax25.py
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR
#
#
# AX.25 v2.2 framing implementation
# Bit stuffing, FCS calculation, address encoding
# Used for UI frames in PACSAT broadcast
#
# This file is part of PyAX25_22.
#

import struct
from typing import List

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
    return fcs ^ 0xFFFF

def bit_stuff(data: bytes) -> bytes:
    """Apply bit stuffing to prevent flag sequences."""
    stuffed = bytearray()
    ones_count = 0
    for byte in data:
        for i in range(8):
            bit = (byte >> i) & 1
            stuffed.append(bit)
            if bit == 1:
                ones_count += 1
                if ones_count == 5:
                    stuffed.append(0)
                    ones_count = 0
            else:
                ones_count = 0
    return bytes(stuffed)

def bit_destuff(data: bytes) -> bytes:
    """Remove bit stuffing."""
    destuffed = bytearray()
    ones_count = 0
    for bit in data:
        if bit == 1:
            ones_count += 1
            if ones_count != 5:
                destuffed.append(1)
        else:
            if ones_count != 5:
                destuffed.append(0)
            ones_count = 0
    return bytes(destuffed)

def encode_address(call: str, ssid: int = 0, c_bit: int = 0, last: int = 0) -> bytes:
    """Encode AX.25 address field."""
    call = call.ljust(6)[:6].upper()
    addr = bytearray()
    for c in call:
        addr.append(ord(c) << 1)
    addr.append((ssid << 1) | c_bit | (last << 7) | 0x60)  # HDLC address extension
    return bytes(addr)

def encode_ui_frame(dest: str, src: str, pid: int, info: bytes) -> bytes:
    """Encode complete AX.25 UI frame."""
    dest_addr = encode_address(dest, last=0)
    src_addr = encode_address(src, last=1)
    
    control = 0x03  # UI frame
    protocol = pid
    
    frame_data = dest_addr + src_addr + bytes([control, protocol]) + info
    
    fcs = fcs_calc(frame_data)
    frame_data += struct.pack('<H', fcs)
    
    # Bit stuffing
    stuffed = bit_stuff(frame_data)
    
    # Flags
    frame = bytes([FLAG]) + stuffed + bytes([FLAG])
    
    return frame

def decode_ui_frame(frame: bytes):
    """Decode AX.25 UI frame (basic)."""
    if frame[0] != FLAG or frame[-1] != FLAG:
        raise ValueError("Invalid flags")
    
    data = frame[1:-1]
    destuffed = bit_destuff(data)
    
    if len(destuffed) < 16:
        raise ValueError("Frame too short")
    
    # Extract addresses (14 bytes each)
    dest = destuffed[:7]
    src = destuffed[7:14]
    
    control = destuffed[14]
    pid = destuffed[15]
    info = destuffed[16:-2]  # Remove FCS
    received_fcs = struct.unpack('<H', destuffed[-2:])[0]
    
    calculated_fcs = fcs_calc(destuffed[:-2])
    if calculated_fcs != received_fcs:
        raise ValueError("FCS mismatch")
    
    return {
        "dest": dest,
        "src": src,
        "control": control,
        "pid": pid,
        "info": info
    }

# DOC: AX.25 framing utilities for PACSAT unproto UI frames
# DOC: Implements bit stuffing, FCS, address encoding per AX.25 v2.2
# DOC: Used by radio interfaces for packet transmission
