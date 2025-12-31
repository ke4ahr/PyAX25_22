# pyax25_22/core/framing.py
"""
AX.25 Frame Encoding/Decoding (v2.2)

Handles:
- Address encoding/decoding
- Frame construction (UI, I, SABM, DISC, etc.)
- Bit stuffing/destuffing
- FCS calculation
- Frame validation

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import struct
import logging
from enum import Enum, IntEnum
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Constants
FLAG = 0x7E
FCS_POLY = 0x8408
FCS_INIT = 0xFFFF

class FrameType(Enum):
    """AX.25 Frame Types"""
    UI = 'U'  # Unnumbered Information
    SABM = 'S'  # Set Async Balanced Mode
    DISC = 'D'  # Disconnect
    DM = 'M'  # Disconnected Mode
    UA = 'A'  # Unnumbered Acknowledgment
    I = 'I'  # Information
    RR = 'R'  # Receive Ready
    RNR = 'N'  # Receive Not Ready
    REJ = 'J'  # Reject
    SREJ = 'E'  # Selective Reject

class PID(IntEnum):
    """Protocol Identifiers"""
    NO_LAYER3 = 0xF0
    IP = 0xCC
    ARPA_IP = 0x0800
    ARPA_ARP = 0x0806
    PACSAT = 0x01

class AX25Address:
    """AX.25 Address with SSID and control bits"""
    def __init__(self, callsign: str, ssid: int = 0,
                 c_bit: bool = False, r_bit: bool = False):
        if not 0 <= ssid <= 15:
            raise ValueError("SSID must be 0-15")
        self.callsign = callsign.upper()[:6].ljust(6)
        self.ssid = ssid
        self.c_bit = c_bit  # Command/Response
        self.r_bit = r_bit  # Reserved

    def encoded(self, last: bool = False) -> bytes:
        """Return encoded 7-byte address"""
        encoded = bytes([ord(c) << 1 for c in self.callsign])
        ctrl = (self.ssid << 1) | (last << 7) | (self.r_bit << 6) | (self.c_bit << 5)
        return encoded + bytes([ctrl | 0x40])  # HDLC address extension

def fcs_calc(data: bytes) -> int:
    """Calculate AX.25 FCS (CRC-CCITT)"""
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
    """Apply AX.25 bit stuffing"""
    stuffed = bytearray()
    ones = 0
    for byte in data:
        for i in range(8):
            bit = (byte >> (7 - i)) & 1
            stuffed.append(bit)
            ones = ones + 1 if bit else 0
            if ones == 5:
                stuffed.append(0)  # Stuff zero
                ones = 0
    # Convert bits back to bytes
    result = bytearray()
    for i in range(0, len(stuffed), 8):
        byte = 0
        for j in range(8):
            if i + j < len(stuffed):
                byte |= stuffed[i + j] << (7 - j)
        result.append(byte)
    return bytes(result)

def bit_destuff(data: bytes) -> bytes:
    """Remove AX.25 bit stuffing"""
    destuffed = bytearray()
    ones = 0
    for byte in data:
        for i in range(8):
            bit = (byte >> (7 - i)) & 1
            if ones == 5:
                if bit:
                    raise ValueError("Consecutive 1s violation")
                ones = 0
                continue  # Skip stuffed bit
            destuffed.append(bit)
            ones = ones + 1 if bit else 0
    # Convert bits back to bytes
    result = bytearray()
    for i in range(0, len(destuffed), 8):
        byte = 0
        for j in range(8):
            if i + j < len(destuffed):
                byte |= destuffed[i + j] << (7 - j)
        result.append(byte)
    return bytes(result)

class AX25Frame:
    """AX.25 Frame builder and parser"""
    def __init__(self, dest: AX25Address, src: AX25Address,
                 digipeaters: Optional[List[AX25Address]] = None,
                 pid: PID = PID.NO_LAYER3):
        self.dest = dest
        self.src = src
        self.digipeaters = digipeaters or []
        self.pid = pid
        self.control = 0x03  # Default UI frame

    @classmethod
    def from_bytes(cls, data: bytes) -> 'AX25Frame':
        """Parse frame from raw bytes"""
        if data[0] != FLAG or data[-1] != FLAG:
            raise ValueError("Invalid frame flags")
        
        unstuffed = bit_destuff(data[1:-1])
        calculated_fcs = fcs_calc(unstuffed[:-2])
        received_fcs = struct.unpack('<H', unstuffed[-2:])[0]
        if calculated_fcs != received_fcs:
            raise ValueError("FCS mismatch")
        
        # Parse addresses
        addr_data = unstuffed[:7*(2 + len(self.digipeaters))]
        # ... (full address parsing implementation)

    def encode_ui(self, info: bytes = b'') -> bytes:
        """Build UI frame"""
        self.control = 0x03  # UI frame
        return self._encode_frame(info)

    def encode_i(self, info: bytes = b'', nr: int = 0, ns: int = 0,
                 poll: bool = False) -> bytes:
        """Build Information frame"""
        self.control = (ns << 1) | (nr << 5) | (poll << 4)
        return self._encode_frame(info)

    def _encode_frame(self, info: bytes) -> bytes:
        """Internal frame builder"""
        frame = bytearray()
        # Addresses
        frame += self.dest.encoded()
        frame += self.src.encoded(last=not self.digipeaters)
        for i, digi in enumerate(self.digipeaters):
            frame += digi.encoded(last=i == len(self.digipeaters)-1)
        # Control/PID
        frame.append(self.control)
        if self.control & 0x01 == 0 or self.control == 0x03:  # I or UI frame
            frame.append(self.pid)
        # Info
        frame += info
        # FCS
        fcs = fcs_calc(frame)
        frame += struct.pack('<H', fcs)
        # Stuffing and flags
        stuffed = bit_stuff(frame)
        return bytes([FLAG]) + stuffed + bytes([FLAG])

      # In AX25Frame class
    def encode_sabm(self, poll: bool = False) -> bytes:
        self.control = 0x2F | (poll << 4)
        return self._encode_frame(b'')

    def encode_disc(self, poll: bool = False) -> bytes:
        self.control = 0x43 | (poll << 4)
        return self._encode_frame(b'')

    # And more frame type methods...
