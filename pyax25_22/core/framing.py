# pyax25_22/core/framing.py
"""
AX.25 Frame Encoding/Decoding (v2.2)

Handles:
- Address encoding/decoding
- All frame types (UI, I, SABM, DISC, etc.)
+ Bit stuffing/destuffing
- FCS calculation
- Complete frame parsing

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
MAX_DIGIPEATERS = 8

class FrameType(Enum):
    """AX.25 Frame Types"""
    UI = 'UI'  # Unnumbered Information
    SABM = 'SABM'  # Set Async Balanced Mode
    DISC = 'DISC'  # Disconnect
    DM = 'DM'  # Disconnected Mode
    UA = 'UA'  # Unnumbered Acknowledgment
    I = 'I'  # Information
    RR = 'RR'  # Receive Ready
    RNR = 'RNR'  # Receive Not Ready
    REJ = 'REJ'  # Reject
    SREJ = 'SREJ'  # Selective Reject

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
            raise ValueError(f"Invalid SSID {ssid}, must be 0-15")
        if len(callsign) < 1 or len(callsign) > 6:
            raise ValueError("Callsign must be 1-6 characters")
        self.callsign = callsign.upper().ljust(6)
        self.ssid = ssid
        self.c_bit = c_bit  # Command/Response
        self.r_bit = r_bit  # Reserved

    def __repr__(self) -> str:
        return (f"AX25Address(callsign='{self.callsign.strip()}', "
                f"ssid={self.ssid}, c_bit={self.c_bit})")

    @classmethod
    def from_bytes(cls, data: bytes) -> Tuple['AX25Address', bool]:
        """Parse 7-byte AX.25 address field"""
        if len(data) != 7:
            raise ValueError("Address must be 7 bytes")
        
        callsign = bytes([b >> 1 for b in data[:6]]).decode().strip()
        ctrl = data[6]
        ssid = (ctrl >> 1) & 0x0F
        last = bool(ctrl & 0x01)
        c_bit = bool(ctrl & 0x80)
        r_bit = bool(ctrl & 0x40)
        
        return cls(callsign, ssid, c_bit, r_bit), last

    def encoded(self, last: bool = False) -> bytes:
        """Return encoded 7-byte address"""
        callsign_bytes = bytes([ord(c) << 1 for c in self.callsign])
        ctrl = (self.ssid << 1) | (last << 0)
        ctrl |= 0x40  # HDLC address extension
        if self.c_bit:
            ctrl |= 0x80
        if self.r_bit:
            ctrl |= 0x40
        return callsign_bytes + bytes([ctrl])

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
        for i in reversed(range(8)):  # MSB first
            bit = (byte >> i) & 1
            stuffed.append(bit)
            ones = ones + 1 if bit else 0
            if ones == 5:
                stuffed.append(0)
                ones = 0
    # Pack bits into bytes
    result = bytearray()
    for i in range(0, len(stuffed), 8):
        byte = 0
        bits = stuffed[i:i+8]
        for j, bit in enumerate(bits):
            byte |= bit << (7 - j)
        result.append(byte)
    return bytes(result)

def bit_destuff(data: bytes) -> bytes:
    """Remove AX.25 bit stuffing"""
    destuffed = bytearray()
    ones = 0
    for byte in data:
        for i in reversed(range(8)):  # MSB first
            bit = (byte >> i) & 1
            if ones == 5:
                if bit:
                    raise ValueError("Consecutive 1s violation")
                ones = 0
                continue
            destuffed.append(bit)
            ones = ones + 1 if bit else 0
    # Pack bits into bytes
    result = bytearray()
    for i in range(0, len(destuffed), 8):
        byte = 0
        bits = destuffed[i:i+8]
        for j, bit in enumerate(bits):
            byte |= bit << (7 - j)
        result.append(byte)
    return bytes(result)

class AX25Frame:
    """AX.25 Frame builder and parser"""
    def __init__(self, 
                 dest: AX25Address,
                 src: AX25Address,
                 digipeaters: Optional[List[AX25Address]] = None,
                 pid: PID = PID.NO_LAYER3):
        self.dest = dest
        self.src = src
        self.digipeaters = digipeaters or []
        if len(self.digipeaters) > MAX_DIGIPEATERS:
            raise ValueError(f"Max {MAX_DIGIPEATERS} digipeaters allowed")
        self.pid = pid
        self.control = 0x03  # Default UI frame
        self.ns = 0  # Send sequence number
        self.nr = 0  # Receive sequence number
        self.poll = False
        self.type = FrameType.UI

    @classmethod
    def from_bytes(cls, data: bytes) -> 'AX25Frame':
        """Parse frame from raw bytes"""
        if len(data) < 32:  # Min frame size (2 flags, 14 addr, 1 ctrl, 1 pid, 2 fcs)
            raise ValueError("Frame too short")
        if data[0] != FLAG or data[-1] != FLAG:
            raise ValueError("Missing frame flags")
        
        # Destuff and verify FCS
        unstuffed = bit_destuff(data[1:-1])
        frame_data = unstuffed[:-2]
        fcs = fcs_calc(frame_data)
        if fcs != struct.unpack('<H', unstuffed[-2:])[0]:
            raise ValueError("FCS mismatch")
        
        # Parse addresses
        addrs = []
        pos = 0
        last = False
        while not last and pos + 7 <= len(frame_data):
            addr, last = AX25Address.from_bytes(frame_data[pos:pos+7])
            addrs.append(addr)
            pos += 7
        
        if len(addrs) < 2:
            raise ValueError("Missing source/destination addresses")
        
        # Frame components
        dest = addrs[0]
        src = addrs[1]
        digis = addrs[2:] if len(addrs) > 2 else []
        control = frame_data[pos]
        pos += 1
        
        # Determine frame type
        frame = cls(dest, src, digis)
        frame._parse_control(control)
        
        if frame.type in [FrameType.UI, FrameType.I]:
            if pos >= len(frame_data):
                raise ValueError("Missing PID")
            frame.pid = PID(frame_data[pos])
            pos += 1
            frame.info = frame_data[pos:]
        else:
            frame.info = frame_data[pos:] if pos < len(frame_data) else b''
        
        return frame

    def _parse_control(self, control: int) -> None:
        """Set frame type based on control byte"""
        if (control & 0x01) == 0:
            self.type = FrameType.I
            self.ns = (control >> 1) & 0x07
            self.nr = (control >> 5) & 0x07
            self.poll = bool(control & 0x10)
        else:
            if (control & 0x0F) == 0x03:
                self.type = FrameType.UI
            elif (control & 0x0F) == 0x0F:
                self.type = FrameType.DM
            elif (control & 0x0F) == 0x2F:
                self.type = FrameType.SABM
            elif (control & 0x0F) == 0x43:
                self.type = FrameType.DISC
            elif (control & 0x0F) == 0x63:
                self.type = FrameType.UA
            else:
                # Supervisory frames
                nr = (control >> 5) & 0x07
                poll = bool(control & 0x10)
                if (control & 0x0F) == 0x01:
                    self.type = FrameType.RR
                elif (control & 0x0F) == 0x05:
                    self.type = FrameType.RNR
                elif (control & 0x0F) == 0x09:
                    self.type = FrameType.REJ
                elif (control & 0x0F) == 0x0D:
                    self.type = FrameType.SREJ
                else:
                    raise ValueError(f"Unknown control byte: 0x{control:02x}")
                self.nr = nr
                self.poll = poll
        self.control = control

    def encode(self) -> bytes:
        """Build frame based on current type"""
        if self.type == FrameType.UI:
            return self.encode_ui()
        elif self.type == FrameType.I:
            return self.encode_i()
        elif self.type == FrameType.SABM:
            return self.encode_sabm()
        # ... handle other frame types
        else:
            raise NotImplementedError(f"Encoding for {self.type} not implemented")

    def encode_ui(self, info: bytes = b'') -> bytes:
        """Build Unnumbered Information frame"""
        self.type = FrameType.UI
        self.control = 0x03 | (self.poll << 4)
        return self._encode_frame(info)

    def encode_i(self, info: bytes = b'', ns: int = 0, nr: int = 0,
                 poll: bool = False) -> bytes:
        """Build Information frame"""
        self.type = FrameType.I
        self.ns = ns & 0x07
        self.nr = nr & 0x07
        self.poll = poll
        self.control = (self.ns << 1) | (self.nr << 5) | (self.poll << 4)
        return self._encode_frame(info)

    def encode_sabm(self, poll: bool = False) -> bytes:
        """Build Set Async Balanced Mode frame"""
        self.type = FrameType.SABM
        self.poll = poll
        self.control = 0x2F | (self.poll << 4)
        return self._encode_frame(b'')

    def encode_disc(self, poll: bool = False) -> bytes:
        """Build Disconnect frame"""
        self.type = FrameType.DISC
        self.poll = poll
        self.control = 0x43 | (self.poll << 4)
        return self._encode_frame(b'')

    def encode_dm(self, poll: bool = False) -> bytes:
        """Build Disconnected Mode frame"""
        self.type = FrameType.DM
        self.poll = poll
        self.control = 0x0F | (self.poll << 4)
        return self._encode_frame(b'')

    def encode_ua(self, poll: bool = False) -> bytes:
        """Build Unnumbered Acknowledgment frame"""
        self.type = FrameType.UA
        self.poll = poll
        self.control = 0x63 | (self.poll << 4)
        return self._encode_frame(b'')

    def _encode_frame(self, info: bytes = b'') -> bytes:
        """Internal frame construction"""
        frame = bytearray()
        
        # Addresses
        frame += self.dest.encoded()
        frame += self.src.encoded(last=not self.digipeaters)
        for i, digi in enumerate(self.digipeaters):
            last = i == len(self.digipeaters) - 1
            frame += digi.encoded(last=last)
        
        # Control field
        frame.append(self.control)
        
        # PID for certain frame types
        if self.type in [FrameType.UI, FrameType.I]:
            frame.append(self.pid)
        
        # Information field
        frame += info
        
        # FCS
        fcs = fcs_calc(frame)
        frame += struct.pack('<H', fcs)
        
        # Bit stuffing and flags
        stuffed = bit_stuff(frame)
        return bytes([FLAG]) + stuffed + bytes([FLAG])

    def __repr__(self) -> str:
        return (f"AX25Frame(dest={self.dest}, src={self.src}, "
                f"type={self.type}, pid={self.pid}, "
                f"digis={len(self.digipeaters)}, "
                f"info_len={len(getattr(self, 'info', b''))})")
