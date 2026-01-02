# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
AX.25 Frame Encoding/Decoding (v2.2)

Handles:
- Address encoding/decoding
- All frame types (UI, I, SABM, DISC, etc.)
- Bit stuffing/destuffing
- FCS calculation
- Complete frame parsing

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import struct
import logging
from enum import Enum, IntEnum
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass

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
    XID = 'XID'  # Exchange Identification
    TEST = 'TEST'  # Test Frame

class PID(IntEnum):
    """Protocol Identifiers"""
    NO_LAYER3 = 0xF0
    IP = 0xCC
    ARPA_IP = 0x0800
    ARPA_ARP = 0x0806
    PACSAT = 0x01
    TEXNET = 0xC3
    NO_L3_EXT = 0x80
    LINK_QUALITY = 0xC4
    APPLETALK = 0x9B
    ARPA_RARP = 0x8035
    FLEXNET = 0xC2
    NETROM = 0x81

class AX25Address:
    """AX.25 Address with SSID and control bits"""
    
    def __init__(self, callsign: str, ssid: int = 0,
                 c_bit: bool = False, r_bit: bool = False):
        """Initialize AX.25 address.
        
        Args:
            callsign: Amateur radio callsign (1-6 characters)
            ssid: Secondary Station Identifier (0-15)
            c_bit: Command/Response bit
            r_bit: Reserved bit (should be 0 for AX.25 v2.2)
            
        Raises:
            ValueError: If callsign or SSID is invalid
        """
        if not isinstance(callsign, str) or len(callsign) < 1 or len(callsign) > 6:
            raise ValueError(f"Invalid callsign '{callsign}': must be 1-6 characters")
        if not 0 <= ssid <= 15:
            raise ValueError(f"Invalid SSID {ssid}: must be 0-15")
        if not isinstance(c_bit, bool):
            raise ValueError("C bit must be boolean")
        if not isinstance(r_bit, bool):
            raise ValueError("R bit must be boolean")
            
        self.callsign = callsign.upper().ljust(6)
        self.ssid = ssid
        self.c_bit = c_bit  # Command/Response
        self.r_bit = r_bit  # Reserved (should be 0 for AX.25 v2.2)
        
        logger.debug(f"Created AX25Address: {self.callsign.strip()}-{self.ssid}")

    def __repr__(self) -> str:
        """String representation of address."""
        return (f"AX25Address(callsign='{self.callsign.strip()}', "
                f"ssid={self.ssid}, c_bit={self.c_bit})")

    @classmethod
    def from_bytes(cls, data: bytes) -> Tuple['AX25Address', bool]:
        """Parse 7-byte AX.25 address field.
        
        Args:
            data: 7-byte address field from frame
            
        Returns:
            Tuple of (AX25Address, last_flag)
            
        Raises:
            ValueError: If data is not 7 bytes
        """
        if len(data) != 7:
            raise ValueError(f"Address must be 7 bytes, got {len(data)}")
        
        try:
            # Extract callsign (shift right by 1 bit)
            callsign_bytes = []
            for i in range(6):
                callsign_bytes.append(data[i] >> 1)
            
            callsign = bytes(callsign_bytes).decode('ascii').strip()
            
            # Extract control field
            ctrl = data[6]
            ssid = (ctrl >> 1) & 0x0F
            last = bool(ctrl & 0x01)
            c_bit = bool(ctrl & 0x80)
            r_bit = bool(ctrl & 0x40)
            
            logger.debug(f"Parsed address: {callsign}-{ssid}, last={last}")
            return cls(callsign, ssid, c_bit, r_bit), last
            
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid ASCII in address field: {e}") from e

    def encoded(self, last: bool = False) -> bytes:
        """Return encoded 7-byte address.
        
        Args:
            last: True if this is the last address in the field
            
        Returns:
            7-byte encoded address
        """
        try:
            # Encode callsign (shift left by 1 bit)
            callsign_bytes = bytearray()
            for char in self.callsign:
                callsign_bytes.append(ord(char) << 1)
            
            # Build control byte
            ctrl = (self.ssid << 1) | (last << 0)
            ctrl |= 0x40  # HDLC address extension bit
            if self.c_bit:
                ctrl |= 0x80
            if self.r_bit:
                ctrl |= 0x40
                
            encoded = bytes(callsign_bytes) + bytes([ctrl])
            logger.debug(f"Encoded address: {encoded.hex()}")
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to encode address {self.callsign}: {e}")
            raise

@dataclass
class FrameMetadata:
    """Metadata extracted from frame parsing."""
    frame_type: FrameType
    ns: int = 0      # Send sequence number
    nr: int = 0      # Receive sequence number
    poll: bool = False
    fcs_valid: bool = False
    pid: Optional[PID] = None
    info_length: int = 0

def fcs_calc(data: bytes) -> int:
    """Calculate AX.25 FCS (CRC-CCITT).
    
    Args:
        data: Frame data to calculate FCS for
        
    Returns:
        16-bit FCS value
        
    Example:
        >>> fcs_calc(b"test")
        0x8E72
    """
    if not isinstance(data, bytes):
        raise TypeError("Data must be bytes")
        
    fcs = FCS_INIT
    for byte in data:
        fcs ^= byte
        for _ in range(8):
            if fcs & 1:
                fcs = (fcs >> 1) ^ FCS_POLY
            else:
                fcs >>= 1
    result = fcs ^ 0xFFFF
    logger.debug(f"FCS for {len(data)} bytes: 0x{result:04X}")
    return result

def bit_stuff(data: bytes) -> bytes:
    """Apply AX.25 bit stuffing.
    
    Args:
        data: Raw frame data to stuff
        
    Returns:
        Bit-stuffed data
        
    Raises:
        TypeError: If data is not bytes
    """
    if not isinstance(data, bytes):
        raise TypeError("Data must be bytes")
        
    if not data:
        logger.debug("Empty data for bit stuffing")
        return data
        
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
        
    logger.debug(f"Bit stuffing: {len(data)} -> {len(result)} bytes")
    return bytes(result)

def bit_destuff(data: bytes) -> bytes:
    """Remove AX.25 bit stuffing.
    
    Args:
        data: Bit-stuffed data to destuff
        
    Returns:
        Destuffed data
        
    Raises:
        ValueError: If bit stuffing violation detected
        TypeError: If data is not bytes
    """
    if not isinstance(data, bytes):
        raise TypeError("Data must be bytes")
        
    if not data:
        logger.debug("Empty data for bit destuffing")
        return data
        
    destuffed = bytearray()
    ones = 0
    
    for byte in data:
        for i in reversed(range(8)):  # MSB first
            bit = (byte >> i) & 1
            if ones == 5:
                if bit:
                    raise ValueError("Consecutive 1s violation in bit destuffing")
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
        
    logger.debug(f"Bit destuffing: {len(data)} -> {len(result)} bytes")
    return bytes(result)

class AX25Frame:
    """AX.25 Frame builder and parser"""
    
    def __init__(self, 
                 dest: AX25Address,
                 src: AX25Address,
                 digipeaters: Optional[List[AX25Address]] = None,
                 pid: PID = PID.NO_LAYER3):
        """Initialize AX.25 frame.
        
        Args:
            dest: Destination address
            src: Source address
            digipeaters: List of digipeater addresses
            pid: Protocol Identifier
        """
        if not isinstance(dest, AX25Address):
            raise TypeError("Destination must be AX25Address")
        if not isinstance(src, AX25Address):
            raise TypeError("Source must be AX25Address")
        if digipeaters and not all(isinstance(d, AX25Address) for d in digipeaters):
            raise TypeError("All digipeaters must be AX25Address")
            
        self.dest = dest
        self.src = src
        self.digipeaters = digipeaters or []
        if len(self.digipeaters) > MAX_DIGIPEATERS:
            raise ValueError(f"Max {MAX_DIGIPEATERS} digipeaters allowed, got {len(self.digipeaters)}")
        self.pid = pid
        self.control = 0x03  # Default UI frame
        self.ns = 0  # Send sequence number
        self.nr = 0  # Receive sequence number
        self.poll = False
        self.type = FrameType.UI
        self.info = b''
        
        logger.debug(f"Created AX25Frame: {self.dest.callsign.strip()} <- {self.src.callsign.strip()}")

    @classmethod
    def from_bytes(cls, data: bytes) -> 'AX25Frame':
        """Parse frame from raw bytes.
        
        Args:
            data: Raw frame bytes including flags
            
        Returns:
            Parsed AX25Frame instance
            
        Raises:
            ValueError: If frame is malformed
            TypeError: If data is not bytes
        """
        if not isinstance(data, bytes):
            raise TypeError("Frame data must be bytes")
            
        if len(data) < 16:  # Min frame: 2 flags + dest + src + ctrl + FCS
            raise ValueError(f"Frame too short ({len(data)} bytes), minimum is 16")
        if data[0] != FLAG or data[-1] != FLAG:
            raise ValueError("Missing frame flags")
        
        try:
            # Remove flags and destuff
            unstuffed = bit_destuff(data[1:-1])
            frame_data = unstuffed[:-2]  # Remove FCS
            fcs_bytes = unstuffed[-2:]
            
            # Verify FCS
            calculated_fcs = fcs_calc(frame_data)
            stored_fcs = struct.unpack('<H', fcs_bytes)[0]
            
            if calculated_fcs != stored_fcs:
                logger.warning(f"FCS mismatch: calculated 0x{calculated_fcs:04X}, stored 0x{stored_fcs:04X}")
                raise ValueError("FCS mismatch")
            
            # Parse addresses
            addresses, addr_end = cls._parse_address_field(frame_data)
            
            if len(addresses) < 2:
                raise ValueError("Missing source/destination addresses")
            
            # Initialize frame
            frame = cls(addresses[0], addresses[1], addresses[2:])
            
            # Parse control field
            if addr_end >= len(frame_data):
                raise ValueError("Missing control field")
                
            control = frame_data[addr_end]
            frame._parse_control(control)
            
            # Parse information field
            info_start = addr_end + 1
            if frame.type in [FrameType.UI, FrameType.I]:
                if info_start >= len(frame_data):
                    raise ValueError("Missing PID field")
                frame.pid = PID(frame_data[info_start])
                info_start += 1
                frame.info = frame_data[info_start:] if info_start < len(frame_data) else b''
            else:
                frame.info = frame_data[info_start:] if info_start < len(frame_data) else b''
            
            logger.debug(f"Parsed frame: type={frame.type}, info_len={len(frame.info)}")
            return frame
            
        except Exception as e:
            logger.error(f"Frame parsing failed: {e}")
            raise

    @staticmethod
    def _parse_address_field(data: bytes) -> Tuple[List[AX25Address], int]:
        """Parse destination, source, and digipeater addresses.
        
        Args:
            data: Frame data starting with addresses
            
        Returns:
            Tuple of (address_list, end_position)
            
        Raises:
            ValueError: If address parsing fails
        """
        if len(data) < 14:  # At least dest + src
            raise ValueError("Insufficient data for address field")
            
        addresses = []
        pos = 0
        last = False
        
        while not last and pos + 7 <= len(data):
            try:
                addr, last = AX25Address.from_bytes(data[pos:pos+7])
                addresses.append(addr)
                pos += 7
                
                # For AX.25 v2.2, the last address bit should be set for the final address
                if last and pos >= len(data):
                    break
                elif last:
                    logger.warning("Last address bit set before end of address field")
                    
            except ValueError as e:
                logger.error(f"Address parsing error at position {pos}: {e}")
                raise
                
        if len(addresses) < 2:
            raise ValueError("At least destination and source addresses required")
            
        logger.debug(f"Parsed {len(addresses)} addresses")
        return addresses, pos

    def _parse_control(self, control: int) -> None:
        """Parse control byte and set frame type, sequence numbers, and flags.
        
        Args:
            control: Raw control byte from frame
            
        Raises:
            ValueError: If control byte is invalid
        """
        if not 0 <= control <= 255:
            raise ValueError(f"Invalid control byte: {control}")
            
        self.control = control
        
        # Check if it's an Information frame (low bit = 0)
        if (control & 0x01) == 0:
            self.type = FrameType.I
            self.ns = (control >> 1) & 0x7F  # Support both modulo 8 and 128
            self.nr = (control >> 8) & 0x7F  # Support both modulo 8 and 128
            self.poll = bool(control & 0x10)
            logger.debug(f"Information frame: NS={self.ns}, NR={self.nr}, P={self.poll}")
            
        # Check if it's an Unnumbered frame (bits 2-3 = 00)
        elif (control & 0x0F) == 0x03:
            self.type = FrameType.UI
            self.poll = bool(control & 0x10)
            logger.debug(f"UI frame: P={self.poll}")
            
        elif (control & 0x0F) == 0x0F:
            self.type = FrameType.DM
            self.poll = bool(control & 0x10)
            logger.debug(f"DM frame: P={self.poll}")
            
        elif (control & 0x0F) == 0x2F:
            self.type = FrameType.SABM
            self.poll = bool(control & 0x10)
            logger.debug(f"SABM frame: P={self.poll}")
            
        elif (control & 0x0F) == 0x43:
            self.type = FrameType.DISC
            self.poll = bool(control & 0x10)
            logger.debug(f"DISC frame: P={self.poll}")
            
        elif (control & 0x0F) == 0x63:
            self.type = FrameType.UA
            self.poll = bool(control & 0x10)
            logger.debug(f"UA frame: P={self.poll}")
            
        elif (control & 0x0F) == 0x8F:
            self.type = FrameType.FRMR
            self.poll = bool(control & 0x10)
            logger.debug(f"FRMR frame: P={self.poll}")
            
        # Supervisory frames (bits 2-3 = 01)
        elif (control & 0x0F) == 0x01:
            self.type = FrameType.RR
            self.nr = (control >> 5) & 0x7F
            self.poll = bool(control & 0x10)
            logger.debug(f"RR frame: NR={self.nr}, P={self.poll}")
            
        elif (control & 0x0F) == 0x05:
            self.type = FrameType.RNR
            self.nr = (control >> 5) & 0x7F
            self.poll = bool(control & 0x10)
            logger.debug(f"RNR frame: NR={self.nr}, P={self.poll}")
            
        elif (control & 0x0F) == 0x09:
            self.type = FrameType.REJ
            self.nr = (control >> 5) & 0x7F
            self.poll = bool(control & 0x10)
            logger.debug(f"REJ frame: NR={self.nr}, P={self.poll}")
            
        elif (control & 0x0F) == 0x0D:
            self.type = FrameType.SREJ
            self.nr = (control >> 5) & 0x7F
            self.poll = bool(control & 0x10)
            logger.debug(f"SREJ frame: NR={self.nr}, P={self.poll}")
            
        # Extended Unnumbered frames
        elif (control & 0x0F) == 0x23:
            self.type = FrameType.XID
            self.poll = bool(control & 0x10)
            logger.debug(f"XID frame: P={self.poll}")
            
        elif (control & 0x0F) == 0x47:
            self.type = FrameType.TEST
            self.poll = bool(control & 0x10)
            logger.debug(f"TEST frame: P={self.poll}")
            
        else:
            raise ValueError(f"Unknown control byte: 0x{control:02X}")

    def encode(self) -> bytes:
        """Complete frame encoding for all frame types.
        
        Returns:
            Complete frame with flags and FCS
            
        Raises:
            NotImplementedError: If frame type encoding is not implemented
        """
        try:
            if self.type == FrameType.UI:
                return self.encode_ui(self.info)
            elif self.type == FrameType.I:
                return self.encode_i(self.info, self.ns, self.nr, self.poll)
            elif self.type == FrameType.SABM:
                return self.encode_u(FrameType.SABM, self.poll)
            elif self.type == FrameType.DISC:
                return self.encode_u(FrameType.DISC, self.poll)
            elif self.type == FrameType.UA:
                return self.encode_u(FrameType.UA, self.poll)
            elif self.type == FrameType.DM:
                return self.encode_u(FrameType.DM, self.poll)
            elif self.type == FrameType.RR:
                return self.encode_s(FrameType.RR, self.nr, self.poll)
            elif self.type == FrameType.RNR:
                return self.encode_s(FrameType.RNR, self.nr, self.poll)
            elif self.type == FrameType.REJ:
                return self.encode_s(FrameType.REJ, self.nr, self.poll)
            elif self.type == FrameType.SREJ:
                return self.encode_s(FrameType.SREJ, self.nr, self.poll)
            elif self.type == FrameType.XID:
                return self.encode_u(FrameType.XID, self.poll)
            elif self.type == FrameType.TEST:
                return self.encode_u(FrameType.TEST, self.poll)
            else:
                raise NotImplementedError(f"Encoding for {self.type} not implemented")
                
        except Exception as e:
            logger.error(f"Frame encoding failed: {e}")
            raise

    def encode_s(self, frame_type: FrameType, nr: int = 0, poll: bool = False) -> bytes:
        """Encode supervisory frames (RR, RNR, REJ, SREJ).
        
        Args:
            frame_type: Type of supervisory frame
            nr: Receive sequence number
            poll: Poll flag
            
        Returns:
            Encoded frame bytes
        """
        if frame_type not in [FrameType.RR, FrameType.RNR, FrameType.REJ, FrameType.SREJ]:
            raise ValueError(f"Invalid supervisory frame type: {frame_type}")
            
        if not 0 <= nr <= 127:  # Support modulo 128
            raise ValueError(f"Invalid NR value: {nr}")
            
        # Build control byte
        control = 0x01  # Base for supervisory frames
        if frame_type == FrameType.RNR:
            control = 0x05
        elif frame_type == FrameType.REJ:
            control = 0x09
        elif frame_type == FrameType.SREJ:
            control = 0x0D
            
        control |= (nr << 5) & 0xE0  # NR in bits 5-7 (modulo 8)
        if nr > 7:  # Modulo 128 extension
            control |= 0x80  # Set E-bit
            # For full modulo 128, we'd need extended sequence numbering
            # This is a simplified implementation
        control |= (poll << 4)
        
        self.control = control
        self.type = frame_type
        self.nr = nr
        self.poll = poll
        
        return self._encode_frame(b'')

    def encode_u(self, frame_type: FrameType, poll: bool = False) -> bytes:
        """Encode unnumbered frames (SABM, DISC, UA, DM, XID, TEST).
        
        Args:
            frame_type: Type of unnumbered frame
            poll: Poll flag
            
        Returns:
            Encoded frame bytes
        """
        if frame_type not in [FrameType.SABM, FrameType.DISC, FrameType.UA, 
                             FrameType.DM, FrameType.XID, FrameType.TEST]:
            raise ValueError(f"Invalid unnumbered frame type: {frame_type}")
            
        # Build control byte
        control = 0x03  # Default UI
        if frame_type == FrameType.SABM:
            control = 0x2F
        elif frame_type == FrameType.DISC:
            control = 0x43
        elif frame_type == FrameType.UA:
            control = 0x63
        elif frame_type == FrameType.DM:
            control = 0x0F
        elif frame_type == FrameType.XID:
            control = 0x23
        elif frame_type == FrameType.TEST:
            control = 0x47
            
        control |= (poll << 4)
        
        self.control = control
        self.type = frame_type
        self.poll = poll
        
        return self._encode_frame(b'')

    def encode_i(self, info: bytes, ns: int = 0, nr: int = 0, poll: bool = False) -> bytes:
        """Encode information frames with proper sequence numbers.
        
        Args:
            info: Information field data
            ns: Send sequence number
            nr: Receive sequence number
            poll: Poll flag
            
        Returns:
            Encoded frame bytes
        """
        if not isinstance(info, bytes):
            raise TypeError("Info must be bytes")
        if not 0 <= ns <= 127:
            raise ValueError(f"Invalid NS value: {ns}")
        if not 0 <= nr <= 127:
            raise ValueError(f"Invalid NR value: {nr}")
            
        # Build control byte for modulo 8
        control = (ns << 1) | 0x00  # I frame format
        control |= (nr << 5) & 0xE0  # NR in bits 5-7
        control |= (poll << 4)
        
        # For modulo 128, we need extended sequence numbering
        if ns > 7 or nr > 7:
            # This would require extended control field implementation
            # Simplified for now - would need to add P/F bit handling
            logger.warning(f"Modulo 128 sequence numbers not fully implemented: NS={ns}, NR={nr}")
        
        self.control = control
        self.type = FrameType.I
        self.ns = ns
        self.nr = nr
        self.poll = poll
        self.info = info
        
        return self._encode_frame(info)

    def encode_ui(self, info: bytes = b'') -> bytes:
        """Encode Unnumbered Information frame.
        
        Args:
            info: Information field data
            
        Returns:
            Encoded frame bytes
        """
        if not isinstance(info, bytes):
            raise TypeError("Info must be bytes")
            
        self.type = FrameType.UI
        self.control = 0x03 | (self.poll << 4)
        self.info = info
        
        return self._encode_frame(info)

    def _encode_frame(self, info: bytes = b'') -> bytes:
        """Internal frame construction.
        
        Args:
            info: Information field data
            
        Returns:
            Complete frame with flags and FCS
        """
        try:
            frame = bytearray()
            
            # Addresses
            frame.extend(self.dest.encoded())
            frame.extend(self.src.encoded(last=not self.digipeaters))
            for i, digi in enumerate(self.digipeaters):
                last = i == len(self.digipeaters) - 1
                frame.extend(digi.encoded(last=last))
            
            # Control field
            frame.append(self.control)
            
            # PID for certain frame types
            if self.type in [FrameType.UI, FrameType.I]:
                frame.append(self.pid)
            
            # Information field
            frame.extend(info)
            
            # FCS
            fcs = fcs_calc(frame)
            frame.extend(struct.pack('<H', fcs))
            
            # Bit stuffing and flags
            stuffed = bit_stuff(frame)
            result = bytes([FLAG]) + stuffed + bytes([FLAG])
            
            logger.debug(f"Encoded frame: {self.type.name}, {len(result)} bytes")
            return result
            
        except Exception as e:
            logger.error(f"Frame construction failed: {e}")
            raise

    def validate(self) -> bool:
        """Validate frame structure and content.
        
        Returns:
            True if frame is valid, False otherwise
        """
        try:
            # Basic validation
            if not isinstance(self.dest, AX25Address):
                logger.error("Invalid destination address")
                return False
            if not isinstance(self.src, AX25Address):
                logger.error("Invalid source address")
                return False
            if not isinstance(self.type, FrameType):
                logger.error("Invalid frame type")
                return False
                
            # Address validation
            if not self.dest.callsign.strip():
                logger.error("Empty destination callsign")
                return False
            if not self.src.callsign.strip():
                logger.error("Empty source callsign")
                return False
                
            # Sequence number validation for I frames
            if self.type == FrameType.I:
                if not 0 <= self.ns <= 127:
                    logger.error(f"Invalid NS: {self.ns}")
                    return False
                if not 0 <= self.nr <= 127:
                    logger.error(f"Invalid NR: {self.nr}")
                    return False
                    
            # Digipeater validation
            if len(self.digipeaters) > MAX_DIGIPEATERS:
                logger.error(f"Too many digipeaters: {len(self.digipeaters)}")
                return False
                
            logger.debug("Frame validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Frame validation failed: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of frame."""
        return (f"AX25Frame(dest={self.dest}, src={self.src}, "
                f"type={self.type}, pid={self.pid}, "
                f"digis={len(self.digipeaters)}, "
                f"info_len={len(self.info)})")

    def to_dict(self) -> Dict[str, Union[str, int, bytes]]:
        """Convert frame to dictionary for debugging.
        
        Returns:
            Dictionary representation of frame
        """
        return {
            'type': self.type.name,
            'destination': self.dest.callsign.strip(),
            'source': self.src.callsign.strip(),
            'digipeaters': [d.callsign.strip() for d in self.digipeaters],
            'control': self.control,
            'ns': self.ns,
            'nr': self.nr,
            'poll': self.poll,
            'pid': self.pid.name if self.pid else None,
            'info_length': len(self.info),
            'info_hex': self.info.hex() if self.info else ''
        }

    def is_valid_ui_frame(self) -> bool:
        """Check if frame is valid UI frame.
        
        Returns:
            True if valid UI frame, False otherwise
        """
        return (self.type == FrameType.UI and 
                self.dest.callsign.strip() and 
                self.src.callsign.strip())

    def is_valid_i_frame(self) -> bool:
        """Check if frame is valid I frame.
        
        Returns:
            True if valid I frame, False otherwise
        """
        return (self.type == FrameType.I and 
                0 <= self.ns <= 127 and 
                0 <= self.nr <= 127 and
                self.dest.callsign.strip() and 
                self.src.callsign.strip())

    def get_sequence_info(self) -> Dict[str, int]:
        """Get sequence number information.
        
        Returns:
            Dictionary with sequence numbers
        """
        return {
            'send_sequence': self.ns,
            'receive_sequence': self.nr,
            'ack_sequence': self.nr,  # NR is acknowledgment for received frames
            'control_byte': self.control
        }

    def calculate_frame_size(self) -> int:
        """Calculate frame size in bytes.
        
        Returns:
            Frame size in bytes
        """
        base_size = 14  # dest (7) + src (7)
        base_size += len(self.digipeaters) * 7  # digipeaters
        base_size += 1  # control
        if self.type in [FrameType.UI, FrameType.I]:
            base_size += 1  # PID
        base_size += len(self.info)  # info
        base_size += 2  # FCS
        return base_size + 2  # flags

    def get_address_info(self) -> Dict[str, str]:
        """Get address information.
        
        Returns:
            Dictionary with address details
        """
        return {
            'destination': f"{self.dest.callsign.strip()}-{self.dest.ssid}",
            'source': f"{self.src.callsign.strip()}-{self.src.ssid}",
            'digipeaters': [f"{d.callsign.strip()}-{d.ssid}" for d in self.digipeaters],
            'has_c_bit': self.src.c_bit,
            'has_r_bit': self.src.r_bit
        }

    def create_response(self, response_type: FrameType = FrameType.UI) -> 'AX25Frame':
        """Create response frame with swapped addresses.
        
        Args:
            response_type: Type of response frame
            
        Returns:
            New AX25Frame with swapped addresses
        """
        return AX25Frame(
            dest=self.src,
            src=self.dest,
            digipeaters=self.digipeaters.copy(),
            pid=self.pid
        )

    def add_digipeater(self, digipeater: AX25Address) -> None:
        """Add digipeater to frame.
        
        Args:
            digipeater: Digipeater address to add
        """
        if len(self.digipeaters) >= MAX_DIGIPEATERS:
            raise ValueError(f"Cannot add more than {MAX_DIGIPEATERS} digipeaters")
        if not isinstance(digipeater, AX25Address):
            raise TypeError("Digipeater must be AX25Address")
            
        self.digipeaters.append(digipeater)
        logger.debug(f"Added digipeater: {digipeater.callsign.strip()}")

    def set_poll(self, poll: bool) -> None:
        """Set poll/final bit.
        
        Args:
            poll: Poll/final flag
        """
        if not isinstance(poll, bool):
            raise TypeError("Poll must be boolean")
            
        self.poll = poll
        if self.type in [FrameType.I, FrameType.RR, FrameType.RNR, FrameType.REJ, FrameType.SREJ]:
            # Update control byte for sequence frames
            if self.poll:
                self.control |= 0x10
            else:
                self.control &= ~0x10
        elif self.type in [FrameType.SABM, FrameType.DISC, FrameType.UA, FrameType.DM, FrameType.XID, FrameType.TEST]:
            # Update control byte for unnumbered frames
            if self.poll:
                self.control |= 0x10
            else:
                self.control &= ~0x10

