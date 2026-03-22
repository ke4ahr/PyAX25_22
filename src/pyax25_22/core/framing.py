# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.framing -- AX.25 frame building and reading.

This file handles the nuts and bolts of turning raw bytes into AX.25
frames and turning AX.25 frames back into bytes.

Think of it like reading and writing a very specific kind of envelope:
  - The envelope has a "To" address and a "From" address.
  - It can have a list of post offices (digipeaters) that help pass it along.
  - Inside is the message (information field).
  - A checksum at the end lets the receiver know if the message got scrambled.

Key pieces in this file:
  - fcs_calc() -- compute the frame checksum (like a ZIP code check)
  - verify_fcs() -- confirm a received checksum is right
  - AX25Address -- one callsign + SSID, can be encoded or decoded
  - AX25Frame -- a complete AX.25 frame with addresses, control, and data

Compliant with AX.25 v2.2 specification (July 1998).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

from .config import AX25Config, DEFAULT_CONFIG_MOD8
from .exceptions import (
    BitStuffingError,
    InvalidAddressError,
    FCSError,
    FrameError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AX.25 constants
# ---------------------------------------------------------------------------

#: HDLC frame delimiter byte.
FLAG: int = 0x7E

#: Starting value for the CRC-16/CCITT-FALSE algorithm.
FCS_INIT: int = 0xFFFF

#: CRC-16 polynomial in reflected (LSB-first) form.
FCS_POLY: int = 0x8408


# ---------------------------------------------------------------------------
# FCS helpers
# ---------------------------------------------------------------------------

def fcs_calc(data: bytes) -> int:
    """Calculate the AX.25 frame check sequence (CRC).

    AX.25 uses a 16-bit checksum called the FCS (Frame Check Sequence).
    This is computed with the CRC-16/CCITT-FALSE algorithm. Think of it
    as a fingerprint for the data -- if even one bit changes in transit,
    the fingerprint will not match.

    Args:
        data: The bytes to checksum. This is everything in the frame
            except the two FCS bytes themselves and the flag bytes.

    Returns:
        A 16-bit integer (0 to 65535) that is the FCS for the given data.

    Raises:
        TypeError: If ``data`` is not bytes or bytearray.

    Example::

        fcs = fcs_calc(b"\\x00\\x01\\x02")
        # Returns a 16-bit int, e.g. 0xABCD
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(
            f"data must be bytes or bytearray, got {type(data).__name__}"
        )

    logger.debug("fcs_calc: computing FCS over %d bytes", len(data))

    fcs = FCS_INIT
    for byte in data:
        fcs ^= byte
        for _ in range(8):
            if fcs & 1:
                fcs = (fcs >> 1) ^ FCS_POLY
            else:
                fcs >>= 1

    result = ~fcs & 0xFFFF
    logger.debug("fcs_calc: result FCS=0x%04X", result)
    return result


def verify_fcs(data: bytes, received_fcs: int) -> bool:
    """Check whether a received FCS matches the data.

    After receiving a frame, call this function to see if the checksum
    in the frame matches the data. If they do not match, the frame was
    probably damaged in transit and should be thrown away.

    Args:
        data: The frame bytes to check (everything except the FCS bytes
            and flag bytes).
        received_fcs: The 16-bit FCS value taken from the end of the frame.

    Returns:
        True if the FCS matches (frame is good), False if it does not
        (frame is damaged).

    Raises:
        TypeError: If ``data`` is not bytes or bytearray.

    Example::

        ok = verify_fcs(frame_body, 0xABCD)
        if not ok:
            raise FCSError("Frame damaged in transit")
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(
            f"data must be bytes or bytearray, got {type(data).__name__}"
        )

    computed = fcs_calc(data)
    match = (computed == received_fcs)

    if match:
        logger.debug("verify_fcs: FCS OK (0x%04X)", received_fcs)
    else:
        logger.warning(
            "verify_fcs: FCS mismatch -- computed=0x%04X received=0x%04X",
            computed, received_fcs,
        )

    return match


# ---------------------------------------------------------------------------
# AX.25 address
# ---------------------------------------------------------------------------

@dataclass
class AX25Address:
    """One AX.25 station address: a callsign and an SSID.

    An AX.25 address is like a postal address for a radio station.
    Every station has a callsign (like KE4AHR) and an optional number
    called the SSID (0 to 15) that lets one station act like several.

    For example, KE4AHR-0 and KE4AHR-1 are two different addresses
    for the same operator.

    When encoded for the air, each callsign character is left-shifted
    by one bit, and the callsign is padded to exactly 6 characters with
    spaces. The SSID and flag bits go in a 7th byte.

    Attributes:
        callsign: The station callsign, 1 to 6 characters, letters and
            digits only. Will be converted to upper case.
        ssid: The Secondary Station Identifier, 0 to 15. Default is 0.
        c_bit: The Command/Response bit. True if this address is the
            source in a command frame or the destination in a response.
        h_bit: The Has-Been-Repeated bit. Set by a digipeater to show
            it has already forwarded this frame. Also called the H bit.

    Raises:
        InvalidAddressError: If the callsign is too long, too short, or
            has bad characters, or if the SSID is out of range.

    Example::

        addr = AX25Address("KE4AHR", ssid=1)
        raw = addr.encode(last=True)  # 7 bytes for the wire
    """

    callsign: str
    ssid: int = 0
    c_bit: bool = False
    h_bit: bool = False

    # Encoded callsign bytes (built in __post_init__, not part of constructor)
    _call_bytes: bytes = field(default=b"", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the callsign and SSID and prepare the encoded bytes.

        Called automatically after the dataclass __init__. Checks that
        the callsign is the right length and contains only letters and
        digits, and that the SSID is in range 0-15.

        Raises:
            InvalidAddressError: If the callsign is bad or the SSID is
                out of range.
        """
        logger.debug(
            "AX25Address.__post_init__: callsign=%r ssid=%d",
            self.callsign, self.ssid,
        )

        if not (0 <= self.ssid <= 15):
            raise InvalidAddressError(
                f"SSID {self.ssid} out of range (must be 0-15)"
            )

        # Strip hyphens (some callers pass "KE4AHR-1") and uppercase
        callsign_clean = self.callsign.upper().strip()
        if "-" in callsign_clean:
            callsign_clean = callsign_clean.split("-")[0]

        if not (1 <= len(callsign_clean) <= 6):
            raise InvalidAddressError(
                f"Callsign '{self.callsign}' must be 1 to 6 characters "
                f"(got {len(callsign_clean)})"
            )

        for ch in callsign_clean:
            if not (ch.isalpha() or ch.isdigit()):
                raise InvalidAddressError(
                    f"Callsign '{self.callsign}' has invalid character '{ch}' "
                    f"(only letters and digits are allowed)"
                )

        # Each character is stored as (ASCII << 1) in the 7-byte address field
        # The callsign is padded to exactly 6 characters with spaces
        object.__setattr__(
            self,
            "_call_bytes",
            bytes((ord(c) << 1) for c in callsign_clean.ljust(6, " ")),
        )

        logger.debug(
            "AX25Address: validated %s-%d, _call_bytes=%s",
            callsign_clean, self.ssid, self._call_bytes.hex(),
        )

    def encode(self, last: bool = False) -> bytes:
        """Encode this address into 7 bytes for the AX.25 address field.

        Produces the 7-byte wire encoding: 6 shifted callsign bytes
        followed by one SSID byte that also carries the C/H bits and
        the end-of-address-field marker.

        Args:
            last: Set to True if this is the last address in the address
                field (the final digipeater, or the source if there are
                no digipeaters). This sets bit 0 of the SSID byte.

        Returns:
            Exactly 7 bytes ready to put into an AX.25 frame.

        Example::

            dest = AX25Address("W1AW")
            src  = AX25Address("KE4AHR", ssid=1)
            raw  = dest.encode(last=False) + src.encode(last=True)
        """
        # Bits 7-6 of SSID byte are always 1 (per AX.25 spec)
        ssid_byte = 0x60
        ssid_byte |= (self.ssid << 1) & 0x1E

        # C bit and H bit share bit 7 in most implementations.
        # The spec says C is bit 7 of the SSID byte for destination/source
        # and H is bit 7 of the SSID byte for digipeaters.
        if self.c_bit or self.h_bit:
            ssid_byte |= 0x80

        if last:
            ssid_byte |= 0x01

        encoded = self._call_bytes + bytes([ssid_byte])
        logger.debug(
            "AX25Address.encode: %s-%d -> %s (last=%s)",
            self.callsign, self.ssid, encoded.hex(), last,
        )
        return encoded

    @classmethod
    def decode(cls, data: bytes) -> Tuple["AX25Address", bool]:
        """Decode 7 bytes from an AX.25 address field into an AX25Address.

        Reads one 7-byte address entry from the address field of an
        AX.25 frame and returns an AX25Address plus a flag that says
        whether this was the last address in the field.

        Args:
            data: At least 7 bytes from the address field. Only the
                first 7 bytes are used.

        Returns:
            A tuple of:
            - AX25Address: The decoded address object.
            - bool: True if this was the last address in the field
              (bit 0 of the SSID byte is set).

        Raises:
            InvalidAddressError: If ``data`` is shorter than 7 bytes or
                the decoded callsign characters are out of range.

        Example::

            addr, is_last = AX25Address.decode(raw_bytes)
            print(f"{addr.callsign}-{addr.ssid}, last={is_last}")
        """
        if len(data) < 7:
            raise InvalidAddressError(
                f"Address field must be at least 7 bytes, got {len(data)}"
            )

        call_bytes = data[:6]
        ssid_byte = data[6]

        # Each callsign byte is the ASCII code shifted left by one bit.
        # Spaces (0x40 after shifting, or 0x20 unshifted) are padding.
        callsign_chars = []
        for b in call_bytes:
            char_code = b >> 1
            if char_code == 0x20:   # space = end of callsign
                break
            ch = chr(char_code)
            if not (ch.isalpha() or ch.isdigit()):
                raise InvalidAddressError(
                    f"Invalid character code 0x{char_code:02X} in address field"
                )
            callsign_chars.append(ch)

        callsign = "".join(callsign_chars).rstrip()

        ssid = (ssid_byte >> 1) & 0x0F
        c_bit = bool(ssid_byte & 0x80)
        h_bit = bool(ssid_byte & 0x80)
        is_last = bool(ssid_byte & 0x01)

        logger.debug(
            "AX25Address.decode: %s-%d c_bit=%s h_bit=%s is_last=%s",
            callsign, ssid, c_bit, h_bit, is_last,
        )

        addr = cls(
            callsign=callsign,
            ssid=ssid,
            c_bit=c_bit,
            h_bit=h_bit,
        )
        return addr, is_last


# ---------------------------------------------------------------------------
# AX.25 frame
# ---------------------------------------------------------------------------

@dataclass
class AX25Frame:
    """One complete AX.25 frame: addresses, control field, and data.

    An AX25Frame holds everything in one AX.25 packet:
      - A destination address (who to send to)
      - A source address (who is sending)
      - An optional list of digipeaters (relay stations to pass through)
      - A control byte (what kind of frame is this: data, ack, connect?)
      - An optional PID byte (what protocol is in the data?)
      - The data payload (information field)

    To send a frame, call encode() to get the raw bytes.
    To receive a frame, call decode() with the raw bytes.

    Attributes:
        destination: The station this frame is addressed to.
        source: The station sending this frame.
        digipeaters: A list of up to 8 relay stations. Usually empty.
        control: The control field value. Determines the frame type.
        pid: The Protocol Identifier. Used in I-frames and UI-frames.
            None if not present.
        info: The data payload (information field). May be empty.
        config: The AX.25 configuration. Controls modulo and encoding.

    Raises:
        FrameError: If a frame cannot be encoded or decoded.
        FCSError: If the checksum in a received frame is wrong.
        BitStuffingError: If the bit stuffing in a received frame is bad.

    Example::

        dest = AX25Address("W1AW")
        src  = AX25Address("KE4AHR", ssid=1)
        frame = AX25Frame(
            destination=dest,
            source=src,
            control=0x03,   # UI frame
            pid=0xF0,
            info=b"Hello",
        )
        raw = frame.encode()
    """

    destination: AX25Address
    source: AX25Address
    digipeaters: List[AX25Address] = field(default_factory=list)
    control: int = 0
    pid: Optional[int] = None
    info: bytes = b""
    config: AX25Config = field(default_factory=lambda: DEFAULT_CONFIG_MOD8)

    def encode(self) -> bytes:
        """Encode this frame into raw bytes ready to send over the air.

        Builds the complete AX.25 frame: address field, control field,
        optional PID, information field, FCS, bit stuffing, and flags.

        The frame format is::

            FLAG | address field | control | [PID] | [info] | FCS | FLAG

        Returns:
            The complete frame as bytes, including the leading and
            trailing FLAG bytes (0x7E).

        Raises:
            FrameError: If the address field or control field cannot be
                built (for example, if a callsign is invalid).

        Example::

            raw = frame.encode()
            serial_port.write(raw)
        """
        logger.debug(
            "AX25Frame.encode: %s -> %s control=0x%02X pid=%s info=%d bytes",
            self.source.callsign, self.destination.callsign,
            self.control, self.pid, len(self.info),
        )

        addr_field = self._build_address_field()
        ctrl_bytes = self._build_control_field()

        payload = ctrl_bytes
        if self.pid is not None:
            payload += bytes([self.pid & 0xFF])
        payload += self.info

        fcs = fcs_calc(addr_field + payload)
        frame_body = addr_field + payload + struct.pack("<H", fcs)

        # Note: bit stuffing is intentionally NOT applied here. For KISS-based
        # interfaces the TNC hardware performs bit stuffing. encode() returns
        # the raw AX.25 frame bytes delimited by FLAG bytes so that the output
        # is self-contained and can be decoded by decode(). If you need
        # HDLC-level bit stuffing for a direct serial connection (no TNC), call
        # _bit_stuff() on frame_body yourself before wrapping.
        result = bytes([FLAG]) + frame_body + bytes([FLAG])

        logger.info(
            "AX25Frame.encode: encoded frame %d bytes (body=%d)",
            len(result), len(frame_body),
        )
        return result

    def _build_address_field(self) -> bytes:
        """Build the AX.25 address field bytes.

        The address field has the destination first, then the source,
        then any digipeaters. The last address has bit 0 of its SSID
        byte set to 1 to mark the end of the address field.

        Returns:
            The encoded address field as bytes.
        """
        has_digis = bool(self.digipeaters)

        addr_field = self.destination.encode(last=False)
        addr_field += self.source.encode(last=not has_digis)

        for i, digi in enumerate(self.digipeaters):
            last = (i == len(self.digipeaters) - 1)
            addr_field += digi.encode(last=last)

        logger.debug(
            "_build_address_field: %d addresses, %d bytes total",
            2 + len(self.digipeaters), len(addr_field),
        )
        return addr_field

    def _build_control_field(self) -> bytes:
        """Build the control field bytes (1 or 2 bytes).

        For modulo-8 frames the control field is 1 byte.
        For modulo-128 I-frames the control field is 2 bytes (only for
        I-frames, where bit 0 of the first byte is 0).

        Returns:
            The control field as 1 or 2 bytes.
        """
        if self.config.modulo == 128 and (self.control & 0x01 == 0):
            # Modulo-128 I-frame: 2-byte control field
            ctrl = struct.pack("<H", self.control & 0xFFFF)
            logger.debug("_build_control_field: modulo-128 I-frame, 2 bytes")
        else:
            # All other frames: 1-byte control field
            ctrl = bytes([self.control & 0xFF])
            logger.debug("_build_control_field: 1-byte control=0x%02X", self.control & 0xFF)
        return ctrl

    @staticmethod
    def _bit_stuff(data: bytes) -> bytes:
        """Apply HDLC bit stuffing to raw frame bytes.

        AX.25 uses HDLC bit stuffing to keep the FLAG byte (0x7E =
        01111110) unique. After every five 1-bits in a row, a 0-bit is
        inserted. The receiver removes these extra 0-bits.

        This operates on bits from LSB to MSB within each byte, as
        required by HDLC/AX.25.

        Args:
            data: The raw frame bytes to stuff (between the two flags,
                not including the flags themselves).

        Returns:
            The bit-stuffed bytes. May be slightly longer than the input.

        Example::

            stuffed = AX25Frame._bit_stuff(frame_body)
        """
        logger.debug("_bit_stuff: input %d bytes", len(data))

        result = bytearray()
        ones_count = 0
        out_byte = 0
        out_pos = 0

        def emit_bit(bit: int) -> None:
            """Write one bit to the output buffer."""
            nonlocal out_byte, out_pos
            out_byte |= (bit << out_pos)
            out_pos += 1
            if out_pos == 8:
                result.append(out_byte)
                out_byte = 0
                out_pos = 0

        for byte in data:
            for i in range(8):
                bit = (byte >> i) & 1
                emit_bit(bit)

                if bit == 1:
                    ones_count += 1
                    if ones_count == 5:
                        # Insert a stuffing 0-bit
                        emit_bit(0)
                        ones_count = 0
                else:
                    ones_count = 0

        # Flush any remaining bits
        if out_pos > 0:
            result.append(out_byte)

        logger.debug("_bit_stuff: output %d bytes", len(result))
        return bytes(result)

    @classmethod
    def _bit_destuff(cls, data: bytes) -> bytes:
        """Remove HDLC bit stuffing from received bytes.

        The reverse of _bit_stuff. After every five 1-bits in a row,
        the next bit should be a stuffed 0-bit that gets removed.

        If six or more 1-bits appear in a row, that is a protocol error
        (it would be an abort sequence or flag, not valid frame data).

        Args:
            data: The stuffed bytes to destuff (between the flag bytes,
                not including the flags themselves).

        Returns:
            The raw frame bytes with stuffing removed.

        Raises:
            BitStuffingError: If six or more consecutive 1-bits are
                found, which indicates a corrupted or aborted frame.

        Example::

            raw = AX25Frame._bit_destuff(received_bytes)
        """
        logger.debug("_bit_destuff: input %d bytes", len(data))

        result = bytearray()
        ones_count = 0
        out_byte = 0
        out_pos = 0

        def emit_bit(bit: int) -> None:
            """Write one bit to the output buffer."""
            nonlocal out_byte, out_pos
            out_byte |= (bit << out_pos)
            out_pos += 1
            if out_pos == 8:
                result.append(out_byte)
                out_byte = 0
                out_pos = 0

        for byte in data:
            for i in range(8):
                bit = (byte >> i) & 1

                if bit == 1:
                    ones_count += 1
                    if ones_count == 6:
                        raise BitStuffingError(
                            "Six consecutive 1-bits found -- frame is corrupted "
                            "or an abort sequence was received"
                        )
                    emit_bit(1)
                else:
                    if ones_count == 5:
                        # This is a stuffed bit -- discard it
                        ones_count = 0
                    else:
                        ones_count = 0
                        emit_bit(0)

        # Flush any remaining bits (partial last byte is normal)
        if out_pos > 0:
            result.append(out_byte)

        logger.debug("_bit_destuff: output %d bytes", len(result))
        return bytes(result)

    @classmethod
    def decode(
        cls,
        raw: bytes,
        config: AX25Config = DEFAULT_CONFIG_MOD8,
    ) -> "AX25Frame":
        """Decode raw bytes into an AX25Frame object.

        Takes the bytes received over the air (or from a TNC) and
        parses them into an AX25Frame. The bytes must include the
        leading and trailing FLAG bytes (0x7E).

        The decoding steps are:
        1. Check the flag bytes.
        2. Remove bit stuffing.
        3. Parse the address field (destination, source, digipeaters).
        4. Parse the control field.
        5. Parse the PID and information field.
        6. Verify the FCS checksum.

        Args:
            raw: The raw frame bytes including leading and trailing
                FLAG bytes.
            config: The AX.25 configuration to use for parsing (needed
                to know if this is modulo-8 or modulo-128).

        Returns:
            A new AX25Frame object with all fields filled in.

        Raises:
            FrameError: If the frame is missing flags, too short, or
                the address field is malformed.
            FCSError: If the checksum does not match the data.
            BitStuffingError: If the bit stuffing is invalid.
            InvalidAddressError: If a callsign in the frame is invalid.

        Example::

            frame = AX25Frame.decode(raw_bytes)
            print(f"From: {frame.source.callsign}")
            print(f"To:   {frame.destination.callsign}")
            print(f"Data: {frame.info}")
        """
        logger.debug("AX25Frame.decode: raw frame %d bytes", len(raw))

        if len(raw) < 2:
            raise FrameError(
                f"Frame too short to contain flags ({len(raw)} bytes)",
                frame_data=bytes(raw),
            )

        if raw[0] != FLAG:
            raise FrameError(
                f"Frame does not start with FLAG byte (got 0x{raw[0]:02X})",
                frame_data=bytes(raw),
            )
        if raw[-1] != FLAG:
            raise FrameError(
                f"Frame does not end with FLAG byte (got 0x{raw[-1]:02X})",
                frame_data=bytes(raw),
            )

        # Remove the flag bytes. Bit destuffing is intentionally NOT applied
        # here -- see note in encode(). The TNC already handles bit stuffing
        # for KISS interfaces.
        destuffed = raw[1:-1]
        logger.debug("AX25Frame.decode: frame body %d bytes", len(destuffed))

        # Minimum frame: destination(7) + source(7) + control(1) + FCS(2) = 17 bytes
        if len(destuffed) < 17:
            raise FrameError(
                f"Frame too short after bit destuffing ({len(destuffed)} bytes, need >= 17)",
                frame_data=bytes(raw),
            )

        # --- Parse address field ---
        offset = 0

        dest, _ = AX25Address.decode(destuffed[offset:offset + 7])
        offset += 7

        src, is_last = AX25Address.decode(destuffed[offset:offset + 7])
        offset += 7

        digipeaters: List[AX25Address] = []
        while not is_last:
            if offset + 7 > len(destuffed):
                raise FrameError(
                    "Address field extends beyond end of frame",
                    frame_data=bytes(raw),
                )
            digi, is_last = AX25Address.decode(destuffed[offset:offset + 7])
            digipeaters.append(digi)
            offset += 7

        logger.debug(
            "AX25Frame.decode: addresses: dest=%s src=%s digis=%d",
            dest.callsign, src.callsign, len(digipeaters),
        )

        # --- Parse control field ---
        if offset >= len(destuffed):
            raise FrameError(
                "Frame ends before control field",
                frame_data=bytes(raw),
            )

        control = destuffed[offset]
        offset += 1

        # Modulo-128 I-frames have a 2-byte control field (bit 0 of first byte is 0)
        if config.modulo == 128 and (control & 0x01 == 0):
            if offset >= len(destuffed):
                raise FrameError(
                    "Frame ends before second control byte (modulo-128 I-frame)",
                    frame_data=bytes(raw),
                )
            control |= destuffed[offset] << 8
            offset += 1

        logger.debug("AX25Frame.decode: control=0x%04X", control)

        # --- Parse PID and information field ---
        # PID is present in I-frames (bit 0 = 0) and UI-frames (control = 0x03)
        pid: Optional[int] = None
        is_i_frame = (control & 0x01 == 0)
        is_ui_frame = (control & 0xFF == 0x03)

        if is_i_frame or is_ui_frame:
            if offset >= len(destuffed) - 2:
                raise FrameError(
                    "Frame ends before PID byte",
                    frame_data=bytes(raw),
                )
            pid = destuffed[offset]
            offset += 1
            logger.debug("AX25Frame.decode: pid=0x%02X", pid)

        # Information field is everything between PID and FCS
        # The last 2 bytes are always the FCS
        info = bytes(destuffed[offset:-2])
        received_fcs = struct.unpack("<H", destuffed[-2:])[0]

        logger.debug(
            "AX25Frame.decode: info=%d bytes FCS=0x%04X",
            len(info), received_fcs,
        )

        # --- Verify FCS ---
        frame_content = bytes(destuffed[:-2])
        if not verify_fcs(frame_content, received_fcs):
            raise FCSError(
                f"FCS mismatch -- frame is corrupted "
                f"(received=0x{received_fcs:04X})",
                frame_data=bytes(raw),
            )

        result = cls(
            destination=dest,
            source=src,
            digipeaters=digipeaters,
            control=control,
            pid=pid,
            info=info,
            config=config,
        )

        logger.info(
            "AX25Frame.decode: OK %s -> %s control=0x%04X pid=%s info=%d bytes",
            src.callsign, dest.callsign, control, pid, len(info),
        )
        return result
