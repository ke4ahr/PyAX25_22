# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/fx25/decoder.py

FX.25 decoder: finds the connect tag, strips FEC, corrects errors, and
returns the original AX.25 frame bytes.

The decoder is a stream-oriented state machine.  Feed received bytes one
at a time (or in chunks) via feed().  When a complete, valid FX.25 block
has been received it calls the on_frame callback with the recovered AX.25
frame bytes and the number of errors corrected.

Usage::

    def got_frame(ax25_bytes, num_errors):
        print(f"Got {len(ax25_bytes)} byte frame, {num_errors} errors corrected")

    decoder = FX25Decoder(on_frame=got_frame)
    for byte in received_bytes:
        decoder.feed(bytes([byte]))
"""

import logging
from enum import Enum, auto
from typing import Callable, Optional, Tuple

from .constants import (
    FX25_TAGS, CTAG_SIZE, PREAMBLE_BYTE, tag_info, wire_to_tag_id,
)
from .rs import rs_decode

logger = logging.getLogger(__name__)


class _State(Enum):
    HUNT = auto()     # Searching for connect tag
    DATA = auto()     # Collecting RS block data


class FX25Decoder:
    """Decode an FX.25 byte stream, calling on_frame for each recovered frame.

    Args:
        on_frame: Callback(ax25_bytes: bytes, num_errors: int) called for each
            successfully decoded frame.  num_errors is the count of RS symbol
            errors corrected (0 means no errors were found).
        on_error: Optional callback(reason: str) called when a block is
            received but cannot be decoded (too many errors or bad frame).
    """

    def __init__(
        self,
        on_frame: Callable[[bytes, int], None],
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.on_frame = on_frame
        self.on_error = on_error

        self._state = _State.HUNT
        # Tag detection shift register: last CTAG_SIZE bytes seen
        self._tag_buf: bytearray = bytearray()
        # Block data buffer: collects data+check bytes after tag
        self._data_buf: bytearray = bytearray()
        self._expected_len: int = 0   # total_bytes expected in data block
        self._ncheck: int = 0
        self._ndata: int = 0
        self._tag_id: int = 0

    def reset(self) -> None:
        """Reset to hunt state (call if stream is interrupted)."""
        self._state = _State.HUNT
        self._tag_buf = bytearray()
        self._data_buf = bytearray()
        self._expected_len = 0

    def feed(self, data: bytes) -> None:
        """Feed received bytes into the decoder.

        Args:
            data: One or more bytes from the received stream.
        """
        for byte in data:
            self._process_byte(byte)

    def _process_byte(self, byte: int) -> None:
        if self._state == _State.HUNT:
            self._hunt(byte)
        else:
            self._collect(byte)

    def _hunt(self, byte: int) -> None:
        """Accumulate bytes looking for a valid connect tag."""
        self._tag_buf.append(byte)
        # Keep only the last CTAG_SIZE bytes
        if len(self._tag_buf) > CTAG_SIZE:
            del self._tag_buf[0]

        if len(self._tag_buf) == CTAG_SIZE:
            try:
                tag_id = wire_to_tag_id(bytes(self._tag_buf))
            except ValueError:
                return  # Not a valid tag yet

            # Valid tag found -- switch to data collection
            self._tag_id = tag_id
            ncheck, ndata, total = tag_info(tag_id)
            self._ncheck = ncheck
            self._ndata = ndata
            self._expected_len = total   # data + check bytes
            self._data_buf = bytearray()
            self._state = _State.DATA
            logger.debug(
                "FX25 tag 0x%02X found: ndata=%d ncheck=%d total=%d",
                tag_id, ndata, ncheck, total,
            )

    def _collect(self, byte: int) -> None:
        """Collect RS block bytes until we have a complete block."""
        self._data_buf.append(byte)
        if len(self._data_buf) < self._expected_len:
            return

        # Full block received -- decode
        self._decode_block()
        # Return to hunt regardless of success
        self._state = _State.HUNT
        self._tag_buf = bytearray()

    def _decode_block(self) -> None:
        """Attempt RS decode on the collected block."""
        codeword = bytes(self._data_buf)
        corrected, num_errors = rs_decode(codeword, self._ncheck)

        if corrected is None:
            reason = (
                f"RS decode failed for tag 0x{self._tag_id:02X} "
                f"(block {len(codeword)} bytes)"
            )
            logger.warning("FX25: %s", reason)
            if self.on_error:
                self.on_error(reason)
            return

        # Strip trailing pad bytes -- AX.25 frame is everything up to first
        # trailing zero sequence, but we cannot know where padding ends without
        # understanding the AX.25 frame structure.  Return the full data field;
        # caller is responsible for stripping pad if necessary.
        logger.debug(
            "FX25 decode: tag=0x%02X errors=%d data=%d",
            self._tag_id, num_errors, len(corrected),
        )
        self.on_frame(corrected, num_errors)
