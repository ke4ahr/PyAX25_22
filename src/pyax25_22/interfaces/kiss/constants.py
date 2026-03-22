# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
interfaces/kiss/constants.py

Shared constants for KISS, XKISS (multi-drop), SMACK, and XOR checksum modes.

References:
- Standard KISS: TAPR KISS specification
- XKISS / Multi-Drop: G8BPQ / Karl Medcalf WK5M documentation
- SMACK: Stuttgart Modified Amateur Radio CRC-KISS (SYMEK/TNC2)
"""

# ---------------------------------------------------------------------------
# Frame delimiters (shared across all modes)
# ---------------------------------------------------------------------------

FEND  = 0xC0   # Frame End -- marks beginning and end of a KISS frame
FESC  = 0xDB   # Frame Escape -- introduces a two-byte escape sequence
TFEND = 0xDC   # Transposed FEND -- follows FESC to represent a raw 0xC0 byte
TFESC = 0xDD   # Transposed FESC -- follows FESC to represent a raw 0xDB byte

# ---------------------------------------------------------------------------
# Standard KISS commands (low nibble of command byte)
# ---------------------------------------------------------------------------

CMD_DATA     = 0x00   # Data frame -- carry an AX.25 frame to/from the TNC
CMD_TXDELAY  = 0x01   # Set TX Delay (units: 10 ms)
CMD_PERSIST  = 0x02   # Set P-Persistence (0-255; probability = (P+1)/256)
CMD_SLOTTIME = 0x03   # Set Slot Time (units: 10 ms)
CMD_TXTAIL   = 0x04   # Set TX Tail (units: 10 ms)
CMD_FULLDUP  = 0x05   # Set Full Duplex (0=half, 1=full)
CMD_HARDWARE = 0x06   # Set Hardware (vendor-specific parameter)
CMD_EXIT     = 0xFF   # Exit KISS mode -- return TNC to normal operation

# ---------------------------------------------------------------------------
# XKISS / Multi-Drop / BPQKISS extensions (low nibble)
# ---------------------------------------------------------------------------

CMD_POLL     = 0x0E   # Poll command -- host requests queued data from TNC
CMD_DATA_ACK = 0x0C   # Extended data with frame acknowledgment (stubbed)

# ---------------------------------------------------------------------------
# SMACK -- Stuttgart Modified Amateur Radio CRC-KISS
# ---------------------------------------------------------------------------

SMACK_FLAG     = 0x80    # Bit 7 of command byte: set to indicate CRC frame
SMACK_POLY     = 0x8005  # CRC-16 polynomial (normal / non-reflected form)
SMACK_INIT     = 0x0000  # Initial CRC register value
SMACK_CRC_SIZE = 2       # Number of CRC bytes appended (LSB-first)

# ---------------------------------------------------------------------------
# Masks for XKISS command byte
# ---------------------------------------------------------------------------

PORT_MASK = 0xF0   # High nibble: TNC port / multi-drop address (0-15)
CMD_MASK  = 0x0F   # Low nibble: KISS command code

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BAUDRATE      = 9600   # Default serial baud rate
DEFAULT_POLL_INTERVAL = 0.1    # Default active-poll interval (seconds)
DEFAULT_MAX_QUEUE     = 100    # Default maximum frames per RX port queue

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

MIN_KISS_FRAME = 1   # Minimum valid KISS frame body: at least a command byte
