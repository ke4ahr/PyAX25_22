# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
interfaces/agw/constants.py

Constants for the AGW Packet Engine (AGWPE) TCP/IP API.

The AGWPE protocol uses a 36-byte header followed by a variable-length data
field.  The ``data_kind`` byte (offset 0 in the header) identifies the frame
type.

References:
- SV2AGW AGWPE API documentation
- Hamradio AGWPE protocol description (various authors)
"""

AGWPE_DEFAULT_PORT = 8000   # Default TCP port for AGWPE server

AGWPE_HEADER_SIZE = 36      # Fixed header length in bytes

# ---------------------------------------------------------------------------
# Data kind codes (ASCII character, stored in header byte 0)
# Each code is the byte value of the ASCII character.
# ---------------------------------------------------------------------------

# Informational / registration
DK_VERSION      = ord('R')   # 'R' -- Request AGWPE version info (host -> app)
DK_REGISTER     = ord('X')   # 'X' -- Register callsign with AGWPE
DK_UNREGISTER   = ord('x')   # 'x' -- Unregister callsign
DK_PORT_INFO    = ord('G')   # 'G' -- Request port information
DK_PORT_CAPS    = ord('g')   # 'g' -- Port capabilities response
DK_EXTENDED_VER = ord('v')   # 'v' -- Extended version info
DK_MEMORY_USAGE = ord('m')   # 'm' -- Memory usage query/response

# Monitoring
DK_ENABLE_MON   = ord('M')   # 'M' -- Enable monitoring on a port
DK_RAW_MON      = ord('K')   # 'K' -- Raw monitored frame (receive)
DK_RAW_SEND     = ord('k')   # 'k' -- Send raw frame (transmit)

# Unproto (UI frames)
DK_UNPROTO      = ord('U')   # 'U' -- Unproto (UI) frame header (receive)
DK_UNPROTO_VIA  = ord('V')   # 'V' -- Unproto with via (digipeat) path
DK_DATA_UNPROTO = ord('D')   # 'D' -- UI data with PID (receive)

# Connected mode
DK_CONNECT      = ord('C')   # 'C' -- Connect request (outgoing)
DK_CON_VIA      = ord('v')   # Overloaded -- also used for connect-via
DK_CONNECT_INC  = ord('c')   # 'c' -- Incoming connection notification
DK_DISC         = ord('d')   # 'd' -- Disconnect / disconnect notification
DK_CONN_DATA    = ord('D')   # 'D' -- Connected data (send/receive) [same as UI]
DK_CONN_INFO    = ord('I')   # 'I' -- Connected frame info

# Flow control
DK_OUTSTANDING  = ord('Y')   # 'Y' -- Outstanding frames count
DK_OUTSTANDING_R = ord('y')  # 'y' -- Outstanding frames query response

# Heard stations
DK_HEARD        = ord('H')   # 'H' -- Heard stations list

# Login / parameters
DK_LOGIN        = ord('T')   # 'T' -- Login (username/password)
DK_PARAMETER    = ord('P')   # 'P' -- Set TNC parameter

# Frame kind byte values as single-byte bytes objects (for comparisons)
KIND_VERSION      = b'R'
KIND_REGISTER     = b'X'
KIND_UNREGISTER   = b'x'
KIND_PORT_INFO    = b'G'
KIND_PORT_CAPS    = b'g'
KIND_EXTENDED_VER = b'v'
KIND_MEMORY_USAGE = b'm'
KIND_ENABLE_MON   = b'M'
KIND_RAW_MON      = b'K'
KIND_RAW_SEND     = b'k'
KIND_UNPROTO      = b'U'
KIND_UNPROTO_VIA  = b'V'
KIND_UNPROTO_DATA = b'D'
KIND_CONNECT      = b'C'
KIND_CONNECT_INC  = b'c'
KIND_DISC         = b'd'
KIND_CONN_DATA    = b'D'
KIND_OUTSTANDING  = b'Y'
KIND_OUTSTANDING_R = b'y'
KIND_HEARD        = b'H'
KIND_LOGIN        = b'T'
KIND_PARAMETER    = b'P'

# AGWPE header field offsets
HDR_DATA_KIND   = 0    # 1 byte: frame type
HDR_RESERVED1   = 1    # 3 bytes: reserved (zero)
HDR_PORT        = 4    # 4 bytes: port number (little-endian uint32)
HDR_RESERVED2   = 8    # 2 bytes: reserved
HDR_CALL_FROM   = 8    # 10 bytes: source callsign (space-padded ASCII)
HDR_CALL_TO     = 18   # 10 bytes: destination callsign (space-padded ASCII)
HDR_DATA_LEN    = 28   # 4 bytes: payload length (little-endian uint32)
HDR_RESERVED3   = 32   # 4 bytes: reserved

# Callsign field width in header
CALLSIGN_WIDTH = 10
