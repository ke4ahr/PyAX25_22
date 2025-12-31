# pyax25_22/core/__init__.py
"""
PyAX25_22 Core Module - AX.25 Layer 2 Implementation

Contains:
- AX.25 frame encoding/decoding (v2.2 specification)
- Connection state machine for connected-mode operation
- Address handling and bit-level utilities

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

# Frame construction and parsing
from .framing import (
    AX25Frame,
    encode_address,
    decode_address,
    fcs_calc,
    bit_stuff,
    bit_destuff,
    FrameType,
    AX25Address
)

# Connected-mode state management
from .statemachine import (
    AX25StateMachine,
    AX25State,
    AX25Modulo,
    SREJManager
)

# Exceptions
from .exceptions import (
    AX25Error,
    FrameDecodeError,
    StateMachineError
)

# Public API
__all__ = [
    # Framing
    'AX25Frame',
    'encode_address',
    'decode_address',
    'fcs_calc',
    'bit_stuff',
    'bit_destuff',
    'FrameType',
    'AX25Address',
    
    # State machine
    'AX25StateMachine',
    'AX25State',
    'AX25Modulo',
    'SREJManager',
    
    # Exceptions
    'AX25Error',
    'FrameDecodeError',
    'StateMachineError'
]

def _verify_versions() -> None:
    """Internal compatibility check."""
    import sys
    if sys.version_info < (3, 8):
        raise RuntimeError("PyAX25_22 requires Python 3.8+")

_verify_versions()
