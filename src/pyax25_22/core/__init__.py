# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.__init__.py

Public API for the core AX.25 v2.2 implementation.

This module exports only the symbols that should be used by applications
and higher-level libraries (PyPACSAT, PyAGW3, PyXKISS).

All internal implementation details are kept private to maintain stability.
"""

from __future__ import annotations

# Core public classes and functions
from .framing import (
    AX25Frame,
    AX25Address,
    fcs_calc,
    verify_fcs,
)
from .statemachine import AX25StateMachine, AX25State
from .flow_control import AX25FlowControl
from .timers import AX25Timers
from .negotiation import build_xid_frame, parse_xid_frame, negotiate_config
from .validation import validate_frame_structure, full_validation

# Connection class is the main user-facing API
from .connected import AX25Connection

# Configuration
from .config import AX25Config, DEFAULT_CONFIG_MOD8, DEFAULT_CONFIG_MOD128

# Exception hierarchy
from .exceptions import (
    AX25Error,
    FrameError,
    InvalidAddressError,
    InvalidControlFieldError,
    FCSError,
    BitStuffingError,
    ConnectionError,
    ConnectionStateError,
    TimeoutError,
    TransportError,
    KISSError,
    AGWPEError,
    ConfigurationError,
    NegotiationError,
)

# Version
__version__ = "0.5.72"

# Public API list – explicitly exported symbols
__all__ = [
    # Framing
    "AX25Frame",
    "AX25Address",
    "fcs_calc",
    "verify_fcs",

    # Validation
    "validate_frame_structure",
    "full_validation",

    # Connection & State
    "AX25Connection",
    "AX25StateMachine",
    "AX25State",

    # Configuration
    "AX25Config",
    "DEFAULT_CONFIG_MOD8",
    "DEFAULT_CONFIG_MOD128",

    # Flow Control & Timers
    "AX25FlowControl",
    "AX25Timers",

    # XID Negotiation
    "build_xid_frame",
    "parse_xid_frame",
    "negotiate_config",

    # Exceptions
    "AX25Error",
    "FrameError",
    "InvalidAddressError",
    "InvalidControlFieldError",
    "FCSError",
    "BitStuffingError",
    "ConnectionError",
    "ConnectionStateError",
    "TimeoutError",
    "TransportError",
    "KISSError",
    "AGWPEError",
    "ConfigurationError",
    "NegotiationError",

    # Metadata
    "__version__",
]

# Optional: Friendly help message when imported interactively
if __name__ == "__main__":
    print(f"PyAX25_22 core module v{__version__}")
    print("This is a library module. Import from your application:")
    print(" from pyax25_22.core import AX25Frame, AX25Connection")
