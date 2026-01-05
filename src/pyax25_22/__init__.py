# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.__init__.py

Top-level package API for PyAX25_22.

Provides convenient access to the most commonly used components:
- Core framing and connection classes
- Transport interfaces (KISS, AGWPE)
- Version information

Users should import from this namespace:
    import pyax25_22
    from pyax25_22 import AX25Frame, KISSInterface
"""

from __future__ import annotations

# Core components
from .core import (
    AX25Frame,
    AX25Address,
    AX25Connection,
    AX25Config,
    DEFAULT_CONFIG_MOD8,
    DEFAULT_CONFIG_MOD128,
    validate_frame_structure,
    full_validation,
)

# Interfaces
from .interfaces import (
    KISSInterface,
    AGWPEInterface,
    TransportInterface,
)

# Version
from .core import __version__

# Explicit public API
__all__ = [
    # Core
    "AX25Frame",
    "AX25Address",
    "AX25Connection",
    "AX25Config",
    "DEFAULT_CONFIG_MOD8",
    "DEFAULT_CONFIG_MOD128",
    "validate_frame_structure",
    "full_validation",

    # Interfaces
    "KISSInterface",
    "AGWPEInterface",
    "TransportInterface",

    # Metadata
    "__version__",
]

# Friendly message when package executed directly
if __name__ == "__main__":
    print(f"PyAX25_22 package v{__version__}")
    print("This is a library package. Import symbols in your application:")
    print("    from pyax25_22 import AX25Frame, KISSInterface")
    print("See README.md and docs/ for usage examples.")