# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25-22.interfaces.__init__.py

Public API for the transport interface layer.

This module exports the concrete interface classes and the abstract base
class used throughout the library.

Applications should import transport implementations from this namespace:
    from pyax25-22.interfaces import KISSInterface, AGWPEInterface

Higher-level libraries (PyXKISS, PyAGW3) will re-export these for convenience.
"""

from __future__ import annotations

# Abstract base class
from .transport import TransportInterface

# Concrete implementations
from .kiss import KISSInterface
from .agwpe import AGWPEInterface

# Version consistency with core
from ..core import __version__

# Explicit public API
__all__ = [
    "TransportInterface",
    "KISSInterface",
    "AGWPEInterface",
    "__version__",
]

# Optional friendly message when imported interactively
if __name__ == "__main__":
    print(f"PyAX25_22 interfaces package version {__version__}")
    print("Available transports:")
    print("  - KISSInterface: Serial/TCP KISS with multi-drop support")
    print("  - AGWPEInterface: Full TCP/IP AGWPE client")
    print("Import example:")
    print("    from pyax25-22.interfaces import KISSInterface")
