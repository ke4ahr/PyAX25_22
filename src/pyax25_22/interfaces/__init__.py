# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.interfaces.__init__.py

Public API for the transport interfaces.

Hierarchy:
    transport.py        -- TransportInterface ABC
    kiss/               -- KISS family (KISSBase, KISSSerial, KISSTCP,
                           XKISS, XKISSTCP, SMACK, SMACKTCP)
    agw/                -- AGW Packet Engine client (AGWPEClient)
    fx25/               -- FX.25 Reed-Solomon FEC (Phase 4, not yet impl.)

Backward-compatible re-exports via legacy shims:
    kiss.py   -> KISSInterface (was the original monolithic KISS class)
    agwpe.py  -> AGWPEInterface (was the original AGWPE class)

Preferred imports for new code:
    from pyax25_22.interfaces.kiss import KISSSerial, KISSTCP, XKISS, SMACK
    from pyax25_22.interfaces.agw import AGWPEClient
"""

from __future__ import annotations

# Abstract base class
from .transport import TransportInterface

# New canonical imports
from .kiss import KISSBase, KISSSerial, KISSTCP, XKISS, XKISSTCP, SMACK, SMACKTCP
from .agw import AGWPEClient, AGWPEFrame

# Legacy shim names (backward compat)
from .kiss import KISSSerial as KISSInterface
from .agw import AGWPEClient as AGWPEInterface

# Version consistency with core
from ..core import __version__

# Explicit public API
__all__ = [
    # Abstract
    "TransportInterface",
    # KISS family
    "KISSBase",
    "KISSSerial",
    "KISSTCP",
    "XKISS",
    "XKISSTCP",
    "SMACK",
    "SMACKTCP",
    # AGW
    "AGWPEClient",
    "AGWPEFrame",
    # Legacy shims
    "KISSInterface",
    "AGWPEInterface",
    "__version__",
]
