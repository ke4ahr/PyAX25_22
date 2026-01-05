# pyax25_22/__init__.py
"""
PyAX25_22 - Pure Python AX.25 Layer 2 Implementation

Provides:
- AX.25 frame encoding/decoding (v2.2)
- KISS and AGWPE transport interfaces
- Connected-mode state machine

License: LGPLv3.0
Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""

__version__ = "0.5.9"

# Core AX.25 functionality
from .core.framing import (
    AX25Frame,
    fcs_calc,
)
from .core.statemachine import (
    AX25StateMachine,
    AX25State
)

# Transport interfaces
from .interfaces.kiss import (
    KISSInterface,
    KISSCommand,
    SerialKISSInterface,
    TCPKISSInterface
)
from .interfaces.kiss_async import (
    AsyncKISSInterface,
    AsyncTCPKISSInterface
)
from .interfaces.agwpe import (
    AGWClient,
    AsyncAGWClient
)

# Exceptions
from .exceptions import (
    AX25Error,
    TransportError,
    KISSProtocolError,
    AGWProtocolError
)

# Utilities
from .utils import (
    configure_logging,
    get_version
)

__all__ = [
    # Core
    'AX25Frame',
    'fcs_calc',
    'AX25StateMachine',
    'AX25State',
    
    # Interfaces
    'KISSInterface',
    'KISSCommand',
    'SerialKISSInterface',
    'TCPKISSInterface',
    'AsyncKISSInterface',
    'AsyncTCPKISSInterface',
    'AGWClient',
    'AsyncAGWClient',
    
    # Exceptions
    'AX25Error',
    'TransportError',
    'KISSProtocolError',
    'AGWProtocolError',
    
    # Utilities
    'configure_logging',
    'get_version',
    
    # Metadata
    '__version__'
]

def configure_logging(level: str = "INFO") -> None:
    """
    Configure package-wide logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s %(name)s %(levelname)s: %(message)s'
    )

def get_version() -> str:
    """Return the package version."""
    return __version__
