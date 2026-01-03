# pyax25_22/utils/__init__.py
"""
PyAX25_22 Utilities Module

Provides common utilities for thread safety, async operations, and other shared functionality.

License: LGPLv3.0
Copyright (C) 2025-2026 Kris Kirby, KE4AHR
"""

from .threadsafe import SharedState, AtomicCounter
from .async_thread import run_in_thread
from typing import List

__all__: List[str] = [
    'SharedState',
    'AtomicCounter',
    'run_in_thread'
]

def __getattr__(name: str):
    """Lazy import helper for better startup performance"""
    if name == 'SharedState':
        from .threadsafe import SharedState
        return SharedState
    if name == 'AtomicCounter':
        from .threadsafe import AtomicCounter
        return AtomicCounter
    if name == 'run_in_thread':
        from .async_thread import run_in_thread
        return run_in_thread
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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

# Add utilities to __all__ if they should be exposed
__all__.extend(['configure_logging'])
