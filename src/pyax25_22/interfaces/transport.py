# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.interfaces.transport.py

Abstract base class for all transport interfaces in PyAX25_22.

Defines the common interface for KISS, AGWPE, and future transports.
Provides validation utilities for transport-specific compliance.

All concrete transports must implement this ABC for consistency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any

from pyax25_22.core.framing import AX25Frame
from pyax25_22.core.exceptions import TransportError

logger = logging.getLogger(__name__)


class TransportInterface(ABC):
    """
    Abstract base class for all AX.25 transport interfaces.

    Defines the standard API for:
    - Connecting/disconnecting
    - Sending/receiving frames
    - Configuration and status
    - Error handling and callbacks

    Concrete implementations (KISS, AGWPE) must inherit from this.
    """

    def __init__(self):
        self.connected: bool = False
        self._callbacks: dict[str, Callable] = {}  # Event -> callback

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the transport medium.

        Raises:
            TransportError on connection failure
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Gracefully disconnect from the transport.

        Raises:
            TransportError on disconnection failure
        """
        pass

    @abstractmethod
    def send_frame(self, frame: AX25Frame, **kwargs: Any) -> None:
        """
        Send an AX.25 frame over the transport.

        Args:
            frame: The frame to transmit
            **kwargs: Transport-specific parameters (e.g., port for AGWPE)

        Raises:
            TransportError on send failure
        """
        pass

    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> AX25Frame:
        """
        Receive next available frame.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Received AX25Frame

        Raises:
            TransportError on timeout or error
        """
        pass

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for transport events.

        Args:
            event: Event name (e.g., 'frame_received', 'connected')
            callback: Function to call
        """
        self._callbacks[event] = callback
        logger.debug(f"Registered callback for event '{event}'")

    def _trigger_callback(self, event: str, *args: Any) -> None:
        """
        Internal: Trigger registered callback if exists.

        Args:
            event: Event name
            *args: Arguments to pass to callback
        """
        if event in self._callbacks:
            try:
                self._callbacks[event](*args)
                logger.debug(f"Triggered callback for '{event}'")
            except Exception as e:
                logger.error(f"Callback error for '{event}': {e}")


def validate_frame_for_transport(frame: AX25Frame, transport_type: str) -> None:
    """
    Validate frame compatibility with specific transport.

    Args:
        frame: Frame to validate
        transport_type: 'KISS' or 'AGWPE'

    Raises:
        TransportError if incompatible
    """
    if transport_type == 'KISS':
        # KISS max frame size typically 256-512 bytes
        if len(frame.encode()) > 512:
            raise TransportError(f"Frame too large for KISS: {len(frame.encode())} bytes")

    elif transport_type == 'AGWPE':
        # AGWPE has no strict limit but practical ~4KB
        if len(frame.encode()) > 4096:
            raise TransportError(f"Frame too large for AGWPE: {len(frame.encode())} bytes")

    logger.debug(f"Frame validated for {transport_type} transport")
