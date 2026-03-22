# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.interfaces.transport -- Base class for all transport interfaces.

A transport interface is the part of the library that actually talks to
the radio hardware or network service. Think of it like a postal worker:
the AX.25 protocol knows what to write on the envelope, but the transport
worker knows how to physically deliver it.

This file defines the TransportInterface abstract base class (ABC). All
concrete transports (KISS over serial, AGWPE over TCP, etc.) must
inherit from it and implement its abstract methods.

Also provides a helper function for checking whether a frame is small
enough to fit through a given transport type.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from pyax25_22.core.framing import AX25Frame
from pyax25_22.core.exceptions import TransportError

logger = logging.getLogger(__name__)


class TransportInterface(ABC):
    """Abstract base class that all AX.25 transport interfaces must implement.

    Concrete subclasses include KISSInterface (serial port) and
    AGWPEInterface (TCP/IP to AGWPE server). Any new transport type must
    also inherit from this class and implement all abstract methods.

    The class also provides a simple callback registry so callers can
    be notified of events like "frame received" or "connected" without
    polling.

    Attributes:
        connected: True if the transport currently has an active link.

    Example::

        class MyTransport(TransportInterface):
            def connect(self): ...
            def disconnect(self): ...
            def send_frame(self, frame): ...
            def receive(self, timeout=None): ...
    """

    def __init__(self) -> None:
        """Initialize the base state (not connected, no callbacks)."""
        self.connected: bool = False
        self._callbacks: Dict[str, Callable] = {}

    # -----------------------------------------------------------------------
    # Abstract interface (must be implemented by subclasses)
    # -----------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Open the transport connection (serial port, TCP socket, etc.).

        Must set ``self.connected = True`` on success.

        Raises:
            TransportError: If the connection cannot be opened. Callers
                should catch this and report it to the user.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the transport connection and clean up resources.

        Must set ``self.connected = False``. Should not raise if the
        transport is already disconnected.

        Raises:
            TransportError: If closing fails in an unexpected way.
        """

    @abstractmethod
    def send_frame(self, frame: AX25Frame, **kwargs: Any) -> None:
        """Send one AX.25 frame over the transport.

        The exact encoding depends on the transport type. KISS wraps
        the frame in FEND bytes with escaping. AGWPE wraps it in a
        36-byte binary header.

        Args:
            frame: The AX25Frame to transmit. Must already be built and
                validated.
            **kwargs: Transport-specific extra parameters. For example,
                AGWPE uses ``port`` to select the radio port.

        Raises:
            TransportError: If the frame cannot be sent. This includes
                serial write errors, TCP errors, and size limits.
        """

    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> AX25Frame:
        """Wait for and return the next incoming frame.

        Blocks until a frame arrives or the timeout expires.

        Args:
            timeout: How many seconds to wait. None means wait forever.
                0 means return immediately (non-blocking check).

        Returns:
            The next received AX25Frame.

        Raises:
            TransportError: If the timeout expires or a read error occurs.
        """

    # -----------------------------------------------------------------------
    # Callback registry (available to all transports)
    # -----------------------------------------------------------------------

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a function to call when a named event happens.

        Common event names are:
          - ``"frame_received"``: A new frame has arrived.
          - ``"connected"``: The transport link came up.
          - ``"disconnected"``: The transport link went down.

        Subclasses may define additional event names.

        Args:
            event: The event name string. Must not be empty.
            callback: A callable that accepts whatever arguments the
                event provides. See the specific transport for details.

        Example::

            def on_frame(frame):
                print(f"Got frame from {frame.source.callsign}")

            transport.register_callback("frame_received", on_frame)
        """
        if not event:
            raise ValueError("event name must not be empty")
        self._callbacks[event] = callback
        logger.debug("Registered callback for event '%s'", event)

    def _trigger_callback(self, event: str, *args: Any) -> None:
        """Fire a registered callback with the given arguments.

        If no callback is registered for the event, does nothing.
        If the callback raises, logs the error and does not re-raise
        (so one bad callback does not break the reader thread).

        Args:
            event: The event name. If not registered, silently ignored.
            *args: Arguments to pass to the callback function.
        """
        if event not in self._callbacks:
            return

        try:
            self._callbacks[event](*args)
            logger.debug("Triggered callback for event '%s'", event)
        except Exception as exc:
            logger.error(
                "Callback for event '%s' raised an exception: %s", event, exc
            )


# ---------------------------------------------------------------------------
# Transport validation helper
# ---------------------------------------------------------------------------

def validate_frame_for_transport(
    frame: AX25Frame,
    transport_type: str,
) -> None:
    """Check that a frame is small enough for the given transport type.

    Different transport types have different practical frame size limits:
      - KISS: Typically up to 512 bytes after bit stuffing.
      - AGWPE: Practical limit around 4096 bytes.

    This function encodes the frame to bytes and checks the encoded length.

    Args:
        frame: The AX25Frame to check. It will be encoded to measure size.
        transport_type: The transport type string: ``"KISS"`` or ``"AGWPE"``.
            Case-sensitive.

    Raises:
        TransportError: If the encoded frame is too large for the transport.

    Example::

        validate_frame_for_transport(my_frame, "KISS")
        transport.send_frame(my_frame)
    """
    encoded = frame.encode()
    size = len(encoded)

    logger.debug(
        "validate_frame_for_transport: transport=%s encoded_size=%d bytes",
        transport_type, size,
    )

    if transport_type == "KISS":
        limit = 512
        if size > limit:
            raise TransportError(
                f"Frame is {size} bytes, which exceeds the KISS transport "
                f"limit of {limit} bytes -- reduce the information field size"
            )

    elif transport_type == "AGWPE":
        limit = 4096
        if size > limit:
            raise TransportError(
                f"Frame is {size} bytes, which exceeds the AGWPE practical "
                f"limit of {limit} bytes -- reduce the information field size"
            )

    else:
        logger.warning(
            "validate_frame_for_transport: unknown transport type '%s' -- "
            "skipping size check",
            transport_type,
        )

    logger.debug(
        "validate_frame_for_transport: %s frame OK (%d bytes)",
        transport_type, size,
    )
