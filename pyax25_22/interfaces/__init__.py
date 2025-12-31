# pyax25_22/interfaces/__init__.py
"""
PyAX25_22 Transport Interfaces

Provides:
- KISS interface (serial/TCP, sync/async)
- AGWPE client (TCP, sync/async)
- Base classes for custom transports

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

# KISS Interface
from .kiss import (
    KISSInterface,
    KISSCommand,
    SerialKISSInterface,
    TCPKISSInterface
)
from .kiss_async import (
    AsyncKISSInterface,
    AsyncSerialKISSInterface,
    AsyncTCPKISSInterface
)

# AGWPE Interface
from .agwpe import (
    AGWClient,
    AGWHeader,
    AGWFrameType
)
from .agwpe_async import (
    AsyncAGWClient
)

# Base Classes
from .transport import (
    BaseTransport,
    AsyncBaseTransport
)

# Exceptions
from .exceptions import (
    TransportError,
    KISSProtocolError,
    AGWProtocolError
)

__all__ = [
    # KISS
    'KISSInterface',
    'KISSCommand',
    'SerialKISSInterface',
    'TCPKISSInterface',
    'AsyncKISSInterface',
    'AsyncSerialKISSInterface',
    'AsyncTCPKISSInterface',
    
    # AGWPE
    'AGWClient',
    'AsyncAGWClient',
    'AGWHeader',
    'AGWFrameType',
    
    # Base
    'BaseTransport',
    'AsyncBaseTransport',
    
    # Exceptions
    'TransportError',
    'KISSProtocolError',
    'AGWProtocolError'
]

def create_transport(connection_string: str, **kwargs) -> BaseTransport:
    """
    Create transport from connection string.
    
    Formats:
    - Serial: "serial:/dev/ttyUSB0:9600"
    - TCP: "tcp:localhost:8001"
    
    Args:
        connection_string: Transport-specific connection string
        **kwargs: Additional transport options
        
    Returns:
        Configured transport instance
    """
    if connection_string.startswith("serial:"):
        _, port, baud = connection_string.split(":")
        return SerialKISSInterface(port, int(baud), **kwargs)
    elif connection_string.startswith("tcp:"):
        _, host, port = connection_string.split(":")
        return TCPKISSInterface(host, int(port), **kwargs)
    else:
        raise TransportError(f"Unknown transport: {connection_string}")
