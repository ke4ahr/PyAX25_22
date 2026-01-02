# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
PyAX25_22 Transport Interfaces

Provides:
- KISS interface (serial/TCP, sync/async)
- AGWPE client (TCP, sync/async)
- Base classes for custom transports

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import logging
import sys
from typing import Optional, Union, Callable, List, Dict, Any, Type, TypeVar, Generic

# Version and metadata
__version__ = "0.1.0"
__author__ = "Kris Kirby"
__email__ = "ke4ahr@example.com"
__description__ = "Transport interfaces for PyAX25_22"
__license__ = "LGPLv3.0"
__copyright__ = "Copyright (C) 2024 Kris Kirby, KE4AHR"

# Import core exceptions for transport interfaces
from ..core.exceptions import (
    AX25Error,
    TransportError as CoreTransportError
)

# Type variables for generic transport types
T = TypeVar('T')  # Synchronous transport type
AsyncT = TypeVar('AsyncT')  # Asynchronous transport type

# Transport interface imports and re-exports
try:
    # KISS Interface
    from .kiss import (
        KISSInterface,
        KISSCommand,
        SerialKISSInterface,
        TCPKISSInterface,
        KISSFrame,
        KISSStatistics,
        KISSProtocolError,
        TransportError as KISSTransportError
    )
    
    # Async KISS Interface  
    from .kiss_async import (
        AsyncKISSInterface,
        AsyncSerialKISSInterface,
        AsyncKISSFrame,
        AsyncFrameQueue,
        AsyncKISSConfig,
        AsyncKISSStats
    )
    
    # KISS TCP Interface
    from .kiss_tcp import (
        TCPKISSInterface as TCPKISSImpl,
        AsyncTCPKISSInterface,
        TCPState,
        TCPKeepaliveManager,
        TCPReconnectManager,
        TCPConnectionInfo,
        TCPKeepaliveConfig,
        TCPReconnectConfig,
        TransportManager as TCPTransportManager
    )
    
    # AGWPE Interface
    from .agwpe import (
        AGWClient,
        AGWHeader,
        AGWFrameType,
        AGWProtocolError,
        TransportError as AGWTransportError
    )
    
    # Async AGWPE Interface
    from .agwpe_async import (
        AsyncAGWClient,
        AsyncAGWHeader,
        AsyncAGWFrameType,
        AsyncAGWProtocolError
    )
    
    # Base Classes
    from .transport import (
        BaseTransport,
        AsyncBaseTransport,
        TransportValidator,
        TransportStatistics,
        TransportManager,
        TransportConfig,
        ConnectionInfo,
        TransportState,
        create_transport
    )
    
    # Exceptions
    from .exceptions import (
        TransportError as InterfaceTransportError,
        KISSProtocolError as InterfaceKISSProtocolError,
        AGWProtocolError as InterfaceAGWProtocolError
    )
    
    _TRANSPORT_IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import transport interfaces: {e}")
    _TRANSPORT_IMPORTS_SUCCESSFUL = False
    raise

# Public API exports
__all__ = [
    # Version and metadata
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    '__license__',
    '__copyright__',
    
    # KISS Interface
    'KISSInterface',
    'KISSCommand',
    'SerialKISSInterface',
    'TCPKISSInterface',
    'KISSFrame',
    'KISSStatistics',
    'KISSProtocolError',
    'KISSTransportError',
    
    # Async KISS Interface
    'AsyncKISSInterface',
    'AsyncSerialKISSInterface',
    'AsyncKISSFrame',
    'AsyncFrameQueue',
    'AsyncKISSConfig',
    'AsyncKISSStats',
    
    # KISS TCP Interface
    'TCPKISSImpl',
    'AsyncTCPKISSInterface',
    'TCPState',
    'TCPKeepaliveManager',
    'TCPReconnectManager',
    'TCPConnectionInfo',
    'TCPKeepaliveConfig',
    'TCPReconnectConfig',
    'TCPTransportManager',
    
    # AGWPE Interface
    'AGWClient',
    'AGWHeader',
    'AGWFrameType',
    'AGWProtocolError',
    'AGWTransportError',
    
    # Async AGWPE Interface
    'AsyncAGWClient',
    'AsyncAGWHeader',
    'AsyncAGWFrameType',
    'AsyncAGWProtocolError',
    
    # Base Classes
    'BaseTransport',
    'AsyncBaseTransport',
    'TransportValidator',
    'TransportStatistics',
    'TransportManager',
    'TransportConfig',
    'ConnectionInfo',
    'TransportState',
    'create_transport',
    
    # Exceptions
    'InterfaceTransportError',
    'InterfaceKISSProtocolError',
    'InterfaceAGWProtocolError',
    
    # Convenience aliases and utilities
    'CoreTransportError',
]

# Transport type definitions for type hints
KISSTransport = Union[SerialKISSInterface, TCPKISSInterface]
AsyncKISSTransport = Union[AsyncKISSInterface, AsyncSerialKISSInterface, AsyncTCPKISSInterface]
AGWPTransport = Union[AGWClient]
AsyncAGWPTransport = Union[AsyncAGWClient]
AnyTransport = Union[KISSTransport, AGWPTransport]
AnyAsyncTransport = Union[AsyncKISSTransport, AsyncAGWPTransport]

# Transport factory and management
class TransportFactory:
    """
    Factory class for creating transport instances with validation.
    
    Provides:
    - Transport creation with automatic type detection
    - Configuration validation
    - Error handling and recovery
    - Transport lifecycle management
    """
    
    @staticmethod
    def create(
        transport_type: str,
        *args,
        async_mode: bool = False,
        **kwargs
    ) -> Union[BaseTransport, AsyncBaseTransport]:
        """
        Create a transport instance with validation.
        
        Args:
            transport_type: Type of transport ('serial', 'tcp', 'kiss', 'agwpe')
            *args: Positional arguments for transport constructor
            async_mode: Whether to create async transport
            **kwargs: Keyword arguments for transport constructor
            
        Returns:
            Configured transport instance
            
        Raises:
            ValueError: If transport type is unknown or invalid
            TypeError: If async_mode doesn't match transport type
        """
        transport_type = transport_type.lower().strip()
        
        # Validate transport type
        valid_types = ['serial', 'tcp', 'kiss', 'agwpe', 'async_kiss', 'async_agwpe']
        if transport_type not in valid_types:
            raise ValueError(f"Unknown transport type: {transport_type}. "
                           f"Valid types: {', '.join(valid_types)}")
        
        # Create transport based on type
        if transport_type in ['serial', 'kiss']:
            if async_mode:
                raise TypeError(f"Transport {transport_type} doesn't support async mode. "
                              f"Use 'async_kiss' instead.")
            return TransportFactory._create_kiss_transport('serial', *args, **kwargs)
            
        elif transport_type == 'tcp':
            if async_mode:
                raise TypeError(f"Transport {transport_type} doesn't support async mode. "
                              f"Use 'async_kiss' instead.")
            return TransportFactory._create_kiss_transport('tcp', *args, **kwargs)
            
        elif transport_type == 'async_kiss':
            return TransportFactory._create_async_kiss_transport(*args, **kwargs)
            
        elif transport_type in ['agwpe', 'async_agwpe']:
            return TransportFactory._create_agwpe_transport(transport_type == 'async_agwpe', *args, **kwargs)
        
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

    @staticmethod
    def _create_kiss_transport(
        transport_subtype: str,
        *args,
        **kwargs
    ) -> Union[SerialKISSInterface, TCPKISSInterface]:
        """Create KISS transport (serial or TCP)."""
        if transport_subtype == 'serial':
            if len(args) < 1:
                raise ValueError("Serial transport requires port argument")
            return SerialKISSInterface(*args, **kwargs)
        elif transport_subtype == 'tcp':
            if len(args) < 2:
                raise ValueError("TCP transport requires host and port arguments")
            return TCPKISSInterface(*args, **kwargs)
        else:
            raise ValueError(f"Unknown KISS transport subtype: {transport_subtype}")

    @staticmethod
    def _create_async_kiss_transport(
        *args,
        **kwargs
    ) -> AsyncKISSInterface:
        """Create async KISS transport."""
        return AsyncKISSInterface(*args, **kwargs)

    @staticmethod
    def _create_agwpe_transport(
        async_mode: bool,
        *args,
        **kwargs
    ) -> Union[AGWClient, AsyncAGWClient]:
        """Create AGWPE transport (sync or async)."""
        if async_mode:
            return AsyncAGWClient(*args, **kwargs)
        else:
            return AGWClient(*args, **kwargs)

    @staticmethod
    def validate_config(transport_type: str, config: Dict[str, Any]) -> List[str]:
        """
        Validate transport configuration.
        
        Args:
            transport_type: Type of transport
            config: Configuration dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Common validation
        if 'timeout' in config:
            if not isinstance(config['timeout'], (int, float)) or config['timeout'] <= 0:
                errors.append("Timeout must be positive number")
        
        # Transport-specific validation
        if transport_type == 'serial':
            if 'port' not in config or not config['port']:
                errors.append("Serial transport requires 'port' parameter")
            if 'baudrate' in config and config['baudrate'] <= 0:
                errors.append("Baudrate must be positive")
                
        elif transport_type in ['tcp', 'agwpe']:
            if 'host' not in config or not config['host']:
                errors.append(f"{transport_type.upper()} transport requires 'host' parameter")
            if 'port' not in config or config['port'] <= 0:
                errors.append(f"{transport_type.upper()} transport requires positive 'port' parameter")
        
        return errors

# Transport connection manager
class TransportConnectionManager:
    """
    Manager for multiple transport connections with pooling and monitoring.
    
    Features:
    - Connection pooling
    - Automatic reconnection
    - Connection monitoring and health checks
    - Resource cleanup and lifecycle management
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self._connections: Dict[str, Union[BaseTransport, AsyncBaseTransport]] = {}
        self._connection_configs: Dict[str, Dict[str, Any]] = {}
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        
        logger = logging.getLogger(__name__)
        logger.info("Transport connection manager initialized")

    def add_connection(
        self,
        name: str,
        transport_type: str,
        *args,
        async_mode: bool = False,
        **kwargs
    ) -> Union[BaseTransport, AsyncBaseTransport]:
        """
        Add a new transport connection.
        
        Args:
            name: Connection name
            transport_type: Type of transport
            *args: Transport constructor arguments
            async_mode: Whether to use async transport
            **kwargs: Transport constructor keyword arguments
            
        Returns:
            Created transport instance
            
        Raises:
            ValueError: If connection name already exists
        """
        if name in self._connections:
            raise ValueError(f"Connection '{name}' already exists")
        
        # Store configuration
        self._connection_configs[name] = {
            'transport_type': transport_type,
            'args': args,
            'kwargs': kwargs,
            'async_mode': async_mode
        }
        
        # Create transport
        transport = TransportFactory.create(transport_type, *args, async_mode=async_mode, **kwargs)
        self._connections[name] = transport
        
        logger = logging.getLogger(__name__)
        logger.info(f"Added transport connection: {name} ({transport_type})")
        
        return transport

    def get_connection(self, name: str) -> Optional[Union[BaseTransport, AsyncBaseTransport]]:
        """
        Get a transport connection by name.
        
        Args:
            name: Connection name
            
        Returns:
            Transport instance or None if not found
        """
        return self._connections.get(name)

    def remove_connection(self, name: str) -> bool:
        """
        Remove a transport connection.
        
        Args:
            name: Connection name
            
        Returns:
            True if connection was removed, False if not found
        """
        if name not in self._connections:
            return False
        
        transport = self._connections.pop(name)
        config = self._connection_configs.pop(name, {})
        
        # Clean up transport
        try:
            if hasattr(transport, 'stop'):
                transport.stop()
            elif hasattr(transport, 'disconnect'):
                transport.disconnect()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Error cleaning up transport {name}: {e}")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Removed transport connection: {name}")
        
        return True

    def reconnect(self, name: str) -> bool:
        """
        Reconnect a transport connection.
        
        Args:
            name: Connection name
            
        Returns:
            True if reconnected successfully, False otherwise
        """
        if name not in self._connections:
            return False
        
        config = self._connection_configs.get(name, {})
        if not config:
            return False
        
        try:
            # Remove old connection
            self.remove_connection(name)
            
            # Create new connection
            transport = TransportFactory.create(
                config['transport_type'],
                *config['args'],
                async_mode=config.get('async_mode', False),
                **config['kwargs']
            )
            
            self._connections[name] = transport
            
            logger = logging.getLogger(__name__)
            logger.info(f"Reconnected transport: {name}")
            
            return True
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to reconnect transport {name}: {e}")
            return False

    def get_all_connections(self) -> Dict[str, Union[BaseTransport, AsyncBaseTransport]]:
        """Get all active connections."""
        return self._connections.copy()

    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections."""
        status = {}
        
        for name, transport in self._connections.items():
            try:
                if hasattr(transport, 'get_status'):
                    status[name] = transport.get_status()
                elif hasattr(transport, 'get_connection_info'):
                    status[name] = transport.get_connection_info()
                elif hasattr(transport, 'is_connected'):
                    status[name] = {
                        'connected': transport.is_connected(),
                        'type': type(transport).__name__
                    }
                else:
                    status[name] = {
                        'status': 'unknown',
                        'type': type(transport).__name__
                    }
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Error getting status for {name}: {e}")
                status[name] = {
                    'status': 'error',
                    'error': str(e),
                    'type': type(transport).__name__
                }
        
        return status

    def cleanup(self) -> None:
        """Clean up all connections."""
        for name in list(self._connections.keys()):
            self.remove_connection(name)

# Convenience functions for common transport operations
def create_kiss_serial(
    port: str,
    baudrate: int = 9600,
    tnc_address: int = 0,
    **kwargs
) -> SerialKISSInterface:
    """
    Create a KISS serial transport.
    
    Args:
        port: Serial port device
        baudrate: Baud rate
        tnc_address: TNC address for multi-drop
        **kwargs: Additional configuration
        
    Returns:
        Configured SerialKISSInterface
        
    Example:
        >>> serial_kiss = create_kiss_serial('/dev/ttyUSB0', 9600, tnc_address=1)
    """
    return SerialKISSInterface(
        port=port,
        baudrate=baudrate,
        tnc_address=tnc_address,
        **kwargs
    )

def create_kiss_tcp(
    host: str,
    port: int = 8001,
    tnc_address: int = 0,
    **kwargs
) -> TCPKISSInterface:
    """
    Create a KISS TCP transport.
    
    Args:
        host: TCP hostname/IP
        port: TCP port
        tnc_address: TNC address for multi-drop
        **kwargs: Additional configuration
        
    Returns:
        Configured TCPKISSInterface
        
    Example:
        >>> tcp_kiss = create_kiss_tcp('localhost', 8001, tnc_address=1)
    """
    return TCPKISSInterface(
        host=host,
        port=port,
        tnc_address=tnc_address,
        **kwargs
    )

def create_agwpe_client(
    host: str = 'localhost',
    port: int = 8000,
    callsign: str = 'NOCALL',
    **kwargs
) -> AGWClient:
    """
    Create an AGWPE client.
    
    Args:
        host: AGWPE server hostname/IP
        port: AGWPE server port
        callsign: Client callsign
        **kwargs: Additional configuration
        
    Returns:
        Configured AGWClient
        
    Example:
        >>> agwpe = create_agwpe_client('localhost', 8000, callsign='MYCALL')
    """
    return AGWClient(
        host=host,
        port=port,
        callsign=callsign,
        **kwargs
    )

# Transport validation and testing utilities
class TransportTester:
    """
    Utilities for testing transport connections and configurations.
    """
    
    @staticmethod
    async def test_connection(
        transport: Union[BaseTransport, AsyncBaseTransport],
        test_data: bytes = b"Test message",
        timeout: float = 5.0
    ) -> Dict[str, Union[bool, str, float]]:
        """
        Test a transport connection.
        
        Args:
            transport: Transport instance to test
            test_data: Test data to send
            timeout: Test timeout in seconds
            
        Returns:
            Test results dictionary
        """
        results = {
            'success': False,
            'error': None,
            'send_time': 0.0,
            'receive_time': 0.0,
            'roundtrip_time': 0.0,
            'data_sent': len(test_data),
            'data_received': 0
        }
        
        try:
            start_time = time.time()
            
            # Test sending
            send_start = time.time()
            if hasattr(transport, 'send_frame'):
                if hasattr(transport, '_send_frame'):  # Async transport
                    await transport.send_frame(test_data)
                else:  # Sync transport
                    transport.send_frame(test_data)
            send_time = time.time() - send_start
            
            # Test receiving (if supported)
            receive_start = time.time()
            received = None
            if hasattr(transport, 'recv_frame'):
                if hasattr(transport, '_read_data'):  # Async transport
                    received = await transport.recv_frame(timeout=timeout)
                else:  # Sync transport
                    received = transport.recv_frame(timeout=timeout)
            receive_time = time.time() - receive_start
            
            total_time = time.time() - start_time
            
            results.update({
                'success': True,
                'send_time': send_time,
                'receive_time': receive_time,
                'roundtrip_time': total_time,
                'data_received': len(received) if received else 0
            })
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results

    @staticmethod
    def validate_transport_config(
        transport_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate transport configuration.
        
        Args:
            transport_type: Type of transport
            config: Configuration to validate
            
        Returns:
            Validation results
        """
        errors = TransportFactory.validate_config(transport_type, config)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

# Utility functions for transport management
def get_available_transports() -> List[str]:
    """
    Get list of available transport types.
    
    Returns:
        List of available transport type names
    """
    return ['serial', 'tcp', 'kiss', 'agwpe', 'async_kiss', 'async_agwpe']

def is_transport_available(transport_type: str) -> bool:
    """
    Check if a transport type is available.
    
    Args:
        transport_type: Type of transport to check
        
    Returns:
        True if transport is available, False otherwise
    """
    available = get_available_transports()
    return transport_type.lower() in available

# Package initialization and configuration
def configure_transport_logging(level: Union[str, int] = "INFO") -> None:
    """
    Configure transport-specific logging.
    
    Args:
        level: Logging level
    """
    logger = logging.getLogger('pyax25_22.interfaces')
    logger.setLevel(getattr(logging, level.upper()) if isinstance(level, str) else level)
    
    # Create handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Also configure submodules
    for module in ['kiss', 'kiss_tcp', 'agwpe', 'transport']:
        mod_logger = logging.getLogger(f'pyax25_22.interfaces.{module}')
        mod_logger.setLevel(logger.level)
        if not mod_logger.handlers:
            mod_logger.addHandler(handler)

# Initialize transport logging at INFO level by default
configure_transport_logging("INFO")

# Clean up imports to avoid polluting namespace
del logging, sys, Optional, Union, Callable, List, Dict, Any, Type, TypeVar, Generic

# Package initialization message
logger = logging.getLogger(__name__)
logger.debug(f"PyAX25_22 interfaces module initialized: version={__version__}")
