# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
Base Transport Interfaces

Defines abstract base classes for both synchronous and asynchronous transports.

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import logging
import threading
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Union, Dict, Any, List, Awaitable, TypeVar, Generic
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Type variables for generic transport types
T = TypeVar('T')
AsyncT = TypeVar('AsyncT')

class TransportError(Exception):
    """Base exception for transport errors"""
    pass

class TransportTimeoutError(TransportError):
    """Timeout exception for transport operations"""
    pass

class TransportConnectionError(TransportError):
    """Connection-related transport errors"""
    pass

class TransportConfigurationError(TransportError):
    """Configuration-related transport errors"""
    pass

class TransportState(Enum):
    """Transport connection states"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()
    ERROR = auto()

@dataclass
class TransportConfig:
    """Base transport configuration"""
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    buffer_size: int = 4096
    auto_reconnect: bool = False
    reconnect_interval: float = 5.0

@dataclass
class ConnectionInfo:
    """Connection status information"""
    state: TransportState
    local_address: Optional[str] = None
    remote_address: Optional[str] = None
    connected_at: Optional[float] = None
    last_activity: Optional[float] = None
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0

class BaseTransport(ABC, Generic[T]):
    """
    Abstract base class for synchronous transports.
    
    Provides:
    - Connection management
    - Frame sending/receiving
    - Error handling
    - Statistics tracking
    - Thread-safe operations
    
    Attributes:
        on_frame_received: Callback for received frames
        on_connection_state: Callback for connection state changes
        on_error: Callback for transport errors
    """
    
    def __init__(self, config: Optional[TransportConfig] = None):
        """Initialize base transport.
        
        Args:
            config: Transport configuration
        """
        self.config = config or TransportConfig()
        
        # Connection state
        self._state = TransportState.DISCONNECTED
        self._connection_info = ConnectionInfo(state=self._state)
        
        # Callbacks
        self.on_frame_received: Optional[Callable[[bytes], None]] = None
        self.on_connection_state: Optional[Callable[[TransportState], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Synchronization
        self._lock = threading.RLock()
        self._running = False
        self._connect_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'connect_attempts': 0,
            'disconnects': 0,
            'frames_sent': 0,
            'frames_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'errors': 0,
            'reconnects': 0,
            'last_error': None,
            'uptime': 0.0
        }
        
        # Connection management
        self._connect_time = 0.0
        self._last_activity = 0.0
        
        logger.info(f"Initialized BaseTransport with config: {self.config}")

    @abstractmethod
    def connect(self) -> None:
        """Connect to the transport medium"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the transport medium"""
        pass

    @abstractmethod
    def send_frame(self, frame: bytes) -> None:
        """
        Send a frame over the transport.
        
        Args:
            frame: Raw bytes to send
        """
        pass

    def start(self) -> None:
        """Start any background processing"""
        if self._running:
            logger.warning("Transport already running")
            return
            
        self._running = True
        self._connect_time = time.time()
        self._last_activity = time.time()
        
        logger.info("BaseTransport started")

    def stop(self) -> None:
        """Stop background processing"""
        self._running = False
        
        # Clean up connection
        try:
            self.disconnect()
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
            
        self._update_uptime()
        logger.info("BaseTransport stopped")

    def _update_state(self, new_state: TransportState) -> None:
        """Update connection state with callback.
        
        Args:
            new_state: New transport state
        """
        with self._lock:
            old_state = self._state
            if old_state != new_state:
                self._state = new_state
                self._connection_info.state = new_state
                
                logger.info(f"Transport state: {old_state.name} -> {new_state.name}")
                
                if self.on_connection_state:
                    try:
                        self.on_connection_state(new_state)
                    except Exception as e:
                        logger.error(f"Connection state callback failed: {e}")

    def _handle_error(self, error: Exception) -> None:
        """Internal error handling.
        
        Args:
            error: Exception that occurred
        """
        with self._lock:
            self._stats['errors'] += 1
            self._stats['last_error'] = str(error)
            self._connection_info.errors += 1
            
            logger.error(f"Transport error: {error}")
            
            if self.on_error:
                try:
                    self.on_error(error)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")

    def _update_activity(self) -> None:
        """Update last activity timestamp."""
        with self._lock:
            self._last_activity = time.time()
            self._connection_info.last_activity = self._last_activity

    def _update_uptime(self) -> None:
        """Update uptime statistics."""
        if self._connect_time > 0:
            self._stats['uptime'] = time.time() - self._connect_time

    def _increment_stats(self, stat_name: str) -> None:
        """Increment a statistics counter.
        
        Args:
            stat_name: Name of statistic to increment
        """
        with self._lock:
            if stat_name in self._stats:
                self._stats[stat_name] += 1
            else:
                logger.warning(f"Unknown statistic: {stat_name}")

    def is_connected(self) -> bool:
        """Check if transport is connected.
        
        Returns:
            True if connected, False otherwise
        """
        with self._lock:
            return self._state == TransportState.CONNECTED

    def get_connection_info(self) -> ConnectionInfo:
        """Get connection information.
        
        Returns:
            ConnectionInfo object with current status
        """
        with self._lock:
            return ConnectionInfo(
                state=self._state,
                local_address=self._connection_info.local_address,
                remote_address=self._connection_info.remote_address,
                connected_at=self._connection_info.connected_at,
                last_activity=self._connection_info.last_activity,
                bytes_sent=self._connection_info.bytes_sent,
                bytes_received=self._connection_info.bytes_received,
                errors=self._connection_info.errors
            )

    def get_stats(self) -> Dict[str, Union[int, float, str, None]]:
        """Get transport statistics.
        
        Returns:
            Dictionary of transport statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats['state'] = self._state.name
            stats['uptime'] = time.time() - self._connect_time if self._connect_time > 0 else 0.0
            stats['last_activity'] = self._last_activity
            return stats

    def reset_stats(self) -> None:
        """Reset transport statistics."""
        with self._lock:
            self._stats = {
                'connect_attempts': 0,
                'disconnects': 0,
                'frames_sent': 0,
                'frames_received': 0,
                'bytes_sent': 0,
                'bytes_received': 0,
                'errors': 0,
                'reconnects': 0,
                'last_error': None,
                'uptime': 0.0
            }
            logger.debug("Transport statistics reset")

    def reconnect(self) -> None:
        """Attempt to reconnect transport."""
        with self._connect_lock:
            self._increment_stats('reconnects')
            logger.info("Attempting transport reconnection")
            
            try:
                self.disconnect()
                time.sleep(self.config.reconnect_interval)
                self.connect()
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                self._handle_error(e)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def __repr__(self) -> str:
        info = self.get_connection_info()
        return (f"BaseTransport(state={info.state.name}, "
                f"local={info.local_address}, remote={info.remote_address})")

class AsyncBaseTransport(ABC, Generic[AsyncT]):
    """
    Abstract base class for asynchronous transports.
    
    Provides:
    - Async connection management
    - Async frame sending/receiving
    - Async error handling
    - Statistics tracking
    - Coroutine-based operations
    
    Attributes:
        on_frame_received: Async callback for received frames
        on_connection_state: Async callback for connection state changes
        on_error: Async callback for transport errors
    """
    
    def __init__(self, config: Optional[TransportConfig] = None, max_queue_size: int = 1000):
        """Initialize async base transport.
        
        Args:
            config: Transport configuration
            max_queue_size: Maximum receive queue size
        """
        self.config = config or TransportConfig()
        self._frame_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Connection state
        self._state = TransportState.DISCONNECTED
        self._connection_info = ConnectionInfo(state=self._state)
        
        # Callbacks
        self.on_frame_received: Optional[Callable[[bytes], None]] = None
        self.on_connection_state: Optional[Callable[[TransportState], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Synchronization
        self._lock = threading.RLock()
        self._running = False
        self._connect_lock = threading.Lock()
        self._connect_event = asyncio.Event()
        
        # Statistics
        self._stats = {
            'connect_attempts': 0,
            'disconnects': 0,
            'frames_sent': 0,
            'frames_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'errors': 0,
            'reconnects': 0,
            'last_error': None,
            'uptime': 0.0
        }
        
        # Connection management
        self._connect_time = 0.0
        self._last_activity = 0.0
        
        logger.info(f"Initialized AsyncBaseTransport with config: {self.config}")

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the transport medium (async)"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the transport medium (async)"""
        pass

    @abstractmethod
    async def send_frame(self, frame: bytes) -> None:
        """
        Send a frame over the transport (async).
        
        Args:
            frame: Raw bytes to send
        """
        pass

    async def start(self) -> None:
        """Start any background processing"""
        if self._running:
            logger.warning("Async transport already running")
            return
            
        self._running = True
        self._connect_time = time.time()
        self._last_activity = time.time()
        
        logger.info("AsyncBaseTransport started")

    async def stop(self) -> None:
        """Stop background processing"""
        self._running = False
        
        # Clean up connection
        try:
            await self.disconnect()
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
            
        self._update_uptime()
        logger.info("AsyncBaseTransport stopped")

    async def recv_frame(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Receive a frame asynchronously.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Received frame or None on timeout
        """
        try:
            if timeout:
                frame = await asyncio.wait_for(
                    self._frame_queue.get(),
                    timeout=timeout
                )
            else:
                frame = await self._frame_queue.get()
            
            if frame:
                await self._frame_queue.task_done()
                self._update_activity()
                
            return frame
        except asyncio.TimeoutError:
            return None

    async def _handle_frame(self, frame: bytes) -> None:
        """Internal frame handling"""
        if self.on_frame_received:
            try:
                if asyncio.iscoroutinefunction(self.on_frame_received):
                    await self.on_frame_received(frame)
                else:
                    self.on_frame_received(frame)
            except Exception as e:
                logger.error(f"Frame handler error: {e}")
        else:
            await self._frame_queue.put(frame)

    async def _handle_error(self, error: Exception) -> None:
        """Internal error handling"""
        with self._lock:
            self._stats['errors'] += 1
            self._stats['last_error'] = str(error)
            self._connection_info.errors += 1
            
            logger.error(f"Async transport error: {error}")
            
            if self.on_error:
                try:
                    if asyncio.iscoroutinefunction(self.on_error):
                        await self.on_error(error)
                    else:
                        self.on_error(error)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")

    async def _update_state(self, new_state: TransportState) -> None:
        """Update connection state with callback.
        
        Args:
            new_state: New transport state
        """
        with self._lock:
            old_state = self._state
            if old_state != new_state:
                self._state = new_state
                self._connection_info.state = new_state
                
                logger.info(f"Async transport state: {old_state.name} -> {new_state.name}")
                
                if self.on_connection_state:
                    try:
                        if asyncio.iscoroutinefunction(self.on_connection_state):
                            await self.on_connection_state(new_state)
                        else:
                            self.on_connection_state(new_state)
                    except Exception as e:
                        logger.error(f"Connection state callback failed: {e}")

    async def _update_activity(self) -> None:
        """Update last activity timestamp."""
        with self._lock:
            self._last_activity = time.time()
            self._connection_info.last_activity = self._last_activity

    async def _update_uptime(self) -> None:
        """Update uptime statistics."""
        if self._connect_time > 0:
            self._stats['uptime'] = time.time() - self._connect_time

    async def _increment_stats(self, stat_name: str) -> None:
        """Increment a statistics counter.
        
        Args:
            stat_name: Name of statistic to increment
        """
        with self._lock:
            if stat_name in self._stats:
                self._stats[stat_name] += 1
            else:
                logger.warning(f"Unknown statistic: {stat_name}")

    async def is_connected(self) -> bool:
        """Check if transport is connected.
        
        Returns:
            True if connected, False otherwise
        """
        with self._lock:
            return self._state == TransportState.CONNECTED

    async def get_connection_info(self) -> ConnectionInfo:
        """Get connection information.
        
        Returns:
            ConnectionInfo object with current status
        """
        with self._lock:
            return ConnectionInfo(
                state=self._state,
                local_address=self._connection_info.local_address,
                remote_address=self._connection_info.remote_address,
                connected_at=self._connection_info.connected_at,
                last_activity=self._connection_info.last_activity,
                bytes_sent=self._connection_info.bytes_sent,
                bytes_received=self._connection_info.bytes_received,
                errors=self._connection_info.errors
            )

    async def get_stats(self) -> Dict[str, Union[int, float, str, None]]:
        """Get transport statistics.
        
        Returns:
            Dictionary of transport statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats['state'] = self._state.name
            stats['uptime'] = time.time() - self._connect_time if self._connect_time > 0 else 0.0
            stats['last_activity'] = self._last_activity
            stats['queue_size'] = self._frame_queue.qsize()
            return stats

    async def reset_stats(self) -> None:
        """Reset transport statistics."""
        with self._lock:
            self._stats = {
                'connect_attempts': 0,
                'disconnects': 0,
                'frames_sent': 0,
                'frames_received': 0,
                'bytes_sent': 0,
                'bytes_received': 0,
                'errors': 0,
                'reconnects': 0,
                'last_error': None,
                'uptime': 0.0
            }
            logger.debug("Async transport statistics reset")

    async def reconnect(self) -> None:
        """Attempt to reconnect transport."""
        async with self._connect_lock:
            await self._increment_stats('reconnects')
            logger.info("Attempting async transport reconnection")
            
            try:
                await self.disconnect()
                await asyncio.sleep(self.config.reconnect_interval)
                await self.connect()
            except Exception as e:
                logger.error(f"Async reconnection failed: {e}")
                await self._handle_error(e)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def __repr__(self) -> str:
        info = self.get_connection_info()
        return (f"AsyncBaseTransport(state={info.state.name}, "
                f"local={info.local_address}, remote={info.remote_address}, "
                f"queue_size={self._frame_queue.qsize()})")

class TransportManager:
    """
    Factory class for creating transports by name/type.
    
    Provides:
    - Transport registration and discovery
    - Transport creation with configuration
    - Transport lifecycle management
    - Statistics aggregation
    """
    
    _transport_types: Dict[str, type] = {}
    _transport_instances: Dict[str, Union[BaseTransport, AsyncBaseTransport]] = {}

    @classmethod
    def register(cls, name: str, transport_class: Union[type[BaseTransport], type[AsyncBaseTransport]]):
        """Register a transport type.
        
        Args:
            name: Transport name (e.g., 'serial', 'tcp', 'kiss')
            transport_class: Transport class to register
        """
        if not issubclass(transport_class, (BaseTransport, AsyncBaseTransport)):
            raise TypeError("Transport class must inherit from BaseTransport or AsyncBaseTransport")
            
        cls._transport_types[name.lower()] = transport_class
        logger.info(f"Registered transport: {name} -> {transport_class.__name__}")

    @classmethod
    def create(
        cls,
        transport_type: str,
        *args,
        async_mode: bool = False,
        **kwargs
    ) -> Union[BaseTransport, AsyncBaseTransport]:
        """
        Create a transport instance.
        
        Args:
            transport_type: Registered transport name
            async_mode: Whether to create async transport
            *args: Positional args for transport constructor
            **kwargs: Keyword args for transport constructor
            
        Returns:
            Configured transport instance
            
        Raises:
            ValueError: If transport type is unknown
            TypeError: If async_mode doesn't match transport type
        """
        trans_type = transport_type.lower()
        if trans_type not in cls._transport_types:
            raise ValueError(f"Unknown transport type: {transport_type}")
            
        transport_class = cls._transport_types[trans_type]
        
        if async_mode:
            if not issubclass(transport_class, AsyncBaseTransport):
                raise TypeError(f"Transport {transport_type} doesn't support async mode")
        else:
            if not issubclass(transport_class, BaseTransport):
                raise TypeError(f"Transport {transport_type} doesn't support sync mode")
                
        instance = transport_class(*args, **kwargs)
        cls._transport_instances[f"{trans_type}_{id(instance)}"] = instance
        
        logger.info(f"Created transport instance: {transport_type} (async={async_mode})")
        return instance

    @classmethod
    def get_registered_transports(cls) -> List[str]:
        """Get list of registered transport names.
        
        Returns:
            List of transport type names
        """
        return list(cls._transport_types.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a transport type.
        
        Args:
            name: Transport name to unregister
        """
        name = name.lower()
        if name in cls._transport_types:
            del cls._transport_types[name]
            logger.info(f"Unregistered transport: {name}")

    @classmethod
    def cleanup(cls) -> None:
        """Clean up all transport instances."""
        for instance_id, instance in list(cls._transport_instances.items()):
            try:
                if isinstance(instance, BaseTransport):
                    instance.stop()
                elif isinstance(instance, AsyncBaseTransport):
                    # Note: Async cleanup would need event loop
                    pass
            except Exception as e:
                logger.error(f"Error cleaning up transport {instance_id}: {e}")
        
        cls._transport_instances.clear()
        logger.info("Transport manager cleaned up")

class TransportValidator:
    """Utility class for validating transport configurations."""
    
    @staticmethod
    def validate_config(config: TransportConfig) -> List[str]:
        """
        Validate transport configuration.
        
        Args:
            config: Transport configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if config.timeout <= 0:
            errors.append("Timeout must be positive")
        if config.retry_attempts < 0:
            errors.append("Retry attempts cannot be negative")
        if config.retry_delay < 0:
            errors.append("Retry delay cannot be negative")
        if config.buffer_size <= 0:
            errors.append("Buffer size must be positive")
        if config.reconnect_interval <= 0 and config.auto_reconnect:
            errors.append("Reconnect interval must be positive when auto_reconnect is enabled")
            
        return errors

    @staticmethod
    def validate_frame(frame: bytes) -> bool:
        """
        Validate frame format.
        
        Args:
            frame: Frame to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(frame, bytes):
            return False
        if len(frame) == 0:
            return False
        # Basic AX.25 frame validation
        if frame[0] != 0x7E or frame[-1] != 0x7E:  # FLAG bytes
            return False
        return True

class TransportStatistics:
    """Aggregated statistics for multiple transports."""
    
    def __init__(self):
        self._transport_stats: Dict[str, Dict[str, Union[int, float]]] = {}
        self._lock = threading.RLock()

    def add_transport_stats(self, transport_name: str, stats: Dict[str, Union[int, float]]) -> None:
        """Add statistics from a transport.
        
        Args:
            transport_name: Name of the transport
            stats: Statistics dictionary from transport
        """
        with self._lock:
            self._transport_stats[transport_name] = stats.copy()

    def get_aggregated_stats(self) -> Dict[str, Union[int, float]]:
        """Get aggregated statistics from all transports.
        
        Returns:
            Aggregated statistics
        """
        with self._lock:
            if not self._transport_stats:
                return {}
            
            # Aggregate basic stats
            total = {
                'connect_attempts': 0,
                'disconnects': 0,
                'frames_sent': 0,
                'frames_received': 0,
                'bytes_sent': 0,
                'bytes_received': 0,
                'errors': 0,
                'reconnects': 0,
                'active_transports': len(self._transport_stats)
            }
            
            for stats in self._transport_stats.values():
                for key in total:
                    if key in stats:
                        total[key] += stats[key]
                        
            return total

    def get_transport_list(self) -> List[str]:
        """Get list of tracked transports.
        
        Returns:
            List of transport names
        """
        with self._lock:
            return list(self._transport_stats.keys())

# Example transport factory functions
def create_transport(connection_string: str, **kwargs) -> BaseTransport:
    """
    Create transport from connection string.
    
    Formats:
    - Serial: "serial:/dev/ttyUSB0:9600"
    - TCP: "tcp:localhost:8001"
    - KISS: "kiss:serial:/dev/ttyUSB0:9600"
    
    Args:
        connection_string: Transport-specific connection string
        **kwargs: Additional transport options
        
    Returns:
        Configured transport instance
    """
    if "://" in connection_string:
        protocol, rest = connection_string.split("://", 1)
    else:
        protocol, rest = connection_string.split(":", 1)
    
    protocol = protocol.lower()
    
    if protocol == "serial":
        # Format: serial:/dev/ttyUSB0:9600
        parts = rest.split(":")
        if len(parts) < 1:
            raise ValueError("Invalid serial connection string")
        port = parts[0]
        baudrate = int(parts[1]) if len(parts) > 1 else 9600
        
        # Import here to avoid circular imports
        from .kiss import SerialKISSInterface
        return SerialKISSInterface(port=port, baudrate=baudrate, **kwargs)
        
    elif protocol == "tcp":
        # Format: tcp:localhost:8001
        parts = rest.split(":")
        if len(parts) < 2:
            raise ValueError("Invalid TCP connection string")
        host = parts[0]
        port = int(parts[1])
        
        from .kiss import TCPKISSInterface
        return TCPKISSInterface(host=host, port=port, **kwargs)
        
    elif protocol.startswith("kiss:"):
        # Format: kiss:serial:/dev/ttyUSB0:9600 or kiss:tcp:localhost:8001
        sub_protocol = protocol.split(":")[1]
        if sub_protocol == "serial":
            parts = rest.split(":")
            port = parts[0]
            baudrate = int(parts[1]) if len(parts) > 1 else 9600
            from .kiss import SerialKISSInterface
            return SerialKISSInterface(port=port, baudrate=baudrate, **kwargs)
        elif sub_protocol == "tcp":
            parts = rest.split(":")
            host = parts[0]
            port = int(parts[1])
            from .kiss import TCPKISSInterface
            return TCPKISSInterface(host=host, port=port, **kwargs)
        else:
            raise ValueError(f"Unknown KISS sub-protocol: {sub_protocol}")
    else:
        raise ValueError(f"Unknown transport protocol: {protocol}")

# Register default transports
def _register_default_transports():
    """Register default transport types."""
    try:
        from .kiss import SerialKISSInterface, TCPKISSInterface
        from .kiss_async import AsyncKISSInterface
        from .agwpe import AGWPEClient
        from .agwpe_async import AsyncAGWPEClient
        
        TransportManager.register("serial", SerialKISSInterface)
        TransportManager.register("tcp", TCPKISSInterface)
        TransportManager.register("async_kiss", AsyncKISSInterface)
        TransportManager.register("agwpe", AGWPEClient)
        TransportManager.register("async_agwpe", AsyncAGWPEClient)
        
        logger.info("Default transports registered")
    except ImportError as e:
        logger.warning(f"Could not register default transports: {e}")

# Auto-register transports on import
_register_default_transports()

# Cleanup on exit
import atexit
atexit.register(TransportManager.cleanup)
