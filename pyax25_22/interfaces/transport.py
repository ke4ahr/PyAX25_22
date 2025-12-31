# pyax25_22/interfaces/transport.py
"""
Base Transport Interfaces

Defines abstract base classes for both synchronous and asynchronous transports.

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, Callable, Union
import asyncio

logger = logging.getLogger(__name__)

class TransportError(Exception):
    """Base exception for transport errors"""

class BaseTransport(ABC):
    """
    Abstract base class for synchronous transports.
    
    Attributes:
        on_frame_received: Callback for received frames
    """
    def __init__(self):
        self.on_frame_received: Optional[Callable[[bytes], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self._lock = threading.Lock()
        self._running = False

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
        self._running = True

    def stop(self) -> None:
        """Stop background processing"""
        self._running = False

    def _handle_error(self, error: Exception) -> None:
        """Internal error handling"""
        logger.error(f"Transport error: {error}")
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class AsyncBaseTransport(ABC):
    """
    Abstract base class for asynchronous transports.
    
    Attributes:
        on_frame_received: Async callback for received frames
    """
    def __init__(self, max_queue_size: int = 100):
        self.on_frame_received: Optional[Callable[[bytes], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self._frame_queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

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

    async def recv_frame(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Receive a frame asynchronously.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Received frame or None on timeout
        """
        try:
            return await asyncio.wait_for(
                self._frame_queue.get(),
                timeout=timeout
            )
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
        logger.error(f"Transport error: {error}")
        if self.on_error:
            try:
                if asyncio.iscoroutinefunction(self.on_error):
                    await self.on_error(error)
                else:
                    self.on_error(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class TransportManager:
    """
    Factory class for creating transports by name/type.
    
    Example:
        TransportManager.create('kiss', device='/dev/ttyUSB0')
    """
    _transport_types = {}

    @classmethod
    def register(cls, name: str, transport_class: Union[BaseTransport, AsyncBaseTransport]):
        """Register a transport type"""
        cls._transport_types[name.lower()] = transport_class

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
        """
        trans_type = transport_type.lower()
        if trans_type not in cls._transport_types:
            raise ValueError(f"Unknown transport type: {transport_type}")
            
        transport_class = cls._transport_types[trans_type]
        
        if async_mode:
            if not issubclass(transport_class, AsyncBaseTransport):
                raise ValueError(f"{transport_type} doesn't support async")
        else:
            if not issubclass(transport_class, BaseTransport):
                raise ValueError(f"{transport_type} doesn't support sync")
                
        return transport_class(*args, **kwargs)

# Example registrations (would be in transport implementations)
# TransportManager.register('kiss', KISSInterface)
# TransportManager.register('agwpe', AGWPEClient)
