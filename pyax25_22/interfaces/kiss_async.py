# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
Asynchronous KISS Interface Implementation

Implements non-blocking KISS communication with multi-drop support.

"""

import asyncio
import logging
import time
import threading
from typing import Optional, Callable, Union, List, Dict, Any, Awaitable, TypeVar, Generic, Tuple
from enum import IntEnum
from dataclasses import dataclass, field
from collections import deque
import serial_asyncio
import socket
from concurrent.futures import ThreadPoolExecutor

from .kiss import (
    KISSInterface,
    KISSCommand,
    KISSProtocolError,
    TransportError,
    KISSFrame
)
from ..utils.async_thread import run_in_thread

logger = logging.getLogger(__name__)

class AsyncKISSState(IntEnum):
    """Async KISS interface states"""
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    DISCONNECTING = 3
    ERROR = 4

@dataclass
class AsyncKISSConfig:
    """Async KISS configuration"""
    timeout: float = 30.0
    buffer_size: int = 4096
    max_queue_size: int = 1000
    read_chunk_size: int = 1024
    write_timeout: float = 5.0
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 10

@dataclass
class AsyncKISSStats:
    """Async KISS statistics"""
    frames_sent: int = 0
    frames_received: int = 0
    escapes_sent: int = 0
    escapes_received: int = 0
    errors: int = 0
    connection_time: float = 0.0
    last_activity: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    reconnect_attempts: int = 0
    protocol_errors: int = 0

T = TypeVar('T')

class AsyncKISSFrame:
    """Async KISS frame with metadata and async operations"""
    
    def __init__(self, data: bytes, tnc_address: int, command: int, timestamp: Optional[float] = None):
        """Initialize async KISS frame.
        
        Args:
            data: Frame payload
            tnc_address: TNC address (0-15)
            command: KISS command byte
            timestamp: Optional timestamp (defaults to current time)
        """
        if not isinstance(data, bytes):
            raise TypeError("Frame data must be bytes")
        if not 0 <= tnc_address <= 15:
            raise ValueError("TNC address must be 0-15")
        if not 0 <= command <= 255:
            raise ValueError("Command must be 0-255")
            
        self.data = data
        self.tnc_address = tnc_address
        self.command = command
        self.timestamp = timestamp or time.time()
        self.size = len(data)
        self.processed = False
        self.processed_at: Optional[float] = None
        
    def mark_processed(self) -> None:
        """Mark frame as processed."""
        self.processed = True
        self.processed_at = time.time()
        
    def is_data_frame(self) -> bool:
        """Check if this is a data frame."""
        return (self.command & 0x0F) == KISSCommand.DATA
        
    def is_poll_frame(self) -> bool:
        """Check if this is a poll frame."""
        return (self.command & 0x0F) == KISSCommand.POLL
        
    def get_command_name(self) -> str:
        """Get human-readable command name."""
        cmd = self.command & 0x0F
        if cmd == KISSCommand.DATA:
            return "DATA"
        elif cmd == KISSCommand.POLL:
            return "POLL"
        elif cmd == KISSCommand.TX_DELAY:
            return "TX_DELAY"
        elif cmd == KISSCommand.PERSIST:
            return "PERSIST"
        elif cmd == KISSCommand.SLOT_TIME:
            return "SLOT_TIME"
        elif cmd == KISSCommand.TX_TAIL:
            return "TX_TAIL"
        elif cmd == KISSCommand.SET_HW:
            return "SET_HW"
        elif cmd == KISSCommand.RETURN:
            return "RETURN"
        else:
            return f"UNKNOWN_{cmd:02X}"

class AsyncFrameQueue:
    """Async-aware frame queue with priority and filtering"""
    
    def __init__(self, maxsize: int = 1000):
        """Initialize async frame queue.
        
        Args:
            maxsize: Maximum queue size
        """
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._priority_queue = asyncio.Queue(maxsize=maxsize // 2)
        self._filter_tnc: Optional[int] = None
        self._filter_command: Optional[int] = None
        self._lock = asyncio.Lock()
        
    async def put(self, frame: AsyncKISSFrame, priority: bool = False) -> None:
        """Put frame in queue.
        
        Args:
            frame: Frame to queue
            priority: Whether to put in priority queue
        """
        if priority:
            await self._priority_queue.put(frame)
        else:
            await self._queue.put(frame)
            
    async def get(self, timeout: Optional[float] = None) -> Optional[AsyncKISSFrame]:
        """Get frame from queue with optional timeout.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Frame or None if timeout
        """
        try:
            if not self._priority_queue.empty():
                return await self._priority_queue.get()
            elif timeout:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                return await self._queue.get()
        except asyncio.TimeoutError:
            return None
            
    async def get_filtered(self, 
                          tnc_address: Optional[int] = None,
                          command: Optional[int] = None,
                          timeout: Optional[float] = None) -> Optional[AsyncKISSFrame]:
        """Get frame with filtering.
        
        Args:
            tnc_address: Filter by TNC address
            command: Filter by command
            timeout: Optional timeout
            
        Returns:
            Frame matching filters or None
        """
        start_time = time.time()
        
        while True:
            frame = await self.get(timeout=timeout)
            if frame is None:
                return None
                
            # Apply filters
            if tnc_address is not None and frame.tnc_address != tnc_address:
                # Put back in queue if doesn't match
                await self.put(frame)
                continue
                
            if command is not None and (frame.command & 0x0F) != command:
                # Put back in queue if doesn't match
                await self.put(frame)
                continue
                
            return frame
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
                
    def qsize(self) -> int:
        """Get total queue size."""
        return self._queue.qsize() + self._priority_queue.qsize()
        
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty() and self._priority_queue.empty()
        
    def clear(self) -> None:
        """Clear all frames from queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        while not self._priority_queue.empty():
            try:
                self._priority_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

class AsyncKISSInterface(KISSInterface):
    """
    Asynchronous KISS interface using asyncio.
    
    Args:
        tnc_address: TNC address for multi-drop (0-15)
        poll_interval: Poll interval in seconds
        config: Async KISS configuration
    """
    
    def __init__(
        self,
        tnc_address: int = 0,
        poll_interval: float = 0.1,
        config: Optional[AsyncKISSConfig] = None
    ):
        """Initialize async KISS interface.
        
        Args:
            tnc_address: TNC address (0-15) for multi-drop
            poll_interval: Poll interval in seconds
            config: Async configuration
        """
        # Initialize parent class without transport specifics
        super().__init__(tnc_address=tnc_address, poll_interval=poll_interval)
        
        self.config = config or AsyncKISSConfig()
        
        # Async-specific state
        self._state = AsyncKISSState.DISCONNECTED
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._running = False
        
        # Async frame queue
        self._frame_queue = AsyncFrameQueue(self.config.max_queue_size)
        
        # Async callbacks
        self.on_frame_received_async: Optional[Callable[[bytes, int], Awaitable[None]]] = None
        self.on_error_async: Optional[Callable[[Exception], Awaitable[None]]] = None
        self.on_status_async: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        
        # Async statistics
        self._async_stats = AsyncKISSStats()
        
        # Async tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Connection management
        self._reconnect_attempts = 0
        self._last_reconnect = 0.0
        
        # Threading for sync-to-async bridge
        self._sync_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AsyncKISS")
        
        logger.info(f"Initialized AsyncKISSInterface: TNC={tnc_address}, "
                   f"poll={poll_interval}s, max_queue={self.config.max_queue_size}")

    async def connect(self, host: str, port: int) -> None:
        """
        Connect to a KISS TCP server.
        
        Args:
            host: Server hostname/IP
            port: Server port
        """
        try:
            self._state = AsyncKISSState.CONNECTING
            self._async_stats.connection_time = time.time()
            self._async_stats.last_activity = time.time()
            
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.config.timeout
            )
            
            self._state = AsyncKISSState.CONNECTED
            self._reconnect_attempts = 0
            self._async_stats.reconnect_attempts = 0
            
            logger.info(f"Async connected to {host}:{port}")
            
        except asyncio.TimeoutError:
            self._state = AsyncKISSState.ERROR
            raise TransportError(f"Connection timeout to {host}:{port}")
        except Exception as e:
            self._state = AsyncKISSState.ERROR
            logger.error(f"Async connection failed: {e}")
            raise TransportError(f"Connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close the connection"""
        self._state = AsyncKISSState.DISCONNECTING
        
        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
            
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            
        # Close connection
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing writer: {e}")
            
        self._reader = None
        self._writer = None
        self._state = AsyncKISSState.DISCONNECTED
        
        logger.info("Async connection closed")

    async def send_frame(
        self,
        data: bytes,
        cmd: int = KISSCommand.DATA
    ) -> None:
        """
        Asynchronously send a KISS frame.
        
        Args:
            data: Frame payload
            cmd: KISS command (default DATA)
        """
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
        if not 0 <= cmd <= 255:
            raise KISSProtocolError(f"Invalid command: {cmd}")
        if self._state != AsyncKISSState.CONNECTED:
            raise TransportError(f"Cannot send in state {self._state.name}")
            
        try:
            cmd_byte = self._encode_command(cmd)
            frame = self._build_frame(cmd_byte, data)
            
            if self._writer:
                self._writer.write(frame)
                await asyncio.wait_for(
                    self._writer.drain(),
                    timeout=self.config.write_timeout
                )
                
                self._async_stats.frames_sent += 1
                self._async_stats.bytes_sent += len(frame)
                self._async_stats.last_activity = time.time()
                
                logger.debug(f"Async sent frame (cmd=0x{cmd:02x}, len={len(data)})")
            else:
                raise TransportError("Writer not available")
                
        except asyncio.TimeoutError:
            logger.error("Async send timeout")
            await self.disconnect()
            raise TransportError("Send timeout")
        except Exception as e:
            logger.error(f"Async send failed: {e}")
            self._async_stats.errors += 1
            await self._handle_error_async(e)
            raise TransportError(f"Send failed: {e}") from e

    async def send_poll(self, target_tnc: int) -> None:
        """
        Send poll command to target TNC.
        
        Args:
            target_tnc: TNC address (0-15)
        """
        if not 0 <= target_tnc <= 15:
            raise ValueError("Target TNC must be 0-15")
            
        cmd_byte = (target_tnc << 4) | KISSCommand.POLL
        await self.send_frame(b'', cmd=cmd_byte)
        self._last_poll = time.time()
        logger.debug(f"Async sent poll to TNC {target_tnc}")

    async def recv_frame(
        self,
        timeout: Optional[float] = None,
        tnc_address: Optional[int] = None,
        command: Optional[int] = None
    ) -> Optional[Tuple[bytes, int]]:
        """
        Receive a frame asynchronously with filtering.
        
        Args:
            timeout: Maximum wait time (seconds)
            tnc_address: Filter by TNC address
            command: Filter by command type
            
        Returns:
            Tuple of (frame, tnc_address) or None on timeout
        """
        try:
            frame = await self._frame_queue.get_filtered(
                tnc_address=tnc_address,
                command=command,
                timeout=timeout
            )
            
            if frame:
                self._async_stats.frames_received += 1
                self._async_stats.last_activity = time.time()
                return frame.data, frame.tnc_address
            else:
                return None
                
        except Exception as e:
            logger.error(f"Async receive failed: {e}")
            self._async_stats.errors += 1
            return None

    async def recv_frame_raw(
        self,
        timeout: Optional[float] = None
    ) -> Optional[AsyncKISSFrame]:
        """
        Receive raw frame object asynchronously.
        
        Args:
            timeout: Maximum wait time (seconds)
            
        Returns:
            AsyncKISSFrame object or None on timeout
        """
        try:
            return await self._frame_queue.get(timeout=timeout)
        except Exception as e:
            logger.error(f"Async raw receive failed: {e}")
            return None

    async def _receive_loop(self) -> None:
        """Main receive loop for async interface"""
        buffer = bytearray()
        in_frame = False
        escaped = False
        
        while self._running and self._reader:
            try:
                # Read data with timeout
                data = await asyncio.wait_for(
                    self._reader.read(self.config.read_chunk_size),
                    timeout=1.0
                )
                
                if not data:
                    logger.info("Async connection closed by remote")
                    break
                
                self._async_stats.bytes_received += len(data)
                self._async_stats.last_activity = time.time()
                
                # Process bytes
                for byte in data:
                    if byte == self.FEND:
                        if in_frame and buffer:
                            await self._process_frame_async(bytes(buffer))
                            buffer.clear()
                        in_frame = True
                        escaped = False
                    elif in_frame:
                        if escaped:
                            if byte == self.TFEND:
                                buffer.append(self.FEND)
                            elif byte == self.TFESC:
                                buffer.append(self.FESC)
                            else:
                                logger.warning(f"Invalid escape byte: 0x{byte:02x}")
                                self._async_stats.protocol_errors += 1
                            escaped = False
                        elif byte == self.FESC:
                            escaped = True
                            self._async_stats.escapes_received += 1
                        else:
                            buffer.append(byte)
                            
            except asyncio.TimeoutError:
                continue  # Check if still running
            except asyncio.IncompleteReadError:
                logger.info("Async connection closed (incomplete read)")
                break
            except Exception as e:
                logger.error(f"Async receive loop error: {e}")
                await self._handle_error_async(e)
                break
                
        self._running = False
        await self.disconnect()

    async def _process_frame_async(self, frame: bytes) -> None:
        """Handle complete received frame asynchronously"""
        if not frame:
            return
            
        try:
            cmd_byte = frame[0]
            tnc_address = (cmd_byte >> 4) & 0x0F
            cmd = cmd_byte & 0x0F
            payload = frame[1:]
            
            # Create async frame object
            async_frame = AsyncKISSFrame(payload, tnc_address, cmd_byte)
            
            # Queue for application processing
            await self._frame_queue.put(async_frame)
            
            # Handle based on command type
            if cmd == KISSCommand.POLL:
                if self._poll_callback:
                    # Call sync poll callback in thread
                    await run_in_thread(self._poll_callback, tnc_address)
            else:
                # Handle data frame
                if self.on_frame_received_async:
                    try:
                        await self.on_frame_received_async(payload, tnc_address)
                    except Exception as e:
                        logger.error(f"Async frame callback failed: {e}")
                
                # Also call sync callback if registered
                if self._rx_callback:
                    await run_in_thread(self._rx_callback, payload, tnc_address)
                        
            # Update status
            if self.on_status_async:
                try:
                    status = {
                        'type': 'frame_received',
                        'tnc_address': tnc_address,
                        'command': cmd,
                        'command_name': async_frame.get_command_name(),
                        'payload_length': len(payload),
                        'timestamp': time.time(),
                        'is_data_frame': async_frame.is_data_frame(),
                        'is_poll_frame': async_frame.is_poll_frame()
                    }
                    await self.on_status_async(status)
                except Exception as e:
                    logger.error(f"Async status callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Async frame processing error: {e}")
            self._async_stats.errors += 1
            await self._handle_error_async(e)

    async def _handle_error_async(self, error: Exception) -> None:
        """Handle async errors."""
        self._async_stats.errors += 1
        
        if self.on_error_async:
            try:
                await self.on_error_async(error)
            except Exception as e:
                logger.error(f"Async error callback failed: {e}")
        
        # Also call sync error callback
        if self._error_callback:
            await run_in_thread(self._error_callback, error)

    async def start(self) -> None:
        """Start async interface and monitoring"""
        if self._running:
            logger.warning("Async interface already running")
            return
            
        self._running = True
        
        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Async KISS interface started")

    async def stop(self) -> None:
        """Stop async interface"""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            
        await self.disconnect()
        logger.info("Async KISS interface stopped")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self._running:
            try:
                # Check connection state periodically
                if self._state == AsyncKISSState.ERROR and self._reconnect_attempts < self.config.max_reconnect_attempts:
                    await asyncio.sleep(self.config.reconnect_delay)
                    # Reconnection would be handled by application
                    
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(1.0)

    async def get_async_stats(self) -> Dict[str, Union[int, float]]:
        """Get async-specific statistics.
        
        Returns:
            Dictionary of async statistics
        """
        return {
            'state': self._state.name,
            'frames_sent': self._async_stats.frames_sent,
            'frames_received': self._async_stats.frames_received,
            'escapes_sent': self._async_stats.escapes_sent,
            'escapes_received': self._async_stats.escapes_received,
            'errors': self._async_stats.errors,
            'protocol_errors': self._async_stats.protocol_errors,
            'connection_time': self._async_stats.connection_time,
            'last_activity': self._async_stats.last_activity,
            'uptime': time.time() - self._async_stats.connection_time if self._async_stats.connection_time else 0,
            'bytes_sent': self._async_stats.bytes_sent,
            'bytes_received': self._async_stats.bytes_received,
            'reconnect_attempts': self._async_stats.reconnect_attempts,
            'queue_size': self._frame_queue.qsize(),
            'queue_empty': self._frame_queue.empty()
        }

    def register_async_frame_callback(self, callback: Callable[[bytes, int], Awaitable[None]]) -> None:
        """Register async frame receive callback.
        
        Args:
            callback: Async function to call when frame is received
        """
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError("Callback must be an async function")
        self.on_frame_received_async = callback

    def register_async_error_callback(self, callback: Callable[[Exception], Awaitable[None]]) -> None:
        """Register async error callback.
        
        Args:
            callback: Async function to call when error occurs
        """
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError("Callback must be an async function")
        self.on_error_async = callback

    def register_async_status_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Register async status callback.
        
        Args:
            callback: Async function to call with status updates
        """
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError("Callback must be an async function")
        self.on_status_async = callback

    def __repr__(self) -> str:
        return (f"AsyncKISSInterface(tnc={self.tnc_address}, "
                f"state={self._state.name}, "
                f"queue_size={self._frame_queue.qsize()})")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

class AsyncSerialKISSInterface:
    """Async serial KISS interface using serial_asyncio"""
    
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        tnc_address: int = 0,
        **kwargs
    ):
        """Initialize async serial KISS interface.
        
        Args:
            port: Serial port device
            baudrate: Serial baud rate
            tnc_address: TNC address for multi-drop
            **kwargs: Additional configuration
        """
        self.port = port
        self.baudrate = baudrate
        self.tnc_address = tnc_address
        
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._running = False
        
        # Frame queue
        self._frame_queue = AsyncFrameQueue(1000)
        
        # Callbacks
        self.on_frame_received: Optional[Callable[[bytes, int], Awaitable[None]]] = None
        self.on_error: Optional[Callable[[Exception], Awaitable[None]]] = None
        
        # Statistics
        self._stats = {
            'frames_sent': 0,
            'frames_received': 0,
            'errors': 0,
            'connection_time': 0.0,
            'last_activity': 0.0
        }
        
        logger.info(f"Initialized AsyncSerialKISSInterface: {port}@{baudrate}, TNC={tnc_address}")

    async def connect(self) -> None:
        """Connect to serial port asynchronously."""
        try:
            self._reader, self._writer = await serial_asyncio.open_serial_connection(
                url=self.port,
                baudrate=self.baudrate
            )
            
            self._stats['connection_time'] = time.time()
            self._stats['last_activity'] = time.time()
            
            logger.info(f"Async serial connected: {self.port}@{self.baudrate}")
            
        except Exception as e:
            logger.error(f"Async serial connection failed: {e}")
            raise TransportError(f"Serial connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from serial port."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._reader = None
        self._writer = None
        logger.info("Async serial disconnected")

    async def send_frame(self, data: bytes) -> None:
        """Send frame over serial asynchronously."""
        if not self._writer:
            raise TransportError("Not connected")
            
        try:
            # Build KISS frame
            cmd_byte = (self.tnc_address << 4) | 0x00  # DATA command
            frame = self._build_frame(cmd_byte, data)
            
            self._writer.write(frame)
            await self._writer.drain()
            
            self._stats['frames_sent'] += 1
            self._stats['last_activity'] = time.time()
            
            logger.debug(f"Async serial sent: {len(data)} bytes")
            
        except Exception as e:
            logger.error(f"Async serial send failed: {e}")
            self._stats['errors'] += 1
            await self._handle_error(e)
            raise TransportError(f"Send failed: {e}") from e

    async def recv_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, int]]:
        """Receive frame from serial asynchronously."""
        try:
            frame = await self._frame_queue.get(timeout=timeout)
            if frame:
                self._stats['frames_received'] += 1
                self._stats['last_activity'] = time.time()
                return frame.data, frame.tnc_address
            return None
            
        except Exception as e:
            logger.error(f"Async serial receive failed: {e}")
            return None

    def _build_frame(self, cmd: int, data: bytes) -> bytes:
        """Build KISS frame with byte stuffing."""
        FEND = 0xC0
        FESC = 0xDB
        TFEND = 0xDC
        TFESC = 0xDD
        
        escaped = (
            bytes([cmd]) + data
            .replace(bytes([FESC]), bytes([FESC, TFESC]))
            .replace(bytes([FEND]), bytes([FESC, TFEND]))
        )
        return bytes([FEND]) + escaped + bytes([FEND])

    async def _receive_loop(self) -> None:
        """Main receive loop for serial."""
        buffer = bytearray()
        in_frame = False
        escaped = False
        
        while self._running and self._reader:
            try:
                # Read single byte
                byte_data = await self._reader.read(1)
                if not byte_data:
                    break
                    
                byte_val = byte_data[0]
                
                if byte_val == 0xC0:  # FEND
                    if in_frame and buffer:
                        await self._process_frame(bytes(buffer))
                        buffer.clear()
                    in_frame = True
                    escaped = False
                elif in_frame:
                    if escaped:
                        if byte_val == 0xDC:  # TFEND
                            buffer.append(0xC0)
                        elif byte_val == 0xDD:  # TFESC
                            buffer.append(0xDB)
                        escaped = False
                    elif byte_val == 0xDB:  # FESC
                        escaped = True
                    else:
                        buffer.append(byte_val)
                        
            except Exception as e:
                logger.error(f"Async serial receive error: {e}")
                await self._handle_error(e)
                break
                
        self._running = False

    async def _process_frame(self, frame: bytes) -> None:
        """Process received frame."""
        if not frame:
            return
            
        cmd_byte = frame[0]
        tnc_address = (cmd_byte >> 4) & 0x0F
        payload = frame[1:]
        
        async_frame = AsyncKISSFrame(payload, tnc_address, cmd_byte)
        await self._frame_queue.put(async_frame)
        
        if self.on_frame_received:
            try:
                await self.on_frame_received(payload, tnc_address)
            except Exception as e:
                logger.error(f"Frame callback failed: {e}")

    async def _handle_error(self, error: Exception) -> None:
        """Handle async serial errors."""
        self._stats['errors'] += 1
        if self.on_error:
            await self.on_error(error)

    async def start(self) -> None:
        """Start async serial interface."""
        await self.connect()
        self._running = True
        
        # Start receive task
        receive_task = asyncio.create_task(self._receive_loop())
        
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            await self.disconnect()

    async def stop(self) -> None:
        """Stop async serial interface."""
        self._running = False
        await self.disconnect()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def __repr__(self) -> str:
        return f"AsyncSerialKISSInterface(port={self.port}, baudrate={self.baudrate}, tnc={self.tnc_address})"

# Example usage and testing
async def test_async_kiss_interfaces():
    """Test async KISS interfaces."""
    logging.basicConfig(level=logging.DEBUG)
    
    # Test async TCP interface
    async def test_async_tcp():
        kiss_tcp = AsyncKISSInterface(tnc_address=1)
        
        def frame_handler(frame: bytes, tnc: int):
            print(f"Sync frame from TNC {tnc}: {frame.hex()}")
        
        async def async_frame_handler(frame: bytes, tnc: int):
            print(f"Async frame from TNC {tnc}: {frame.hex()}")
        
        kiss_tcp.register_rx_callback(frame_handler)
        kiss_tcp.register_async_frame_callback(async_frame_handler)
        
        try:
            await kiss_tcp.connect("localhost", 8001)
            await kiss_tcp.start()
            
            # Send test frames
            await kiss_tcp.send_frame(b"Hello Async TCP!")
            await kiss_tcp.send_poll(2)
            
            # Receive frames
            for i in range(5):
                frame = await kiss_tcp.recv_frame(timeout=2.0)
                if frame:
                    print(f"Received: {frame[0].hex()}")
                await asyncio.sleep(1.0)
                
            await kiss_tcp.stop()
            
        except Exception as e:
            logger.error(f"Async TCP test failed: {e}")
            await kiss_tcp.stop()
    
    # Test async serial interface
    async def test_async_serial():
        kiss_serial = AsyncSerialKISSInterface("/dev/ttyUSB0", 9600, tnc_address=1)
        
        async def frame_handler(frame: bytes, tnc: int):
            print(f"Async serial frame from TNC {tnc}: {frame.hex()}")
        
        kiss_serial.on_frame_received = frame_handler
        
        try:
            await kiss_serial.start()
            
            # Send test frame
            await kiss_serial.send_frame(b"Hello Async Serial!")
            
            # Wait for responses
            await asyncio.sleep(5.0)
            
            await kiss_serial.stop()
            
        except Exception as e:
            logger.error(f"Async serial test failed: {e}")
            await kiss_serial.stop()
    
    # Run tests
    print("Testing async TCP interface...")
    await test_async_tcp()
    
    print("\nTesting async serial interface...")
    await test_async_serial()
    
    print("\nAsync KISS interface tests completed")

if __name__ == "__main__":
    asyncio.run(test_async_kiss_interfaces())
