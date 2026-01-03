# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
TCP KISS Interface Implementation

Implements KISS over TCP with multi-drop support and keepalive handling.

License: LGPLv3.0
Copyright (C) 2025-2026 Kris Kirby, KE4AHR
"""

import socket
import threading
import time
import logging
import select
import struct
from typing import Optional, Callable, Union, List, Dict, Any, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .kiss import (
    KISSInterface,
    KISSCommand,
    KISSProtocolError,
    TransportError,
    KISSFrame
)

logger = logging.getLogger(__name__)

class TCPState(Enum):
    """TCP connection states"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()
    ERROR = auto()

@dataclass
class TCPConnectionInfo:
    """TCP connection information"""
    host: str
    port: int
    local_address: Optional[Tuple[str, int]] = None
    remote_address: Optional[Tuple[str, int]] = None
    connected_at: Optional[float] = None
    last_activity: Optional[float] = None
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    reconnect_count: int = 0

@dataclass
class TCPKeepaliveConfig:
    """TCP keepalive configuration"""
    enabled: bool = True
    idle_time: int = 60      # Time before sending keepalive probes (seconds)
    interval: int = 10       # Interval between keepalive probes (seconds)
    count: int = 3          # Number of keepalive probes before disconnect

@dataclass
class TCPReconnectConfig:
    """TCP reconnection configuration"""
    enabled: bool = True
    interval: float = 10.0   # Base reconnection interval (seconds)
    max_interval: float = 300.0  # Maximum reconnection interval (5 minutes)
    backoff_factor: float = 2.0  # Exponential backoff factor
    max_attempts: int = 10   # Maximum reconnection attempts before giving up

class TCPKeepaliveManager:
    """Manages TCP keepalive probes"""
    
    def __init__(self, socket_obj: socket.socket, config: TCPKeepaliveConfig):
        """Initialize keepalive manager.
        
        Args:
            socket_obj: Socket to manage keepalive for
            config: Keepalive configuration
        """
        self.socket = socket_obj
        self.config = config
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._lock = threading.RLock()
        
        if self.config.enabled:
            self._setup_keepalive()
    
    def _setup_keepalive(self) -> None:
        """Configure socket keepalive options"""
        try:
            # Enable keepalive
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            # Set keepalive parameters (platform dependent)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.config.idle_time)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.config.interval)
            if hasattr(socket, 'TCP_KEEPCNT'):
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.config.count)
                
            logger.debug(f"Keepalive configured: idle={self.config.idle_time}s, "
                        f"interval={self.config.interval}s, count={self.config.count}")
                        
        except OSError as e:
            logger.warning(f"Could not configure keepalive: {e}")
    
    def start(self) -> None:
        """Start keepalive monitoring"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._schedule_probe()
    
    def stop(self) -> None:
        """Stop keepalive monitoring"""
        with self._lock:
            self._running = False
            if self._timer:
                self._timer.cancel()
                self._timer = None
    
    def _schedule_probe(self) -> None:
        """Schedule next keepalive probe"""
        if not self._running:
            return
            
        # For this implementation, we rely on OS-level keepalive
        # But we can add application-level keepalive if needed
        pass
    
    def reset(self) -> None:
        """Reset keepalive timer"""
        # Application-level keepalive reset would go here
        pass

class TCPReconnectManager:
    """Manages TCP reconnection logic with exponential backoff"""
    
    def __init__(self, config: TCPReconnectConfig):
        """Initialize reconnection manager.
        
        Args:
            config: Reconnection configuration
        """
        self.config = config
        self._current_interval = config.interval
        self._attempt_count = 0
        self._last_attempt = 0.0
        self._lock = threading.RLock()
    
    def should_attempt_reconnect(self) -> bool:
        """Check if reconnection should be attempted.
        
        Returns:
            True if reconnection should be attempted, False otherwise
        """
        with self._lock:
            if not self.config.enabled:
                return False
                
            if self._attempt_count >= self.config.max_attempts:
                return False
                
            now = time.time()
            if now - self._last_attempt < self._current_interval:
                return False
                
            return True
    
    def record_attempt(self) -> None:
        """Record a reconnection attempt"""
        with self._lock:
            self._attempt_count += 1
            self._last_attempt = time.time()
    
    def record_success(self) -> None:
        """Record a successful reconnection"""
        with self._lock:
            self._attempt_count = 0
            self._current_interval = self.config.interval
    
    def record_failure(self) -> None:
        """Record a failed reconnection attempt"""
        with self._lock:
            self._attempt_count += 1
            self._current_interval = min(
                self._current_interval * self.config.backoff_factor,
                self.config.max_interval
            )
            self._last_attempt = time.time()
    
    def get_next_interval(self) -> float:
        """Get next reconnection interval.
        
        Returns:
            Time until next reconnection attempt (seconds)
        """
        with self._lock:
            if not self.config.enabled:
                return 0.0
                
            if self._attempt_count >= self.config.max_attempts:
                return 0.0  # No more attempts
                
            now = time.time()
            elapsed = now - self._last_attempt
            remaining = max(0, self._current_interval - elapsed)
            return remaining
    
    def reset(self) -> None:
        """Reset reconnection state"""
        with self._lock:
            self._attempt_count = 0
            self._current_interval = self.config.interval
            self._last_attempt = 0.0

class TCPKISSInterface(KISSInterface):
    """
    TCP-based KISS interface with keepalive and reconnection support.
    
    Args:
        host: TCP hostname/IP
        port: TCP port (default 8001)
        tnc_address: TNC address for multi-drop (0-15)
        poll_interval: Poll interval in seconds
        timeout: Socket timeout (seconds)
        reconnect_interval: Reconnect delay (seconds)
        keepalive_config: Keepalive configuration
        reconnect_config: Reconnection configuration
    """
    
    def __init__(
        self,
        host: str,
        port: int = 8001,
        tnc_address: int = 0,
        poll_interval: float = 0.1,
        timeout: float = 5.0,
        reconnect_interval: float = 10.0,
        keepalive_config: Optional[TCPKeepaliveConfig] = None,
        reconnect_config: Optional[TCPReconnectConfig] = None,
        max_queue_size: int = 1000,
        buffer_size: int = 4096
    ):
        """Initialize TCP KISS interface.
        
        Args:
            host: TCP hostname/IP
            port: TCP port (default 8001)
            tnc_address: TNC address for multi-drop (0-15)
            poll_interval: Poll interval in seconds
            timeout: Socket timeout (seconds)
            reconnect_interval: Reconnect delay (seconds)
            keepalive_config: Keepalive configuration
            reconnect_config: Reconnection configuration
            max_queue_size: Maximum receive queue size
            buffer_size: Socket buffer size
        """
        super().__init__(tnc_address=tnc_address, poll_interval=poll_interval)
        
        self.host = host
        self.port = port
        self.timeout = timeout
        self.buffer_size = buffer_size
        self._socket: Optional[socket.socket] = None
        self._connect_time = 0.0
        self._last_activity = 0.0
        self._reconnect_count = 0
        
        # Configuration
        self.keepalive_config = keepalive_config or TCPKeepaliveConfig()
        self.reconnect_config = reconnect_config or TCPReconnectConfig(
            interval=reconnect_interval
        )
        
        # Connection management
        self._state = TCPState.DISCONNECTED
        self._connection_info = TCPConnectionInfo(
            host=host,
            port=port
        )
        
        # Managers
        self._keepalive_manager: Optional[TCPKeepaliveManager] = None
        self._reconnect_manager = TCPReconnectManager(self.reconnect_config)
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connect_lock = threading.Lock()
        self._socket_lock = threading.RLock()
        
        # Buffers
        self._receive_buffer = bytearray()
        self._send_buffer = bytearray()
        
        # Statistics
        self._tcp_stats = {
            'connections_made': 0,
            'disconnections': 0,
            'reconnections': 0,
            'connection_failures': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'keepalive_packets': 0,
            'keepalive_failures': 0,
            'select_timeouts': 0,
            'socket_errors': 0,
            'partial_sends': 0,
            'buffer_overruns': 0
        }
        
        # Async support
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TCPKISS")
        self._async_future: Optional[asyncio.Future] = None
        
        logger.info(f"Initialized TCPKISSInterface: {host}:{port}, TNC={tnc_address}")

    def _configure_socket(self, sock: socket.socket) -> None:
        """Configure socket options"""
        try:
            # Basic socket configuration
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.buffer_size)
            
            # Keepalive configuration
            if self.keepalive_config.enabled:
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.keepalive_config.idle_time)
                if hasattr(socket, 'TCP_KEEPINTVL'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.keepalive_config.interval)
                if hasattr(socket, 'TCP_KEEPCNT'):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.keepalive_config.count)
            
            # Timeout
            sock.settimeout(self.timeout)
            
            logger.debug(f"Socket configured: timeout={self.timeout}s, "
                        f"buffer_size={self.buffer_size}, "
                        f"keepalive={self.keepalive_config.enabled}")
            
        except OSError as e:
            logger.warning(f"Socket configuration failed: {e}")

    def _ensure_connected(self) -> bool:
        """Establish TCP connection if needed"""
        with self._connect_lock:
            if self._socket:
                return True
                
            if not self._reconnect_manager.should_attempt_reconnect():
                remaining = self._reconnect_manager.get_next_interval()
                logger.debug(f"Reconnection cooldown: {remaining:.1f}s remaining")
                return False
            
            try:
                self._reconnect_manager.record_attempt()
                self._connection_info.reconnect_count += 1
                
                self._socket = socket.create_connection(
                    (self.host, self.port),
                    timeout=self.timeout
                )
                self._configure_socket(self._socket)
                self._connect_time = time.time()
                self._last_activity = time.time()
                self._reconnect_count = 0
                self._tcp_stats['connections_made'] += 1
                
                # Update connection info
                self._connection_info.connected_at = self._connect_time
                try:
                    self._connection_info.local_address = self._socket.getsockname()
                    self._connection_info.remote_address = self._socket.getpeername()
                except OSError:
                    pass
                
                # Initialize keepalive manager
                self._keepalive_manager = TCPKeepaliveManager(self._socket, self.keepalive_config)
                
                self._change_state(TCPState.CONNECTED)
                self._reconnect_manager.record_success()
                
                logger.info(f"Connected to {self.host}:{self.port}")
                return True
                
            except socket.error as e:
                logger.error(f"Connection failed: {e}")
                self._tcp_stats['connection_failures'] += 1
                self._reconnect_manager.record_failure()
                self._socket = None
                return False

    def _change_state(self, new_state: TCPState) -> None:
        """Update connection state with callback."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            logger.info(f"TCP state: {old_state.name} -> {new_state.name}")
            
            # Call status callback if registered
            if self._status_callback:
                try:
                    status = {
                        'type': 'connection_state',
                        'old_state': old_state.name,
                        'new_state': new_state.name,
                        'host': self.host,
                        'port': self.port,
                        'timestamp': time.time()
                    }
                    self._status_callback(status)
                except Exception as e:
                    logger.error(f"Status callback failed: {e}")

    def _send_raw(self, data: bytes) -> None:
        """Send raw data over TCP with error handling"""
        with self._socket_lock:
            if not self._socket:
                raise TransportError("Not connected")
                
            try:
                # Use select to check if socket is writable
                ready, _, error = select.select([], [self._socket], [self._socket], 0.1)
                
                if error:
                    raise TransportError("Socket error during send")
                if not ready:
                    raise TransportError("Socket not ready for writing")
                
                total_sent = 0
                while total_sent < len(data):
                    sent = self._socket.send(data[total_sent:])
                    if sent == 0:
                        raise TransportError("Connection closed during send")
                    total_sent += sent
                    self._tcp_stats['bytes_sent'] += sent
                    self._connection_info.bytes_sent += sent
                
                self._last_activity = time.time()
                self._connection_info.last_activity = self._last_activity
                
            except socket.error as e:
                logger.error(f"TCP send failed: {e}")
                self._handle_socket_error(e)
                raise TransportError(f"Send failed: {e}") from e
            except Exception as e:
                logger.error(f"Send error: {e}")
                self._tcp_stats['socket_errors'] += 1
                raise TransportError(f"Send error: {e}") from e

    def _handle_socket_error(self, error: socket.error) -> None:
        """Handle socket errors and update statistics"""
        self._tcp_stats['socket_errors'] += 1
        self._connection_info.errors += 1
        
        # Disconnect on any socket error
        self._disconnect()
        
        # Update error callback
        if self._error_callback:
            try:
                self._error_callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def _disconnect(self) -> None:
        """Close TCP connection"""
        with self._socket_lock:
            if self._socket:
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._socket.close()
                self._socket = None
            
            if self._keepalive_manager:
                self._keepalive_manager.stop()
                self._keepalive_manager = None
            
            self._tcp_stats['disconnections'] += 1
            self._connection_info.last_activity = time.time()
            
            self._change_state(TCPState.DISCONNECTED)
            logger.info(f"Disconnected from {self.host}:{self.port}")

    def _read_data(self) -> bytes:
        """Read data from socket with select-based timeout"""
        with self._socket_lock:
            if not self._socket:
                return b''
                
            try:
                # Use select to check if socket is readable
                ready, _, error = select.select([self._socket], [], [self._socket], 0.01)
                
                if error:
                    raise TransportError("Socket error during read")
                if not ready:
                    return b''  # No data available
                
                data = self._socket.recv(self.buffer_size)
                if not data:  # Connection closed
                    self._disconnect()
                    return b''
                
                self._last_activity = time.time()
                self._connection_info.last_activity = self._last_activity
                self._tcp_stats['bytes_received'] += len(data)
                self._connection_info.bytes_received += len(data)
                
                return data
                
            except socket.timeout:
                self._tcp_stats['select_timeouts'] += 1
                return b''
            except socket.error as e:
                logger.error(f"TCP receive error: {e}")
                self._handle_socket_error(e)
                return b''
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self._tcp_stats['socket_errors'] += 1
                return b''

    def start(self) -> None:
        """Start the KISS interface and receiver thread"""
        if self._running:
            logger.warning("Interface already running")
            return
            
        try:
            if not self._ensure_connected():
                raise TransportError("Initial connection failed")
                
            self._running = True
            self._thread = threading.Thread(
                target=self._receive_loop,
                name=f"TCPKISS-{self.host}:{self.port}",
                daemon=True
            )
            self._thread.start()
            
            if self.keepalive_config.enabled:
                if self._keepalive_manager:
                    self._keepalive_manager.start()
            
            logger.info(f"Started TCP KISS interface to {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start TCP KISS interface: {e}")
            self._disconnect()
            raise

    def stop(self) -> None:
        """Stop the interface and receiver thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        self._disconnect()
        logger.info(f"Stopped TCP KISS interface to {self.host}:{self.port}")

    def _receive_loop(self) -> None:
        """Main receive loop with reconnection handling"""
        buffer = bytearray()
        in_frame = False
        escaped = False
        
        while self._running:
            try:
                # Reconnect if needed
                if not self._ensure_connected():
                    time.sleep(1.0)  # Wait before retrying
                    continue
                
                # Handle receives
                data = self._read_data()
                if not data:
                    # Check for inactivity timeout
                    if self._last_activity > 0 and time.time() - self._last_activity > self.timeout * 2:
                        logger.warning("Inactivity timeout, reconnecting")
                        self._disconnect()
                    continue
                
                # Process bytes using parent class method
                for byte in data:
                    if byte == self.FEND:
                        if in_frame and buffer:
                            self._process_frame(bytes(buffer))
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
                                self._tcp_stats['socket_errors'] += 1
                            escaped = False
                        elif byte == self.FESC:
                            escaped = True
                        else:
                            buffer.append(byte)
                            
            except Exception as e:
                logger.error(f"TCP receive loop error: {e}")
                self._disconnect()
                time.sleep(1.0)

    def send_keepalive(self) -> None:
        """Send keepalive packet to maintain connection."""
        try:
            # Send a zero-length data frame as keepalive
            self.send_frame(b'', KISSCommand.DATA)
            self._tcp_stats['keepalive_packets'] += 1
            logger.debug("Sent keepalive packet")
        except Exception as e:
            logger.warning(f"Keepalive failed: {e}")
            self._tcp_stats['keepalive_failures'] += 1

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information.
        
        Returns:
            Dictionary of connection status
        """
        with self._socket_lock:
            connected = self._socket is not None
            uptime = time.time() - self._connect_time if connected and self._connect_time > 0 else 0
            
            return {
                'connected': connected,
                'host': self.host,
                'port': self.port,
                'uptime': uptime,
                'reconnect_count': self._reconnect_count,
                'last_activity': self._last_activity,
                'timeout': self.timeout,
                'buffer_size': self.buffer_size,
                'connection_info': {
                    'local_address': self._connection_info.local_address,
                    'remote_address': self._connection_info.remote_address,
                    'connected_at': self._connection_info.connected_at,
                    'last_activity': self._connection_info.last_activity,
                    'bytes_sent': self._connection_info.bytes_sent,
                    'bytes_received': self._connection_info.bytes_received,
                    'errors': self._connection_info.errors,
                    'reconnect_count': self._connection_info.reconnect_count
                },
                'tcp_statistics': self._tcp_stats.copy(),
                'keepalive_config': {
                    'enabled': self.keepalive_config.enabled,
                    'idle_time': self.keepalive_config.idle_time,
                    'interval': self.keepalive_config.interval,
                    'count': self.keepalive_config.count
                },
                'reconnect_config': {
                    'enabled': self.reconnect_config.enabled,
                    'interval': self.reconnect_config.interval,
                    'max_interval': self.reconnect_config.max_interval,
                    'backoff_factor': self.reconnect_config.backoff_factor,
                    'max_attempts': self.reconnect_config.max_attempts,
                    'next_interval': self._reconnect_manager.get_next_interval()
                }
            }

    def get_state(self) -> TCPState:
        """Get current connection state.
        
        Returns:
            Current TCP state
        """
        return self._state

    def is_connected(self) -> bool:
        """Check if connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self._state == TCPState.CONNECTED

    def force_reconnect(self) -> None:
        """Force a reconnection."""
        logger.info("Forcing reconnection")
        self._reconnect_manager.reset()
        self._disconnect()
        # Connection will be re-established by receive loop

    def set_keepalive_config(self, config: TCPKeepaliveConfig) -> None:
        """Update keepalive configuration.
        
        Args:
            config: New keepalive configuration
        """
        self.keepalive_config = config
        if self._keepalive_manager:
            self._keepalive_manager.config = config
            self._keepalive_manager._setup_keepalive()
        logger.info(f"Keepalive config updated: {config}")

    def set_reconnect_config(self, config: TCPReconnectConfig) -> None:
        """Update reconnection configuration.
        
        Args:
            config: New reconnection configuration
        """
        self.reconnect_config = config
        self._reconnect_manager.config = config
        logger.info(f"Reconnect config updated: {config}")

    def clear_statistics(self) -> None:
        """Clear TCP-specific statistics."""
        with self._socket_lock:
            self._tcp_stats = {k: 0 for k in self._tcp_stats}
            self._connection_info.bytes_sent = 0
            self._connection_info.bytes_received = 0
            self._connection_info.errors = 0
            self._connection_info.reconnect_count = 0
        logger.debug("TCP statistics cleared")

    def __repr__(self) -> str:
        connected = "connected" if self._socket else "disconnected"
        return (f"TCPKISSInterface(host={self.host}, port={self.port}, "
                f"tnc={self.tnc_address}, {connected}, state={self._state.name})")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

class AsyncTCPKISSInterface:
    """
    Asynchronous TCP KISS interface using asyncio.
    
    Provides non-blocking I/O operations for high-performance applications.
    """
    
    def __init__(
        self,
        host: str,
        port: int = 8001,
        tnc_address: int = 0,
        timeout: float = 5.0,
        buffer_size: int = 4096,
        **kwargs
    ):
        """Initialize async TCP KISS interface.
        
        Args:
            host: TCP hostname/IP
            port: TCP port (default 8001)
            tnc_address: TNC address for multi-drop (0-15)
            timeout: Socket timeout (seconds)
            buffer_size: Socket buffer size
            **kwargs: Additional configuration
        """
        self.host = host
        self.port = port
        self.tnc_address = tnc_address
        self.timeout = timeout
        self.buffer_size = buffer_size
        
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_frame_received: Optional[Callable[[bytes, int], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_status_change: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Statistics
        self._stats = {
            'connections_made': 0,
            'disconnections': 0,
            'reconnections': 0,
            'errors': 0,
            'bytes_sent': 0,
            'bytes_received': 0
        }
        
        logger.info(f"Initialized AsyncTCPKISSInterface: {host}:{port}, TNC={tnc_address}")

    async def connect(self) -> None:
        """Connect to TCP server asynchronously."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout
            )
            
            self._stats['connections_made'] += 1
            logger.info(f"Async connected to {self.host}:{self.port}")
            
        except asyncio.TimeoutError:
            raise TransportError(f"Connection timeout to {self.host}:{self.port}")
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Async connection failed: {e}")
            raise TransportError(f"Connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from TCP server asynchronously."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
                self._stats['disconnections'] += 1
                logger.info(f"Async disconnected from {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Async disconnect error: {e}")
        
        self._reader = None
        self._writer = None

    async def send_frame(self, data: bytes, cmd: int = KISSCommand.DATA) -> None:
        """Send frame asynchronously.
        
        Args:
            data: Frame payload
            cmd: KISS command
        """
        if not self._writer:
            raise TransportError("Not connected")
            
        try:
            # Build KISS frame
            cmd_byte = (self.tnc_address << 4) | (cmd & 0x0F)
            frame = self._build_frame(cmd_byte, data)
            
            self._writer.write(frame)
            await self._writer.drain()
            
            self._stats['bytes_sent'] += len(frame)
            logger.debug(f"Async sent frame: {len(data)} bytes")
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Async send failed: {e}")
            await self.disconnect()
            raise TransportError(f"Send failed: {e}") from e

    async def recv_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, int]]:
        """Receive frame asynchronously.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Tuple of (frame_data, tnc_address) or None
        """
        if not self._reader:
            return None
            
        try:
            if timeout:
                data = await asyncio.wait_for(
                    self._reader.read(self.buffer_size),
                    timeout=timeout
                )
            else:
                data = await self._reader.read(self.buffer_size)
            
            if not data:
                await self.disconnect()
                return None
                
            self._stats['bytes_received'] += len(data)
            
            # Process frame (simplified - would need full parsing)
            # This is a basic implementation
            if len(data) >= 3 and data[0] == KISSInterface.FEND and data[-1] == KISSInterface.FEND:
                # Basic frame validation
                cmd_byte = data[1]  # Simplified - actual parsing needed
                tnc_address = (cmd_byte >> 4) & 0x0F
                payload = data[2:-1]  # Simplified extraction
                return payload, tnc_address
            
            return None
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Async receive failed: {e}")
            await self.disconnect()
            return None

    def _build_frame(self, cmd: int, data: bytes) -> bytes:
        """Build KISS frame with byte stuffing."""
        FEND = KISSInterface.FEND
        FESC = KISSInterface.FESC
        TFEND = KISSInterface.TFEND
        TFESC = KISSInterface.TFESC
        
        escaped = (
            bytes([cmd]) + data
            .replace(bytes([FESC]), bytes([FESC, TFESC]))
            .replace(bytes([FEND]), bytes([FESC, TFEND]))
        )
        return bytes([FEND]) + escaped + bytes([FEND])

    async def _receive_loop(self) -> None:
        """Main receive loop for async interface."""
        buffer = bytearray()
        in_frame = False
        escaped = False
        
        while self._running and self._reader:
            try:
                data = await self._reader.read(1024)
                if not data:
                    break  # Connection closed
                    
                for byte in data:
                    if byte == KISSInterface.FEND:
                        if in_frame and buffer:
                            await self._process_frame(bytes(buffer))
                            buffer.clear()
                        in_frame = True
                        escaped = False
                    elif in_frame:
                        if escaped:
                            if byte == KISSInterface.TFEND:
                                buffer.append(KISSInterface.FEND)
                            elif byte == KISSInterface.TFESC:
                                buffer.append(KISSInterface.FESC)
                            else:
                                logger.warning(f"Invalid escape byte: 0x{byte:02x}")
                            escaped = False
                        elif byte == KISSInterface.FESC:
                            escaped = True
                        else:
                            buffer.append(byte)
            except Exception as e:
                logger.error(f"Async receive loop error: {e}")
                break
                
        self._running = False

    async def _process_frame(self, frame: bytes) -> None:
        """Process received frame."""
        if not frame:
            return
            
        cmd_byte = frame[0]
        tnc_address = (cmd_byte >> 4) & 0x0F
        cmd = cmd_byte & 0x0F
        payload = frame[1:]
        
        if cmd == KISSCommand.DATA and self.on_frame_received:
            try:
                self.on_frame_received(payload, tnc_address)
            except Exception as e:
                logger.error(f"Frame callback failed: {e}")

    async def start(self) -> None:
        """Start async interface."""
        await self.connect()
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info("Async TCP KISS interface started")

    async def stop(self) -> None:
        """Stop async interface."""
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        await self.disconnect()
        logger.info("Async TCP KISS interface stopped")

    def get_stats(self) -> Dict[str, int]:
        """Get async interface statistics."""
        return self._stats.copy()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def __repr__(self) -> str:
        connected = "connected" if self._writer else "disconnected"
        return (f"AsyncTCPKISSInterface(host={self.host}, port={self.port}, "
                f"tnc={self.tnc_address}, {connected})")

# Example usage and testing
async def test_async_tcp_kiss():
    """Test async TCP KISS interface."""
    try:
        async_kiss = AsyncTCPKISSInterface("localhost", 8001, tnc_address=1)
        
        def frame_handler(frame: bytes, tnc: int):
            print(f"Async frame from TNC {tnc}: {frame.hex()}")
        
        async_kiss.on_frame_received = frame_handler
        
        await async_kiss.start()
        
        # Send test frame
        await async_kiss.send_frame(b"Hello Async KISS!")
        
        # Wait for responses
        for _ in range(10):
            frame = await async_kiss.recv_frame(timeout=1.0)
            if frame:
                print(f"Received: {frame[0].hex()}")
            await asyncio.sleep(1.0)
            
        await async_kiss.stop()
        
    except Exception as e:
        logger.error(f"Async test failed: {e}")

if __name__ == "__main__":
    # Test basic TCP KISS functionality
    logging.basicConfig(level=logging.DEBUG)
    
    def frame_handler(frame: bytes, tnc: int) -> None:
        print(f"Frame from TNC {tnc}: {frame.hex()}")
    
    def status_handler(status: Dict[str, Any]) -> None:
        print(f"Status: {status}")
    
    # Create TCP interface
    kiss = TCPKISSInterface("localhost", 8001, tnc_address=1)
    kiss.register_rx_callback(frame_handler)
    kiss.register_status_callback(status_handler)
    
    try:
        kiss.start()
        
        # Test operations
        kiss.send_frame(b"Hello TCP KISS!")
        kiss.send_poll(2)
        
        # Monitor for 30 seconds
        import time
        time.sleep(30)
        
        kiss.stop()
        
    except KeyboardInterrupt:
        kiss.stop()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        kiss.stop()
    
    print("TCP KISS interface test completed")

