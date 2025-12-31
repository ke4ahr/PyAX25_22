# pyax25_22/interfaces/kiss_tcp.py
"""
TCP KISS Interface Implementation

Implements KISS over TCP with multi-drop support and keepalive handling.

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import socket
import threading
import time
import logging
from typing import Optional, Callable

from .kiss import (
    KISSInterface,
    KISSCommand,
    KISSProtocolError,
    TransportError
)

logger = logging.getLogger(__name__)

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
    """
    def __init__(
        self,
        host: str,
        port: int = 8001,
        tnc_address: int = 0,
        poll_interval: float = 0.1,
        timeout: float = 5.0,
        reconnect_interval: float = 10.0
    ):
        super().__init__(tnc_address=tnc_address, poll_interval=poll_interval)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reconnect_interval = reconnect_interval
        
        self._socket: Optional[socket.socket] = None
        self._socket_lock = threading.Lock()
        self._connect_time = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _configure_socket(self, sock: socket.socket) -> None:
        """Configure socket options"""
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, 'TCP_KEEPIDLE'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        if hasattr(socket, 'TCP_KEEPINTVL'):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        sock.settimeout(self.timeout)

    def _ensure_connected(self) -> bool:
        """Establish TCP connection if needed"""
        with self._socket_lock:
            if self._socket:
                return True
                
            try:
                self._socket = socket.create_connection(
                    (self.host, self.port),
                    timeout=self.timeout
                )
                self._configure_socket(self._socket)
                self._connect_time = time.time()
                logger.info(f"Connected to {self.host}:{self.port}")
                return True
            except socket.error as e:
                self._socket = None
                logger.error(f"Connection failed: {e}")
                return False

    def _send_raw(self, data: bytes) -> None:
        """Send raw data over TCP (thread-safe)"""
        with self._socket_lock:
            if not self._socket:
                raise TransportError("Not connected")
                
            try:
                sent = self._socket.send(data)
                if sent != len(data):
                    raise TransportError(f"Partial send ({sent}/{len(data)} bytes)")
                logger.debug(f"Sent {len(data)} bytes to {self.host}:{self.port}")
            except socket.error as e:
                self._disconnect()
                raise TransportError(f"Send failed: {e}") from e

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
                logger.info(f"Disconnected from {self.host}:{self.port}")

    def _read_data(self) -> bytes:
        """Read data from socket (thread-safe)"""
        with self._socket_lock:
            if not self._socket:
                return b''
                
            try:
                data = self._socket.recv(1024)
                if not data:  # Connection closed
                    self._disconnect()
                    return b''
                return data
            except socket.timeout:
                return b''
            except OSError as e:
                self._disconnect()
                raise TransportError(f"Receive error: {e}") from e

    def start(self) -> None:
        """Start the KISS interface and receiver thread"""
        if self._running:
            return
            
        if not self._ensure_connected():
            raise TransportError("Initial connection failed")
            
        self._running = True
        self._thread = threading.Thread(
            target=self._reconnect_loop,
            name=f"KISS-TCP-{self.host}:{self.port}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"Started KISS TCP interface to {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the interface and receiver thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._disconnect()
        logger.info(f"Stopped KISS TCP interface to {self.host}:{self.port}")

    def _reconnect_loop(self) -> None:
        """Maintain connection and handle receive"""
        buffer = bytearray()
        in_frame = False
        escaped = False
        
        while self._running:
            try:
                # Reconnect if needed
                if not self._ensure_connected():
                    time.sleep(self.reconnect_interval)
                    continue
                
                # Handle receives
                data = self._read_data()
                if not data:
                    if time.time() - self._connect_time > self.reconnect_interval:
                        self._disconnect()
                    continue
                
                # Process bytes
                for byte in data:
                    if byte == self.FEND:
                        if in_frame:
                            if buffer:
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
                            escaped = False
                        elif byte == self.FESC:
                            escaped = True
                        else:
                            buffer.append(byte)
            except TransportError as e:
                logger.error(f"Transport error: {e}")
                self._disconnect()
            except Exception as e:
                logger.exception(f"Unexpected error in receive loop: {e}")

    def __repr__(self) -> str:
        return (f"TCPKISSInterface(host={self.host}, port={self.port}, "
                f"tnc={self.tnc_address}, connected={self._socket is not None})")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    def frame_handler(frame: bytes, tnc: int) -> None:
        print(f"Received frame from TNC {tnc}: {frame.hex()}")
    
    kiss = TCPKISSInterface("localhost", 8001, tnc_address=1)
    kiss.register_rx_callback(frame_handler)
    kiss.start()
    
    try:
        while True:
            # Send poll every 5 seconds
            if time.time() - kiss._last_poll > 5:
                kiss.send_poll(2)  # Poll TNC 2
            time.sleep(0.1)
    except KeyboardInterrupt:
        kiss.stop()
