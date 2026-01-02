# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
KISS Interface Implementation (Multi-Drop Compatible)

Implements:
- Standard KISS framing (RFC 1055)
- Multi-drop extensions (Image 0/G8BPQ spec)
- Serial and TCP transports
- Thread-safe operation
- Hardware error handling

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import serial
import socket
import threading
import time
import logging
import struct
from typing import Optional, Callable, Union, List, Dict, Any, Tuple
from enum import IntEnum
from collections import deque
import select

logger = logging.getLogger(__name__)

class KISSCommand(IntEnum):
    """
    KISS command bytes (multi-drop compatible)
    High nibble = TNC address (0-15)
    Low nibble = command
    """
    DATA = 0x00        # Normal data frame
    TX_DELAY = 0x0A    # Set TX delay (deprecated)
    PERSIST = 0x0B     # Set persistence (deprecated)
    SLOT_TIME = 0x0C   # Set slot time (deprecated)
    TX_TAIL = 0x0D     # Set TX tail time (deprecated)
    POLL = 0x0E        # Poll command (multi-drop)
    SET_HW = 0x10      # Hardware-specific commands
    RETURN = 0xFF      # Exit KISS mode

class KISSProtocolError(Exception):
    """Base exception for KISS protocol errors"""
    pass

class TransportError(Exception):
    """Base exception for transport errors"""
    pass

class FrameError(KISSProtocolError):
    """Frame-related errors"""
    pass

class KISSFrame:
    """KISS frame with metadata"""
    
    def __init__(self, data: bytes, tnc_address: int, command: int):
        """Initialize KISS frame.
        
        Args:
            data: Frame payload
            tnc_address: TNC address (0-15)
            command: KISS command byte
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
        self.timestamp = time.time()
        self.size = len(data)
        
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

class KISSStatistics:
    """KISS interface statistics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        """Reset all statistics"""
        self.frames_sent = 0
        self.frames_received = 0
        self.escapes_sent = 0
        self.escapes_received = 0
        self.errors = 0
        self.connection_time = 0.0
        self.last_activity = 0.0
        self.bytes_sent = 0
        self.bytes_received = 0
        
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics as dictionary."""
        return {
            'frames_sent': self.frames_sent,
            'frames_received': self.frames_received,
            'escapes_sent': self.escapes_sent,
            'escapes_received': self.escapes_received,
            'errors': self.errors,
            'connection_time': self.connection_time,
            'last_activity': self.last_activity,
            'uptime': time.time() - self.connection_time if self.connection_time else 0,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received
        }

class KISSInterface:
    """
    Base KISS interface implementation
    
    Args:
        tnc_address: TNC address (0-15) for multi-drop
        poll_interval: Poll interval in seconds
    """
    FEND = 0xC0
    FESC = 0xDB
    TFEND = 0xDC
    TFESC = 0xDD

    def __init__(
        self,
        tnc_address: int = 0,
        poll_interval: float = 0.1
    ):
        """Initialize KISS interface.
        
        Args:
            tnc_address: TNC address (0-15) for multi-drop
            poll_interval: Poll interval in seconds
            
        Raises:
            ValueError: If TNC address is invalid
        """
        if not 0 <= tnc_address <= 15:
            raise ValueError("TNC address must be 0-15")
        if not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
            raise ValueError("Poll interval must be positive")
            
        self.tnc_address = tnc_address
        self.poll_interval = poll_interval
        
        # Frame processing
        self._receive_buffer = bytearray()
        self._in_frame = False
        self._escaped = False
        self._frame_queue = deque(maxlen=100)
        self._frame_buffer = bytearray()
        
        # Callbacks
        self._rx_callback: Optional[Callable[[bytes, int], None]] = None
        self._poll_callback: Optional[Callable[[int], None]] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._status_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._frame_callback: Optional[Callable[[KISSFrame], None]] = None
        
        # Statistics
        self._stats = KISSStatistics()
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._last_poll = 0.0
        
        # Connection state
        self._connected = False
        self._connection_errors = 0
        self._last_error = None
        
        logger.info(f"Initialized KISS interface: TNC={tnc_address}, poll={poll_interval}s")

    def register_rx_callback(self, callback: Callable[[bytes, int], None]) -> None:
        """Register frame receive callback (frame, TNC_address).
        
        Args:
            callback: Function to call when frame is received
        """
        if not callable(callback):
            raise TypeError("Callback must be callable")
        with self._lock:
            self._rx_callback = callback
            logger.debug("RX callback registered")

    def register_poll_callback(self, callback: Callable[[int], None]) -> None:
        """Register poll callback (polling_TNC_address).
        
        Args:
            callback: Function to call when poll is received
        """
        if not callable(callback):
            raise TypeError("Callback must be callable")
        with self._lock:
            self._poll_callback = callback
            logger.debug("Poll callback registered")

    def register_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Register error callback.
        
        Args:
            callback: Function to call when error occurs
        """
        if not callable(callback):
            raise TypeError("Callback must be callable")
        with self._lock:
            self._error_callback = callback
            logger.debug("Error callback registered")

    def register_status_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register status callback.
        
        Args:
            callback: Function to call with status updates
        """
        if not callable(callback):
            raise TypeError("Callback must be callable")
        with self._lock:
            self._status_callback = callback
            logger.debug("Status callback registered")

    def register_frame_callback(self, callback: Callable[[KISSFrame], None]) -> None:
        """Register frame callback for detailed frame information.
        
        Args:
            callback: Function to call with KISSFrame objects
        """
        if not callable(callback):
            raise TypeError("Callback must be callable")
        with self._lock:
            self._frame_callback = callback
            logger.debug("Frame callback registered")

    def _encode_command(self, cmd: int) -> int:
        """
        Pack command byte with TNC address
        
        Returns:
            byte with high nibble = tnc_address, low nibble = cmd
        """
        if not 0 <= cmd <= 255:
            raise ValueError(f"Invalid command: {cmd}")
        return (self.tnc_address << 4) | (cmd & 0x0F)

    def send_frame(
        self,
        data: bytes,
        cmd: int = KISSCommand.DATA
    ) -> None:
        """
        Send KISS frame
        
        Args:
            data: Frame payload (excluding command byte)
            cmd: KISS command (default DATA)
        
        Raises:
            KISSProtocolError: On invalid command
            TransportError: On send failure
        """
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
        if not 0 <= cmd <= 255:
            raise KISSProtocolError(f"Invalid command: {cmd}")
            
        try:
            cmd_byte = self._encode_command(cmd)
            frame = self._build_frame(cmd_byte, data)
            
            with self._lock:
                self._send_raw(frame)
                self._stats.frames_sent += 1
                self._stats.bytes_sent += len(frame)
                
            logger.debug(f"Sent frame (cmd=0x{cmd:02x}, len={len(data)}, TNC={self.tnc_address})")
            
        except Exception as e:
            logger.error(f"Send failed: {e}")
            self._stats.errors += 1
            self._handle_error(e)
            raise TransportError(f"Send failed: {e}") from e

    def send_poll(self, target_tnc: int) -> None:
        """
        Send poll command to target TNC
        
        Args:
            target_tnc: TNC address (0-15)
        """
        if not 0 <= target_tnc <= 15:
            raise ValueError("Target TNC must be 0-15")
            
        cmd_byte = (target_tnc << 4) | KISSCommand.POLL
        self.send_frame(b'', cmd=cmd_byte)
        self._last_poll = time.time()
        logger.debug(f"Sent poll to TNC {target_tnc}")

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get interface statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return self._stats.get_stats()

    def reset_stats(self) -> None:
        """Reset interface statistics."""
        with self._lock:
            self._stats.reset()
            self._stats.connection_time = time.time()
            logger.debug("Statistics reset")

    def get_status(self) -> Dict[str, Any]:
        """Get interface status.
        
        Returns:
            Dictionary with interface status
        """
        with self._lock:
            return {
                'tnc_address': self.tnc_address,
                'poll_interval': self.poll_interval,
                'connected': self._connected,
                'running': self._running,
                'last_poll': self._last_poll,
                'connection_errors': self._connection_errors,
                'last_error': str(self._last_error) if self._last_error else None,
                'buffer_size': len(self._frame_buffer),
                'queue_size': len(self._frame_queue),
                'statistics': self._stats.get_stats()
            }

    def _build_frame(self, cmd: int, data: bytes) -> bytes:
        """Construct full KISS frame with stuffing"""
        try:
            escaped = self._escape(bytes([cmd]) + data)
            frame = bytes([self.FEND]) + escaped + bytes([self.FEND])
            
            # Count escapes for statistics
            escape_count = escaped.count(self.FESC)
            with self._lock:
                self._stats.escapes_sent += escape_count
                
            return frame
            
        except Exception as e:
            logger.error(f"Frame building failed: {e}")
            raise KISSProtocolError(f"Frame building failed: {e}") from e

    def _escape(self, data: bytes) -> bytes:
        """Apply KISS byte stuffing"""
        try:
            return (
                data
                .replace(bytes([self.FESC]), bytes([self.FESC, self.TFESC]))
                .replace(bytes([self.FEND]), bytes([self.FESC, self.TFEND]))
            )
        except Exception as e:
            logger.error(f"Byte stuffing failed: {e}")
            raise KISSProtocolError(f"Byte stuffing failed: {e}") from e

    def _unescape(self, data: bytes) -> bytes:
        """Remove KISS byte stuffing"""
        try:
            return (
                data
                .replace(bytes([self.FESC, self.TFEND]), bytes([self.FEND]))
                .replace(bytes([self.FESC, self.TFESC]), bytes([self.FESC]))
            )
        except Exception as e:
            logger.error(f"Byte destuffing failed: {e}")
            raise KISSProtocolError(f"Byte destuffing failed: {e}") from e

    def _send_raw(self, data: bytes) -> None:
        """Internal raw send (implemented by transport subclass)"""
        raise NotImplementedError("Transport must implement _send_raw")

    def _read_data(self) -> bytes:
        """Read raw data from transport (implemented by subclass)"""
        raise NotImplementedError("Transport must implement _read_data")

    def _handle_error(self, error: Exception) -> None:
        """Handle transport errors.
        
        Args:
            error: Exception that occurred
        """
        with self._lock:
            self._last_error = error
            self._connection_errors += 1
            
            if self._error_callback:
                try:
                    self._error_callback(error)
                except Exception as e:
                    logger.error(f"Error callback failed: {e}")

    def start(self) -> None:
        """Start the KISS interface and receiver thread"""
        if self._running:
            logger.warning("Interface already running")
            return
            
        self._running = True
        self._thread = threading.Thread(
            target=self._receive_loop,
            name=f"KISS-Receiver-TNC{self.tnc_address}",
            daemon=True
        )
        self._thread.start()
        self._stats.connection_time = time.time()
        self._stats.last_activity = time.time()
        self._connected = True
        logger.info(f"Started KISS interface (TNC {self.tnc_address})")

    def stop(self) -> None:
        """Stop the interface and receiver thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._connected = False
        logger.info(f"Stopped KISS interface (TNC {self.tnc_address})")

    def _receive_loop(self) -> None:
        """Main receive loop (runs in thread)"""
        buffer = bytearray()
        in_frame = False
        escaped = False
        
        while self._running:
            try:
                data = self._read_data()
                if not data:
                    time.sleep(0.01)  # Prevent busy waiting
                    continue
                
                self._stats.bytes_received += len(data)
                self._stats.last_activity = time.time()
                
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
                                self._stats.errors += 1
                                self._handle_error(FrameError(f"Invalid escape byte: 0x{byte:02x}"))
                            escaped = False
                        elif byte == self.FESC:
                            escaped = True
                            self._stats.escapes_received += 1
                        else:
                            buffer.append(byte)
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self._stats.errors += 1
                self._handle_error(e)
                time.sleep(0.1)  # Prevent error loop

    def _process_frame(self, frame: bytes) -> None:
        """Handle complete, unstuffed frame"""
        if not frame:
            return
            
        try:
            cmd_byte = frame[0]
            tnc_address = (cmd_byte >> 4) & 0x0F
            cmd = cmd_byte & 0x0F
            payload = frame[1:]
            
            with self._lock:
                self._stats.frames_received += 1
                
                # Create KISSFrame object
                kiss_frame = KISSFrame(payload, tnc_address, cmd_byte)
                
                # Call frame callback if registered
                if self._frame_callback:
                    try:
                        self._frame_callback(kiss_frame)
                    except Exception as e:
                        logger.error(f"Frame callback failed: {e}")
                
                # Handle based on command type
                if cmd == KISSCommand.POLL:
                    if self._poll_callback:
                        try:
                            self._poll_callback(tnc_address)
                        except Exception as e:
                            logger.error(f"Poll callback failed: {e}")
                else:
                    if self._rx_callback:
                        try:
                            self._rx_callback(payload, tnc_address)
                        except Exception as e:
                            logger.error(f"RX callback failed: {e}")
                        
                # Update status
                if self._status_callback:
                    status = {
                        'type': 'frame_received',
                        'tnc_address': tnc_address,
                        'command': cmd,
                        'command_name': kiss_frame.get_command_name(),
                        'payload_length': len(payload),
                        'timestamp': time.time(),
                        'is_data_frame': kiss_frame.is_data_frame(),
                        'is_poll_frame': kiss_frame.is_poll_frame()
                    }
                    try:
                        self._status_callback(status)
                    except Exception as e:
                        logger.error(f"Status callback failed: {e}")
                        
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self._stats.errors += 1
            self._handle_error(e)

    def __repr__(self) -> str:
        return (f"KISSInterface(tnc={self.tnc_address}, "
                f"poll={self.poll_interval}s, "
                f"connected={self._connected}, "
                f"running={self._running})")

class SerialKISSInterface(KISSInterface):
    """Serial port KISS interface with hardware error handling"""
    
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 1.0,
        rtscts: bool = False,
        dsrdtr: bool = False,
        xonxoff: bool = False,
        bytesize: int = serial.EIGHTBITS,
        parity: str = serial.PARITY_NONE,
        stopbits: int = serial.STOPBITS_ONE,
        **kwargs
    ):
        """Initialize serial KISS interface.
        
        Args:
            port: Serial port device (e.g., '/dev/ttyUSB0' or 'COM1')
            baudrate: Serial baud rate
            timeout: Serial timeout in seconds
            rtscts: Enable RTS/CTS hardware flow control
            dsrdtr: Enable DSR/DTR hardware flow control
            xonxoff: Enable XON/XOFF software flow control
            bytesize: Serial byte size
            parity: Serial parity setting
            stopbits: Serial stop bits
            **kwargs: Additional KISS interface arguments
        """
        super().__init__(**kwargs)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.rtscts = rtscts
        self.dsrdtr = dsrdtr
        self.xonxoff = xonxoff
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        
        self._serial: Optional[serial.Serial] = None
        self._port_settings = {
            'port': port,
            'baudrate': baudrate,
            'bytesize': bytesize,
            'parity': parity,
            'stopbits': stopbits,
            'timeout': timeout,
            'rtscts': rtscts,
            'dsrdtr': dsrdtr,
            'xonxoff': xonxoff
        }
        
        # Serial-specific statistics
        self._serial_stats = {
            'port_opened': 0,
            'port_closed': 0,
            'read_errors': 0,
            'write_errors': 0,
            'buffer_overruns': 0,
            'framing_errors': 0,
            'parity_errors': 0
        }

    def _send_raw(self, data: bytes) -> None:
        """Send raw data over serial port."""
        if not self._serial or not self._serial.is_open:
            raise TransportError("Serial port not open")
            
        try:
            sent = self._serial.write(data)
            if sent != len(data):
                raise TransportError(f"Partial send ({sent}/{len(data)} bytes)")
            self._serial.flush()
            
        except serial.SerialException as e:
            logger.error(f"Serial send failed: {e}")
            self._handle_serial_error(e)
            raise TransportError(f"Serial send failed: {e}") from e
        except Exception as e:
            logger.error(f"Serial write error: {e}")
            self._serial_stats['write_errors'] += 1
            raise TransportError(f"Serial write error: {e}") from e

    def _read_data(self) -> bytes:
        """Read data from serial port."""
        if not self._serial or not self._serial.is_open:
            return b''
            
        try:
            # Use select-like behavior with timeout
            if self._serial.in_waiting > 0:
                return self._serial.read(self._serial.in_waiting)
            else:
                return b''
                
        except serial.SerialTimeoutException:
            return b''
        except serial.SerialException as e:
            logger.error(f"Serial read failed: {e}")
            self._handle_serial_error(e)
            return b''
        except Exception as e:
            logger.error(f"Serial read error: {e}")
            self._serial_stats['read_errors'] += 1
            return b''

    def _handle_serial_error(self, error: serial.SerialException) -> None:
        """Handle serial port errors.
        
        Args:
            error: Serial exception
        """
        logger.error(f"Serial port error: {error}")
        
        # Track error types
        if "overrun" in str(error).lower():
            self._serial_stats['buffer_overruns'] += 1
        elif "framing" in str(error).lower():
            self._serial_stats['framing_errors'] += 1
        elif "parity" in str(error).lower():
            self._serial_stats['parity_errors'] += 1
        
        # Try to recover
        try:
            if self._serial:
                self._serial.close()
        except:
            pass
            
        # Notify error callback
        if self._error_callback:
            self._error_callback(error)

    def start(self) -> None:
        """Open serial port before starting KISS interface."""
        try:
            # Close existing connection
            if self._serial:
                self._serial.close()
                
            # Open serial port
            self._serial = serial.Serial(**self._port_settings)
            self._serial_stats['port_opened'] += 1
            
            # Configure DTR/RTS if not using hardware flow control
            if not self.rtscts and not self.dsrdtr:
                self._serial.dtr = True
                self._serial.rts = False
                
            logger.info(f"Opened serial port {self.port} @ {self.baudrate} baud")
            super().start()
            
        except serial.SerialException as e:
            logger.error(f"Serial port open failed: {e}")
            self._serial = None
            self._serial_stats['port_opened'] -= 1
            raise TransportError(f"Serial open failed: {e}") from e
        except Exception as e:
            logger.error(f"Serial port configuration failed: {e}")
            raise TransportError(f"Serial configuration failed: {e}") from e

    def stop(self) -> None:
        """Stop KISS interface and close serial port."""
        super().stop()
        if self._serial:
            try:
                self._serial.close()
                self._serial_stats['port_closed'] += 1
                logger.info(f"Closed serial port {self.port}")
            except serial.SerialException as e:
                logger.warning(f"Error closing serial port: {e}")
            finally:
                self._serial = None

    def get_port_info(self) -> Dict[str, Any]:
        """Get serial port information.
        
        Returns:
            Dictionary of port settings and status
        """
        if not self._serial:
            return {'status': 'closed', 'port': self.port}
            
        return {
            'status': 'open' if self._serial.is_open else 'closed',
            'port': self._serial.port,
            'baudrate': self._serial.baudrate,
            'bytesize': self._serial.bytesize,
            'parity': self._serial.parity,
            'stopbits': self._serial.stopbits,
            'timeout': self._serial.timeout,
            'rtscts': self._serial.rtscts,
            'dsrdtr': self._serial.dsrdtr,
            'xonxoff': self._serial.xonxoff,
            'dtr': self._serial.dtr,
            'rts': self._serial.rts,
            'cd': self._serial.cd,
            'cts': self._serial.cts,
            'dsr': self._serial.dsr,
            'ri': self._serial.ri,
            'in_waiting': self._serial.in_waiting if self._serial.is_open else 0,
            'out_waiting': self._serial.out_waiting if self._serial.is_open else 0,
            'settings': self._port_settings,
            'statistics': self._serial_stats
        }

    def configure_port(self, **settings) -> None:
        """Reconfigure serial port settings.
        
        Args:
            **settings: Serial port configuration parameters
        """
        # Update port settings
        for key, value in settings.items():
            if key in self._port_settings:
                self._port_settings[key] = value
            else:
                logger.warning(f"Unknown serial setting: {key}")
        
        # If port is open, reopen with new settings
        if self._serial and self._serial.is_open:
            self.stop()
            self.start()

    def clear_buffers(self) -> None:
        """Clear serial port buffers."""
        if self._serial and self._serial.is_open:
            try:
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()
                logger.debug("Serial buffers cleared")
            except serial.SerialException as e:
                logger.error(f"Failed to clear buffers: {e}")

    def __repr__(self) -> str:
        status = "open" if self._serial and self._serial.is_open else "closed"
        return (f"SerialKISSInterface(port={self.port}, baudrate={self.baudrate}, "
                f"tnc={self.tnc_address}, status={status})")

class TCPKISSInterface(KISSInterface):
    """TCP KISS interface with connection management"""
    
    def __init__(
        self,
        host: str,
        port: int = 8001,
        timeout: float = 5.0,
        reconnect_interval: float = 10.0,
        max_reconnect_attempts: int = 5,
        **kwargs
    ):
        """Initialize TCP KISS interface.
        
        Args:
            host: TCP hostname/IP
            port: TCP port (default 8001)
            timeout: Socket timeout (seconds)
            reconnect_interval: Reconnect delay (seconds)
            max_reconnect_attempts: Maximum reconnection attempts
            **kwargs: Additional KISS interface arguments
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._socket: Optional[socket.socket] = None
        self._connect_time = 0.0
        self._last_activity = 0.0
        self._reconnect_count = 0
        self._reconnect_attempts = 0
        
        # TCP-specific statistics
        self._tcp_stats = {
            'connections_made': 0,
            'disconnections': 0,
            'reconnections': 0,
            'connection_failures': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'keepalive_packets': 0
        }

    def _configure_socket(self, sock: socket.socket) -> None:
        """Configure socket options"""
        try:
            # Basic socket options
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Keepalive settings (platform dependent)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            if hasattr(socket, 'TCP_KEEPINTVL'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            if hasattr(socket, 'TCP_KEEPCNT'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                
            sock.settimeout(self.timeout)
            
        except OSError as e:
            logger.warning(f"Socket configuration failed: {e}")

    def _ensure_connected(self) -> bool:
        """Establish TCP connection if needed"""
        if self._socket:
            return True
            
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) exceeded")
            return False
            
        try:
            self._socket = socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout
            )
            self._configure_socket(self._socket)
            self._connect_time = time.time()
            self._last_activity = time.time()
            self._reconnect_count = 0
            self._reconnect_attempts = 0
            self._tcp_stats['connections_made'] += 1
            
            logger.info(f"Connected to {self.host}:{self.port}")
            return True
            
        except socket.error as e:
            logger.error(f"Connection failed: {e}")
            self._tcp_stats['connection_failures'] += 1
            self._reconnect_attempts += 1
            self._socket = None
            return False

    def _send_raw(self, data: bytes) -> None:
        """Send raw data over TCP (thread-safe)"""
        if not self._socket:
            raise TransportError("Not connected")
            
        try:
            sent = self._socket.send(data)
            if sent != len(data):
                raise TransportError(f"Partial send ({sent}/{len(data)} bytes)")
            self._last_activity = time.time()
            self._tcp_stats['bytes_sent'] += sent
            
        except socket.error as e:
            logger.error(f"TCP send failed: {e}")
            self._disconnect()
            raise TransportError(f"Send failed: {e}") from e

    def _disconnect(self) -> None:
        """Close TCP connection"""
        if self._socket:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
            self._tcp_stats['disconnections'] += 1
            logger.info(f"Disconnected from {self.host}:{self.port}")

    def _read_data(self) -> bytes:
        """Read data from socket (thread-safe)"""
        if not self._socket:
            return b''
            
        try:
            data = self._socket.recv(1024)
            if not data:  # Connection closed
                self._disconnect()
                return b''
            self._last_activity = time.time()
            self._tcp_stats['bytes_received'] += len(data)
            return data
            
        except socket.timeout:
            return b''
        except OSError as e:
            logger.error(f"TCP receive error: {e}")
            self._disconnect()
            return b''

    def start(self) -> None:
        """Connect to TCP server before starting KISS interface"""
        if not self._ensure_connected():
            raise TransportError("Initial connection failed")
            
        super().start()
        logger.info(f"Started TCP KISS interface to {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop KISS interface and close TCP connection"""
        super().stop()
        self._disconnect()
        logger.info(f"Stopped TCP KISS interface to {self.host}:{self.port}")

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
                    self._reconnect_count += 1
                    self._tcp_stats['reconnections'] += 1
                    continue
                
                # Handle receives
                data = self._read_data()
                if not data:
                    # Check for inactivity timeout
                    if time.time() - self._last_activity > self.timeout * 2:
                        logger.warning("Inactivity timeout, reconnecting")
                        self._disconnect()
                    continue
                
                self._stats.bytes_received += len(data)
                
                # Process bytes
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
                            escaped = False
                        elif byte == self.FESC:
                            escaped = True
                        else:
                            buffer.append(byte)
            except Exception as e:
                logger.error(f"TCP receive loop error: {e}")
                self._disconnect()
                time.sleep(1.0)

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information.
        
        Returns:
            Dictionary of connection status
        """
        connected = self._socket is not None
        uptime = time.time() - self._connect_time if connected else 0
        
        return {
            'connected': connected,
            'host': self.host,
            'port': self.port,
            'uptime': uptime,
            'reconnect_count': self._reconnect_count,
            'reconnect_attempts': self._reconnect_attempts,
            'last_activity': self._last_activity,
            'timeout': self.timeout,
            'max_reconnect_attempts': self.max_reconnect_attempts,
            'statistics': self._tcp_stats
        }

    def send_keepalive(self) -> None:
        """Send keepalive packet to maintain connection."""
        try:
            # Send a zero-length data frame as keepalive
            self.send_frame(b'', KISSCommand.DATA)
            self._tcp_stats['keepalive_packets'] += 1
            logger.debug("Sent keepalive packet")
        except Exception as e:
            logger.warning(f"Keepalive failed: {e}")

    def __repr__(self) -> str:
        connected = "connected" if self._socket else "disconnected"
        return (f"TCPKISSInterface(host={self.host}, port={self.port}, "
                f"tnc={self.tnc_address}, {connected})")

# Example usage and testing functions
def test_kiss_frame_building():
    """Test KISS frame building and parsing."""
    kiss = KISSInterface(tnc_address=1)
    
    # Test frame building
    test_data = b"Hello KISS!"
    cmd = KISSCommand.DATA
    frame = kiss._build_frame(kiss._encode_command(cmd), test_data)
    
    # Test frame parsing
    buffer = bytearray()
    in_frame = False
    escaped = False
    
    for byte in frame:
        if byte == KISSInterface.FEND:
            if in_frame and buffer:
                print(f"Received frame: {bytes(buffer)}")
                buffer.clear()
            in_frame = True
            escaped = False
        elif in_frame:
            if escaped:
                if byte == KISSInterface.TFEND:
                    buffer.append(KISSInterface.FEND)
                elif byte == KISSInterface.TFESC:
                    buffer.append(KISSInterface.FESC)
                escaped = False
            elif byte == KISSInterface.FESC:
                escaped = True
            else:
                buffer.append(byte)
    
    print("KISS frame test completed")

if __name__ == "__main__":
    # Run basic tests
    test_kiss_frame_building()
    
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    
    def frame_handler(frame: bytes, tnc: int) -> None:
        print(f"Frame from TNC {tnc}: {frame.hex()}")
    
    # Create serial interface
    # serial_kiss = SerialKISSInterface("/dev/ttyUSB0", 9600, tnc_address=1)
    # serial_kiss.register_rx_callback(frame_handler)
    
    # Create TCP interface
    # tcp_kiss = TCPKISSInterface("localhost", 8001, tnc_address=1)
    # tcp_kiss.register_rx_callback(frame_handler)
    
    print("KISS interface examples ready")
