# pyax25_22/interfaces/kiss.py
"""
KISS Interface Implementation (Multi-Drop Compatible)

Implements:
- Standard KISS framing (RFC 1055)
- Multi-drop extensions (Image 0/G8BPQ spec)
- Serial and TCP transports
- Thread-safe operation

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import serial
import socket
import threading
import time
import logging
from typing import Optional, Callable, Union

logger = logging.getLogger(__name__)

class KISSProtocolError(Exception):
    """Base exception for KISS protocol errors"""

class TransportError(Exception):
    """Base exception for transport errors"""

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
        if not 0 <= tnc_address <= 15:
            raise ValueError("TNC address must be 0-15")
        self.tnc_address = tnc_address
        self.poll_interval = poll_interval
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._rx_callback: Optional[Callable[[bytes, int], None]] = None
        self._poll_callback: Optional[Callable[[int], None]] = None
        self._last_poll = 0.0

    def _encode_command(self, cmd: int) -> int:
        """
        Pack command byte with TNC address
        
        Returns:
            byte with high nibble = tnc_address, low nibble = cmd
        """
        return (self.tnc_address << 4) | (cmd & 0x0F)

    def register_rx_callback(self, callback: Callable[[bytes, int], None]) -> None:
        """Register frame receive callback (frame, TNC_address)"""
        with self._lock:
            self._rx_callback = callback

    def register_poll_callback(self, callback: Callable[[int], None]) -> None:
        """Register poll callback (polling_TNC_address)"""
        with self._lock:
            self._poll_callback = callback

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
        if not 0 <= cmd <= 0xFF:
            raise KISSProtocolError(f"Invalid command: {cmd}")
            
        cmd_byte = self._encode_command(cmd)
        frame = self._build_frame(cmd_byte, data)
        
        try:
            with self._lock:
                self._send_raw(frame)
            logger.debug(f"Sent frame (cmd=0x{cmd:02x}, len={len(data)})")
        except Exception as e:
            raise TransportError(f"Send failed: {e}") from e

    def send_poll(self, target_tnc: int) -> None:
        """
        Send poll command to target TNC
        
        Args:
            target_tnc: TNC address (0-15)
        """
        if not 0 <= target_tnc <= 15:
            raise ValueError("Target TNC must be 0-15")
            
        # Poll command high nibble = target TNC
        cmd_byte = (target_tnc << 4) | KISSCommand.POLL
        self.send_frame(b'', cmd=cmd_byte)

    def _build_frame(self, cmd: int, data: bytes) -> bytes:
        """Construct full KISS frame with stuffing"""
        escaped = self._escape(bytes([cmd]) + data)
        return bytes([self.FEND]) + escaped + bytes([self.FEND])

    def _escape(self, data: bytes) -> bytes:
        """Apply KISS byte stuffing"""
        return (
            data
            .replace(bytes([self.FESC]), bytes([self.FESC, self.TFESC]))
            .replace(bytes([self.FEND]), bytes([self.FESC, self.TFEND]))
        )

    def _unescape(self, data: bytes) -> bytes:
        """Remove KISS byte stuffing"""
        return (
            data
            .replace(bytes([self.FESC, self.TFEND]), bytes([self.FEND]))
            .replace(bytes([self.FESC, self.TFESC]), bytes([self.FESC]))
        )

    def _send_raw(self, data: bytes) -> None:
        """Internal raw send (implemented by transport)"""
        raise NotImplementedError

    def start(self) -> None:
        """Start receiver thread"""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(
            target=self._receive_loop,
            name="KISS-Receiver",
            daemon=True
        )
        self._thread.start()
        logger.info("KISS interface started")

    def stop(self) -> None:
        """Stop receiver thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("KISS interface stopped")

    def _receive_loop(self) -> None:
        """Main receive loop (runs in thread)"""
        buffer = bytearray()
        in_frame = False
        escaped = False
        
        while self._running:
            try:
                data = self._read_data()
                if not data:
                    continue
                
                for byte in data:
                    if byte == self.FEND:
                        if in_frame and len(buffer) > 0:
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
                                logger.warning("Invalid escape byte: 0x%02x", byte)
                            escaped = False
                        elif byte == self.FESC:
                            escaped = True
                        else:
                            buffer.append(byte)
            except Exception as e:
                logger.error("Receive error: %s", e)

    def _process_frame(self, frame: bytes) -> None:
        """Handle complete, unstuffed frame"""
        if not frame:
            return
            
        cmd_byte = frame[0]
        tnc_address = (cmd_byte >> 4) & 0x0F
        cmd = cmd_byte & 0x0F
        payload = frame[1:]
        
        try:
            if cmd == KISSCommand.POLL:
                with self._lock:
                    if self._poll_callback:
                        self._poll_callback(tnc_address)
            else:
                with self._lock:
                    if self._rx_callback:
                        self._rx_callback(payload, tnc_address)
        except Exception as e:
            logger.error("Frame handler error: %s", e)

    def _read_data(self) -> bytes:
        """Read raw data from transport (implemented by subclass)"""
        raise NotImplementedError

class SerialKISSInterface(KISSInterface):
    """Serial port KISS interface"""
    
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial: Optional[serial.Serial] = None

    def _send_raw(self, data: bytes) -> None:
        if not self._serial:
            raise TransportError("Not connected")
        self._serial.write(data)

    def _read_data(self) -> bytes:
        if not self._serial or not self._serial.is_open:
            raise TransportError("Not connected")
        return self._serial.read(self._serial.in_waiting or 1)

    def start(self) -> None:
        """Open serial port before starting"""
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            logger.info(f"Opened serial port {self.port}@{self.baudrate}")
        except serial.SerialException as e:
            raise TransportError(f"Serial open failed: {e}") from e
        super().start()

    def stop(self) -> None:
        """Close serial port after stopping"""
        super().stop()
        if self._serial:
            self._serial.close()
            self._serial = None
        logger.info("Closed serial port")

class TCPKISSInterface(KISSInterface):
    """TCP KISS interface"""
    
    def __init__(
        self,
        host: str,
        port: int = 8001,
        timeout: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None

    def _send_raw(self, data: bytes) -> None:
        if not self._socket:
            raise TransportError("Not connected")
        self._socket.sendall(data)

    def _read_data(self) -> bytes:
        if not self._socket:
            raise TransportError("Not connected")
        try:
            return self._socket.recv(1024)
        except socket.timeout:
            return b''
        except OSError as e:
            raise TransportError(f"Socket error: {e}") from e

    def start(self) -> None:
        """Connect to TCP server before starting"""
        try:
            self._socket = socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout
            )
            self._socket.settimeout(self.timeout)
            logger.info(f"Connected to {self.host}:{self.port}")
        except socket.error as e:
            raise TransportError(f"Connection failed: {e}") from e
        super().start()

    def stop(self) -> None:
        """Close socket after stopping"""
        super().stop()
        if self._socket:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        logger.info("Closed TCP connection")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    def frame_handler(frame: bytes, tnc: int) -> None:
        print(f"Frame from TNC {tnc}: {frame.hex()}")
    
    kiss = TCPKISSInterface("localhost", 8001, tnc_address=1)
    kiss.register_rx_callback(frame_handler)
    kiss.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        kiss.stop()
