# pyax25_22/interfaces/agwpe.py
"""
AGWPE Interface Implementation

Implements the AGWPE TCP protocol as specified in:
- Image 1 (Header format)
- Image 4-5 (Registration and operation)

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import socket
import struct
import threading
import logging
from typing import Optional, Callable, Tuple, Dict

logger = logging.getLogger(__name__)

class AGWProtocolError(Exception):
    """Base exception for AGWPE protocol errors"""

class TransportError(Exception):
    """Base exception for transport errors"""

class AGWFrameType(IntEnum):
    """AGWPE frame types"""
    DATA = ord('D')     # Data frame (UI)
    RAW = ord('K')      # Raw AX.25 frame
    MONITOR = ord('M')  # Enable monitoring
    CONNECT = ord('C')  # Connect request
    DX = ord('d')       # Connected data
    DISCONNECT = ord('D')  # Disconnect
    REGISTER = ord('R') # Registration
    VERSION = ord('v')  # Version request
    HEARD = ord('H')    # Heard list request

AGW_HEADER_FORMAT = '<I4sBBBBBBBB8s8sI'
AGW_HEADER_SIZE = 36
DEFAULT_PORT = 8000

class AGWHeader(NamedTuple):
    """Parsed AGWPE header"""
    port: int            # Port index
    data_kind: int       # Frame type (AGWFrameType)
    pid: int             # Protocol ID
    call_from: bytes     # Source callsign (8 bytes)
    call_to: bytes       # Destination callsign (8 bytes)
    data_len: int        # Payload length

class AGWPEClient:
    """
    AGWPE TCP client implementation
    
    Args:
        host: AGWPE server host
        port: AGWPE server port (default 8000)
        callsign: Client callsign
        timeout: Socket timeout in seconds
    """
    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        callsign: str = "NOCALL",
        timeout: float = 30.0
    ):
        self.host = host
        self.port = port
        self.callsign = callsign.upper().ljust(8).encode('ascii')
        self.timeout = timeout
        
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_callbacks: Dict[int, Callable[[bytes, str, str], None]] = {}
        self._version_callback: Optional[Callable[[str], None]] = None
        
    def connect(self) -> None:
        """Connect to AGWPE server and register"""
        if self._running:
            return
            
        try:
            self._socket = socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout
            )
            self._socket.settimeout(self.timeout)
            
            # Send registration frame
            reg_frame = self._build_header(
                port=0,
                data_kind=AGWFrameType.REGISTER,
                call_from=self.callsign
            )
            self._socket.sendall(reg_frame)
            
            # Wait for registration response
            resp = self._socket.recv(AGW_HEADER_SIZE)
            if len(resp) < AGW_HEADER_SIZE or resp[4] != ord('X'):
                raise AGWProtocolError("Registration failed")
                
            self._running = True
            self._thread = threading.Thread(
                target=self._receive_loop,
                daemon=True,
                name=f"AGWPE-Rx-{self.host}:{self.port}"
            )
            self._thread.start()
            logger.info(f"Connected to AGWPE at {self.host}:{self.port}")
            
        except socket.error as e:
            raise TransportError(f"Connection failed: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from AGWPE server"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
        logger.info(f"Disconnected from {self.host}:{self.port}")

    def register_frame_callback(
        self,
        frame_type: AGWFrameType,
        callback: Callable[[bytes, str, str], None]
    ) -> None:
        """Register callback for specific frame type"""
        with self._lock:
            self._frame_callbacks[frame_type] = callback

    def register_version_callback(
        self,
        callback: Callable[[str], None]
    ) -> None:
        """Register callback for version response"""
        with self._lock:
            self._version_callback = callback

    def send_frame(
        self,
        data: bytes,
        frame_type: AGWFrameType = AGWFrameType.DATA,
        port: int = 0,
        dest: Optional[str] = None
    ) -> None:
        """
        Send frame to AGWPE server
        
        Args:
            data: Frame payload
            frame_type: AGWFrameType (default DATA)
            port: Virtual TNC port
            dest: Destination callsign (optional)
        """
        if not self._socket:
            raise TransportError("Not connected")
            
        call_to = dest.upper().ljust(8).encode('ascii') if dest else b' ' * 8
        
        header = self._build_header(
            port=port,
            data_kind=frame_type,
            call_from=self.callsign,
            call_to=call_to,
            data_len=len(data)
        )
        
        try:
            with self._lock:
                self._socket.sendall(header + data)
            logger.debug(f"Sent {frame_type.name} frame ({len(data)} bytes)")
        except OSError as e:
            raise TransportError(f"Send failed: {e}") from e

    def _build_header(
        self,
        port: int = 0,
        data_kind: int = AGWFrameType.DATA,
        call_from: bytes = b'',
        call_to: bytes = b'',
        data_len: int = 0
    ) -> bytes:
        """Construct AGWPE header (Image 1 spec)"""
        return struct.pack(
            AGW_HEADER_FORMAT,
            0,             # Cookie
            b'',           # Reserved
            port & 0xff,   # Port index
            data_kind,     # Frame type
            0, 0, 0, 0, 0, # Reserved
            call_from.ljust(8, b' ')[:8],
            call_to.ljust(8, b' ')[:8],
            data_len
        )

    def _parse_header(self, data: bytes) -> Optional[AGWHeader]:
        """Parse AGWPE header from bytes"""
        if len(data) < AGW_HEADER_SIZE:
            return None
            
        fields = struct.unpack(AGW_HEADER_FORMAT, data)
        return AGWHeader(
            port=fields[2],
            data_kind=fields[3],
            pid=fields[4],  # PID is in reserved field4
            call_from=fields[9].strip(),
            call_to=fields[10].strip(),
            data_len=fields[11]
        )

    def _receive_loop(self) -> None:
        """Main receive loop (runs in thread)"""
        buffer = b''
        while self._running and self._socket:
            try:
                # Read header
                header_data = self._socket.recv(AGW_HEADER_SIZE)
                if not header_data:
                    break  # Connection closed
                    
                header = self._parse_header(header_data)
                if not header:
                    continue
                
                # Read payload
                payload = b''
                remaining = header.data_len
                while remaining > 0:
                    chunk = self._socket.recv(min(remaining, 4096))
                    if not chunk:
                        break
                    payload += chunk
                    remaining -= len(chunk)
                
                # Dispatch
                self._handle_frame(header, payload)
                
            except socket.timeout:
                continue
            except OSError as e:
                logger.error(f"Receive error: {e}")
                break
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                break
                
        self._running = False

    def _handle_frame(self, header: AGWHeader, payload: bytes) -> None:
        """Process received frame"""
        call_from = header.call_from.decode('ascii', 'ignore').strip()
        call_to = header.call_to.decode('ascii', 'ignore').strip()
        
        with self._lock:
            if header.data_kind == AGWFrameType.VERSION and self._version_callback:
                version = payload.decode('ascii', 'ignore').strip()
                self._version_callback(version)
            elif header.data_kind in self._frame_callbacks:
                self._frame_callbacks[header.data_kind](payload, call_from, call_to)
            else:
                logger.debug(f"Unhandled frame type: {chr(header.data_kind)}")

    def __repr__(self) -> str:
        return (f"AGWPEClient(host={self.host}, port={self.port}, "
                f"callsign={self.callsign.decode().strip()}, "
                f"connected={self._running})")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    def frame_handler(data: bytes, src: str, dest: str) -> None:
        print(f"Frame from {src} to {dest}: {data.hex()}")
    
    agw = AGWPEClient("localhost", callsign="MYCALL")
    agw.connect()
    agw.register_frame_callback(AGWFrameType.DATA, frame_handler)
    
    try:
        # Send test frame
        agw.send_frame(b"Hello AGWPE!", dest="TEST")
        
        while True:
            time.sleep(1)
    finally:
        agw.disconnect()
