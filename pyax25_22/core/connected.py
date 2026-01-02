# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
Connected Mode Handler

Implements:
- Connection establishment/teardown
- Frame handling and processing
- Sequence number management
- Error recovery and retransmission
- Integration with state machine

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import logging
import asyncio
import threading
import time
from enum import Enum, auto
from typing import Optional, Callable, List, Dict, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ConnectionError(Exception):
    """Base exception for connection errors"""

class ConnectionTimeout(ConnectionError):
    """Connection timeout error"""

class FrameValidationError(ConnectionError):
    """Frame validation error"""

class SequenceError(ConnectionError):
    """Sequence number error"""

class WindowError(ConnectionError):
    """Window management error"""

class AX25State(Enum):
    DISCONNECTED = auto()
    AWAITING_CONNECTION = auto()
    CONNECTED = auto()
    AWAITING_RELEASE = auto()
    TIMEOUT = auto()

@dataclass
class ConnectionConfig:
    """Connection configuration parameters"""
    t1_timeout: float = 10.0      # Frame retransmission timeout
    t3_timeout: float = 30.0      # Inactivity timeout
    n2_retries: int = 3           # Maximum retransmission attempts
    k_window: int = 4             # Window size
    max_frame_size: int = 256     # Maximum frame payload size
    poll_timeout: float = 5.0     # Poll timeout
    receive_buffer_size: int = 1000  # Receive buffer size
    send_buffer_size: int = 1000     # Send buffer size

class ConnectedModeHandler:
    """
    High-level connected mode handler that manages connections
    and integrates with the AX.25 state machine.
    """
    
    def __init__(
        self,
        my_call: str,
        send_frame_fn: Callable[[bytes], None],
        frame_callback: Optional[Callable[[bytes], None]] = None,
        config: Optional[ConnectionConfig] = None
    ):
        """Initialize connected mode handler.
        
        Args:
            my_call: Local callsign
            send_frame_fn: Function to send frames to transport
            frame_callback: Callback for received frames
            config: Connection configuration
        """
        if not isinstance(my_call, str) or not my_call.strip():
            raise ValueError("Callsign must be a non-empty string")
        if not callable(send_frame_fn):
            raise TypeError("send_frame_fn must be callable")
            
        self.my_call = my_call.strip().upper()
        self.send_frame_fn = send_frame_fn
        self.frame_callback = frame_callback
        self.config = config or ConnectionConfig()
        
        # Connection state
        self.state = AX25State.DISCONNECTED
        self.dest_call: Optional[str] = None
        
        # Sequence numbers
        self.vs = 0      # Send sequence number
        self.vr = 0      # Receive sequence number
        self.va = 0      # Acknowledged sequence number
        
        # Frame management
        self.send_queue: List[Dict[str, Any]] = []
        self.received_frames: Dict[int, bytes] = {}
        self.out_of_order_frames: Dict[int, bytes] = {}
        
        # Buffer management
        self.receive_buffer = bytearray()
        self.send_buffer = bytearray()
        
        # Timers
        self.t1_timer: Optional[threading.Timer] = None
        self.t3_timer: Optional[threading.Timer] = None
        self._timer_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'frames_sent': 0,
            'frames_received': 0,
            'retransmissions': 0,
            'errors': 0,
            'connect_attempts': 0,
            'disconnects': 0,
            'sequence_errors': 0,
            'window_violations': 0,
            'acknowledgments_sent': 0,
            'acknowledgments_received': 0,
            'out_of_order_frames': 0,
            'frames_delivered': 0,
            'buffer_overflows': 0,
            'connection_time': 0.0,
            'last_activity': 0.0
        }
        
        # Locking
        self._state_lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._recv_lock = threading.RLock()
        
        # Event handlers
        self.on_state_change: Optional[Callable[[AX25State], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_frame_delivered: Optional[Callable[[bytes], None]] = None
        self.on_connection_established: Optional[Callable[[], None]] = None
        self.on_connection_terminated: Optional[Callable[[], None]] = None
        
        # Connection flags
        self._poll_pending = False
        self._busy = False
        self._retransmission_pending = False
        
        logger.info(f"Initialized ConnectedModeHandler for {self.my_call}")

    def connect(self, dest_call: str) -> None:
        """Initiate connection to destination.
        
        Args:
            dest_call: Destination callsign
            
        Raises:
            ConnectionError: If connection fails
        """
        if not isinstance(dest_call, str) or not dest_call.strip():
            raise ValueError("Destination callsign must be non-empty string")
            
        with self._state_lock:
            if self.state != AX25State.DISCONNECTED:
                raise ConnectionError(f"Already connected or connecting (state: {self.state.name})")
            
            self.dest_call = dest_call.strip().upper()
            self.stats['connect_attempts'] += 1
            self.stats['connection_time'] = time.time()
            
            try:
                self._change_state(AX25State.AWAITING_CONNECTION)
                self._send_sabm()
                self._start_t1_timer()
                
                logger.info(f"Connection attempt to {self.dest_call}")
                
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self._change_state(AX25State.DISCONNECTED)
                raise ConnectionError(f"Connection to {dest_call} failed: {e}") from e

    def disconnect(self) -> None:
        """Initiate disconnect from current connection."""
        with self._state_lock:
            if self.state not in [AX25State.CONNECTED, AX25State.AWAITING_CONNECTION]:
                logger.warning(f"Cannot disconnect from state {self.state.name}")
                return
                
            try:
                self._change_state(AX25State.AWAITING_RELEASE)
                self._send_disc()
                self._start_t1_timer()
                
                logger.info("Disconnect initiated")
                
            except Exception as e:
                logger.error(f"Disconnect failed: {e}")
                # Force disconnect on error
                self._change_state(AX25State.DISCONNECTED)
                if self.on_error:
                    self.on_error(e)

    def send_data(self, data: bytes, timeout: Optional[float] = None) -> bool:
        """Send data over the connection.
        
        Args:
            data: Data to send
            timeout: Optional timeout for sending
            
        Returns:
            True if sent successfully, False otherwise
            
        Raises:
            ConnectionError: If not connected or send fails
        """
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
        if self.state != AX25State.CONNECTED:
            raise ConnectionError(f"Cannot send data in state {self.state.name}")
        if len(data) > self.config.max_frame_size:
            raise ValueError(f"Data too large ({len(data)} > {self.config.max_frame_size})")
            
        with self._send_lock:
            try:
                if not self._can_send():
                    raise ConnectionError("Send window full")
                
                if self._busy:
                    raise ConnectionError("Remote station busy")
                
                # Build and send I-frame
                ns = self.vs
                nr = self.vr
                frame = self._build_i_frame(data, ns, nr)
                
                self._send_frame(frame)
                
                # Queue for retransmission
                self._queue_frame_for_retransmission(frame, ns)
                
                self.vs = (self.vs + 1) % 8  # Modulo 8 for now
                self._reset_t3_timer()
                self.stats['frames_sent'] += 1
                
                logger.debug(f"Sent I-frame: NS={ns}, size={len(data)}")
                return True
                
            except Exception as e:
                logger.error(f"Send failed: {e}")
                self.stats['errors'] += 1
                if self.on_error:
                    self.on_error(e)
                return False

    def receive_frame(self, frame: bytes) -> None:
        """Process received frame.
        
        Args:
            frame: Raw frame bytes
        """
        try:
            # Validate frame
            self._validate_frame(frame)
            
            # Parse frame
            parsed_frame = self._parse_frame(frame)
            
            # Reset inactivity timer
            self._reset_t3_timer()
            self.stats['last_activity'] = time.time()
            
            # Process based on frame type
            if parsed_frame['type'] == 'I':
                self._handle_i_frame(parsed_frame)
            elif parsed_frame['type'] == 'RR':
                self._handle_rr_frame(parsed_frame)
            elif parsed_frame['type'] == 'RNR':
                self._handle_rnr_frame(parsed_frame)
            elif parsed_frame['type'] == 'REJ':
                self._handle_rej_frame(parsed_frame)
            elif parsed_frame['type'] == 'SREJ':
                self._handle_srej_frame(parsed_frame)
            elif parsed_frame['type'] == 'UA':
                self._handle_ua_frame(parsed_frame)
            elif parsed_frame['type'] == 'DM':
                self._handle_dm_frame(parsed_frame)
            elif parsed_frame['type'] == 'SABM':
                self._handle_sabm_frame(parsed_frame)
            elif parsed_frame['type'] == 'DISC':
                self._handle_disc_frame(parsed_frame)
            else:
                logger.warning(f"Unknown frame type: {parsed_frame['type']}")
                
        except FrameValidationError as e:
            logger.error(f"Frame validation failed: {e}")
            self.stats['errors'] += 1
            if self.on_error:
                self.on_error(e)
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            self.stats['errors'] += 1
            if self.on_error:
                self.on_error(e)

    def _validate_frame(self, frame: bytes) -> None:
        """Validate received frame.
        
        Args:
            frame: Raw frame bytes
            
        Raises:
            FrameValidationError: If frame is invalid
        """
        if not isinstance(frame, bytes) or len(frame) < 16:
            raise FrameValidationError("Invalid frame format or size")
        
        # Basic checksum/FCS validation would go here
        # For now, we assume the transport layer handles this
        
        logger.debug("Frame validation passed")

    def _parse_frame(self, frame: bytes) -> Dict[str, Any]:
        """Parse frame into components.
        
        Args:
            frame: Raw frame bytes
            
        Returns:
            Parsed frame data
            
        Raises:
            FrameValidationError: If parsing fails
        """
        try:
            from .framing import AX25Frame
            
            ax25_frame = AX25Frame.from_bytes(frame)
            
            parsed = {
                'type': ax25_frame.type.name,
                'ns': getattr(ax25_frame, 'ns', 0),
                'nr': getattr(ax25_frame, 'nr', 0),
                'poll': getattr(ax25_frame, 'poll', False),
                'info': getattr(ax25_frame, 'info', b''),
                'pid': getattr(ax25_frame, 'pid', None)
            }
            
            logger.debug(f"Parsed frame: {parsed['type']}, NS={parsed['ns']}, NR={parsed['nr']}")
            return parsed
            
        except Exception as e:
            raise FrameValidationError(f"Frame parsing failed: {e}") from e

    def _handle_i_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Information frame.
        
        Args:
            frame_data: Parsed frame data
        """
        with self._recv_lock:
            ns = frame_data['ns']
            nr = frame_data['nr']
            info = frame_data['info']
            
            logger.debug(f"Handling I-frame: NS={ns}, NR={nr}")
            
            # Update send window based on NR
            self._update_send_window(nr)
            
            # Check sequence number validity
            if not self._is_valid_receive_sequence(ns):
                logger.warning(f"Invalid receive sequence: NS={ns}, VR={self.vr}")
                self.stats['sequence_errors'] += 1
                if not self._busy:
                    self._send_rej(self.vr)
                return
            
            # Check receive window
            if self._is_in_receive_window(ns):
                if ns == self.vr:
                    # In-order frame
                    self._deliver_frame(info)
                    self.vr = (self.vr + 1) % 8
                    
                    # Send acknowledgment
                    self._send_rr()
                else:
                    # Out-of-order frame
                    self.out_of_order_frames[ns] = info
                    self.stats['out_of_order_frames'] += 1
                    self._send_rr()  # Send ACK for last in-order frame
            else:
                # Out of window
                logger.warning(f"I-frame out of receive window: NS={ns}, VR={self.vr}")
                self.stats['window_violations'] += 1
                if not self._busy:
                    self._send_rej(self.vr)

    def _handle_rr_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Receive Ready frame.
        
        Args:
            frame_data: Parsed frame data
        """
        nr = frame_data['nr']
        
        with self._send_lock:
            logger.debug(f"Handling RR frame: NR={nr}")
            
            # Update send window
            self._update_send_window(nr)
            
            # Clean up out-of-order frames that are now acknowledged
            self._cleanup_acked_out_of_order(nr)
            
            # Clear retransmission pending flag
            self._retransmission_pending = False
            
            self.stats['acknowledgments_received'] += 1

    def _handle_rnr_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Receive Not Ready frame.
        
        Args:
            frame_data: Parsed frame data
        """
        nr = frame_data['nr']
        
        with self._send_lock:
            logger.warning(f"Handling RNR frame: NR={nr}")
            
            # Update send window
            self._update_send_window(nr)
            
            # Set busy flag - stop sending
            self._busy = True
            
            # Stop T1 timer since we're not expecting ACKs
            self._stop_t1_timer()
            
            self.stats['acknowledgments_received'] += 1

    def _handle_rej_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Reject frame.
        
        Args:
            frame_data: Parsed frame data
        """
        nr = frame_data['nr']
        
        with self._send_lock:
            logger.warning(f"Handling REJ frame: NR={nr}")
            
            # Clear busy flag
            self._busy = False
            
            # Retransmit all frames from NR onwards
            self._retransmit_from(nr)
            self._start_t1_timer()
            
            self.stats['acknowledgments_received'] += 1

    def _handle_srej_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Selective Reject frame.
        
        Args:
            frame_data: Parsed frame data
        """
        nr = frame_data['nr']
        
        with self._send_lock:
            logger.warning(f"Handling SREJ frame: NR={nr}")
            
            # Clear busy flag
            self._busy = False
            
            # Retransmit only the specific frame
            self._retransmit_frame(nr)
            self._start_t1_timer()
            
            self.stats['acknowledgments_received'] += 1

    def _handle_ua_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Unnumbered Acknowledgment frame."""
        if self.state == AX25State.AWAITING_CONNECTION:
            self._change_state(AX25State.CONNECTED)
            self._stop_t1_timer()
            self._busy = False
            logger.info("Connection established")
            if self.on_connection_established:
                self.on_connection_established()
        elif self.state == AX25State.AWAITING_RELEASE:
            self._change_state(AX25State.DISCONNECTED)
            self._stop_t1_timer()
            self._busy = False
            self.stats['disconnects'] += 1
            logger.info("Connection released")
            if self.on_connection_terminated:
                self.on_connection_terminated()

    def _handle_dm_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Disconnected Mode frame."""
        if self.state == AX25State.AWAITING_CONNECTION:
            self._change_state(AX25State.DISCONNECTED)
            self._busy = False
            logger.info("Connection rejected")

    def _handle_sabm_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Set Async Balanced Mode frame."""
        if self.state == AX25State.DISCONNECTED:
            # Respond to connection request
            ua_frame = self._build_ua_frame()
            self._send_frame(ua_frame)
            self._change_state(AX25State.CONNECTED)
            self._busy = False
            logger.info("Connection accepted")
            if self.on_connection_established:
                self.on_connection_established()

    def _handle_disc_frame(self, frame_data: Dict[str, Any]) -> None:
        """Handle Disconnect frame."""
        if self.state == AX25State.CONNECTED:
            # Respond with UA and disconnect
            ua_frame = self._build_ua_frame()
            self._send_frame(ua_frame)
            self._change_state(AX25State.DISCONNECTED)
            self._busy = False
            self.stats['disconnects'] += 1
            logger.info("Connection terminated by remote")
            if self.on_connection_terminated:
                self.on_connection_terminated()

    def _build_i_frame(self, data: bytes, ns: int, nr: int) -> bytes:
        """Build Information frame.
        
        Args:
            data: Frame payload
            ns: Send sequence number
            nr: Receive sequence number
            
        Returns:
            Encoded I-frame bytes
        """
        try:
            from .framing import AX25Frame, AX25Address
            
            dest_addr = AX25Address(self.dest_call, 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame = AX25Frame(dest_addr, src_addr)
            encoded = frame.encode_i(data, ns, nr, poll=False)
            
            logger.debug(f"Built I-frame: NS={ns}, NR={nr}, size={len(data)}")
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to build I-frame: {e}")
            raise ConnectionError(f"Failed to build I-frame: {e}") from e

    def _build_ua_frame(self) -> bytes:
        """Build Unnumbered Acknowledgment frame.
        
        Returns:
            Encoded UA frame bytes
        """
        try:
            from .framing import AX25Frame, AX25Address
            
            dest_addr = AX25Address(self.dest_call, 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame = AX25Frame(dest_addr, src_addr)
            encoded = frame.encode_u(frame.type.UA, poll=False)
            
            logger.debug("Built UA frame")
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to build UA frame: {e}")
            raise ConnectionError(f"Failed to build UA frame: {e}") from e

    def _send_sabm(self) -> None:
        """Send Set Async Balanced Mode frame."""
        try:
            from .framing import AX25Frame, AX25Address
            
            dest_addr = AX25Address(self.dest_call, 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame = AX25Frame(dest_addr, src_addr)
            encoded = frame.encode_sabm(poll=True)
            
            self._send_frame(encoded)
            logger.debug("Sent SABM frame")
            
        except Exception as e:
            logger.error(f"Failed to send SABM: {e}")
            raise ConnectionError(f"Failed to send SABM: {e}") from e

    def _send_disc(self) -> None:
        """Send Disconnect frame."""
        try:
            from .framing import AX25Frame, AX25Address
            
            dest_addr = AX25Address(self.dest_call, 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame = AX25Frame(dest_addr, src_addr)
            encoded = frame.encode_disc(poll=True)
            
            self._send_frame(encoded)
            logger.debug("Sent DISC frame")
            
        except Exception as e:
            logger.error(f"Failed to send DISC: {e}")
            raise ConnectionError(f"Failed to send DISC: {e}") from e

    def _send_rr(self) -> None:
        """Send Receive Ready frame."""
        try:
            from .framing import AX25Frame, AX25Address, FrameType
            
            dest_addr = AX25Address(self.dest_call, 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame = AX25Frame(dest_addr, src_addr)
            encoded = frame.encode_s(FrameType.RR, self.vr, poll=False)
            
            self._send_frame(encoded)
            self.stats['acknowledgments_sent'] += 1
            logger.debug(f"Sent RR frame: NR={self.vr}")
            
        except Exception as e:
            logger.error(f"Failed to send RR: {e}")

    def _send_rej(self, nr: int) -> None:
        """Send Reject frame.
        
        Args:
            nr: Sequence number to reject
        """
        try:
            from .framing import AX25Frame, AX25Address, FrameType
            
            dest_addr = AX25Address(self.dest_call, 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame = AX25Frame(dest_addr, src_addr)
            encoded = frame.encode_s(FrameType.REJ, nr, poll=False)
            
            self._send_frame(encoded)
            self.stats['acknowledgments_sent'] += 1
            logger.warning(f"Sent REJ frame: NR={nr}")
            
        except Exception as e:
            logger.error(f"Failed to send REJ: {e}")

    def _send_srej(self, nr: int) -> None:
        """Send Selective Reject frame.
        
        Args:
            nr: Sequence number to selectively reject
        """
        try:
            from .framing import AX25Frame, AX25Address, FrameType
            
            dest_addr = AX25Address(self.dest_call, 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame = AX25Frame(dest_addr, src_addr)
            encoded = frame.encode_s(FrameType.SREJ, nr, poll=False)
            
            self._send_frame(encoded)
            self.stats['acknowledgments_sent'] += 1
            logger.warning(f"Sent SREJ frame: NR={nr}")
            
        except Exception as e:
            logger.error(f"Failed to send SREJ: {e}")

    def _send_frame(self, frame: bytes) -> None:
        """Send frame via transport.
        
        Args:
            frame: Raw frame bytes
        """
        try:
            self.send_frame_fn(frame)
            logger.debug(f"Transport sent {len(frame)} bytes")
        except Exception as e:
            logger.error(f"Transport send failed: {e}")
            raise ConnectionError(f"Transport send failed: {e}") from e

    def _queue_frame_for_retransmission(self, frame: bytes, ns: int) -> None:
        """Queue frame for retransmission.
        
        Args:
            frame: Frame bytes
            ns: Sequence number
        """
        entry = {
            'frame': frame,
            'ns': ns,
            'timestamp': time.time(),
            'retries': 0
        }
        self.send_queue.append(entry)
        logger.debug(f"Queued frame NS={ns} for retransmission")

    def _can_send(self) -> bool:
        """Check if we can send more frames.
        
        Returns:
            True if window allows sending, False otherwise
        """
        unacked_frames = len([entry for entry in self.send_queue if entry['ns'] >= self.va])
        return unacked_frames < self.config.k_window

    def _is_valid_receive_sequence(self, ns: int) -> bool:
        """Check if receive sequence number is valid.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if valid, False otherwise
        """
        return 0 <= ns <= 7  # Modulo 8 for now

    def _is_in_receive_window(self, ns: int) -> bool:
        """Check if sequence number is in receive window.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if in window, False otherwise
        """
        window_size = self.config.k_window
        diff = (ns - self.vr) % 8
        return 0 <= diff < window_size

    def _update_send_window(self, nr: int) -> None:
        """Update send window based on acknowledgment.
        
        Args:
            nr: Acknowledged sequence number
        """
        while self.va != nr:
            self.va = (self.va + 1) % 8
            
            # Clean up send queue
            self.send_queue = [entry for entry in self.send_queue if entry['ns'] != self.va]
            
        self._stop_t1_timer()

    def _cleanup_acked_out_of_order(self, nr: int) -> None:
        """Clean up acknowledged out-of-order frames.
        
        Args:
            nr: Acknowledged sequence number
        """
        # Remove any out-of-order frames that are now acknowledged
        self.out_of_order_frames = {ns: frame for ns, frame in self.out_of_order_frames.items() 
                                   if ns >= nr}

    def _retransmit_from(self, nr: int) -> None:
        """Retransmit all frames from NR onwards.
        
        Args:
            nr: Sequence number to start retransmission from
        """
        logger.info(f"Retransmitting from NR={nr}")
        
        frames_to_resend = [entry for entry in self.send_queue if entry['ns'] >= nr]
        
        for entry in frames_to_resend:
            entry['retries'] += 1
            if entry['retries'] > self.config.n2_retries:
                logger.error(f"Max retries exceeded for frame NS={entry['ns']}")
                self._handle_connection_timeout()
                return
                
            self._send_frame(entry['frame'])
            entry['timestamp'] = time.time()
            self.stats['retransmissions'] += 1

    def _retransmit_frame(self, ns: int) -> None:
        """Retransmit specific frame.
        
        Args:
            ns: Sequence number of frame to retransmit
        """
        frame_entry = next((entry for entry in self.send_queue if entry['ns'] == ns), None)
        if frame_entry:
            frame_entry['retries'] += 1
            if frame_entry['retries'] > self.config.n2_retries:
                logger.error(f"Max retries exceeded for frame NS={ns}")
                return
                
            self._send_frame(frame_entry['frame'])
            frame_entry['timestamp'] = time.time()
            self.stats['retransmissions'] += 1
            logger.info(f"Retransmitted frame NS={ns}")

    def _deliver_frame(self, data: bytes) -> None:
        """Deliver frame to application.
        
        Args:
            data: Frame payload data
        """
        self.stats['frames_received'] += 1
        self.stats['frames_delivered'] += 1
        
        # Check for buffer overflow
        if len(self.receive_buffer) + len(data) > self.config.receive_buffer_size:
            logger.warning("Receive buffer overflow")
            self.stats['buffer_overflows'] += 1
            # Clear buffer and start fresh
            self.receive_buffer.clear()
        
        # Add to receive buffer
        self.receive_buffer.extend(data)
        
        # Call frame callback
        if self.frame_callback:
            try:
                self.frame_callback(data)
            except Exception as e:
                logger.error(f"Frame callback failed: {e}")
        
        # Call frame delivered callback
        if self.on_frame_delivered:
            try:
                self.on_frame_delivered(data)
            except Exception as e:
                logger.error(f"Frame delivered callback failed: {e}")

    def _start_t1_timer(self) -> None:
        """Start T1 retransmission timer."""
        with self._timer_lock:
            self._stop_t1_timer()
            self.t1_timer = threading.Timer(self.config.t1_timeout, self._handle_t1_timeout)
            self.t1_timer.daemon = True
            self.t1_timer.start()
            self._retransmission_pending = True
            logger.debug(f"T1 timer started ({self.config.t1_timeout}s)")

    def _stop_t1_timer(self) -> None:
        """Stop T1 retransmission timer."""
        with self._timer_lock:
            if self.t1_timer:
                self.t1_timer.cancel()
                self.t1_timer = None
            self._retransmission_pending = False
            logger.debug("T1 timer stopped")

    def _reset_t3_timer(self) -> None:
        """Reset T3 inactivity timer."""
        with self._timer_lock:
            if self.t3_timer:
                self.t3_timer.cancel()
            self.t3_timer = threading.Timer(self.config.t3_timeout, self._handle_t3_timeout)
            self.t3_timer.daemon = True
            self.t3_timer.start()
            logger.debug(f"T3 timer reset ({self.config.t3_timeout}s)")

    def _stop_t3_timer(self) -> None:
        """Stop T3 inactivity timer."""
        with self._timer_lock:
            if self.t3_timer:
                self.t3_timer.cancel()
                self.t3_timer = None
            logger.debug("T3 timer stopped")

    def _handle_t1_timeout(self) -> None:
        """Handle T1 retransmission timeout."""
        logger.warning("T1 timeout - retransmitting unacknowledged frames")
        
        # Check retry count
        unacked_entries = [entry for entry in self.send_queue if entry['retries'] < self.config.n2_retries]
        
        if not unacked_entries:
            logger.error("All frames exceeded retry limit")
            self._handle_connection_timeout()
            return
        
        # Retransmit oldest unacknowledged frame
        oldest_entry = min(unacked_entries, key=lambda x: x['timestamp'])
        self._retransmit_frame(oldest_entry['ns'])
        
        # Restart T1
        self._start_t1_timer()

    def _handle_t3_timeout(self) -> None:
        """Handle T3 inactivity timeout."""
        logger.warning("T3 timeout - connection inactivity")
        self._handle_connection_timeout()

    def _handle_connection_timeout(self) -> None:
        """Handle connection timeout."""
        logger.error("Connection timeout - terminating connection")
        self._change_state(AX25State.TIMEOUT)
        
        # Clean up timers
        self._stop_t1_timer()
        self._stop_t3_timer()
        
        # Notify error handler
        if self.on_error:
            self.on_error(ConnectionTimeout("Connection timeout"))

    def _change_state(self, new_state: AX25State) -> None:
        """Update connection state.
        
        Args:
            new_state: New state to transition to
        """
        old_state = self.state
        if old_state != new_state:
            self.state = new_state
            logger.info(f"Connection state: {old_state.name} -> {new_state.name}")
            
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception as e:
                    logger.error(f"State change callback failed: {e}")

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get connection statistics.
        
        Returns:
            Dictionary of connection statistics
        """
        with self._state_lock:
            stats = self.stats.copy()
            stats['uptime'] = time.time() - stats['connection_time'] if stats['connection_time'] else 0
            stats['last_activity_ago'] = time.time() - stats['last_activity'] if stats['last_activity'] else 0
            return stats

    def reset_stats(self) -> None:
        """Reset connection statistics."""
        with self._state_lock:
            self.stats = {
                'frames_sent': 0,
                'frames_received': 0,
                'retransmissions': 0,
                'errors': 0,
                'connect_attempts': 0,
                'disconnects': 0,
                'sequence_errors': 0,
                'window_violations': 0,
                'acknowledgments_sent': 0,
                'acknowledgments_received': 0,
                'out_of_order_frames': 0,
                'frames_delivered': 0,
                'buffer_overflows': 0,
                'connection_time': time.time(),
                'last_activity': time.time()
            }
            logger.debug("Statistics reset")

    def get_connection_info(self) -> Dict[str, Union[str, int, float, bool]]:
        """Get connection information.
        
        Returns:
            Dictionary of connection status
        """
        return {
            'state': self.state.name,
            'my_call': self.my_call,
            'dest_call': self.dest_call or 'None',
            'vs': self.vs,
            'vr': self.vr,
            'va': self.va,
            'send_queue_length': len(self.send_queue),
            'out_of_order_count': len(self.out_of_order_frames),
            'receive_buffer_size': len(self.receive_buffer),
            'busy': self._busy,
            'retransmission_pending': self._retransmission_pending,
            'uptime': time.time() - self.stats['connection_time'] if self.stats['connection_time'] else 0,
            'window_size': self.config.k_window,
            'max_frame_size': self.config.max_frame_size,
            't1_timeout': self.config.t1_timeout,
            't3_timeout': self.config.t3_timeout
        }

    def clear_buffers(self) -> None:
        """Clear receive and send buffers."""
        with self._recv_lock:
            self.receive_buffer.clear()
        with self._send_lock:
            self.send_buffer.clear()
        logger.debug("Buffers cleared")

    def is_connected(self) -> bool:
        """Check if connection is established.
        
        Returns:
            True if connected, False otherwise
        """
        return self.state == AX25State.CONNECTED

    def get_receive_buffer(self) -> bytes:
        """Get receive buffer contents.
        
        Returns:
            Receive buffer data
        """
        with self._recv_lock:
            return bytes(self.receive_buffer)

    def clear_receive_buffer(self) -> None:
        """Clear receive buffer."""
        with self._recv_lock:
            self.receive_buffer.clear()

    def get_send_queue_info(self) -> List[Dict[str, Union[int, float]]]:
        """Get information about queued frames.
        
        Returns:
            List of queued frame information
        """
        with self._send_lock:
            return [
                {
                    'ns': entry['ns'],
                    'retries': entry['retries'],
                    'age': time.time() - entry['timestamp']
                }
                for entry in self.send_queue
            ]

    def cleanup(self) -> None:
        """Clean up connection resources."""
        self._stop_t1_timer()
        self._stop_t3_timer()
        self.send_queue.clear()
        self.received_frames.clear()
        self.out_of_order_frames.clear()
        self.receive_buffer.clear()
        self.send_buffer.clear()
        logger.debug("Connection cleaned up")

    def __repr__(self) -> str:
        return (f"ConnectedModeHandler(state={self.state.name}, "
                f"dest={self.dest_call}, "
                f"vs={self.vs}, vr={self.vr}, va={self.va}, "
                f"queue={len(self.send_queue)})")

