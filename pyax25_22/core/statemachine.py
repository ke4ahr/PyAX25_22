# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
AX.25 Connected-Mode State Machine

Handles:
- Connection establishment/teardown
- Information frames (I-frames)
- Timers (T1, T2, T3)
- Sequence numbering (modulo 8/128)
- Retransmissions
- Flow control

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import threading
import time
import logging
from enum import Enum, IntEnum
from typing import Optional, List, Dict, Callable, Union, Any
from collections import deque

logger = logging.getLogger(__name__)

class AX25State(Enum):
    """Connection states"""
    DISCONNECTED = 0
    AWAITING_CONNECTION = 1
    CONNECTED = 2
    AWAITING_RELEASE = 3
    TIMEOUT = 4

class AX25Modulo(IntEnum):
    """Window modulus"""
    MOD8 = 8
    MOD128 = 128

class FrameReject(Enum):
    """REJ handling options"""
    REJ = 1      # Standard REJ
    SREJ = 2     # Selective REJ
    NONE = 3     # No REJ (rely on timeout)

class SREJManager:
    """Selective Reject (SREJ) handler"""
    def __init__(self, modulo: AX25Modulo = AX25Modulo.MOD8):
        """Initialize SREJ manager.
        
        Args:
            modulo: Window modulus (MOD8 or MOD128)
        """
        if modulo not in [AX25Modulo.MOD8, AX25Modulo.MOD128]:
            raise ValueError(f"Invalid modulo: {modulo}")
            
        self.modulo = modulo
        self._srej_queue: Dict[int, bytes] = {}
        self._expected_ns = 0
        
        logger.debug(f"Initialized SREJ manager with modulo {modulo}")
        
    def add_frame(self, ns: int, frame: bytes) -> None:
        """Store out-of-sequence frame.
        
        Args:
            ns: Sequence number
            frame: Frame data to store
        """
        if not 0 <= ns < self.modulo:
            raise ValueError(f"Invalid sequence number {ns} for modulo {self.modulo}")
        if not isinstance(frame, bytes):
            raise TypeError("Frame must be bytes")
            
        if ns not in self._srej_queue:
            self._srej_queue[ns] = frame
            logger.debug(f"SREJ: Stored frame NS={ns}")
        else:
            logger.warning(f"SREJ: Duplicate frame NS={ns}, ignoring")
            
    def get_next(self) -> Optional[bytes]:
        """Get next in-order frame if available.
        
        Returns:
            Frame data if available, None otherwise
        """
        frame = self._srej_queue.get(self._expected_ns)
        if frame is not None:
            self._srej_queue.pop(self._expected_ns)
            self._expected_ns = (self._expected_ns + 1) % self.modulo
            logger.debug(f"SREJ: Delivered frame NS={self._expected_ns - 1}")
        return frame
        
    def needs_srej(self, ns: int) -> bool:
        """Check if NS requires SREJ.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if SREJ needed, False otherwise
        """
        if not 0 <= ns < self.modulo:
            raise ValueError(f"Invalid sequence number {ns} for modulo {self.modulo}")
            
        expected = self._expected_ns
        diff = (ns - expected) % self.modulo
        result = 0 < diff <= (self.modulo // 2)
        logger.debug(f"SREJ check: NS={ns}, expected={expected}, needs SREJ={result}")
        return result

class FrameEntry:
    """Queued frame metadata"""
    def __init__(self, frame: bytes, ns: int, timestamp: float):
        """Initialize frame entry.
        
        Args:
            frame: Raw frame data
            ns: Sequence number
            timestamp: Time frame was sent
        """
        if not isinstance(frame, bytes):
            raise TypeError("Frame must be bytes")
        if not 0 <= ns <= 127:
            raise ValueError(f"Invalid sequence number: {ns}")
        if not isinstance(timestamp, (int, float)):
            raise TypeError("Timestamp must be numeric")
            
        self.frame = frame
        self.ns = ns
        self.timestamp = timestamp
        self.retries = 0
        
        logger.debug(f"Created frame entry: NS={ns}, size={len(frame)}")

class TimerManager:
    """AX.25 timer management"""
    def __init__(self,
                 t1_timeout: float = 10.0,
                 t2_timeout: float = 2.0, 
                 t3_timeout: float = 30.0):
        """Initialize timer manager.
        
        Args:
            t1_timeout: Frame retransmission timeout (seconds)
            t2_timeout: Acknowledgment timeout (seconds)
            t3_timeout: Inactivity timeout (seconds)
        """
        if not all(isinstance(t, (int, float)) and t > 0 for t in [t1_timeout, t2_timeout, t3_timeout]):
            raise ValueError("All timeouts must be positive numbers")
            
        self.t1_timeout = t1_timeout
        self.t2_timeout = t2_timeout
        self.t3_timeout = t3_timeout
        
        # Timer states
        self.t1_running = False
        self.t2_running = False
        self.t3_running = False
        
        self.t1_timer: Optional[threading.Timer] = None
        self.t2_timer: Optional[threading.Timer] = None
        self.t3_timer: Optional[threading.Timer] = None
        
        # Callbacks
        self.on_t1_expired: Optional[Callable[[], None]] = None
        self.on_t2_expired: Optional[Callable[[], None]] = None
        self.on_t3_expired: Optional[Callable[[], None]] = None
        
        logger.debug(f"Initialized timer manager: T1={t1_timeout}s, T2={t2_timeout}s, T3={t3_timeout}s")

    def _t1_expired(self) -> None:
        """Handle T1 retransmission timeout"""
        logger.warning(f"T1 timeout expired ({self.t1_timeout}s)")
        self.t1_running = False
        if self.on_t1_expired:
            try:
                self.on_t1_expired()
            except Exception as e:
                logger.error(f"T1 callback failed: {e}")

    def _t2_expired(self) -> None:
        """Handle T2 acknowledgment timeout"""
        logger.warning(f"T2 timeout expired ({self.t2_timeout}s)")
        self.t2_running = False
        if self.on_t2_expired:
            try:
                self.on_t2_expired()
            except Exception as e:
                logger.error(f"T2 callback failed: {e}")

    def _t3_expired(self) -> None:
        """Handle T3 inactivity timeout"""
        logger.warning(f"T3 timeout expired ({self.t3_timeout}s)")
        self.t3_running = False
        if self.on_t3_expired:
            try:
                self.on_t3_expired()
            except Exception as e:
                logger.error(f"T3 callback failed: {e}")

    def start_t1(self) -> None:
        """Start T1 retransmission timer"""
        if self.t1_running:
            logger.debug("T1 already running, ignoring start request")
            return
            
        self.t1_running = True
        self.t1_timer = threading.Timer(self.t1_timeout, self._t1_expired)
        self.t1_timer.daemon = True
        self.t1_timer.start()
        logger.debug(f"T1 started ({self.t1_timeout}s)")

    def stop_t1(self) -> None:
        """Stop T1 retransmission timer"""
        if not self.t1_running:
            logger.debug("T1 not running, ignoring stop request")
            return
            
        self.t1_running = False
        if self.t1_timer:
            self.t1_timer.cancel()
            self.t1_timer = None
        logger.debug("T1 stopped")

    def start_t2(self) -> None:
        """Start T2 acknowledgment timer"""
        if self.t2_running:
            logger.debug("T2 already running, ignoring start request")
            return
            
        self.t2_running = True
        self.t2_timer = threading.Timer(self.t2_timeout, self._t2_expired)
        self.t2_timer.daemon = True
        self.t2_timer.start()
        logger.debug(f"T2 started ({self.t2_timeout}s)")

    def stop_t2(self) -> None:
        """Stop T2 acknowledgment timer"""
        if not self.t2_running:
            logger.debug("T2 not running, ignoring stop request")
            return
            
        self.t2_running = False
        if self.t2_timer:
            self.t2_timer.cancel()
            self.t2_timer = None
        logger.debug("T2 stopped")

    def start_t3(self) -> None:
        """Start T3 inactivity timer"""
        if self.t3_running:
            logger.debug("T3 already running, ignoring start request")
            return
            
        self.t3_running = True
        self.t3_timer = threading.Timer(self.t3_timeout, self._t3_expired)
        self.t3_timer.daemon = True
        self.t3_timer.start()
        logger.debug(f"T3 started ({self.t3_timeout}s)")

    def stop_t3(self) -> None:
        """Stop T3 inactivity timer"""
        if not self.t3_running:
            logger.debug("T3 not running, ignoring stop request")
            return
            
        self.t3_running = False
        if self.t3_timer:
            self.t3_timer.cancel()
            self.t3_timer = None
        logger.debug("T3 stopped")

    def reset_t3(self) -> None:
        """Reset T3 on any frame activity"""
        self.stop_t3()
        self.start_t3()
        logger.debug("T3 reset")

    def cleanup(self) -> None:
        """Clean up all timers"""
        self.stop_t1()
        self.stop_t2()
        self.stop_t3()
        logger.debug("All timers cleaned up")

class AX25StateMachineError(Exception):
    """Custom exception for state machine errors"""
    pass

class AX25StateMachine:
    """
    AX.25 Connected-Mode State Machine
    
    Implements:
    - V(S) - Send state variable
    - V(R) - Receive state variable
    - V(A) - Acknowledgment state variable
    - N2 - Retry counter
    - T1 - Frame retransmission timer
    """
    
    def __init__(
        self,
        my_call: str,
        modulo: AX25Modulo = AX25Modulo.MOD8,
        t1_timeout: float = 10.0,
        n2_retries: int = 10,
        k_window: int = 4,
        rej_policy: FrameReject = FrameReject.REJ
    ):
        """Initialize AX.25 state machine.
        
        Args:
            my_call: Local callsign
            modulo: Window modulus (MOD8 or MOD128)
            t1_timeout: T1 retransmission timeout (seconds)
            n2_retries: Maximum retransmission attempts
            k_window: Window size
            rej_policy: REJ handling policy
        """
        if not isinstance(my_call, str) or not my_call.strip():
            raise ValueError("Callsign must be a non-empty string")
        if modulo not in [AX25Modulo.MOD8, AX25Modulo.MOD128]:
            raise ValueError(f"Invalid modulo: {modulo}")
        if not isinstance(t1_timeout, (int, float)) or t1_timeout <= 0:
            raise ValueError("T1 timeout must be positive")
        if not isinstance(n2_retries, int) or n2_retries < 1:
            raise ValueError("N2 retries must be positive integer")
        if not isinstance(k_window, int) or k_window < 1:
            raise ValueError("Window size must be positive integer")
        if rej_policy not in FrameReject:
            raise ValueError(f"Invalid REJ policy: {rej_policy}")
            
        self.my_call = my_call.strip().upper()
        self.modulo = modulo
        self.t1_timeout = t1_timeout
        self.n2_retries = n2_retries
        self.k_window = k_window
        self.rej_policy = rej_policy
        
        # State variables
        self.state = AX25State.DISCONNECTED
        self.vs = 0      # Send sequence number
        self.vr = 0      # Receive sequence number  
        self.va = 0      # Acknowledged sequence number
        
        # Frame management
        self.send_queue: List[FrameEntry] = []
        self.received_frames: Dict[int, bytes] = {}
        self.srej = SREJManager(modulo)
        
        # Timer management
        self.timers = TimerManager(t1_timeout)
        self._setup_timer_callbacks()
        
        # Locking
        self._state_lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._recv_lock = threading.RLock()
        
        # Callbacks
        self.on_frame_received: Optional[Callable[[bytes], None]] = None
        self.on_state_change: Optional[Callable[[AX25State], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_frame_sent: Optional[Callable[[int, bytes], None]] = None
        self.on_frame_acked: Optional[Callable[[int], None]] = None
        self.on_retransmit: Optional[Callable[[int], None]] = None
        
        # Statistics
        self._stats = {
            'frames_sent': 0,
            'frames_received': 0,
            'frames_acked': 0,
            'retransmissions': 0,
            'errors': 0,
            'connect_attempts': 0,
            'disconnects': 0,
            'sequence_errors': 0,
            'window_violations': 0,
            't1_timeouts': 0,
            't3_timeouts': 0
        }
        
        logger.info(f"Initialized state machine for {self.my_call}, modulo {modulo}, window {k_window}")

    def _setup_timer_callbacks(self) -> None:
        """Set up timer expiration callbacks"""
        self.timers.on_t1_expired = self._handle_t1_timeout
        self.timers.on_t2_expired = self._handle_t2_timeout
        self.timers.on_t3_expired = self._handle_t3_timeout
        logger.debug("Timer callbacks configured")

    def _send_frame(self, frame: bytes) -> None:
        """Send raw frame to transport layer.
        
        Args:
            frame: Raw frame bytes to send
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Transport must implement _send_frame")

    def _build_u_frame(self, control: int) -> bytes:
        """Build unnumbered frame using AX25Frame.
        
        Args:
            control: Control byte for the frame
            
        Returns:
            Encoded frame bytes
            
        Raises:
            ImportError: If AX25Frame is not available
        """
        try:
            from .framing import AX25Frame, AX25Address
            
            # Create dummy addresses for U-frames (address handling depends on transport)
            dest_addr = AX25Address("DEST", 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame_obj = AX25Frame(dest_addr, src_addr)
            frame_obj.control = control
            frame_obj.type = self._control_to_frame_type(control)
            
            encoded = frame_obj.encode()
            logger.debug(f"Built U-frame: control=0x{control:02X}, type={frame_obj.type}")
            return encoded
            
        except ImportError as e:
            logger.error(f"Failed to import AX25Frame: {e}")
            raise ImportError("AX25Frame not available") from e

    def _control_to_frame_type(self, control: int) -> Any:
        """Convert control byte to frame type.
        
        Args:
            control: Control byte
            
        Returns:
            FrameType enum value
        """
        from .framing import FrameType
        
        if (control & 0x01) == 0:
            return FrameType.I
        elif (control & 0x0F) == 0x03:
            return FrameType.UI
        elif (control & 0x0F) == 0x0F:
            return FrameType.DM
        elif (control & 0x0F) == 0x2F:
            return FrameType.SABM
        elif (control & 0x0F) == 0x43:
            return FrameType.DISC
        elif (control & 0x0F) == 0x63:
            return FrameType.UA
        elif (control & 0x0F) == 0x01:
            return FrameType.RR
        elif (control & 0x0F) == 0x05:
            return FrameType.RNR
        elif (control & 0x0F) == 0x09:
            return FrameType.REJ
        elif (control & 0x0F) == 0x0D:
            return FrameType.SREJ
        else:
            return FrameType.UI  # Default

    def _build_i_frame(self, data: bytes, ns: int) -> bytes:
        """Build information frame with proper sequence numbers.
        
        Args:
            data: Information field data
            ns: Send sequence number
            
        Returns:
            Encoded I-frame bytes
        """
        try:
            from .framing import AX25Frame, AX25Address
            
            dest_addr = AX25Address("DEST", 0)  # Transport will set proper destination
            src_addr = AX25Address(self.my_call, 0)
            
            frame_obj = AX25Frame(dest_addr, src_addr)
            encoded = frame_obj.encode_i(data, ns, self.vr, poll=False)
            
            logger.debug(f"Built I-frame: NS={ns}, VR={self.vr}, size={len(data)}")
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to build I-frame: {e}")
            raise AX25StateMachineError(f"Failed to build I-frame: {e}") from e

    def _build_s_frame(self, frame_type: Any, nr: int = 0, poll: bool = False) -> bytes:
        """Build supervisory frame.
        
        Args:
            frame_type: Type of supervisory frame
            nr: Receive sequence number
            poll: Poll flag
            
        Returns:
            Encoded S-frame bytes
        """
        try:
            from .framing import AX25Frame, AX25Address, FrameType
            
            dest_addr = AX25Address("DEST", 0)
            src_addr = AX25Address(self.my_call, 0)
            
            frame_obj = AX25Frame(dest_addr, src_addr)
            
            if frame_type == FrameType.RR:
                encoded = frame_obj.encode_s(FrameType.RR, nr, poll)
            elif frame_type == FrameType.RNR:
                encoded = frame_obj.encode_s(FrameType.RNR, nr, poll)
            elif frame_type == FrameType.REJ:
                encoded = frame_obj.encode_s(FrameType.REJ, nr, poll)
            elif frame_type == FrameType.SREJ:
                encoded = frame_obj.encode_s(FrameType.SREJ, nr, poll)
            else:
                raise ValueError(f"Invalid supervisory frame type: {frame_type}")
                
            logger.debug(f"Built S-frame: type={frame_type.name}, NR={nr}, P={poll}")
            return encoded
            
        except Exception as e:
            logger.error(f"Failed to build S-frame: {e}")
            raise AX25StateMachineError(f"Failed to build S-frame: {e}") from e

    def _handle_frame(self, frame: Any) -> None:
        """Complete frame handling with proper state transitions.
        
        Args:
            frame: Parsed AX25Frame object
        """
        try:
            with self._recv_lock:
                logger.debug(f"Handling frame: {frame.type.name}")
                
                if frame.type == FrameType.UI:
                    self._handle_ui_frame(frame)
                elif frame.type == FrameType.I:
                    self._handle_i_frame(frame)
                elif frame.type in [FrameType.RR, FrameType.RNR]:
                    self._handle_s_frame(frame)
                elif frame.type == FrameType.REJ:
                    self._handle_rej_frame(frame)
                elif frame.type == FrameType.SREJ:
                    self._handle_srej_frame(frame)
                elif frame.type == FrameType.UA:
                    self._handle_ua_frame(frame)
                elif frame.type == FrameType.DM:
                    self._handle_dm_frame(frame)
                elif frame.type in [FrameType.SABM, FrameType.DISC]:
                    self._handle_u_frame(frame)
                else:
                    logger.warning(f"Unhandled frame type: {frame.type.name}")
                    self._stats['errors'] += 1
                    
        except Exception as e:
            logger.error(f"Frame handling failed: {e}")
            self._stats['errors'] += 1
            if self.on_error:
                self.on_error(e)

    def _handle_ui_frame(self, frame: Any) -> None:
        """Handle Unnumbered Information frame."""
        logger.debug("Handling UI frame")
        # UI frames are unsequenced, pass directly to application
        if self.on_frame_received and frame.info:
            self.on_frame_received(frame.info)
        self._stats['frames_received'] += 1

    def _handle_i_frame(self, frame: Any) -> None:
        """Handle Information frame."""
        with self._recv_lock:
            ns = frame.ns
            nr = frame.nr
            
            logger.debug(f"Handling I-frame: NS={ns}, NR={nr}")
            
            # Update send window based on NR
            self._update_send_window(nr)
            
            # Check sequence number validity
            if not self._is_valid_receive_sequence(ns):
                logger.warning(f"Invalid receive sequence number: NS={ns}, VR={self.vr}")
                self._stats['sequence_errors'] += 1
                if self.rej_policy == FrameReject.REJ:
                    self._send_rej(self.vr)
                return
            
            # Check receive window
            if self._is_in_receive_window(ns):
                # Accept frame
                self.received_frames[ns] = frame.info
                self._advance_receive_window()
                
                # Send acknowledgment
                self._send_rr()
                
                # Deliver to application
                if self.on_frame_received:
                    self.on_frame_received(frame.info)
                
                self._stats['frames_received'] += 1
            else:
                # Out of window
                logger.warning(f"I-frame out of receive window: NS={ns}, VR={self.vr}")
                self._stats['window_violations'] += 1
                if self.rej_policy == FrameReject.REJ:
                    self._send_rej(self.vr)
                elif self.rej_policy == FrameReject.SREJ:
                    self._send_srej(ns)

    def _handle_s_frame(self, frame: Any) -> None:
        """Handle Supervisory frame (RR, RNR)."""
        with self._send_lock:
            nr = frame.nr
            
            logger.debug(f"Handling S-frame: {frame.type.name}, NR={nr}")
            
            # Update send window
            self._update_send_window(nr)
            
            if frame.type == FrameType.RNR:
                logger.warning("Remote station busy (RNR received)")
                # Handle busy condition - stop sending

    def _handle_rej_frame(self, frame: Any) -> None:
        """Handle Reject frame."""
        with self._send_lock:
            nr = frame.nr
            
            logger.warning(f"Handling REJ frame: NR={nr}")
            
            # Retransmit all frames starting from NR
            self._retransmit_from(nr)
            self.timers.start_t1()
            self._stats['retransmissions'] += 1

    def _handle_srej_frame(self, frame: Any) -> None:
        """Handle Selective Reject frame."""
        with self._send_lock:
            nr = frame.nr
            
            logger.warning(f"Handling SREJ frame: NR={nr}")
            
            # Retransmit only the specific frame
            self._retransmit_frame(nr)
            self.timers.start_t1()
            self._stats['retransmissions'] += 1

    def _handle_ua_frame(self, frame: Any) -> None:
        """Handle Unnumbered Acknowledgment frame."""
        if self.state == AX25State.AWAITING_CONNECTION:
            self._change_state(AX25State.CONNECTED)
            self.timers.stop_t1()
            logger.info("Connection established")
        elif self.state == AX25State.AWAITING_RELEASE:
            self._change_state(AX25State.DISCONNECTED)
            self.timers.stop_t1()
            logger.info("Connection released")

    def _handle_dm_frame(self, frame: Any) -> None:
        """Handle Disconnected Mode frame."""
        if self.state == AX25State.AWAITING_CONNECTION:
            self._change_state(AX25State.DISCONNECTED)
            logger.info("Connection rejected")

    def _handle_u_frame(self, frame: Any) -> None:
        """Handle Unnumbered frames (SABM, DISC)."""
        if frame.type == FrameType.SABM and self.state == AX25State.DISCONNECTED:
            # Respond to connection request
            ua_frame = self._build_u_frame(0x63)  # UA
            self._send_frame(ua_frame)
            self._change_state(AX25State.CONNECTED)
            logger.info("Connection accepted")

    def _is_valid_receive_sequence(self, ns: int) -> bool:
        """Check if receive sequence number is valid.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if valid, False otherwise
        """
        # For modulo 8: 0-7, for modulo 128: 0-127
        return 0 <= ns < self.modulo

    def _is_in_receive_window(self, ns: int) -> bool:
        """Check if sequence number is in receive window.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if in window, False otherwise
        """
        window_size = self.k_window
        diff = (ns - self.vr) % self.modulo
        return 0 <= diff < window_size

    def _advance_receive_window(self) -> None:
        """Advance receive window for contiguous frames."""
        while self.vr in self.received_frames:
            frame_data = self.received_frames.pop(self.vr)
            self.vr = (self.vr + 1) % self.modulo
            
            # Deliver to application
            if self.on_frame_received:
                self.on_frame_received(frame_data)

    def _update_send_window(self, nr: int) -> None:
        """Update send window based on acknowledgment.
        
        Args:
            nr: Acknowledged sequence number
        """
        while self.va != nr:
            # Remove acknowledged frame from queue
            self.va = (self.va + 1) % self.modulo
            
            # Clean up send queue
            self.send_queue = [entry for entry in self.send_queue if entry.ns != self.va]
            
            if self.on_frame_acked:
                self.on_frame_acked(self.va)
            
        self._stats['frames_acked'] += 1
        self.timers.stop_t1()

    def _retransmit_from(self, nr: int) -> None:
        """Retransmit all frames from NR onwards.
        
        Args:
            nr: Sequence number to start retransmission from
        """
        logger.info(f"Retransmitting from NR={nr}")
        
        # Find frames to retransmit
        frames_to_resend = [entry for entry in self.send_queue if entry.ns >= nr]
        
        for entry in frames_to_resend:
            entry.retries += 1
            if entry.retries > self.n2_retries:
                logger.error(f"Max retries exceeded for frame NS={entry.ns}")
                self._handle_connection_timeout()
                return
                
            self._send_frame(entry.frame)
            entry.timestamp = time.time()
            
            if self.on_retransmit:
                self.on_retransmit(entry.ns)

    def _retransmit_frame(self, ns: int) -> None:
        """Retransmit specific frame.
        
        Args:
            ns: Sequence number of frame to retransmit
        """
        frame_entry = next((entry for entry in self.send_queue if entry.ns == ns), None)
        if frame_entry:
            frame_entry.retries += 1
            if frame_entry.retries > self.n2_retries:
                logger.error(f"Max retries exceeded for frame NS={ns}")
                return
                
            self._send_frame(frame_entry.frame)
            frame_entry.timestamp = time.time()
            
            if self.on_retransmit:
                self.on_retransmit(ns)
                
            logger.info(f"Retransmitted frame NS={ns}")

    def _send_rr(self) -> None:
        """Send Receive Ready frame."""
        try:
            rr_frame = self._build_s_frame(FrameType.RR, self.vr, poll=False)
            self._send_frame(rr_frame)
            logger.debug(f"Sent RR: NR={self.vr}")
        except Exception as e:
            logger.error(f"Failed to send RR: {e}")

    def _send_rnr(self) -> None:
        """Send Receive Not Ready frame."""
        try:
            rnr_frame = self._build_s_frame(FrameType.RNR, self.vr, poll=False)
            self._send_frame(rnr_frame)
            logger.warning(f"Sent RNR: NR={self.vr}")
        except Exception as e:
            logger.error(f"Failed to send RNR: {e}")

    def _send_rej(self, nr: int) -> None:
        """Send Reject frame."""
        try:
            rej_frame = self._build_s_frame(FrameType.REJ, nr, poll=False)
            self._send_frame(rej_frame)
            logger.warning(f"Sent REJ: NR={nr}")
        except Exception as e:
            logger.error(f"Failed to send REJ: {e}")

    def _send_srej(self, nr: int) -> None:
        """Send Selective Reject frame."""
        try:
            srej_frame = self._build_s_frame(FrameType.SREJ, nr, poll=False)
            self._send_frame(srej_frame)
            logger.warning(f"Sent SREJ: NR={nr}")
        except Exception as e:
            logger.error(f"Failed to send SREJ: {e}")

    def _handle_t1_timeout(self) -> None:
        """Handle T1 retransmission timeout."""
        logger.warning("T1 timeout - retransmitting unacknowledged frames")
        self._stats['t1_timeouts'] += 1
        
        # Retransmit oldest unacknowledged frame
        if self.send_queue:
            oldest_entry = min(self.send_queue, key=lambda x: x.timestamp)
            self._retransmit_frame(oldest_entry.ns)
            self.timers.start_t1()

    def _handle_t2_timeout(self) -> None:
        """Handle T2 acknowledgment timeout."""
        logger.warning("T2 timeout - no acknowledgment received")
        self._stats['t2_timeouts'] += 1

    def _handle_t3_timeout(self) -> None:
        """Handle T3 inactivity timeout."""
        logger.warning("T3 timeout - connection inactivity")
        self._stats['t3_timeouts'] += 1
        self._handle_connection_timeout()

    def _handle_connection_timeout(self) -> None:
        """Handle connection timeout."""
        logger.error("Connection timeout - terminating connection")
        self._change_state(AX25State.TIMEOUT)
        
        # Clean up timers
        self.timers.cleanup()
        
        # Notify error handler
        if self.on_error:
            self.on_error(TimeoutError("Connection timeout"))

    def _change_state(self, new_state: AX25State) -> None:
        """Update state with callback.
        
        Args:
            new_state: New state to transition to
        """
        with self._state_lock:
            old_state = self.state
            if old_state != new_state:
                self.state = new_state
                logger.info(f"State change: {old_state.name} -> {new_state.name}")
                if self.on_state_change:
                    try:
                        self.on_state_change(new_state)
                    except Exception as e:
                        logger.error(f"State change callback failed: {e}")
                        if self.on_error:
                            self.on_error(e)

    def connect(self) -> None:
        """Initiate connection."""
        with self._state_lock:
            if self.state != AX25State.DISCONNECTED:
                raise AX25StateMachineError(f"Cannot connect from state {self.state.name}")
                
            self._change_state(AX25State.AWAITING_CONNECTION)
            sabm_frame = self._build_u_frame(0x2F)  # SABM
            self._send_frame(sabm_frame)
            self.timers.start_t1()
            self._stats['connect_attempts'] += 1
            
            logger.info("Connection attempt initiated")

    def disconnect(self) -> None:
        """Initiate disconnect."""
        with self._state_lock:
            if self.state not in [AX25State.CONNECTED, AX25State.AWAITING_CONNECTION]:
                logger.warning(f"Cannot disconnect from state {self.state.name}")
                return
                
            self._change_state(AX25State.AWAITING_RELEASE)
            disc_frame = self._build_u_frame(0x43)  # DISC
            self._send_frame(disc_frame)
            self.timers.start_t1()
            
            logger.info("Disconnect attempt initiated")

    def send_info(self, data: bytes) -> None:
        """Queue information frame for sending.
        
        Args:
            data: Frame payload data
        """
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
            
        with self._send_lock:
            if self.state != AX25State.CONNECTED:
                raise AX25StateMachineError(f"Cannot send data in state {self.state.name}")
                
            if not self._can_send():
                raise AX25StateMachineError("Send window full")
                
            try:
                i_frame = self._build_i_frame(data, self.vs)
                self._send_frame(i_frame)
                
                # Queue for retransmission
                entry = FrameEntry(i_frame, self.vs, time.time())
                self.send_queue.append(entry)
                
                if self.on_frame_sent:
                    self.on_frame_sent(self.vs, data)
                
                self.vs = (self.vs + 1) % self.modulo
                self.timers.start_t1()
                self._stats['frames_sent'] += 1
                
                logger.debug(f"Sent I-frame: NS={self.vs - 1}, size={len(data)}")
                
            except Exception as e:
                logger.error(f"Failed to send information frame: {e}")
                self._stats['errors'] += 1
                raise AX25StateMachineError(f"Failed to send information frame: {e}") from e

    def receive_frame(self, frame_data: bytes) -> None:
        """Handle received frame data.
        
        Args:
            frame_data: Raw frame bytes
        """
        try:
            from .framing import AX25Frame
            
            # Parse frame
            frame = AX25Frame.from_bytes(frame_data)
            
            # Reset T3 on any activity
            self.timers.reset_t3()
            
            # Handle frame
            self._handle_frame(frame)
            
        except Exception as e:
            logger.error(f"Frame receive failed: {e}")
            self._stats['errors'] += 1
            if self.on_error:
                self.on_error(e)

    def _can_send(self) -> bool:
        """Check if we can send more frames.
        
        Returns:
            True if window allows sending, False otherwise
        """
        unacked_frames = len([entry for entry in self.send_queue if entry.ns >= self.va])
        return unacked_frames < self.k_window

    def get_stats(self) -> Dict[str, int]:
        """Get connection statistics.
        
        Returns:
            Dictionary of connection statistics
        """
        with self._state_lock:
            return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset connection statistics."""
        with self._state_lock:
            self._stats = {
                'frames_sent': 0,
                'frames_received': 0,
                'frames_acked': 0,
                'retransmissions': 0,
                'errors': 0,
                'connect_attempts': 0,
                'disconnects': 0,
                'sequence_errors': 0,
                'window_violations': 0,
                't1_timeouts': 0,
                't3_timeouts': 0
            }

    def cleanup(self) -> None:
        """Clean up state machine resources."""
        self.timers.cleanup()
        self.send_queue.clear()
        self.received_frames.clear()
        logger.debug("State machine cleaned up")

    def get_sequence_info(self) -> Dict[str, int]:
        """Get sequence number information.
        
        Returns:
            Dictionary with sequence numbers
        """
        return {
            'send_sequence': self.vs,
            'receive_sequence': self.vr,
            'ack_sequence': self.va,
            'modulo': self.modulo,
            'window_size': self.k_window
        }

    def get_state_info(self) -> Dict[str, Union[str, int]]:
        """Get state machine information.
        
        Returns:
            Dictionary with state information
        """
        return {
            'state': self.state.name,
            'callsign': self.my_call,
            'modulo': self.modulo.name,
            'send_queue_length': len(self.send_queue),
            'received_frames': len(self.received_frames),
            'send_window_size': self.k_window,
            'unacknowledged_frames': len([e for e in self.send_queue if e.ns >= self.va])
        }

    def __repr__(self) -> str:
        return (f"AX25StateMachine(state={self.state.name}, "
                f"vs={self.vs}, vr={self.vr}, va={self.va}, "
                f"queued={len(self.send_queue)})")

