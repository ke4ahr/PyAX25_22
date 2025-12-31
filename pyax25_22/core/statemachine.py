# pyax25_22/core/statemachine.py
"""
AX.25 Connected-Mode State Machine

Handles:
- Connection establishment/teardown
Beeing frames
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
from typing import Optional, List, Dict, Callable, Union
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
        self.modulo = modulo
        self._srej_queue: Dict[int, bytes] = {}
        self._expected_ns = 0
        
    def add_frame(self, ns: int, frame: bytes) -> None:
        """Store out-of-sequence frame"""
        if ns not in self._srej_queue:
            self._srej_queue[ns] = frame
            
    def get_next(self) -> Optional[bytes]:
        """Get next in-order frame if available"""
        if self._expected_ns in self._srej_queue:
            frame = self._srej_queue.pop(self._expected_ns)
            self._expected_ns = (self._expected_ns + 1) % self.modulo
            return frame
        return None
        
    def needs_srej(self, ns: int) -> bool:
        """Check if NS requires SREJ"""
        expected = self._expected_ns
        diff = (ns - expected) % self.modulo
        return 0 < diff <= (self.modulo // 2)

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
        self.my_call = my_call
        self.modulo = modulo
        self.t1_timeout = t1_timeout
        self.n2_retries = n2_retries
        self.k_window = k_window
        self.rej_policy = rej_policy
        
        self.state = AX25State.DISCONNECTED
        self.vs = 0      # Send sequence number
        self.vr = 0      # Receive sequence number
        self.va = 0      # Acknowledged sequence number
        
        self.send_queue: List[FrameEntry] = []
        self.srej = SREJManager(modulo)
        self.t1_timer: Optional[threading.Timer] = None
        self.t1_lock = threading.Lock()
        
        # Callbacks
        self.on_frame_received: Optional[Callable[[bytes], None]] = None
        self.on_state_change: Optional[Callable[[AX25State], None]] = None
        
        logger.info(f"Initialized state machine for {my_call}, modulo {modulo}")

    class FrameEntry:
        """Queued frame metadata"""
        def __init__(self, frame: bytes, ns: int, timestamp: float):
            self.frame = frame
            self.ns = ns
            self.timestamp = timestamp
            self.retries = 0
            
    def _change_state(self, new_state: AX25State) -> None:
        """Update state with callback"""
        old_state = self.state
        self.state = new_state
        logger.debug(f"State change: {old_state} -> {new_state}")
        if self.on_state_change:
            try:
                self.on_state_change(new_state)
            except Exception as e:
                logger.error(f"State callback failed: {e}")

    def _start_t1(self) -> None:
        """(Re)start T1 timer"""
        with self.t1_lock:
            if self.t1_timer:
                self.t1_timer.cancel()
            self.t1_timer = threading.Timer(
                self.t1_timeout,
                self._handle_timeout
            )
            self.t1_timer.daemon = True
            self.t1_timer.start()

    def _stop_t1(self) -> None:
        """Stop T1 timer"""
        with self.t1_lock:
            if self.t1_timer:
                self.t1_timer.cancel()
                self.t1_timer = None

    def _handle_timeout(self) -> None:
        """T1 timeout handler"""
        logger.warning(f"T1 timeout in state {self.state}")
        if self.state not in [AX25State.CONNECTED, AX25State.AWAITING_CONNECTION]:
            return
            
        # Resend unacknowledged frames
        resent = False
        for entry in self.send_queue:
            if entry.ns == self.va:
                entry.retries += 1
                if entry.retries > self.n2_retries:
                    logger.error("N2 retries exceeded, disconnecting")
                    self._change_state(AX25State.TIMEOUT)
                    self.disconnect()
                    return
                # Resend frame
                self._send_frame(entry.frame)
                resent = True
                
        if resent:
            self._start_t1()
        else:
            self._stop_t1()

    def _send_frame(self, frame: bytes) -> None:
        """Send raw frame (implement in transport)"""
        raise NotImplementedError("Transport must implement _send_frame")

    def connect(self) -> None:
        """Initiate connection"""
        if self.state != AX25State.DISCONNECTED:
            raise RuntimeError("Already connected or connecting")
            
        self._change_state(AX25State.AWAITING_CONNECTION)
        frame = self._build_sabm()
        self._send_frame(frame)
        self.send_queue.append(
            self.FrameEntry(frame, self.vs, time.time())
        )
        self._start_t1()

    def disconnect(self) -> None:
        """Initiate disconnect"""
        if self.state in [AX25State.DISCONNECTED, AX25State.TIMEOUT]:
            return
            
        self._change_state(AX25State.AWAITING_RELEASE)
        frame = self._build_disc()
        self._send_frame(frame)
        self.send_queue.append(
            self.FrameEntry(frame, self.vs, time.time())
        )
        self._start_t1()

    def send_info(self, data: bytes) -> None:
        """Queue information frame"""
        if self.state != AX25State.CONNECTED:
            raise RuntimeError("Not connected")
            
        if (self.vs - self.va) % self.modulo >= self.k_window:
            raise RuntimeError("Window full")
            
        frame = self._build_i_frame(data, self.vs)
        self._send_frame(frame)
        self.send_queue.append(
            self.FrameEntry(frame, self.vs, time.time())
        )
        self.vs = (self.vs + 1) % self.modulo
        if not self.t1_timer:
            self._start_t1()

    def receive_frame(self, frame: bytes) -> None:
        """Handle received frame"""
        # Parse frame and dispatch...
        # Implement full parsing based on framing.AX25Frame
        pass

    def _build_sabm(self, poll: bool = True) -> bytes:
        """Build SABM frame"""
        control = 0x2F | (poll << 4)
        return self._build_u_frame(control)

    def _build_disc(self, poll: bool = True) -> bytes:
        """Build DISC frame"""
        control = 0x43 | (poll << 4)
        return self._build_u_frame(control)

    def _build_u_frame(self, control: int) -> bytes:
        """Build unnumbered frame"""
        # Implement using framing.AX25Frame
        pass

    def _build_i_frame(self, data: bytes, ns: int) -> bytes:
        """Build information frame"""
        control = (ns << 1) | (self.vr << (5 if self.modulo==AX25Modulo.MOD8 else 13))
        return self._build_iframe(control, data)

    # ... more builder methods

    def __repr__(self) -> str:
        return (f"AX25StateMachine({self.state}, "
                f"vs={self.vs}, vr={self.vr}, va={self.va})")
