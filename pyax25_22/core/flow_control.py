# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
Flow Control Implementation

Implements window-based flow control and selective reject handling.

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import logging
import threading
import time
from enum import IntEnum
from typing import Optional, Callable, Dict, List, Any, Union
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class WindowMode(IntEnum):
    """Window modulus"""
    MOD8 = 8
    MOD128 = 128

class FlowController:
    """
    Flow control implementation with window management and selective reject.
    
    Features:
    - Window-based flow control
    - Selective reject (SREJ) handling
    - Retransmission management
    - Buffer overflow prevention
    """
    
    def __init__(self, 
                 send_callback: Callable[[bytes], None],
                 window_size: int = 4,
                 mode: WindowMode = WindowMode.MOD8):
        """
        Initialize flow controller.
        
        Args:
            send_callback: Function to send frames
            window_size: Window size (1-7 for MOD8, 1-127 for MOD128)
            mode: Window modulus
            
        Raises:
            ValueError: If window size is invalid for the mode
            TypeError: If send_callback is not callable
        """
        if not 1 <= window_size <= (mode - 1):
            raise ValueError(f"Window size must be 1-{mode-1} for {mode.name}")
        if not callable(send_callback):
            raise TypeError("send_callback must be callable")
            
        self.send_callback = send_callback
        self.window_size = window_size
        self.mode = mode
        
        # State variables
        self.vs = 0      # Send sequence number
        self.vr = 0      # Receive sequence number
        self.va = 0      # Acknowledged sequence number
        
        # Window management
        self.send_window: deque = deque(maxlen=window_size)
        self.recv_buffer: Dict[int, bytes] = {}
        self.out_of_order_frames: Dict[int, bytes] = {}
        
        # Retransmission management
        self.sent_frames: Dict[int, Dict[str, Any]] = {}
        self.retransmission_queue: List[int] = []
        
        # Selective reject management
        self.srej_pending = False
        self.srej_queue: Dict[int, bytes] = {}
        self.srej_timer: Optional[threading.Timer] = None
        
        # Locking
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'frames_sent': 0,
            'frames_received': 0,
            'retransmissions': 0,
            'srej_requests': 0,
            'srej_responses': 0,
            'window_violations': 0,
            'sequence_errors': 0,
            'buffer_overflows': 0,
            'acknowledgments_sent': 0,
            'acknowledgments_received': 0,
            'duplicate_frames': 0
        }
        
        # Configuration
        self._config = {
            'max_retries': 3,
            'srej_timeout': 2.0,
            'window_check_interval': 0.1
        }
        
        logger.info(f"Initialized flow controller: window={window_size}, mode={mode.name}")

    def can_send(self) -> bool:
        """Check if we can send more frames.
        
        Returns:
            True if window allows sending, False otherwise
        """
        with self._lock:
            unacked_frames = len([entry for entry in self.send_window if entry['ns'] >= self.va])
            return unacked_frames < self.window_size

    def send_frame(self, frame_data: bytes, ns: int) -> int:
        """
        Send frame with flow control.
        
        Args:
            frame_data: Frame payload
            ns: Sequence number
            
        Returns:
            Sequence number assigned to frame
            
        Raises:
            RuntimeError: If send window is full
            ValueError: If sequence number is invalid
        """
        if not isinstance(frame_data, bytes):
            raise TypeError("Frame data must be bytes")
        if not 0 <= ns < self.mode:
            raise ValueError(f"Invalid sequence number: {ns}, must be 0-{self.mode-1}")
            
        with self._lock:
            if not self.can_send():
                self._stats['window_violations'] += 1
                logger.warning(f"Send window full, cannot send NS={ns}")
                raise RuntimeError("Send window full")
            
            # Create frame entry
            entry = {
                'ns': ns,
                'data': frame_data,
                'timestamp': time.time(),
                'retries': 0,
                'acknowledged': False
            }
            
            # Add to send window
            self.send_window.append(entry)
            self.sent_frames[ns] = entry
            
            # Send frame
            try:
                self.send_callback(frame_data)
                self._stats['frames_sent'] += 1
                logger.debug(f"Sent frame NS={ns}")
                return ns
            except Exception as e:
                logger.error(f"Send failed for NS={ns}: {e}")
                self._stats['errors'] += 1
                raise

    def handle_ack(self, nr: int) -> int:
        """
        Handle acknowledgment and update send window.
        
        Args:
            nr: Acknowledged sequence number (NR)
            
        Returns:
            Number of frames acknowledged
            
        Raises:
            ValueError: If sequence number is invalid
        """
        if not 0 <= nr < self.mode:
            raise ValueError(f"Invalid sequence number: {nr}, must be 0-{self.mode-1}")
            
        with self._lock:
            acknowledged = 0
            
            # Remove acknowledged frames from window
            while self.send_window and self.send_window[0]['ns'] < nr:
                entry = self.send_window.popleft()
                entry['acknowledged'] = True
                acknowledged += 1
                
                # Update VA
                self.va = (self.va + 1) % self.mode
                
                # Remove from sent frames
                if entry['ns'] in self.sent_frames:
                    del self.sent_frames[entry['ns']]
            
            # Clean up retransmission queue
            self.retransmission_queue = [ns for ns in self.retransmission_queue if ns >= nr]
            
            if acknowledged > 0:
                self._stats['acknowledgments_received'] += acknowledged
                logger.debug(f"Acknowledged {acknowledged} frames, NR={nr}, VA={self.va}")
                
            return acknowledged

    def handle_rej(self, nr: int) -> None:
        """
        Handle reject (REJ) frame - retransmit all frames from NR onwards.
        
        Args:
            nr: Sequence number to start retransmission from
            
        Raises:
            ValueError: If sequence number is invalid
        """
        if not 0 <= nr < self.mode:
            raise ValueError(f"Invalid sequence number: {nr}, must be 0-{self.mode-1}")
            
        with self._lock:
            self._stats['retransmissions'] += 1
            logger.warning(f"Handling REJ for NR={nr}")
            
            # Find frames to retransmit
            frames_to_resend = [entry for entry in self.send_window if entry['ns'] >= nr]
            
            for entry in frames_to_resend:
                entry['retries'] += 1
                if entry['retries'] > self._config['max_retries']:
                    logger.error(f"Max retries exceeded for frame NS={entry['ns']}")
                    continue
                    
                try:
                    self.send_callback(entry['data'])
                    logger.debug(f"Retransmitted frame NS={entry['ns']}")
                except Exception as e:
                    logger.error(f"Retransmission failed for NS={entry['ns']}: {e}")

    def handle_srej(self, nr: int) -> None:
        """
        Handle selective reject (SREJ) frame - retransmit specific frame.
        
        Args:
            nr: Sequence number to selectively retransmit
            
        Raises:
            ValueError: If sequence number is invalid
        """
        if not 0 <= nr < self.mode:
            raise ValueError(f"Invalid sequence number: {nr}, must be 0-{self.mode-1}")
            
        with self._lock:
            self._stats['srej_requests'] += 1
            logger.warning(f"Handling SREJ for NR={nr}")
            
            # Find specific frame to retransmit
            if nr in self.sent_frames:
                entry = self.sent_frames[nr]
                entry['retries'] += 1
                
                if entry['retries'] > self._config['max_retries']:
                    logger.error(f"Max retries exceeded for SREJ frame NS={nr}")
                    return
                
                try:
                    self.send_callback(entry['data'])
                    self._stats['srej_responses'] += 1
                    logger.debug(f"Retransmitted SREJ frame NS={nr}")
                except Exception as e:
                    logger.error(f"SREJ retransmission failed for NS={nr}: {e}")

    def receive_frame(self, frame_data: bytes, ns: int) -> Optional[int]:
        """
        Handle received frame and manage receive window.
        
        Args:
            frame_data: Frame payload
            ns: Sequence number
            
        Returns:
            Next expected sequence number if frame accepted, None if rejected
            
        Raises:
            ValueError: If sequence number is invalid
        """
        if not isinstance(frame_data, bytes):
            raise TypeError("Frame data must be bytes")
        if not 0 <= ns < self.mode:
            raise ValueError(f"Invalid sequence number: {ns}, must be 0-{self.mode-1}")
            
        with self._lock:
            # Check for duplicate frame
            if ns in self.out_of_order_frames and self.out_of_order_frames[ns] == frame_data:
                self._stats['duplicate_frames'] += 1
                logger.debug(f"Duplicate frame received NS={ns}")
                return self.vr
            
            # Check sequence number validity
            if not self._is_valid_sequence(ns):
                self._stats['sequence_errors'] += 1
                logger.warning(f"Invalid sequence number: NS={ns}, VR={self.vr}")
                return None
            
            # Check receive window
            if self._is_in_receive_window(ns):
                if ns == self.vr:
                    # In-order frame
                    self.vr = (self.vr + 1) % self.mode
                    
                    # Process any buffered out-of-order frames
                    self._process_buffered_frames()
                    
                    self._stats['frames_received'] += 1
                    logger.debug(f"Accepted in-order frame NS={ns}, VR={self.vr}")
                    return self.vr
                else:
                    # Out-of-order frame
                    if len(self.out_of_order_frames) >= self.window_size:
                        self._stats['buffer_overflows'] += 1
                        logger.warning(f"Receive buffer full, dropping frame NS={ns}")
                        return None
                        
                    self.out_of_order_frames[ns] = frame_data
                    self._stats['frames_received'] += 1
                    logger.debug(f"Stored out-of-order frame NS={ns}")
                    return self.vr
            else:
                # Out of window
                self._stats['window_violations'] += 1
                logger.warning(f"Frame out of receive window: NS={ns}, VR={self.vr}")
                return None

    def _is_valid_sequence(self, ns: int) -> bool:
        """Check if sequence number is valid.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if valid, False otherwise
        """
        return 0 <= ns < self.mode

    def _is_in_receive_window(self, ns: int) -> bool:
        """Check if sequence number is in receive window.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if in window, False otherwise
        """
        window_size = self.window_size
        diff = (ns - self.vr) % self.mode
        return 0 <= diff < window_size

    def _process_buffered_frames(self) -> None:
        """Process any buffered out-of-order frames that can now be delivered."""
        while True:
            next_ns = self.vr
            if next_ns in self.out_of_order_frames:
                frame_data = self.out_of_order_frames.pop(next_ns)
                self.vr = (self.vr + 1) % self.mode
                # Frame would be delivered to application layer
                logger.debug(f"Delivered buffered frame NS={next_ns}")
            else:
                break

    def send_ack(self, nr: int) -> None:
        """
        Send acknowledgment for received frames.
        
        Args:
            nr: Sequence number to acknowledge
            
        Raises:
            ValueError: If sequence number is invalid
        """
        if not 0 <= nr < self.mode:
            raise ValueError(f"Invalid sequence number: {nr}, must be 0-{self.mode-1}")
            
        with self._lock:
            # This would typically send an RR frame
            # Implementation depends on the transport layer
            self._stats['acknowledgments_sent'] += 1
            logger.debug(f"Sent acknowledgment NR={nr}")

    def get_window_status(self) -> Dict[str, Union[int, bool]]:
        """Get current window status.
        
        Returns:
            Dictionary with window status information
        """
        with self._lock:
            unacked_count = len([entry for entry in self.send_window if entry['ns'] >= self.va])
            return {
                'vs': self.vs,
                'vr': self.vr,
                'va': self.va,
                'window_size': self.window_size,
                'unacked_frames': unacked_count,
                'window_full': unacked_count >= self.window_size,
                'buffer_size': len(self.out_of_order_frames),
                'mode': self.mode.name
            }

    def get_stats(self) -> Dict[str, int]:
        """Get flow control statistics.
        
        Returns:
            Dictionary with flow control statistics
        """
        with self._lock:
            return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset flow control statistics."""
        with self._lock:
            self._stats = {k: 0 for k in self._stats}
            logger.debug("Flow control statistics reset")

    def set_config(self, key: str, value: Any) -> None:
        """
        Set flow control configuration.
        
        Args:
            key: Configuration parameter name
            value: Configuration parameter value
        """
        with self._lock:
            if key in self._config:
                self._config[key] = value
                logger.debug(f"Flow control config updated: {key} = {value}")
            else:
                logger.warning(f"Unknown flow control config parameter: {key}")

    def get_config(self) -> Dict[str, Any]:
        """Get current flow control configuration.
        
        Returns:
            Dictionary with current configuration
        """
        with self._lock:
            return self._config.copy()

    def cleanup(self) -> None:
        """Clean up flow controller resources."""
        with self._lock:
            self.send_window.clear()
            self.recv_buffer.clear()
            self.out_of_order_frames.clear()
            self.sent_frames.clear()
            self.retransmission_queue.clear()
            
            if self.srej_timer:
                self.srej_timer.cancel()
                self.srej_timer = None
                
        logger.debug("Flow controller cleaned up")

    def __repr__(self) -> str:
        status = self.get_window_status()
        return (f"FlowController(window={status['window_size']}, "
                f"vs={status['vs']}, vr={status['vr']}, va={status['va']}, "
                f"unacked={status['unacked_frames']}, mode={status['mode']})")

class SREJManager:
    """Selective Reject (SREJ) handler"""
    
    def __init__(self, modulo: WindowMode = WindowMode.MOD8):
        """Initialize SREJ manager.
        
        Args:
            modulo: Window modulus (MOD8 or MOD128)
            
        Raises:
            ValueError: If modulo is invalid
        """
        if modulo not in [WindowMode.MOD8, WindowMode.MOD128]:
            raise ValueError(f"Invalid modulo: {modulo}")
            
        self.modulo = modulo
        self._srej_queue: Dict[int, bytes] = {}
        self._expected_ns = 0
        self._lock = threading.RLock()
        
        logger.debug(f"Initialized SREJ manager with modulo {modulo}")

    def add_frame(self, ns: int, frame: bytes) -> None:
        """Store out-of-sequence frame.
        
        Args:
            ns: Sequence number
            frame: Frame data to store
            
        Raises:
            ValueError: If sequence number is invalid
            TypeError: If frame is not bytes
        """
        if not 0 <= ns < self.modulo:
            raise ValueError(f"Invalid sequence number {ns} for modulo {self.modulo}")
        if not isinstance(frame, bytes):
            raise TypeError("Frame must be bytes")
            
        with self._lock:
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
        with self._lock:
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
            
        Raises:
            ValueError: If sequence number is invalid
        """
        if not 0 <= ns < self.modulo:
            raise ValueError(f"Invalid sequence number {ns} for modulo {self.modulo}")
            
        with self._lock:
            expected = self._expected_ns
            diff = (ns - expected) % self.modulo
            result = 0 < diff <= (self.modulo // 2)
            logger.debug(f"SREJ check: NS={ns}, expected={expected}, needs SREJ={result}")
            return result

    def reset(self) -> None:
        """Reset SREJ manager."""
        with self._lock:
            self._srej_queue.clear()
            self._expected_ns = 0
            logger.debug("SREJ manager reset")

    def get_queue_size(self) -> int:
        """Get current queue size.
        
        Returns:
            Number of frames in SREJ queue
        """
        with self._lock:
            return len(self._srej_queue)

    def __repr__(self) -> str:
        with self._lock:
            return f"SREJManager(modulo={self.modulo.name}, queue_size={len(self._srej_queue)})"

class WindowManager:
    """Window management for flow control"""
    
    def __init__(self, window_size: int, mode: WindowMode):
        """Initialize window manager.
        
        Args:
            window_size: Size of the window
            mode: Window modulus mode
        """
        self.window_size = window_size
        self.mode = mode
        self._lock = threading.RLock()
        
        # Window boundaries
        self.send_base = 0      # Left edge of send window
        self.send_next = 0      # Next sequence number to send
        self.recv_base = 0      # Left edge of receive window
        self.recv_next = 0      # Next expected sequence number
        
    def is_send_window_full(self) -> bool:
        """Check if send window is full.
        
        Returns:
            True if window is full, False otherwise
        """
        with self._lock:
            return self.send_next - self.send_base >= self.window_size

    def is_sequence_in_send_window(self, ns: int) -> bool:
        """Check if sequence number is in send window.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if in window, False otherwise
        """
        with self._lock:
            diff = (ns - self.send_base) % self.mode
            return 0 <= diff < self.window_size

    def is_sequence_in_recv_window(self, ns: int) -> bool:
        """Check if sequence number is in receive window.
        
        Args:
            ns: Sequence number to check
            
        Returns:
            True if in window, False otherwise
        """
        with self._lock:
            diff = (ns - self.recv_base) % self.mode
            return 0 <= diff < self.window_size

    def advance_send_window(self, ns: int) -> None:
        """Advance send window after acknowledgment.
        
        Args:
            ns: Acknowledged sequence number
        """
        with self._lock:
            if self.is_sequence_in_send_window(ns):
                self.send_base = (ns + 1) % self.mode

    def advance_recv_window(self, ns: int) -> None:
        """Advance receive window after frame acceptance.
        
        Args:
            ns: Accepted sequence number
        """
        with self._lock:
            if ns == self.recv_base:
                self.recv_base = (self.recv_base + 1) % self.mode
                self.recv_next = (self.recv_next + 1) % self.mode

    def get_send_window_status(self) -> Dict[str, int]:
        """Get send window status.
        
        Returns:
            Dictionary with send window information
        """
        with self._lock:
            return {
                'base': self.send_base,
                'next': self.send_next,
                'size': self.window_size,
                'mode': self.mode.value
            }

    def get_recv_window_status(self) -> Dict[str, int]:
        """Get receive window status.
        
        Returns:
            Dictionary with receive window information
        """
        with self._lock:
            return {
                'base': self.recv_base,
                'next': self.recv_next,
                'size': self.window_size,
                'mode': self.mode.value
            }

    def __repr__(self) -> str:
        send_status = self.get_send_window_status()
        recv_status = self.get_recv_window_status()
        return (f"WindowManager(send_base={send_status['base']}, "
                f"send_next={send_status['next']}, recv_base={recv_status['base']}, "
                f"recv_next={recv_status['next']})")

# Convenience function for creating flow controllers
def create_flow_controller(
    send_callback: Callable[[bytes], None],
    window_size: int = 4,
    mode: WindowMode = WindowMode.MOD8
) -> FlowController:
    """
    Create a flow controller with the specified parameters.
    
    Args:
        send_callback: Function to send frames
        window_size: Window size
        mode: Window modulus mode
        
    Returns:
        Configured FlowController instance
    """
    return FlowController(send_callback, window_size, mode)

# Default flow controller factory
def create_default_flow_controller(send_callback: Callable[[bytes], None]) -> FlowController:
    """
    Create a default flow controller with standard parameters.
    
    Args:
        send_callback: Function to send frames
        
    Returns:
        Configured FlowController with default settings
    """
    return FlowController(send_callback, window_size=4, mode=WindowMode.MOD8)

# Export public API
__all__ = [
    'WindowMode',
    'FlowController',
    'SREJManager',
    'WindowManager',
    'create_flow_controller',
    'create_default_flow_controller'
]
