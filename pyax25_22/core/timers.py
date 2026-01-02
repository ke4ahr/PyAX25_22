# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
AX25 Timer Management

Provides:
- T1, T2, T3 timer implementation
- Timer state management
- Thread-safe timer operations
- Callback system for timer expiration
- Timer cleanup and resource management

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import threading
import time
import logging
from typing import Optional, Callable, Dict, Any, Union

logger = logging.getLogger(__name__)

class TimerError(Exception):
    """Base exception for timer errors"""
    pass

class TimerNotRunningError(TimerError):
    """Exception raised when trying to stop a timer that isn't running"""
    pass

class TimerAlreadyRunningError(TimerError):
    """Exception raised when trying to start a timer that's already running"""
    pass

class AX25Timer:
    """
    AX.25 Protocol Timer
    
    Implements a single timer with the following features:
    - Configurable timeout duration
    - Automatic expiration callback
    - Thread-safe operations
    - State tracking (running/stopped)
    - Resource cleanup
    """
    
    def __init__(self, 
                 timeout: float,
                 callback: Callable[[], None],
                 name: str = "timer"):
        """Initialize AX.25 timer.
        
        Args:
            timeout: Timeout duration in seconds
            callback: Function to call when timer expires
            name: Human-readable timer name for logging
            
        Raises:
            ValueError: If timeout is not positive
            TypeError: If callback is not callable
        """
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"Timeout must be positive number, got {timeout}")
        if not callable(callback):
            raise TypeError("Callback must be callable")
        if not isinstance(name, str):
            raise TypeError("Name must be string")
            
        self.timeout = float(timeout)
        self.callback = callback
        self.name = name
        
        # Timer state
        self._task: Optional[threading.Timer] = None
        self._running = False
        self._last_started = 0.0
        self._expires_at = 0.0
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'starts': 0,
            'stops': 0,
            'expirations': 0,
            'cancellations': 0,
            'total_uptime': 0.0
        }
        
        logger.debug(f"Initialized {self.name} timer: timeout={timeout}s")

    def start(self) -> None:
        """Start the timer.
        
        If the timer is already running, this will restart it.
        """
        with self._lock:
            # Cancel existing timer if running
            if self._running:
                self.cancel()
                logger.debug(f"Restarting {self.name} timer")
            else:
                logger.debug(f"Starting {self.name} timer")
            
            # Create and start new timer
            self._task = threading.Timer(self.timeout, self._timer_expired)
            self._task.daemon = True
            self._task.name = f"AX25Timer-{self.name}"
            self._task.start()
            
            # Update state
            self._running = True
            self._last_started = time.time()
            self._expires_at = self._last_started + self.timeout
            
            # Update statistics
            self._stats['starts'] += 1
            
            logger.debug(f"{self.name} timer started, expires at {self._expires_at:.3f}")

    def cancel(self) -> None:
        """Stop the timer if it's running.
        
        This will prevent the callback from being called.
        """
        with self._lock:
            if not self._running:
                logger.debug(f"{self.name} timer not running, ignoring cancel")
                return
                
            if self._task:
                self._task.cancel()
                self._task = None
                
            # Update statistics
            self._stats['cancellations'] += 1
            self._update_uptime()
            
            self._running = False
            self._expires_at = 0.0
            
            logger.debug(f"{self.name} timer cancelled")

    def stop(self) -> None:
        """Stop the timer (alias for cancel)."""
        self.cancel()

    def reset(self) -> None:
        """Reset the timer by stopping and starting it."""
        with self._lock:
            self.cancel()
            self.start()
            logger.debug(f"{self.name} timer reset")

    def is_running(self) -> bool:
        """Check if the timer is currently running.
        
        Returns:
            True if timer is running, False otherwise
        """
        with self._lock:
            return self._running

    def get_remaining_time(self) -> float:
        """Get remaining time until expiration.
        
        Returns:
            Remaining time in seconds, 0.0 if not running
        """
        with self._lock:
            if not self._running:
                return 0.0
            remaining = self._expires_at - time.time()
            return max(0.0, remaining)

    def get_elapsed_time(self) -> float:
        """Get time elapsed since timer was started.
        
        Returns:
            Elapsed time in seconds, 0.0 if not running
        """
        with self._lock:
            if not self._running:
                return 0.0
            return time.time() - self._last_started

    def _timer_expired(self) -> None:
        """Internal method called when timer expires."""
        with self._lock:
            self._running = False
            self._update_uptime()
            self._expires_at = 0.0
            self._stats['expirations'] += 1
            
            logger.debug(f"{self.name} timer expired")
            
        # Call callback outside of lock to avoid deadlocks
        try:
            self.callback()
        except Exception as e:
            logger.error(f"Timer {self.name} callback failed: {e}")
            # Don't re-raise - timer expiration should not propagate errors

    def _update_uptime(self) -> None:
        """Update total uptime statistics."""
        if self._last_started > 0:
            elapsed = time.time() - self._last_started
            self._stats['total_uptime'] += elapsed

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get timer statistics.
        
        Returns:
            Dictionary containing timer statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats['running'] = self._running
            stats['timeout'] = self.timeout
            stats['expires_at'] = self._expires_at
            stats['last_started'] = self._last_started
            stats['remaining_time'] = self.get_remaining_time()
            stats['elapsed_time'] = self.get_elapsed_time()
            return stats

    def cleanup(self) -> None:
        """Clean up timer resources."""
        self.cancel()
        logger.debug(f"{self.name} timer cleaned up")

    def __enter__(self):
        """Context manager entry - start timer."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop timer."""
        self.cancel()

    def __repr__(self) -> str:
        return (f"AX25Timer(name='{self.name}', timeout={self.timeout}, "
                f"running={self._running}, remaining={self.get_remaining_time():.3f})")

class TimerManager:
    """
    AX.25 Protocol Timer Manager
    
    Manages multiple timers (T1, T2, T3) with the following features:
    - Centralized timer management
    - State coordination between timers
    - Resource cleanup
    - Statistics tracking
    - Thread-safe operations
    """
    
    def __init__(self,
                 t1_timeout: float = 10.0,
                 t2_timeout: float = 2.0, 
                 t3_timeout: float = 30.0):
        """Initialize timer manager.
        
        Args:
            t1_timeout: T1 retransmission timeout (seconds)
            t2_timeout: T2 acknowledgment timeout (seconds)
            t3_timeout: T3 inactivity timeout (seconds)
            
        Raises:
            ValueError: If any timeout is not positive
        """
        if not all(isinstance(t, (int, float)) and t > 0 for t in [t1_timeout, t2_timeout, t3_timeout]):
            raise ValueError("All timeouts must be positive numbers")
            
        # Timer timeouts
        self.t1_timeout = float(t1_timeout)
        self.t2_timeout = float(t2_timeout)
        self.t3_timeout = float(t3_timeout)
        
        # Timer states
        self.t1_running = False
        self.t2_running = False
        self.t3_running = False
        
        # Timer objects
        self.t1_timer: Optional[AX25Timer] = None
        self.t2_timer: Optional[AX25Timer] = None
        self.t3_timer: Optional[AX25Timer] = None
        
        # Callbacks
        self.on_t1_expired: Optional[Callable[[], None]] = None
        self.on_t2_expired: Optional[Callable[[], None]] = None
        self.on_t3_expired: Optional[Callable[[], None]] = None
        
        # Locking
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            't1_starts': 0,
            't1_stops': 0,
            't1_expirations': 0,
            't1_cancellations': 0,
            't1_total_uptime': 0.0,
            't2_starts': 0,
            't2_stops': 0,
            't2_expirations': 0,
            't2_cancellations': 0,
            't2_total_uptime': 0.0,
            't3_starts': 0,
            't3_stops': 0,
            't3_expirations': 0,
            't3_cancellations': 0,
            't3_total_uptime': 0.0,
            'last_activity': 0.0
        }
        
        logger.info(f"Initialized TimerManager: T1={t1_timeout}s, T2={t2_timeout}s, T3={t3_timeout}s")

    def _t1_expired(self) -> None:
        """Handle T1 retransmission timeout"""
        logger.warning(f"T1 timeout expired ({self.t1_timeout}s)")
        self.t1_running = False
        self._stats['last_activity'] = time.time()
        if self.on_t1_expired:
            try:
                self.on_t1_expired()
            except Exception as e:
                logger.error(f"T1 callback failed: {e}")

    def _t2_expired(self) -> None:
        """Handle T2 acknowledgment timeout"""
        logger.warning(f"T2 timeout expired ({self.t2_timeout}s)")
        self.t2_running = False
        self._stats['last_activity'] = time.time()
        if self.on_t2_expired:
            try:
                self.on_t2_expired()
            except Exception as e:
                logger.error(f"T2 callback failed: {e}")

    def _t3_expired(self) -> None:
        """Handle T3 inactivity timeout"""
        logger.warning(f"T3 timeout expired ({self.t3_timeout}s)")
        self.t3_running = False
        self._stats['last_activity'] = time.time()
        if self.on_t3_expired:
            try:
                self.on_t3_expired()
            except Exception as e:
                logger.error(f"T3 callback failed: {e}")

    def start_t1(self) -> None:
        """Start T1 retransmission timer"""
        with self._lock:
            if self.t1_running:
                logger.debug("T1 already running, resetting")
                self.stop_t1()
            
            self.t1_timer = AX25Timer(
                timeout=self.t1_timeout,
                callback=self._t1_expired,
                name="T1"
            )
            self.t1_timer.start()
            self.t1_running = True
            
            self._stats['t1_starts'] += 1
            self._stats['last_activity'] = time.time()
            
            logger.debug(f"T1 started ({self.t1_timeout}s)")

    def stop_t1(self) -> None:
        """Stop T1 retransmission timer"""
        with self._lock:
            if not self.t1_running:
                logger.debug("T1 not running, ignoring stop request")
                return
                
            if self.t1_timer:
                self.t1_timer.cancel()
                self.t1_timer = None
                
            self.t1_running = False
            self._stats['t1_stops'] += 1
            self._stats['last_activity'] = time.time()
            
            logger.debug("T1 stopped")

    def start_t2(self) -> None:
        """Start T2 acknowledgment timer"""
        with self._lock:
            if self.t2_running:
                logger.debug("T2 already running, resetting")
                self.stop_t2()
            
            self.t2_timer = AX25Timer(
                timeout=self.t2_timeout,
                callback=self._t2_expired,
                name="T2"
            )
            self.t2_timer.start()
            self.t2_running = True
            
            self._stats['t2_starts'] += 1
            self._stats['last_activity'] = time.time()
            
            logger.debug(f"T2 started ({self.t2_timeout}s)")

    def stop_t2(self) -> None:
        """Stop T2 acknowledgment timer"""
        with self._lock:
            if not self.t2_running:
                logger.debug("T2 not running, ignoring stop request")
                return
                
            if self.t2_timer:
                self.t2_timer.cancel()
                self.t2_timer = None
                
            self.t2_running = False
            self._stats['t2_stops'] += 1
            self._stats['last_activity'] = time.time()
            
            logger.debug("T2 stopped")

    def start_t3(self) -> None:
        """Start T3 inactivity timer"""
        with self._lock:
            if self.t3_running:
                logger.debug("T3 already running, resetting")
                self.stop_t3()
            
            self.t3_timer = AX25Timer(
                timeout=self.t3_timeout,
                callback=self._t3_expired,
                name="T3"
            )
            self.t3_timer.start()
            self.t3_running = True
            
            self._stats['t3_starts'] += 1
            self._stats['last_activity'] = time.time()
            
            logger.debug(f"T3 started ({self.t3_timeout}s)")

    def stop_t3(self) -> None:
        """Stop T3 inactivity timer"""
        with self._lock:
            if not self.t3_running:
                logger.debug("T3 not running, ignoring stop request")
                return
                
            if self.t3_timer:
                self.t3_timer.cancel()
                self.t3_timer = None
                
            self.t3_running = False
            self._stats['t3_stops'] += 1
            self._stats['last_activity'] = time.time()
            
            logger.debug("T3 stopped")

    def reset_t3(self) -> None:
        """Reset T3 on any frame activity"""
        with self._lock:
            # Stop T3
            self.stop_t3()
            # Start T3
            self.start_t3()
            logger.debug("T3 reset")

    def notify_activity(self) -> None:
        """Reset T3 on any frame activity"""
        self.reset_t3()

    def get_t1_remaining(self) -> float:
        """Get T1 remaining time.
        
        Returns:
            Remaining time in seconds, 0.0 if not running
        """
        with self._lock:
            if self.t1_timer:
                return self.t1_timer.get_remaining_time()
            return 0.0

    def get_t2_remaining(self) -> float:
        """Get T2 remaining time.
        
        Returns:
            Remaining time in seconds, 0.0 if not running
        """
        with self._lock:
            if self.t2_timer:
                return self.t2_timer.get_remaining_time()
            return 0.0

    def get_t3_remaining(self) -> float:
        """Get T3 remaining time.
        
        Returns:
            Remaining time in seconds, 0.0 if not running
        """
        with self._lock:
            if self.t3_timer:
                return self.t3_timer.get_remaining_time()
            return 0.0

    def are_any_running(self) -> bool:
        """Check if any timers are running.
        
        Returns:
            True if any timer is running, False otherwise
        """
        with self._lock:
            return self.t1_running or self.t2_running or self.t3_running

    def get_status(self) -> Dict[str, Union[bool, float, Dict]]:
        """Get overall timer status.
        
        Returns:
            Dictionary with timer status information
        """
        with self._lock:
            return {
                't1_running': self.t1_running,
                't2_running': self.t2_running,
                't3_running': self.t3_running,
                't1_remaining': self.get_t1_remaining(),
                't2_remaining': self.get_t2_remaining(),
                't3_remaining': self.get_t3_remaining(),
                'any_running': self.are_any_running(),
                'timeouts': {
                    't1': self.t1_timeout,
                    't2': self.t2_timeout,
                    't3': self.t3_timeout
                },
                'last_activity': self._stats['last_activity']
            }

    def get_all_stats(self) -> Dict[str, Union[int, float]]:
        """Get all timer statistics.
        
        Returns:
            Dictionary with comprehensive timer statistics
        """
        with self._lock:
            stats = self._stats.copy()
            
            # Add individual timer stats if available
            if self.t1_timer:
                t1_stats = self.t1_timer.get_stats()
                stats['t1_running'] = t1_stats['running']
                stats['t1_last_started'] = t1_stats['last_started']
                
            if self.t2_timer:
                t2_stats = self.t2_timer.get_stats()
                stats['t2_running'] = t2_stats['running']
                stats['t2_last_started'] = t2_stats['last_started']
                
            if self.t3_timer:
                t3_stats = self.t3_timer.get_stats()
                stats['t3_running'] = t3_stats['running']
                stats['t3_last_started'] = t3_stats['last_started']
            
            return stats

    def cleanup(self) -> None:
        """Clean up all timers"""
        with self._lock:
            self.stop_t1()
            self.stop_t2()
            self.stop_t3()
            
            # Clear timer references
            self.t1_timer = None
            self.t2_timer = None
            self.t3_timer = None
            
            logger.debug("All timers cleaned up")

    def reset_all(self) -> None:
        """Reset all running timers"""
        with self._lock:
            if self.t1_running:
                self.start_t1()
            if self.t2_running:
                self.start_t2()
            if self.t3_running:
                self.start_t3()
            logger.debug("All running timers reset")

    def stop_all(self) -> None:
        """Stop all timers"""
        self.stop_t1()
        self.stop_t2()
        self.stop_t3()
        logger.debug("All timers stopped")

    def __enter__(self):
        """Context manager entry - timers start when used"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup timers"""
        self.cleanup()

    def __repr__(self) -> str:
        status = self.get_status()
        return (f"TimerManager(T1={status['t1_running']}, T2={status['t2_running']}, "
                f"T3={status['t3_running']})")

class TimerGroup:
    """
    Group of related timers with coordinated management.
    
    Useful for managing timer sets that should be started/stopped together.
    """
    
    def __init__(self, name: str = "TimerGroup"):
        """Initialize timer group.
        
        Args:
            name: Group name for logging
        """
        self.name = name
        self.timers: Dict[str, AX25Timer] = {}
        self.callbacks: Dict[str, Callable[[], None]] = {}
        self._lock = threading.RLock()
        
        logger.debug(f"Initialized TimerGroup: {name}")

    def add_timer(self, 
                  name: str, 
                  timeout: float, 
                  callback: Callable[[], None]) -> None:
        """Add a timer to the group.
        
        Args:
            name: Timer name
            timeout: Timeout duration
            callback: Expiration callback
        """
        if not isinstance(name, str):
            raise TypeError("Timer name must be string")
        if name in self.timers:
            raise ValueError(f"Timer '{name}' already exists")
            
        timer = AX25Timer(timeout, callback, f"{self.name}-{name}")
        self.timers[name] = timer
        self.callbacks[name] = callback
        
        logger.debug(f"Added timer '{name}' to group '{self.name}'")

    def start_timer(self, name: str) -> None:
        """Start a specific timer.
        
        Args:
            name: Timer name
        """
        with self._lock:
            if name not in self.timers:
                raise ValueError(f"Timer '{name}' not found")
            self.timers[name].start()

    def stop_timer(self, name: str) -> None:
        """Stop a specific timer.
        
        Args:
            name: Timer name
        """
        with self._lock:
            if name not in self.timers:
                raise ValueError(f"Timer '{name}' not found")
            self.timers[name].stop()

    def start_all(self) -> None:
        """Start all timers in the group."""
        with self._lock:
            for timer in self.timers.values():
                timer.start()

    def stop_all(self) -> None:
        """Stop all timers in the group."""
        with self._lock:
            for timer in self.timers.values():
                timer.stop()

    def reset_all(self) -> None:
        """Reset all timers in the group."""
        with self._lock:
            for timer in self.timers.values():
                timer.reset()

    def get_status(self) -> Dict[str, Dict[str, Union[bool, float]]]:
        """Get status of all timers.
        
        Returns:
            Dictionary with timer status
        """
        with self._lock:
            return {
                name: {
                    'running': timer.is_running(),
                    'remaining': timer.get_remaining_time(),
                    'elapsed': timer.get_elapsed_time()
                }
                for name, timer in self.timers.items()
            }

    def cleanup(self) -> None:
        """Clean up all timers."""
        with self._lock:
            for timer in self.timers.values():
                timer.cleanup()
            self.timers.clear()
            self.callbacks.clear()
            
        logger.debug(f"TimerGroup '{self.name}' cleaned up")

    def __repr__(self) -> str:
        return f"TimerGroup(name='{self.name}', timers={list(self.timers.keys())})"

