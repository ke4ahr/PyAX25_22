# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.timers.py

AX.25 v2.2 compliant timer implementation.

Provides:
- T1: Acknowledgment timer with adaptive SRTT (Smoothed Round-Trip Time)
- T3: Idle channel probe timer
- Support for both synchronous (threading) and asynchronous (asyncio) operation

Fully compliant with AX.25 v2.2 Section 4.3.3.5 (Timer procedures).
"""

from __future__ import annotations

import time
import threading
import asyncio
from typing import Callable, Optional, Coroutine
import logging

from .config import AX25Config
from .exceptions import TimeoutError

logger = logging.getLogger(__name__)

class AX25Timers:
    """
    Comprehensive timer manager for AX.25 connections.

    Implements T1 (adaptive) and T3 (idle probe) with both sync and async interfaces.
    Uses Jacobson/Karels algorithm for T1 SRTT calculation.
    """

    def __init__(self, config: AX25Config):
        """
        Initialize timers with configuration.

        Args:
            config: AX.25 configuration containing base timer values
        """
        self.config = config

        # Current timer values (updated dynamically)
        self.t1_current: float = config.t1_timeout
        self.t3_current: float = config.t3_timeout

        # SRTT algorithm variables (T1 adaptive)
        self.srtt: float = config.t1_timeout  # Smoothed RTT
        self.rttvar: float = config.t1_timeout / 2  # RTT variance
        self.rto: float = self.srtt + max(1.0, 4 * self.rttvar)  # Retransmission timeout

        # Active timer handles
        self._t1_thread_timer: Optional[threading.Timer] = None
        self._t3_thread_timer: Optional[threading.Timer] = None
        self._t1_async_task: Optional[asyncio.Task] = None
        self._t3_async_task: Optional[asyncio.Task] = None

        # Timestamp of last acknowledgment
        self._last_ack_time: float = time.time()

        logger.info(
            f"Timers initialized: T1 base={config.t1_timeout}s, T3={config.t3_timeout}s"
        )

    # Synchronous timer operations

    def start_t1_sync(self, callback: Callable[[], None]) -> None:
        """
        Start T1 acknowledgment timer (synchronous/threading).

        Args:
            callback: Function to call on timeout
        """
        self.stop_t1_sync()
        try:
            self._t1_thread_timer = threading.Timer(self.rto, self._t1_timeout_handler(callback))
            self._t1_thread_timer.daemon = True
            self._t1_thread_timer.start()
            logger.debug(f"T1 started (sync): {self.rto:.2f}s")
        except Exception as e:
            logger.error(f"Failed to start T1 timer: {e}")
            raise TimeoutError(f"Failed to start T1 timer: {e}")

    def stop_t1_sync(self) -> None:
        """Stop T1 timer if running."""
        if self._t1_thread_timer:
            try:
                self._t1_thread_timer.cancel()
                self._t1_thread_timer = None
                logger.debug("T1 stopped (sync)")
            except Exception as e:
                logger.error(f"Error stopping T1 timer: {e}")

    def start_t3_sync(self, callback: Callable[[], None]) -> None:
        """
        Start T3 idle probe timer (synchronous).

        Args:
            callback: Function to call on timeout
        """
        self.stop_t3_sync()
        try:
            self._t3_thread_timer = threading.Timer(self.t3_current, self._t3_timeout_handler(callback))
            self._t3_thread_timer.daemon = True
            self._t3_thread_timer.start()
            logger.debug(f"T3 started (sync): {self.t3_current}s")
        except Exception as e:
            logger.error(f"Failed to start T3 timer: {e}")
            raise TimeoutError(f"Failed to start T3 timer: {e}")

    def stop_t3_sync(self) -> None:
        """Stop T3 timer if running."""
        if self._t3_thread_timer:
            try:
                self._t3_thread_timer.cancel()
                self._t3_thread_timer = None
                logger.debug("T3 stopped (sync)")
            except Exception as e:
                logger.error(f"Error stopping T3 timer: {e}")

    # Asynchronous timer operations

    async def start_t1_async(self, callback: Callable[[], Coroutine[None, None, None]]) -> None:
        """
        Start T1 acknowledgment timer (asyncio).

        Args:
            callback: Async function to call on timeout
        """
        self.stop_t1_async()
        try:
            self._t1_async_task = asyncio.create_task(self._t1_async_wait(callback))
            logger.debug(f"T1 started (async): {self.rto:.2f}s")
        except Exception as e:
            logger.error(f"Failed to start async T1 timer: {e}")
            raise TimeoutError(f"Failed to start async T1 timer: {e}")

    async def stop_t1_async(self) -> None:
        """Cancel T1 async task if running."""
        if self._t1_async_task:
            try:
                self._t1_async_task.cancel()
                await self._t1_async_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error stopping async T1 timer: {e}")
            finally:
                self._t1_async_task = None
                logger.debug("T1 stopped (async)")

    async def start_t3_async(self, callback: Callable[[], Coroutine[None, None, None]]) -> None:
        """
        Start T3 idle probe timer (asyncio).

        Args:
            callback: Async function to call on timeout
        """
        self.stop_t3_async()
        try:
            self._t3_async_task = asyncio.create_task(self._t3_async_wait(callback))
            logger.debug(f"T3 started (async): {self.t3_current}s")
        except Exception as e:
            logger.error(f"Failed to start async T3 timer: {e}")
            raise TimeoutError(f"Failed to start async T3 timer: {e}")

    async def stop_t3_async(self) -> None:
        """Cancel T3 async task if running."""
        if self._t3_async_task:
            try:
                self._t3_async_task.cancel()
                await self._t3_async_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error stopping async T3 timer: {e}")
            finally:
                self._t3_async_task = None
                logger.debug("T3 stopped (async)")

    # Internal handlers

    def _t1_timeout_handler(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Create thread-safe T1 timeout handler."""
        def handler() -> None:
            try:
                logger.warning(f"T1 timeout after {self.rto:.2f}s")
                callback()
            except Exception as e:
                logger.error(f"T1 timeout handler failed: {e}")
                raise TimeoutError("T1 acknowledgment timeout")
        return handler

    def _t3_timeout_handler(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Create thread-safe T3 timeout handler."""
        def handler() -> None:
            try:
                logger.warning(f"T3 idle timeout after {self.t3_current}s")
                callback()
            except Exception as e:
                logger.error(f"T3 timeout handler failed: {e}")
        return handler

    async def _t1_async_wait(self, callback: Callable[[], Coroutine[None, None, None]]) -> None:
        """Async implementation of T1 timeout."""
        try:
            await asyncio.sleep(self.rto)
            logger.warning(f"Async T1 timeout after {self.rto:.2f}s")
            await callback()
            raise TimeoutError("Async T1 acknowledgment timeout")
        except asyncio.CancelledError:
            logger.debug("Async T1 cancelled")
        except Exception as e:
            logger.error(f"Async T1 wait failed: {e}")
            raise TimeoutError(f"Async T1 wait failed: {e}")

    async def _t3_async_wait(self, callback: Callable[[], Coroutine[None, None, None]]) -> None:
        """Async implementation of T3 timeout."""
        try:
            await asyncio.sleep(self.t3_current)
            logger.warning(f"Async T3 idle timeout after {self.t3_current}s")
            await callback()
        except asyncio.CancelledError:
            logger.debug("Async T3 cancelled")
        except Exception as e:
            logger.error(f"Async T3 wait failed: {e}")

    # RTT measurement and adaptation

    def record_acknowledgment(self) -> None:
        """
        Record receipt of acknowledgment to update SRTT.

        Call when an RR/RNR/REJ/SREJ with N(R) acknowledging new frames arrives.
        """
        measured_rtt = time.time() - self._last_ack_time

        # Jacobson/Karels algorithm
        delta = measured_rtt - self.srtt
        self.srtt += 0.125 * delta
        self.rttvar += 0.25 * (abs(delta) - self.rttvar)
        self.rto = self.srtt + max(1.0, 4 * self.rttvar)

        # Apply bounds to prevent extreme values
        self.rto = max(1.0, min(self.rto, 60.0))  # 1s min, 60s max

        self._last_ack_time = time.time()
        logger.debug(f"RTT recorded: {measured_rtt:.3f}s → SRTT={self.srtt:.3f}s, RTO={self.rto:.3f}s")

    def reset(self) -> None:
        """Reset all timers and state."""
        self.stop_t1_sync()
        self.stop_t3_sync()

        # Stop async timers if they're running
        try:
            asyncio.run(self.stop_t1_async())
        except RuntimeError:
            # No event loop running
            pass

        try:
            asyncio.run(self.stop_t3_async())
        except RuntimeError:
            # No event loop running
            pass

        # Reset SRTT algorithm
        self.srtt = self.config.t1_timeout
        self.rttvar = self.config.t1_timeout / 2
        self.rto = self.srtt + max(1.0, 4 * self.rttvar)

        logger.info("Timers reset to initial state")

    def update_t1_timeout(self, new_timeout: float) -> None:
        """
        Update T1 timeout value.

        Args:
            new_timeout: New T1 timeout in seconds
        """
        if not (0.1 <= new_timeout <= 60.0):
            raise ValueError("T1 timeout must be between 0.1 and 60.0 seconds")

        self.t1_current = new_timeout
        self.rto = new_timeout  # Reset RTO to new base value
        logger.info(f"T1 timeout updated to {new_timeout}s")

    def update_t3_timeout(self, new_timeout: float) -> None:
        """
        Update T3 timeout value.

        Args:
            new_timeout: New T3 timeout in seconds
        """
        if not (10.0 <= new_timeout <= 3600.0):
            raise ValueError("T3 timeout must be between 10.0 and 3600.0 seconds")

        self.t3_current = new_timeout
        logger.info(f"T3 timeout updated to {new_timeout}s")

    def get_timer_status(self) -> dict:
        """
        Get current timer status.

        Returns:
            Dictionary containing timer status information
        """
        return {
            't1_running': self._t1_thread_timer is not None or self._t1_async_task is not None,
            't3_running': self._t3_thread_timer is not None or self._t3_async_task is not None,
            't1_current': self.t1_current,
            't3_current': self.t3_current,
            'srtt': self.srtt,
            'rttvar': self.rttvar,
            'rto': self.rto,
            'last_ack_time': self._last_ack_time
        }
