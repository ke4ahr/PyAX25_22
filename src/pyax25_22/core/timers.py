# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.timers -- T1 and T3 timers for AX.25 connections.

AX.25 uses two timers to keep connections healthy:

  T1 (Acknowledgment Timer):
    Started whenever we send a frame that needs an ack. If the ack
    does not arrive in time, we re-send. T1 adjusts itself based on
    how long acks normally take (using the Jacobson/Karels algorithm,
    the same math that TCP uses for its retransmit timer).

  T3 (Idle Probe Timer):
    A much longer timer. If the link has been quiet for a while, T3
    fires and we send a probe to make sure the other side is still
    there. If we do not get an answer, we declare the link dead.

Both timers work in two modes:
  - Synchronous (threading): Uses Python threading.Timer. Good for
    simple scripts and when you are not using asyncio.
  - Asynchronous (asyncio): Uses asyncio.create_task. Good for async
    applications.

Compliant with AX.25 v2.2 Section 4.3.3.5 (Timer procedures).
"""

from __future__ import annotations

import time
import threading
import asyncio
from typing import Callable, Coroutine, Optional
import logging

from .config import AX25Config
from .exceptions import TimeoutError

logger = logging.getLogger(__name__)


class AX25Timers:
    """T1 and T3 timer manager for one AX.25 connection.

    Tracks both timers in synchronous and asynchronous modes.
    T1 uses an adaptive timeout (SRTT) based on measured round-trip
    times, so it automatically adjusts to changing network conditions.

    You should only use one mode (sync or async) per connection.
    Mixing sync and async timer calls in the same connection may
    produce unexpected results.

    Attributes:
        config: The AX.25 configuration with base timer values.
        t1_current: Current T1 base timeout (before SRTT adjustment).
        t3_current: Current T3 idle probe timeout.
        srtt: Smoothed Round-Trip Time estimate (seconds).
        rttvar: Round-Trip Time variance estimate (seconds).
        rto: Retransmission TimeOut -- actual T1 delay used.

    Example::

        timers = AX25Timers(config)
        timers.start_t1_sync(callback=on_t1_timeout)
        # ... frames are exchanged ...
        timers.record_acknowledgment()  # updates SRTT
        timers.stop_t1_sync()
    """

    def __init__(self, config: AX25Config) -> None:
        """Set up timers from the given configuration.

        Sets the initial SRTT and RTO values from config.t1_timeout.
        The Jacobson/Karels algorithm will refine these over time.

        Args:
            config: The AX.25 configuration containing:
                - t1_timeout: Base acknowledgment timer (seconds).
                - t3_timeout: Idle probe timer (seconds).
        """
        self.config = config

        self.t1_current: float = config.t1_timeout
        self.t3_current: float = config.t3_timeout

        # Jacobson/Karels adaptive T1 variables
        self.srtt: float = config.t1_timeout
        self.rttvar: float = config.t1_timeout / 2.0
        self.rto: float = self.srtt + max(1.0, 4.0 * self.rttvar)

        # Active timer handles (threading)
        self._t1_thread_timer: Optional[threading.Timer] = None
        self._t3_thread_timer: Optional[threading.Timer] = None

        # Active timer tasks (asyncio)
        self._t1_async_task: Optional[asyncio.Task] = None
        self._t3_async_task: Optional[asyncio.Task] = None

        # Timestamp for RTT measurement
        self._last_ack_time: float = time.monotonic()

        logger.info(
            "AX25Timers initialized: T1_base=%.1fs, T3=%.1fs, RTO=%.1fs",
            config.t1_timeout, config.t3_timeout, self.rto,
        )

    # -----------------------------------------------------------------------
    # Synchronous (threading) T1
    # -----------------------------------------------------------------------

    def start_t1_sync(self, callback: Callable[[], None]) -> None:
        """Start the T1 acknowledgment timer using a background thread.

        If T1 is already running, it is stopped first and restarted.
        The timer will call ``callback`` when it fires. The callback
        runs in a daemon thread, so it must be thread-safe.

        Args:
            callback: A callable with no arguments to invoke when T1
                expires. Called in a daemon thread.

        Raises:
            TimeoutError: If the timer thread cannot be created.

        Example::

            timers.start_t1_sync(on_t1_timeout)
        """
        self.stop_t1_sync()

        try:
            self._t1_thread_timer = threading.Timer(
                self.rto, self._make_t1_handler(callback)
            )
            self._t1_thread_timer.daemon = True
            self._t1_thread_timer.start()
            logger.debug("T1 started (sync): %.2fs", self.rto)
        except Exception as exc:
            logger.error("Failed to start T1 sync timer: %s", exc)
            raise TimeoutError(f"Failed to start T1 timer: {exc}")

    def stop_t1_sync(self) -> None:
        """Cancel the T1 timer if it is running (synchronous mode).

        Safe to call even if T1 is not running. Does nothing in that case.
        """
        if self._t1_thread_timer is not None:
            try:
                self._t1_thread_timer.cancel()
                logger.debug("T1 stopped (sync)")
            except Exception as exc:
                logger.warning("Error cancelling T1 sync timer: %s", exc)
            finally:
                self._t1_thread_timer = None

    # -----------------------------------------------------------------------
    # Synchronous (threading) T3
    # -----------------------------------------------------------------------

    def start_t3_sync(self, callback: Callable[[], None]) -> None:
        """Start the T3 idle probe timer using a background thread.

        If T3 is already running, it is stopped first and restarted.
        T3 is typically much longer than T1 (minutes rather than seconds).

        Args:
            callback: A callable with no arguments to invoke when T3
                expires. Called in a daemon thread.

        Raises:
            TimeoutError: If the timer thread cannot be created.

        Example::

            timers.start_t3_sync(on_idle_timeout)
        """
        self.stop_t3_sync()

        try:
            self._t3_thread_timer = threading.Timer(
                self.t3_current, self._make_t3_handler(callback)
            )
            self._t3_thread_timer.daemon = True
            self._t3_thread_timer.start()
            logger.debug("T3 started (sync): %.1fs", self.t3_current)
        except Exception as exc:
            logger.error("Failed to start T3 sync timer: %s", exc)
            raise TimeoutError(f"Failed to start T3 timer: {exc}")

    def stop_t3_sync(self) -> None:
        """Cancel the T3 timer if it is running (synchronous mode).

        Safe to call even if T3 is not running. Does nothing in that case.
        """
        if self._t3_thread_timer is not None:
            try:
                self._t3_thread_timer.cancel()
                logger.debug("T3 stopped (sync)")
            except Exception as exc:
                logger.warning("Error cancelling T3 sync timer: %s", exc)
            finally:
                self._t3_thread_timer = None

    # -----------------------------------------------------------------------
    # Asynchronous (asyncio) T1
    # -----------------------------------------------------------------------

    async def start_t1_async(
        self,
        callback: Callable[[], Coroutine],
    ) -> None:
        """Start the T1 acknowledgment timer as an asyncio task.

        If T1 is already running, it is cancelled first and restarted.
        The callback is an async function (coroutine) that will be
        awaited when T1 expires.

        Must be called from an async context (inside an async function
        or event loop).

        Args:
            callback: An async callable with no arguments. Will be
                awaited when T1 expires.

        Raises:
            TimeoutError: If the asyncio task cannot be created.

        Example::

            await timers.start_t1_async(on_t1_timeout)
        """
        await self.stop_t1_async()

        try:
            self._t1_async_task = asyncio.create_task(
                self._t1_async_wait(callback)
            )
            logger.debug("T1 started (async): %.2fs", self.rto)
        except Exception as exc:
            logger.error("Failed to start T1 async timer: %s", exc)
            raise TimeoutError(f"Failed to start async T1 timer: {exc}")

    async def stop_t1_async(self) -> None:
        """Cancel the T1 async task if it is running.

        Safe to call even if the task is not running. Awaits the
        cancelled task to ensure it is fully cleaned up.
        """
        if self._t1_async_task is not None:
            try:
                self._t1_async_task.cancel()
                await self._t1_async_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("Error stopping T1 async timer: %s", exc)
            finally:
                self._t1_async_task = None
                logger.debug("T1 stopped (async)")

    # -----------------------------------------------------------------------
    # Asynchronous (asyncio) T3
    # -----------------------------------------------------------------------

    async def start_t3_async(
        self,
        callback: Callable[[], Coroutine],
    ) -> None:
        """Start the T3 idle probe timer as an asyncio task.

        If T3 is already running, it is cancelled and restarted.

        Args:
            callback: An async callable with no arguments. Will be
                awaited when T3 expires.

        Raises:
            TimeoutError: If the asyncio task cannot be created.

        Example::

            await timers.start_t3_async(on_idle_timeout)
        """
        await self.stop_t3_async()

        try:
            self._t3_async_task = asyncio.create_task(
                self._t3_async_wait(callback)
            )
            logger.debug("T3 started (async): %.1fs", self.t3_current)
        except Exception as exc:
            logger.error("Failed to start T3 async timer: %s", exc)
            raise TimeoutError(f"Failed to start async T3 timer: {exc}")

    async def stop_t3_async(self) -> None:
        """Cancel the T3 async task if it is running.

        Safe to call even if the task is not running.
        """
        if self._t3_async_task is not None:
            try:
                self._t3_async_task.cancel()
                await self._t3_async_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("Error stopping T3 async timer: %s", exc)
            finally:
                self._t3_async_task = None
                logger.debug("T3 stopped (async)")

    # -----------------------------------------------------------------------
    # Internal timer implementations
    # -----------------------------------------------------------------------

    def _make_t1_handler(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Create a wrapper function that logs before calling the T1 callback.

        Args:
            callback: The user-supplied function to call on T1 timeout.

        Returns:
            A wrapper function suitable for threading.Timer.
        """
        def handler() -> None:
            logger.warning("T1 timeout fired (RTO=%.2fs)", self.rto)
            try:
                callback()
            except Exception as exc:
                logger.error("T1 timeout callback raised an exception: %s", exc)
        return handler

    def _make_t3_handler(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Create a wrapper function that logs before calling the T3 callback.

        Args:
            callback: The user-supplied function to call on T3 timeout.

        Returns:
            A wrapper function suitable for threading.Timer.
        """
        def handler() -> None:
            logger.warning("T3 idle timeout fired (%.1fs)", self.t3_current)
            try:
                callback()
            except Exception as exc:
                logger.error("T3 timeout callback raised an exception: %s", exc)
        return handler

    async def _t1_async_wait(
        self, callback: Callable[[], Coroutine]
    ) -> None:
        """Internal async coroutine that sleeps then calls the T1 callback.

        Args:
            callback: Async function to await when the timer fires.
        """
        try:
            await asyncio.sleep(self.rto)
            logger.warning("T1 timeout fired (async, RTO=%.2fs)", self.rto)
            await callback()
        except asyncio.CancelledError:
            logger.debug("T1 async timer cancelled")
            raise
        except Exception as exc:
            logger.error("T1 async timer callback failed: %s", exc)

    async def _t3_async_wait(
        self, callback: Callable[[], Coroutine]
    ) -> None:
        """Internal async coroutine that sleeps then calls the T3 callback.

        Args:
            callback: Async function to await when the timer fires.
        """
        try:
            await asyncio.sleep(self.t3_current)
            logger.warning("T3 idle timeout fired (async, %.1fs)", self.t3_current)
            await callback()
        except asyncio.CancelledError:
            logger.debug("T3 async timer cancelled")
            raise
        except Exception as exc:
            logger.error("T3 async timer callback failed: %s", exc)

    # -----------------------------------------------------------------------
    # RTT measurement (Jacobson/Karels)
    # -----------------------------------------------------------------------

    def record_acknowledgment(self) -> None:
        """Update the SRTT estimate after receiving an acknowledgment.

        Call this every time we receive a frame that acknowledges new
        I-frames (any frame with an N(R) that advances V(A)).

        Uses the Jacobson/Karels algorithm (RFC 6298 style):
          - srtt = srtt + 0.125 * (rtt - srtt)
          - rttvar = rttvar + 0.25 * (|rtt - srtt| - rttvar)
          - rto = srtt + 4 * rttvar (clamped to 1..60 seconds)

        This is the same algorithm used by TCP for its retransmit timer.

        Example::

            # When an RR with N(R) > V(A) is received:
            timers.record_acknowledgment()
        """
        now = time.monotonic()
        measured_rtt = now - self._last_ack_time

        # Jacobson/Karels update
        delta = measured_rtt - self.srtt
        self.srtt += 0.125 * delta
        self.rttvar += 0.25 * (abs(delta) - self.rttvar)
        new_rto = self.srtt + max(1.0, 4.0 * self.rttvar)

        # Clamp RTO: minimum 1 second, maximum 60 seconds
        new_rto = max(1.0, min(new_rto, 60.0))

        logger.debug(
            "record_acknowledgment: rtt=%.3fs srtt=%.3fs rttvar=%.3fs rto %.3f->%.3fs",
            measured_rtt, self.srtt, self.rttvar, self.rto, new_rto,
        )
        self.rto = new_rto
        self._last_ack_time = now

    def update_t1_timeout(self, new_timeout: float) -> None:
        """Override the T1 base timeout (seconds).

        Also resets the RTO to the new value, discarding any adaptive
        SRTT measurements accumulated so far.

        Args:
            new_timeout: New T1 timeout in seconds. Must be 0.1 to 60.0.

        Raises:
            ValueError: If new_timeout is out of range.
        """
        if not (0.1 <= new_timeout <= 60.0):
            raise ValueError(
                f"T1 timeout must be 0.1-60.0 seconds, got {new_timeout}"
            )
        self.t1_current = new_timeout
        self.rto = new_timeout
        logger.info("T1 timeout updated to %.1fs, RTO reset", new_timeout)

    def update_t3_timeout(self, new_timeout: float) -> None:
        """Override the T3 idle probe timeout (seconds).

        Args:
            new_timeout: New T3 timeout in seconds. Must be 10.0 to 3600.0.

        Raises:
            ValueError: If new_timeout is out of range.
        """
        if not (10.0 <= new_timeout <= 3600.0):
            raise ValueError(
                f"T3 timeout must be 10.0-3600.0 seconds, got {new_timeout}"
            )
        self.t3_current = new_timeout
        logger.info("T3 timeout updated to %.1fs", new_timeout)

    # -----------------------------------------------------------------------
    # Status and reset
    # -----------------------------------------------------------------------

    def get_timer_status(self) -> dict:
        """Return a snapshot of the current timer state.

        Useful for logging and diagnostics.

        Returns:
            A dict with these keys:
              - ``t1_running`` (bool): True if T1 is active.
              - ``t3_running`` (bool): True if T3 is active.
              - ``t1_current`` (float): Base T1 timeout.
              - ``t3_current`` (float): T3 timeout.
              - ``srtt`` (float): Smoothed RTT estimate.
              - ``rttvar`` (float): RTT variance.
              - ``rto`` (float): Current retransmission timeout.
              - ``last_ack_time`` (float): monotonic timestamp of last ack.

        Example::

            status = timers.get_timer_status()
            print(f"RTO is {status['rto']:.1f}s")
        """
        return {
            "t1_running": (
                self._t1_thread_timer is not None
                or self._t1_async_task is not None
            ),
            "t3_running": (
                self._t3_thread_timer is not None
                or self._t3_async_task is not None
            ),
            "t1_current": self.t1_current,
            "t3_current": self.t3_current,
            "srtt": self.srtt,
            "rttvar": self.rttvar,
            "rto": self.rto,
            "last_ack_time": self._last_ack_time,
        }

    def reset(self) -> None:
        """Stop all timers and reset SRTT/RTO to the original config values.

        Call this when a connection is closed or reset. After reset,
        the timers behave as if newly created with the original config.
        """
        self.stop_t1_sync()
        self.stop_t3_sync()

        # Reset SRTT to base values from config
        self.srtt = self.config.t1_timeout
        self.rttvar = self.config.t1_timeout / 2.0
        self.rto = self.srtt + max(1.0, 4.0 * self.rttvar)
        self._last_ack_time = time.monotonic()

        logger.info("AX25Timers reset to initial state")
