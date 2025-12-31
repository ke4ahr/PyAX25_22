# pyax25_22/core/timers.py
import asyncio
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class AX25Timer:
    def __init__(self, 
                 timeout: float,
                 callback: Callable[[], None],
                 name: str = "timer"):
        self.timeout = timeout
        self.callback = callback
        self.name = name
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def _timer_task(self):
        try:
            await asyncio.sleep(self.timeout)
            if self._running:
                logger.debug(f"Timer {self.name} expired")
                self.callback()
        except asyncio.CancelledError:
            logger.debug(f"Timer {self.name} cancelled")

    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._timer_task(), name=self.name)

    def cancel(self):
        if self._running and self._task:
            self._running = False
            self._task.cancel()

    def reset(self):
        self.cancel()
        self.start()

class TimerManager:
    def __init__(self,
                 t1_timeout: float = 10.0,
                 t2_timeout: float = 2.0, 
                 t3_timeout: float = 30.0):
        # Timers
        self.t1 = AX25Timer(t1_timeout, self._t1_expired, "T1")
        self.t2 = AX25Timer(t2_timeout, self._t2_expired, "T2") 
        self.t3 = AX25Timer(t3_timeout, self._t3_expired, "T3")
        
        # State
        self.ack_pending = False
        self.t1_running = False

    def _t1_expired(self):
        """Handle T1 retransmission timeout"""
        pass  # Implemented in ConnectedModeHandler

    def _t2_expired(self):
        """Handle T2 acknowledgment timeout"""
        pass  # Implemented in ConnectedModeHandler

    def _t3_expired(self):
        """Handle T3 inactivity timeout"""
        pass  # Implemented in ConnectedModeHandler

    def notify_activity(self):
        """Reset T3 on any frame activity"""
        self.t3.reset()

    def start_t1(self):
        if not self.t1_running:
            self.t1_running = True
            self.t1.start()

    def stop_t1(self):
        if self.t1_running:
            self.t1_running = False
            self.t1.cancel()

