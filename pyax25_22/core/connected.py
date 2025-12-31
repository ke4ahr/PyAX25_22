# pyax25_22/core/connected.py
import logging
import asyncio
from enum import Enum, auto
from typing import Optional, Callable
from .framing import FrameType, AX25Frame

logger = logging.getLogger(__name__)

class AX25State(Enum):
    DISCONNECTED = auto()
    AWAITING_CONNECTION = auto()
    CONNECTED = auto()
    AWAITING_RELEASE = auto()
    TIMEOUT = auto()

class ConnectedModeHandler:
    def __init__(self, 
                 my_call: str,
                 send_frame_fn: Callable[[bytes], None],
                 frame_callback: Optional[Callable[[bytes], None]] = None,
                 t1_timeout: float = 10.0,
                 n2_retries: int = 3):
        self.my_call = my_call
        self.send_frame = send_frame_fn
        self.frame_callback = frame_callback
        self.t1_timeout = t1_timeout
        self.n2_retries = n2_retries
        self.state = AX25State.DISCONNECTED
        self.t1_task: Optional[asyncio.Task] = None
        self.retry_count = 0
        self.vs = 0  # Send sequence number
        self.vr = 0  # Receive sequence number

    async def connect(self, dest_call: str):
        if self.state != AX25State.DISCONNECTED:
            raise RuntimeError("Already connected or connecting")
        
        self.dest_call = dest_call
        self.state = AX25State.AWAITING_CONNECTION
        self._send_sabm()
        await self._start_t1_timer()

    async def disconnect(self):
        if self.state not in (AX25State.CONNECTED, AX25State.AWAITING_CONNECTION):
            return
            
        self.state = AX25State.AWAITING_RELEASE
        self._send_disc()
        await self._start_t1_timer()

    def _send_sabm(self):
        frame = AX25Frame(
            dest=self.dest_call,
            src=self.my_call,
        ).encode_sabm()
        self.send_frame(frame)

    def _send_disc(self):
        frame = AX25Frame(
            dest=self.dest_call,
            src=self.my_call,
        ).encode_disc()
        self.send_frame(frame)

    def _send_ua(self):
        frame = AX25Frame(
            dest=self.dest_call,
            src=self.my_call,
        ).encode_ua()
        self.send_frame(frame)

    def _send_dm(self):
        frame = AX25Frame(
            dest=self.dest_call,
            src=self.my_call,
        ).encode_dm()
        self.send_frame(frame)

    async def _start_t1_timer(self):
        if self.t1_task and not self.t1_task.done():
            self.t1_task.cancel()
        
        self.t1_task = asyncio.create_task(self._t1_expired())
        try:
            await asyncio.wait_for(self.t1_task, self.t1_timeout)
        except asyncio.TimeoutError:
            if self.retry_count < self.n2_retries:
                self.retry_count += 1
                logger.warning(f"Retry {self.retry_count}/{self.n2_retries}")
                if self.state == AX25State.AWAITING_CONNECTION:
                    self._send_sabm()
                elif self.state == AX25State.AWAITING_RELEASE:
                    self._send_disc()
                await self._start_t1_timer()
            else:
                logger.error("Max retries reached, disconnecting")
                self.state = AX25State.TIMEOUT

    async def _t1_expired(self):
        await asyncio.sleep(self.t1_timeout)

    def handle_frame(self, frame: AX25Frame):
        if frame.type == FrameType.SABM:
            if self.state == AX25State.DISCONNECTED:
                self.state = AX25State.CONNECTED
                self._send_ua()
            else:
                self._send_dm()
                
        elif frame.type == FrameType.DISC:
            if self.state == AX25State.CONNECTED:
                self.state = AX25State.DISCONNECTED
                self._send_ua()
                
        elif frame.type == FrameType.UA:
            if self.state == AX25State.AWAITING_CONNECTION:
                self.state = AX25State.CONNECTED
                self.t1_task.cancel()
            elif self.state == AX25State.AWAITING_RELEASE:
                self.state = AX25State.DISCONNECTED
                self.t1_task.cancel()
                
        elif frame.type == FrameType.DM:
            self.state = AX25State.DISCONNECTED
            if self.t1_task:
                self.t1_task.cancel()

        elif frame.type == FrameType.I:
            if self.state != AX25State.CONNECTED:
                self._send_dm()
            else:
                self._handle_i_frame(frame)

    def _handle_i_frame(self, frame: AX25Frame):
        # Basic receive handling - no flow control yet
        ns = (frame.control >> 1) & 0x07
        if ns == self.vr:
            self.vr = (self.vr + 1) % 8
            if self.frame_callback:
                self.frame_callback(frame.info)
