# pyax25_22/core/flowcontrol.py
import logging
from enum import IntEnum
from typing import Deque, Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)

class WindowMode(IntEnum):
    MOD8 = 8
    MOD128 = 128

class FlowController:
    def __init__(self, 
                 send_callback: Callable[[bytes], None],
                 window_size: int = 4,
                 mode: WindowMode = WindowMode.MOD8):
        self.send_callback = send_callback
        self.window_size = window_size
        self.mode = mode
        
        # Send state
        self.send_window: Deque[Optional[bytes]] = deque(maxlen=window_size)
        self.vs = 0  # Send sequence number
        self.va = 0  # Acknowledged sequence number
        
        # Receive state
        self.vr = 0  # Receive sequence number
        self.rej_sent = False
        self.busy = False
        
        # Buffers
        self.sent_frames: Dict[int, bytes] = {}
        self.recv_buffer: Dict[int, bytes] = {}

    def can_send(self) -> bool:
        return len(self.send_window) < self.window_size

    def send_frame(self, frame: bytes) -> int:
        if not self.can_send():
            raise RuntimeError("Send window full")
            
        seq = self.vs
        self.send_window.append(frame)
        self.sent_frames[seq] = frame
        self.vs = (self.vs + 1) % self.mode
        return seq

    def handle_ack(self, nr: int) -> int:
        acked = 0
        while self.va != nr:
            if self.va in self.send_window:
                self.send_window.remove(self.va)
            if self.va in self.sent_frames:
                del self.sent_frames[self.va]
            self.va = (self.va + 1) % self.mode
            acked += 1
        return acked

    def handle_rej(self, nr: int) -> None:
        seq = nr
        while seq != self.vs:
            if seq in self.sent_frames:
                self.send_callback(self.sent_frames[seq])
            seq = (seq + 1) % self.mode

    def handle_srej(self, nr: int) -> None:
        if nr in self.sent_frames:
            self.send_callback(self.sent_frames[nr])

    def receive_frame(self, frame: bytes, ns: int) -> Optional[int]:
        if ns == self.vr:
            # In-order delivery
            self.vr = (self.vr + 1) % self.mode
            return ns
        elif (ns - self.vr) % self.mode <= self.window_size:
            # Out-of-order but within window
            self.recv_buffer[ns] = frame
            return None
        else:
            # Outside window - protocol error
            return -1

    def get_window_status(self) -> tuple[int, bool]:
        """Returns (nr, busy)"""
        return (self.vr, self.busy)

    def process_recv_buffer(self) -> list[bytes]:
        frames = []
        while self.vr in self.recv_buffer:
            frames.append(self.recv_buffer.pop(self.vr))
            self.vr = (self.vr + 1) % self.mode
        return frames
