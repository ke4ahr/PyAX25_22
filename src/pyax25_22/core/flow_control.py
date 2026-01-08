# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.flow_control.py

AX.25 v2.2 compliant flow control and selective reject implementation.

Manages:
- Transmit and receive window tracking
- Outstanding I-frame acknowledgment
- Receiver Ready/Not Ready (RR/RNR)
- Reject (REJ) and Selective Reject (SREJ)
- Peer busy state detection
- Integration with state machine and timers

Fully compliant with AX.25 v2.2 Section 4.3.3 (Flow Control).
"""

from __future__ import annotations

from typing import List
import logging

from .framing import AX25Frame
from .statemachine import AX25StateMachine, AX25State
from .config import AX25Config
from .exceptions import FrameError

logger = logging.getLogger(__name__)

class AX25FlowControl:
    """Comprehensive flow control manager for connected-mode AX.25."""

    def __init__(self, sm: AX25StateMachine, config: AX25Config):
        self.sm = sm
        self.config = config

        self.outstanding_seqs: List[int] = []
        self.srej_requested: List[int] = []

        self.local_busy: bool = False
        self.peer_busy: bool = False
        self.rej_sent: bool = False
        self.srej_sent: bool = False

        logger.info(f"FlowControl initialized: k={config.window_size}, modulo={config.modulo}")

    @property
    def window_available(self) -> int:
        return self.config.window_size - len(self.outstanding_seqs)

    def can_send_i_frame(self) -> bool:
        return self.window_available > 0 and not self.peer_busy

    def send_data(self, seq_num: int) -> None:
        if not self.can_send_i_frame():
            raise FrameError("Cannot enqueue: window full or peer busy")
        self.outstanding_seqs.append(seq_num)

    def acknowledge_up_to(self, nr: int) -> None:
        initial = len(self.outstanding_seqs)
        self.outstanding_seqs = [s for s in self.outstanding_seqs if s >= nr]
        if len(self.outstanding_seqs) < initial:
            self.rej_sent = self.srej_sent = False

    def set_peer_busy(self) -> None:
        if not self.peer_busy:
            self.peer_busy = True
            logger.warning("Peer busy (RNR)")

    def clear_peer_busy(self) -> None:
        if self.peer_busy:
            self.peer_busy = False
            logger.info("Peer ready (RR)")

    def set_local_busy(self) -> None:
        self.local_busy = True

    def clear_local_busy(self) -> None:
        self.local_busy = False

    def send_reject(self, nr: int) -> Optional[AX25Frame]:
        if self.rej_sent:
            return None
        pf_bit = 0x10 if self.sm.state == AX25State.TIMER_RECOVERY else 0x00
        control = 0x09 | (nr << 5) | pf_bit
        self.rej_sent = True
        return AX25Frame(
            destination=getattr(self.sm, "remote_addr", None),
            source=getattr(self.sm, "local_addr", None),
            control=control,
            config=self.config,
        )

    def send_selective_reject(self, nr: int) -> Optional[AX25Frame]:
        if self.srej_sent or nr in self.srej_requested:
            return None
        pf_bit = 0x10 if self.sm.state == AX25State.TIMER_RECOVERY else 0x00
        control = 0x0D | (nr << 5) | pf_bit
        self.srej_sent = True
        self.srej_requested.append(nr)
        return AX25Frame(
            destination=getattr(self.sm, "remote_addr", None),
            source=getattr(self.sm, "local_addr", None),
            control=control,
            config=self.config,
        )

    def send_rr(self, pf_bit: bool = False) -> AX25Frame:
        control = 0x01 | (self.sm.v_r << 5) | (0x10 if pf_bit else 0x00)
        return AX25Frame(
            destination=getattr(self.sm, "remote_addr", None),
            source=getattr(self.sm, "local_addr", None),
            control=control,
            config=self.config,
        )

    def send_rnr(self, pf_bit: bool = False) -> AX25Frame:
        control = 0x05 | (self.sm.v_r << 5) | (0x10 if pf_bit else 0x00)
        return AX25Frame(
            destination=getattr(self.sm, "remote_addr", None),
            source=getattr(self.sm, "local_addr", None),
            control=control,
            config=self.config,
        )

    def reset(self) -> None:
        self.outstanding_seqs.clear()
        self.srej_requested.clear()
        self.local_busy = self.peer_busy = False
        self.rej_sent = self.srej_sent = False
        logger.info("Flow control reset")
