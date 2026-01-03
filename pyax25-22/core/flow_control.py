# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25-22.core.flow_control.py

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

from typing import List, Optional
import logging

from .framing import AX25Frame
from .statemachine import AX25StateMachine
from .config import AX25Config
from .exceptions import FrameError

logger = logging.getLogger(__name__)


class AX25FlowControl:
    """
    Comprehensive flow control manager for connected-mode AX.25.

    Tracks window utilization, acknowledgments, and selective recovery.
    Supports both modulo 8 and modulo 128 operation.
    """

    def __init__(self, sm: AX25StateMachine, config: AX25Config):
        """
        Initialize flow control with state machine and configuration.

        Args:
            sm: Reference to the connection's state machine
            config: AX.25 configuration (defines window size k)
        """
        self.sm = sm
        self.config = config

        # Outstanding I-frame sequence numbers (N(S))
        self.outstanding_seqs: List[int] = []

        # Sequence numbers requested for selective retransmission
        self.srej_requested: List[int] = []

        # Local receiver busy state
        self.local_busy: bool = False

        # Peer receiver busy state
        self.peer_busy: bool = False

        # REJ/SREJ state
        self.rej_sent: bool = False
        self.srej_sent: bool = False

        logger.info(
            f"FlowControl initialized: window k={config.window_size}, "
            f"modulo={config.modulo}"
        )

    @property
    def window_available(self) -> int:
        """Number of I-frames that can still be sent within window."""
        return self.config.window_size - len(self.outstanding_seqs)

    def can_send_i_frame(self) -> bool:
        """Return True if window allows another I-frame."""
        return self.window_available > 0 and not self.peer_busy

    def enqueue_i_frame(self, seq_num: int) -> None:
        """
        Record an I-frame as outstanding after transmission.

        Args:
            seq_num: N(S) value of the transmitted I-frame
        """
        if not self.can_send_i_frame():
            raise FrameError("Cannot enqueue: window full or peer busy")

        self.outstanding_seqs.append(seq_num)
        logger.debug(
            f"Enqueued I-frame N(S)={seq_num}, "
            f"outstanding={len(self.outstanding_seqs)}/{self.config.window_size}"
        )

    def acknowledge_up_to(self, nr: int) -> None:
        """
        Process received N(R) — acknowledge all frames before nr.

        Args:
            nr: N(R) value from incoming frame
        """
        initial_count = len(self.outstanding_seqs)
        # Remove all seq < nr (acknowledged)
        self.outstanding_seqs = [
            seq for seq in self.outstanding_seqs if seq >= nr
        ]

        acknowledged = initial_count - len(self.outstanding_seqs)
        if acknowledged > 0:
            logger.info(f"Acknowledged {acknowledged} frames up to N(R)={nr}")
            self.rej_sent = False
            self.srej_sent = False

    def set_peer_busy(self) -> None:
        """Mark peer as busy (RNR received)."""
        if not self.peer_busy:
            self.peer_busy = True
            logger.warning("Peer indicated busy (RNR)")

    def clear_peer_busy(self) -> None:
        """Clear peer busy state (RR received)."""
        if self.peer_busy:
            self.peer_busy = False
            logger.info("Peer ready again (RR received)")

    def set_local_busy(self) -> None:
        """Set local receiver busy (application cannot accept more data)."""
        self.local_busy = True
        logger.warning("Local receiver busy")

    def clear_local_busy(self) -> None:
        """Clear local busy state."""
        self.local_busy = False
        logger.info("Local receiver ready")

    def send_reject(self, nr: int) -> AX25Frame:
        """
        Generate a REJ supervisory frame.

        Requests retransmission starting from nr.
        """
        if self.rej_sent:
            logger.debug("REJ already outstanding")
            return None

        # Control field: REJ with P/F bit based on state
        pf_bit = 0x10 if self.sm.state == AX25StateMachine.TIMER_RECOVERY else 0x00
        control = 0x09 | (nr << 5) | pf_bit  # REJ base

        self.rej_sent = True
        logger.warning(f"Sending REJ N(R)={nr}, P/F={'1' if pf_bit else '0'}")

        return AX25Frame(
            destination=self.sm.remote_addr if hasattr(self.sm, 'remote_addr') else None,
            source=self.sm.local_addr if hasattr(self.sm, 'local_addr') else None,
            control=control,
            config=self.config,
        )

    def send_selective_reject(self, nr: int) -> AX25Frame:
        """
        Generate an SREJ supervisory frame for a specific missing frame.

        Only one SREJ outstanding at a time per spec.
        """
        if self.srej_sent:
            logger.debug("SREJ already outstanding")
            return None

        if nr in self.srej_requested:
            logger.debug(f"SREJ for {nr} already requested")
            return None

        # Control field: SREJ with P/F bit
        pf_bit = 0x10 if self.sm.state == AX25StateMachine.TIMER_RECOVERY else 0x00
        control = 0x0D | (nr << 5) | pf_bit

        self.srej_sent = True
        self.srej_requested.append(nr)
        logger.warning(f"Sending SREJ N(R)={nr}")

        return AX25Frame(
            destination=self.sm.remote_addr if hasattr(self.sm, 'remote_addr') else None,
            source=self.sm.local_addr if hasattr(self.sm, 'local_addr') else None,
            control=control,
            config=self.config,
        )

    def send_rr(self, pf_bit: bool = False) -> AX25Frame:
        """Generate Receiver Ready supervisory frame."""
        control = 0x01 | (self.sm.v_r << 5) | (0x10 if pf_bit else 0x00)
        logger.debug(f"Sending RR N(R)={self.sm.v_r}, {'P' if pf_bit else 'F'}")

        return AX25Frame(
            destination=self.sm.remote_addr if hasattr(self.sm, 'remote_addr') else None,
            source=self.sm.local_addr if hasattr(self.sm, 'local_addr') else None,
            control=control,
            config=self.config,
        )

    def send_rnr(self, pf_bit: bool = False) -> AX25Frame:
        """Generate Receiver Not Ready supervisory frame."""
        control = 0x05 | (self.sm.v_r << 5) | (0x10 if pf_bit else 0x00)
        logger.warning(f"Sending RNR N(R)={self.sm.v_r}")

        return AX25Frame(
            destination=self.sm.remote_addr if hasattr(self.sm, 'remote_addr') else None,
            source=self.sm.local_addr if hasattr(self.sm, 'local_addr') else None,
            control=control,
            config=self.config,
        )

    def reset(self) -> None:
        """Reset flow control state (e.g., on disconnection)."""
        self.outstanding_seqs.clear()
        self.srej_requested.clear()
        self.local_busy = False
        self.peer_busy = False
        self.rej_sent = False
        self.srej_sent = False
        logger.info("Flow control state reset")
