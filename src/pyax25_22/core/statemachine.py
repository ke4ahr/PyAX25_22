# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.statemachine.py

AX.25 v2.2 compliant state machine implementation.

Based on AX.25 v2.2 Specification Section 4 (Link Layer State Diagrams - SDL).
Supports both modulo 8 and modulo 128 operation modes.

This module provides:
- Enumeration of all AX.25 states
- State machine class with transition validation
- Integration points for timers, flow control, and frame processing
"""

from __future__ import annotations

from enum import Enum, auto
import logging
from typing import Optional

from .config import AX25Config
from .exceptions import ConnectionStateError

logger = logging.getLogger(__name__)


class AX25State(Enum):
    """
    AX.25 connection states as defined in v2.2 specification.

    These correspond to the states in the SDL diagrams (Section 4).
    """
    DISCONNECTED = auto()             # State 1: Disconnected
    AWAITING_CONNECTION = auto()      # State 2: Awaiting connection establishment
    AWAITING_RELEASE = auto()         # State 3: Awaiting connection release
    CONNECTED = auto()                # State 4: Connected (normal operation)
    TIMER_RECOVERY = auto()           # State 5: Timer recovery (awaiting ack after timeout)
    AWAITING_XID = auto()             # Optional: Awaiting XID response for negotiation


class AX25StateMachine:
    """
    AX.25 state machine manager.

    Handles all valid state transitions per AX.25 v2.2 SDL.
    Integrates with configuration for modulo-specific sequence handling.
    """

    def __init__(self, config: Optional[AX25Config] = None) -> None:
        """
        Initialize state machine.

        Args:
            config: AX.25 configuration (defaults to mod 8)
        """
        self.config = config or AX25Config()
        self.state: AX25State = AX25State.DISCONNECTED

        # Sequence variables (V(S), V(R), V(A))
        self.v_s: int = 0  # Send state variable (next N(S) to send)
        self.v_r: int = 0  # Receive state variable (expected N(S))
        self.v_a: int = 0  # Acknowledge state variable (last acked N(S))

        # Flags for protocol state
        self.peer_busy: bool = False     # Peer RNR state
        self.reject_sent: bool = False    # REJ outstanding
        self.srej_sent: bool = False      # SREJ outstanding
        self.layer3_initiated: bool = False  # Layer 3 initiated connection

        logger.info(
            f"StateMachine initialized in {self.state.name}, "
            f"modulo={self.config.modulo}"
        )

    @property
    def modulo_mask(self) -> int:
        """Sequence number mask based on modulo."""
        return 0x07 if self.config.modulo == 8 else 0x7F

    def increment_seq(self, seq: int) -> int:
        """
        Increment sequence number with modulo wrap-around.

        Args:
            seq: Current sequence number

        Returns:
            Incremented sequence
        """
        return (seq + 1) & self.modulo_mask

    def transition(self, event: str, frame_type: Optional[str] = None) -> None:
        """
        Perform state transition based on event.

        Validates transitions per AX.25 v2.2 SDL diagrams.
        Updates internal variables on certain transitions.

        Args:
            event: Event name (e.g., 'connect_request', 'SABM_received')
            frame_type: Optional frame type for supervisory frames

        Raises:
            ConnectionStateError if transition invalid
        """
        old_state = self.state
        logger.debug(f"Transition attempt: {old_state.name} --[{event}]--> ?")

        # DISCONNECTED state transitions (State 1)
        if self.state == AX25State.DISCONNECTED:
            if event == "connect_request" and self.layer3_initiated:
                self.state = AX25State.AWAITING_CONNECTION
                self.v_s = self.v_r = self.v_a = 0
                self.peer_busy = self.reject_sent = self.srej_sent = False
            elif event == "SABM_received" or event == "SABME_received":
                self.state = AX25State.CONNECTED
                self.v_s = self.v_r = self.v_a = 0
                self.peer_busy = self.reject_sent = self.srej_sent = False
            elif event == "DISC_received":
                # Send DM response, remain disconnected
                pass
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in DISCONNECTED")

        # AWAITING_CONNECTION transitions (State 2)
        elif self.state == AX25State.AWAITING_CONNECTION:
            if event == "UA_received" or event == "DM_received":
                self.state = AX25State.DISCONNECTED
            elif event == "T1_timeout":
                self.state = AX25State.DISCONNECTED
            elif event == "FRMR_received":
                self.state = AX25State.DISCONNECTED
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in AWAITING_CONNECTION")

        # CONNECTED transitions (State 4)
        elif self.state == AX25State.CONNECTED:
            if event == "disconnect_request":
                self.state = AX25State.AWAITING_RELEASE
            elif event == "DISC_received":
                self.state = AX25State.DISCONNECTED
            elif event == "T3_timeout":
                # Probe channel state
                pass
            elif event == "T1_timeout":
                self.state = AX25State.TIMER_RECOVERY
            elif frame_type == "RNR":
                self.peer_busy = True
            elif frame_type == "RR":
                self.peer_busy = False
            elif frame_type == "REJ":
                self.reject_sent = True
            elif frame_type == "SREJ":
                self.srej_sent = True
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in CONNECTED")

        # TIMER_RECOVERY transitions (State 5)
        elif self.state == AX25State.TIMER_RECOVERY:
            if event == "RR_response" or event == "RNR_response" or event == "REJ_response" or event == "SREJ_response":
                self.state = AX25State.CONNECTED
                self.reject_sent = self.srej_sent = False
            elif event == "T1_timeout":
                # Increment retry counter
                pass
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in TIMER_RECOVERY")

        # AWAITING_RELEASE transitions (State 3)
        elif self.state == AX25State.AWAITING_RELEASE:
            if event == "UA_received" or event == "DM_received":
                self.state = AX25State.DISCONNECTED
            elif event == "T1_timeout":
                self.state = AX25State.DISCONNECTED
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in AWAITING_RELEASE")

        # AWAITING_XID transitions (Optional state for negotiation)
        elif self.state == AX25State.AWAITING_XID:
            if event == "XID_response":
                self.state = AX25State.CONNECTED
            elif event == "T1_timeout":
                self.state = AX25State.DISCONNECTED
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in AWAITING_XID")

        logger.info(f"Transition: {old_state.name} -> {self.state.name} on '{event}'")
