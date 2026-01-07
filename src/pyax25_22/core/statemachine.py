# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.statemachine.py

AX.25 v2.2 Layer 2 state machine implementation.

Implements all states and transitions per AX.25 v2.2 SDL diagrams.

States:
- DISCONNECTED
- AWAITING_CONNECTION
- AWAITING_RELEASE
- CONNECTED
- TIMER_RECOVERY
- AWAITING_XID
"""

from __future__ import annotations

from enum import Enum
import logging

from .exceptions import ConnectionStateError

logger = logging.getLogger(__name__)


class AX25State(Enum):
    DISCONNECTED = "disconnected"
    AWAITING_CONNECTION = "awaiting_connection"
    AWAITING_RELEASE = "awaiting_release"
    CONNECTED = "connected"
    TIMER_RECOVERY = "timer_recovery"
    AWAITING_XID = "awaiting_xid"


class AX25StateMachine:
    """AX.25 Layer 2 state machine."""

    def __init__(self, layer3_initiated: bool = True):
        self.layer3_initiated = layer3_initiated
        self.state = AX25State.DISCONNECTED
        self.modulo = 8
        self.modulo_mask = 0x07
        self.v_s = self.v_r = self.v_a = 0
        self.peer_busy = False
        self.reject_sent = False
        self.srej_sent = False

    def increment_vs(self) -> None:
        """Increment V(S) with modulo wrap."""
        self.v_s = (self.v_s + 1) & self.modulo_mask

    def transition(self, event: str, frame_type: Optional[str] = None) -> None:
        """Perform state transition based on event.

        Validates transitions per AX.25 v2.2 SDL diagrams.
        """
        old_state = self.state
        logger.debug(f"Transition attempt: {old_state.value} --[{event}]--> ?")

        # Map legacy supervisory events for backward compatibility
        if event.endswith("_received") and event[:-9] in {"RR", "RNR", "REJ", "SREJ"}:
            frame_type = event[:-9]
            event = "supervisory_received"

        # DISCONNECTED state transitions
        if self.state == AX25State.DISCONNECTED:
            if event == "connect_request":
                if not self.layer3_initiated:
                    raise ConnectionStateError("Connect request not allowed without layer3 initiation")
                self.state = AX25State.AWAITING_CONNECTION
                self.v_s = self.v_r = self.v_a = 0
                self.peer_busy = self.reject_sent = self.srej_sent = False
            elif event in ("SABM_received", "SABME_received"):
                self.state = AX25State.CONNECTED
                self.v_s = self.v_r = self.v_a = 0
                self.peer_busy = self.reject_sent = self.srej_sent = False
            elif event == "DISC_received":
                # Send DM response, remain disconnected
                pass
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in DISCONNECTED")

        # AWAITING_CONNECTION transitions
        elif self.state == AX25State.AWAITING_CONNECTION:
            if event in ("UA_received", "DM_received", "FRMR_received"):
                self.state = AX25State.DISCONNECTED
            elif event == "T1_timeout":
                self.state = AX25State.DISCONNECTED
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in AWAITING_CONNECTION")

        # CONNECTED transitions
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
            elif event == "supervisory_received":
                if frame_type == "RNR":
                    self.peer_busy = True
                elif frame_type == "RR":
                    self.peer_busy = False
                elif frame_type == "REJ":
                    self.reject_sent = True
                elif frame_type == "SREJ":
                    self.srej_sent = True
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in CONNECTED")

        # TIMER_RECOVERY transitions
        elif self.state == AX25State.TIMER_RECOVERY:
            if event == "ack_received":
                self.state = AX25State.CONNECTED
            elif event == "T1_timeout":
                self.state = AX25State.CONNECTED
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in TIMER_RECOVERY")

        # AWAITING_RELEASE transitions
        elif self.state == AX25State.AWAITING_RELEASE:
            if event == "UA_received":
                self.state = AX25State.DISCONNECTED
            elif event == "T1_timeout":
                self.state = AX25State.DISCONNECTED
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in AWAITING_RELEASE")

        # AWAITING_XID transitions
        elif self.state == AX25State.AWAITING_XID:
            if event == "XID_received":
                self.state = AX25State.CONNECTED
            elif event == "T1_timeout":
                self.state = AX25State.DISCONNECTED
            else:
                raise ConnectionStateError(f"Invalid event '{event}' in AWAITING_XID")

        else:
            raise ConnectionStateError(f"Unknown state {self.state}")

        logger.debug(f"Transition: {old_state.value} -> {self.state.value}")