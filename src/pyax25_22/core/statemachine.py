# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.statemachine -- AX.25 connection state machine.

This file tracks what "mode" an AX.25 connection is in at any moment.
Think of it like a traffic light: it can only be in one color at a time,
and it changes from one color to another based on specific events.

The AX.25 states are:
  - DISCONNECTED: No connection. Not talking to anyone.
  - AWAITING_CONNECTION: Sent a connect request, waiting for an answer.
  - AWAITING_RELEASE: Sent a disconnect request, waiting to be released.
  - CONNECTED: Link is up. Data can flow.
  - TIMER_RECOVERY: Link had a timeout. Trying to recover.
  - AWAITING_XID: Waiting for the other side to agree on settings.

The state machine also tracks a few counters that change with each
transition: V(S) (next send), V(R) (next receive), V(A) (last acked).

Fully compliant with AX.25 v2.2 specification, July 1998.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
import logging

from .config import AX25Config, DEFAULT_CONFIG_MOD8
from .exceptions import ConnectionStateError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enumeration
# ---------------------------------------------------------------------------

class AX25State(Enum):
    """All possible states for an AX.25 connection.

    Each name matches the state name in the AX.25 v2.2 specification.
    The value is a short string used in log messages.

    Example::

        if sm.state == AX25State.CONNECTED:
            print("Ready to send data")
    """

    DISCONNECTED = "disconnected"
    AWAITING_CONNECTION = "awaiting_connection"
    AWAITING_RELEASE = "awaiting_release"
    CONNECTED = "connected"
    TIMER_RECOVERY = "timer_recovery"
    AWAITING_XID = "awaiting_xid"


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class AX25StateMachine:
    """Tracks the state of one AX.25 connection.

    This class is the "brain" of an AX.25 connection. It knows what
    state the link is in and only allows legal transitions based on the
    AX.25 v2.2 specification. If you try to do something illegal (like
    send data before connecting), it raises ConnectionStateError.

    It also tracks the three sequence number variables:
      - V(S): The sequence number to use on the next outgoing I-frame.
      - V(R): The sequence number we expect on the next incoming I-frame.
      - V(A): The last I-frame that has been acknowledged by the other side.

    Attributes:
        config: The AX.25 configuration (controls modulo/window size).
        layer3_initiated: True if the layer above (like a TCP app) told
            us to connect. False if we are waiting for the other side.
        state: The current connection state (an AX25State enum value).
        modulo_mask: Used to wrap sequence numbers (7 for mod-8, 127 for mod-128).
        v_s: Send state variable V(S). Next outgoing I-frame sequence number.
        v_r: Receive state variable V(R). Next expected incoming sequence number.
        v_a: Acknowledge state variable V(A). Last acknowledged sequence number.
        peer_busy: True if the remote station sent RNR (Receiver Not Ready).
        reject_sent: True if we sent a REJ frame and have not gotten recovery yet.
        srej_sent: True if we sent a SREJ frame and have not gotten recovery yet.

    Raises:
        ConnectionStateError: If a state transition is not allowed.

    Example::

        sm = AX25StateMachine(config)
        sm.transition("connect_request")
        # state is now AWAITING_CONNECTION
        sm.transition("UA_received")
        # state is now CONNECTED
    """

    def __init__(
        self,
        config: AX25Config = DEFAULT_CONFIG_MOD8,
        layer3_initiated: bool = True,
    ) -> None:
        """Set up a new state machine in the DISCONNECTED state.

        Args:
            config: The AX.25 configuration to use. Controls modulo and
                window size for sequence number masking.
            layer3_initiated: Set to True if the layer above (application)
                initiates the connection. False if we accept incoming
                connections. Default is True.
        """
        self.config = config
        self.layer3_initiated = layer3_initiated
        self.state = AX25State.DISCONNECTED

        self._modulo = config.modulo
        self.modulo_mask = 0x07 if self._modulo == 8 else 0x7F

        # Sequence number variables (reset to 0 on connect)
        self.v_s: int = 0
        self.v_r: int = 0
        self.v_a: int = 0

        # Link status flags
        self.peer_busy: bool = False
        self.reject_sent: bool = False
        self.srej_sent: bool = False

        logger.debug(
            "AX25StateMachine created: modulo=%d, layer3_initiated=%s",
            self._modulo, layer3_initiated,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def modulo(self) -> int:
        """The sequence number modulo (8 or 128).

        Returns:
            8 for standard mode, 128 for extended mode.
        """
        return self._modulo

    @modulo.setter
    def modulo(self, value: int) -> None:
        """Change the modulo and update the sequence number mask.

        Args:
            value: Must be 8 or 128.

        Raises:
            ValueError: If value is not 8 or 128.
        """
        if value not in (8, 128):
            raise ValueError(f"Modulo must be 8 or 128, got {value}")
        self._modulo = value
        self.modulo_mask = 0x07 if value == 8 else 0x7F
        logger.debug("modulo changed to %d", value)

    # -----------------------------------------------------------------------
    # Sequence number helpers
    # -----------------------------------------------------------------------

    def increment_vs(self) -> None:
        """Advance V(S) by one, wrapping at the modulo boundary.

        Call this after sending each I-frame to advance the send
        sequence number. The value wraps from (modulo - 1) back to 0.

        Example::

            sm.increment_vs()  # v_s goes from 7 to 0 in modulo-8
        """
        old = self.v_s
        self.v_s = (self.v_s + 1) & self.modulo_mask
        logger.debug("V(S): %d -> %d", old, self.v_s)

    def increment_vr(self) -> None:
        """Advance V(R) by one, wrapping at the modulo boundary.

        Call this after accepting each in-sequence I-frame to show that
        we expect the next one.

        Example::

            sm.increment_vr()  # v_r goes from 7 to 0 in modulo-8
        """
        old = self.v_r
        self.v_r = (self.v_r + 1) & self.modulo_mask
        logger.debug("V(R): %d -> %d", old, self.v_r)

    def reset_sequence_variables(self) -> None:
        """Reset V(S), V(R), and V(A) to zero and clear status flags.

        Called when a connection is established (after SABM/UA exchange)
        or when reconnecting.
        """
        self.v_s = 0
        self.v_r = 0
        self.v_a = 0
        self.peer_busy = False
        self.reject_sent = False
        self.srej_sent = False
        logger.debug("Sequence variables reset to zero")

    # -----------------------------------------------------------------------
    # State transition
    # -----------------------------------------------------------------------

    def transition(self, event: str, frame_type: Optional[str] = None) -> None:
        """Move the state machine to a new state based on an event.

        Checks that the event is legal in the current state. If it is,
        updates the state. If it is not allowed, raises ConnectionStateError.

        The event names match those used in the AX.25 v2.2 SDL diagrams.
        Some older names like "RR_received" are accepted as aliases.

        Args:
            event: The name of the event that happened. For example:
                "connect_request", "UA_received", "T1_timeout",
                "DISC_received", "disconnect_request", "ack_received",
                "XID_received", "SABM_received", "SABME_received",
                "DM_received", "FRMR_received", "supervisory_received".
            frame_type: When event is "supervisory_received", this
                says which frame was received: "RR", "RNR", "REJ",
                or "SREJ". Not used for other events.

        Raises:
            ConnectionStateError: If the event is not legal in the
                current state, or if the current state is unknown.

        Example::

            sm.transition("connect_request")
            sm.transition("UA_received")
            sm.transition("supervisory_received", frame_type="RNR")
        """
        old_state = self.state

        logger.debug(
            "transition: %s --[%s]--> ?",
            old_state.value, event,
        )

        # Accept legacy names like "RR_received" as "supervisory_received"
        if (event.endswith("_received")
                and event[:-9] in ("RR", "RNR", "REJ", "SREJ")):
            frame_type = event[:-9]
            event = "supervisory_received"

        # --- DISCONNECTED ---
        if self.state == AX25State.DISCONNECTED:
            if event == "connect_request":
                if not self.layer3_initiated:
                    raise ConnectionStateError(
                        "connect_request is not allowed: layer3_initiated is False"
                    )
                self.state = AX25State.AWAITING_CONNECTION
                self.reset_sequence_variables()

            elif event in ("SABM_received", "SABME_received"):
                # Remote side is connecting to us
                self.state = AX25State.CONNECTED
                self.reset_sequence_variables()

            elif event == "DISC_received":
                # Remote side sent DISC while we are already disconnected.
                # The spec says send DM. State does not change.
                logger.debug("DISC received in DISCONNECTED state -- send DM, stay disconnected")

            elif event == "T1_timeout":
                # Ignore stale T1 timeouts in DISCONNECTED state
                logger.debug("T1 timeout ignored in DISCONNECTED state")

            else:
                raise ConnectionStateError(
                    f"Event '{event}' is not allowed in state {self.state.value}"
                )

        # --- AWAITING_CONNECTION ---
        elif self.state == AX25State.AWAITING_CONNECTION:
            if event == "UA_received":
                self.state = AX25State.CONNECTED

            elif event in ("DM_received", "FRMR_received"):
                self.state = AX25State.DISCONNECTED

            elif event == "T1_timeout":
                # Exceeded retry limit while waiting for UA
                self.state = AX25State.DISCONNECTED

            else:
                raise ConnectionStateError(
                    f"Event '{event}' is not allowed in state {self.state.value}"
                )

        # --- CONNECTED ---
        elif self.state == AX25State.CONNECTED:
            if event == "disconnect_request":
                self.state = AX25State.AWAITING_RELEASE

            elif event == "DISC_received":
                self.state = AX25State.DISCONNECTED

            elif event == "T3_timeout":
                # Send a probe (RR with P=1) -- state does not change
                logger.debug("T3 idle timeout in CONNECTED -- probe channel")

            elif event == "T1_timeout":
                # Enter timer recovery mode
                self.state = AX25State.TIMER_RECOVERY

            elif event == "supervisory_received":
                # Update peer status based on frame type
                if frame_type == "RNR":
                    self.peer_busy = True
                    logger.debug("Peer busy (RNR received)")
                elif frame_type == "RR":
                    self.peer_busy = False
                    logger.debug("Peer ready (RR received)")
                elif frame_type == "REJ":
                    self.reject_sent = True
                    logger.debug("REJ received -- will retransmit")
                elif frame_type == "SREJ":
                    self.srej_sent = True
                    logger.debug("SREJ received -- will retransmit specific frame")

            else:
                raise ConnectionStateError(
                    f"Event '{event}' is not allowed in state {self.state.value}"
                )

        # --- TIMER_RECOVERY ---
        elif self.state == AX25State.TIMER_RECOVERY:
            if event == "ack_received":
                self.state = AX25State.CONNECTED

            elif event == "T1_timeout":
                # Another timeout -- stay in TIMER_RECOVERY and retry
                # (the caller increments the retry counter and may move to DISCONNECTED)
                self.state = AX25State.CONNECTED

            else:
                raise ConnectionStateError(
                    f"Event '{event}' is not allowed in state {self.state.value}"
                )

        # --- AWAITING_RELEASE ---
        elif self.state == AX25State.AWAITING_RELEASE:
            if event == "UA_received":
                self.state = AX25State.DISCONNECTED

            elif event == "T1_timeout":
                # Gave up waiting for UA -- force disconnect
                self.state = AX25State.DISCONNECTED

            else:
                raise ConnectionStateError(
                    f"Event '{event}' is not allowed in state {self.state.value}"
                )

        # --- AWAITING_XID ---
        elif self.state == AX25State.AWAITING_XID:
            if event == "XID_received":
                self.state = AX25State.CONNECTED

            elif event == "T1_timeout":
                self.state = AX25State.DISCONNECTED

            else:
                raise ConnectionStateError(
                    f"Event '{event}' is not allowed in state {self.state.value}"
                )

        else:
            raise ConnectionStateError(
                f"Unknown state: {self.state!r}"
            )

        if self.state != old_state:
            logger.info(
                "State: %s -> %s (event=%s)",
                old_state.value, self.state.value, event,
            )
        else:
            logger.debug(
                "State unchanged at %s (event=%s)",
                self.state.value, event,
            )
