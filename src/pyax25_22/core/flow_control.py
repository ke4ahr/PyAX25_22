# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.flow_control -- AX.25 send/receive window management.

This file manages how many packets can be "in flight" at once on a
connected AX.25 link. Think of it like a bucket brigade:

  - Only so many buckets (I-frames) can be passed at once.
  - If the bucket line gets full, we stop until some come back.
  - If the other side is busy, we also stop.
  - When the other side says it is ready, we start again.

This also handles the Selective Reject (SREJ) list, which lets the
receiver ask for just one missing frame to be re-sent without asking
for everything to be repeated.

Fully compliant with AX.25 v2.2 Section 4.3.3 (Flow Control).
"""

from __future__ import annotations

from typing import List, Optional
import logging

from .framing import AX25Frame
from .statemachine import AX25StateMachine, AX25State
from .config import AX25Config
from .exceptions import FrameError, ResourceExhaustionError

logger = logging.getLogger(__name__)


class AX25FlowControl:
    """Manages the transmit and receive windows for one AX.25 connection.

    Keeps track of which I-frames have been sent but not yet
    acknowledged, whether the remote station is busy, and whether
    we have sent a Reject or Selective Reject.

    Also builds the Supervisory (S) frames that report our state to
    the remote station: RR (ready), RNR (busy), REJ (go back), or
    SREJ (resend one frame).

    Attributes:
        sm: The state machine for this connection.
        config: The AX.25 configuration (window size, modulo).
        outstanding_seqs: Sequence numbers of sent-but-not-acked frames.
        srej_requested: Sequence numbers we have asked to be re-sent.
        local_busy: True if we sent RNR (our receive buffer is full).
        peer_busy: True if the remote station sent RNR.
        rej_sent: True if we sent REJ and have not yet recovered.
        srej_sent: True if we sent SREJ and have not yet recovered.

    Raises:
        ResourceExhaustionError: If you try to send when the window is full.
        FrameError: If an invalid sequence number is given.

    Example::

        fc = AX25FlowControl(state_machine, config)
        if fc.can_send_i_frame():
            fc.enqueue_i_frame(seq_num=0)
    """

    def __init__(self, sm: AX25StateMachine, config: AX25Config) -> None:
        """Set up flow control for a new connection.

        Args:
            sm: The state machine that tracks the connection state.
                Used to check the current state when building frames.
            config: The AX.25 configuration. The window_size and modulo
                fields are used to limit in-flight frames.
        """
        self.sm = sm
        self.config = config

        self.outstanding_seqs: List[int] = []
        self.srej_requested: List[int] = []

        self.local_busy: bool = False
        self.peer_busy: bool = False
        self.rej_sent: bool = False
        self.srej_sent: bool = False

        logger.info(
            "AX25FlowControl initialized: window_size=%d, modulo=%d",
            config.window_size, config.modulo,
        )

    # -----------------------------------------------------------------------
    # Window status
    # -----------------------------------------------------------------------

    @property
    def window_available(self) -> int:
        """How many more I-frames we can send right now.

        This is the window size minus the number of sent-but-not-acked
        frames. When this reaches zero, we must wait for acknowledgments
        before sending more.

        Returns:
            Number of frames we can still send (0 or more).
        """
        available = self.config.window_size - len(self.outstanding_seqs)
        logger.debug(
            "window_available: %d (window=%d, outstanding=%d)",
            available, self.config.window_size, len(self.outstanding_seqs),
        )
        return available

    def can_send_i_frame(self) -> bool:
        """Check whether it is safe to send another I-frame right now.

        All three of these must be true to send:
          1. The window is not full (room for more unacked frames).
          2. The remote station is not busy (no RNR received).
          3. We are not busy (we did not send RNR).

        Returns:
            True if we may send an I-frame, False if we must wait.

        Example::

            if fc.can_send_i_frame():
                send_next_frame()
        """
        result = (
            self.window_available > 0
            and not self.peer_busy
            and not self.local_busy
        )
        logger.debug(
            "can_send_i_frame: %s (avail=%d, peer_busy=%s, local_busy=%s)",
            result, self.window_available, self.peer_busy, self.local_busy,
        )
        return result

    # -----------------------------------------------------------------------
    # Frame tracking
    # -----------------------------------------------------------------------

    def enqueue_i_frame(self, seq_num: int) -> None:
        """Record that an I-frame with this sequence number has been sent.

        Call this right after putting an I-frame on the wire. It adds
        the sequence number to the list of frames waiting for an ack.

        Args:
            seq_num: The N(S) sequence number of the frame just sent.
                Must be in range 0 to (modulo - 1).

        Raises:
            ResourceExhaustionError: If the window is full or the peer
                is busy, so we should not be sending right now.
            FrameError: If this sequence number is already in the
                outstanding list (would indicate a logic error).
        """
        if not self.can_send_i_frame():
            raise ResourceExhaustionError(
                f"Cannot enqueue seq {seq_num}: "
                f"window full ({len(self.outstanding_seqs)}/{self.config.window_size}) "
                f"or peer_busy={self.peer_busy}"
            )
        if seq_num in self.outstanding_seqs:
            raise FrameError(
                f"Sequence number {seq_num} is already outstanding -- "
                f"this is a logic error"
            )

        self.outstanding_seqs.append(seq_num)
        logger.debug(
            "enqueue_i_frame: seq=%d, outstanding=%s",
            seq_num, self.outstanding_seqs,
        )

    def acknowledge_up_to(self, nr: int) -> None:
        """Remove all outstanding frames with sequence numbers below nr.

        When the remote side sends N(R) in any frame, it is saying
        "I have received everything up to but not including N(R)."
        We remove those from our outstanding list.

        After an acknowledgment clears the list, we also clear the
        REJ/SREJ sent flags so we are ready for normal operation again.

        Args:
            nr: The N(R) value from the received frame. All outstanding
                frames with sequence number less than nr are acknowledged.
        """
        before = len(self.outstanding_seqs)
        self.outstanding_seqs = [
            s for s in self.outstanding_seqs if s >= nr
        ]
        after = len(self.outstanding_seqs)
        acked = before - after

        if acked > 0:
            logger.debug(
                "acknowledge_up_to(%d): acknowledged %d frame(s), %d still outstanding",
                nr, acked, after,
            )
            self.rej_sent = False
            self.srej_sent = False
        else:
            logger.debug(
                "acknowledge_up_to(%d): nothing to acknowledge (outstanding=%s)",
                nr, self.outstanding_seqs,
            )

    # -----------------------------------------------------------------------
    # Peer busy state
    # -----------------------------------------------------------------------

    def handle_rr(self) -> None:
        """Process a received RR (Receiver Ready) frame.

        The remote station is telling us it is ready to receive more
        I-frames. Clear the peer_busy flag so we can resume sending.
        """
        logger.debug("handle_rr: clearing peer_busy")
        self.clear_peer_busy()

    def handle_rnr(self) -> None:
        """Process a received RNR (Receiver Not Ready) frame.

        The remote station is telling us it cannot accept more I-frames
        right now (its buffer may be full). Set the peer_busy flag.
        """
        logger.debug("handle_rnr: setting peer_busy")
        self.set_peer_busy()

    def set_peer_busy(self) -> None:
        """Mark the remote station as busy.

        Call this when an RNR frame is received. While peer_busy is
        True, we must not send I-frames.
        """
        if not self.peer_busy:
            self.peer_busy = True
            logger.warning(
                "Peer is busy (RNR received) -- suspending I-frame transmission"
            )
        else:
            logger.debug("set_peer_busy: already busy")

    def clear_peer_busy(self) -> None:
        """Mark the remote station as ready.

        Call this when an RR frame is received or when the peer's
        P/F bit situation resolves. Once cleared, sending may resume.
        """
        if self.peer_busy:
            self.peer_busy = False
            logger.info("Peer is ready (RR received) -- I-frame transmission may resume")
        else:
            logger.debug("clear_peer_busy: was not busy")

    def set_local_busy(self) -> None:
        """Mark our own receive buffer as busy.

        Call this when our receive buffer is full and we need to
        send RNR to tell the other side to stop. While local_busy is
        True, we will not accept more I-frames.
        """
        if not self.local_busy:
            self.local_busy = True
            logger.warning("Local busy -- will send RNR to peer")
        else:
            logger.debug("set_local_busy: already busy")

    def clear_local_busy(self) -> None:
        """Mark our own receive buffer as ready again.

        Call this when buffer space has been freed and we are ready
        to accept more I-frames. We should then send RR to tell the
        remote station it may resume.
        """
        if self.local_busy:
            self.local_busy = False
            logger.info("Local buffer available -- will send RR to peer")
        else:
            logger.debug("clear_local_busy: was not busy")

    # -----------------------------------------------------------------------
    # Supervisory frame builders
    # -----------------------------------------------------------------------

    def send_rr(self, pf_bit: bool = False) -> AX25Frame:
        """Build a Receiver Ready (RR) supervisory frame.

        RR tells the remote station that we are ready to receive more
        I-frames. It also carries our current V(R) as N(R), which
        acknowledges all frames up to that sequence number.

        Args:
            pf_bit: Set to True to set the Poll/Final bit in the frame.
                Used during timer recovery procedures.

        Returns:
            An AX25Frame ready to be encoded and sent.

        Raises:
            FrameError: If the state machine does not have address info.
        """
        control = 0x01 | (self.sm.v_r << 5) | (0x10 if pf_bit else 0x00)
        frame = AX25Frame(
            destination=getattr(self.sm, "remote_addr", None),
            source=getattr(self.sm, "local_addr", None),
            control=control,
            config=self.config,
        )
        logger.debug("send_rr: control=0x%02X pf_bit=%s", control, pf_bit)
        return frame

    def send_rnr(self, pf_bit: bool = False) -> AX25Frame:
        """Build a Receiver Not Ready (RNR) supervisory frame.

        RNR tells the remote station that we cannot accept more I-frames
        right now. The remote station must stop sending I-frames until
        it receives an RR from us.

        Args:
            pf_bit: Set to True to set the Poll/Final bit in the frame.

        Returns:
            An AX25Frame ready to be encoded and sent.
        """
        control = 0x05 | (self.sm.v_r << 5) | (0x10 if pf_bit else 0x00)
        frame = AX25Frame(
            destination=getattr(self.sm, "remote_addr", None),
            source=getattr(self.sm, "local_addr", None),
            control=control,
            config=self.config,
        )
        logger.debug("send_rnr: control=0x%02X pf_bit=%s", control, pf_bit)
        return frame

    def send_reject(self, nr: int) -> Optional[AX25Frame]:
        """Build a Reject (REJ) supervisory frame if not already sent.

        REJ tells the remote station that frame N(R) was not received
        correctly and all frames from N(R) onward must be re-sent.

        If we already sent a REJ that has not been resolved yet,
        returns None to avoid sending duplicate REJs.

        Args:
            nr: The sequence number of the first bad frame. This will
                be the N(R) in the REJ frame.

        Returns:
            An AX25Frame to send, or None if REJ already sent.
        """
        if self.rej_sent:
            logger.debug("send_reject: REJ already outstanding, not sending again")
            return None

        pf_bit = 0x10 if self.sm.state == AX25State.TIMER_RECOVERY else 0x00
        control = 0x09 | (nr << 5) | pf_bit
        self.rej_sent = True

        frame = AX25Frame(
            destination=getattr(self.sm, "remote_addr", None),
            source=getattr(self.sm, "local_addr", None),
            control=control,
            config=self.config,
        )
        logger.debug("send_reject: nr=%d control=0x%02X", nr, control)
        return frame

    def send_selective_reject(self, nr: int) -> Optional[AX25Frame]:
        """Build a Selective Reject (SREJ) frame to request one frame.

        SREJ asks the remote station to re-send just the one I-frame
        with sequence number N(R), without re-sending everything after it.
        This is more efficient than REJ when only one frame is missing.

        Returns None if we already sent SREJ for this sequence number,
        to avoid duplicate requests.

        Args:
            nr: The sequence number of the missing frame.

        Returns:
            An AX25Frame to send, or None if already requested.
        """
        if self.srej_sent or nr in self.srej_requested:
            logger.debug(
                "send_selective_reject: already requested nr=%d, skipping",
                nr,
            )
            return None

        pf_bit = 0x10 if self.sm.state == AX25State.TIMER_RECOVERY else 0x00
        control = 0x0D | (nr << 5) | pf_bit
        self.srej_sent = True
        self.srej_requested.append(nr)

        frame = AX25Frame(
            destination=getattr(self.sm, "remote_addr", None),
            source=getattr(self.sm, "local_addr", None),
            control=control,
            config=self.config,
        )
        logger.debug(
            "send_selective_reject: nr=%d control=0x%02X srej_list=%s",
            nr, control, self.srej_requested,
        )
        return frame

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all flow control state to the initial values.

        Call this when a new connection is established or after a link
        reset (SABM). Clears all outstanding frames, SREJ requests, and
        busy flags.
        """
        self.outstanding_seqs.clear()
        self.srej_requested.clear()
        self.local_busy = False
        self.peer_busy = False
        self.rej_sent = False
        self.srej_sent = False
        logger.info("Flow control state reset")
