# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2026 Kris Kirby, KE4AHR

"""
pyax25_22.core.connected -- Manages one connected AX.25 session.

This module handles the life of a single connected AX.25 link from
start to finish. Think of it like managing a phone call:

  1. You dial the other station (SABM/SABME frame).
  2. They pick up and say hello (UA frame).
  3. You talk back and forth (I-frames carrying data).
  4. If a packet is garbled, you ask for it again (REJ/SREJ).
  5. When done, one side hangs up (DISC/UA).

The AX25Connection class ties together all the other pieces:
  - AX25StateMachine: tracks whether the link is up, connecting, etc.
  - AX25FlowControl: limits how many packets can be in flight.
  - AX25Timers: handles the T1 retry timer and T3 idle probe timer.
  - Negotiation: agrees on window size, modulo, and max frame size.

Both synchronous (threaded) and asynchronous (asyncio) use is supported.

Compliant with AX.25 v2.2 specification, July 1998.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from .framing import AX25Frame, AX25Address
from .statemachine import AX25StateMachine, AX25State
from .flow_control import AX25FlowControl
from .timers import AX25Timers
from .negotiation import build_xid_frame, parse_xid_frame, negotiate_config
from .config import AX25Config, DEFAULT_CONFIG_MOD8
from .exceptions import (
    ConnectionStateError,
    FrameError,
    NegotiationError,
    TransportError,
)

logger = logging.getLogger(__name__)


class AX25Connection:
    """A single connected AX.25 session between two stations.

    Manages the complete life cycle of a connected AX.25 link:
    connection setup, data transfer, flow control, parameter
    negotiation, and clean disconnection.

    Each AX25Connection represents exactly one logical link. To talk
    to two remote stations at once, create two AX25Connection objects.

    Attributes:
        local_addr: The local station address (our callsign/SSID).
        remote_addr: The address of the station we are connected to.
        config: The AX.25 configuration for this link. May be updated
            after XID negotiation.
        transport: The interface used to send and receive raw frames.
            May be None for testing without a radio.
        sm: The state machine tracking the link state.
        flow: The flow control manager.
        timers: The T1/T3 timer manager.
        v_s: Send state variable -- sequence number for the next I-frame.
        v_r: Receive state variable -- next expected incoming I-frame number.
        v_a: Acknowledge state variable -- last frame the remote has acked.
        outgoing_queue: Data bytes waiting to be sent as I-frames.
        incoming_buffer: Received I-frame data waiting to be read.
        negotiated_config: The final negotiated config (set after XID).
        retry_count: How many times we have retransmitted after a timeout.

    Raises:
        ConnectionStateError: If you try to do something not allowed in
            the current state (for example, send data when not connected).
        TransportError: If sending a frame fails at the transport layer.
        FrameError: If a frame is malformed.

    Example::

        local  = AX25Address("KE4AHR", ssid=1)
        remote = AX25Address("W1AW")
        conn = AX25Connection(local, remote, config=my_config, initiate=True)
        await conn.connect()
        await conn.send_data(b"Hello W1AW!")
        await conn.disconnect()
    """

    def __init__(
        self,
        local_addr: AX25Address,
        remote_addr: AX25Address,
        config: Optional[AX25Config] = None,
        initiate: bool = False,
        transport=None,
    ) -> None:
        """Set up a new AX.25 connection object.

        Creates all the internal components (state machine, flow control,
        timers) and optionally kicks off the connection sequence.

        Args:
            local_addr: Our station address (callsign + SSID).
            remote_addr: The address of the other station.
            config: AX.25 configuration to use. Defaults to the standard
                modulo-8 configuration if not given.
            initiate: If True, this side initiates the connection by
                sending SABM when ``connect()`` is called. If False,
                this side waits for an incoming SABM.
            transport: An optional transport object that provides
                ``send_frame(frame)`` and ``receive_frame()`` methods.
                Pass None for testing without hardware.
        """
        self.local_addr = local_addr
        self.remote_addr = remote_addr
        self.config = config if config is not None else DEFAULT_CONFIG_MOD8
        self.transport = transport

        # Build the internal components
        self.sm = AX25StateMachine(self.config, layer3_initiated=initiate)
        self.flow = AX25FlowControl(self.sm, self.config)
        self.timers = AX25Timers(self.config)

        # Sequence number variables (mirrored from state machine for convenience)
        self.v_s: int = 0
        self.v_r: int = 0
        self.v_a: int = 0

        # Data buffers
        self.outgoing_queue: List[bytes] = []
        self.incoming_buffer: List[bytes] = []

        # Negotiation
        self.negotiated_config: Optional[AX25Config] = None
        self.xid_pending: bool = False

        # Retry counter (tracks T1 timeouts since last success)
        self.retry_count: int = 0

        if initiate:
            # Trigger the state machine into AWAITING_CONNECTION
            self.sm.transition("connect_request")

        logger.info(
            "AX25Connection created: %s-%d <-> %s-%d, initiate=%s, modulo=%d",
            local_addr.callsign, local_addr.ssid,
            remote_addr.callsign, remote_addr.ssid,
            initiate, self.config.modulo,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def peer_busy(self) -> bool:
        """True if the remote station has sent RNR (Receiver Not Ready).

        When this is True we must not send I-frames. The flow control
        manager tracks this flag.
        """
        return self.flow.peer_busy

    @property
    def state(self) -> AX25State:
        """The current connection state from the state machine.

        Returns:
            An AX25State enum value (e.g. CONNECTED, DISCONNECTED).
        """
        return self.sm.state

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    async def connect(self) -> AX25Frame:
        """Send the connection request frame (SABM or SABME).

        Builds and sends the appropriate connect frame:
          - SABM (0x2F) for modulo-8 connections.
          - SABME (0x6F) for modulo-128 connections.

        The P (Poll) bit is set to 1. Starts T1 to wait for the UA reply.

        Returns:
            The SABM or SABME frame that was sent.

        Raises:
            ConnectionStateError: If the state machine is not in
                AWAITING_CONNECTION state (call initiate=True in constructor
                before calling connect).

        Example::

            sabm = await conn.connect()
        """
        if self.sm.state != AX25State.AWAITING_CONNECTION:
            raise ConnectionStateError(
                f"connect() called in state {self.sm.state.value} -- "
                f"must be AWAITING_CONNECTION (set initiate=True in constructor)"
            )

        # SABM = 0x2F (mod 8), SABME = 0x6F (mod 128), both with P=1
        control = 0x2F if self.config.modulo == 8 else 0x6F

        sabm_frame = AX25Frame(
            destination=self.remote_addr,
            source=self.local_addr,
            control=control,
            config=self.config,
        )

        await self._send_frame(sabm_frame)
        self.timers.start_t1_sync(self._on_t1_timeout)
        logger.info(
            "connect: sent %s to %s",
            "SABM" if self.config.modulo == 8 else "SABME",
            self.remote_addr.callsign,
        )
        return sabm_frame

    async def disconnect(self) -> AX25Frame:
        """Send the disconnect frame (DISC) to close the link.

        Only allowed when the link is CONNECTED or in TIMER_RECOVERY.
        Transitions to AWAITING_RELEASE and starts T1 to wait for UA.

        Returns:
            The DISC frame that was sent.

        Raises:
            ConnectionStateError: If the connection is not currently open.

        Example::

            disc = await conn.disconnect()
        """
        if self.sm.state not in (AX25State.CONNECTED, AX25State.TIMER_RECOVERY):
            raise ConnectionStateError(
                f"disconnect() called in state {self.sm.state.value} -- "
                f"must be CONNECTED or TIMER_RECOVERY"
            )

        self.sm.transition("disconnect_request")

        disc_frame = AX25Frame(
            destination=self.remote_addr,
            source=self.local_addr,
            control=0x43,   # DISC with P=1
            config=self.config,
        )

        await self._send_frame(disc_frame)
        self.timers.start_t1_sync(self._on_t1_timeout)
        logger.info("disconnect: sent DISC to %s", self.remote_addr.callsign)
        return disc_frame

    # -----------------------------------------------------------------------
    # Data transfer
    # -----------------------------------------------------------------------

    async def send_data(self, data: bytes) -> None:
        """Queue application data to be sent as an I-frame.

        Puts the data in the outgoing queue. If the link is ready (not
        peer_busy and window is open), it tries to send the frame right
        away. Otherwise the data waits until the link clears.

        Args:
            data: The bytes to send. Must not exceed N1 (max_frame) bytes.
                Empty data is silently ignored.

        Raises:
            ConnectionStateError: If the link is not in CONNECTED state.
            FrameError: If the data is longer than N1.

        Example::

            await conn.send_data(b"Hello world")
        """
        if self.sm.state != AX25State.CONNECTED:
            raise ConnectionStateError(
                f"send_data() called in state {self.sm.state.value} -- "
                f"must be CONNECTED"
            )

        if not data:
            logger.warning("send_data: called with empty data -- ignoring")
            return

        if len(data) > self.config.max_frame:
            raise FrameError(
                f"send_data: {len(data)} bytes exceeds N1={self.config.max_frame} -- "
                f"split into smaller chunks"
            )

        self.outgoing_queue.append(data)
        logger.debug(
            "send_data: queued %d bytes (queue now has %d items)",
            len(data), len(self.outgoing_queue),
        )

        if not self.peer_busy:
            await self._transmit_pending()

    async def _transmit_pending(self) -> None:
        """Send as many queued I-frames as the window allows.

        Sends I-frames from the outgoing_queue until the window is full,
        the peer is busy, or the queue is empty. Starts T1 on the first
        frame sent if T1 is not already running.
        """
        modulo = 128 if self.config.modulo == 128 else 8

        while (
            self.flow.can_send_i_frame()
            and self.outgoing_queue
        ):
            data = self.outgoing_queue.pop(0)
            i_frame = self._build_i_frame(data, p_bit=False)
            await self._send_frame(i_frame)
            self.flow.enqueue_i_frame(self.v_s)
            self.v_s = (self.v_s + 1) % modulo

            logger.debug(
                "_transmit_pending: sent I-frame V(S)=%d, queue=%d remaining",
                (self.v_s - 1) % modulo, len(self.outgoing_queue),
            )

            # Start T1 on the first outstanding frame
            if len(self.flow.outstanding_seqs) == 1:
                self.timers.start_t1_sync(self._on_t1_timeout)

    # -----------------------------------------------------------------------
    # Frame builders (internal)
    # -----------------------------------------------------------------------

    def _build_i_frame(self, info: bytes, p_bit: bool = False) -> AX25Frame:
        """Build a data I-frame with the current sequence numbers.

        Packs V(S) and V(R) into the control field according to modulo
        (1 or 2 byte control field).

        Args:
            info: The data payload to send in this I-frame.
            p_bit: True to set the Poll bit in the control field.

        Returns:
            An AX25Frame with the correct control field for the current
            sequence state.
        """
        if self.config.modulo == 8:
            # Modulo-8: control = 0 | V(S)<<1 | P | V(R)<<5
            ns = (self.v_s & 0x07) << 1
            nr = (self.v_r & 0x07) << 5
            p_flag = 0x10 if p_bit else 0x00
            control = ns | nr | p_flag
        else:
            # Modulo-128: 2-byte control field
            # byte 0: V(S)<<1 | 0 | P
            # byte 1: V(R)<<1 | 0
            ns = (self.v_s & 0x7F) << 1
            nr = (self.v_r & 0x7F) << 1
            p_flag = 0x10 if p_bit else 0x00
            control = ns | p_flag | (nr << 8)

        logger.debug(
            "_build_i_frame: V(S)=%d V(R)=%d control=0x%04X info=%d bytes",
            self.v_s, self.v_r, control, len(info),
        )

        return AX25Frame(
            destination=self.remote_addr,
            source=self.local_addr,
            control=control,
            pid=0xF0,   # No Layer 3 protocol
            info=info,
            config=self.config,
        )

    # -----------------------------------------------------------------------
    # Incoming frame processing
    # -----------------------------------------------------------------------

    def process_frame(self, frame: AX25Frame) -> None:
        """Handle an incoming frame and update the connection state.

        Dispatches to the appropriate handler based on frame type
        (U-frame, S-frame, or I-frame), then resets the T3 idle timer.

        Args:
            frame: A decoded AX25Frame received from the transport.

        Example::

            raw = transport.receive_frame()
            frame = AX25Frame.decode(raw, config=conn.config)
            conn.process_frame(frame)
        """
        logger.debug(
            "process_frame: state=%s control=0x%02X",
            self.sm.state.value, frame.control,
        )

        # Dispatch by frame type (bits 0-1 of control field)
        if frame.control & 0x03 == 0x03:
            self._handle_u_frame(frame)
        elif frame.control & 0x03 == 0x01:
            self._handle_s_frame(frame)
        elif frame.control & 0x01 == 0x00:
            self._handle_i_frame(frame)

        # Reset the T3 idle probe timer on any valid received frame
        self.timers.start_t3_sync(self._on_t3_timeout)

    def _handle_u_frame(self, frame: AX25Frame) -> None:
        """Handle an Unnumbered (U) frame.

        Processes UA, SABM, SABME, DISC, XID, DM, and FRMR frames.

        Args:
            frame: The received U-frame.
        """
        # Mask out the P/F bit (bit 4) to get the pure command code
        cmd = frame.control & ~0x10
        p_f = bool(frame.control & 0x10)

        logger.debug(
            "_handle_u_frame: cmd=0x%02X p_f=%s state=%s",
            cmd, p_f, self.sm.state.value,
        )

        # UA (Unnumbered Acknowledgement): 0x63 (mod-8) or 0x6F (mod-128)
        if cmd in (0x63, 0x6F):
            if self.sm.state == AX25State.AWAITING_CONNECTION:
                self.sm.transition("UA_received")
                self.timers.stop_t1_sync()
                logger.info(
                    "Connection established with %s (UA received, modulo %d)",
                    self.remote_addr.callsign,
                    8 if cmd == 0x63 else 128,
                )
                return
            if self.sm.state == AX25State.AWAITING_RELEASE:
                self.sm.transition("UA_received")
                self.timers.stop_t1_sync()
                logger.info(
                    "Disconnection completed (UA received from %s)",
                    self.remote_addr.callsign,
                )
                return

        # SABM (0x2F) or SABME (0x6F): incoming connection request
        elif cmd in (0x2F, 0x6F):
            if self.sm.state not in (AX25State.AWAITING_CONNECTION,):
                event = "SABM_received" if cmd == 0x2F else "SABME_received"
                self.sm.transition(event)
                self._send_ua()
                logger.info(
                    "Incoming connection from %s (%s received)",
                    self.remote_addr.callsign,
                    "SABM" if cmd == 0x2F else "SABME",
                )

        # DISC (0x43): remote station is disconnecting
        elif cmd == 0x43:
            self.sm.transition("DISC_received")
            self._send_ua()
            self.timers.stop_t1_sync()
            self.timers.stop_t3_sync()
            logger.info("Disconnected by peer (%s)", self.remote_addr.callsign)

        # DM (0x0F): remote station is refusing / disconnected
        elif cmd == 0x0F:
            self.sm.transition("DM_received")
            self.timers.stop_t1_sync()
            logger.warning("DM received from %s", self.remote_addr.callsign)

        # FRMR (0x87 with different bit pattern): frame reject
        elif cmd == 0x87:
            self.sm.transition("FRMR_received")
            logger.error("FRMR received from %s", self.remote_addr.callsign)

        # XID (0xAF): parameter negotiation
        elif cmd == 0xAF:
            if frame.info:
                try:
                    remote_params = parse_xid_frame(frame.info)
                    self.negotiated_config = negotiate_config(
                        self.config, remote_params
                    )
                    logger.info(
                        "XID negotiation complete with %s: "
                        "modulo=%d k=%d N1=%d",
                        self.remote_addr.callsign,
                        self.negotiated_config.modulo,
                        self.negotiated_config.window_size,
                        self.negotiated_config.max_frame,
                    )
                    if p_f:
                        self._send_xid_response()
                except NegotiationError as exc:
                    logger.error("XID negotiation failed: %s", exc)
            else:
                logger.warning("XID frame received with no info field -- ignoring")

        else:
            logger.warning(
                "_handle_u_frame: unknown U-frame command 0x%02X -- ignoring", cmd
            )

    def _handle_s_frame(self, frame: AX25Frame) -> None:
        """Handle a Supervisory (S) frame (RR, RNR, REJ, SREJ).

        Extracts N(R) and the P/F bit, updates the acknowledgment state,
        and handles the specific frame type.

        Args:
            frame: The received S-frame.
        """
        modulo_mask = 0x07 if self.config.modulo == 8 else 0x7F
        nr_shift = 5 if self.config.modulo == 8 else 9

        s_type = (frame.control >> 2) & 0x03
        nr = (frame.control >> nr_shift) & modulo_mask
        p_f = bool(frame.control & 0x10)

        logger.debug(
            "_handle_s_frame: s_type=%d nr=%d p_f=%s", s_type, nr, p_f
        )

        # Acknowledge frames up to N(R)
        self.flow.acknowledge_up_to(nr)
        self.v_a = nr

        if s_type == 0x00:   # RR -- Receiver Ready
            self.flow.handle_rr()
            if self.outgoing_queue:
                asyncio.create_task(self._transmit_pending())

        elif s_type == 0x01:  # RNR -- Receiver Not Ready
            self.flow.handle_rnr()

        elif s_type == 0x02:  # REJ -- Go Back N
            self._retransmit_from(nr)

        elif s_type == 0x03:  # SREJ -- Selective Reject
            self._retransmit_specific(nr)

        # Respond to poll with final
        if p_f:
            self._send_rr(f_bit=True)

    def _handle_i_frame(self, frame: AX25Frame) -> None:
        """Handle an Information (I) frame carrying data.

        Checks the sequence number against V(R). If in sequence,
        stores the data and acknowledges with RR. If out of sequence,
        requests retransmission with SREJ.

        Args:
            frame: The received I-frame.
        """
        modulo_mask = 0x07 if self.config.modulo == 8 else 0x7F
        ns_shift = 1
        nr_shift = 5 if self.config.modulo == 8 else 9

        ns = (frame.control >> ns_shift) & modulo_mask
        nr = (frame.control >> nr_shift) & modulo_mask
        p_bit = bool(frame.control & 0x10)

        logger.debug(
            "_handle_i_frame: N(S)=%d expected V(R)=%d N(R)=%d", ns, self.v_r, nr
        )

        # Acknowledge frames the other side says we sent
        self.flow.acknowledge_up_to(nr)
        self.v_a = nr

        if ns != self.v_r:
            logger.warning(
                "_handle_i_frame: out-of-sequence N(S)=%d, expected %d -- sending SREJ",
                ns, self.v_r,
            )
            self._send_srej(self.v_r)
            return

        # In-sequence: accept the data
        modulo = 128 if self.config.modulo == 128 else 8
        self.v_r = (self.v_r + 1) % modulo
        self.incoming_buffer.append(frame.info)

        logger.debug(
            "_handle_i_frame: accepted %d bytes, V(R) now %d",
            len(frame.info), self.v_r,
        )

        # Acknowledge receipt
        self._send_rr(f_bit=p_bit)

    # -----------------------------------------------------------------------
    # Frame sender helpers (internal)
    # -----------------------------------------------------------------------

    async def _send_frame(self, frame: AX25Frame) -> None:
        """Send a frame via the transport layer.

        If no transport is attached, logs the frame bytes instead so the
        class can be used in testing without hardware.

        Args:
            frame: The frame to send.

        Raises:
            TransportError: If the transport raises an exception while
                sending.
        """
        if self.transport is not None:
            try:
                self.transport.send_frame(frame)
                logger.debug(
                    "_send_frame: sent control=0x%02X to %s",
                    frame.control, frame.destination.callsign,
                )
            except Exception as exc:
                logger.error("_send_frame: transport error: %s", exc)
                raise TransportError(
                    f"Failed to send frame to {frame.destination.callsign}: {exc}"
                ) from exc
        else:
            logger.debug(
                "_send_frame (no transport): would send %d bytes: %s",
                len(frame.encode()), frame.encode().hex(),
            )

    def _send_ua(self) -> None:
        """Send Unnumbered Acknowledgement (UA) with F=1."""
        ua_frame = AX25Frame(
            destination=self.remote_addr,
            source=self.local_addr,
            control=0x63,   # UA with F=1
            config=self.config,
        )
        asyncio.create_task(self._send_frame(ua_frame))
        logger.debug("_send_ua: sending UA")

    def _send_rr(self, f_bit: bool = False) -> None:
        """Send Receiver Ready (RR) supervisory frame."""
        control = 0x01 | (self.v_r << 5) | (0x10 if f_bit else 0x00)
        rr_frame = AX25Frame(
            destination=self.remote_addr,
            source=self.local_addr,
            control=control,
            config=self.config,
        )
        asyncio.create_task(self._send_frame(rr_frame))
        logger.debug("_send_rr: V(R)=%d f_bit=%s", self.v_r, f_bit)

    def _send_rnr(self, f_bit: bool = False) -> None:
        """Send Receiver Not Ready (RNR) supervisory frame."""
        control = 0x05 | (self.v_r << 5) | (0x10 if f_bit else 0x00)
        rnr_frame = AX25Frame(
            destination=self.remote_addr,
            source=self.local_addr,
            control=control,
            config=self.config,
        )
        asyncio.create_task(self._send_frame(rnr_frame))
        logger.debug("_send_rnr: V(R)=%d f_bit=%s", self.v_r, f_bit)

    def _send_srej(self, nr: int) -> None:
        """Send Selective Reject (SREJ) for one specific frame."""
        control = 0x0D | (nr << 5) | 0x10   # SREJ with F=1
        srej_frame = AX25Frame(
            destination=self.remote_addr,
            source=self.local_addr,
            control=control,
            config=self.config,
        )
        asyncio.create_task(self._send_frame(srej_frame))
        logger.debug("_send_srej: nr=%d control=0x%02X", nr, control)

    def _send_xid_response(self) -> None:
        """Send an XID response frame with our negotiated parameters."""
        cfg = self.negotiated_config or self.config
        xid_data = build_xid_frame(cfg)
        xid_frame = AX25Frame(
            destination=self.remote_addr,
            source=self.local_addr,
            control=0xAF,   # XID with F=1
            info=xid_data,
            config=cfg,
        )
        asyncio.create_task(self._send_frame(xid_frame))
        logger.debug("_send_xid_response: %d bytes of XID data", len(xid_data))

    # -----------------------------------------------------------------------
    # Timer callbacks
    # -----------------------------------------------------------------------

    def _on_t1_timeout(self) -> None:
        """Handle T1 acknowledgment timer expiration.

        Increments the retry counter. If we have exceeded N2 retries,
        gives up and lets the state machine move to DISCONNECTED.
        Otherwise retransmits outstanding frames and restarts T1.
        """
        self.retry_count += 1
        logger.warning(
            "T1 timeout (retry %d of %d)",
            self.retry_count, self.config.retry_count,
        )

        if self.retry_count >= self.config.retry_count:
            logger.error(
                "T1 exceeded maximum retries (%d) -- disconnecting",
                self.config.retry_count,
            )
            self.sm.transition("T1_timeout")
            self.timers.stop_t1_sync()
            self.timers.stop_t3_sync()
            return

        # Retransmit based on current state -- do NOT call sm.transition here,
        # because in AWAITING_CONNECTION that would immediately move to DISCONNECTED
        # (the state machine's T1_timeout event is reserved for when retries are
        # exhausted).  We just re-send the frame and restart the timer.
        if self.sm.state in (
            AX25State.AWAITING_CONNECTION,
            AX25State.CONNECTED,
            AX25State.TIMER_RECOVERY,
        ):
            self._retransmit_all_sync()
            self.timers.start_t1_sync(self._on_t1_timeout)

    def _on_t3_timeout(self) -> None:
        """Handle T3 idle channel probe timer expiration.

        Sends a probe frame (RR with P=1) to check if the remote
        station is still reachable.
        """
        logger.warning("T3 idle timeout -- probing %s", self.remote_addr.callsign)
        self._send_rr(f_bit=True)

    async def _process_timers(self) -> None:
        """Process any pending timer events (async stub).

        This is an async entry point for timer processing in asyncio-driven
        applications.  Synchronous timer callbacks (T1/T3) fire on daemon
        threads and directly call _on_t1_timeout / _on_t3_timeout, so there
        is usually nothing extra to do here.  Override or extend in subclasses
        that use a purely async timer approach.
        """
        # Nothing to flush -- sync timer callbacks run inline.
        return

    # -----------------------------------------------------------------------
    # Retransmission helpers (internal)
    # -----------------------------------------------------------------------

    def _retransmit_all_sync(self) -> None:
        """Retransmit the connect frame or all outstanding I-frames (sync).

        If we are in AWAITING_CONNECTION, re-sends the SABM/SABME.
        Otherwise re-sends I-frames starting from V(A).

        This is the synchronous version used by T1 timeout callbacks,
        which run in a daemon thread and cannot use asyncio.
        """
        logger.warning("_retransmit_all_sync: retransmitting outstanding frames")

        if self.sm.state == AX25State.AWAITING_CONNECTION:
            control = 0x2F if self.config.modulo == 8 else 0x6F
            sabm_frame = AX25Frame(
                destination=self.remote_addr,
                source=self.local_addr,
                control=control,
                config=self.config,
            )
            if self.transport is not None:
                try:
                    self.transport.send_frame(sabm_frame)
                    logger.debug("_retransmit_all_sync: re-sent SABM/SABME")
                except Exception as exc:
                    logger.error("_retransmit_all_sync: send failed: %s", exc)
            else:
                logger.debug(
                    "_retransmit_all_sync (no transport): %s",
                    sabm_frame.encode().hex(),
                )
        else:
            logger.info(
                "_retransmit_all_sync: I-frame retransmission from V(A)=%d "
                "not yet implemented",
                self.v_a,
            )

    def _retransmit_from(self, nr: int) -> None:
        """Retransmit all I-frames from N(R) onward (after REJ).

        This is called when we receive a REJ frame. The remote station
        is saying it has not received frame nr and wants it (and all
        following frames) re-sent.

        Args:
            nr: The sequence number of the first frame to re-send.
        """
        logger.warning(
            "_retransmit_from: REJ received -- retransmitting from N(R)=%d", nr
        )
        # Full retransmit from nr is not yet implemented.
        # When implemented, this will resend all I-frames with N(S) >= nr.

    def _retransmit_specific(self, nr: int) -> None:
        """Retransmit a single I-frame after SREJ.

        This is called when we receive a SREJ frame. The remote station
        is asking us to re-send just one specific frame.

        Args:
            nr: The sequence number of the single frame to re-send.
        """
        logger.warning(
            "_retransmit_specific: SREJ received -- retransmitting frame N(S)=%d", nr
        )
        # Single-frame retransmit is not yet implemented.
        # When implemented, this will find the buffered copy of frame nr and resend it.

    # -----------------------------------------------------------------------
    # Async frame processing helpers
    # -----------------------------------------------------------------------

    async def _process_incoming(self) -> None:
        """Read one frame from the transport and process it.

        Designed to be called in an event loop. Reads one frame from the
        transport (if available), decodes it, and passes it to
        process_frame().

        Does nothing if no transport is attached or no frame is available.
        """
        if self.transport is None:
            return

        frame_data = self.transport.receive_frame()
        if frame_data:
            try:
                frame = AX25Frame.decode(frame_data, self.config)
                self.process_frame(frame)
            except Exception as exc:
                logger.error("_process_incoming: could not decode frame: %s", exc)
