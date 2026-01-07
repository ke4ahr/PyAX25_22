# tests/test_statemachine.py

Unit tests for the AX.25 state machine.

Covers all states and transitions per AX.25 v2.2 SDL diagrams.


    import pytest

    from pyax25_22.core.statemachine import AX25StateMachine, AX25State
    from pyax25_22.core.exceptions import ConnectionStateError


    @pytest.fixture
    def sm_mod8():
        """State machine with modulo 8."""
        return AX25StateMachine(layer3_initiated=True)


    @pytest.fixture
    def sm_mod128():
        """State machine with modulo 128."""
        sm = AX25StateMachine(layer3_initiated=True)
        sm.modulo = 128
        return sm


    def test_initial_state(sm_mod8):
        """Initial state is DISCONNECTED."""
        assert sm_mod8.state == AX25State.DISCONNECTED


    def test_connect_request_without_layer3(sm_mod8):
        """Connect request without layer3_initiated raises error."""
        sm = AX25StateMachine(layer3_initiated=False)
        with pytest.raises(ConnectionStateError):
            sm.transition("connect_request")


    def test_connect_request(sm_mod8):
        """Connect request from DISCONNECTED -> AWAITING_CONNECTION."""
        sm_mod8.transition("connect_request")
        assert sm_mod8.state == AX25State.AWAITING_CONNECTION


    def test_sabm_received_from_disconnected(sm_mod8):
        """SABM received -> CONNECTED."""
        sm_mod8.transition("SABM_received")
        assert sm_mod8.state == AX25State.CONNECTED


    def test_sabme_received_mod128(sm_mod128):
        """SABME received -> CONNECTED (mod128)."""
        sm_mod128.transition("SABME_received")
        assert sm_mod128.state == AX25State.CONNECTED


    def test_disc_from_disconnected(sm_mod8):
        """DISC in DISCONNECTED is ignored."""
        sm_mod8.transition("DISC_received")  # No change
        assert sm_mod8.state == AX25State.DISCONNECTED


    def test_ua_from_awaiting_connection(sm_mod8):
        """UA in AWAITING_CONNECTION -> DISCONNECTED."""
        sm_mod8.transition("connect_request")
        sm_mod8.transition("UA_received")
        assert sm_mod8.state == AX25State.DISCONNECTED


    def test_timeout_from_awaiting_connection(sm_mod8):
        """T1 timeout in AWAITING_CONNECTION -> DISCONNECTED."""
        sm_mod8.transition("connect_request")
        sm_mod8.transition("T1_timeout")
        assert sm_mod8.state == AX25State.DISCONNECTED


    def test_disconnect_request_from_connected(sm_mod8):
        """Disconnect request in CONNECTED -> AWAITING_RELEASE."""
        sm_mod8.transition("SABM_received")
        sm_mod8.transition("disconnect_request")
        assert sm_mod8.state == AX25State.AWAITING_RELEASE


    def test_disc_from_connected(sm_mod8):
        """DISC in CONNECTED -> DISCONNECTED."""
        sm_mod8.transition("SABM_received")
        sm_mod8.transition("DISC_received")
        assert sm_mod8.state == AX25State.DISCONNECTED


    def test_rnr_in_connected(sm_mod8):
        """RNR supervisory in connected sets peer_busy."""
        sm_mod8.transition("SABM_received")
        assert not sm_mod8.peer_busy
        sm_mod8.transition("supervisory_received", frame_type="RNR")
        assert sm_mod8.peer_busy


    def test_rr_in_connected(sm_mod8):
        """RR supervisory in connected clears peer_busy."""
        sm_mod8.transition("SABM_received")
        sm_mod8.peer_busy = True
        sm_mod8.transition("supervisory_received", frame_type="RR")
        assert not sm_mod8.peer_busy


    def test_rej_in_connected(sm_mod8):
        """REJ supervisory in connected sets reject_sent."""
        sm_mod8.transition("SABM_received")
        assert not sm_mod8.reject_sent
        sm_mod8.transition("supervisory_received", frame_type="REJ")
        assert sm_mod8.reject_sent


    def test_srej_in_connected(sm_mod8):
        """SREJ supervisory in connected sets srej_sent."""
        sm_mod8.transition("SABM_received")
        assert not sm_mod8.srej_sent
        sm_mod8.transition("supervisory_received", frame_type="SREJ")
        assert sm_mod8.srej_sent


    def test_t1_timeout_from_connected(sm_mod8):
        """T1 timeout in CONNECTED -> TIMER_RECOVERY."""
        sm_mod8.transition("SABM_received")
        sm_mod8.transition("T1_timeout")
        assert sm_mod8.state == AX25State.TIMER_RECOVERY


    def test_ack_response_from_timer_recovery(sm_mod8):
        """Ack in TIMER_RECOVERY -> CONNECTED."""
        sm_mod8.transition("SABM_received")
        sm_mod8.transition("T1_timeout")
        sm_mod8.transition("ack_received")
        assert sm_mod8.state == AX25State.CONNECTED


    def test_t1_timeout_from_timer_recovery(sm_mod8):
        """T1 timeout in TIMER_RECOVERY -> CONNECTED."""
        sm_mod8.transition("SABM_received")
        sm_mod8.transition("T1_timeout")
        sm_mod8.transition("T1_timeout")
        assert sm_mod8.state == AX25State.CONNECTED


    def test_ua_from_awaiting_release(sm_mod8):
        """UA in AWAITING_RELEASE -> DISCONNECTED."""
        sm_mod8.transition("SABM_received")
        sm_mod8.transition("disconnect_request")
        sm_mod8.transition("UA_received")
        assert sm_mod8.state == AX25State.DISCONNECTED


    def test_t1_timeout_from_awaiting_release(sm_mod8):
        """T1 timeout in AWAITING_RELEASE -> DISCONNECTED."""
        sm_mod8.transition("SABM_received")
        sm_mod8.transition("disconnect_request")
        sm_mod8.transition("T1_timeout")
        assert sm_mod8.state == AX25State.DISCONNECTED


    def test_xid_from_awaiting_xid(sm_mod8):
        """XID in AWAITING_XID -> CONNECTED."""
        sm_mod8.state = AX25State.AWAITING_XID
        sm_mod8.transition("XID_received")
        assert sm_mod8.state == AX25State.CONNECTED


    def test_t1_timeout_from_awaiting_xid(sm_mod8):
        """T1 timeout in AWAITING_XID -> DISCONNECTED."""
        sm_mod8.state = AX25State.AWAITING_XID
        sm_mod8.transition("T1_timeout")
        assert sm_mod8.state == AX25State.DISCONNECTED


    def test_invalid_event_raises_error(sm_mod8):
        """Invalid event raises ConnectionStateError."""
        with pytest.raises(ConnectionStateError):
            sm_mod8.transition("invalid_event")


    def test_sequence_increment_mod8(sm_mod8):
        """Sequence numbers wrap at 7 in mod8."""
        sm_mod8.v_s = 7
        sm_mod8.increment_vs()
        assert sm_mod8.v_s == 0


    def test_sequence_increment_mod128(sm_mod128):
        """Sequence numbers wrap at 127 in mod128."""
        sm_mod128.v_s = 127
        sm_mod128.increment_vs()
        assert sm_mod128.v_s == 0


    def test_modulo_mask(sm_mod8, sm_mod128):
        """Modulo mask is correct."""
        assert sm_mod8.modulo_mask == 0x07
        assert sm_mod128.modulo_mask == 0x7F"