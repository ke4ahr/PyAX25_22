@pytest.mark.asyncio
async def test_mod128_lifecycle(mock_connection_mod128):
    """Test connection with modulo 128."""
    conn = mock_connection_mod128

    await conn.connect()
    # Simulate UA response (not SABME - that would be for incoming connection)
    conn.transport.inject_frame(
        AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("DEST"),
            control=0x6F,  # UA (extended)
        ).encode()
    )
    await conn._process_incoming()

    assert conn.state == AX25State.CONNECTED
    assert conn.config.modulo == 128

    await conn.disconnect()
    assert conn.state == AX25State.AWAITING_RELEASE

    # Simulate UA
    conn.transport.inject_frame(
        AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("DEST"),
            control=0x6F,
        ).encode()
    )
    await conn._process_incoming()

    assert conn.state == AX25State.DISCONNECTED

@pytest.mark.asyncio
async def test_retransmission_on_timeout(mock_connection_mod8):
    """Test retransmission behavior when T1 expires."""
    conn = mock_connection_mod8

    # Create new config with short timeout
    new_config = AX25Config(
        modulo=conn.config.modulo,
        max_frame=conn.config.max_frame,
        window_size=conn.config.window_size,
        t1_timeout=0.5,
        t3_timeout=conn.config.t3_timeout,
        retry_count=conn.config.retry_count,
        tx_delay=conn.config.tx_delay,
        tx_tail=conn.config.tx_tail,
        persistence=conn.config.persistence,
        slot_time=conn.config.slot_time
    )
    conn.config = new_config
    conn.timers.rto = 0.5

    await conn.connect()
    assert conn.state == AX25State.AWAITING_CONNECTION

    # Wait for T1 timeout
    await asyncio.sleep(0.7)

    # Trigger timeout manually
    conn._on_t1_timeout()

    # Should have retransmitted
    assert len(conn.transport.sent_frames) >= 2  # Original + retransmit

@pytest.mark.asyncio
async def test_idle_timeout(mock_connection_mod8):
    """Test T3 idle timeout handling."""
    conn = mock_connection_mod8

    await conn.connect()
    # Simulate UA
    conn.transport.inject_frame(
        AX25Frame(
            destination=AX25Address("TEST"),
            source=AX25Address("DEST"),
            control=0x63,
        ).encode()
    )
    await conn._process_incoming()

    # Use minimum valid T3 timeout (10 seconds)
    conn.timers.t3_current = 10.0

    # Start T3 timer
    conn.timers.start_t3_sync(lambda: None)

    # Verify timer was started
    assert conn.timers._t3_thread_timer is not None

    # Clean up
    conn.timers.stop_t3_sync()
