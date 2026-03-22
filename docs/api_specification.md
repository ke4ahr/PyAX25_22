# PyAX25_22 API Reference

## Core Module (`pyax25_22.core`)

### Framing

#### `AX25Frame`
Complete AX.25 frame with encoding/decoding.

    frame = AX25Frame(
        destination=AX25Address(\"DEST\", ssid=1),
        source=AX25Address(\"SRC\", ssid=2),
        control=0x03,
        pid=0xF0,
        info=b\"Hello\"
    )
    raw = frame.encode()
    decoded = AX25Frame.decode(raw)

#### `AX25Address`
Address field with callsign, SSID, C/H bits.

    addr = AX25Address(\"KE4AHR\", ssid=1, c_bit=True)

### Connection Management

#### `AX25Connection`
Connected mode session.

    conn = AX25Connection(local, remote, initiate=True)
    conn.connect()
    conn.send_data(b\"Test\")
    conn.disconnect()

#### `AX25StateMachine`
State transitions per v2.2 SDL.

### Configuration

#### `AX25Config`
Immutable protocol parameters.

    config = AX25Config(modulo=128, window_size=63)

### Flow Control

#### `AX25FlowControl`
Window, REJ/SREJ, busy handling.

### Timers

#### `AX25Timers`
Adaptive T1 and T3 management.

### Negotiation

#### XID Functions
`build_xid_frame()`, `parse_xid_frame()`, `negotiate_config()`

### Exceptions
Full hierarchy under `AX25Error`.

## Interfaces Module (`pyax25_22.interfaces`)

#### `TransportInterface`
Abstract base class for all transports (`interfaces/transport.py`).

### KISS Subpackage (`pyax25_22.interfaces.kiss`)

#### `KISSBase`
Abstract transport-agnostic KISS framing (`_stuff`, `_destuff`, `_process_byte`).

#### `KISSSerial`
KISS over serial port (pyserial, threaded reader).

    tnc = KISSSerial("/dev/ttyUSB0")

#### `KISSTCP`
KISS over TCP socket with keepalive.

    tnc = KISSTCP("localhost", 8001)

#### `XKISS` / `XKISSTCP`
G8BPQ multi-drop KISS with active/passive polling and per-port DIGIPEAT.

    xk = XKISS("/dev/ttyUSB0", our_calls={"KE4AHR"}, digipeat_ports={0})
    xk.enable_digipeat(0)

#### `SMACK` / `SMACKTCP`
CRC-16 KISS (Stuttgart Modified Amateur Radio CRC-KISS) with auto-switch.

    smack = SMACK("/dev/ttyUSB0")

#### `AsyncKISSTCP`
asyncio-based KISS over TCP. Supports plain and coroutine `on_frame` callbacks
and `asyncio.Queue` delivery.

    tnc = AsyncKISSTCP("localhost", 8001, on_frame=my_callback)
    await tnc.connect()
    await tnc.send(raw_ax25_bytes)

#### `KISSInterface` (alias)
Backward-compatible alias for `KISSSerial`.

### AGW Subpackage (`pyax25_22.interfaces.agw`)

#### `AGWPEClient`
Full AGWPE TCP client. Exponential backoff, TCP keepalive, all frame types.
Callbacks: `on_frame`, `on_connected_data`, `on_outstanding`, `on_heard_stations`.

    client = AGWPEClient("localhost", 8000, callsign="KE4AHR")
    client.connect()
    client.send_ui(port=0, dest="APRS", src="KE4AHR", pid=0xF0, info=b"Hello")

#### `AsyncAGWPEClient`
asyncio-based AGWPE client. All callbacks support plain or coroutine functions.

    client = AsyncAGWPEClient("localhost", 8000, callsign="KE4AHR")
    await client.connect()
    await client.send_ui(port=0, dest="APRS", src="KE4AHR", pid=0xF0, info=b"Hello")

#### `AGWSerial`
TCP server bridging AGWPE protocol clients to a serial KISS TNC.

    bridge = AGWSerial("/dev/ttyUSB0", 9600, agw_host="0.0.0.0", agw_port=8000)
    bridge.start()

### FX.25 Subpackage (`pyax25_22.interfaces.fx25`)

#### `FX25Encoder`
Wraps AX.25 frames with Reed-Solomon FEC (TAPR FX.25). Auto-selects smallest tag.

    enc = FX25Encoder()
    packet = enc.encode(raw_ax25_frame)

#### `FX25Decoder`
Stream state machine (HUNT / DATA states) that strips FX.25 framing and
error-corrects the enclosed AX.25 frame.

    dec = FX25Decoder(on_frame=my_callback)
    dec.feed(received_bytes)

### Legacy Shims

#### `KISSInterface` / `AGWPEInterface`
Legacy module-level aliases kept for backward compatibility.
Import from `pyax25_22.interfaces.kiss` / `pyax25_22.interfaces.agw` for new code.
## Utilities

#### Logging
`get_logger(__name__)` for structured logs.

All public APIs have type hints, docstrings, logging, and tests.
License: LGPL-3.0-or-later  
Copyright (C) 2025-2026 Kris Kirby, KE4AHR

