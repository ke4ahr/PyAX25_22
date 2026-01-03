# PyAX25_22 API Reference

## Core Module (`pyax25-22.core`)

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

## Interfaces Module (`pyax25-22.interfaces`)

#### `KISSInterface`
Multi-drop KISS serial/TCP.

    kiss = KISSInterface(\"/dev/ttyUSB0\", tnc_address=1)
    kiss.connect()

#### `AGWPEInterface`
Full TCP/IP AGWPE client.

    agwpe = AGWPEInterface()
    agwpe.connect()
    agwpe.enable_monitoring()

#### `TransportInterface`
Abstract base for all transports.

## Utilities

#### Logging
`get_logger(__name__)` for structured logs.

All public APIs have type hints, docstrings, logging, and tests.
License: LGPLv3.0  
Copyright (C) 2025-2026 Kris Kirby, KE4AHR
