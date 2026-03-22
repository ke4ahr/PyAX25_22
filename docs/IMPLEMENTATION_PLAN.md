# PyAX25_22 Integration & Enhancement Implementation Plan

**Date:** 2026-03-21
**Author:** Kris Kirby, KE4AHR
**Status:** In Progress -- Phase 0 (Analysis Complete)

## Objective

Combine all functionality of:
- PyXKISS (KISS, XKISS, SMACK protocol interfaces)
- PyAGW3 (AGW Packet Engine TCP client)
- Reference PyPACSAT (high-level consumer of these libraries)

...into PyAX25_22, and add FX.25 (Forward Error Correction) support.

**End goal:** PyAX25_22 becomes the single core packet radio library for PyPACSAT.
PyPACSAT may be ahead in features -- PyAX25_22 must evolve to be a stable dependency,
eliminating the need for PyPACSAT to vendor PyXKISS/PyAGW3 separately. The result will be
a single, complete AX.25 Level 2.2 library that supports:

- Standard KISS over serial port
- KISS over TCP
- Extended KISS (XKISS / Multi-Drop / BPQKISS) with G8BPQ addressing
- SMACK (Stuttgart Modified Amateur Radio CRC-KISS) with CRC-16
- Polled KISS (host initiates polls) and polling KISS (responds to external polls)
- AGW over TCP (full AGWPE protocol client)
- AGW protocol interfacing to KISS TNCs on serial ports
- FX.25 (Reed-Solomon FEC wrapper for AX.25 frames)
- Synchronous (threaded) and asynchronous (asyncio) operation

---

## Phase 0 -- Analysis (COMPLETE)

Analysis of all source repos complete. See HANDOFF.md for detailed findings.

### Key Findings

1. **PyAX25_22** has a solid core (framing, state machine, connected mode) but thin interfaces.
2. **PyXKISS** has the most complete KISS/XKISS/SMACK implementation -- should be primary source.
3. **PyAGW3** has a more robust AGWPE client than PyAX25_22 -- should be merged in.
4. **PyPACSAT** is a higher-level application that depends on these. Will benefit from the merge.
5. **FX.25** is not implemented anywhere -- new implementation required.

---

## Phase 1 -- Interface Module Restructure

### 1.1 Expand KISS interface hierarchy

Target: `src/pyax25_22/interfaces/`

```
interfaces/
  transport.py     -- TransportInterface ABC (exists, keep)
  kiss/
    __init__.py    -- Re-exports
    base.py        -- KISSBase (framing, escaping, destuffing)
    serial.py      -- KISSSerial (serial port, replaces kiss.py)
    tcp.py         -- KISSTCP (KISS over TCP, new)
    xkiss.py       -- XKISS extends KISSBase (from PyXKISS)
    smack.py       -- SMACK extends XKISS (from PyXKISS)
    constants.py   -- FEND/FESC/CMD_* constants (from PyXKISS)
    exceptions.py  -- KISS-specific exceptions
  agw/
    __init__.py    -- Re-exports
    client.py      -- AGWPEClient (enhanced from PyAGW3)
    serial.py      -- AGWSerial (AGW to serial KISS bridge)
    constants.py   -- AGWPE frame types and constants
    exceptions.py  -- AGW-specific exceptions
  fx25/
    __init__.py    -- Re-exports
    encoder.py     -- FX25Encoder (wraps AX.25 frame with FEC)
    decoder.py     -- FX25Decoder (strips FEC, recovers AX.25)
    rs.py          -- Reed-Solomon FEC implementation
    constants.py   -- FX.25 tag bytes and constants
```

### 1.2 Preserve backward compatibility

Keep `interfaces/kiss.py` and `interfaces/agwpe.py` as thin re-export shims pointing to
the new submodule structure.

---

## Phase 2 -- KISS/XKISS/SMACK Integration (from PyXKISS)

### 2.1 KISSBase

Extract framing logic from PyXKISS kiss.py into a transport-agnostic base class:
- `send(payload, cmd)` -- encode and write KISS frame
- `_destuff(data)` -- FESC/TFEND/TFESC removal
- `_process_byte(byte)` -- state machine for frame assembly
- Must NOT depend on serial or TCP -- delegate I/O to subclass

### 2.2 KISSSerial

Extends KISSBase with pyserial:
- Constructor: `device`, `baudrate`, `timeout`
- Opens serial port
- Starts reader thread
- Implements `write(data)` called by KISSBase

### 2.3 KISSTCP

Extends KISSBase with TCP socket:
- Constructor: `host`, `port`, `timeout`
- Opens TCP connection (optionally with TCP keepalive)
- Starts reader thread
- Implements `write(data)` called by KISSBase

### 2.4 XKISS (from PyXKISS xkiss.py)

Extends KISSBase (or KISSSerial/KISSTCP via composition):
- Multi-drop addressing: high nibble = TNC address (0-15)
- Active polling: background thread sends 0x0E poll frames
- Passive polling: queues incoming data, flushes on poll receipt
- Optional XOR checksum (Kantronics/BPQ32 CHECKSUM mode)
- Per-port receive queues with configurable max size

### 2.5 SMACK (from PyXKISS smack.py)

Extends XKISS:
- Bit 7 flag on data command byte (0x80+)
- CRC-16 (poly 0x8005, init 0x0000, LSB-first append)
- Auto-switch: TX starts plain, enables CRC after first valid CRC frame received
- Fix bug: add missing `import threading` in smack.py

---

## Phase 3 -- AGW Integration (from PyAGW3)

### 3.1 Enhanced AGWPEClient

Merge PyAGW3's AGWPEClient into `interfaces/agw/client.py`:
- All frame types: R, X, G, m, H, Y, y, K, k, U, V, D, d, C, c, v, T, P
- Exponential backoff retry on connect (from PyAGW3)
- TCP keepalive (SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT)
- Callback dispatch per DataKind
- Synchronous (threaded) and async (asyncio) variants

### 3.2 AGWSerial (AGW to Serial KISS Bridge)

New class: `interfaces/agw/serial.py`
- Implements AGW framing on top of a serial KISS TNC
- Translates between AGW frame types and KISS frames
- Allows BPQ32-style AGW clients to use serial TNCs directly
- Reference: BPQ32 source for AGW-to-KISS mapping

---

## Phase 4 -- FX.25 Support

FX.25 is a forward error correction extension for AX.25 frames, using Reed-Solomon codes.
Reference: TAPR FX.25 specification (fx25.pdf).

### 4.1 FX.25 Frame Structure

```
[PREAMBLE: 0x00 * N]
[CONNECT TAG: 8 bytes]  -- identifies correlation tag + RS parameters
[AX.25 FRAME DATA]     -- original AX.25 frame (0 to N1 bytes)
[PAD BYTES]            -- zeros to reach block size
[CHECK BYTES]          -- RS check symbols
```

### 4.2 Correlation Tags (FX.25 Standard)

| Tag      | Data Bytes | Check Bytes | Total |
|----------|-----------|------------|-------|
| 0x01     | 239       | 16         | 255   |
| 0x02     | 239       | 32         | 255+16 = 271 |
| 0x03     | 239       | 64         | 303   |
| 0x04     | 128       | 16         | 144   |
| 0x05     | 128       | 32         | 160   |
| 0x06     | 128       | 64         | 192   |
| 0x07     | 64        | 16         | 80    |
| 0x08     | 64        | 32         | 96    |
| 0x09     | 32        | 16         | 48    |
| 0x0A     | 32        | 32         | 64    |

### 4.3 Implementation Steps

1. `fx25/rs.py` -- Reed-Solomon GF(2^8) encoder/decoder (pure Python)
2. `fx25/constants.py` -- Tag bytes, RS parameters
3. `fx25/encoder.py` -- Wraps AX.25 frame bytes with FX.25 preamble + FEC
4. `fx25/decoder.py` -- Strips FX.25 wrapper, corrects errors, returns AX.25 frame
5. Integration with KISSSerial/KISSTCP as optional layer

---

## Phase 5 -- Async Support

### 5.1 AsyncKISSTCP

asyncio-native version of KISSTCP:
- `await connect()`
- `await send(payload)`
- `await receive()` -- returns next frame
- Frame callback via asyncio Queue

### 5.2 AsyncAGWPEClient

asyncio-native AGWPE client:
- Already partially designed in interfaces/agwpe.py
- Full async send/receive loop using asyncio streams

---

## Phase 6 -- Testing & Documentation

### 6.1 Tests

For each new/merged module:
- Unit tests with mocked serial/socket
- Integration tests (where hardware not required)
- FX.25 codec tests with known vectors

### 6.2 Documentation

- Update docs/api_specification.md with new interface classes
- Update docs/compliance.md with FX.25 support
- Create docs/fx25_spec.md
- Create docs/kiss_tcp_spec.md
- Update README.md

---

## Implementation Order (recommended)

1. Phase 1: Restructure interfaces/ directory
2. Phase 2: KISS/XKISS/SMACK (most critical for PyPACSAT)
3. Phase 3.1: Enhanced AGWPE client
4. Phase 4: FX.25 (separate, lower dependency)
5. Phase 3.2: AGWSerial bridge
6. Phase 5: Async support
7. Phase 6: Tests and documentation

---

## Security Notes

- Serial port access: validate device path, no shell injection
- TCP connections: validate host/port, use timeout
- Frame size limits: enforce max frame size to prevent memory exhaustion
- FEC decode: limit error correction attempts, reject obviously malformed frames
- No credentials stored in plain text (AGW login uses 'T' frame -- document risk)

---

*This document is updated as implementation progresses.*
*License: LGPL-3.0-or-later*
*Copyright (C) 2025-2026 Kris Kirby, KE4AHR*

Copyright (C) 2026 Kris Kirby, KE4AHR
