# PyAX25_22 Integration Project -- Handoff Document

**Version:** 1.0
**Date:** 2026-03-21
**Author:** Kris Kirby, KE4AHR
**Purpose:** Session handoff and technical reference for developers continuing this work.

---

## 1. Project Overview

**End Goal:** PyAX25_22 becomes the unified packet radio core library that PyPACSAT and future
applications depend on. PyPACSAT may be ahead of the other repos in features, so the library
must evolve to support it as a stable dependency rather than PyPACSAT vendoring everything.

The immediate goal is to combine the functionality of four related amateur packet radio Python
libraries into a single, complete AX.25 Level 2.2 implementation:

| Repository | Purpose | Status |
|-----------|---------|--------|
| PyAX25_22 | Core AX.25 framing + state machine + interfaces | Active -- primary target |
| PyXKISS   | KISS/XKISS/SMACK TNC interface library | Merge source |
| PyAGW3    | AGW Packet Engine TCP client library | Merge source |
| PyPACSAT  | PACSAT ground station application | High-level consumer |

Plus adding:
- **FX.25** support (Reed-Solomon FEC for AX.25)
- **KISS over TCP** support
- **AGW-to-KISS serial bridge**
- Full **async (asyncio)** transport variants

Reference implementation for protocol details: BPQ32 (https://github.com/g8bpq/BPQ32)

---

## 2. Repository Locations

```
~/tmp/code/ham/packet/PyAX25_22/PyAX25_22    -- target (this repo)
~/tmp/code/ham/packet/PyAGW3/PyAGW3          -- merge source
~/tmp/code/ham/packet/PyXKISS/PyXKISS        -- merge source
~/tmp/code/ham/packet/PyPACSAT/PyPACSAT      -- high-level reference
```

---

## 3. Current State Analysis

### 3.1 PyAX25_22 (v1.0.1 -- all phases complete; CHANGELOG current)

**Strengths:**
- Complete AX.25 frame encoding/decoding (AX25Frame, AX25Address)
- Proper bit stuffing/destuffing (CRC-16/CCITT-FALSE)
- Full state machine (6 states per v2.2 SDL)
- AX25Connection class for connected-mode sessions
- AX25Config with validated parameters (mod 8 and mod 128)
- AX25FlowControl, AX25Timers, XID negotiation
- TransportInterface ABC
- KISSInterface (basic serial, multi-drop addressing)
- AGWPEInterface (basic TCP client)

**Weaknesses / Gaps:**
- No XKISS (multi-drop polling, XOR checksum)
- No SMACK (CRC-16 protection)
- No KISS over TCP
- No FX.25
- AGWPEInterface missing: exponential backoff, TCP keepalive, frame types T/P/v/m/H
- State machine incomplete in some edge cases (TIMER_RECOVERY transitions)
- `KISSInterface.set_parameter()` is broken -- passes empty AX25Frame instead of raw bytes
- `KISSInterface._reader_thread()` references `self.config` which is not defined
- `Dict` type hint used without import in kiss.py

**File Structure:**
```
src/pyax25_22/
  core/
    config.py        -- AX25Config (immutable, validated)
    connected.py     -- AX25Connection
    exceptions.py    -- Full exception hierarchy
    flow_control.py  -- AX25FlowControl
    framing.py       -- AX25Frame, AX25Address, FCS, bit stuffing
    negotiation.py   -- XID build/parse/negotiate
    statemachine.py  -- AX25StateMachine (6 states)
    timers.py        -- AX25Timers (T1 adaptive, T3)
    validation.py    -- validation utilities
  interfaces/
    agwpe.py         -- AGWPEInterface (basic)
    kiss.py          -- KISSInterface (basic, has bugs)
    transport.py     -- TransportInterface (ABC)
  utils/
    logging.py
```

**Known bugs in PyAX25_22:**
- `kiss.py:92` -- `Dict` not imported (should be `from typing import ... Dict`)
- `kiss.py:172` -- `set_parameter()` passes `AX25Frame()` instead of raw bytes
- `kiss.py:215` -- references `self.config` (undefined)
- `statemachine.py:71` -- `Optional` not imported (missing `from typing import Optional`)
- `connected.py:75` -- `config = config or AX25Config()` -- AX25Config has no default constructor (dataclass with required slot)

### 3.2 PyXKISS (v1.0.0) -- Merge Source

**Complete implementation, ready to integrate:**
- `constants.py` -- All KISS/XKISS/SMACK constants
- `kiss.py` -- KISS base class (serial, threaded reader, destuffing)
- `xkiss.py` -- XKISS extends KISS (multi-drop, polling, XOR checksum, passive queuing)
- `smack.py` -- SMACK extends XKISS (CRC-16, auto-switch)
- `exceptions.py` -- XKISSException, SerialError, ChecksumError
- `interface.py` -- (not reviewed in detail -- examine before merge)

**Bug in smack.py:**
- `smack.py:21` -- Uses `threading.Lock()` but `import threading` is missing
- Fix: add `import threading` at top of smack.py

**Architecture note:**
- PyXKISS uses inheritance: SMACK extends XKISS extends KISS
- This is compatible with PyAX25_22's TransportInterface ABC
- Strategy: Refactor to use transport-agnostic KISSBase + KISSSerial/KISSTCP subclasses

### 3.3 PyAGW3 -- Merge Source

**More complete than PyAX25_22's AGWPEInterface:**
- `AGWPEFrame` dataclass
- `AGWPEClient` with:
  - Exponential backoff retry (`connect(max_retries, base_delay)`)
  - TCP keepalive (SO_KEEPALIVE + TCP_KEEPIDLE/INTVL/CNT)
  - More frame types: K (raw), Y (outstanding), H (heard), v (extended version), m (memory), T (login), P (parameter)
  - Callbacks: `on_frame`, `on_connected_data`, `on_outstanding`, `on_heard_stations`, `on_extended_version`, `on_memory_usage`

**Bug/inconsistency:**
- `agwpe.py:77` -- Sends `b'R'` to register but AGWPE spec uses `b'X'` for callsign registration. The `R` command requests version info. Review against spec.
- Header packing differs from PyAX25_22 (different field order in struct.pack_into)

### 3.4 PyPACSAT (v1.0.0) -- Reference

**Status:** "DOES NOT WORK AT THE MOMENT" (README)

**Files:**
- `pacsat/broadcast.py` -- PACSAT broadcast protocol (PID 0xBD/0xBB)
- `pacsat/ftl0_server.py` -- FTL0 file transfer server
- `pacsat/groundstation.py` -- Ground station coordination
- `pacsat/pfh.py` -- PACSAT File Header handling
- `pacsat/radio_connected.py` -- Connected mode radio interface
- `pacsat/telemetry.py` -- Telemetry parsing
- `pacsat/file_storage.py` -- File storage with subdirectory hierarchy

**Dependencies present in tree:**
- `PyAX25_22/`, `PyXKISS/`, `PyAGW3/`, `PyHamREST1/` (all vendored in tree)

**This project will be fixed once PyAX25_22 integration is complete.**

---

## 4. Integration Plan Summary

See `docs/IMPLEMENTATION_PLAN.md` for the full detailed plan.

### Phases at a glance:

| Phase | Description | Priority |
|-------|------------|---------|
| 0 | Analysis (DONE) | -- |
| 1 | Restructure interfaces/ to subpackages | High |
| 2 | KISS/XKISS/SMACK integration from PyXKISS | High |
| 3 | AGW integration from PyAGW3 | High |
| 4 | FX.25 implementation (new) | Medium |
| 5 | Async transport support | Medium |
| 6 | Tests and documentation | Ongoing |

---

## 5. Protocol Reference

### 5.1 KISS Protocol

- Standard KISS: FEND=0xC0, FESC=0xDB, TFEND=0xDC, TFESC=0xDD
- Commands: 0x00=Data, 0x01=TXDelay, 0x02=Persistence, 0x03=SlotTime, 0x04=TXTail, 0x05=FullDuplex, 0x06=SetHardware, 0xFF=Exit
- Multi-drop (XKISS/BPQKISS): High nibble of command byte = TNC address (0-15)
- Extended commands: 0x0C=DataACK, 0x0E=Poll

### 5.2 XKISS / Multi-Drop Extensions (G8BPQ / WK5M)

- Command byte: bits 7-4 = port/address, bits 3-0 = command
- Poll command (0x0E): host requests queued data from TNC
- Optional XOR checksum: 1 byte appended before FEND
- BPQ32 CHECKSUM option: XOR of all bytes between FENDs

### 5.3 SMACK (Stuttgart Modified Amateur Radio CRC-KISS)

- Data command with bit 7 set (0x80)
- CRC-16: poly 0x8005 (normal), init 0x0000, LSB-first append (2 bytes)
- Auto-switch: TNC enables CRC after first valid CRC frame received
- Invalid CRC: frame dropped silently

### 5.4 AGW Packet Engine (SV2AGW Protocol)

Header format (36 bytes):
```
offset  size  field
0       4     Port (LOWORD = radio port index)
4       4     DataKind (LOWORD = ASCII char code, e.g. 0x44='D')
8       10    CallFrom (NULL-terminated ASCII)
18      10    CallTo (NULL-terminated ASCII)
28      4     DataLen (length of data field)
32      4     USER (reserved, always 0)
[36+]         Data (DataLen bytes)
```

Key DataKinds:
- `X` -- Register callsign (send) / success reply (receive)
- `G` -- Port capabilities query (send/receive)
- `R` -- Version info (send/receive)
- `m` -- Enable monitoring (send) / monitoring frame (receive)
- `k` -- Enable raw frames (send) / raw AX.25 frame (receive)
- `U` -- Send unproto (send) / unproto monitor (receive)
- `V` -- Send unproto via digipeaters (send)
- `D` -- Send data to connected station (send) / connected data (receive)
- `d` -- Disconnect (send) / disconnected notification (receive)
- `c` -- Connect without digipeaters (send) / new connection (receive)
- `v` -- Connect with digipeaters (send)
- `H` -- Heard list (send request / receive data)
- `Y` -- Outstanding frames (send request / receive count)
- `T` -- Login credentials
- `P` -- Set radio port parameter

### 5.5 FX.25 (TAPR Forward Error Correction)

- Wraps AX.25 frames with Reed-Solomon FEC
- 8-byte correlation tag identifies RS parameters
- Supported RS modes: (255,239,8), (255,223,16), (255,191,32), (160,128,16), etc.
- Preamble: 0x00 bytes before tag (TNC-specific, often 0-4)
- Frame limit: max 239 bytes of AX.25 data in single FX.25 block (larger needs segmentation)

---

## 6. Known Issues / Bugs to Fix

### PyAX25_22

1. `interfaces/kiss.py` -- Missing `from typing import Dict`
2. `interfaces/kiss.py:172` -- `set_parameter()` passes `AX25Frame()` -- should send raw bytes
3. `interfaces/kiss.py:215` -- `self.config` undefined in `_reader_thread`
4. `core/statemachine.py:71` -- Missing `from typing import Optional`
5. `core/connected.py:75` -- `AX25Config()` has no zero-arg constructor (frozen dataclass)
6. `core/statemachine.py` TIMER_RECOVERY: T1_timeout should transition to DISCONNECTED after N2 retries, not CONNECTED

### PyXKISS

7. `smack.py` -- Missing `import threading`

### PyAGW3

8. `agwpe.py:77` -- Uses `b'R'` for callsign registration -- should be `b'X'`

---

## 7. Testing Strategy

### Unit Tests (no hardware)
- Frame encode/decode round-trips
- KISS framing/destuffing with known byte sequences
- SMACK CRC-16 with known vectors
- FX.25 encode/decode with known vectors
- State machine transitions for all states/events

### Integration Tests (mock sockets/serial)
- KISSSerial with mock serial port
- KISSTCP with mock socket
- AGWPEClient with mock TCP server
- XKISS polling with mock serial

### Protocol Compliance Tests
- AX.25 v2.2 SDL compliance (state machine)
- KISS framing compliance
- SMACK CRC compliance
- FX.25 tag recognition

---

## 8. Security Considerations

- **Serial port validation**: Accept only known device paths (e.g. /dev/ttyUSB*) -- prevent path traversal
- **TCP connection limits**: Always set socket timeouts -- prevent blocking forever
- **Frame size enforcement**: Reject oversized frames (DoS prevention)
- **AGW login ('T' frame)**: Credentials sent in plaintext -- document risk, consider TLS tunnel
- **FEC error amplification**: Limit RS decode attempts -- malformed frames can be expensive
- **Thread safety**: All shared state (queues, connected flag) must be protected by locks

---

## 9. Current Implementation Status

### Core Modules (all rewritten with docstrings, error handling, logging)

- [x] `core/config.py` -- AX25Config (immutable dataclass, validated)
- [x] `core/framing.py` -- AX25Frame, AX25Address, FCS, encode/decode
- [x] `core/statemachine.py` -- 6-state AX.25 v2.2 state machine
- [x] `core/flow_control.py` -- AX25FlowControl (window, RR/RNR/REJ/SREJ)
- [x] `core/timers.py` -- AX25Timers (T1 adaptive Jacobson, T3)
- [x] `core/negotiation.py` -- XID build/parse/negotiate
- [x] `core/validation.py` -- validate_frame_structure, full_validation
- [x] `core/connected.py` -- AX25Connection (full lifecycle management)
- [x] `core/exceptions.py` -- Complete exception hierarchy

### Interface Modules

- [x] `interfaces/transport.py` -- TransportInterface ABC (unchanged)
- [x] `interfaces/kiss/` -- **NEW subpackage** (Phase 1 + Phase 2 complete)
  - [x] `constants.py` -- All KISS/XKISS/SMACK constants
  - [x] `exceptions.py` -- KISS-specific exceptions
  - [x] `base.py` -- KISSBase (abstract, transport-agnostic)
  - [x] `serial.py` -- KISSSerial (pyserial transport)
  - [x] `tcp.py` -- KISSTCP (TCP socket transport) -- new
  - [x] `xkiss.py` -- XKISSMixin, XKISS, XKISSTCP (multi-drop + DIGIPEAT)
  - [x] `smack.py` -- SMACKMixin, SMACK, SMACKTCP (CRC-16)
  - [x] `__init__.py` -- re-exports + KISSInterface alias
- [x] `interfaces/agw/` -- **NEW subpackage** (Phase 3.1 complete)
  - [x] `constants.py` -- AGWPE frame kind codes + header offsets
  - [x] `exceptions.py` -- AGW-specific exceptions
  - [x] `client.py` -- AGWPEClient (full, from PyAGW3 + fixes)
  - [x] `async_client.py` -- AsyncAGWPEClient (asyncio, Phase 5 complete)
  - [x] `serial.py` -- AGWSerial bridge (Phase 3.2 complete)
  - [x] `__init__.py` -- re-exports
- [x] `interfaces/fx25/` -- FX.25 FEC (Phase 4 complete)
  - [x] `rs.py` -- Reed-Solomon GF(2^8) encoder/decoder (Berlekamp-Massey)
  - [x] `encoder.py` -- FX25Encoder (tag selection, padding, preamble)
  - [x] `decoder.py` -- FX25Decoder (stream state machine, HUNT/DATA)
  - [x] `constants.py` -- FX25 tag table, wire format constants
  - [x] `__init__.py` -- re-exports
- [x] `interfaces/kiss/async_tcp.py` -- AsyncKISSTCP (asyncio, Phase 5 complete)
- [x] `interfaces/kiss.py` -- LEGACY (still exists, backward compat shim)
- [x] `interfaces/agwpe.py` -- LEGACY (still exists, backward compat shim)
- [x] `interfaces/__init__.py` -- updated with new exports

### Tests

- [x] `tests/test_config.py` -- 23 tests
- [x] `tests/test_framing.py` -- existing
- [x] `tests/test_statemachine.py` -- existing
- [x] `tests/test_flow_control.py` -- existing + ResourceExhaustionError fix
- [x] `tests/test_timers.py` -- 18 tests (new)
- [x] `tests/test_negotiation.py` -- 11 tests (new)
- [x] `tests/test_validation.py` -- 14 tests (new)
- [x] `tests/test_exceptions.py` -- 17 tests (new)
- [x] `tests/test_integration.py` -- existing + 2 fixes (_process_timers, retry logic)
- [x] `tests/test_kiss_base.py` -- tests for KISSBase/Serial/TCP
- [x] `tests/test_xkiss.py` -- tests for XKISS/SMACK
- [x] `tests/test_agw_client.py` -- tests for AGWPEClient
- [x] `tests/test_agw_serial.py` -- 33 tests for AGWSerial bridge (new)
- [x] `tests/test_fx25.py` -- 38 tests for FX.25/RS (new)
- [x] `tests/test_async_kiss.py` -- 31 tests for AsyncKISSTCP (new)
- [x] `tests/test_async_agw.py` -- 13 tests for AsyncAGWPEClient (new)
- [x] `tests/test_transport_compliance.py` -- fixed 2 pre-existing mock serial failures

**Current test count:** 324 passed, 0 failed

### Documentation

- [x] `docs/IMPLEMENTATION_PLAN.md` -- phased plan
- [x] `docs/HANDOFF.md` -- this file
- [x] `docs/CLIFF_NOTES.md` -- session-by-session cliff notes
- [x] `docs/architecture_paper.md` -- scientific paper (9 sections, 10 appendices)
- [x] `docs/api_specification.md` -- API spec
- [x] `docs/compliance.md` -- protocol compliance notes
- [x] `docs/agwpe_spec.md` -- AGWPE protocol reference
- [x] `docs/kiss_spec.md` -- KISS/XKISS/SMACK reference
- [x] `docs/smack_spec.md` -- SMACK CRC-KISS reference

## 10. Hand-Off Checklist

- [x] Analysis of all three repos complete
- [x] Implementation plan created (IMPLEMENTATION_PLAN.md)
- [x] Handoff document created (this file)
- [x] Memory files created for future sessions
- [x] Phase 0: Analysis complete
- [x] Phase 1: interfaces/ restructure to subpackages
- [x] Phase 2: KISS/XKISS/SMACK integration (KISSBase, KISSSerial, KISSTCP, XKISS, SMACK)
- [x] Phase 3.1: AGW client integration (AGWPEClient from PyAGW3)
- [x] DIGIPEAT per port feature (enable_digipeat/disable_digipeat/set_digipeat)
- [x] Phase 3.2: AGWSerial bridge (AGW to serial KISS TNC)
- [x] Phase 4: FX.25 Reed-Solomon FEC
- [x] Phase 5: AsyncKISSTCP, AsyncAGWPEClient
- [x] Phase 6.1: Tests for new kiss/ and agw/ modules (324/324 passing)
- [x] Fix 2 pre-existing transport compliance mock tests
- [ ] PyPACSAT: Update to use merged PyAX25_22

---

## 10. Quick Start for Next Developer

```bash
# Clone / navigate to main repo
cd ~/tmp/code/ham/packet/PyAX25_22/PyAX25_22

# Install dev dependencies
pip install -e .[dev]

# Run existing tests
pytest -v

# Review implementation plan
cat docs/IMPLEMENTATION_PLAN.md

# Key source repos to reference
ls ~/tmp/code/ham/packet/PyXKISS/PyXKISS/
ls ~/tmp/code/ham/packet/PyAGW3/PyAGW3/
```

---

*License: LGPL-3.0-or-later*
*Copyright (C) 2025-2026 Kris Kirby, KE4AHR*
*73 de KE4AHR*

