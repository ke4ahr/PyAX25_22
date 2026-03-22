# Session Cliff Notes

---

## Session 2026-03-21 (First Session)

**Goal:** Analyze all repos, create documentation, plan integration.

### Done

1. Analyzed all source repos (PyAX25_22, PyXKISS, PyAGW3, PyPACSAT)
2. Identified bugs in each codebase
3. Created `docs/IMPLEMENTATION_PLAN.md` with phased plan
4. Created `docs/HANDOFF.md` with detailed findings
5. Created memory files for future sessions
6. Fixed em dashes in compliance.md

### Key Decisions

- Single library target: all code goes into PyAX25_22
- Subpackage structure: `interfaces/kiss/`, `interfaces/agw/`, `interfaces/fx25/`
- Use PyXKISS as primary KISS/XKISS/SMACK source
- Use PyAGW3 as primary AGWPE source
- Transport-agnostic KISSBase design
- SMACK extends XKISS extends KISSBase

---

## Session 2026-03-21 (Main Implementation Session)

**Goals:**
- Add sixth-grade docstrings to all functions (Task 6)
- Add comprehensive error handling and logging (Task 7)
- Write unit tests for all modules (Task 8)
- Write architecture scientific paper (Task 9)

### Done

1. **Rewrote all core modules** with sixth-grade Google-style docstrings,
   comprehensive logging, specific exception types:
   - `core/framing.py` -- AX25Address, AX25Frame; removed bit stuffing from
     encode/decode for KISS-based operation (TNC handles it in hardware)
   - `core/statemachine.py` -- AX25StateMachine; added increment_vr(),
     reset_sequence_variables(), fixed missing Optional import
   - `core/flow_control.py` -- AX25FlowControl; changed enqueue_i_frame to
     raise ResourceExhaustionError (not FrameError)
   - `core/timers.py` -- AX25Timers; split T1/T3 handlers into factory methods,
     use time.monotonic() for RTT
   - `core/negotiation.py` -- fixed negotiate_config() to include all AX25Config
     fields so constructor call succeeds
   - `core/validation.py` -- full docstrings and logging
   - `core/connected.py` -- fixed missing TransportError import, fixed
     AX25Config() constructor call, added _on_t3_timeout(), fixed mod-128 path
   - `interfaces/transport.py` -- fixed missing Callable/Dict imports
   - `interfaces/kiss.py` -- fixed missing Dict import, fixed set_parameter()
     raw KISS frame, fixed frame_config reference in reader thread
   - `interfaces/agwpe.py` -- use 'X' for callsign registration (not 'R')

2. **New test files:**
   - `tests/test_config.py` -- 23 tests for AX25Config
   - `tests/test_negotiation.py` -- 11 tests for XID
   - `tests/test_validation.py` -- 14 tests for frame structure rules
   - `tests/test_exceptions.py` -- 17 tests for exception hierarchy
   - `tests/test_timers.py` -- 18 tests for AX25Timers

3. **Architecture paper:** `docs/architecture_paper.md` (9 sections + 10 appendices)

4. **Bug fixes this session:**
   - Bit stuffing removed from encode/decode (was breaking FCS round-trip)
   - ResourceExhaustionError vs FrameError mismatch in flow control tests
   - test_integration.py async import wrapped in try/except
   - typing_extensions upgraded to fix pytest plugin loading error
   - Two integration test fixes:
     - `_on_t1_timeout` was calling `sm.transition("T1_timeout")` on intermediate
       retries (wrongly moving state to DISCONNECTED); now only called when
       retries are exhausted
     - Added missing `_process_timers()` async no-op method

**Final test count:** 143 passed, 2 pre-existing failures (transport compliance
mock serial tests -- were failing before these sessions too)

---

## Session 2026-03-21 (Interface Restructure Session)

**Goals:**
- Phase 1: Restructure interfaces/ into kiss/, agw/, fx25/ subpackages
- Phase 2: Implement KISS/XKISS/SMACK from PyXKISS
- Phase 3 (partial): Implement AGWPEClient from PyAGW3
- Add DIGIPEAT=ON/OFF per port feature

### Done

1. **Created `interfaces/kiss/` subpackage:**
   - `constants.py` -- FEND/FESC/CMD_*/SMACK_*/XKISS constants
   - `exceptions.py` -- KISSTransportError, KISSSerialError, KISSTCPError,
     KISSFrameError, KISSChecksumError, KISSQueueFullError
   - `base.py` -- **KISSBase** (abstract, transport-agnostic):
     `_stuff()`, `_destuff()`, `_process_byte()`, `_dispatch()`, abstract `write()`
   - `serial.py` -- **KISSSerial(KISSBase)**: opens pyserial port, reader thread,
     write lock
   - `tcp.py` -- **KISSTCP(KISSBase)**: TCP socket, keepalive, reader thread
   - `xkiss.py` -- **XKISSMixin(KISSBase)** + **XKISS(XKISSMixin, KISSSerial)**
     + **XKISSTCP(XKISSMixin, KISSTCP)**:
     - Multi-drop port addressing (high nibble of cmd byte)
     - Active polling thread
     - Passive polling + per-port RX queues
     - Optional XOR checksum
     - **DIGIPEAT per port**: `enable_digipeat(port)`, `disable_digipeat(port)`,
       `set_digipeat(port, bool)`, `get_digipeat(port)` -- re-transmits frames
       whose first un-repeated digipeater address matches our_calls set
     - Fixed bug: original PyXKISS XKISS.send() did not KISS-escape payload
   - `smack.py` -- **SMACKMixin(XKISSMixin)** + **SMACK(SMACKMixin, XKISS)**
     + **SMACKTCP(SMACKMixin, XKISSTCP)**:
     - CRC-16 (poly 0x8005, init 0x0000, LSB-first)
     - Auto-switch on first valid CRC frame received
     - Fixed bug: original PyXKISS smack.py missing `import threading`
   - `__init__.py` -- re-exports all + KISSInterface alias

2. **Created `interfaces/agw/` subpackage:**
   - `constants.py` -- AGWPE_DEFAULT_PORT, frame kind codes, header offsets
   - `exceptions.py` -- AGWConnectionError, AGWFrameError, AGWLoginError,
     AGWTimeoutError
   - `client.py` -- **AGWPEClient** (from PyAGW3, enhanced):
     - Uses 'X' for callsign registration (PyAGW3 used 'R' which is version info)
     - Exponential backoff reconnect
     - TCP keepalive
     - All frame types dispatched
     - Safety limit: 64 KiB max data_len
   - `__init__.py` -- re-exports

3. **Created `interfaces/fx25/` stub:**
   - `__init__.py` -- raises NotImplementedError (Phase 4 placeholder)

4. **Updated `interfaces/__init__.py`:**
   - Exports KISSBase, KISSSerial, KISSTCP, XKISS, XKISSTCP, SMACK, SMACKTCP
   - Exports AGWPEClient, AGWPEFrame
   - KISSInterface and AGWPEInterface aliases for backward compat

### MRO (Method Resolution Order)

```
XKISS -> XKISSMixin -> KISSSerial -> KISSBase -> ABC -> object
SMACK -> SMACKMixin -> XKISS -> XKISSMixin -> KISSSerial -> KISSBase -> ABC
```

### Bug Fixes from PyXKISS

| Bug | Location | Fix |
|-----|----------|-----|
| Missing `import threading` | smack.py | Added (SMACKMixin uses threading.Lock) |
| XKISS.send() no KISS escaping | xkiss.py | Call `_stuff()` for proper escaping |
| checksum computed over unescaped bytes | xkiss.py | Fixed: checksum computed before stuffing |

### Known Issues (Pre-existing)

- `tests/test_transport_compliance.py::test_transport_validation_kiss` --
  FAILS because it calls `KISSInterface("mock_port")` which tries to open a
  real serial port. Was failing before these sessions. Needs mock patch.
- `tests/test_transport_compliance.py::test_kiss_send_receive_mock` -- same

### Next Steps

- Phase 3.2: AGWSerial bridge (AGW over serial KISS TNC)
- Phase 4: FX.25 Reed-Solomon FEC implementation
- Phase 5: AsyncKISSTCP and AsyncAGWPEClient
- Write tests for new kiss/, agw/ modules
- Fix the 2 pre-existing transport compliance mock tests

---

## Session 2026-03-22 (FX.25 + AGWSerial + Async Transports)

**Goals:** Phase 3.2 (AGWSerial), Phase 4 (FX.25 FEC), Phase 5 (async transports),
fix pre-existing transport compliance test failures.

### Done

1. **Phase 4: FX.25 Reed-Solomon FEC** (`interfaces/fx25/`)
   - `rs.py` -- GF(2^8) arithmetic, Berlekamp-Massey decoder, Chien search, Forney algorithm
   - `encoder.py` -- FX25Encoder: auto-select smallest tag, pad frame, add preamble
   - `decoder.py` -- FX25Decoder: stream state machine with HUNT and DATA states
   - 38 tests (`tests/test_fx25.py`) all passing

2. **Phase 3.2: AGWSerial bridge** (`interfaces/agw/serial.py`)
   - TCP server that accepts AGWPE clients and bridges to a serial KISS TNC
   - AX.25 UI frame encode/decode helpers
   - Per-client monitoring and registration tracking
   - 33 tests (`tests/test_agw_serial.py`) all passing

3. **Phase 5: Async transports**
   - `interfaces/kiss/async_tcp.py` -- AsyncKISSTCP: asyncio KISS over TCP
     - `_stuff()` / `_destuff()` / `_process_byte()` (same logic as KISSBase)
     - `send(payload, cmd)` with asyncio.Lock, writer.drain()
     - on_frame callback (plain or coroutine) + asyncio.Queue delivery
   - `interfaces/agw/async_client.py` -- AsyncAGWPEClient: asyncio AGWPE client
     - Full dispatch: on_frame, on_connect, on_connected_data, on_disconnect,
       on_outstanding, on_heard_stations -- all support plain or coroutine callbacks
     - `send_ui`, `send_raw`, `send_connect`, `send_disconnect`, `send_connected_data`
   - 31 tests (`tests/test_async_kiss.py`), 13 tests (`tests/test_async_agw.py`)

4. **Fixed 2 pre-existing transport compliance failures**
   - Both `test_transport_validation_kiss` and `test_kiss_send_receive_mock` called
     `KISSInterface("mock_port")` which tried to open a real serial port
   - Fix: wrap constructor call with `patch("serial.Serial", return_value=serial)`

5. **Added `conftest.py`** at repo root
   - Adds `src/` to `sys.path` so pytest finds `pyax25_22` without installing the package
   - Needed because pytest 6.2.5 does not support `pythonpath` in pyproject.toml

### Key Bug Fixes in RS Decoder

The BM + Forney RS decoder had 5 bugs:

| Bug | Root cause | Fix |
|-----|-----------|-----|
| BM padding corrupted C[0] | `C = [0]+C` prepends in little-endian | `C.append(0)` |
| BM returned little-endian | gf_poly_eval expects big-endian | `return list(reversed(C))` |
| Forney Lambda derivative wrong index | `locator[j]` instead of `locator[d-j]` | Use `locator[d-j]` |
| Forney Omega wrong truncation | took first ncheck (high-degree) terms | take last ncheck terms |
| Forney magnitude extra xi factor | `gf_mul(xi^(1-FCR), ...)` for FCR=1 = no-op | remove xi factor |

### Final State

**324/324 tests passing.** All phases complete. PyPACSAT migration is the only remaining work.

---

## Quick Reference

### Import Guide

```python
# Standard KISS over serial
from pyax25_22.interfaces.kiss import KISSSerial
tnc = KISSSerial("/dev/ttyUSB0")

# KISS over TCP (e.g., Dire Wolf)
from pyax25_22.interfaces.kiss import KISSTCP
tnc = KISSTCP("localhost", 8001)

# XKISS over serial (multi-drop, polling, digipeat)
from pyax25_22.interfaces.kiss import XKISS
xk = XKISS("/dev/ttyUSB0", our_calls={"KE4AHR"}, digipeat_ports={0})

# SMACK over serial (CRC-16, auto-switch)
from pyax25_22.interfaces.kiss import SMACK
smack = SMACK("/dev/ttyUSB0")

# AGW Packet Engine
from pyax25_22.interfaces.agw import AGWPEClient
client = AGWPEClient("localhost", 8000, callsign="KE4AHR")
client.connect()
```

### DIGIPEAT API

```python
xk = XKISS("/dev/ttyUSB0", our_calls={"KE4AHR"})
xk.enable_digipeat(0)        # DIGIPEAT=ON port 0
xk.disable_digipeat(1)       # DIGIPEAT=OFF port 1
xk.set_digipeat(0, True)     # Same as enable_digipeat(0)
is_on = xk.get_digipeat(0)   # Query current setting
```

### Test Commands

```bash
# Run all tests (conftest.py adds src/ to sys.path automatically)
python3.10 -m pytest -q

# Run specific test file
python3.10 -m pytest tests/test_timers.py -v

# Quick syntax check
python3.10 -m py_compile src/pyax25_22/interfaces/kiss/async_tcp.py
```
