# CHANGELOG - PyAX25_22

## Version 1.0.1 (2026-03-22)

### Changes

- **License:** Changed to GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)
- **Version:** Bumped to 1.0.1 across all files (pyproject.toml, setup.py, `__version__`, man page, docs)
- **Copyright:** Standardized to 2025-2026 across all source and documentation files
- **Docs:** Removed duplicate copyright notices from all documentation files
- **Docs:** Fixed stale license strings (`LGPLv3.0` → `LGPL-3.0-or-later`) in spec documents
- **Docs:** Updated IMPLEMENTATION_PLAN, HANDOFF, architecture_paper, compliance with current status
- **Docs:** Fixed broken `api_reference.md` link in README to `api_specification.md`
- **Docs:** Added v0.6.5 and v1.0.1 entries to CHANGELOG
- **Man page:** Updated date, license, and copyright in `man/pyax25_22.3`
- **Docstrings:** Added missing docstrings to `rs._get_generator` and `decoder._process_byte`

---

## Version 0.6.5 (2026-03-22)

### New Features

- **Phase 1 -- Interface Restructure:** `interfaces/kiss/` and `interfaces/agw/` subpackages
  replacing monolithic `kiss.py` / `agwpe.py`; backward-compat `KISSInterface` alias retained
- **Phase 2 -- KISS/XKISS/SMACK:**
  - `KISSBase` (abstract, transport-agnostic framing)
  - `KISSSerial` (pyserial transport with reader thread)
  - `KISSTCP` (TCP socket transport with keepalive)
  - `XKISSMixin` + `XKISS` + `XKISSTCP` (G8BPQ multi-drop, active/passive polling,
    optional XOR checksum, per-port DIGIPEAT: `enable_digipeat` / `disable_digipeat`)
  - `SMACKMixin` + `SMACK` + `SMACKTCP` (CRC-16 auto-switch)
- **Phase 3.1 -- AGWPE Client:** `AGWPEClient` merged from PyAGW3 with fixes:
  uses `X` (not `R`) for callsign registration; exponential backoff; TCP keepalive;
  all frame types dispatched
- **Phase 3.2 -- AGWSerial Bridge:** `AGWSerial` TCP server bridging AGWPE clients
  to a serial KISS TNC; AX.25 UI frame encode/decode helpers
- **Phase 4 -- FX.25 Reed-Solomon FEC:**
  - `rs.py`: GF(2^8) arithmetic, Berlekamp-Massey, Chien search, Forney algorithm
  - `FX25Encoder`: tag auto-selection, preamble, padding
  - `FX25Decoder`: stream state machine (HUNT / DATA states), error correction
- **Phase 5 -- Async Transports:**
  - `AsyncKISSTCP`: asyncio KISS over TCP; `on_frame` callback (plain or coroutine)
    + `asyncio.Queue` delivery
  - `AsyncAGWPEClient`: asyncio AGWPE client; full dispatch callbacks for
    `on_frame`, `on_connect`, `on_connected_data`, `on_disconnect`, `on_outstanding`,
    `on_heard_stations` -- all support plain or coroutine functions

### Infrastructure

- `conftest.py` at repo root: adds `src/` to `sys.path` (pytest 6.x compatibility)
- Fixed 2 pre-existing mock serial failures in `test_transport_compliance.py`

### Test Count

324/324 tests passing.

---

## Version 0.5.27 (2026-01-06)

###  Bug Fixes

#### Critical Fixes in `src/pyax25_22/core/framing.py`

1. **Address Encoding - Reserved Bit (HIGH)**
   - Fixed missing reserved bit in SSID byte
   - Changed initialization from `0x60` to `0x80` to set bit 7=1 as required by AX.25 v2.2
   - **Impact:** Frames now properly compliant with spec section 3.12.2
   - **Closes:** Test failure in `test_address_encoding`

2. **Address Decoding - Right Shift (HIGH)**
   - Added missing right shift by 1 bit to decode callsign characters
   - Changed from `char_code = b` to `char_code = b >> 1`
   - **Impact:** Callsigns now decode correctly instead of showing garbage
   - **Closes:** Test failure in `test_address_decoding`

3. **Bit Stuffing Algorithm (CRITICAL)**
   - Complete rewrite of `_bit_stuff()` and `_bit_destuff()` methods
   - Implemented proper HDLC-style bit-level processing with state tracking
   - Fixed algorithm to correctly insert/remove stuffing bits after five consecutive 1s
   - **Impact:** Frames can now be properly encoded and decoded without corruption
   - **Closes:** Test failure in `test_bit_stuffing`

4. **Extended Control Field - Modulo 128 (HIGH)**
   - Fixed encode() to write both bytes of 16-bit control field
   - Fixed decode() to read both bytes of 16-bit control field
   - Added proper detection of extended I-frames
   - **Impact:** Modulo 128 mode now works correctly
   - **Closes:** Test failure in `test_extended_mod128`

###  Test Results

- **Before:** 4 failed, 4 passed
- **After:** 8 passed, 0 failed

###  Changed Files

- `src/pyax25_22/core/framing.py` - 4 major bug fixes
- `tests/test_framing.py` - Updated expectations and test methodology
- `setup.py` - Version bump to 0.5.27

###  Documentation

- Updated inline code documentation
- Added rationale for each fix

###  Compliance

All changes restore full AX.25 v2.2 specification compliance:
- Section 3.12.2: Address Field Encoding  
- Section 3.8: Bit Stuffing
- Section 3.9: Extended Control Field

---

## Version 0.5.26 (2026-01-02)

*Previous version - contained 4 critical bugs in framing module*

[Earlier versions omitted for brevity]

Copyright (C) 2025-2026 Kris Kirby, KE4AHR
