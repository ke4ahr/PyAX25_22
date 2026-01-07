# CHANGELOG - PyAX25_22

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
