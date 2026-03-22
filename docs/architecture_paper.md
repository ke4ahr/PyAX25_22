# PyAX25_22: Architecture and Design of a Complete AX.25 v2.2 Library

**Author:** Kris Kirby, KE4AHR
**Date:** 2026-03-21
**Version:** 0.5.97
**License:** LGPL-3.0-or-later

---

## Abstract

PyAX25_22 is a Python implementation of the AX.25 v2.2 Data Link Layer protocol for amateur packet radio. This paper describes the architecture, design decisions, protocol compliance, and internal structure of the library. The library provides both synchronous (threaded) and asynchronous (asyncio) operation, supports KISS and AGWPE transport interfaces, and is designed to be the core packet radio library for the PyPACSAT satellite ground station software. This document includes appendices that catalog every public function and method with its signature, purpose, parameters, return values, and raised exceptions.

---

## 1. Introduction

### 1.1 Background

AX.25 is the Data Link Layer (Layer 2) protocol used in amateur packet radio. It was derived from X.25 and adapted for radio use by the Amateur Radio community in the 1980s. The current version, AX.25 v2.2 (July 1998), defines connected-mode data transfer, unnumbered frames, supervisory frames, and parameter negotiation via XID (Exchange Identification) frames.

Amateur packet radio is still in active use for:

- APRS (Automatic Packet Reporting System) -- position beaconing and telemetry
- BBS (Bulletin Board Systems) -- store-and-forward message systems
- PACSAT -- packet satellite ground stations
- Winlink -- global radio email network
- Digital modes bridging -- connecting legacy packet nodes

### 1.2 Goals

PyAX25_22 was designed to:

1. Provide a complete, correct AX.25 v2.2 implementation in pure Python.
2. Support multiple transport interfaces: KISS (serial), KISS (TCP), AGWPE.
3. Support both synchronous (threading) and asynchronous (asyncio) operation.
4. Serve as the single packet radio dependency for PyPACSAT.
5. Integrate KISS, XKISS, SMACK, and AGWPE from separate libraries (PyXKISS, PyAGW3) into one cohesive library.
6. Add FX.25 forward error correction support (planned).

### 1.3 Scope

This paper covers the currently implemented core and interface layers. The FX.25 encoder/decoder is planned but not yet implemented. Async wrappers (AsyncKISSTCP, AsyncAGWPEClient) are designed but not yet complete.

---

## 2. High-Level Architecture

```
Application Layer (PyPACSAT, custom code)
        |
        | AX25Connection API
        v
+---------------------------+
|   AX25Connection          |  connected.py
|   - send_data()           |
|   - process_frame()       |
|   - connect() / disconnect()|
+---------------------------+
        |
        | Uses:
        v
+----------------+  +------------------+  +-----------+
| AX25StateMachine| | AX25FlowControl  |  | AX25Timers|
| statemachine.py | | flow_control.py  |  | timers.py |
+----------------+  +------------------+  +-----------+
        |
        | Exchanges:
        v
+---------------------------+
|   AX25Frame               |  framing.py
|   AX25Address             |
|   fcs_calc / verify_fcs   |
+---------------------------+
        |
        | Sent/received via:
        v
+------------------+  +------------------+
|  KISSInterface   |  |  AGWPEInterface   |
|  interfaces/kiss |  |  interfaces/agwpe |
+------------------+  +------------------+
        |
        v
  Radio / TNC / AGWPE server
```

### 2.1 Layer Separation

The library is organized into two main packages:

- **`pyax25_22.core`**: Protocol logic (framing, state machine, flow control, timers, negotiation, validation).
- **`pyax25_22.interfaces`**: Hardware/network transport adapters (KISS, AGWPE).

This separation means the core can be tested and used independently of any hardware. Mock transports are used in tests. The core never imports from interfaces.

---

## 3. Core Module Design

### 3.1 Configuration (config.py)

`AX25Config` is an immutable dataclass (frozen=True, slots=True) that holds all protocol parameters for one connection. Immutability is deliberate: changing configuration while a connection is open could cause subtle bugs. All field values are validated in `__post_init__` against the ranges specified in AX.25 v2.2. A `ConfigurationError` is raised immediately if any value is out of range.

**Key design decision**: Pre-built configuration objects (`DEFAULT_CONFIG_MOD8`, `DEFAULT_CONFIG_MOD128`, `CONFIG_APRS`, `CONFIG_PACSAT_BROADCAST`) are provided as module-level constants. Users can create connections with a single argument rather than specifying all parameters.

**Validated parameters**:
- `modulo`: 8 (standard) or 128 (extended). Controls sequence number range.
- `window_size` (k): 1-7 for modulo 8; 1-127 for modulo 128. Maximum in-flight frames.
- `max_frame` (N1): 1-4096 bytes. Maximum information field size.
- `t1_timeout`: 0.0-300.0 seconds. Retransmit timer base value.
- `t3_timeout`: 10.0-3600.0 seconds. Idle channel probe timer.
- `retry_count` (N2): 0-255. How many times to retry before giving up.
- `tx_delay`: 0+ seconds. Transmitter key-up delay.
- `tx_tail`: 0+ seconds. Transmitter hold time after last byte.
- `persistence`: 0-255. P-persistence CSMA aggressiveness.
- `slot_time`: 0.01-1.0 seconds. CSMA slot time.

### 3.2 Framing (framing.py)

The framing module handles encoding and decoding of AX.25 frames. It contains:

- `fcs_calc(data)`: Computes the 16-bit CRC-16/CCITT-FALSE checksum used as the Frame Check Sequence (FCS). Uses the reflected polynomial 0x8408 with initial value 0xFFFF and inverted output.
- `verify_fcs(data, received_fcs)`: Compares computed FCS with received FCS, logs a warning on mismatch.
- `AX25Address`: A dataclass representing one station address (callsign + SSID). Encodes to 7 bytes (6 shifted callsign bytes + 1 SSID byte) and decodes from 7 bytes.
- `AX25Frame`: A dataclass representing a complete AX.25 frame with destination, source, optional digipeaters, control field, optional PID, and information field.

**FCS algorithm**: CRC-16/CCITT-FALSE (also called CRC-16/ARC when reflected). Polynomial is 0x8408 (reflected 0x8005). Initial value 0xFFFF. Output is bitwise inverted. This is the standard for AX.25 and matches the HDLC standard.

**Address encoding**: Each character of the callsign is left-shifted by one bit before encoding. The callsign is padded to exactly 6 characters with spaces. The 7th byte contains: bit 7 (C/H flag), bits 6-5 (reserved, always 1), bits 4-1 (SSID, 4 bits), bit 0 (end-of-address marker).

**Control field encoding**: For modulo-8 connections, the control field is 1 byte. For modulo-128 I-frames, the control field is 2 bytes (little-endian). S-frames and U-frames are always 1 byte regardless of modulo.

**Bit stuffing**: HDLC bit stuffing (`_bit_stuff`, `_bit_destuff`) is implemented as utility methods but is NOT applied in `encode()`/`decode()`. For KISS-based interfaces, the TNC hardware performs bit stuffing. Direct HDLC-over-serial users can call these methods explicitly.

**Frame format**:
```
FLAG(0x7E) | dest(7B) | src(7B) | [digi(7B) ...] | ctrl(1-2B) | [pid(1B)] | [info(0-N1 B)] | FCS(2B) | FLAG(0x7E)
```

### 3.3 State Machine (statemachine.py)

`AX25StateMachine` implements the AX.25 v2.2 Layer 2 state machine as specified in the SDL diagrams of the specification. The six states are:

| State | Meaning |
|-------|---------|
| DISCONNECTED | No link. Waiting for events. |
| AWAITING_CONNECTION | Sent SABM/SABME; waiting for UA. |
| AWAITING_RELEASE | Sent DISC; waiting for UA. |
| CONNECTED | Link is up; data can flow. |
| TIMER_RECOVERY | T1 expired; retrying. |
| AWAITING_XID | Waiting for XID parameter negotiation response. |

**Transition rules** are enforced: calling `transition()` with an event that is not legal in the current state raises `ConnectionStateError`. This prevents the application layer from accidentally putting the protocol into an illegal state.

The state machine also maintains three sequence number variables:
- **V(S)**: Next send sequence number (incremented after each I-frame sent).
- **V(R)**: Next expected receive sequence number (incremented after each in-order I-frame received).
- **V(A)**: Last acknowledged sequence number (updated when N(R) advances).

And two status flags:
- **peer_busy**: Set when RNR is received; cleared when RR is received.
- **reject_sent**: Set when REJ is sent; cleared when the go-back-N recovery completes.
- **srej_sent**: Set when SREJ is sent; cleared when the selective retransmit completes.

### 3.4 Flow Control (flow_control.py)

`AX25FlowControl` manages the sliding window protocol:

- Tracks the list of outstanding (sent but not yet acknowledged) I-frame sequence numbers.
- Enforces the window size limit: prevents sending more than `k` unacknowledged frames.
- Tracks peer_busy (remote sent RNR) and local_busy (we are full, need to send RNR).
- Builds Supervisory frames: RR, RNR, REJ, SREJ.

**Window available formula**: `window_size - len(outstanding_seqs)`

**Acknowledgment**: `acknowledge_up_to(nr)` removes all outstanding sequence numbers less than N(R). This implements the "cumulative acknowledgment" property of AX.25: a frame with N(R) = 5 acknowledges all frames with N(S) < 5.

**Reject vs. Selective Reject**: REJ requests go-back-N retransmission from N(R). SREJ requests retransmission of a single missing frame. Only one REJ or SREJ may be outstanding at a time (tracked by `rej_sent`/`srej_sent` flags).

### 3.5 Timers (timers.py)

`AX25Timers` manages the two AX.25 timers:

**T1 (Acknowledgment Timer)**: Started when an I-frame is sent. If the ack does not arrive before T1 expires, the frame is retransmitted and T1 restarts. T1 uses an adaptive algorithm (Jacobson/Karels, same as TCP) to automatically adjust the retransmit timeout based on measured round-trip times:

```
delta  = measured_rtt - srtt
srtt   = srtt + 0.125 * delta
rttvar = rttvar + 0.25 * (|delta| - rttvar)
rto    = srtt + max(1.0, 4.0 * rttvar)
rto    = clamp(rto, 1.0, 60.0)
```

This is the same algorithm described in RFC 6298 for TCP retransmission timers.

**T3 (Idle Probe Timer)**: A much longer timer (default 300 seconds). If the link is idle for T3, we send an RR with P=1 to probe whether the remote station is still reachable. If no response comes before T1 expires (with retries), the link is declared dead.

Both timers support synchronous (threading.Timer) and asynchronous (asyncio.Task) modes. The sync mode is used by the AX25Connection when running in a threaded application. The async mode is used when running inside an asyncio event loop.

### 3.6 XID Negotiation (negotiation.py)

Before a modulo-128 connection is established, the two stations can exchange XID (Exchange Identification) frames to agree on operating parameters. The negotiation module provides:

- `build_xid_frame(config)`: Encodes the local configuration into TLV (Type-Length-Value) format. Mandatory parameters: modulo, window size, N1. Optional: SREJ support, retry count.
- `parse_xid_frame(info)`: Decodes a received XID information field from TLV format into a dictionary.
- `negotiate_config(local, remote_params)`: Applies the AX.25 negotiation rules:
  - Modulo must match exactly (no fallback).
  - Window size is the minimum of local and remote.
  - N1 (max frame) is the minimum of local and remote.
  - Retry count is taken from remote if provided.
  - SREJ is only enabled if both stations support it.

**TLV parameter codes** (from AX.25 v2.2 Section 4.3.4):

| Code | Parameter | Length |
|------|-----------|--------|
| 0x01 | Modulo | 1 byte |
| 0x02 | Window size k | 1 byte |
| 0x03 | Max frame N1 | 2 bytes (little-endian) |
| 0x04 | Retry count N2 | 1 byte |
| 0x08 | SREJ support | 1 byte (0/1) |

### 3.7 Frame Validation (validation.py)

`validate_frame_structure()` checks AX.25 compliance rules that go beyond what the basic decoder checks:

- **Digipeater count**: At most 8 digipeaters in the address field.
- **I-frame**: Must have PID; info field length must not exceed N1.
- **S-frame** (RR, RNR, REJ, SREJ): Must have no PID and no info field.
- **UI-frame**: Must have PID.
- **Other U-frames** (SABM, SABME, DISC, UA, DM, XID, FRMR, TEST): Must have no PID.

This validation is called after decoding and FCS verification, providing a defense-in-depth check that the frame is well-formed.

### 3.8 Connected Mode (connected.py)

`AX25Connection` integrates all the core components into a high-level API for managing one connected AX.25 session. It is the primary entry point for applications.

**Lifecycle**:
1. Create with `local_addr`, `remote_addr`, `config`, and optional `transport`.
2. Call `await connect()` (initiating side) or wait for `process_frame()` to receive SABM.
3. Call `await send_data(bytes)` to queue data for transmission.
4. Call `process_frame(frame)` for each received frame to advance the state machine.
5. Call `await disconnect()` to close the link gracefully.

**Frame dispatch**: `process_frame()` examines the control field to determine the frame type (U/S/I) and calls the appropriate internal handler.

**Retransmission**: `_on_t1_timeout()` is the T1 callback. It increments the retry counter and either retransmits or, if the maximum retry count is exceeded, forces disconnection. The retransmission logic currently handles SABM retransmission during connection setup; full I-frame retransmission from V(A) is a placeholder for future implementation.

---

## 4. Interface Module Design

### 4.1 Transport Abstraction (interfaces/transport.py)

`TransportInterface` is an abstract base class (ABC) that all transport implementations must inherit. It defines:

- `connect()`: Open the transport connection.
- `disconnect()`: Close the transport connection.
- `send_frame(frame)`: Transmit an AX.25 frame.
- `receive(timeout)`: Block for and return the next received frame.
- `register_callback(event, callback)`: Register event-driven callbacks.

The `validate_frame_for_transport()` helper checks encoded frame size against transport limits (512 bytes for KISS, 4096 bytes for AGWPE).

### 4.2 KISS Interface (interfaces/kiss.py)

`KISSInterface` implements the KISS protocol (TAPR, 1986) for communicating with a TNC over a serial port.

**Frame format**:
```
FEND(0xC0) | cmd_byte | escaped_data | FEND(0xC0)
```

**Escaping**: FEND (0xC0) within data is replaced by FESC(0xDB) + TFEND(0xDC). FESC within data is replaced by FESC(0xDB) + TFESC(0xDD).

**Command byte**: High nibble = TNC address (0-15, for Multi-Drop KISS/XKISS). Low nibble = command code (0x00=data, 0x01=TXDELAY, 0x02=persistence, 0x03=SLOTTIME, 0x04=TXTAIL, 0x05=FULLDUPLEX, 0x06=SETHARDWARE, 0xFF=EXIT).

**Multi-Drop KISS (XKISS)**: The high nibble of the command byte addresses up to 16 TNCs on one serial bus. `KISSInterface` supports this via the `tnc_address` parameter.

**Bugs fixed from pre-existing code**:
1. Added missing `Dict` import (was causing NameError at runtime).
2. Fixed `set_parameter()` which was calling `AX25Frame()` with no arguments. Now sends a raw KISS command frame.
3. Fixed `_reader_thread` which referenced `self.config` (undefined). Now uses `self.frame_config`.

**Reader thread**: A daemon thread continuously reads bytes from the serial port, processes FESC escaping, assembles complete KISS frames between FEND markers, decodes the AX.25 frame, and puts it in a `queue.Queue` for `receive()` to retrieve.

### 4.3 AGWPE Interface (interfaces/agwpe.py)

`AGWPEInterface` implements the AGWPE TCP/IP socket API (SV2AGW, 2000) for communicating with a TNC through the AGWPE Windows software or compatible servers.

**Frame header** (36 bytes, little-endian):
```
Port     (4 bytes)  -- radio port index
DataKind (4 bytes)  -- ASCII command byte
CallFrom (10 bytes) -- NULL-terminated source callsign
CallTo   (10 bytes) -- NULL-terminated destination callsign
DataLen  (4 bytes)  -- number of data bytes following header
USER     (4 bytes)  -- reserved, always 0
```

**DataKind values**: 'D' (connected data), 'U' (unproto monitor), 'X' (registration reply), 'R' (version), 'g' (port capabilities), 'H' (heard list), 'k' (raw frames), 'm' (monitoring toggle), 'c' (new connection), 'd' (disconnect).

**Note**: Uses `'X'` for callsign registration (the AGWPE spec), not `'R'` (which is a bug seen in some implementations -- this library uses the correct command).

---

## 5. Exception Hierarchy

All library exceptions descend from `AX25Error`. The full hierarchy:

```
AX25Error
+-- FrameError
|   +-- InvalidAddressError     callsign/SSID out of spec
|   +-- InvalidControlFieldError  control field malformed
|   +-- FCSError                checksum mismatch
|   +-- BitStuffingError        6+ consecutive 1-bits (abort)
|   +-- SegmentationError       segment reassembly failure
+-- ConnectionError
|   +-- ConnectionStateError    illegal state transition
|   +-- TimeoutError            T1 or T3 expired fatally
|   +-- ProtocolViolationError  remote violated AX.25 rules
|   +-- NegotiationError        XID parameter mismatch
+-- TransportError
|   +-- KISSError               serial port / KISS error
|   +-- AGWPEError              TCP / AGWPE error
+-- ConfigurationError          parameter out of allowed range
+-- ResourceExhaustionError     window full / buffer overflow
```

Every exception is logged at ERROR level when created, providing a complete audit trail in log files even if the caller does not handle the exception.

---

## 6. Logging

The library uses Python's `logging` module with module-level loggers:

```python
logger = logging.getLogger(__name__)
```

Log levels used:
- **DEBUG**: Frame bytes, sequence numbers, timer values, state machine steps.
- **INFO**: Connection established/closed, XID negotiation complete, config validated.
- **WARNING**: FCS mismatch, peer busy, T1/T3 timeout, retransmitting.
- **ERROR**: Serial/TCP errors, failed sends, exception creation, callback failures.

Applications configure logging as desired. The library does not configure any handlers.

---

## 7. Testing

The test suite uses pytest and covers:

| Test file | Coverage |
|-----------|----------|
| test_framing.py | Address encoding/decoding, FCS, frame round-trip |
| test_statemachine.py | All states and transitions, sequence variables |
| test_flow_control.py | Window management, busy states, REJ/SREJ |
| test_config.py | All field validations, pre-built configs |
| test_negotiation.py | XID build/parse round-trip, negotiate_config rules |
| test_validation.py | Frame structure rules, I/S/U/UI frame checks |
| test_exceptions.py | Exception hierarchy and inheritance |
| test_timers.py | Timer start/stop, SRTT update, value clamping |
| test_transport_compliance.py | Transport validation helpers |
| test_integration.py | End-to-end connection lifecycle (requires pytest-asyncio) |

Total: 131 passing tests (non-async); 14 async integration tests pending pytest-asyncio.

---

## 8. Known Bugs and Limitations

| ID | Location | Description | Status |
|----|----------|-------------|--------|
| B1 | connected.py | I-frame retransmission from V(A) after REJ is a stub (logs but does not resend) | Known, planned |
| B2 | connected.py | SREJ single-frame retransmission is a stub | Known, planned |
| B3 | interfaces/* | No TCP KISS transport yet (KISSTCP class) | Planned -- Phase 1 |
| B4 | interfaces/* | No XKISS (G8BPQ multi-drop) implementation yet | Planned -- Phase 2 |
| B5 | interfaces/* | No SMACK (Stuttgart CRC-KISS) implementation yet | Planned -- Phase 2 |
| B6 | -- | No FX.25 (Reed-Solomon FEC) implementation yet | Planned -- Phase 4 |
| B7 | -- | No asyncio transport implementations yet | Planned -- Phase 5 |

---

## 9. Protocol Compliance

| Feature | Status |
|---------|--------|
| AX.25 v2.2 frame encoding | Implemented |
| AX.25 v2.2 frame decoding | Implemented |
| FCS (CRC-16/CCITT-FALSE) | Implemented |
| Address field (dest + src + 0-8 digipeaters) | Implemented |
| Modulo-8 I-frame sequencing | Implemented |
| Modulo-128 I-frame sequencing | Implemented |
| S-frames: RR, RNR, REJ, SREJ | Implemented |
| U-frames: SABM, SABME, UA, DISC, DM, XID, FRMR | Implemented |
| State machine (all 6 states) | Implemented |
| T1 adaptive (Jacobson/Karels SRTT) | Implemented |
| T3 idle probe | Implemented |
| XID parameter negotiation | Implemented |
| Frame structure validation | Implemented |
| KISS serial transport | Implemented |
| AGWPE TCP transport | Implemented |
| Multi-Drop KISS (XKISS) | Planned |
| SMACK (CRC-KISS) | Planned |
| KISS over TCP | Planned |
| FX.25 (Reed-Solomon FEC) | Planned |
| Async transports | Planned |

---

## Appendix A: Function Reference -- core/config.py

### `AX25Config.__post_init__(self) -> None`

Called automatically by the dataclass machinery after `__init__`. Validates all configuration fields against the AX.25 v2.2 allowed ranges. Logs validation start at DEBUG and success at INFO. Raises `ConfigurationError` for any out-of-range value.

**Parameters**: None (reads from self)
**Returns**: None
**Raises**: `ConfigurationError` if any field is invalid

---

## Appendix B: Function Reference -- core/framing.py

### `fcs_calc(data: bytes) -> int`

Compute the CRC-16/CCITT-FALSE frame check sequence for the given bytes.
**Parameters**: `data` -- bytes to checksum
**Returns**: 16-bit int (0-65535)
**Raises**: `TypeError` if data is not bytes/bytearray
**Logs**: DEBUG with byte count and result FCS

### `verify_fcs(data: bytes, received_fcs: int) -> bool`

Check whether the FCS of data matches received_fcs.
**Parameters**: `data` -- frame bytes; `received_fcs` -- FCS from frame
**Returns**: True if match, False if mismatch
**Raises**: `TypeError` if data is not bytes/bytearray
**Logs**: DEBUG on match; WARNING on mismatch with both FCS values

### `AX25Address.__post_init__(self) -> None`

Validate callsign and SSID; build `_call_bytes` for encoding.
**Raises**: `InvalidAddressError` for bad callsign or SSID out of 0-15
**Logs**: DEBUG with callsign, SSID, and encoded bytes

### `AX25Address.encode(self, last: bool = False) -> bytes`

Encode this address into 7 bytes for the AX.25 address field.
**Parameters**: `last` -- True if this is the last address
**Returns**: Exactly 7 bytes
**Logs**: DEBUG with encoded hex

### `AX25Address.decode(cls, data: bytes) -> Tuple[AX25Address, bool]`

Decode 7 bytes from an AX.25 address field.
**Parameters**: `data` -- at least 7 bytes
**Returns**: (AX25Address, is_last)
**Raises**: `InvalidAddressError` if data < 7 bytes or bad character codes
**Logs**: DEBUG with decoded callsign, SSID, flags

### `AX25Frame.encode(self) -> bytes`

Encode this frame to FLAG + addr + ctrl + [pid] + [info] + FCS + FLAG.
**Returns**: Complete frame bytes
**Raises**: `FrameError` if address build fails
**Logs**: DEBUG with addresses and sizes; INFO with total frame size

### `AX25Frame._build_address_field(self) -> bytes`

Build the address field (destination + source + digipeaters) with correct last-address markers.
**Returns**: Address field bytes
**Logs**: DEBUG with address count and total length

### `AX25Frame._build_control_field(self) -> bytes`

Build the 1-byte (modulo-8) or 2-byte (modulo-128 I-frame) control field.
**Returns**: 1 or 2 bytes
**Logs**: DEBUG with control value

### `AX25Frame._bit_stuff(data: bytes) -> bytes`

Apply HDLC bit stuffing (insert 0-bit after every 5 consecutive 1-bits). Not used in encode() for KISS-based operation.
**Parameters**: `data` -- raw frame bytes to stuff
**Returns**: Bit-stuffed bytes (may be longer)
**Logs**: DEBUG with input and output sizes

### `AX25Frame._bit_destuff(cls, data: bytes) -> bytes`

Remove HDLC bit stuffing. Not used in decode() for KISS-based operation.
**Parameters**: `data` -- stuffed bytes to destuff
**Returns**: Destuffed bytes
**Raises**: `BitStuffingError` if 6+ consecutive 1-bits are found
**Logs**: DEBUG with input and output sizes

### `AX25Frame.decode(cls, raw: bytes, config: AX25Config = DEFAULT_CONFIG_MOD8) -> AX25Frame`

Decode raw bytes into an AX25Frame. Strips FLAG bytes, parses address/control/pid/info, verifies FCS.
**Parameters**: `raw` -- complete frame including FLAGS; `config` -- AX.25 config
**Returns**: Populated AX25Frame
**Raises**: `FrameError` (bad flags or too short), `FCSError` (bad checksum), `BitStuffingError` (bad stuffing), `InvalidAddressError` (bad callsign)
**Logs**: DEBUG at each parse step; INFO on success

---

## Appendix C: Function Reference -- core/statemachine.py

### `AX25StateMachine.__init__(self, config, layer3_initiated) -> None`

Initialize state machine in DISCONNECTED state.
**Parameters**: `config` -- AX25Config; `layer3_initiated` -- True to allow connect_request
**Logs**: DEBUG with modulo and layer3_initiated

### `AX25StateMachine.increment_vs(self) -> None`

Advance V(S) by 1 with modulo wrap.
**Logs**: DEBUG with old and new V(S)

### `AX25StateMachine.increment_vr(self) -> None`

Advance V(R) by 1 with modulo wrap.
**Logs**: DEBUG with old and new V(R)

### `AX25StateMachine.reset_sequence_variables(self) -> None`

Reset V(S), V(R), V(A) to 0 and clear peer_busy/reject_sent/srej_sent.
**Logs**: DEBUG

### `AX25StateMachine.transition(self, event: str, frame_type: Optional[str] = None) -> None`

Move to a new state based on the event. Enforces AX.25 v2.2 SDL rules.
**Parameters**: `event` -- event name (e.g., "connect_request", "UA_received"); `frame_type` -- "RR"/"RNR"/"REJ"/"SREJ" for supervisory events
**Raises**: `ConnectionStateError` if event is illegal in current state
**Logs**: DEBUG at start; INFO on state change

---

## Appendix D: Function Reference -- core/flow_control.py

### `AX25FlowControl.__init__(self, sm, config) -> None`

Initialize flow control with empty outstanding list.
**Logs**: INFO with window_size and modulo

### `AX25FlowControl.window_available (property) -> int`

Calculate available window space (window_size - outstanding count).
**Returns**: Non-negative int
**Logs**: DEBUG

### `AX25FlowControl.can_send_i_frame(self) -> bool`

Return True if window is open, peer is not busy, and we are not busy.
**Logs**: DEBUG

### `AX25FlowControl.enqueue_i_frame(self, seq_num: int) -> None`

Record a sent-but-not-acked frame.
**Raises**: `ResourceExhaustionError` if window full or peer busy; `FrameError` if seq_num already outstanding
**Logs**: DEBUG with seq and outstanding list

### `AX25FlowControl.acknowledge_up_to(self, nr: int) -> None`

Remove all outstanding seqs < nr. Clears rej_sent/srej_sent on any ack.
**Logs**: DEBUG with ack count and remaining outstanding

### `AX25FlowControl.handle_rr(self) -> None`

Process received RR: clear peer_busy.

### `AX25FlowControl.handle_rnr(self) -> None`

Process received RNR: set peer_busy.

### `AX25FlowControl.set_peer_busy(self) -> None`

Mark remote as busy. Logs WARNING on first transition.

### `AX25FlowControl.clear_peer_busy(self) -> None`

Mark remote as ready. Logs INFO on transition.

### `AX25FlowControl.set_local_busy(self) -> None`

Mark our buffer as full. Logs WARNING on first transition.

### `AX25FlowControl.clear_local_busy(self) -> None`

Mark our buffer as available. Logs INFO on transition.

### `AX25FlowControl.send_rr(self, pf_bit: bool = False) -> AX25Frame`

Build an RR supervisory frame with current V(R) as N(R).
**Returns**: AX25Frame (not yet sent)
**Logs**: DEBUG

### `AX25FlowControl.send_rnr(self, pf_bit: bool = False) -> AX25Frame`

Build an RNR supervisory frame.
**Returns**: AX25Frame

### `AX25FlowControl.send_reject(self, nr: int) -> Optional[AX25Frame]`

Build a REJ frame for N(R), or return None if already outstanding.
**Returns**: AX25Frame or None
**Logs**: DEBUG

### `AX25FlowControl.send_selective_reject(self, nr: int) -> Optional[AX25Frame]`

Build an SREJ frame for the specific missing sequence number, or None if already requested.
**Returns**: AX25Frame or None
**Logs**: DEBUG with SREJ list

### `AX25FlowControl.reset(self) -> None`

Clear all state: outstanding list, SREJ list, busy flags.
**Logs**: INFO

---

## Appendix E: Function Reference -- core/timers.py

### `AX25Timers.__init__(self, config: AX25Config) -> None`

Initialize SRTT/RTO from config.t1_timeout. No timers running.
**Logs**: INFO with T1/T3 base and initial RTO

### `AX25Timers.start_t1_sync(self, callback) -> None`

Start T1 with threading.Timer. Cancels any existing T1 first.
**Raises**: `TimeoutError` if Timer cannot be created
**Logs**: DEBUG

### `AX25Timers.stop_t1_sync(self) -> None`

Cancel T1 threading.Timer if running. Safe to call when not running.
**Logs**: DEBUG

### `AX25Timers.start_t3_sync(self, callback) -> None`

Start T3 with threading.Timer.
**Raises**: `TimeoutError` if Timer cannot be created
**Logs**: DEBUG

### `AX25Timers.stop_t3_sync(self) -> None`

Cancel T3 threading.Timer if running.
**Logs**: DEBUG

### `AX25Timers.start_t1_async(self, callback) -> None` (coroutine)

Start T1 as asyncio.Task. Must be called from async context.
**Raises**: `TimeoutError` if task cannot be created

### `AX25Timers.stop_t1_async(self) -> None` (coroutine)

Cancel T1 asyncio task if running.

### `AX25Timers.start_t3_async(self, callback) -> None` (coroutine)

Start T3 as asyncio.Task.

### `AX25Timers.stop_t3_async(self) -> None` (coroutine)

Cancel T3 asyncio task if running.

### `AX25Timers.record_acknowledgment(self) -> None`

Update SRTT/RTTVAR/RTO using Jacobson/Karels algorithm. Call when N(R) advances.
**Logs**: DEBUG with RTT sample and updated values

### `AX25Timers.update_t1_timeout(self, new_timeout: float) -> None`

Override T1 base and reset RTO to the new value.
**Raises**: `ValueError` if not in 0.1-60.0 range
**Logs**: INFO

### `AX25Timers.update_t3_timeout(self, new_timeout: float) -> None`

Override T3 timeout.
**Raises**: `ValueError` if not in 10.0-3600.0 range
**Logs**: INFO

### `AX25Timers.get_timer_status(self) -> dict`

Return snapshot: t1_running, t3_running, t1_current, t3_current, srtt, rttvar, rto, last_ack_time.

### `AX25Timers.reset(self) -> None`

Stop all timers and restore SRTT/RTO to config values.
**Logs**: INFO

---

## Appendix F: Function Reference -- core/negotiation.py

### `build_xid_frame(config: AX25Config) -> bytes`

Build TLV-encoded XID information field bytes from local config.
**Parameters**: `config` -- local AX25Config
**Returns**: Bytes for XID frame info field
**Logs**: DEBUG with parameters; INFO with total byte count

### `parse_xid_frame(info: bytes) -> Dict[int, int]`

Parse TLV bytes from received XID info field into a dict.
**Parameters**: `info` -- raw XID info bytes
**Returns**: Dict mapping parameter type code to integer value
**Raises**: `NegotiationError` if truncated, bad length, or extra bytes
**Logs**: DEBUG for each TLV; INFO with total count

### `negotiate_config(local: AX25Config, remote_params: Dict[int, int]) -> AX25Config`

Apply AX.25 negotiation rules to produce a final agreed configuration.
**Parameters**: `local` -- local config; `remote_params` -- from parse_xid_frame
**Returns**: New AX25Config with negotiated values
**Raises**: `NegotiationError` if modulo mismatch or negotiated values are invalid
**Logs**: DEBUG with input; INFO with final result

---

## Appendix G: Function Reference -- core/validation.py

### `validate_frame_structure(frame: AX25Frame, config: Optional[AX25Config] = None) -> None`

Check I/S/U frame structure rules per AX.25 v2.2.
**Parameters**: `frame` -- decoded frame; `config` -- optional override for N1 check
**Raises**: `InvalidAddressError` (too many digis), `InvalidControlFieldError` (bad PID presence), `FrameError` (info too long or S-frame with info)
**Logs**: DEBUG on pass

### `full_validation(frame: AX25Frame, config: Optional[AX25Config] = None) -> None`

Run all validators. Currently calls validate_frame_structure.
**Logs**: DEBUG at start; INFO on pass

---

## Appendix H: Function Reference -- core/connected.py

### `AX25Connection.__init__(self, local_addr, remote_addr, config, initiate, transport) -> None`

Create a connection object. If initiate=True, immediately triggers connect_request transition.
**Logs**: INFO with addresses, initiate flag, modulo

### `AX25Connection.connect(self) -> AX25Frame` (async)

Send SABM or SABME and start T1. Must be in AWAITING_CONNECTION state.
**Returns**: The SABM/SABME frame sent
**Raises**: `ConnectionStateError` if wrong state
**Logs**: INFO with frame type and destination

### `AX25Connection.disconnect(self) -> AX25Frame` (async)

Send DISC and start T1. Must be in CONNECTED or TIMER_RECOVERY state.
**Returns**: The DISC frame sent
**Raises**: `ConnectionStateError` if wrong state
**Logs**: INFO with destination

### `AX25Connection.send_data(self, data: bytes) -> None` (async)

Queue bytes for I-frame transmission. Triggers immediate send if peer is ready.
**Parameters**: `data` -- payload bytes (max N1)
**Raises**: `ConnectionStateError` if not CONNECTED; `FrameError` if data too long
**Logs**: DEBUG with byte count and queue size

### `AX25Connection.process_frame(self, frame: AX25Frame) -> None`

Dispatch a received frame to the appropriate handler and reset T3.
**Logs**: DEBUG with state and control field

### `AX25Connection._on_t1_timeout(self) -> None`

T1 expiry callback. Increments retry count. Disconnects if N2 exceeded. Otherwise retransmits and restarts T1.
**Logs**: WARNING with retry count; ERROR if max exceeded

### `AX25Connection._on_t3_timeout(self) -> None`

T3 expiry callback. Sends RR probe with P=1.
**Logs**: WARNING with destination

---

## Appendix I: Function Reference -- interfaces/kiss.py

### `KISSInterface.__init__(self, port, baudrate, tnc_address, timeout, frame_config) -> None`

Set up KISS interface. Does not open the port.
**Logs**: INFO with port, baud, TNC address

### `KISSInterface.connect(self) -> None`

Open serial port and start reader thread.
**Raises**: `KISSError` if serial port cannot be opened
**Logs**: INFO

### `KISSInterface.disconnect(self) -> None`

Close serial port and join reader thread. Safe to call when not connected.
**Logs**: INFO

### `KISSInterface.send_frame(self, frame, cmd) -> None`

Encode frame in KISS format and write to serial port.
**Parameters**: `frame` -- AX25Frame; `cmd` -- KISS command low nibble (default CMD_DATA)
**Raises**: `KISSError` if not connected or send fails
**Logs**: INFO with cmd byte and frame sizes

### `KISSInterface.set_parameter(self, cmd, value) -> None`

Send a KISS parameter command (TXDELAY, PERSIST, etc.) as a raw frame.
**Parameters**: `cmd` -- command nibble; `value` -- 0-255 parameter byte
**Raises**: `KISSError` if not connected or send fails
**Logs**: DEBUG

### `KISSInterface.receive(self, timeout) -> Tuple[int, int, AX25Frame]`

Block for and return next received frame as (tnc_addr, port, frame).
**Raises**: `KISSError` on timeout

### `KISSInterface.register_callback(self, cmd, callback) -> None`

Register callable(tnc_addr, port, frame) for a KISS command low nibble.
**Logs**: DEBUG

---

## Appendix J: Function Reference -- interfaces/agwpe.py

### `AGWPEInterface.__init__(self, host, port, timeout) -> None`

Set up AGWPE client. Does not connect.
**Logs**: INFO with host, port, timeout

### `AGWPEInterface.connect(self) -> None`

Open TCP socket to AGWPE server and start reader thread.
**Raises**: `AGWPEConnectionError` if TCP connect fails
**Logs**: INFO

### `AGWPEInterface.disconnect(self) -> None`

Close socket and join reader thread. Safe when not connected.
**Logs**: INFO

### `AGWPEInterface.register_callsign(self, callsign: str) -> None`

Send 'X' command to register callsign. Server replies with 'X'.
**Raises**: `AGWPEConnectionError` if not connected
**Logs**: INFO

### `AGWPEInterface.enable_monitoring(self) -> None`

Send 'm' command to enable frame monitoring.
**Logs**: INFO

### `AGWPEInterface.disable_monitoring(self) -> None`

Send 'm' command to disable frame monitoring.
**Logs**: INFO

### `AGWPEInterface.query_outstanding_frames(self, port, callsign) -> None`

Send 'y' command. Server replies with 'Y' containing count.
**Logs**: DEBUG

### `AGWPEInterface.query_port_capabilities(self, port) -> None`

Send 'g' command. Server replies with 'g' containing capabilities.
**Logs**: DEBUG

### `AGWPEInterface.query_version(self) -> None`

Send 'R' command. Server replies with 'R' containing version string.
**Logs**: DEBUG

### `AGWPEInterface.enable_raw_frames(self) -> None`

Send 'k' command to enable raw AX.25 frame monitoring.
**Logs**: INFO

### `AGWPEInterface.register_callback(self, data_kind, callback) -> None`

Register callback(port, call_from, call_to, data) for a DataKind.
**Raises**: `ValueError` if data_kind is not a single character
**Logs**: DEBUG

### `AGWPEInterface.receive(self, timeout) -> Tuple[int, str, str, str, bytes]`

Block for and return next frame as (port, data_kind, call_from, call_to, data).
**Raises**: `AGWPEConnectionError` on timeout

---

Copyright (C) 2026 Kris Kirby, KE4AHR
