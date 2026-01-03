# AGWPE Protocol Specification (PyAX25_22 Implementation)

## Introduction
AGWPE (AGW Packet Engine) is a TCP-based protocol for interfacing with packet radio software/hardware. This document covers version-compatible implementation in PyAX25_22.

**License**: LGPLv3.0  
**Copyright** (C) 2025-2026 Kris Kirby, KE4AHR


## Frame Structure

### Header Format (36 bytes)
    Offset  Length  Field       Description
    0       4       Cookie      0x00 (reserved)
    4       4       DataKind    Frame type identifier
    8       1       Port        Virtual port number
    9       1       Reserved    
    10      4       PID         Protocol ID
    14      10      CallFrom    Source callsign
    24      10      CallTo      Destination callsign
    34      2       DataLen     Payload length (little-endian)

### Data Kinds
| Type | ASCII | Description                |
|------|-------|----------------------------|
| 0x44 | D     | UI Frame                   |
| 0x4B | K     | Raw AX.25 Frame            |
| 0x52 | R     | Registration               |
| 0x58 | X     | Registration Response      |
| 0x48 | H     | Heard List Request         |
| 0x64 | d     | Connected Data             |

## Connection Setup

### Registration Process
1. Client connects to TCP port 8000
2. Client sends registration frame:

    00000000 52 00 00 00 00 00 00 00 00 00 4D 59 43 41 4C 4C
    00000010 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    
    Fields:
    - DataKind: R (0x52)
    - CallFrom: MYCALL (padded to 10 bytes)

3. Server responds with registration acknowledgment:

    00000000 58 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    00000010 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00

## Data Frames

### UI Frame Example
    Header:
    00000000 00 00 00 00 44 00 00 00 00 00 00 00 00 00 54 45
    00000010 53 54 20 20 20 20 20 20 44 45 53 54 20 20 20 20
    00000020 20 20 04 00
    
    Payload:
    0xF0 Hello

    Fields:
    - DataKind: D (UI)
    - Port: 0
    - CallFrom: TEST
    - CallTo: DEST
    - PID: 0xF0
    - DataLen: 4 (little-endian)

### Raw Frame Example
    Header:
    00000000 00 00 00 00 4B 00 00 00 00 00 00 00 00 00 54 45
    00000010 53 54 20 20 20 20 20 20 44 45 53 54 20 20 20 20
    00000020 20 20 08 00
    
    Payload:
    0x82 0xA0 0x9C 0x6E 0x96 0x88 0x8A 0x40

## Error Conditions

| Error Type              | Handling                     |
|-------------------------|------------------------------|
| Registration Failed     | Close connection             |
| Invalid Header          | Discard frame                |
| Payload Length Mismatch | Discard partial frame        |
| Keepalive Timeout       | Reconnect with backoff       |

## TCP Implementation

### Connection Parameters
- **Port**: 8000 (default)
- **Keepalive**: 60s idle, 10s interval, 5 retries
- **MTU**: 4096 bytes
- **Encoding**: ASCII callsigns

## Examples

### Complete Frame Exchange
1. Registration:

    Client -> Server:
    00000000 52 00 00 00 00 00 00 00 00 00 4D 59 43 41 4C 4C
    00000010 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00

    Server -> Client:
    00000000 58 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    00000010 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00

2. Data Transmission:

    Client -> Server:
    00000000 00 00 00 00 44 00 00 00 00 00 00 00 00 00 54 45
    00000010 53 54 20 20 20 20 20 20 44 45 53 54 20 20 20 20
    00000020 20 20 05 00
    Payload: Hello

## References
1. AGWPE Protocol Documentation
2. PyAX25_22 Source Code
3. AX.25 Link Layer Protocol
