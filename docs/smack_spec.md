# SMACK Protocol Specification

**SMACK -- Stuttgart Modified Amateurradio-CRC-KISS**

For communication via the TNC -- PC serial interface.

**Original Authors:** Jan Schiefer, DL5UE/G0TRR and Dieter Deyke, DK5SG/N0PRA
**Original Date:** February 1991
**Source:** http://www.symek.com/g/smack.html (SYMEK GmbH, Stuttgart)
**Imported:** 2026-03-21

---

## 1. Background

KISS was proposed in 1986 by Phil Karn, KA9Q. For his TCP/IP software a protocol was required
that made access to packet radio possible below the AX.25 protocol level. KISS provides Level 2a
operation. The TNC handles only media access control (data carrier detect, channel-busy detection,
p-persistence CSMA) and conversion of synchronous HDLC data to asynchronous RS-232 format.

The KISS protocol manages separation of packets by delimiters, treatment of delimiter characters
that may appear within the data stream, and a simple set of commands for setting TNC parameters.

In February 1991, Jan Schiefer DL5UE/G0TRR and Dieter Deyke DK5SG/N0PRA proposed an enhanced
and fully backward-compatible version of KISS that adds a CRC checksum over KISS data packets to
ensure data integrity over the RS-232 line. This enhanced KISS was called SMACK.

---

## 2. KISS Command Byte Structure

The second byte of a KISS frame is the command byte:

```
 d7   d6   d5   d4   d3   d2   d1   d0
+----+----+----+----+----+----+----+----+
| SM | P4  | P2  | P1  | C3  | C2  | C1  | C0 |
+----+----+----+----+----+----+----+----+

SM     = SMACK/CRC flag (0 = standard KISS, 1 = SMACK with CRC)
P4,P2,P1 = port number bits (binary encoded: port = P4*4 + P2*2 + P1*1), range 0-7
C3-C0  = command code (low nibble), range 0x0-0xF
```

### Standard Command Codes (low nibble)

| Code | Command | Description |
|------|---------|-------------|
| 0x0  | DATA    | Data frame follows |
| 0x1  | TXDELAY | TX delay parameter follows |
| 0x2  | PERSIST | P-persistence parameter follows |
| 0x3  | SLOTTIME | Slot time parameter follows |
| 0x4  | TXTAIL  | TX tail parameter follows |
| 0x5  | FULLDUP | Full duplex on/off follows |
| 0xFF | EXIT    | Exit KISS / return to command mode |

---

## 3. SMACK -- CRC Extension

SMACK uses the most significant bit (d7) of the command byte as the CRC flag:

- **d7 = 0**: Standard KISS frame (no CRC). Compatible with all KISS implementations.
- **d7 = 1**: SMACK frame -- a 2-byte CRC checksum is appended after the data.

Only data frames (command code 0x0) carry a CRC. Command frames (TXDELAY, PERSIST, etc.)
are never CRC-protected. This ensures that TNC parameters can always be set even if the
host and TNC are operating in different modes.

---

## 4. Frame Formats

### 4.1 Standard KISS Frame (no CRC) -- Port 1 (0x00)

```
+------+------+------+------+-----+------+------+
| FEND | 0x00 | DATA | DATA | ... | DATA | FEND |
+------+------+------+------+-----+------+------+
```

### 4.2 Standard KISS Frame -- Port 2 (0x10)

```
+------+------+------+------+-----+------+------+
| FEND | 0x10 | DATA | DATA | ... | DATA | FEND |
+------+------+------+------+-----+------+------+
```

### 4.3 SMACK Frame with CRC -- Port 1 (0x80)

```
+------+------+------+------+-----+------+---------+---------+------+
| FEND | 0x80 | DATA | DATA | ... | DATA | CRC_LOW | CRC_HIGH | FEND |
+------+------+------+------+-----+------+---------+---------+------+
```

### 4.4 SMACK Frame with CRC -- Port 2 (0x90)

```
+------+------+------+------+-----+------+---------+---------+------+
| FEND | 0x90 | DATA | DATA | ... | DATA | CRC_LOW | CRC_HIGH | FEND |
+------+------+------+------+-----+------+---------+---------+------+
```

**Note:** CRC_LOW is the least significant byte (LSB-first ordering).

---

## 5. Multiport TNC Addressing Table

```
Command Byte Bit Layout:

d7   d6   d5   d4   d3   d2   d1   d0
SMCK  P4   P2   P1   CMD3 CMD2 CMD1 CMD0

Where:
  SMCK (d7): 0 = KISS mode, 1 = SMACK/CRC mode
  P4   (d6): Port bit for value 4
  P2   (d5): Port bit for value 2
  P1   (d4): Port bit for value 1
  CMD  (d3-d0): Command code (0=data, 1=txdelay, ...)

Port number = (d6*4) + (d5*2) + (d4*1), range 0-7 (8 ports maximum)
```

### Command Byte Examples

| Byte | Binary   | SMACK? | Port | Command |
|------|----------|--------|------|---------|
| 0x00 | 00000000 | No     | 0    | Data    |
| 0x10 | 00010000 | No     | 1    | Data    |
| 0x20 | 00100000 | No     | 2    | Data    |
| 0x30 | 00110000 | No     | 3    | Data    |
| 0x80 | 10000000 | Yes    | 0    | Data    |
| 0x90 | 10010000 | Yes    | 1    | Data    |
| 0xA0 | 10100000 | Yes    | 2    | Data    |
| 0xB0 | 10110000 | Yes    | 3    | Data    |
| 0x01 | 00000001 | No     | 0    | TXDelay |
| 0x11 | 00010001 | No     | 1    | TXDelay |

---

## 6. CRC Algorithm

SMACK uses CRC-16 with the following parameters:

- **Polynomial:** X^16 + X^15 + X^2 + 1 (0x8005 in normal form, 0xA001 reflected)
- **Name:** CRC-16/ARC, CRC-16/IBM, CRC-16/LHA
- **Initial value:** 0x0000 (preset CRC generator to zero)
- **Input reflection:** Yes (LSB first processing)
- **Output reflection:** Yes
- **XOR output:** 0x0000
- **Check value:** 0xBB3D (CRC of ASCII "123456789")
- **CRC bytes appended:** LSB first (low byte, then high byte)

### What is Covered by the CRC?

The CRC is calculated over:
1. The command byte (e.g., 0x80 or 0x90)
2. All data bytes

IMPORTANT: The CRC is calculated BEFORE KISS SLIP-encoding (escaping of FEND/FESC bytes)
and verified AFTER SLIP-decoding. This is because:
- The CRC bytes themselves may contain FEND (0xC0) or FESC (0xDB) values that need escaping
- The SLIP encoder/decoder in some implementations is separate from the KISS layer

### Verification

To verify a received frame, process all bytes including the two CRC bytes through the CRC
algorithm. If the result is zero (0x0000), the frame is valid. If not, a transmission error
occurred and the frame must be dropped.

### CRC Table Generation (C Reference Code)

From the original SYMEK specification:

```c
unsigned short Table[256];
const int Rest[8] = { 0xC0C1, 0xC181, 0xC301, 0xC601,
                      0xCC01, 0xD801, 0xF001, 0xA001 };

main() {
    int i, j;
    unsigned short value;
    for (i = 0; i < 256; i++) {
        value = 0;
        for (j = 0; j < 8; j++)
            if (i & (1 << j))
                value ^= Rest[j];
        Table[i] = value;
    }
}
```

### Python CRC-16/ARC Implementation

```python
def smack_crc16(data: bytes) -> int:
    """Compute SMACK CRC-16/ARC over the given bytes.

    This uses the reflected CRC-16/ARC algorithm (polynomial 0x8005,
    initial value 0x0000, LSB-first processing).

    Args:
        data: Bytes to compute CRC over (command byte + data bytes)

    Returns:
        16-bit CRC value (int). Append as little-endian 2 bytes.
    """
    crc = 0x0000
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF
```

---

## 7. Mode Switching (Auto-Detection)

A SMACK TNC powers up in standard KISS mode (no CRC). Both host and TNC use auto-detection
to switch modes:

### Case 1: Standard KISS TNC connected to SMACK-capable host

| Host | TNC |
|------|-----|
| Sends one packet with CRC (command byte 0x80), then switches TX back to plain KISS | |
| | Receives frame with unknown command byte 0x80 -- drops/ignores it (per KISS spec) |
| Continues transmitting KISS data without CRC | |
| | Transmits all data without CRC |

**Result:** Both sides operate in plain KISS mode. The SMACK probe is silently ignored.

### Case 2: SMACK TNC connected to SMACK-capable host

| Host | TNC |
|------|-----|
| Sends one packet with CRC (0x80), switches TX back to plain KISS | |
| | Receives frame with CRC flag -- switches TX to CRC mode |
| Transmits data WITHOUT CRC | |
| | Sends first frame WITH CRC |
| Receives frame with CRC bit set -- switches TX to CRC mode | |
| | Sends all frames WITH CRC |
| Uses SMACK data with CRC | Uses SMACK data with CRC |

**Result:** Both sides negotiate to SMACK mode automatically.

### Frame Reception Rules (always applied, regardless of TX mode)

| Received Frame | Action |
|---------------|--------|
| No CRC (command bit 7 = 0) | Process frame normally |
| With CRC, checksum OK | Process frame, possibly switch TX to CRC mode |
| With CRC, checksum FAILED | Drop frame silently |

---

## 8. KISS Escaping and SMACK

KISS uses SLIP-style escaping:
- FEND (0xC0) within data is replaced by FESC (0xDB) + TFEND (0xDC)
- FESC (0xDB) within data is replaced by FESC (0xDB) + TFESC (0xDD)

The SMACK CRC bytes are appended BEFORE escaping. This means:
- CRC bytes that happen to equal 0xC0 or 0xDB will be properly escaped
- The CRC covers the raw (pre-escape) content
- Verification decodes escaping first, then checks CRC

---

## 9. XKISS vs SMACK -- Important Distinction

PyXKISS and BPQ32's XKISS use a DIFFERENT command byte format from SMACK:

| Format | Bit 7 | Bits 4-6 | Bits 0-3 |
|--------|-------|----------|----------|
| SMACK  | CRC flag | Port (3 bits, binary, 0-7) | Command |
| XKISS  | Port MSB | Port lower 3 bits (addr 0-15) | Command |

In XKISS, the full high nibble (bits 4-7) encodes the TNC address (0-15).
In SMACK, bit 7 is reserved for the CRC flag, leaving only bits 4-6 for the port (0-7).

These formats conflict for addresses 8-15 (bits 7+4 both set in XKISS = 0x80+ command byte,
which SMACK would interpret as CRC mode). They cannot be used simultaneously on the same bus.

---

## 10. Known Implementations

As of 1991, SMACK was implemented in:
- WAMPES (Westfalia Amateur Packet Experiment Software)
- SMACK firmware v1.3 for TNC2 (by Jan Schiefer DL5UE, based on K3MC TNC2-KISS)
- NORD><LINK firmware from version 2.4 (used in most European TNC2 hardware)
- NOS TCP/IP software
- TNC3/31 and TNC4 firmware by Jimi Scherer DL1GJI

---

## 11. References

1. Karn, Phil, KA9Q; Proposed "Raw" TNC Functional Spec; 6 August 1986; USENET News
2. Rohner, Michael, DC4OX; Was ist CRC?; Packet-Radio Mailbox-Netz; May 1988
3. FTP Software, Inc.; PC/TCP Version 1.09 Packet Driver Specification; Wakefield, MA 1989
4. Schiefer, Jan, DL5UE; WAMPES -- Weiterentwicklung; 5. Uberregionales Packet-Radio-Treffen; Frankfurt 1989
5. Original specification: http://www.symek.com/g/smack.html

---

Copyright (C) 2025-2026 Kris Kirby, KE4AHR
