# KISS Protocol Specification (PyAX25_22 Implementation)

## Introduction
The KISS (Keep It Simple, Stupid) protocol provides a hardware-agnostic interface for AX.25 packet communication. This document covers both standard RFC 1055 KISS and PyAX25_22's multi-drop extensions.

**License:** LGPLv3.0
**Copyright** (C) 2025-2026 Kris Kirby, KE4AHR


## Frame Format

### Basic Structure
    [FEND][Command][Data...][FEND]

- **FEND**: Frame End (0xC0)
- **Command**: 1 byte combining TNC address and command type
- **Data**: Variable-length payload

### Command Byte Structure
    MSB                           LSB
    7 6 5 4 | 3 2 1 0
    --------+--------
    TNC Addr| Command

- **Bits 4-7**: TNC address (0-15)
- **Bits 0-3**: Command code

## Byte Stuffing

| Original Byte | Stuffed Sequence |
|---------------|------------------|
| 0xC0 (FEND)   | 0xDB 0xDC        |
| 0xDB (FESC)   | 0xDB 0xDD        |

Example:
    Original: 0x01 0xC0 0xDB
    Stuffed:  0x01 0xDB 0xDC 0xDB 0xDD

## Command Set

### Standard Commands
| Command | Hex | Description          |
|---------|-----|----------------------|
| DATA    | 0x00| Normal data frame    |
| TXDELAY | 0x01| Transmitter delay    |
| PERSIST | 0x02| Persistence value    |
| SLOTTIME| 0x03| Slot interval        |
| TXTAIL  | 0x04| Transmitter tail     |
| FULLDUP | 0x05| Duplex mode          |
| SET_HW  | 0x06| Hardware config      |
| RETURN  | 0xFF| Exit KISS mode       |

### Multi-Drop Extensions
| Command | Hex | Description          |
|---------|-----|----------------------|
| POLL    | 0x0E| TNC poll command     |
| XDATA   | 0x0F| Extended data frame  |

## Multi-Drop Operation

### Addressing Scheme
- **Master**: Address 0 (controller)
- **Slaves**: Addresses 1-15 (devices)

### Polling Mechanism
1. Master sends poll frame:
  
    0xC0 0x2E 0xC0  # Poll TNC 2 (0x2E = (2<<4)|0x0E)

2. Slave responds with data frame:

    0xC0 0x20 0x... 0xC0  # TNC 2 response (0x20 = (2<<4)|0x00)

## TCP Transport Implementation

### Connection Parameters
- **Default Port**: 8001
- **Keepalive**: Enabled (60s idle, 10s interval)
- **Reconnect**: Automatic with backoff

### Frame Format over TCP
    [FEND][Command][Escaped Data][FEND]

## Error Handling

| Condition               | Action                      |
|-------------------------|-----------------------------|
| Invalid FEND sequences  | Discard frame               |
| Bad byte stuffing       | Discard frame               |
| Unknown command         | Log and ignore              |
| Poll timeout            | Retry (configurable)        |

## Examples

### Basic UI Frame
    Frame:
    0xC0 0x00 H e l l o 0xC0
    
    Structure:
    [FEND][DATA][Payload][FEND]

### Multi-Drop Poll/Response
    Master -> Slave:
    0xC0 0x1E 0xC0  # Poll TNC 1
    
    Slave -> Master:
    0xC0 0x10 D a t a 0xC0  # Response from TNC 1

## References
1. RFC 1055 - KISS Protocol Specification
2. G8BPQ Multi-drop Extension Document
3. AX.25 Link Layer Protocol
4. PyAX25_22 Source Code

