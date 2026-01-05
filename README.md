
# PyAX25_22 – Pure Python AX.25 v2.2 Layer 2 Implementation

**A complete, modern, and fully compliant AX.25 v2.2 Link Layer library for amateur packet radio.**

**Author:** Kris Kirby, KE4AHR  
**License:** GNU Lesser General Public License v3 or later (LGPL-3.0-or-later)  
**Version:** 0.5.15  
**Release Date:** January 05, 2026  
**Repository:** https://github.com/ke4ahr/PyAX25_22

## Overview

PyAX25_22 is a pure-Python implementation of the **AX.25 v2.2** (July 1998) Link Layer protocol used in amateur packet radio. It provides a clean, well-tested, and production-ready foundation for applications including:

- PACSAT ground stations (via future PyPACSAT)
- Packet radio nodes and BBS systems
- APRS monitoring and iGates
- Software TNC integration (Dire Wolf, soundmodem)
- Custom packet tools

The library focuses exclusively on **Layer 2** – no higher-layer protocols (NET/ROM, FBB/B2F, etc.) are included, ensuring maximum reusability.

## Key Features

- **Full AX.25 v2.2 compliance**
  - All frame types: I, S (RR/RNR/REJ/SREJ), U (SABM/SABME/UA/DISC/DM/FRMR/UI/XID/TEST)
  - Modulo 8 and Modulo 128 operation
  - Up to 8 digipeaters with proper H-bit handling
  - Bit stuffing/destuffing and CRC-16/CCITT-FALSE FCS
  - XID parameter negotiation (modulo, window size k, max frame N1, SREJ)
- **Connected and unconnected modes**
  - Complete state machine per SDL diagrams
  - Adaptive T1 timer with SRTT algorithm
  - Full flow control (RR/RNR/REJ/SREJ)
- **Transport interfaces**
  - **KISS** with full **multi-drop support** (G8BPQ extension)
  - **AGWPE TCP/IP API** complete client (registration, monitoring, queue query)
- **Concurrency**
  - Synchronous and asynchronous APIs
  - Thread-safe background I/O handling
- **Quality**
  - Comprehensive error handling with custom exception hierarchy
  - Structured, configurable logging
  - >95% test coverage
  - Full type hints and Google-style docstrings

## Installation

    pip install pyax25-22

For development (includes testing and documentation tools):

    pip install pyax25-22[dev]

## Quick Examples

### Send a UI Beacon

    from pyax25_22.core.framing import AX25Frame, AX25Address

    dest = AX25Address("APRS", ssid=0)
    src = AX25Address("KE4AHR", ssid=1)

    frame = AX25Frame(
        destination=dest,
        source=src,
        control=0x03,      # UI frame
        pid=0xF0,          # No Layer 3 protocol
        info=b"PyAX25_22 beacon test"
    )

    print(f"Encoded frame ({len(frame.encode())} bytes): {frame.encode().hex()}")

### Monitor via Multi-Drop KISS

    from pyax25_22.interfaces.kiss import KISSInterface

    kiss = KISSInterface("/dev/ttyUSB0", baudrate=9600, tnc_address=1)

    def on_frame(tnc_addr, port, frame):
        print(f"TNC {tnc_addr} | {frame.source.callsign}-{frame.source.ssid} → {frame.destination.callsign}")

    kiss.register_callback(0x00, on_frame)  # Data frames
    kiss.connect()

    try:
        while True:
            tnc_addr, port, frame = kiss.receive()
            # Handled by callback
    except KeyboardInterrupt:
        kiss.disconnect()

### Monitor via AGWPE

    from pyax25_22.interfaces.agwpe import AGWPEInterface

    agwpe = AGWPEInterface()

    def on_monitored(port, fr, to, data):
        print(f"[{port}] {fr} → {to}: {data.decode(errors='ignore')}")

    agwpe.register_callback('M', on_monitored)
    agwpe.connect()
    agwpe.enable_monitoring()

    try:
        while True:
            port, kind, fr, to, data = agwpe.receive()
            print(f"{kind}: {fr}->{to}")
    except KeyboardInterrupt:
        agwpe.disconnect()

## Documentation

- Compliance Report: [docs/compliance.md](docs/compliance.md)
- API Reference: [docs/api_reference.md](docs/api_reference.md)
- Examples: [examples/](examples/)

## References

Built from primary sources:

- AX.25 Link Access Protocol for Amateur Packet Radio v2.2 (July 1998)
- Multi-Drop KISS Operation – Karl Medcalf WK5M
- AGW TCP/IP Socket Interface – George Rossopoulos SV2AGW (2000)
- AGWPE TCP/IP API Tutorial – Pedro E. Colla LU7DID & SV2AGW
- PACSAT File Header Definition – Jeff Ward G0/K8KA & Harold Price NK6K

## Contributing

Contributions are welcome! Please:
- Follow existing code style
- Add tests for new features
- Update documentation

## License

Licensed under **LGPL-3.0-or-later**.

Full license text: [LICENSE](LICENSE)

---

73 de KE4AHR

Copyright (C) 2025-2026 Kris Kirby, KE4AHR
