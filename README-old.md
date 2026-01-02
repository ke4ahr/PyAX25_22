# PyAX25_22 - Python AX.25 Protocol Suite

![License](https://img.shields.io/badge/License-LGPLv3.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## Overview

PyAX25_22 is a pure Python implementation of the AX.25 packet radio protocol (v2.2) supporting:

- KISS interfaces (serial/TCP)
- AGWPE protocol
- Multi-drop operation
- Both synchronous and asynchronous APIs
- Full UI and connected-mode framing

**Key Features**:
- Strict RFC 1055 compliance + extensions
- Thread-safe core components
- Asyncio-friendly interfaces
- Comprehensive test suite
- Minimal dependencies

## Installation

### From PyPI
    pip install pyax25_22

### From Source
    git clone https://github.com/ke4ahr/PyAX25_22.git
    cd PyAX25_22
    pip install -e .

## Usage

### Basic KISS Interface
    from pyax25_22 import SerialKISSInterface

    kiss = SerialKISSInterface(port="/dev/ttyUSB0", baudrate=9600)
    kiss.start()

    # Send UI frame
    kiss.send_frame(b"Hello APRS!")

    # Register receive callback
    def frame_handler(data, tnc):
        print(f"From TNC {tnc}: {data}")
    kiss.register_rx_callback(frame_handler)

### Async AGWPE Client
    import asyncio
    from pyax25_22 import AsyncAGWClient

    async def main():
        agw = AsyncAGWClient(host="localhost", call="MYCALL")
        await agw.connect()
        
        # Send UI frame
        await agw.send_frame(b"Test", dest="APRS")

        # Receive frames
        while True:
            frame = await agw.recv_frame()
            print(f"Received: {frame}")

    asyncio.run(main())

## Documentation

### Protocol Specifications
- [KISS Protocol](docs/kiss_spec.md)
- [AGWPE Protocol](docs/agwpe_spec.md)

### API Reference
See the `docs/` directory for detailed module documentation.

## Features

### Core
- AX.25 v2.2 framing
- Bit stuffing/destuffing
- FCS calculation
- Address encoding

### Transports
- Serial KISS
- TCP KISS
- AGWPE (sync + async)
- Multi-drop operation

### Advanced
- Connected-mode state machine
- Modulo 8/128 support
- Selective Reject (SREJ)
- Retransmission logic

## Project Status

**Current**:
- Stable core implementation
- Complete KISS/AGWPE support
- Basic connected mode

**Planned**:
- Enhanced diagnostics
- Performance optimizations
- Extended frame support

## Contributing

Contributions are welcome! Please see:
- [Issue Tracker](https://github.com/ke4ahr/PyAX25_22/issues)
- [Contributing Guidelines](CONTRIBUTING.md)

## License

LGPLv3.0 - See [LICENSE](LICENSE) for details.

## Contact

Author: Kris Kirby (KE4AHR)  
Email: ke4ahr@example.com  
GitHub: [ke4ahr](https://github.com/ke4ahr)

