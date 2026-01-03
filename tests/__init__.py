# tests/__init__.py
"""
PyAX25_22 Test Package Initialization

Provides:
- Common test fixtures
- Shared test utilities
- Project-wide test configuration

License: LGPLv3.0
Copyright (C) 2025-2026 Kris Kirby, KE4AHR
"""

import pytest
import logging
from typing import Generator

# Project-wide test configuration
pytest_plugins = []

@pytest.fixture(scope="session", autouse=True)
def configure_logging() -> Generator[None, None, None]:
    """Configure logging for all tests"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    yield

@pytest.fixture
def sample_ax25_frame() -> bytes:
    """Sample valid AX.25 frame for testing"""
    from pyax25_22.core.framing import (
        AX25Frame,
        AX25Address,
        PID,
        bit_stuff,
        fcs_calc
    )
    
    frame = AX25Frame(
        dest=AX25Address("DEST", 1),
        src=AX25Address("SRC", 2)
    ).encode_ui(b"test payload")
    
    return frame

@pytest.fixture
def mock_serial() -> Generator[Mock, None, None]:
    """Mock serial port fixture"""
    with patch('serial.Serial') as mock:
        yield mock()

@pytest.fixture
def mock_socket() -> Generator[Mock, None, None]:
    """Mock socket fixture"""
    with patch('socket.socket') as mock:
        yield mock()

def pytest_configure(config: pytest.Config) -> None:
    """Pytest configuration hook"""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", 
        "hardware: mark test that requires actual hardware"
    )

# Enable shared test objects
__all__ = [
    'configure_logging',
    'sample_ax25_frame',
    'mock_serial',
    'mock_socket'
]

