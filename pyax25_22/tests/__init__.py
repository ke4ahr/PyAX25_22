# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
PyAX25_22 Test Package Initialization

Provides:
- Common test fixtures
- Shared test utilities
- Project-wide test configuration
- Test data generators
- Performance benchmarking utilities

"""

import pytest
import logging
import time
import random
import string
import threading
import asyncio
import socket
import serial
from typing import List, Tuple, Dict, Any, Optional, Callable, Generator, Union
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from contextlib import contextmanager
import tempfile
import os
import json
import statistics
from dataclasses import dataclass, asdict
from enum import Enum
import sys

# Test configuration and metadata
__version__ = "0.1.0"
__author__ = "QA Team"
__email__ = "qa@example.com"
__description__ = "Test suite for PyAX25_22"
__license__ = "LGPLv3.0"
__copyright__ = "Copyright (C) 2024 QA Team"

# Test environment configuration
TEST_TIMEOUT_DEFAULT = 30.0
TEST_RETRIES_DEFAULT = 3
TEST_DATA_DIR = "test_data"
TEST_LOG_LEVEL = logging.DEBUG

class TestEnvironment(Enum):
    """Test environment types"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    STRESS = "stress"
    MOCK = "mock"

class TestDataGenerator:
    """Generate test data for various AX.25 components"""
    
    @staticmethod
    def generate_callsign(length: Optional[int] = None) -> str:
        """Generate random amateur radio callsign.
        
        Args:
            length: Length of callsign (1-6 characters)
            
        Returns:
            Random callsign string
        """
        if length is None:
            length = random.randint(1, 6)
        
        # Amateur radio callsign patterns: letter + numbers + letters
        prefix = random.choice(string.ascii_uppercase)
        middle = ''.join(random.choices(string.digits, k=random.randint(0, 3)))
        suffix = ''.join(random.choices(string.ascii_uppercase, k=length - len(prefix) - len(middle)))
        
        return (prefix + middle + suffix)[:6].ljust(6, ' ')

    @staticmethod
    def generate_ssid() -> int:
        """Generate random SSID (0-15)."""
        return random.randint(0, 15)

    @staticmethod
    def generate_frame_data(
        min_size: int = 0,
        max_size: int = 1024,
        include_special: bool = True
    ) -> bytes:
        """Generate random frame data.
        
        Args:
            min_size: Minimum data size
            max_size: Maximum data size
            include_special: Include special characters
            
        Returns:
            Random frame data bytes
        """
        size = random.randint(min_size, max_size)
        
        if include_special:
            # Include FLAG, ESC, and other special bytes
            special_chars = [0x7E, 0x7D, 0x00, 0xFF, 0x0D, 0x0A]
            data = bytes(random.choices(
                list(range(256)),
                weights=[0.9 if b not in special_chars else 0.1 for b in range(256)],
                k=size
            ))
        else:
            # Only printable ASCII
            data = bytes(random.choices(
                [ord(c) for c in string.printable],
                k=size
            ))
        
        return data

    @staticmethod
    def generate_ax25_address() -> Tuple[str, int]:
        """Generate random AX.25 address.
        
        Returns:
            Tuple of (callsign, ssid)
        """
        callsign = TestDataGenerator.generate_callsign()
        ssid = TestDataGenerator.generate_ssid()
        return callsign, ssid

    @staticmethod
    def generate_test_frames(count: int = 10) -> List[Dict[str, Any]]:
        """Generate test frame data.
        
        Args:
            count: Number of frames to generate
            
        Returns:
            List of frame test data dictionaries
        """
        frames = []
        
        for i in range(count):
            dest_callsign, dest_ssid = TestDataGenerator.generate_ax25_address()
            src_callsign, src_ssid = TestDataGenerator.generate_ax25_address()
            
            # Ensure different addresses
            while dest_callsign == src_callsign and dest_ssid == src_ssid:
                src_callsign, src_ssid = TestDataGenerator.generate_ax25_address()
            
            frame_data = {
                'dest_callsign': dest_callsign,
                'dest_ssid': dest_ssid,
                'src_callsign': src_callsign,
                'src_ssid': src_ssid,
                'info': TestDataGenerator.generate_frame_data(10, 100),
                'pid': random.choice([0xF0, 0xCC, 0x01]),  # Common PIDs
                'sequence_ns': random.randint(0, 7),  # Modulo 8
                'sequence_nr': random.randint(0, 7),
                'poll': random.choice([True, False])
            }
            frames.append(frame_data)
        
        return frames

    @staticmethod
    def generate_kiss_commands(count: int = 5) -> List[Tuple[int, bytes]]:
        """Generate KISS command test data.
        
        Args:
            count: Number of commands to generate
            
        Returns:
            List of (command, data) tuples
        """
        commands = []
        kiss_commands = [0x00, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x10, 0xFF]  # KISS commands
        
        for _ in range(count):
            cmd = random.choice(kiss_commands)
            data = TestDataGenerator.generate_frame_data(0, 50)
            commands.append((cmd, data))
        
        return commands

class TestConfiguration:
    """Test configuration management"""
    
    def __init__(self):
        self.config = self._load_default_config()
        self._override_config = {}
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default test configuration."""
        return {
            'test_timeout': TEST_TIMEOUT_DEFAULT,
            'test_retries': TEST_RETRIES_DEFAULT,
            'log_level': TEST_LOG_LEVEL,
            'mock_network': True,
            'mock_serial': True,
            'performance_thresholds': {
                'frame_encoding_ms': 10.0,
                'frame_decoding_ms': 10.0,
                'fcs_calculation_ms': 1.0,
                'bit_stuffing_ms': 5.0
            },
            'test_data_size': {
                'small': 100,
                'medium': 1000,
                'large': 10000,
                'xlarge': 100000
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if key in self._override_config:
            return self._override_config[key]
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._override_config[key] = value
    
    def reset(self) -> None:
        """Reset to default configuration."""
        self._override_config.clear()

# Global test configuration
test_config = TestConfiguration()

@dataclass
class PerformanceResult:
    """Performance test result data"""
    test_name: str
    operation_count: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    operations_per_second: float
    memory_usage: Optional[int] = None
    additional_metrics: Dict[str, Any] = None

class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    @staticmethod
    def benchmark_operation(
        operation: Callable[[], Any],
        iterations: int = 100,
        warmup_iterations: int = 10
    ) -> PerformanceResult:
        """Benchmark an operation.
        
        Args:
            operation: Function to benchmark
            iterations: Number of iterations to measure
            warmup_iterations: Number of warmup iterations
            
        Returns:
            PerformanceResult with timing statistics
        """
        # Warmup
        for _ in range(warmup_iterations):
            operation()
        
        # Benchmark
        times = []
        total_start = time.perf_counter()
        
        for _ in range(iterations):
            start = time.perf_counter()
            result = operation()
            end = time.perf_counter()
            times.append(end - start)
        
        total_time = time.perf_counter() - total_start
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        ops_per_second = iterations / total_time
        
        return PerformanceResult(
            test_name=operation.__name__ if hasattr(operation, '__name__') else 'anonymous',
            operation_count=iterations,
            total_time=total_time,
            average_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            operations_per_second=ops_per_second
        )
    
    @staticmethod
    def benchmark_memory(operation: Callable[[], Any], iterations: int = 100) -> Dict[str, Any]:
        """Benchmark memory usage of an operation.
        
        Args:
            operation: Function to benchmark
            iterations: Number of iterations
            
        Returns:
            Memory usage statistics
        """
        import tracemalloc
        
        tracemalloc.start()
        
        for _ in range(iterations):
            operation()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'current_memory': current,
            'peak_memory': peak,
            'average_memory': current / iterations if iterations > 0 else 0
        }

class TestUtilities:
    """Common test utilities and helpers"""
    
    @staticmethod
    @contextmanager
    def temporary_file(content: bytes = b"", suffix: str = ".tmp") -> str:
        """Create a temporary file for testing.
        
        Args:
            content: Content to write to file
            suffix: File suffix
            
        Yields:
            Temporary file path
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            yield temp_path
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    @staticmethod
    @contextmanager
    def mock_serial_port() -> Mock:
        """Create a mock serial port for testing.
        
        Yields:
            Mock serial port object
        """
        mock_serial = Mock(spec=serial.Serial)
        mock_serial.is_open = True
        mock_serial.in_waiting = 0
        mock_serial.write = Mock(return_value=10)
        mock_serial.read = Mock(return_value=b'')
        mock_serial.flush = Mock()
        mock_serial.close = Mock()
        mock_serial.port = '/dev/ttyUSB0'
        mock_serial.baudrate = 9600
        
        yield mock_serial
    
    @staticmethod
    @contextmanager
    def mock_socket() -> Mock:
        """Create a mock socket for testing.
        
        Yields:
            Mock socket object
        """
        mock_socket = Mock(spec=socket.socket)
        mock_socket.connect = Mock()
        mock_socket.send = Mock(return_value=10)
        mock_socket.recv = Mock(return_value=b'')
        mock_socket.shutdown = Mock()
        mock_socket.close = Mock()
        mock_socket.settimeout = Mock()
        mock_socket.setsockopt = Mock()
        
        yield mock_socket
    
    @staticmethod
    async def async_timeout(timeout: float, coro: Callable[[], Any]) -> Any:
        """Run a coroutine with timeout.
        
        Args:
            timeout: Timeout in seconds
            coro: Coroutine to run
            
        Returns:
            Coroutine result
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        return await asyncio.wait_for(coro(), timeout=timeout)
    
    @staticmethod
    def retry_with_backoff(
        operation: Callable[[], Any],
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0
    ) -> Any:
        """Retry operation with exponential backoff.
        
        Args:
            operation: Function to retry
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            backoff_factor: Exponential backoff factor
            
        Returns:
            Operation result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    raise
                
                delay = base_delay * (backoff_factor ** attempt)
                time.sleep(delay)
        
        raise last_exception

# Pytest fixtures and configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers",
        "stress: mark test as stress test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "network: mark test as requiring network"
    )
    config.addinivalue_line(
        "markers",
        "serial: mark test as requiring serial port"
    )
    config.addinivalue_line(
        "markers",
        "hardware: mark test as requiring hardware"
    )

@pytest.fixture(scope="session", autouse=True)
def configure_logging() -> None:
    """Configure logging for all tests."""
    logging.basicConfig(
        level=test_config.get('log_level'),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]",
        force=True  # Override any existing configuration
    )
    
    # Configure specific loggers
    logger = logging.getLogger('pyax25_22')
    logger.setLevel(test_config.get('log_level'))
    
    # Add test-specific handlers if needed
    test_logger = logging.getLogger('pyax25_22.tests')
    test_logger.info("Test suite initialized")

@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator."""
    return TestDataGenerator()

@pytest.fixture
def test_config_manager() -> TestConfiguration:
    """Provide test configuration manager."""
    return test_config

@pytest.fixture
def performance_benchmark() -> PerformanceBenchmark:
    """Provide performance benchmarking utilities."""
    return PerformanceBenchmark()

@pytest.fixture
def test_utilities() -> TestUtilities:
    """Provide test utilities."""
    return TestUtilities()

@pytest.fixture(params=['small', 'medium', 'large'])
def test_data_size(request) -> int:
    """Provide different test data sizes."""
    return test_config.get('test_data_size', {}).get(request.param, 1000)

@pytest.fixture
def mock_transport() -> Mock:
    """Provide mock transport for testing."""
    mock = Mock()
    mock.send_frame = Mock()
    mock.read_data = Mock(return_value=b'')
    mock.connect = Mock()
    mock.disconnect = Mock()
    return mock

@pytest.fixture
def sample_ax25_frame_data() -> Dict[str, Any]:
    """Provide sample AX.25 frame test data."""
    return {
        'dest_call': 'DEST',
        'dest_ssid': 1,
        'src_call': 'SRC',
        'src_ssid': 2,
        'info': b'Hello AX.25!',
        'pid': 0xF0,
        'ns': 3,
        'nr': 5,
        'poll': False
    }

@pytest.fixture
def sample_kiss_frame_data() -> Tuple[int, bytes]:
    """Provide sample KISS frame test data."""
    return (0x00, b'Test KISS frame data')

@pytest.fixture
def temp_directory() -> str:
    """Provide temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
async def async_test_event_loop() -> asyncio.AbstractEventLoop:
    """Provide async test event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

# Test decorators and utilities
def skip_if_no_network(func: Callable) -> Callable:
    """Decorator to skip tests requiring network."""
    @pytest.mark.network
    def wrapper(*args, **kwargs):
        if test_config.get('mock_network', True):
            pytest.skip("Network tests disabled in mock mode")
        return func(*args, **kwargs)
    return wrapper

def skip_if_no_serial(func: Callable) -> Callable:
    """Decorator to skip tests requiring serial port."""
    @pytest.mark.serial
    def wrapper(*args, **kwargs):
        if test_config.get('mock_serial', True):
            pytest.skip("Serial tests disabled in mock mode")
        return func(*args, **kwargs)
    return wrapper

def performance_test(threshold_ms: float = 100.0):
    """Decorator for performance tests with thresholds."""
    def decorator(func: Callable) -> Callable:
        @pytest.mark.performance
        @pytest.mark.slow
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            if duration_ms > threshold_ms:
                pytest.fail(f"Performance test exceeded threshold: {duration_ms:.2f}ms > {threshold_ms}ms")
            
            return result
        return wrapper
    return decorator

def stress_test(iterations: int = 1000):
    """Decorator for stress tests."""
    def decorator(func: Callable) -> Callable:
        @pytest.mark.stress
        @pytest.mark.slow
        def wrapper(*args, **kwargs):
            errors = []
            for i in range(iterations):
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    errors.append(f"Iteration {i}: {e}")
                    
                    # Allow some errors in stress testing
                    if len(errors) > iterations * 0.01:  # More than 1% failure rate
                        pytest.fail(f"Stress test failed with {len(errors)} errors: {errors}")
            
            if errors:
                pytest.warns(UserWarning, match=f"Stress test had {len(errors)} errors")
            
            return result
        return wrapper
    return decorator

# Test data persistence utilities
class TestDataPersistence:
    """Utilities for saving and loading test data."""
    
    @staticmethod
    def save_test_data(data: Any, filename: str, format: str = 'json') -> str:
        """Save test data to file.
        
        Args:
            data: Data to save
            filename: Output filename
            format: Format ('json', 'pickle')
            
        Returns:
            Path to saved file
        """
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'pickle':
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return filename
    
    @staticmethod
    def load_test_data(filename: str, format: str = 'json') -> Any:
        """Load test data from file.
        
        Args:
            filename: Input filename
            format: Format ('json', 'pickle')
            
        Returns:
            Loaded data
        """
        if format == 'json':
            with open(filename, 'r') as f:
                return json.load(f)
        elif format == 'pickle':
            import pickle
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")

# Test result aggregation and reporting
class TestResults:
    """Aggregate and report test results."""
    
    def __init__(self):
        self.results = []
        self.performance_results = []
        self.error_count = 0
        self.warning_count = 0
    
    def add_result(self, test_name: str, success: bool, duration: float, error: Optional[str] = None):
        """Add test result."""
        self.results.append({
            'test_name': test_name,
            'success': success,
            'duration': duration,
            'error': error,
            'timestamp': time.time()
        })
        
        if not success:
            self.error_count += 1
    
    def add_performance_result(self, result: PerformanceResult):
        """Add performance test result."""
        self.performance_results.append(result)
    
    def add_warning(self, warning: str):
        """Add warning."""
        self.warning_count += 1
        logger = logging.getLogger('pyax25_22.tests')
        logger.warning(f"Test warning: {warning}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(r['duration'] for r in self.results) if self.results else 0
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'performance_tests': len(self.performance_results)
        }
    
    def save_results(self, filename: str):
        """Save results to file."""
        data = {
            'summary': self.get_summary(),
            'test_results': self.results,
            'performance_results': [asdict(r) for r in self.performance_results],
            'metadata': {
                'version': __version__,
                'timestamp': time.time(),
                'python_version': sys.version
            }
        }
        
        TestDataPersistence.save_test_data(data, filename)

# Global test results tracker
test_results = TestResults()

# Package initialization
def initialize_test_environment():
    """Initialize test environment."""
    # Create test data directory
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Set random seed for reproducible tests
    random.seed(42)
    
    # Configure test logging
    logger = logging.getLogger('pyax25_22.tests')
    logger.info(f"Test environment initialized: {__version__}")

# Initialize on import
initialize_test_environment()

# Export public API
__all__ = [
    # Configuration
    'test_config',
    'TestConfiguration',
    'test_results',
    'TestResults',
    
    # Data generation
    'TestDataGenerator',
    'TestEnvironment',
    
    # Performance testing
    'PerformanceBenchmark',
    'PerformanceResult',
    'performance_test',
    'stress_test',
    
    # Test utilities
    'TestUtilities',
    'TestDataPersistence',
    
    # Pytest fixtures (available for import)
    'test_data_generator',
    'test_config_manager',
    'performance_benchmark',
    'test_utilities',
    'test_data_size',
    'mock_transport',
    'sample_ax25_frame_data',
    'sample_kiss_frame_data',
    'temp_directory',
    'async_test_event_loop',
    
    # Test decorators
    'skip_if_no_network',
    'skip_if_no_serial',
    
    # Constants
    'TEST_TIMEOUT_DEFAULT',
    'TEST_RETRIES_DEFAULT',
    'TEST_DATA_DIR',
    'TEST_LOG_LEVEL',
]

# Clean up imports
del pytest, logging, time, random, string, threading, asyncio, socket, serial
del List, Tuple, Dict, Any, Optional, Callable, Generator, Union, Enum
del contextmanager, tempfile, json, statistics, dataclass, asdict, sys
