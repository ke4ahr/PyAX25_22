# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2025-2026 Kris Kirby, KE4AHR

"""
PyAX25_22 Core Module - AX.25 Layer 2 Implementation

Contains:
- AX.25 frame encoding/decoding (v2.2 specification)
- Connection state machine for connected-mode operation
- Address handling and bit-level utilities

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import logging
import sys
import warnings
from typing import List, Optional, Union, Dict, Any

# Version information
__version__ = "0.1.0"
__author__ = "Kris Kirby"
__email__ = "ke4ahr@example.com"

# Module metadata
__description__ = "Pure Python implementation of AX.25 v2.2 protocol suite"
__license__ = "LGPLv3.0"
__copyright__ = "Copyright (C) 2024 Kris Kirby, KE4AHR"

# Minimum Python version requirement
_REQUIRED_PYTHON_VERSION = (3, 8)

def _verify_python_version() -> None:
    """Verify Python version compatibility."""
    if sys.version_info < _REQUIRED_PYTHON_VERSION:
        raise RuntimeError(
            f"PyAX25_22 requires Python { '.'.join(map(str, _REQUIRED_PYTHON_VERSION)) } "
            f"or later, but {sys.version} is installed"
        )

# Perform version check on import
_verify_python_version()

# Import core components
try:
    # Frame construction and parsing
    from .framing import (
        AX25Frame,
        encode_address,
        decode_address,
        fcs_calc,
        bit_stuff,
        bit_destuff,
        FrameType,
        AX25Address,
        FrameMetadata,
        PID
    )
    
    # Connected-mode state management
    from .statemachine import (
        AX25StateMachine,
        AX25State,
        AX25Modulo,
        SREJManager,
        FrameReject,
        TimerManager,
        AX25Timer
    )
    
    # Connection management
    from .connected import (
        ConnectedModeHandler,
        ConnectionConfig,
        ConnectionError,
        ConnectionTimeout,
        FrameValidationError,
        SequenceError,
        WindowError
    )
    
    # Flow control
    from .flow_control import (
        FlowController,
        WindowMode,
        SREJManager as FlowSREJManager
    )
    
    # Timer management
    from .timers import (
        AX25Timer as CoreTimer,
        TimerManager as CoreTimerManager,
        TimerError,
        TimerNotRunningError,
        TimerAlreadyRunningError
    )
    
    # Exceptions
    from .exceptions import (
        AX25Error,
        FrameDecodeError,
        StateMachineError,
        TransportError as CoreTransportError,
        KISSProtocolError,
        AGWProtocolError
    )
    
    _IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import core components: {e}")
    _IMPORTS_SUCCESSFUL = False
    # Re-raise to prevent partial import
    raise

# Public API exports
__all__ = [
    # Version and metadata
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    '__license__',
    '__copyright__',
    
    # Framing
    'AX25Frame',
    'encode_address',
    'decode_address',
    'fcs_calc',
    'bit_stuff',
    'bit_destuff',
    'FrameType',
    'AX25Address',
    'FrameMetadata',
    'PID',
    
    # State machine
    'AX25StateMachine',
    'AX25State',
    'AX25Modulo',
    'SREJManager',
    'FrameReject',
    'TimerManager',
    'AX25Timer',
    
    # Connection management
    'ConnectedModeHandler',
    'ConnectionConfig',
    'ConnectionError',
    'ConnectionTimeout',
    'FrameValidationError',
    'SequenceError',
    'WindowError',
    
    # Flow control
    'FlowController',
    'WindowMode',
    'FlowSREJManager',
    
    # Timer management
    'CoreTimer',
    'CoreTimerManager',
    'TimerError',
    'TimerNotRunningError',
    'TimerAlreadyRunningError',
    
    # Exceptions
    'AX25Error',
    'FrameDecodeError',
    'StateMachineError',
    'CoreTransportError',
    'KISSProtocolError',
    'AGWProtocolError',
]

# Compatibility aliases for common use cases
# These provide shorter names for frequently used classes
AX25 = AX25Frame  # Alias for the main frame class
Address = AX25Address  # Alias for address class
StateMachine = AX25StateMachine  # Alias for state machine
Timer = AX25Timer  # Alias for timer class

# Convenience functions for common operations
def create_ax25_frame(dest: str, src: str, info: bytes, 
                     dest_ssid: int = 0, src_ssid: int = 0,
                     pid: Union[PID, int] = PID.NO_LAYER3) -> AX25Frame:
    """
    Convenience function to create an AX.25 frame.
    
    Args:
        dest: Destination callsign
        src: Source callsign  
        info: Frame information payload
        dest_ssid: Destination SSID (0-15)
        src_ssid: Source SSID (0-15)
        pid: Protocol Identifier
        
    Returns:
        Configured AX25Frame instance
        
    Example:
        >>> frame = create_ax25_frame("DEST-1", "SRC-2", b"Hello World")
        >>> encoded = frame.encode_ui()
    """
    if not isinstance(dest, str) or not dest.strip():
        raise ValueError("Destination callsign must be non-empty string")
    if not isinstance(src, str) or not src.strip():
        raise ValueError("Source callsign must be non-empty string")
    if not isinstance(info, bytes):
        raise TypeError("Info must be bytes")
    if not 0 <= dest_ssid <= 15:
        raise ValueError("Destination SSID must be 0-15")
    if not 0 <= src_ssid <= 15:
        raise ValueError("Source SSID must be 0-15")
        
    dest_addr = AX25Address(dest.strip(), dest_ssid)
    src_addr = AX25Address(src.strip(), src_ssid)
    
    frame = AX25Frame(dest_addr, src_addr, pid=pid)
    return frame

def parse_ax25_frame(data: bytes) -> AX25Frame:
    """
    Convenience function to parse an AX.25 frame from raw bytes.
    
    Args:
        data: Raw frame bytes including flags
        
    Returns:
        Parsed AX25Frame instance
        
    Example:
        >>> frame = parse_ax25_frame(b'\\x7E...\\x7E')
        >>> print(f"From: {frame.src.callsign}")
    """
    if not isinstance(data, bytes):
        raise TypeError("Frame data must be bytes")
    if len(data) < 16:  # Minimum frame size
        raise ValueError("Frame data too short")
        
    return AX25Frame.from_bytes(data)

def calculate_fcs(data: bytes) -> int:
    """
    Convenience function to calculate AX.25 Frame Check Sequence.
    
    Args:
        data: Data to calculate FCS for
        
    Returns:
        16-bit FCS value
        
    Example:
        >>> fcs = calculate_fcs(b"Hello World")
        >>> print(f"FCS: 0x{fcs:04X}")
    """
    return fcs_calc(data)

def encode_kiss_frame(data: bytes, tnc_address: int = 0, 
                     command: int = 0x00) -> bytes:
    """
    Convenience function to encode data as KISS frame.
    
    Args:
        data: Frame payload
        tnc_address: TNC address (0-15)
        command: KISS command byte
        
    Returns:
        Encoded KISS frame bytes
        
    Example:
        >>> kiss_frame = encode_kiss_frame(b"Hello", tnc_address=1)
    """
    if not isinstance(data, bytes):
        raise TypeError("Data must be bytes")
    if not 0 <= tnc_address <= 15:
        raise ValueError("TNC address must be 0-15")
    if not 0 <= command <= 255:
        raise ValueError("Command must be 0-255")
        
    # This would need KISS interface import, but for core __init__.py
    # we'll provide a placeholder that indicates the functionality
    raise NotImplementedError("KISS frame encoding requires transport interface")

# Utility functions for common operations
def get_version_info() -> Dict[str, Union[str, tuple]]:
    """
    Get detailed version information about PyAX25_22.
    
    Returns:
        Dictionary with version information
        
    Example:
        >>> info = get_version_info()
        >>> print(f"Version: {info['version']}")
    """
    return {
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': __description__,
        'license': __license__,
        'copyright': __copyright__,
        'python_version': sys.version_info,
        'platform': sys.platform,
        'imports_successful': _IMPORTS_SUCCESSFUL
    }

def check_compatibility() -> Dict[str, Union[bool, str]]:
    """
    Check compatibility and return status information.
    
    Returns:
        Dictionary with compatibility status
        
    Example:
        >>> compat = check_compatibility()
        >>> if not compat['compatible']:
        ...     print("Compatibility issues detected")
    """
    issues = []
    compatible = True
    
    # Check Python version
    if sys.version_info < _REQUIRED_PYTHON_VERSION:
        issues.append(f"Python version {sys.version_info} < {_REQUIRED_PYTHON_VERSION}")
        compatible = False
    
    # Check imports
    if not _IMPORTS_SUCCESSFUL:
        issues.append("Core component imports failed")
        compatible = False
    
    # Check optional dependencies
    try:
        import serial
        serial_version = getattr(serial, '__version__', 'unknown')
    except ImportError:
        issues.append("pyserial not available (required for serial transport)")
        serial_version = "not available"
    
    try:
        import asyncio
        asyncio_available = True
    except ImportError:
        issues.append("asyncio not available (required for async operations)")
        asyncio_available = False
    
    return {
        'compatible': compatible,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'required_python': f"{_REQUIRED_PYTHON_VERSION[0]}.{_REQUIRED_PYTHON_VERSION[1]}",
        'pyserial': serial_version,
        'asyncio': asyncio_available,
        'issues': issues,
        'imports_successful': _IMPORTS_SUCCESSFUL
    }

def configure_logging(level: Union[str, int] = "INFO", 
                     format_string: Optional[str] = None) -> None:
    """
    Configure package-wide logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL or int)
        format_string: Custom format string for log messages
        
    Example:
        >>> configure_logging("DEBUG")
        >>> configure_logging(logging.INFO, "%(asctime)s - %(levelname)s - %(message)s")
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s '
            '[%(filename)s:%(lineno)d]'
        )
    
    # Set up root logger for pyax25_22 package
    logger = logging.getLogger('pyax25_22')
    logger.setLevel(getattr(logging, level.upper()) if isinstance(level, str) else level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()) if isinstance(level, str) else level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Also configure root logger if not already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.WARNING)  # Don't spam with too much info
    
    logger.info(f"PyAX25_22 logging configured: level={level}, format={format_string}")

def enable_deprecation_warnings() -> None:
    """
    Enable deprecation warnings for the package.
    
    This will show warnings for deprecated features and APIs.
    """
    warnings.filterwarnings('default', category=DeprecationWarning, module='pyax25_22')
    warnings.filterwarnings('default', category=PendingDeprecationWarning, module='pyax25_22')
    logging.getLogger('pyax25_22').info("Deprecation warnings enabled")

def disable_deprecation_warnings() -> None:
    """
    Disable deprecation warnings for the package.
    
    This will suppress warnings for deprecated features.
    """
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pyax25_22')
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning, module='pyax25_22')
    logging.getLogger('pyax25_22').info("Deprecation warnings disabled")

# Module-level convenience variables
# These provide easy access to commonly used constants and enums
FRAME_TYPES = FrameType
ADDRESS_TYPES = AX25Address
PID_TYPES = PID
STATE_TYPES = AX25State
MODULO_TYPES = AX25Modulo

# Backward compatibility warnings for common misuses
def __getattr__(name: str) -> Any:
    """
    Handle dynamic attribute access with helpful error messages.
    
    Args:
        name: Attribute name being accessed
        
    Returns:
        Attribute value if found
        
    Raises:
        AttributeError: If attribute not found with helpful message
    """
    # Common misspellings and corrections
    corrections = {
        'ax25frame': 'AX25Frame',
        'ax25_address': 'AX25Address', 
        'ax25_state_machine': 'AX25StateMachine',
        'ax25_state': 'AX25State',
        'ax25_timer': 'AX25Timer',
        'ax25_fcs': 'fcs_calc',
        'ax25_bit_stuff': 'bit_stuff',
        'ax25_bit_destuff': 'bit_destuff',
        'ax25_encode_address': 'encode_address',
        'ax25_decode_address': 'decode_address'
    }
    
    if name in corrections:
        correct_name = corrections[name]
        warnings.warn(
            f"Attribute '{name}' not found. Did you mean '{correct_name}'? "
            f"Use '{correct_name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return globals()[correct_name]
    
    # Check for common module imports that might be expected
    if name in ['interfaces', 'transport', 'utils']:
        raise AttributeError(
            f"Module '{name}' is not part of core. "
            f"Import from 'pyax25_22.{name}' instead."
        )
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Initialize logging at WARNING level by default (non-intrusive)
configure_logging("WARNING")

# Export additional utility functions
__all__.extend([
    'create_ax25_frame',
    'parse_ax25_frame', 
    'calculate_fcs',
    'encode_kiss_frame',
    'get_version_info',
    'check_compatibility',
    'configure_logging',
    'enable_deprecation_warnings',
    'disable_deprecation_warnings',
    # Convenience aliases
    'AX25',
    'Address', 
    'StateMachine',
    'Timer',
    # Common type references
    'FRAME_TYPES',
    'ADDRESS_TYPES',
    'PID_TYPES',
    'STATE_TYPES',
    'MODULO_TYPES'
])

# Package initialization message (only in debug mode)
logger = logging.getLogger(__name__)
logger.debug(f"PyAX25_22 core module initialized: version={__version__}, "
            f"Python={sys.version_info.major}.{sys.version_info.minor}")

# Clean up temporary variables
del logging, sys, warnings, List, Optional, Union, Dict, Any
