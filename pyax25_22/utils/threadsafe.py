# pyax25_22/utils/threadsafe.py
"""
Thread-safe Data Structures

Provides:
- AtomicCounter: Thread-safe integer counter
- SharedState: Thread-safe key-value store

License: LGPLv3.0
Copyright (C) 2024 Kris Kirby, KE4AHR
"""

import threading
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class AtomicCounter:
    """
    Thread-safe atomic counter with increment/decrement operations.
    
    Args:
        initial: Initial value (default 0)
    """
    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.RLock()
        
    def increment(self, amount: int = 1) -> int:
        """
        Increment counter and return new value.
        
        Args:
            amount: Value to add (default 1)
        """
        with self._lock:
            self._value += amount
            logger.debug(f"Counter incremented by {amount}, now {self._value}")
            return self._value
            
    def decrement(self, amount: int = 1) -> int:
        """
        Decrement counter and return new value.
        
        Args:
            amount: Value to subtract (default 1)
        """
        with self._lock:
            self._value -= amount
            logger.debug(f"Counter decremented by {amount}, now {self._value}")
            return self._value
            
    def set(self, value: int) -> None:
        """Set counter to specific value."""
        with self._lock:
            self._value = value
            logger.debug(f"Counter set to {value}")
            
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
            
    def __repr__(self) -> str:
        return f"AtomicCounter(value={self.get()})"

class SharedState:
    """
    Thread-safe key-value store with lock protection.
    
    Example:
        state = SharedState()
        state.set('key', 'value')
        print(state.get('key'))
    """
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
    def set(self, key: str, value: Any) -> None:
        """Set key to value."""
        with self._lock:
            self._data[key] = value
            logger.debug(f"Set {key}={repr(value)}")
            
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get value for key, returning default if not found.
        
        Args:
            key: Key to retrieve
            default: Value if key missing (default None)
        """
        with self._lock:
            logger.debug(f"Getting {key} (exists: {key in self._data})")
            return self._data.get(key, default)
            
    def update(self, key: str, updater: Callable[[Any], Any]) -> Any:
        """
        Update key's value using a function.
        
        Args:
            key: Key to update
            updater: Function(old_value) -> new_value
            
        Returns:
            The new value
            
        Raises:
            KeyError if key doesn't exist
        """
        with self._lock:
            if key not in self._data:
                raise KeyError(f"Key {key} not found")
            self._data[key] = updater(self._data[key])
            logger.debug(f"Updated {key} via function")
            return self._data[key]
            
    def delete(self, key: str) -> None:
        """Delete key from state."""
        with self._lock:
            try:
                del self._data[key]
                logger.debug(f"Deleted {key}")
            except KeyError:
                pass
                
    def contains(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            return key in self._data
            
    def __repr__(self) -> str:
        with self._lock:
            keys = list(self._data.keys())
            return f"SharedState(keys={keys})"

if __name__ == "__main__":
    # Example usage
    import time
    from concurrent.futures import ThreadPoolExecutor
    
    logging.basicConfig(level=logging.DEBUG)
    
    counter = AtomicCounter()
    state = SharedState()
    state.set('test', 0)
    
    def worker():
        for _ in range(1000):
            counter.increment()
            state.update('test', lambda x: x + 1)
            
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(10)]
        
    print(f"Final counter: {counter.get()}")
    print(f"Final test value: {state.get('test')}")
