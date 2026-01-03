# pyax25_22/utils/async_thread.py
"""
Asynchronous Thread Utilities

Provides:
- run_in_thread: Execute blocking calls in thread pool without blocking event loop

License: LGPLv3.0
Copyright (C) 2025-2026 Kris Kirby, KE4AHR
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Global thread pool for blocking operations
_DEFAULT_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix='AsyncThreadPool'
)

async def run_in_thread(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Run blocking function in thread pool executor.
    
    Args:
        func: Blocking callable to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of func(*args, **kwargs)
        
    Example:
        async def main():
            result = await run_in_thread(blocking_io_function, arg1, arg2)
    """
    loop = asyncio.get_running_loop()
    wrapped = partial(func, *args, **kwargs)
    logger.debug(f"Executing {func.__name__} in thread pool")
    try:
        result = await loop.run_in_executor(_DEFAULT_EXECUTOR, wrapped)
    except Exception as e:
        logger.error(f"Thread execution failed: {e}")
        raise
    logger.debug(f"Completed {func.__name__} in thread pool")
    return result

def get_executor() -> ThreadPoolExecutor:
    """Get the global thread pool executor"""
    return _DEFAULT_EXECUTOR

def set_executor(executor: ThreadPoolExecutor) -> None:
    """Replace the global thread pool executor"""
    global _DEFAULT_EXECUTOR
    _DEFAULT_EXECUTOR.shutdown(wait=True)
    _DEFAULT_EXECUTOR = executor

async def __close_executor() -> None:
    """Cleanup executor on shutdown"""
    _DEFAULT_EXECUTOR.shutdown(wait=False)

try:
    from atexit import register
    register(lambda: asyncio.run(__close_executor()))
except ImportError:
    pass

if __name__ == "__main__":
    # Example usage
    import time
    
    logging.basicConfig(level=logging.DEBUG)
    
    def blocking_task(duration: float) -> float:
        time.sleep(duration)
        return duration
    
    async def main():
        print("Running blocking task for 1 second...")
        result = await run_in_thread(blocking_task, 1.0)
        print(f"Got result: {result}")
        
    asyncio.run(main())
