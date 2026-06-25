"""
utils/retry.py
Retry decorator with exponential backoff for flaky API calls.
"""

import functools
import logging
import time
from typing import Callable, Type

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator that retries a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts before giving up
        delay:        Initial delay between retries in seconds
        backoff:      Multiplier applied to delay after each retry
        exceptions:   Tuple of exception types to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{getattr(func, '__name__', repr(func))} failed after {max_attempts} attempts: {e}"
                        )
                        return None
                    logger.warning(
                        f"{getattr(func, '__name__', repr(func))} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                        )
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None
        return wrapper
    return decorator
