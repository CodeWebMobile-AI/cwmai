#!/usr/bin/env python3
"""Safe disk I/O operations wrapper."""

import functools
import time
import logging
import sqlite3

logger = logging.getLogger(__name__)

def safe_disk_operation(max_retries=3, delay=1.0):
    """Decorator for safe disk operations with retry logic."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (IOError, OSError, sqlite3.OperationalError) as e:
                    last_error = e
                    if "disk I/O error" in str(e) or "database is locked" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(f"Disk I/O error in {func.__name__}, attempt {attempt + 1}/{max_retries}: {e}")
                            time.sleep(delay * (attempt + 1))  # Exponential backoff
                            continue
                    raise
            
            # If we get here, all retries failed
            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_error
        
        return wrapper
    return decorator

# Example usage:
# @safe_disk_operation(max_retries=5)
# def save_state(self):
#     with open('state.json', 'w') as f:
#         json.dump(self.state, f)
