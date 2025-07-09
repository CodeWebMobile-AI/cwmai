"""
AI-Generated Tool: how_many_active_workers_do_we_
Description: How many active workers do we have?
Generated: 2025-06-15T12:15:46.136610+00:00
Requirements: count active worker processes by checking system processes
"""

from typing import Dict, List, Optional, Union

import asyncio
import psutil
import time
import logging

from scripts.state_manager import StateManager  # Ensure this import works correctly.  Remove if unused.

"""
Module for counting active worker processes.
"""

# Configure logging (important for error handling and debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


__description__ = "How many active workers do we have?"
__parameters__ = {}
__examples__ = [
    {"command": "how_many_active_workers_do_we_", "description": "Counts the number of active worker processes."}
]

# Global cache for active worker count and timestamp.  Use carefully with long-running processes.
_active_workers_cache: Optional[Dict] = None
_cache_timestamp: Optional[float] = None
_cache_expiration_seconds: int = 60  # Cache expires after 60 seconds

# Add a global lock to prevent race conditions when updating the cache.
_cache_lock = asyncio.Lock()


def _is_worker_process(process: psutil.Process, worker_process_names: List[str]) -> bool:
    """
    Helper function to determine if a process is a worker process.

    Args:
        process: The psutil.Process object.
        worker_process_names: A list of worker process names to check against.

    Returns:
        True if the process is a worker process, False otherwise.
    """
    try:
        # Check process name
        if process.info['name'] in worker_process_names:
            return True
        # Also check command line for python scripts
        elif process.info['cmdline'] and len(process.info['cmdline']) > 1:
            for cmd_part in process.info['cmdline']:
                if any(worker_name in str(cmd_part) for worker_name in worker_process_names):
                    return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        # Ignore processes that have terminated or we don't have access to
        pass
    except Exception as e:
        logging.warning(f"Unexpected error while checking process {process.info.get('pid', 'Unknown PID')}: {e}")
    return False


async def _get_active_worker_count(worker_process_names: List[str]) -> int:
    """
    Counts active worker processes by checking system processes.

    Args:
        worker_process_names: A list of worker process names to check against.

    Returns:
        The number of active worker processes.
    """
    active_workers = 0
    process_errors = 0 # Track errors to avoid infinite loops due to persistent issues.
    max_process_errors = 5 # Limit number of errors before giving up.
    for process in psutil.process_iter(['name', 'cmdline']):
        try:
            if _is_worker_process(process, worker_process_names):
                active_workers += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Ignore processes that have terminated or we don't have access to
            pass
        except Exception as e:
            process_errors += 1
            logging.error(f"Error processing process {process.info.get('pid', 'Unknown PID')}: {e}")
            if process_errors > max_process_errors:
                logging.error("Too many errors processing processes. Aborting worker count.")
                raise RuntimeError("Too many errors during worker process enumeration.")

    return active_workers


async def _clear_cache():
    """
    Clears the cache by setting it to None.
    """
    global _active_workers_cache, _cache_timestamp
    async with _cache_lock:  # Use lock to prevent race conditions
        _active_workers_cache = None
        _cache_timestamp = None
    logging.info("Active worker count cache cleared.")


async def how_many_active_workers_do_we_(clear_cache: bool = False, **kwargs) -> Dict[str, Union[int, str]]:
    """
    Counts active worker processes by checking system processes, using caching for performance.

    Args:
        clear_cache:  If True, clears the existing cache before counting.  Useful for forcing a refresh.
        **kwargs:  Optional keyword arguments (currently unused, but included for potential future expansion).

    Returns:
        Dict: A dictionary containing the total number of active worker processes and optionally a timestamp.
              Example: {"active_workers": 5, "timestamp": "2024-10-27T10:00:00"}

    Raises:
        RuntimeError: If there are too many errors during process enumeration.
    """
    global _active_workers_cache, _cache_timestamp

    # Optional: Add argument parsing/validation for kwargs if expecting specific inputs.
    # For example, to allow the user to specify the worker process names:
    # worker_process_names = kwargs.get("worker_process_names", ["worker_process", "worker.py"])


    state_manager = StateManager()  # Instantiate state manager (even if not used).  Remove if unused.


    worker_process_names = ["worker_process", "worker.py", "ai_worker_agent.py"]  # Define worker process names. Adapt to the real names.

    try:

        if clear_cache:
            await _clear_cache()

        # Check if cache is valid
        async with _cache_lock:  # Acquire lock before accessing cache
            if _active_workers_cache and _cache_timestamp and (time.time() - _cache_timestamp < _cache_expiration_seconds):
                logging.info("Using cached active worker count.")
                return _active_workers_cache

        # Get the active worker count
        active_workers = await _get_active_worker_count(worker_process_names)

        result = {"active_workers": active_workers, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

        async with _cache_lock: # Acquire lock before updating cache
            _active_workers_cache = result  # Update cache
            _cache_timestamp = time.time()  # Update timestamp

        logging.info(f"Active worker count: {active_workers}")
        return result

    except Exception as e:
        error_message = f"Error counting active workers: {e}"
        logging.exception(error_message)  # Log the full exception, including traceback
        await _clear_cache() # In case of a major error, invalidate the cache.
        raise RuntimeError(error_message) from e #Re-raise the exception.