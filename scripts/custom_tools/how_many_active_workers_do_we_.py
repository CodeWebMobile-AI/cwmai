"""
AI-Generated Tool: how_many_active_workers_do_we_
Description: How many active workers do we have?
Generated: 2025-06-15T12:15:46.136610+00:00
Requirements: count active worker processes by checking system processes
"""

from typing import Dict

import asyncio
import psutil

from scripts.state_manager import StateManager


"""
Module for counting active worker processes.
"""




__description__ = "How many active workers do we have?"
__parameters__ = {}
__examples__ = [
    {"command": "how_many_active_workers_do_we_", "description": "Counts the number of active worker processes."}
]


async def how_many_active_workers_do_we_(**kwargs) -> Dict:
    """
    Counts active worker processes by checking system processes.

    Returns:
        Dict: A dictionary containing the total number of active worker processes.
              Example: {"active_workers": 5}
    """
    state_manager = StateManager()

    try:
        # Load state (if needed for your specific logic, otherwise remove)
        state = state_manager.load_state()

        worker_process_names = ["worker_process", "worker.py", "ai_worker_agent.py"]  # Define worker process names. Adapt to the real names.

        active_workers = 0
        for process in psutil.process_iter(['name', 'cmdline']):
            try:
                # Check process name
                if process.info['name'] in worker_process_names:
                    active_workers += 1
                # Also check command line for python scripts
                elif process.info['cmdline'] and len(process.info['cmdline']) > 1:
                    for cmd_part in process.info['cmdline']:
                        if any(worker_name in str(cmd_part) for worker_name in worker_process_names):
                            active_workers += 1
                            break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Ignore processes that have terminated or we don't have access to
                pass

        result = {"active_workers": active_workers}
        return result

    except Exception as e:
        error_message = f"Error counting active workers: {e}"
        # Log error without using non-existent method
        print(f"[ERROR] {error_message}")
        raise RuntimeError(error_message) from e
