"""
AI-Generated Tool: whats_the_disk_usage_of_log_fi
Description: What's the disk usage of log files?
Generated: 2025-06-15T12:15:50.612552+00:00
Requirements: calculate total disk space used by all .log files
"""

import os

from scripts.state_manager import StateManager


"""
This module provides an asynchronous function to calculate the total disk space
used by all .log files in a specified directory.

Tool Name: whats_the_disk_usage_of_log_fi
Description: What's the disk usage of log files?
Requirements: Calculate total disk space used by all .log files.
"""


__description__ = "Calculate the total disk space used by all .log files in a directory."
__parameters__ = "None"
__examples__ = "await whats_the_disk_usage_of_log_fi(path='/var/log')"

async def whats_the_disk_usage_of_log_fi(**kwargs):
    """
    Calculate the total disk space used by all .log files in a specified directory.

    Parameters:
    kwargs (dict): Keyword arguments, expecting 'path' to specify the directory.

    Returns:
    dict: A dictionary containing the total disk space used by .log files.
    """
    state_manager = StateManager()
    state = state_manager.load_state()

    # Validate and extract the path parameter
    path = kwargs.get('path', '.')
    if not os.path.isdir(path):
        return {"error": "Invalid directory path provided."}

    total_size = 0
    file_count = 0

    try:
        # Traverse the directory tree
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.log'):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
    except Exception as e:
        return {"error": str(e)}

    return {"total_size": total_size, "file_count": file_count, "summary": f"Found {file_count} log files using {total_size} bytes"}
