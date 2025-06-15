"""
AI-Generated Tool: count_python_files_in_directory
Description: 
CREATE_TOOL: count_python_files_in_directory

This tool should take a directory path as input and return the number of Python files (files ending with the ".py" extension) present in that directory. It would need to recursively search the directory and its subdirectories.

Generated: 2025-06-15T12:33:07.373751+00:00
Requirements: 
        Based on user request: How many Python files are in the scripts directory?
        Create a tool that: 
CREATE_TOOL: count_python_files_in_directory

This tool should take a directory path as input and return the number of Python files (files ending with the ".py" extension) present in that directory. It would need to recursively search the directory and its subdirectories.

        Tool should integrate with existing system components
        
"""

from typing import Any
from typing import Dict
from typing import Dict, Any

import asyncio
import os


"""
Module for counting Python files in a directory recursively.
"""


__description__ = "Counts the number of Python files in a directory (recursively)."
__parameters__ = {
    "directory_path": {
        "type": "string",
        "description": "The path to the directory to search.",
        "required": True,
    }
}
__examples__ = [
    {
        "command": "count_python_files_in_directory(directory_path='/path/to/scripts')",
        "description": "Counts Python files in the /path/to/scripts directory.",
    }
]


async def count_python_files_in_directory(**kwargs: Dict[str, Any]) -> Dict[str, int]:
    """
    Counts the number of Python files in a given directory recursively.

    Args:
        directory_path: The path to the directory to search.

    Returns:
        A dictionary containing the total count of Python files.
        Example: {"total": 10}
    """

    directory_path = kwargs.get("directory_path")

    if not directory_path:
        raise ValueError("directory_path is a required parameter.")

    if not isinstance(directory_path, str):
        raise TypeError("directory_path must be a string.")

    total_python_files = 0

    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".py"):
                    total_python_files += 1
    except OSError as e:
        raise OSError(f"Error accessing directory: {e}") from e

    return {"total": total_python_files}
