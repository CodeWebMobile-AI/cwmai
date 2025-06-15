"""
AI-Generated Tool: count_python_files
Description: Count all Python files in the project
Generated: 2025-06-15T10:59:39.894214+00:00
Requirements: 
        1. Count all .py files in the project
        2. Group by directory
        3. Show total lines of code
        4. Return structured data with counts and statistics
        
"""

"""
Module: count_python_files

Description: Counts all Python files in the project, groups by directory,
and shows the total lines of code. Returns structured data with counts and statistics.
"""

import asyncio
import os
import glob
from typing import Dict, Any
from scripts.state_manager import StateManager  # Assuming this exists
from scripts.file_manager import FileManager  # Assuming this exists
from scripts.error_handler import ErrorHandler  # Assuming this exists

__description__ = "Count all Python files in the project"
__parameters__ = {}
__examples__ = [
    "count_python_files"
]


async def count_python_files(**kwargs) -> Dict[str, Any]:
    """
    Counts all Python files in the project, groups by directory, and shows the total lines of code.

    Returns:
        A dictionary containing the counts and statistics.  Example:
        {
            "total_files": 5,
            "total_lines": 1234,
            "directory_summary": {
                "path/to/dir1": {"file_count": 2, "line_count": 500},
                "path/to/dir2": {"file_count": 3, "line_count": 734}
            }
        }
    """
    state_manager = StateManager()  # Assuming this class is defined
    file_manager = FileManager()  # Assuming this class is defined
    error_handler = ErrorHandler()  # Assuming this class is defined

    state = state_manager.load_state()

    try:
        directory_summary: Dict[str, Dict[str, int]] = {}
        total_files = 0
        total_lines = 0

        for root, _, files in os.walk("."):  # Start from the current directory
            py_files = [f for f in files if f.endswith(".py")]
            if py_files:
                directory_summary[root] = {"file_count": 0, "line_count": 0}
                for file in py_files:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            line_count = len(lines)

                            directory_summary[root]["file_count"] += 1
                            directory_summary[root]["line_count"] += line_count
                            total_files += 1
                            total_lines += line_count
                    except Exception as e:
                        error_handler.log_error(f"Error reading file {filepath}: {e}")
                        continue  # Skip to the next file in case of error
        
        result = {
            "total_files": total_files,
            "total_lines": total_lines,
            "directory_summary": directory_summary,
        }

        return result

    except Exception as e:
        error_handler.log_error(f"An unexpected error occurred: {e}")
        return {"error": str(e)}
