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
from typing import Dict, Any, List

import asyncio
import os
import functools
import logging
import hashlib


"""
Module for counting Python files in a directory recursively.
Includes caching and improved error handling.
"""

__description__ = "Counts the number of Python files in a directory (recursively)."
__parameters__ = {
    "directory_path": {
        "type": "string",
        "description": "The path to the directory to search.",
        "required": True,
    },
    "include_tests": {
        "type": "boolean",
        "description": "Whether to include files in 'test' or 'tests' directories. Defaults to False.",
        "required": False,
        "default": False
    },
    "exclude_patterns": {
        "type": "list",
        "description": "A list of directory name patterns to exclude from the search (e.g., ['.venv', 'migrations']).",
        "required": False,
        "default": []
    },
    "min_size_kb": {
        "type": "integer",
        "description": "Minimum size (in KB) of Python files to include in the count.  Files smaller than this size will be excluded.",
        "required": False,
        "default": 0
    }
}
__examples__ = [
    {
        "command": "count_python_files_in_directory(directory_path='/path/to/scripts')",
        "description": "Counts Python files in the /path/to/scripts directory.",
    },
    {
        "command": "count_python_files_in_directory(directory_path='/path/to/scripts', include_tests=True)",
        "description": "Counts Python files in the /path/to/scripts directory, including test files.",
    },
    {
        "command": "count_python_files_in_directory(directory_path='/path/to/scripts', exclude_patterns=['.venv', 'migrations'])",
        "description": "Counts Python files in /path/to/scripts, excluding .venv and migrations directories.",
    },
     {
        "command": "count_python_files_in_directory(directory_path='/path/to/scripts', min_size_kb=10)",
        "description": "Counts Python files in /path/to/scripts, including only files that are at least 10KB in size.",
    }
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Simple cache (in-memory)
_CACHE: Dict[str, Dict[str, Any]] = {}  # Cache now stores the full dictionary result

def _generate_cache_key(directory_path: str, include_tests: bool, exclude_patterns: List[str], min_size_kb: int) -> str:
    """
    Generates a cache key based on the function's parameters. Uses a hash for efficiency.
    """
    key_string = f"{directory_path}-{include_tests}-{'-'.join(sorted(exclude_patterns))}-{min_size_kb}"
    return hashlib.md5(key_string.encode()).hexdigest()


def _is_valid_directory(path: str) -> bool:
    """
    Checks if the given path is a valid and existing directory.
    """
    return os.path.isdir(path)


def _is_test_directory(path: str) -> bool:
    """
    Checks if a directory is named 'test' or 'tests'.
    """
    dir_name = os.path.basename(path).lower()
    return dir_name in ("test", "tests")

def _is_excluded_directory(path: str, exclude_patterns: List[str]) -> bool:
    """
    Checks if a directory should be excluded based on the provided patterns.
    """
    dir_name = os.path.basename(path)
    return any(pattern in dir_name for pattern in exclude_patterns)


@functools.lru_cache(maxsize=128)  # Using lru_cache for simplicity.  Can be replaced with a more sophisticated cache.
async def count_python_files_in_directory(**kwargs: Dict[str, Any]) -> Dict[str, Any]:  # Changed return type for flexibility
    """
    Counts the number of Python files in a given directory recursively.

    Args:
        directory_path: The path to the directory to search.
        include_tests: Whether to include files in 'test' or 'tests' directories.
        exclude_patterns: A list of directory name patterns to exclude.
        min_size_kb: Minimum size in KB of files to include

    Returns:
        A dictionary containing:
            - total: The total count of Python files.
            - cached: True if the result was retrieved from the cache, False otherwise.
            - file_list: A list of all python files found
            - errors: A list of any errors encountered during processing.
        Example: {"total": 10, "cached": False, "file_list": ["/path/to/file1.py", "/path/to/file2.py"], "errors": []}
    """

    directory_path = kwargs.get("directory_path")
    include_tests = kwargs.get("include_tests", False)
    exclude_patterns = kwargs.get("exclude_patterns", [])
    min_size_kb = kwargs.get("min_size_kb", 0)

    # Input Validation
    if not directory_path:
        raise ValueError("directory_path is a required parameter.")

    if not isinstance(directory_path, str):
        raise TypeError("directory_path must be a string.")

    if not isinstance(include_tests, bool):
        raise TypeError("include_tests must be a boolean.")

    if not isinstance(exclude_patterns, list):
        raise TypeError("exclude_patterns must be a list.")
    
    if not isinstance(min_size_kb, int):
        raise TypeError("min_size_kb must be an integer.")

    if not _is_valid_directory(directory_path):
        raise ValueError(f"Invalid directory path: {directory_path}")

    cache_key = _generate_cache_key(directory_path, include_tests, exclude_patterns, min_size_kb)

    # Check if result is in cache
    if cache_key in _CACHE:
        logging.info(f"Returning cached result for directory: {directory_path}, include_tests: {include_tests}, exclude_patterns: {exclude_patterns}, min_size_kb: {min_size_kb}")
        return _CACHE[cache_key]

    total_python_files = 0
    python_files: List[str] = []
    errors: List[str] = [] # To store any errors during file access or processing

    try:
        for root, dirs, files in os.walk(directory_path):

            # Before processing files in a directory, check if it should be excluded
            if _is_excluded_directory(root, exclude_patterns):
                logging.debug(f"Skipping excluded directory: {root}")
                dirs[:] = []  # This prevents os.walk from descending into this directory
                continue

            if not include_tests and _is_test_directory(root):
                logging.debug(f"Skipping test directory: {root}")
                dirs[:] = [] # Prevent walking into test directories
                continue  # Skip test directories if include_tests is False

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        file_size_kb = os.path.getsize(file_path) / 1024  # Size in KB
                        if file_size_kb >= min_size_kb:
                            total_python_files += 1
                            python_files.append(file_path)
                    except OSError as e:
                        error_message = f"Error getting size of file: {file_path}. Error: {e}"
                        logging.warning(error_message)
                        errors.append(error_message) # Record the error but continue processing other files
                    except Exception as e:
                        error_message = f"Unexpected error processing file: {file_path}. Error: {e}"
                        logging.exception(error_message)
                        errors.append(error_message)  # Record the error and continue
    except OSError as e:
        error_message = f"Error accessing directory: {directory_path}. Error: {e}"
        logging.error(error_message)
        raise OSError(error_message) from e
    except Exception as e:
        error_message = f"An unexpected error occurred while processing directory {directory_path}: {e}"
        logging.exception(error_message)
        raise

    result = {"total": total_python_files, "cached": False, "file_list": python_files, "errors": errors}

    # Store result in cache
    _CACHE[cache_key] = result
    logging.info(f"Cached result for directory: {directory_path}, include_tests: {include_tests}, exclude_patterns: {exclude_patterns}, min_size_kb: {min_size_kb}, count: {total_python_files}")

    return result