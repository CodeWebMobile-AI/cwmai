"""
AI-Generated Tool: count_how_many_python
Description: Count the number of Python files with more than 100 lines of code in a directory.
Generated: 2024-11-03
Requirements: Count Python files with > 100 lines in a directory.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import json
import asyncio

__description__ = "Count Python files with more than 100 lines of code in a directory."
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "Directory to search for Python files.  Defaults to current directory.",
        "required": False,
        "default": "."
    },
    "recursive": {
        "type": "boolean",
        "description": "Whether to recursively search subdirectories. Defaults to False.",
        "required": False,
        "default": False
    },
    "min_lines": {
        "type": "integer",
        "description": "Minimum number of lines of code a Python file must have to be counted. Defaults to 100.",
        "required": False,
        "default": 100
    },
    "include_filenames": {
        "type": "boolean",
        "description": "Whether to include a list of filenames that meet the criteria in the output. Defaults to False.",
        "required": False,
        "default": False
    },
    "exclude_patterns": {
        "type": "string",
        "description": "Comma-separated list of filename patterns to exclude (e.g., test*.py, *_test.py).",
        "required": False,
        "default": ""
    },
    "max_files": {
        "type": "integer",
        "description": "Maximum number of files to process.  Useful for very large directories. Defaults to no limit.",
        "required": False,
        "default": 0  # 0 means no limit
    }
}
__examples__ = [
    {"description": "Count Python files in current directory", "code": "await count_how_many_python()"},
    {"description": "Count Python files in 'scripts' directory", "code": "await count_how_many_python(directory='scripts')"},
    {"description": "Recursively count Python files in current directory", "code": "await count_how_many_python(recursive=True)"},
    {"description": "Recursively count Python files in 'scripts' directory", "code": "await count_how_many_python(directory='scripts', recursive=True)"},
    {"description": "Count Python files with at least 50 lines of code", "code": "await count_how_many_python(min_lines=50)"},
    {"description": "Count Python files and include filenames", "code": "await count_how_many_python(include_filenames=True)"},
    {"description": "Exclude test files", "code": "await count_how_many_python(exclude_patterns='test*.py,*_test.py')"},
    {"description": "Limit the number of files processed", "code": "await count_how_many_python(max_files=100)"}
]

_CACHE: Dict[str, Dict[str, Any]] = {}


def _generate_cache_key(directory: str, recursive: bool, min_lines: int, exclude_patterns: str, max_files: int) -> str:
    """Generates a cache key based on the input parameters."""
    cache_data = {"directory": directory, "recursive": recursive, "min_lines": min_lines, "exclude_patterns": exclude_patterns, "max_files": max_files}
    cache_key = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode('utf-8')).hexdigest()
    return cache_key


def _matches_exclude_pattern(filename: str, exclude_patterns: List[str]) -> bool:
    """Checks if a filename matches any of the exclude patterns."""
    for pattern in exclude_patterns:
        if pattern.replace("*", "") in filename: #SImple contains check, could use regex for advanced pattern matching
            return True
    return False


async def count_how_many_python(**kwargs) -> Dict[str, Any]:
    """Count Python files with more than a specified number of lines of code."""
    directory: str = kwargs.get('directory', '.')
    recursive: bool = kwargs.get('recursive', False)
    min_lines: int = kwargs.get('min_lines', 100)
    include_filenames: bool = kwargs.get('include_filenames', False)
    exclude_patterns_str: str = kwargs.get('exclude_patterns', '')
    max_files: int = kwargs.get('max_files', 0)  # 0 means no limit

    # Input validation
    if not isinstance(directory, str):
        return {"error": "Directory must be a string."}
    if not isinstance(recursive, bool):
        return {"error": "Recursive must be a boolean."}
    if not isinstance(min_lines, int):
        return {"error": "min_lines must be an integer."}
    if min_lines <= 0:
        return {"error": "min_lines must be a positive integer."}
    if not isinstance(include_filenames, bool):
        return {"error": "include_filenames must be a boolean."}
    if not isinstance(exclude_patterns_str, str):
        return {"error": "exclude_patterns must be a string."}
    if not isinstance(max_files, int):
        return {"error": "max_files must be an integer."}
    if max_files < 0:
        return {"error": "max_files must be a non-negative integer."}

    exclude_patterns: List[str] = [p.strip() for p in exclude_patterns_str.split(',') if p.strip()]

    cache_key = _generate_cache_key(directory, recursive, min_lines, exclude_patterns_str, max_files)

    if cache_key in _CACHE:
        print("Returning cached result.")
        return _CACHE[cache_key]

    try:
        path = Path(directory)
        if not path.exists():
            return {"error": f"Directory not found: {directory}"}

        if not path.is_dir():
            return {"error": f"Not a directory: {directory}"}

        python_file_count: int = 0
        matching_filenames: List[str] = []
        files_processed: int = 0

        async def process_file(item: Path):
            nonlocal python_file_count, matching_filenames, files_processed

            if max_files > 0 and files_processed >= max_files:
                return  # Stop processing if limit is reached

            if _matches_exclude_pattern(item.name, exclude_patterns):
                return

            try:
                with open(item, 'r', encoding='utf-8', errors='ignore') as f:  # Ignore decoding errors
                    lines = f.readlines()
                    if len(lines) > min_lines:
                        python_file_count += 1
                        if include_filenames:
                            matching_filenames.append(str(item.absolute()))
            except FileNotFoundError:
                print(f"Warning: File not found {item}.  It may have been deleted.")
            except Exception as e:
                print(f"Warning: Error reading file {item}: {str(e)}") #Non-critical error, so just print a warning
            finally:
                files_processed += 1  # Increment even if there was an error

        async def process_directory(search_path: Path, recurse: bool):
            tasks = []
            if recurse:
                for item in search_path.rglob("*.py"):
                    if item.is_file():
                        tasks.append(asyncio.create_task(process_file(item)))
            else:
                for item in search_path.glob("*.py"):
                    if item.is_file():
                        tasks.append(asyncio.create_task(process_file(item)))
            await asyncio.gather(*tasks)


        await process_directory(path, recursive)

        result = {
            "directory": str(path.absolute()),
            "recursive": recursive,
            "min_lines": min_lines,
            "python_file_count": python_file_count,
            "summary": f"Found {python_file_count} Python files with more than {min_lines} lines of code.",
            "files_processed": files_processed,
        }

        if include_filenames:
            result["filenames"] = matching_filenames
        if exclude_patterns:
            result["exclude_patterns": exclude_patterns] = exclude_patterns

        _CACHE[cache_key] = result  # Cache the result
        return result

    except Exception as e:
        return {"error": f"Error counting Python files: {str(e)}"}