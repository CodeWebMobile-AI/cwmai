"""
AI-Generated Tool: find_async_functions
Description: Find all functions that use async/await
Generated: 2025-06-15T12:45:11.724188+00:00
Requirements: Search all Python files for async function definitions and await statements
"""

from typing import Any
from typing import Dict
from typing import Dict, Any, List
from typing import List

import asyncio
import os
import re


"""
Module: find_async_functions

Description: Find all functions that use async/await in Python files.
"""


# Assume StateManager and any other dependencies are in 'scripts' or installed packages
# from scripts.state_manager import StateManager # Removed to avoid undefined name error
# from scripts.logger import logger # Removed to avoid undefined name error

__description__ = "Find all functions that use async/await in Python files."
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "The directory to search for Python files.",
        "required": True,
    }
}
__examples__ = [
    {
        "description": "Find async functions in the current directory.",
        "input": {"directory": "."},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3},
    }
]


async def find_async_functions(**kwargs) -> Dict[str, Any]:
    """
    Finds all functions that use async/await in Python files within a specified directory.

    Args:
        **kwargs: Keyword arguments containing the 'directory' to search.

    Returns:
        A dictionary containing the counts of async functions, await statements, and files analyzed.
    """

    # state_manager = StateManager() # Removed to avoid undefined name error
    # state = state_manager.load_state() # Removed to avoid undefined name error

    directory = kwargs.get("directory")

    if not directory:
        raise ValueError("Directory must be specified.")

    if not isinstance(directory, str):
        raise TypeError("Directory must be a string.")

    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")

    async_function_count = 0
    await_statement_count = 0
    files_analyzed = 0

    async def process_file(filepath: str) -> None:
        nonlocal async_function_count, await_statement_count, files_analyzed
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

                # Find async function definitions
                async_functions = re.findall(r"async def\s+\w+\s*\(", content)
                async_function_count += len(async_functions)

                # Find await statements
                await_statements = re.findall(r"await\s+\w+\(", content)
                await_statement_count += len(await_statements)

                files_analyzed += 1

        except Exception as e:
            # logger.error(f"Error processing file {filepath}: {e}") # Removed to avoid undefined name error
            print(f"Error processing file {filepath}: {e}")

    async def find_python_files(search_directory: str) -> List[str]:
        python_files = []
        for root, _, files in os.walk(search_directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

    python_files = await find_python_files(directory)
    await asyncio.gather(*(process_file(filepath) for filepath in python_files))

    result = {
        "async_function_count": async_function_count,
        "await_statement_count": await_statement_count,
        "files_analyzed": files_analyzed,
    }

    # state_manager.update_state({"last_run": "success"}) # Removed to avoid undefined name error

    return result
