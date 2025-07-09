"""
AI-Generated Tool: list_files
Description: List items for files
Generated: 2025-06-22T22:47:12.718848+00:00
Requirements: 
        Tool Name: list_files
        Intent: List items for files
        Expected Parameters: {"path": "'.'", "include_directories": "True", "pattern": "'.git'"}
        Category: query
        
        The tool should:
        1. List all items with relevant details
2. Support filtering and pagination
3. Return structured data
        
"""

"""
Module for listing files with filtering and pagination.
"""

import asyncio
import os
import re
from typing import Dict, Any, List

from scripts.state_manager import StateManager


__description__ = "List items for files."
__parameters__ = {
    "path": "'.'",
    "include_directories": "True",
    "pattern": "'.git'"
}
__examples__ = [
    "list_files path='.' include_directories='True' pattern='*.py'",
    "list_files path='/home/user/documents' include_directories='False' pattern='*.txt'",
    "list_files pattern='README.md'"
]


async def list_files(**kwargs) -> Dict[str, Any]:
    """
    Lists files and directories based on provided parameters.

    Args:
        **kwargs: A dictionary containing the following optional parameters:
            - path (str): The path to list files from. Defaults to '.'.
            - include_directories (str): Whether to include directories in the listing. Defaults to 'True'.
            - pattern (str): A pattern to filter files by. Defaults to "'.git'".

    Returns:
        A dictionary containing the list of files and directories, along with summary information.
    """
    state_manager = StateManager()
    state = state_manager.get_state()

    path = kwargs.get('path', '.')
    include_directories_str = kwargs.get('include_directories', 'True')
    pattern = kwargs.get('pattern', "'.git'")

    include_directories = include_directories_str.lower() == 'true'

    try:
        items: List[Dict[str, str]] = []
        total_files = 0
        total_directories = 0

        for item in os.listdir(path):
            full_path = os.path.join(path, item)

            if re.search(pattern, item):
                if os.path.isfile(full_path):
                    items.append({
                        "name": item,
                        "type": "file",
                        "size": str(os.path.getsize(full_path)),
                        "path": full_path
                    })
                    total_files += 1
                elif os.path.isdir(full_path) and include_directories:
                    items.append({
                        "name": item,
                        "type": "directory",
                        "path": full_path
                    })
                    total_directories += 1
        result = {
                "items": items,
                "total_files": total_files,
                "total_directories": total_directories,
                "path": path,
                "pattern": pattern,
                "include_directories": include_directories

            }
        return result
    except FileNotFoundError as e:
        return {"error": f"Path not found: {path}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
