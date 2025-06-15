"""
AI-Generated Tool: show_me_files_changed_today
Description: Show me files changed today
Generated: 2025-06-15T12:16:19.345744+00:00
Requirements: list all files modified in the last 24 hours
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from typing import Dict, List
from typing import List

import datetime
import os
import time

from scripts.state_manager import StateManager


"""File Change Tracker Module

This module provides functionality to list all files modified in the last 24 hours.
It scans the current directory and all subdirectories recursively to find files
with modification timestamps within the last day.
"""


__description__ = "Show me files changed today"
__parameters__ = {
    "directory": {
        "type": "str",
        "required": False,
        "default": ".",
        "description": "Directory path to search (default: current directory)"
    },
    "recursive": {
        "type": "bool",
        "required": False,
        "default": True,
        "description": "Whether to search subdirectories recursively"
    }
}
__examples__ = [
    {"description": "Find changed files in current directory",
     "code": "await show_me_files_changed_today()"},
    {"description": "Find changed files in specific directory non-recursively",
     "code": "await show_me_files_changed_today(directory='/path/to/dir', recursive=False)"}
]

async def show_me_files_changed_today(**kwargs) -> Dict[str, List[Dict[str, str]]]:
    """List all files modified in the last 24 hours.
    
    Args:
        directory (str, optional): Directory to search. Defaults to current directory.
        recursive (bool, optional): Search subdirectories. Defaults to True.
    
    Returns:
        Dict: Dictionary containing list of files with their paths and modification times
              Example: {"files": [{"path": "/path/to/file", "modified": "2023-01-01 12:00:00"}]}
    
    Raises:
        ValueError: If the specified directory doesn't exist
    """
    state_manager = StateManager()
    state = state_manager.load_state()
    
    # Get parameters with defaults
    directory = kwargs.get('directory', '.')
    recursive = kwargs.get('recursive', True)
    
    # Validate directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Calculate cutoff time (24 hours ago)
    cutoff_time = time.time() - 24 * 60 * 60
    
    # Prepare results container
    changed_files = []
    
    # Walk through directory
    walk = os.walk(directory) if recursive else [next(os.walk(directory))]
    
    for root, _, files in walk:
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                mod_time = os.path.getmtime(filepath)
                if mod_time >= cutoff_time:
                    changed_files.append({
                        "path": filepath,
                        "modified": datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                    })
            except (OSError, PermissionError):
                continue  # Skip files we can't access
    
    return {"files": changed_files, "count": len(changed_files)}
