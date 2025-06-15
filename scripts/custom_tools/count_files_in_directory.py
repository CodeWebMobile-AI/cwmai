"""
AI-Generated Tool: count_files_in_directory
Description: Count items for files in directory
Generated: 2025-06-15T12:22:34.615491+00:00
Requirements: 
        Tool Name: count_files_in_directory
        Intent: Count items for files in directory
        Expected Parameters: {}
        Category: analytics
        
        The tool should:
        1. Count all items in the system
2. Return total count with breakdown by status/type
3. Include summary statistics
        
"""

from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Dict, Any

import os

from scripts.state_manager import StateManager


"""
Directory File Counter Tool

This module provides functionality to count files in a directory and return
statistics about the items found, including total count and breakdown by type.
"""


__description__ = "Count items for files in directory"
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "Directory path to count files in (defaults to current directory)",
        "required": False
    }
}
__examples__ = [
    {"description": "Count files in current directory", "code": "await count_files_in_directory()"},
    {"description": "Count files in specific directory", "code": "await count_files_in_directory(directory='/path/to/dir')"}
]

async def count_files_in_directory(**kwargs) -> Dict[str, Any]:
    """
    Count all items in a directory and return statistics.

    Returns:
        Dictionary containing:
        - total: Total count of all items
        - files: Count of regular files
        - directories: Count of directories
        - symlinks: Count of symbolic links
        - other: Count of other item types
        - summary: Human-readable summary string
        - statistics: Dictionary with size statistics
    """
    try:
        state_manager = StateManager()
        state = state_manager.load_state()
        
        # Get directory from kwargs first, then state, then default to current directory
        directory = kwargs.get('directory')
        if not directory:
            directory = state.get('current_directory', os.getcwd())
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        counts = defaultdict(int)
        size_stats = {
            'total_size': 0,
            'largest_file': {'name': '', 'size': 0},
            'smallest_file': {'name': '', 'size': float('inf')},
            'average_size': 0
        }
        file_count = 0
        
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    counts['files'] += 1
                    file_size = entry.stat().st_size
                    size_stats['total_size'] += file_size
                    
                    # Track largest file
                    if file_size > size_stats['largest_file']['size']:
                        size_stats['largest_file'] = {'name': entry.name, 'size': file_size}
                    
                    # Track smallest file
                    if file_size < size_stats['smallest_file']['size']:
                        size_stats['smallest_file'] = {'name': entry.name, 'size': file_size}
                    
                    file_count += 1
                elif entry.is_dir():
                    counts['directories'] += 1
                elif entry.is_symlink():
                    counts['symlinks'] += 1
                else:
                    counts['other'] += 1
        
        # Calculate average if there are files
        if file_count > 0:
            size_stats['average_size'] = size_stats['total_size'] / file_count
        else:
            size_stats['smallest_file'] = {'name': 'N/A', 'size': 0}
        
        total_items = sum(counts.values())
        summary = (
            f"Found {total_items} items: "
            f"{counts['files']} files, "
            f"{counts['directories']} directories, "
            f"{counts['symlinks']} symlinks, "
            f"{counts['other']} other items"
        )
        
        return {
            'total': total_items,
            'files': counts['files'],
            'directories': counts['directories'],
            'symlinks': counts['symlinks'],
            'other': counts['other'],
            'summary': summary,
            'statistics': size_stats,
            'directory': directory
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'total': 0,
            'files': 0,
            'directories': 0,
            'symlinks': 0,
            'other': 0,
            'summary': f"Error counting files: {str(e)}",
            'statistics': {
                'total_size': 0,
                'largest_file': {'name': 'N/A', 'size': 0},
                'smallest_file': {'name': 'N/A', 'size': 0},
                'average_size': 0
            }
        }
