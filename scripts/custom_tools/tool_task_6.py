"""
AI-Generated Tool: tool_task_6
Description: Calculate the number of comment lines in each file by subtracting the non-comment code lines from the total lines.
Generated: 2025-06-15T12:42:58.726582+00:00
Requirements: Tool to Calculate the number of comment lines in each file by subtracting the non-comment code lines from the total lines.
"""

from typing import Any
from typing import Dict
from typing import Dict, Any

import os

from scripts.state_manager import StateManager


"""
Module for calculating comment lines in files by subtracting non-comment code lines from total lines.

This tool analyzes files to determine the number of comment lines by comparing the total lines
with the count of non-comment code lines. It supports various file types and comment syntaxes.
"""


__description__ = "Calculate the number of comment lines in each file by subtracting non-comment code lines from total lines."
__parameters__ = {
    "file_path": "str: Path to the file to analyze",
    "comment_symbols": "List[str]: List of comment symbols for the file type (e.g., ['#'] for Python)"
}
__examples__ = [
    {
        "file_path": "example.py",
        "comment_symbols": ["#"]
    }
]

async def tool_task_6(**kwargs) -> Dict[str, Any]:
    """
    Calculate comment lines in a file by subtracting non-comment code lines from total lines.
    
    Args:
        **kwargs: Must include 'file_path' and 'comment_symbols'.
        
    Returns:
        Dict with keys:
            - 'total_lines': int
            - 'code_lines': int
            - 'comment_lines': int
            - 'file_path': str
            
    Raises:
        ValueError: If required parameters are missing or file doesn't exist.
    """
    state_manager = StateManager()
    state = state_manager.load_state()
    
    # Validate required parameters
    if 'file_path' not in kwargs:
        raise ValueError("Missing required parameter: 'file_path'")
    if 'comment_symbols' not in kwargs:
        raise ValueError("Missing required parameter: 'comment_symbols'")
    
    file_path = kwargs['file_path']
    comment_symbols = kwargs['comment_symbols']
    
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    total_lines = 0
    code_lines = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            total_lines += 1
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                continue
                
            # Check if line starts with any comment symbol
            is_comment = any(stripped_line.startswith(symbol) for symbol in comment_symbols)
            if not is_comment:
                code_lines += 1
    
    comment_lines = total_lines - code_lines
    
    return {
        'total_lines': total_lines,
        'code_lines': code_lines,
        'comment_lines': comment_lines,
        'file_path': file_path
    }
