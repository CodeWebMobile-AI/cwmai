"""
AI-Generated Tool: tool_task_8
Description: Calculate the overall code-to-comment ratio for the entire repository.
Generated: 2025-06-15T12:43:05.161854+00:00
Requirements: Tool to Calculate the overall code-to-comment ratio for the entire repository.
"""

from typing import Dict, Any
import glob
import os
import re
from pathlib import Path

from scripts.state_manager import StateManager


"""
Module: tool_task_8

Description: Calculate the overall code-to-comment ratio for the entire repository.
"""


__description__ = "Calculate the overall code-to-comment ratio for the entire repository."
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "Directory to analyze (defaults to current working directory)",
        "required": False
    }
}
__examples__ = [
    {"description": "Analyze current directory", "code": "await tool_task_8()"},
    {"description": "Analyze specific directory", "code": "await tool_task_8(directory='/path/to/repo')"}
]


async def tool_task_8(**kwargs) -> Dict[str, Any]:
    """
    Calculates the overall code-to-comment ratio for the entire repository.

    Args:
        **kwargs: Optional directory parameter

    Returns:
        A dictionary containing the code-to-comment ratio.

        Example:
        {
            "code_lines": 1500,
            "comment_lines": 500,
            "code_to_comment_ratio": 3.0,
            "summary": "Code-to-comment ratio: 3.0:1"
        }
    """
    state_manager = StateManager()
    
    # Get directory from kwargs or use current directory
    repo_dir = kwargs.get('directory', os.getcwd())
    
    if not os.path.isdir(repo_dir):
        return {"error": f"Directory not found: {repo_dir}"}

    total_code_lines = 0
    total_comment_lines = 0
    files_analyzed = 0

    try:
        # Find Python files
        python_files = glob.glob(os.path.join(repo_dir, "**/*.py"), recursive=True)
        
        for file_path in python_files:
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    lines = content.splitlines()
                    code_lines, comment_lines = count_code_and_comment_lines(lines)
                    total_code_lines += code_lines
                    total_comment_lines += comment_lines
                    files_analyzed += 1

                except Exception as e:
                    # Skip files that can't be read
                    continue

        if total_comment_lines == 0:
            code_to_comment_ratio = float('inf') if total_code_lines > 0 else 0.0
            ratio_str = "∞:1" if total_code_lines > 0 else "0:0"
        else:
            code_to_comment_ratio = total_code_lines / total_comment_lines
            ratio_str = f"{code_to_comment_ratio:.2f}:1"

        result = {
            "code_lines": total_code_lines,
            "comment_lines": total_comment_lines,
            "code_to_comment_ratio": code_to_comment_ratio,
            "files_analyzed": files_analyzed,
            "summary": f"Code-to-comment ratio: {ratio_str} ({files_analyzed} files analyzed)"
        }
        
        # Save to state
        state = state_manager.load_state()
        state["last_code_analysis"] = result
        state_manager.save_state(state)

        return result

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


def count_code_and_comment_lines(lines):
    """Counts the number of code and comment lines in a list of lines."""
    code_lines = 0
    comment_lines = 0
    in_multiline_comment = False

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Check for multiline comment delimiters
        if '"""' in line or "'''" in line:
            # Count occurrences to handle inline usage
            if line.count('"""') % 2 == 1 or line.count("'''") % 2 == 1:
                in_multiline_comment = not in_multiline_comment
            if in_multiline_comment or line.startswith('"""') or line.startswith("'''"):
                comment_lines += 1
            else:
                code_lines += 1
            continue

        if in_multiline_comment:
            comment_lines += 1
        elif line.startswith("#"):
            comment_lines += 1
        else:
            code_lines += 1

    return code_lines, comment_lines