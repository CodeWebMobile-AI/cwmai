"""
AI-Generated Tool: count_how_many_python
Description: Count the number of Python files with more than 100 lines of code in a directory.
Generated: 2024-11-03
Requirements: Count Python files with > 100 lines in a directory.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

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
    }
}
__examples__ = [
    {"description": "Count Python files in current directory", "code": "await count_how_many_python()"},
    {"description": "Count Python files in 'scripts' directory", "code": "await count_how_many_python(directory='scripts')"},
    {"description": "Recursively count Python files in current directory", "code": "await count_how_many_python(recursive=True)"},
    {"description": "Recursively count Python files in 'scripts' directory", "code": "await count_how_many_python(directory='scripts', recursive=True)"}
]


async def count_how_many_python(**kwargs) -> Dict[str, Any]:
    """Count Python files with more than 100 lines of code."""
    directory: str = kwargs.get('directory', '.')
    recursive: bool = kwargs.get('recursive', False)

    try:
        path = Path(directory)
        if not path.exists():
            return {"error": f"Directory not found: {directory}"}

        if not path.is_dir():
            return {"error": f"Not a directory: {directory}"}

        python_file_count: int = 0

        if recursive:
            for item in path.rglob("*.py"):
                if item.is_file():
                    try:
                        with open(item, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if len(lines) > 100:
                                python_file_count += 1
                    except Exception as e:
                        return {"error": f"Error reading file {item}: {str(e)}"}
        else:
            for item in path.glob("*.py"):
                if item.is_file():
                    try:
                        with open(item, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if len(lines) > 100:
                                python_file_count += 1
                    except Exception as e:
                        return {"error": f"Error reading file {item}: {str(e)}"}

        return {
            "directory": str(path.absolute()),
            "recursive": recursive,
            "python_file_count": python_file_count,
            "summary": f"Found {python_file_count} Python files with more than 100 lines of code."
        }

    except Exception as e:
        return {"error": f"Error counting Python files: {str(e)}"}