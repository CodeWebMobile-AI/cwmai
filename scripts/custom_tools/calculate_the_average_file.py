"""
AI-Generated Tool: calculate_the_average_file
Description: Calculates the average file size within each directory of a given directory structure.
Generated: 2024-02-29
Requirements: Recursive directory traversal, file size retrieval, directory grouping, average calculation, error handling.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

__description__ = "Calculates the average file size within each directory of a given directory structure."
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "Root directory to analyze. Defaults to current directory.",
        "required": False,
        "default": "."
    },
    "max_depth": {
        "type": "integer",
        "description": "Maximum depth of directory traversal.  Defaults to no limit.",
        "required": False,
        "default": -1  # Represents no limit
    }
}
__examples__ = [
    {"description": "Calculate average file size in the current directory.", "code": "await calculate_the_average_file()"},
    {"description": "Calculate average file size in a specific directory.", "code": "await calculate_the_average_file(directory='path/to/directory')"},
    {"description": "Calculate average file size with a maximum depth of 2.", "code": "await calculate_the_average_file(directory='path/to/directory', max_depth=2)"}
]


async def calculate_the_average_file(**kwargs) -> Dict[str, Any]:
    """Calculates the average file size within each directory of a given directory structure."""
    directory: str = kwargs.get('directory', '.')
    max_depth: int = kwargs.get('max_depth', -1)

    try:
        path = Path(directory)
        if not path.exists():
            return {"error": f"Directory not found: {directory}"}

        if not path.is_dir():
            return {"error": f"Not a directory: {directory}"}

        directory_sizes: Dict[str, Dict[str, Any]] = {}

        def traverse_directory(current_path: Path, depth: int) -> None:
            """Recursively traverses the directory structure."""
            if max_depth != -1 and depth > max_depth:
                return

            files_in_dir: List[int] = []
            for item in current_path.iterdir():
                if item.is_file():
                    try:
                        files_in_dir.append(item.stat().st_size)
                    except Exception as e:
                        print(f"Error getting file size for {item}: {e}")  # or log
                elif item.is_dir():
                    traverse_directory(item, depth + 1)  # Recursive call

            if files_in_dir:
                average_size = sum(files_in_dir) / len(files_in_dir)
                directory_sizes[str(current_path.absolute())] = {
                    "average_size": average_size,
                    "file_count": len(files_in_dir),
                    "total_size": sum(files_in_dir)
                }
            else:
                directory_sizes[str(current_path.absolute())] = {
                    "average_size": 0,
                    "file_count": 0,
                    "total_size": 0
                }

        traverse_directory(path, 0)

        results = []
        for dir_path, data in directory_sizes.items():
            results.append({
                "directory": dir_path,
                "average_size": data["average_size"],
                "file_count": data["file_count"],
                "total_size": data["total_size"]
            })

        summary = f"Analyzed {len(directory_sizes)} directories."
        return {
            "results": results,
            "summary": summary
        }

    except Exception as e:
        return {"error": f"Error calculating average file sizes: {str(e)}"}