"""
AI-Generated Tool: find_all_functions_that
Description: Find all functions in a directory that exceed a specified length limit.
Generated: 2025-10-27
Requirements: Identify and locate all functions within a codebase that exceed a specified length limit, measured in lines of code.
"""

import os
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional

__description__ = "Find all functions in a directory that exceed a specified length limit."
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "Directory path to search for files.",
        "required": True
    },
    "language": {
        "type": "string",
        "description": "Programming language of the files (e.g., 'python'). Currently only supports 'python'.",
        "required": False,
        "default": "python"
    },
    "length_limit": {
        "type": "integer",
        "description": "Maximum allowed length of a function in lines of code.",
        "required": False,
        "default": 50
    },
    "ignore_comments_and_blanks": {
        "type": "boolean",
        "description": "Whether to ignore comments and blank lines when counting lines of code.",
        "required": False,
        "default": True
    }
}
__examples__ = [
    {"description": "Find long functions in the current directory", "code": "await find_all_functions_that(directory='.')"},
    {"description": "Find functions longer than 100 lines in the 'src' directory", "code": "await find_all_functions_that(directory='src', length_limit=100)"}
]


async def find_all_functions_that(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Find all functions in a directory that exceed a specified length limit."""
    directory = kwargs.get('directory')
    language = kwargs.get('language', 'python')
    length_limit = kwargs.get('length_limit', 50)
    ignore_comments_and_blanks = kwargs.get('ignore_comments_and_blanks', True)

    if not isinstance(directory, str):
        return {"error": "Directory must be a string."}
    if not isinstance(language, str):
        return {"error": "Language must be a string."}
    if not isinstance(length_limit, int):
        return {"error": "Length limit must be an integer."}
    if not isinstance(ignore_comments_and_blanks, bool):
        return {"error": "ignore_comments_and_blanks must be a boolean."}

    if language.lower() != 'python':
        return {"error": "Only python is supported at the moment."}

    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return {"error": f"Directory not found: {directory}"}
        if not directory_path.is_dir():
            return {"error": f"Not a directory: {directory}"}

        long_functions: List[Dict[str, Any]] = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, "r") as f:
                            code = f.read()
                            tree = ast.parse(code)
                            for node in ast.walk(tree):
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    function_name = node.name
                                    start_line = node.lineno
                                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno # Handle cases where end_lineno is not available

                                    # Get the function's code block
                                    function_code = code.splitlines()[start_line-1:end_line]

                                    # Remove comments and blank lines if requested
                                    if ignore_comments_and_blanks:
                                        function_code_filtered = [
                                            line for line in function_code
                                            if line.strip() and not line.strip().startswith('#')
                                        ]
                                    else:
                                        function_code_filtered = function_code

                                    function_length = len(function_code_filtered)

                                    if function_length > length_limit:
                                        long_functions.append({
                                            "file_path": str(file_path.resolve()),
                                            "function_name": function_name,
                                            "start_line": start_line,
                                            "end_line": end_line,
                                            "length": function_length
                                        })
                    except Exception as e:
                        return {"error": f"Error processing file {file_path}: {str(e)}"}

        summary = f"Found {len(long_functions)} functions exceeding the length limit of {length_limit} lines."
        return {
            "long_functions": long_functions,
            "length_limit": length_limit,
            "directory": str(directory_path.resolve()),
            "summary": summary
        }

    except Exception as e:
        return {"error": f"Error finding long functions: {str(e)}"}