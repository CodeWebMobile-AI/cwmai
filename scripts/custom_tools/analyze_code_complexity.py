"""
AI-Generated Tool: analyze_code_complexity
Description: Analyze code complexity metrics
Generated: 2025-06-15T11:45:35.628691+00:00
Requirements: 
        This tool should:
        1. Use pathlib to navigate directories
        2. Use ast to parse Python files
        3. Calculate cyclomatic complexity
        4. Return metrics as JSON
        
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Dict, Any, List
from typing import List

import ast
import asyncio
import json
import os


"""
Module: analyze_code_complexity

Description: Analyzes code complexity metrics for Python files in a directory.
"""


__description__ = "Analyze code complexity metrics for Python files."

__parameters__ = {
    "directory": {
        "type": "string",
        "description": "Path to the directory containing Python files.",
        "required": True,
    },
    "max_complexity": {
        "type": "integer",
        "description": "Maximum acceptable cyclomatic complexity.",
        "required": False,
        "default": 10,
    },
}

__examples__ = [
    {
        "description": "Analyze complexity of Python files in the 'src' directory.",
        "code": 'analyze_code_complexity(directory="src")',
    },
    {
        "description": "Analyze complexity with a maximum complexity threshold of 5.",
        "code": 'analyze_code_complexity(directory="src", max_complexity=5)',
    },
]


async def analyze_code_complexity(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes code complexity metrics for Python files in a given directory.

    Args:
        **kwargs: Keyword arguments containing 'directory' (path to the directory) and
                  optionally 'max_complexity' (maximum acceptable cyclomatic complexity).

    Returns:
        A dictionary containing the analysis results.
    """

    directory = kwargs.get("directory")
    max_complexity = kwargs.get("max_complexity", 10)

    if not directory:
        raise ValueError("Directory must be specified.")

    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory '{directory}' not found.")
        if not directory_path.is_dir():
            raise NotADirectoryError(f"'{directory}' is not a directory.")

        total_files = 0
        complex_files = 0
        complexity_details: List[Dict[str, Any]] = []

        for file_path in directory_path.glob("**/*.py"):
            total_files += 1
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                complexity = calculate_cyclomatic_complexity(tree)

                if complexity > max_complexity:
                    complex_files += 1
                    complexity_details.append({
                        "file": str(file_path),
                        "complexity": complexity
                    })

            except Exception as e:
                print(f"Error processing file {file_path}: {e}") # Log error, don't stop execution.

        return {
            "total_files": total_files,
            "complex_files": complex_files,
            "max_complexity": max_complexity,
            "complexity_details": complexity_details
        }

    except (ValueError, FileNotFoundError, NotADirectoryError) as e:
        raise e  # Re-raise validation errors
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def calculate_cyclomatic_complexity(tree: ast.AST) -> int:
    """Calculates cyclomatic complexity of a Python code block represented by an AST."""

    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
             complexity += 1 #List Comprehension
    return complexity
