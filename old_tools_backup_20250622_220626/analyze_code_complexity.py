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
import hashlib
import logging
from functools import lru_cache
import re
import time  # Import time module


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
    "exclude_patterns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of file patterns to exclude from analysis (e.g., ['test*.py', '*_test.py']).",
        "required": False,
        "default": [],
    },
     "include_docstrings": {
        "type": "boolean",
        "description": "Whether to include docstrings in the complexity analysis.  If True, docstrings are treated as code.",
        "required": False,
        "default": False,
    },
    "report_file": {
        "type": "string",
        "description": "Path to save a detailed report of the analysis in JSON format. If not provided, no report is saved.",
        "required": False,
    },
    "enable_profiling":{
        "type": "boolean",
        "description": "Enable detailed profiling of the tool's execution time. Creates a profiling report in the directory being scanned.",
        "required": False,
        "default": False
    },
     "ignore_import_errors": {
        "type": "boolean",
        "description": "If True, import errors will be ignored when extracting dependencies.",
        "required": False,
        "default": True
    }
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
    {
        "description": "Analyze complexity, excluding test files.",
        "code": 'analyze_code_complexity(directory="src", exclude_patterns=["test*.py", "*_test.py"])',
    },
    {
        "description": "Analyze complexity including docstrings.",
        "code": 'analyze_code_complexity(directory="src", include_docstrings=True)',
    },
    {
        "description": "Analyze complexity and save a detailed report.",
        "code": 'analyze_code_complexity(directory="src", report_file="complexity_report.json")',
    },
    {
         "description": "Analyze complexity and enable profiling.",
         "code": 'analyze_code_complexity(directory="src", enable_profiling=True)'
    },
    {
        "description": "Analyze complexity and ignore import errors.",
        "code": 'analyze_code_complexity(directory="src", ignore_import_errors=True)'
    }
]


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_file_excluded(file_path: Path, exclude_patterns: List[str]) -> bool:
    """Checks if a file path matches any of the exclude patterns."""
    for pattern in exclude_patterns:
        if file_path.match(pattern):
            return True
    return False

@lru_cache(maxsize=128)  # Cache complexity results for performance
def cached_calculate_cyclomatic_complexity(file_path: str, include_docstrings: bool = False) -> int:
    """Calculates cyclomatic complexity of a Python code block represented by an AST.  Uses caching"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
            if not include_docstrings:
               source = remove_docstrings(source)
            tree = ast.parse(source)
        return calculate_cyclomatic_complexity(tree)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return -1  # Indicate an error occurred
    except SyntaxError as e:
         logging.error(f"Syntax Error in {file_path}: {e}")
         return -1
    except Exception as e:
        logging.error(f"Error parsing or calculating complexity for {file_path} during cached operation: {e}")
        return -1  # Indicate an error occurred

def calculate_file_hash(file_path: str) -> str:
    """Calculates the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating hash for {file_path}: {e}")
        return ""

def remove_docstrings(source: str) -> str:
    """Removes docstrings from a Python source string."""
    try:
        import re
        return re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', source, flags=re.DOTALL)
    except Exception as e:
        logging.warning(f"Failed to remove docstrings: {e}")
        return source

def validate_exclude_patterns(exclude_patterns: List[str]) -> None:
    """Validates that exclude patterns are valid strings."""
    if not isinstance(exclude_patterns, list):
        raise ValueError("Exclude patterns must be a list of strings.")
    for pattern in exclude_patterns:
        if not isinstance(pattern, str):
            raise ValueError("Exclude patterns must be a list of strings.")
        try:
            re.compile(pattern)  # Test if pattern is a valid regex
        except re.error:
            raise ValueError(f"Invalid exclude pattern: {pattern}")

async def analyze_code_complexity(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes code complexity metrics for Python files in a given directory.

    Args:
        **kwargs: Keyword arguments containing 'directory' (path to the directory),
                  optionally 'max_complexity' (maximum acceptable cyclomatic complexity),
                  optionally 'exclude_patterns' (list of file patterns to exclude),
                  optionally 'include_docstrings' (boolean to include docstrings in analysis),
                  and optionally 'report_file' (path to save a detailed report).

    Returns:
        A dictionary containing the analysis results.
    """

    start_time = time.time() #Start Time for profiler

    directory = kwargs.get("directory")
    max_complexity = kwargs.get("max_complexity", 10)
    exclude_patterns = kwargs.get("exclude_patterns", [])
    include_docstrings = kwargs.get("include_docstrings", False)
    report_file = kwargs.get("report_file")
    enable_profiling = kwargs.get("enable_profiling", False)
    ignore_import_errors = kwargs.get("ignore_import_errors", True)


    # Input validation
    if not isinstance(directory, str) or not directory:
        raise ValueError("Directory must be a non-empty string.")

    if not isinstance(max_complexity, int):
        raise ValueError("Max complexity must be an integer.")
    if max_complexity <= 0:
        raise ValueError("Max complexity must be a positive integer.")
    
    validate_exclude_patterns(exclude_patterns)

    if not isinstance(include_docstrings, bool):
        raise ValueError("include_docstrings must be a boolean.")
    
    if report_file is not None and not isinstance(report_file, str):
        raise ValueError("report_file must be a string or None.")
    
    if not isinstance(enable_profiling, bool):
        raise ValueError("enable_profiling must be a boolean.")
    
    if not isinstance(ignore_import_errors, bool):
        raise ValueError("ignore_import_errors must be a boolean.")

    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory '{directory}' not found.")
        if not directory_path.is_dir():
            raise NotADirectoryError(f"'{directory}' is not a directory.")

        total_files = 0
        complex_files = 0
        complexity_details: List[Dict[str, Any]] = []
        skipped_files: List[str] = []
        errored_files: List[str] = []
        file_sizes: Dict[str, int] = {} # Store file sizes
        file_dependencies: Dict[str, List[str]] = {} #Store imported modules per file
        average_line_length: Dict[str, float] = {} #Store average line length for each file

        for file_path in directory_path.glob("**/*.py"):
            total_files += 1

            if is_file_excluded(file_path, exclude_patterns):
                skipped_files.append(str(file_path))
                continue
            try:
                file_path_str = str(file_path)
                complexity = cached_calculate_cyclomatic_complexity(file_path_str, include_docstrings)  # Use the cached function

                file_size = os.path.getsize(file_path_str)  # Get file size
                file_sizes[file_path_str] = file_size
                try:
                    file_dependencies[file_path_str] = get_imported_modules(file_path_str, ignore_import_errors)
                except Exception as e:
                    logging.error(f"Error extracting imports from {file_path}: {e}")
                    file_dependencies[file_path_str] = []  # Ensure it is initialized even if there is an error

                average_line_length[file_path_str] = calculate_average_line_length(file_path_str)

                if complexity > max_complexity and complexity != -1:
                    complex_files += 1
                    complexity_details.append({
                        "file": file_path_str,
                        "complexity": complexity,
                        "file_hash": calculate_file_hash(file_path_str),  # Add file hash
                        "file_size": file_size,
                        "average_line_length": average_line_length[file_path_str]
                    })
                elif complexity == -1:
                    errored_files.append(file_path_str)  # Track files with calculation errors

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                errored_files.append(str(file_path))
                

        results = {
            "total_files": total_files,
            "complex_files": complex_files,
            "max_complexity": max_complexity,
            "complexity_details": complexity_details,
            "skipped_files": skipped_files,
            "errored_files": errored_files,
        }

        if report_file:
             report_data = {
                 "results": results,
                 "file_sizes": file_sizes,
                 "include_docstrings": include_docstrings,
                 "file_dependencies": file_dependencies,
                 "exclude_patterns": exclude_patterns, #Include exclude patterns in report
                 "average_line_length": average_line_length
             }
             try:
                 with open(report_file, "w") as f:
                    json.dump(report_data, f, indent=4)
                 logging.info(f"Detailed report saved to {report_file}")
             except Exception as e:
                 logging.error(f"Error saving report to {report_file}: {e}")

        end_time = time.time() #End time for profiler
        execution_time = end_time - start_time

        results["execution_time"] = execution_time

        if enable_profiling:
            profiling_report = {
                "execution_time": execution_time,
                "total_files": total_files,
                "complex_files": complex_files,
            }

            profiling_file = Path(directory) / "complexity_profiling.json"

            try:
                with open(profiling_file, "w") as f:
                    json.dump(profiling_report, f, indent=4)
                logging.info(f"Profiling report saved to {profiling_file}")
            except Exception as e:
                 logging.error(f"Error saving profiling report to {profiling_file}: {e}")


        return results

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
        elif isinstance(node, ast.FunctionDef):  # Consider function definitions as a branching point
            complexity += 1
        elif isinstance(node, ast.AsyncFunctionDef):
            complexity += 1

    return complexity

def get_imported_modules(file_path: str, ignore_errors: bool = True) -> List[str]:
    """Extracts imported modules from a Python file."""
    imported_modules: List[str] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module)
    except Exception as e:
        if ignore_errors:
            logging.warning(f"Error extracting imports from {file_path} (ignored): {e}")
        else:
            raise e
    return imported_modules

def calculate_average_line_length(file_path: str) -> float:
    """Calculates the average line length of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            return 0.0
        total_length = sum(len(line.strip()) for line in lines)
        return total_length / len(lines)
    except Exception as e:
        logging.error(f"Error calculating average line length for {file_path}: {e}")
        return 0.0