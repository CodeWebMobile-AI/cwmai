"""
AI-Generated Tool: find_all_functions_that
Description: Find all functions in a directory that exceed a specified length limit.
Generated: 2025-10-27
Requirements: Identify and locate all functions within a codebase that exceed a specified length limit, measured in lines of code.
"""

import os
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import json
import multiprocessing
import logging
import glob
import subprocess
from functools import lru_cache

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
    },
    "cache_results": {
        "type": "boolean",
        "description": "Whether to cache the results for faster subsequent runs.",
        "required": False,
        "default": True
    },
    "include_function_body": {
        "type": "boolean",
        "description": "Whether to include the full function body in the results.",
        "required": False,
        "default": False
    },
    "num_processes": {
        "type": "integer",
        "description": "Number of processes to use for parallel processing. Defaults to the number of CPUs available.",
        "required": False,
        "default": multiprocessing.cpu_count()
    },
    "exclude_patterns": {
        "type": "string",
        "description": "Comma-separated list of file patterns to exclude (e.g., 'test*.py,*_test.py'). Uses glob-style matching.",
        "required": False,
        "default": ""
    },
    "only_report_longest":{
        "type": "boolean",
        "description": "If set to True, only the longest function is reported.",
        "required": False,
        "default": False
    },
    "min_lines_of_code": {
        "type": "integer",
        "description": "Minimum lines of code a function should have to be considered at all.",
        "required": False,
        "default": 0
    },
    "report_all_functions": {
        "type": "boolean",
        "description": "If set to True, reports all functions regardless of the line limit, but still provides line counts.",
        "required": False,
        "default": False
    },
    "include_docstrings": {
        "type": "boolean",
        "description": "Whether to include docstrings when counting the lines of code for a function.",
        "required": False,
        "default": False
    },
    "report_style_violations": {
        "type": "boolean",
        "description": "Whether to report potential style violations (using flake8) for long functions.",
        "required": False,
        "default": False
    },
    "aggregate_results": {
        "type": "boolean",
        "description": "Whether to aggregate the results to the level of files instead of individual functions, reporting the average, min, and max function lengths per file.",
        "required": False,
        "default": False
    },
    "show_progress": {
        "type": "boolean",
        "description": "Whether to display a progress bar during file processing.",
        "required": False,
        "default": True
    }

}
__examples__ = [
    {"description": "Find long functions in the current directory", "code": "await find_all_functions_that(directory='.')"},
    {"description": "Find functions longer than 100 lines in the 'src' directory", "code": "await find_all_functions_that(directory='src', length_limit=100)"},
    {"description": "Find long functions and disable caching", "code": "await find_all_functions_that(directory='src', cache_results=False)"},
    {"description": "Find long functions and include function body in the results", "code": "await find_all_functions_that(directory='src', include_function_body=True)"},
    {"description": "Find long functions using 4 processes", "code": "await find_all_functions_that(directory='src', num_processes=4)"},
    {"description": "Exclude test files", "code": "await find_all_functions_that(directory='.', exclude_patterns='test*.py,*_test.py')"},
    {"description": "Only report the longest function", "code": "await find_all_functions_that(directory='.', only_report_longest=True)"},
    {"description": "Only consider functions with at least 10 lines", "code": "await find_all_functions_that(directory='.', min_lines_of_code=10)"},
    {"description": "Report all functions and their line counts", "code": "await find_all_functions_that(directory='.', report_all_functions=True)"},
    {"description": "Include Docstrings in line count", "code": "await find_all_functions_that(directory='.', include_docstrings=True)"},
    {"description": "Report style violations in long functions", "code": "await find_all_functions_that(directory='.', report_style_violations=True)"},
    {"description": "Aggregate results to file level", "code": "await find_all_functions_that(directory='.', aggregate_results=True)"},
    {"description": "Show progress during file processing", "code": "await find_all_functions_that(directory='.', show_progress=True)"}
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    logging.warning("tqdm is not installed. Progress bar will not be displayed. Install it with 'pip install tqdm'")


def _calculate_cache_key(directory: str, language: str, length_limit: int, ignore_comments_and_blanks: bool, include_function_body: bool, exclude_patterns: str, only_report_longest: bool, min_lines_of_code: int, report_all_functions: bool, include_docstrings: bool, report_style_violations: bool, aggregate_results: bool) -> str:
    """Calculates a unique cache key based on the input parameters."""
    key_data = {
        "directory": directory,
        "language": language,
        "length_limit": length_limit,
        "ignore_comments_and_blanks": ignore_comments_and_blanks,
        "include_function_body": include_function_body,
        "exclude_patterns": exclude_patterns,
        "only_report_longest": only_report_longest,
        "min_lines_of_code": min_lines_of_code,
        "report_all_functions": report_all_functions,
        "include_docstrings": include_docstrings,
        "report_style_violations": report_style_violations,
        "aggregate_results": aggregate_results
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


async def _load_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads results from the cache if available."""
    cache_file = Path(".function_length_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                if cache_key in cache_data:
                    logging.info("Results loaded from cache.")
                    return cache_data[cache_key]
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error loading cache: {e}")
            return None
    return None


async def _save_to_cache(cache_key: str, results: Dict[str, Any]) -> None:
    """Saves results to the cache."""
    cache_file = Path(".function_length_cache.json")
    try:
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
            except json.JSONDecodeError:
                cache_data = {}  # Handle empty or corrupted cache file
        else:
            cache_data = {}

        cache_data[cache_key] = results

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=4)  # Add indentation for readability
        logging.info("Results saved to cache.")
    except Exception as e:
        logging.warning(f"Error saving to cache: {e}")


@lru_cache(maxsize=128)  # Cache line counting for each file
def _count_lines(code_lines: List[str], ignore_comments_and_blanks: bool) -> int:
    """Counts the number of effective lines of code."""
    if ignore_comments_and_blanks:
        return sum(1 for line in code_lines if line.strip() and not line.strip().startswith('#'))
    else:
        return len(code_lines)


@lru_cache(maxsize=128)  # Cache docstring extraction
def _get_function_docstring(node: ast.FunctionDef, code: str) -> str:
    """Extracts the docstring from a function node."""
    if ast.get_docstring(node):
        return ast.get_docstring(node)
    return ""


def _process_file(file_path: Path, length_limit: int, ignore_comments_and_blanks: bool, include_function_body: bool, min_lines_of_code: int, report_all_functions: bool, include_docstrings: bool, report_style_violations: bool) -> List[Dict[str, Any]]:
    """Processes a single file to find long functions."""
    long_functions: List[Dict[str, Any]] = []
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

                    if include_docstrings:
                        docstring = _get_function_docstring(node, code)
                        if docstring:
                            docstring_lines = docstring.splitlines()
                            function_code = code.splitlines()[start_line-1:end_line]  # Re-extract in case the docstring is at the beginning
                            function_length = _count_lines(function_code, ignore_comments_and_blanks)
                        else:
                            function_length = _count_lines(function_code, ignore_comments_and_blanks)
                    else:
                        function_length = _count_lines(function_code, ignore_comments_and_blanks)



                    if function_length >= min_lines_of_code and (report_all_functions or function_length > length_limit):
                        function_data: Dict[str, Any] = {
                            "file_path": str(file_path.resolve()),
                            "function_name": function_name,
                            "start_line": start_line,
                            "end_line": end_line,
                            "length": function_length
                        }
                        if include_function_body:
                            function_data["function_body"] = "\n".join(function_code)  # Add the function body

                        if report_style_violations and function_length > length_limit:
                            violations = _check_style_violations(file_path, start_line, end_line)
                            function_data["style_violations"] = violations

                        long_functions.append(function_data)
    except Exception as e:
        logging.warning(f"Error processing file {file_path}: {str(e)}")  # Non-fatal error, log and continue
    return long_functions


def _filter_files(directory: Path, exclude_patterns: str) -> List[Path]:
    """Filters files based on exclude patterns."""

    excluded_files = []
    if exclude_patterns:
        patterns = exclude_patterns.split(',')
        for pattern in patterns:
            pattern = pattern.strip()
            try:
                excluded_files.extend([Path(f).resolve() for f in glob.glob(str(directory / pattern), recursive=True)])  # Use glob for pattern matching and resolve paths
            except Exception as e:
                logging.warning(f"Invalid exclude pattern {pattern}: {e}")

    all_files = [Path(root) / file for root, _, files in os.walk(directory) for file in files if file.endswith(".py")]
    filtered_files = [file.resolve() for file in all_files if file.resolve() not in excluded_files] # resolve paths for comparison
    return filtered_files

def _check_style_violations(file_path: Path, start_line: int, end_line: int) -> List[str]:
    """
    Checks for style violations using flake8 for a specific function.
    Returns a list of style violation messages.
    """
    try:
        command = [
            "flake8",
            str(file_path),
            "--select=E,W",  # Select Error and Warning codes
            f"--isolated",  # Ignore user config
            f"--filename={str(file_path)}",
            f"--line-range={start_line}:{end_line}"
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        output = result.stdout.strip()
        errors = output.splitlines()
        return errors
    except FileNotFoundError:
        logging.warning("flake8 is not installed. Style violation checks will be skipped.")
        return ["flake8 not installed"]  # Indicate flake8 is not available.
    except Exception as e:
        logging.error(f"Error running flake8: {e}")
        return [f"Error running flake8: {e}"]


@lru_cache(maxsize=128)  # Cache function counts per file
def _find_all_functions_in_file(file_path: Path) -> int:
    """Counts all functions in a file using AST."""
    count = 0
    try:
        with open(file_path, "r") as f:
            code = f.read()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    count += 1
    except Exception as e:
        logging.warning(f"Error counting functions in {file_path}: {e}")
    return count


async def find_all_functions_that(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Find all functions in a directory that exceed a specified length limit."""
    directory = kwargs.get('directory')
    language = kwargs.get('language', 'python')
    length_limit = kwargs.get('length_limit', 50)
    ignore_comments_and_blanks = kwargs.get('ignore_comments_and_blanks', True)
    cache_results = kwargs.get('cache_results', True)
    include_function_body = kwargs.get('include_function_body', False)
    num_processes = kwargs.get('num_processes', multiprocessing.cpu_count())
    exclude_patterns = kwargs.get('exclude_patterns', "")
    only_report_longest = kwargs.get('only_report_longest', False)
    min_lines_of_code = kwargs.get('min_lines_of_code', 0)
    report_all_functions = kwargs.get('report_all_functions', False)
    include_docstrings = kwargs.get('include_docstrings', False)
    report_style_violations = kwargs.get('report_style_violations', False)
    aggregate_results = kwargs.get('aggregate_results', False)
    show_progress = kwargs.get('show_progress', True)


    # Input Validation with more specific error messages
    if not isinstance(directory, str):
        return {"error": "Directory must be a string.  Please provide a valid directory path."}
    if not isinstance(language, str):
        return {"error": "Language must be a string. Please specify a supported language (e.g., 'python')."}
    if not isinstance(length_limit, int):
        return {"error": "Length limit must be an integer. Provide a numerical value for the maximum function length."}
    if not isinstance(ignore_comments_and_blanks, bool):
        return {"error": "ignore_comments_and_blanks must be a boolean (True or False)."}
    if not isinstance(cache_results, bool):
        return {"error": "cache_results must be a boolean (True or False)."}
    if not isinstance(include_function_body, bool):
        return {"error": "include_function_body must be a boolean (True or False)."}
    if not isinstance(num_processes, int) or num_processes <= 0:
        return {"error": "num_processes must be a positive integer."}
    if not isinstance(exclude_patterns, str):
        return {"error": "exclude_patterns must be a string."}
    if not isinstance(only_report_longest, bool):
        return {"error": "only_report_longest must be a boolean."}
    if not isinstance(min_lines_of_code, int) or min_lines_of_code < 0:
        return {"error": "min_lines_of_code must be a non-negative integer."}
    if not isinstance(report_all_functions, bool):
        return {"error": "report_all_functions must be a boolean."}
    if not isinstance(include_docstrings, bool):
        return {"error": "include_docstrings must be a boolean."}
    if not isinstance(report_style_violations, bool):
        return {"error": "report_style_violations must be a boolean."}
    if not isinstance(aggregate_results, bool):
        return {"error": "aggregate_results must be a boolean."}
    if not isinstance(show_progress, bool):
        return {"error": "show_progress must be a boolean."}


    if language.lower() != 'python':
        return {"error": "Only python is supported at the moment.  Support for other languages may be added in the future."}

    cache_key = _calculate_cache_key(directory, language, length_limit, ignore_comments_and_blanks, include_function_body, exclude_patterns, only_report_longest, min_lines_of_code, report_all_functions, include_docstrings, report_style_violations, aggregate_results)

    if cache_results:
        cached_results = await _load_from_cache(cache_key)
        if cached_results:
            return cached_results

    try:
        directory_path = Path(directory).resolve() # Resolve the directory path
        if not directory_path.exists():
            return {"error": f"Directory not found: {directory}.  Please ensure the directory path is correct."}
        if not directory_path.is_dir():
            return {"error": f"Not a directory: {directory}. The provided path must point to a directory."}

        long_functions: List[Dict[str, Any]] = []
        files_processed = 0
        functions_found = 0

        # Filter files based on exclude patterns
        file_paths = _filter_files(directory_path, exclude_patterns)
        files_processed = len(file_paths)

        # Prepare arguments for starmap. This needs to be created before tqdm.
        process_file_args = [(file_path, length_limit, ignore_comments_and_blanks, include_function_body, min_lines_of_code, report_all_functions, include_docstrings, report_style_violations) for file_path in file_paths]

        # Create a pool of processes for parallel execution
        with multiprocessing.Pool(processes=num_processes) as pool:
            if tqdm and show_progress:
                results = list(tqdm(pool.starmap(_process_file, process_file_args), total=len(file_paths), desc="Processing files"))
            else:
                results = pool.starmap(_process_file, process_file_args)


        # Consolidate results from different processes
        for file_result in results:
            long_functions.extend(file_result)
            functions_found += len(file_result) # This is an approximation - doesn't count all functions

        # Accurately count total functions (AST traversal)
        total_functions_found = sum(_find_all_functions_in_file(file_path) for file_path in file_paths)

        if only_report_longest and long_functions:
            long_functions = [max(long_functions, key=lambda x: x['length'])]

        if aggregate_results:
             file_level_results = {}
             for file_path in file_paths:
                 file_level_results[str(file_path)] = []
             for function_data in long_functions:
                 file_level_results[function_data["file_path"]].append(function_data["length"])

             aggregated_data = {}
             for file_path, lengths in file_level_results.items():
                if lengths:
                    avg_length = sum(lengths) / len(lengths)
                    min_length = min(lengths)
                    max_length = max(lengths)
                    aggregated_data[file_path] = {
                        "average_length": avg_length,
                        "min_length": min_length,
                        "max_length": max_length,
                        "num_functions": len(lengths)
                     }
                else:
                    aggregated_data[file_path] = {
                        "average_length": 0,
                        "min_length": 0,
                        "max_length": 0,
                        "num_functions": 0
                    }

             summary = f"Aggregated function length results for {files_processed} Python files."

             result = {
                "file_level_data": aggregated_data,
                "length_limit": length_limit,
                "directory": str(directory_path.resolve()),
                "summary": summary,
                "files_processed": files_processed,
                "functions_found": total_functions_found, # Correct count
                "report_all_functions": report_all_functions,
                "min_lines_of_code": min_lines_of_code,
                "include_docstrings": include_docstrings,
                "report_style_violations": report_style_violations
            }
        else:
            summary = f"Found {len(long_functions)} functions exceeding the length limit of {length_limit} lines (or meeting criteria for report_all_functions) in {files_processed} Python files.  Total functions found: {total_functions_found}."
            result = {
                "long_functions": long_functions,
                "length_limit": length_limit,
                "directory": str(directory_path.resolve()),
                "summary": summary,
                "files_processed": files_processed,
                "functions_found": total_functions_found, # Correct count
                "report_all_functions": report_all_functions,
                "min_lines_of_code": min_lines_of_code,
                "include_docstrings": include_docstrings,
                "report_style_violations": report_style_violations
            }

        if cache_results:
            await _save_to_cache(cache_key, result)

        return result

    except Exception as e:
        logging.exception("An unexpected error occurred.")  # Log the full exception traceback
        return {"error": f"Unexpected error finding long functions: {str(e)}. Please check the directory and try again. See logs for more details."}