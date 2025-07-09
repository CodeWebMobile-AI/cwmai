"""
AI-Generated Tool: find_async_functions
Description: Find all functions that use async/await
Generated: 2025-06-15T12:45:11.724188+00:00
Requirements: Search all Python files for async function definitions and await statements
"""

from typing import Any
from typing import Dict
from typing import Dict, Any, List
from typing import List

import asyncio
import os
import re
import hashlib
import json
import logging
import concurrent.futures
from collections import defaultdict
from fnmatch import fnmatch

"""
Module: find_async_functions

Description: Find all functions that use async/await in Python files.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


__description__ = "Find all functions that use async/await in Python files."
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "The directory to search for Python files.",
        "required": True,
    },
    "cache_file": {
        "type": "string",
        "description": "Path to the cache file.  If not provided, caching is disabled.",
        "required": False,
    },
    "exclude_patterns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of regex patterns to exclude files.",
        "required": False,
    },
    "include_venv": {
        "type": "boolean",
        "description": "Include virtual environment directories in the search.",
        "required": False,
        "default": False,
    },
    "max_depth": {
        "type": "integer",
        "description": "Maximum directory depth to traverse. Defaults to unlimited.",
        "required": False,
        "default": None
    },
    "report_file": {
        "type": "string",
        "description": "Path to a file to save the analysis report (JSON format).  If not provided, no report is saved.",
        "required": False,
    },
    "concurrency": {
        "type": "integer",
        "description": "Number of concurrent tasks to use for file processing. Defaults to 4.",
        "required": False,
        "default": 4,
    },
    "include_file_content": {
        "type": "boolean",
        "description": "Whether to include matched async functions and await statements along with their context in the report. Defaults to False.",
        "required": False,
        "default": False,
    },
    "context_length": {
        "type": "integer",
        "description": "The number of characters of context to include around each match when include_file_content is True. Defaults to 50.",
        "required": False,
        "default": 50
    },
    "include_line_numbers": {
        "type": "boolean",
        "description": "Whether to include line numbers for matches when include_file_content is True. Defaults to False",
        "required": False,
        "default": False
    },
    "output_format": {
        "type": "string",
        "description": "The format of the report ('json' or 'text'). Defaults to 'json'.",
        "required": False,
        "default": "json"
    },
    "aggregate_by_file": {
        "type": "boolean",
        "description": "Aggregate the results by file (count of async functions and awaits per file). Defaults to False.",
        "required": False,
        "default": False
    },
    "pattern_type": {
        "type": "string",
        "description": "The pattern matching type for exclude_patterns ('regex' or 'glob'). Defaults to 'regex'.",
        "required": False,
        "default": "regex"
    },
    "skip_hidden": {
        "type": "boolean",
        "description": "Skip hidden files and directories (starting with a dot). Defaults to True.",
        "required": False,
        "default": True
    },
}
__examples__ = [
    {
        "description": "Find async functions in the current directory.",
        "input": {"directory": "."},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    },
    {
        "description": "Find async functions in the current directory, excluding test files.",
        "input": {"directory": ".", "exclude_patterns": ["test_.*\\.py"]},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 1, "errors": []},
    },
    {
        "description": "Find async functions in the current directory, using a cache file.",
        "input": {"directory": ".", "cache_file": "async_function_cache.json"},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    },
    {
        "description": "Find async functions, excluding virtual environments.",
        "input": {"directory": ".", "include_venv": False},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    },
        {
        "description": "Find async functions in the current directory with maximum depth of 2.",
        "input": {"directory": ".", "max_depth": 2},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    },
     {
        "description": "Find async functions and save the report to a file.",
        "input": {"directory": ".", "report_file": "async_report.json"},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    },
    {
        "description": "Find async functions using more concurrency.",
        "input": {"directory": ".", "concurrency": 8},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    },
    {
        "description": "Find async functions including matched content.",
        "input": {"directory": ".", "include_file_content": True},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    },
    {
        "description": "Find async functions including matched content and line numbers.",
        "input": {"directory": ".", "include_file_content": True, "include_line_numbers": True},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    },
    {
       "description": "Find async functions and aggregate results by file",
       "input": {"directory": ".", "aggregate_by_file": True},
       "expected_output": {"files_analyzed": 3, "files_skipped": 0, "errors": [], "file_analysis": {"./file1.py": {"async_function_count": 2, "await_statement_count": 5}, "./file2.py": {"async_function_count": 1, "await_statement_count": 3}}}
    },
    {
        "description": "Find async functions and save report in text format.",
        "input": {"directory": ".", "report_file": "async_report.txt", "output_format": "text"},
        "expected_output": "See async_report.txt",
    },
    {
        "description": "Find async functions using glob patterns for exclusion.",
        "input": {"directory": ".", "exclude_patterns": ["test_*.py", "*_test.py"], "pattern_type": "glob"},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 2, "errors": []},
    },
    {
        "description": "Find async functions, skipping hidden files and directories.",
        "input": {"directory": ".", "skip_hidden": True},
        "expected_output": {"async_function_count": 5, "await_statement_count": 12, "files_analyzed": 3, "files_skipped": 0, "errors": []},
    }

]


async def find_async_functions(**kwargs) -> Dict[str, Any]:
    """
    Finds all functions that use async/await in Python files within a specified directory.

    Args:
        **kwargs: Keyword arguments containing the 'directory' to search.  Also supports optional arguments:
                   'cache_file': Path to a JSON file to store cache data.
                   'exclude_patterns': A list of regex patterns of filenames to exclude.
                   'include_venv': Whether to include virtual environment directories (default: False)
                   'max_depth': Maximum directory depth to traverse (default: None, unlimited)
                   'report_file': Path to save analysis report.
                   'concurrency': Number of concurrent tasks (default: 4)
                   'include_file_content': Include matched content in the report (default: False)
                   'context_length': Length of context around matches (default: 50)
                   'include_line_numbers': Whether to include line numbers (default: False)
                   'output_format': The format of the report ('json' or 'text'). Defaults to 'json'.
                   'aggregate_by_file': Aggregate results by file.
                   'pattern_type': Type of pattern for exclude_patterns ('regex' or 'glob'). Defaults to 'regex'.
                   'skip_hidden': Skip hidden files/directories. Defaults to True.

    Returns:
        A dictionary containing the counts of async functions, await statements, files analyzed, skipped files, and any errors encountered.
    """

    directory = kwargs.get("directory")
    cache_file = kwargs.get("cache_file")
    exclude_patterns = kwargs.get("exclude_patterns", [])
    include_venv = kwargs.get("include_venv", False)
    max_depth = kwargs.get("max_depth")
    report_file = kwargs.get("report_file")
    concurrency = kwargs.get("concurrency", 4)
    include_file_content = kwargs.get("include_file_content", False)
    context_length = kwargs.get("context_length", 50)
    include_line_numbers = kwargs.get("include_line_numbers", False)
    output_format = kwargs.get("output_format", "json").lower()
    aggregate_by_file = kwargs.get("aggregate_by_file", False)
    pattern_type = kwargs.get("pattern_type", "regex").lower()
    skip_hidden = kwargs.get("skip_hidden", True)

    # Input Validation
    if not directory:
        raise ValueError("Directory must be specified.")

    if not isinstance(directory, str):
        raise TypeError("Directory must be a string.")

    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")

    if exclude_patterns and not isinstance(exclude_patterns, list):
        raise TypeError("exclude_patterns must be a list of strings.")
    if exclude_patterns:
        for pattern in exclude_patterns:
            if not isinstance(pattern, str):
                raise TypeError("Each exclude pattern must be a string.")

    if max_depth is not None and not isinstance(max_depth, int):
        raise TypeError("max_depth must be an integer or None.")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be a positive integer.")

    if report_file is not None and not isinstance(report_file, str):
        raise TypeError("report_file must be a string.")

    if not isinstance(concurrency, int):
        raise TypeError("concurrency must be an integer.")
    if concurrency <= 0:
        raise ValueError("concurrency must be a positive integer.")
    if not isinstance(include_file_content, bool):
        raise TypeError("include_file_content must be a boolean.")
    if not isinstance(context_length, int):
        raise TypeError("context_length must be an integer.")
    if context_length <= 0:
        raise ValueError("context_length must be a positive integer.")
    if not isinstance(include_line_numbers, bool):
        raise TypeError("include_line_numbers must be a boolean.")
    if output_format not in ("json", "text"):
        raise ValueError("output_format must be 'json' or 'text'.")
    if not isinstance(aggregate_by_file, bool):
        raise TypeError("aggregate_by_file must be a boolean.")
    if pattern_type not in ("regex", "glob"):
        raise ValueError("pattern_type must be 'regex' or 'glob'.")
    if not isinstance(skip_hidden, bool):
        raise TypeError("skip_hidden must be a boolean.")



    async_function_count = 0
    await_statement_count = 0
    files_analyzed = 0
    files_skipped = 0
    errors = []
    detailed_results = [] # Store matched content when include_file_content is enabled
    file_analysis = defaultdict(lambda: {"async_function_count": 0, "await_statement_count": 0}) #Aggregate counts per file

    cache = {}
    if cache_file:
        cache = await load_cache(cache_file, errors)

    def get_line_number(content: str, match_obj: re.Match) -> int:
        """Get line number of a match object in the content string."""
        return content[:match_obj.start()].count('\n') + 1

    def matches_exclude_pattern(filepath: str, exclude_patterns: List[str], pattern_type: str) -> bool:
        """Checks if the filepath matches any of the exclude patterns."""
        for pattern in exclude_patterns:
            if pattern_type == "regex":
                if re.search(pattern, filepath):
                    return True
            elif pattern_type == "glob":
                if fnmatch(filepath, pattern):
                    return True
        return False

    async def process_file(filepath: str) -> None:
        nonlocal async_function_count, await_statement_count, files_analyzed, errors, detailed_results, file_analysis
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

                # Create a hash of the file content for cache invalidation
                file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

                if cache_file and filepath in cache and cache[filepath]['hash'] == file_hash:
                    # Use cached results
                    cached_data = cache[filepath]
                    async_function_count += cached_data['async_function_count']
                    await_statement_count += cached_data['await_statement_count']
                    logger.info(f"Using cached data for {filepath}")
                    if include_file_content and 'detailed_results' in cached_data:
                         detailed_results.extend(cached_data['detailed_results'])

                    if aggregate_by_file:
                        file_analysis[filepath]["async_function_count"] += cached_data['async_function_count']
                        file_analysis[filepath]["await_statement_count"] += cached_data['await_statement_count']

                else:
                    # Find async function definitions
                    async_functions = re.findall(r"async def\s+\w+\s*\(", content)
                    func_count = len(async_functions)
                    async_function_count += func_count

                    # Find await statements
                    await_statements = re.findall(r"await\s+\w+(?:\([^()]*\))?", content) #Improved regex for await
                    await_count = len(await_statements)
                    await_statement_count += await_count

                    if aggregate_by_file:
                        file_analysis[filepath]["async_function_count"] += func_count
                        file_analysis[filepath]["await_statement_count"] += await_count

                    file_details = []
                    if include_file_content:
                        for m, match in zip(re.finditer(r"async def\s+\w+\s*\(", content), async_functions):
                            context = content[max(0, m.start() - context_length):min(len(content), m.end() + context_length)]
                            detail = {
                                "type": "async_function",
                                "match": match,
                                "context": context  # Add context around the match
                            }
                            if include_line_numbers:
                                detail["line_number"] = get_line_number(content, m)
                            file_details.append(detail)

                        for m, match in zip(re.finditer(r"await\s+\w+(?:\([^()]*\))?", content), await_statements):
                            context = content[max(0, m.start() - context_length):min(len(content), m.end() + context_length)]
                            detail = {
                                "type": "await_statement",
                                "match": match,
                                "context": context  # Add context around the match
                            }
                            if include_line_numbers:
                                detail["line_number"] = get_line_number(content, m)
                            file_details.append(detail)

                        detailed_results.extend(file_details)


                    # Update cache
                    if cache_file:
                        cache[filepath] = {
                            'hash': file_hash,
                            'async_function_count': func_count,
                            'await_statement_count': await_count,
                            'detailed_results': file_details if include_file_content else [] #Cache detailed results
                        }
                    logger.info(f"Processed {filepath}")

                files_analyzed += 1

        except FileNotFoundError as e:
            error_message = f"Error processing file {filepath}: File not found - {e}"
            logger.error(error_message)
            errors.append({"file": filepath, "error": error_message, "type": "FileNotFoundError"})
        except PermissionError as e:
            error_message = f"Error processing file {filepath}: Permission denied - {e}"
            logger.error(error_message)
            errors.append({"file": filepath, "error": error_message, "type": "PermissionError"})
        except UnicodeDecodeError as e:
            error_message = f"Error processing file {filepath}: UnicodeDecodeError - {e}"
            logger.error(error_message)
            errors.append({"file": filepath, "error": error_message, "type": "UnicodeDecodeError"})
        except Exception as e:
            error_message = f"Error processing file {filepath}: {e}"
            logger.exception(error_message)
            errors.append({"file": filepath, "error": str(e), "type": "Exception"})

    async def find_python_files(search_directory: str, current_depth: int = 0) -> List[str]:
        python_files = []
        try:
            for root, dirs, files in os.walk(search_directory):
                # Skip virtual environment directories unless include_venv is True
                if not include_venv and os.path.basename(root) == 'venv':
                    dirs[:] = []  # Don't traverse into venv
                    logger.info(f"Skipping venv directory: {root}")
                    continue

                # Enforce maximum depth
                if max_depth is not None and current_depth >= max_depth:
                    logger.info(f"Reached maximum depth of {max_depth} at {root}. Skipping subdirectories.")
                    dirs[:] = []  # Don't traverse into deeper directories
                    continue

                # Skip hidden directories
                if skip_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]

                for file in files:
                    # Skip hidden files
                    if skip_hidden and file.startswith('.'):
                        continue

                    if file.endswith(".py"):
                        filepath = os.path.join(root, file)
                        if matches_exclude_pattern(filepath, exclude_patterns, pattern_type):
                                files_skipped += 1
                                logger.info(f"Skipping file {filepath} due to exclusion pattern.")
                                continue
                        python_files.append(filepath)

                # Increment depth for the next level of directories
                if dirs:
                    current_depth +=1
        except OSError as e:
            logger.error(f"OSError during directory traversal: {e}")
            errors.append({"directory": search_directory, "error": str(e), "type": "OSError"})
        except Exception as e:
            logger.exception(f"Unexpected error during directory traversal: {e}")
            errors.append({"directory": search_directory, "error": str(e), "type": "Exception"})


        return python_files

    python_files = await find_python_files(directory)

    # Process files concurrently using a ThreadPoolExecutor for CPU-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        loop = asyncio.get_event_loop()
        file_processing_tasks = [
            loop.run_in_executor(executor, lambda fp=filepath: asyncio.run(process_file(fp)), filepath)
            for filepath in python_files
        ]

        await asyncio.gather(*file_processing_tasks, return_exceptions=True) # Process files concurrently

    result = {
        "async_function_count": async_function_count,
        "await_statement_count": await_statement_count,
        "files_analyzed": files_analyzed,
        "files_skipped": files_skipped,
        "errors": errors,
    }

    if include_file_content:
        result["detailed_results"] = detailed_results

    if aggregate_by_file:
        result["file_analysis"] = dict(file_analysis)

    # Save the cache
    if cache_file:
        await save_cache(cache_file, cache, errors)

    if report_file:
        if output_format == "json":
            await save_report_json(report_file, result, errors)
        else:
            await save_report_text(report_file, result)


    return result

async def save_report_json(report_file: str, result: Dict[str, Any], errors: List[Dict[str, Any]]) -> None:
    """Saves the analysis report to a JSON file. Handles potential errors gracefully."""
    try:
        with open(report_file, "w") as f:
            json.dump(result, f, indent=4)
        logger.info(f"Report saved to {report_file}")
    except Exception as e:
        error_message = f"Error saving report to {report_file}: {e}"
        logger.error(error_message)
        errors.append({"file": report_file, "error": str(e), "type": "Exception"})

async def save_report_text(report_file: str, result: Dict[str, Any]) -> None:
    """Saves the analysis report to a text file."""
    try:
        with open(report_file, "w") as f:
            f.write("Async Function Analysis Report\n")
            f.write("===============================\n")
            f.write(f"Total Async Functions: {result['async_function_count']}\n")
            f.write(f"Total Await Statements: {result['await_statement_count']}\n")
            f.write(f"Files Analyzed: {result['files_analyzed']}\n")
            f.write(f"Files Skipped: {result['files_skipped']}\n")

            if "file_analysis" in result:
                f.write("\nAnalysis by File:\n")
                for filepath, analysis in result["file_analysis"].items():
                    f.write(f"  {filepath}:\n")
                    f.write(f"    - Async Functions: {analysis['async_function_count']}\n")
                    f.write(f"    - Await Statements: {analysis['await_statement_count']}\n")

            if "detailed_results" in result and result["detailed_results"]:
                f.write("\nDetailed Results:\n")
                for detail in result["detailed_results"]:
                    f.write(f"  Type: {detail['type']}\n")
                    f.write(f"  Match: {detail['match']}\n")
                    f.write(f"  Context: {detail['context']}\n")
                    if "line_number" in detail:
                        f.write(f"  Line Number: {detail['line_number']}\n")
                    f.write("\n")

            if result['errors']:
                f.write("\nErrors:\n")
                for error in result['errors']:
                    f.write(f"  - File: {error.get('file', 'N/A')}\n")
                    f.write(f"    Error: {error['error']}\n")
        logger.info(f"Text report saved to {report_file}")

    except Exception as e:
        logger.error(f"Error saving text report to {report_file}: {e}")

async def load_cache(cache_file: str, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Loads the cache from a JSON file. Handles potential errors gracefully."""
    cache = {}
    try:
        with open(cache_file, "r") as f:
            cache = json.load(f)
        logger.info(f"Cache loaded from {cache_file}")
    except FileNotFoundError:
        logger.info(f"Cache file {cache_file} not found. Starting with an empty cache.")
    except json.JSONDecodeError as e:
        logger.warning(f"Could not decode cache file {cache_file}. Starting with an empty cache. Error: {e}")
        errors.append({"file": cache_file, "error": f"JSONDecodeError: {e}", "type": "JSONDecodeError"})
    except Exception as e:
        logger.error(f"Could not read cache file {cache_file}. Starting with an empty cache. Error: {e}")
        errors.append({"file": cache_file, "error": str(e), "type": "Exception"})
    return cache

async def save_cache(cache_file: str, cache: Dict[str, Any], errors: List[Dict[str, Any]]) -> None:
    """Saves the cache to a JSON file. Handles potential errors gracefully."""
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)
        logger.info(f"Cache saved to {cache_file}")
    except Exception as e:
        error_message = f"Error saving cache to {cache_file}: {e}"
        logger.error(error_message)
        errors.append({"file": cache_file, "error": str(e), "type": "Exception"})