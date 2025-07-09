"""
AI-Generated Tool: count_python_files
Description: Count all Python files in the project
Generated: 2025-06-15T10:59:39.894214+00:00
Requirements: 
        1. Count all .py files in the project
        2. Group by directory
        3. Show total lines of code
        4. Return structured data with counts and statistics
        
"""

"""
Module: count_python_files

Description: Counts all Python files in the project, groups by directory,
and shows the total lines of code. Returns structured data with counts and statistics.
"""

import asyncio
import os
import glob
from typing import Dict, Any, Optional, List, Tuple
from scripts.state_manager import StateManager  # Assuming this exists
from scripts.file_manager import FileManager  # Assuming this exists
from scripts.error_handler import ErrorHandler  # Assuming this exists
import json
import hashlib
import logging
import time
import re

__description__ = "Count all Python files in the project"
__parameters__ = {
    "include_tests": {
        "type": "boolean",
        "description": "Include files in 'test' or 'tests' directories.",
        "default": False
    },
    "min_line_length": {
        "type": "integer",
        "description": "Minimum line length to consider for counting.",
        "default": 0
    },
    "cache_file": {
        "type": "string",
        "description": "Path to the cache file for storing results.",
        "default": ".python_file_count_cache.json"
    },
    "exclude_patterns": {
        "type": "array",
        "description": "List of glob patterns to exclude files/directories.",
        "default": []
    },
    "max_file_size":{
        "type":"integer",
        "description": "Maximum file size in bytes to process.  Files larger than this will be skipped.",
        "default": 1000000 # 1MB Default
    },
    "line_ending_type": {
        "type": "string",
        "description": "Type of line ending to use ('unix', 'windows', or 'auto').",
        "default": "auto"
    },
    "encoding": {
        "type": "string",
        "description": "Encoding to use when reading files.  'auto' will attempt to detect encoding.",
        "default": "auto"
    },
    "ignore_shebang": {
        "type": "boolean",
        "description": "Ignore shebang lines (#!...) when counting lines of code.",
        "default": True
    },
    "file_batch_size": {
        "type": "integer",
        "description": "Number of files to process in each batch.",
        "default": 50
    }
}
__examples__ = [
    "count_python_files",
    "count_python_files include_tests=True",
    "count_python_files min_line_length=10",
    "count_python_files cache_file=my_cache.json",
    "count_python_files exclude_patterns=['**/migrations/*.py', '**/__init__.py']",
    "count_python_files max_file_size=500000",
    "count_python_files line_ending_type=windows",
    "count_python_files encoding=latin-1",
    "count_python_files ignore_shebang=True",
    "count_python_files file_batch_size=100"
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def count_python_files(**kwargs) -> Dict[str, Any]:
    """
    Counts all Python files in the project, groups by directory, and shows the total lines of code.

    Args:
        include_tests (bool, optional): Include files in 'test' or 'tests' directories. Defaults to False.
        min_line_length (int, optional): Minimum line length to consider for counting. Defaults to 0.
        cache_file (str, optional): Path to the cache file. Defaults to ".python_file_count_cache.json".
        exclude_patterns (list, optional): List of glob patterns to exclude files/directories. Defaults to [].
        max_file_size (int, optional):  Maximum file size in bytes to process. Defaults to 1MB.
        line_ending_type (str, optional): Type of line ending to use ('unix', 'windows', or 'auto'). Defaults to 'auto'.
        encoding (str, optional): Encoding to use when reading files. Defaults to 'auto'.
        ignore_shebang (bool, optional): Ignore shebang lines when counting lines of code. Defaults to True.
        file_batch_size (int, optional): Number of files to process in each batch. Defaults to 50.

    Returns:
        A dictionary containing the counts and statistics.  Example:
        {
            "total_files": 5,
            "total_lines": 1234,
            "directory_summary": {
                "path/to/dir1": {"file_count": 2, "line_count": 500},
                "path/to/dir2": {"file_count": 3, "line_count": 734}
            }
        }
    """
    start_time = time.time()

    state_manager = StateManager()  # Assuming this class is defined
    file_manager = FileManager()  # Assuming this class is defined
    error_handler = ErrorHandler()  # Assuming this class is defined

    state = state_manager.load_state()

    include_tests = kwargs.get("include_tests", False)
    min_line_length = kwargs.get("min_line_length", 0)
    cache_file = kwargs.get("cache_file", ".python_file_count_cache.json")
    exclude_patterns = kwargs.get("exclude_patterns", [])
    max_file_size = kwargs.get("max_file_size", 1000000)
    line_ending_type = kwargs.get("line_ending_type", "auto").lower()
    encoding = kwargs.get("encoding", "auto").lower()
    ignore_shebang = kwargs.get("ignore_shebang", True)
    file_batch_size = kwargs.get("file_batch_size", 50)

    # Input Validation
    if not isinstance(include_tests, bool):
        error_handler.log_error("Invalid input: include_tests must be a boolean.")
        return {"error": "Invalid input: include_tests must be a boolean."}
    if not isinstance(min_line_length, int) or min_line_length < 0:
        error_handler.log_error("Invalid input: min_line_length must be a non-negative integer.")
        return {"error": "Invalid input: min_line_length must be a non-negative integer."}
    if not isinstance(cache_file, str):
        error_handler.log_error("Invalid input: cache_file must be a string.")
        return {"error": "Invalid input: cache_file must be a string."}
    if not isinstance(exclude_patterns, list):
        error_handler.log_error("Invalid input: exclude_patterns must be a list.")
        return {"error": "Invalid input: exclude_patterns must be a list."}
    if not isinstance(max_file_size, int) or max_file_size <= 0:
         error_handler.log_error("Invalid input: max_file_size must be a positive integer.")
         return {"error": "Invalid input: max_file_size must be a positive integer."}
    if line_ending_type not in ["unix", "windows", "auto"]:
        error_handler.log_error("Invalid input: line_ending_type must be 'unix', 'windows', or 'auto'.")
        return {"error": "Invalid input: line_ending_type must be 'unix', 'windows', or 'auto'."}

    if not isinstance(encoding, str):
        error_handler.log_error("Invalid input: encoding must be a string.")
        return {"error": "Invalid input: encoding must be a string."}

    if not isinstance(ignore_shebang, bool):
        error_handler.log_error("Invalid input: ignore_shebang must be a boolean.")
        return {"error": "Invalid input: ignore_shebang must be a boolean."}

    if not isinstance(file_batch_size, int) or file_batch_size <= 0:
        error_handler.log_error("Invalid input: file_batch_size must be a positive integer.")
        return {"error": "Invalid input: file_batch_size must be a positive integer."}

    if not validate_exclude_patterns(exclude_patterns):
        error_handler.log_error("Invalid input: exclude_patterns contains invalid glob patterns.")
        return {"error": "Invalid input: exclude_patterns contains invalid glob patterns."}



    # Generate Cache Key
    cache_key = generate_cache_key(include_tests, min_line_length, exclude_patterns, max_file_size, line_ending_type, encoding, ignore_shebang, file_batch_size)

    # Load from Cache
    cached_result = load_from_cache(cache_file, cache_key)
    if cached_result:
        logging.info("Loaded from cache.")
        cached_result["cache_hit"] = True  # Add cache hit indicator
        return cached_result

    try:
        directory_summary: Dict[str, Dict[str, int]] = {}
        total_files = 0
        total_lines = 0
        excluded_files = 0
        skipped_files_size = 0
        error_files = 0
        empty_files = 0
        file_sizes: List[int] = [] # Track file sizes for statistics
        file_paths: List[str] = [] # Track file paths for additional stats.
        file_data: List[Tuple[str, int]] = []  # Store file path and line count


        all_py_files: List[str] = []
        for root, _, files in os.walk("."):  # Start from the current directory

            # Skip test directories if include_tests is False
            if not include_tests and ("test" in root.lower() or "tests" in root.lower()):
                continue

            py_files = [os.path.join(root, f) for f in files if f.endswith(".py")] # full path
            all_py_files.extend(py_files)



        # Batch Processing
        for i in range(0, len(all_py_files), file_batch_size):
            batch_files = all_py_files[i:i + file_batch_size]

            for filepath in batch_files:
                # Check if file should be excluded
                if any(glob.has_match(filepath, pattern) for pattern in exclude_patterns):
                    excluded_files += 1
                    logging.debug(f"Excluded file: {filepath} due to exclude pattern.")
                    continue

                try:
                    file_size = os.path.getsize(filepath)
                    file_sizes.append(file_size) # Add to sizes

                    if file_size > max_file_size:
                        skipped_files_size += 1
                        logging.warning(f"Skipping file {filepath} due to size ({file_size} bytes > {max_file_size} bytes)")
                        continue

                    # Determine encoding
                    effective_encoding = encoding
                    if effective_encoding == "auto":
                        effective_encoding = get_file_encoding(filepath)

                    try:
                        with open(filepath, "r", encoding=effective_encoding) as f:
                            lines = f.readlines()
                    except UnicodeDecodeError as e:
                        error_handler.log_error(f"UnicodeDecodeError reading file {filepath} with encoding {effective_encoding}: {e}.  Consider specifying a different encoding.")
                        logging.error(f"UnicodeDecodeError reading file {filepath} with encoding {effective_encoding}: {e}.  Consider specifying a different encoding.")
                        error_files += 1
                        continue

                    if not lines:
                        empty_files += 1
                        logging.debug(f"Skipping empty file: {filepath}.")
                        continue

                    # Determine line endings
                    if line_ending_type == "auto":
                        if all("\r\n" in line for line in lines):
                            line_ending = "\r\n"
                        else:
                            line_ending = "\n"
                    elif line_ending_type == "windows":
                        line_ending = "\r\n"
                    else:  # "unix"
                        line_ending = "\n"

                    # Normalize line endings
                    normalized_lines = [line.replace("\r\n", line_ending).replace("\r", "") for line in lines]

                    # Filter lines based on minimum length
                    filtered_lines = [line for line in normalized_lines if len(line.strip()) >= min_line_length]

                    # Ignore shebang lines if specified
                    if ignore_shebang and filtered_lines and filtered_lines[0].startswith("#!"):
                        filtered_lines = filtered_lines[1:]

                    line_count = len(filtered_lines)

                    root_dir = os.path.dirname(filepath)  # Get the directory
                    if root_dir not in directory_summary:
                        directory_summary[root_dir] = {"file_count": 0, "line_count": 0}


                    directory_summary[root_dir]["file_count"] += 1
                    directory_summary[root_dir]["line_count"] += line_count
                    total_files += 1
                    total_lines += line_count

                    file_data.append((filepath, line_count))
                    file_paths.append(filepath)  # Keep track of all files processed.

                except FileNotFoundError as e:
                    error_handler.log_error(f"File not found: {filepath}. Skipping. Error: {e}")
                    logging.error(f"File not found: {filepath}. Skipping. Error: {e}")
                    error_files += 1
                except PermissionError as e:
                    error_handler.log_error(f"Permission denied: {filepath}. Skipping. Error: {e}")
                    logging.error(f"Permission denied: {filepath}. Skipping. Error: {e}")
                    error_files += 1
                except Exception as e:
                    error_handler.log_error(f"Error reading file {filepath}: {e}")
                    logging.exception(f"Error reading file {filepath}: {e}") # Log the full exception
                    error_files += 1
                    continue  # Skip to the next file in case of error

        end_time = time.time()
        execution_time = end_time - start_time

        # Calculate file size statistics
        avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
        max_file_size_found = max(file_sizes) if file_sizes else 0
        min_file_size_found = min(file_sizes) if file_sizes else 0

        # Calculate lines per file statistics
        lines_per_file = [data[1] for data in file_data]
        avg_lines_per_file = sum(lines_per_file) / len(lines_per_file) if lines_per_file else 0
        max_lines_per_file = max(lines_per_file) if lines_per_file else 0
        min_lines_per_file = min(lines_per_file) if lines_per_file else 0


        result = {
            "total_files": total_files,
            "total_lines": total_lines,
            "directory_summary": directory_summary,
            "include_tests": include_tests,
            "min_line_length": min_line_length,
            "exclude_patterns": exclude_patterns,
            "excluded_files": excluded_files,
            "skipped_files_size": skipped_files_size,
            "empty_files": empty_files,
            "error_files": error_files,
            "cache_key": cache_key,
            "execution_time": execution_time,
            "max_file_size": max_file_size,
            "line_ending_type": line_ending_type,
            "encoding": encoding,
            "ignore_shebang": ignore_shebang,
            "cache_hit": False,  # Indicate cache miss
            "file_size_stats": {
                "average": avg_file_size,
                "maximum": max_file_size_found,
                "minimum": min_file_size_found,
                "count": len(file_sizes)
            },
             "lines_per_file_stats":{
                "average": avg_lines_per_file,
                "maximum": max_lines_per_file,
                "minimum": min_lines_per_file,
                "count": len(lines_per_file)
             },
            "file_batch_size": file_batch_size
        }

        # Save to Cache
        save_to_cache(cache_file, cache_key, result)

        return result

    except Exception as e:
        error_handler.log_error(f"An unexpected error occurred: {e}")
        logging.exception(f"An unexpected error occurred: {e}")  # Log the full exception
        return {"error": str(e)}

def generate_cache_key(include_tests: bool, min_line_length: int, exclude_patterns: list, max_file_size: int, line_ending_type: str, encoding: str, ignore_shebang: bool, file_batch_size: int) -> str:
    """Generates a unique cache key based on input parameters."""
    data = {
        "include_tests": include_tests,
        "min_line_length": min_line_length,
        "exclude_patterns": exclude_patterns,
        "max_file_size": max_file_size,
        "line_ending_type": line_ending_type,
        "encoding": encoding,
        "ignore_shebang": ignore_shebang,
        "file_batch_size": file_batch_size
    }
    cache_string = json.dumps(data, sort_keys=True).encode('utf-8')
    return hashlib.md5(cache_string).hexdigest()


def load_from_cache(cache_file: str, cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads the result from a cache file, if it exists and the key matches."""
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
            if "cache_entries" in cache_data and cache_key in cache_data["cache_entries"]:
                logging.debug(f"Cache hit for key: {cache_key}")
                return cache_data["cache_entries"][cache_key]
            else:
                logging.debug(f"Cache miss for key: {cache_key}")
                return None  # No matching cache entry

    except FileNotFoundError:
        logging.debug(f"Cache file not found: {cache_file}")
        return None
    except json.JSONDecodeError:
        logging.warning(f"Cache file {cache_file} is corrupted. Ignoring cache.")
        return None  # Corrupted cache

def save_to_cache(cache_file: str, cache_key: str, data: Dict[str, Any]):
    """Saves the result to a cache file, using a cache key."""
    try:
        # Load existing cache (if any)
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cache_data = {"cache_entries": {}}

        # Add/Update the cache entry
        cache_data["cache_entries"][cache_key] = data

        # Save the updated cache
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=4)  # Pretty print the JSON
        logging.debug(f"Saved to cache with key: {cache_key}")


    except Exception as e:
        logging.error(f"Error saving to cache file {cache_file}: {e}")

# Utility function to clear the cache
def clear_cache(cache_file: str):
    """Clears the cache file."""
    try:
        os.remove(cache_file)
        logging.info(f"Cache file '{cache_file}' cleared.")
    except FileNotFoundError:
        logging.info(f"Cache file '{cache_file}' not found.")
    except Exception as e:
        logging.error(f"Error clearing cache file '{cache_file}': {e}")

def validate_exclude_patterns(patterns: list) -> bool:
    """Validates that exclude patterns are valid glob patterns."""
    try:
        for pattern in patterns:
            glob.compile(pattern)  # Try to compile the pattern
        return True
    except Exception:
        return False

def get_file_encoding(filepath: str) -> str:
    """Attempts to detect the encoding of a file."""
    try:
        import chardet
        with open(filepath, 'rb') as f:
            rawdata = f.read()
        result = chardet.detect(rawdata)
        return result['encoding']
    except Exception as e:
        logging.warning(f"Could not detect encoding for {filepath}.  Assuming UTF-8. Error: {e}")
        return "utf-8"  # Default encoding

def print_directory_summary(directory_summary: Dict[str, Dict[str, int]]):
    """Prints a formatted directory summary."""
    print("\nDirectory Summary:")
    for directory, data in directory_summary.items():
        print(f"  Directory: {directory}")
        print(f"    - File Count: {data['file_count']}")
        print(f"    - Line Count: {data['line_count']}")

def get_largest_files(file_data: List[Tuple[str, int]], n: int = 5) -> list:
    """Returns the n largest files (by line count)."""
    sorted_files = sorted(file_data, key=lambda x: x[1], reverse=True)
    return sorted_files[:n]


def display_results(results: Dict[str, Any]):
    """Nicely formats and prints the results."""
    print("\n--- Results ---")
    print(f"Total files: {results['total_files']}")
    print(f"Total lines: {results['total_lines']}")
    print(f"Excluded files: {results['excluded_files']}")
    print(f"Skipped files (size): {results['skipped_files_size']}")
    print(f"Empty files: {results['empty_files']}")
    print(f"Error files: {results['error_files']}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")
    print(f"Cache hit: {results['cache_hit']}")
    print(f"File batch size: {results['file_batch_size']}")


    print_directory_summary(results['directory_summary'])

    print("\nFile Size Statistics:")
    print(f"  Average file size: {results['file_size_stats']['average']:.2f} bytes")
    print(f"  Maximum file size: {results['file_size_stats']['maximum']} bytes")
    print(f"  Minimum file size: {results['file_size_stats']['minimum']} bytes")
    print(f"  File size count: {results['file_size_stats']['count']}")

    print("\nLines per File Statistics:")
    print(f"  Average lines per file: {results['lines_per_file_stats']['average']:.2f} lines")
    print(f"  Maximum lines per file: {results['lines_per_file_stats']['maximum']} lines")
    print(f"  Minimum lines per file: {results['lines_per_file_stats']['minimum']} lines")
    print(f"  Lines per file count: {results['lines_per_file_stats']['count']}")


    # Show largest files
    # Reconstruct file_data for get_largest_files
    file_data = []
    for directory, data in results['directory_summary'].items():
        # Find .py files in the directory and get their line counts
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                    line_count = len(lines)
                    file_data.append((filepath, line_count))
                except Exception as e:
                    logging.warning(f"Could not read file {filepath} to determine line count. Skipping. Error: {e}")
    largest_files = get_largest_files(file_data)


    if largest_files:
        print("\nLargest Files (by line count):")
        for file, line_count in largest_files:
            print(f"  - {file}: {line_count} lines")

    if results['error_files'] > 0:
        print("\nConsider reviewing the error logs for more details on file processing issues.")



# Example Usage (for testing)
if __name__ == '__main__':
    async def main():
        # Example 1: Basic usage
        result = await count_python_files(include_tests=True, min_line_length=5, cache_file="my_cache.json", max_file_size=500000, ignore_shebang=True, file_batch_size=25)
        print("Example 1 Result:")
        #print(json.dumps(result, indent=4))  # Pretty print the JSON output
        display_results(result)

        # Example 2: Run again to test cache
        print("\nRunning again to test cache...")
        result2 = await count_python_files(include_tests=True, min_line_length=5, cache_file="my_cache.json", max_file_size=500000, ignore_shebang=True, file_batch_size=25)
        print("Example 2 Result (should be from cache):")
        #print(json.dumps(result2, indent=4))
        display_results(result2)

        # Example 3: With exclude patterns
        print("\nRunning with exclude patterns...")
        result3 = await count_python_files(include_tests=True, min_line_length=5, cache_file="my_cache.json", exclude_patterns=["**/test_file.py"], max_file_size=500000, ignore_shebang=True, file_batch_size=25)
        print("Example 3 Result:")
        #print(json.dumps(result3, indent=4))
        display_results(result3)

        # Example 4: Clear Cache
        # clear_cache("my_cache.json") # Uncomment to clear the cache before running again

        # Example 5: Test line ending type
        print("\nRunning with specific line ending type (windows)...")
        result4 = await count_python_files(include_tests=True, min_line_length=0, cache_file="my_cache.json", line_ending_type="windows", ignore_shebang=True, file_batch_size=25)
        print("Example 4 Result:")
        #print(json.dumps(result4, indent=4))
        display_results(result4)

        # Example 6: Test encoding
        print("\nRunning with specific encoding (latin-1)...")
        result5 = await count_python_files(include_tests=True, min_line_length=0, cache_file="my_cache.json", encoding="latin-1", ignore_shebang=True, file_batch_size=25)
        print("Example 5 Result:")
        display_results(result5)

        # Example 7: Test ignore_shebang = False
        print("\nRunning with ignore_shebang = False...")
        result6 = await count_python_files(include_tests=True, min_line_length=0, cache_file="my_cache.json", ignore_shebang=False, file_batch_size=25)
        print("Example 6 Result:")
        display_results(result6)

        # Example 8: Test file batch size
        print("\nRunning with file_batch_size = 10...")
        result7 = await count_python_files(include_tests=True, min_line_length=0, cache_file="my_cache.json", ignore_shebang=False, file_batch_size=10)
        print("Example 7 Result:")
        display_results(result7)


    asyncio.run(main())