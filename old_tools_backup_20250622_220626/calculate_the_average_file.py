"""
AI-Generated Tool: calculate_the_average_file
Description: Calculates the average file size within each directory of a given directory structure.
Generated: 2024-02-29
Requirements: Recursive directory traversal, file size retrieval, directory grouping, average calculation, error handling.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

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
    },
    "cache_file": {
        "type": "string",
        "description": "Path to the cache file. If provided, results are cached and reused.",
        "required": False,
        "default": None
    },
    "ignore_errors": {
        "type": "boolean",
        "description": "If true, ignore errors when accessing file sizes and directories.",
        "required": False,
        "default": False
    },
    "output_format":{
        "type":"string",
        "description":"Format the output (json or list). Default is list",
        "required":False,
        "default": "list"
    },
     "min_file_size": {
        "type": "integer",
        "description": "Minimum file size (in bytes) to include in the average calculation. Defaults to 0.",
        "required": False,
        "default": 0
    },
    "include_hidden": {
        "type": "boolean",
        "description": "If true, include hidden files and directories in the analysis. Defaults to False.",
        "required": False,
        "default": False
    },
    "log_file": {
        "type": "string",
        "description": "Path to the log file.  If provided, errors and warnings are logged.",
        "required": False,
        "default": None
    },
    "report_empty_dirs": {
        "type": "boolean",
        "description": "If true, reports directories that have no files matching the size criteria.",
        "required": False,
        "default": False
    },
    "delete_cache": {
        "type": "boolean",
        "description": "If true, delete the cache file before execution. Useful for ensuring fresh results.",
        "required": False,
        "default": False
    },
    "clear_log": {
        "type": "boolean",
        "description": "If true, clears the log file before execution.  Requires log_file to be specified.",
        "required": False,
        "default": False
    },
    "include_symlinks": {
        "type": "boolean",
        "description": "If true, include symlinks in the analysis.  Defaults to False, which means symlinks are skipped.",
        "required": False,
        "default": False
    },
        "exclude_patterns": {
        "type": "string",
        "description": "Comma-separated list of filename patterns to exclude (e.g., '*.txt,*.log').",
        "required": False,
        "default": None
    }

}
__examples__ = [
    {"description": "Calculate average file size in the current directory.", "code": "await calculate_the_average_file()"},
    {"description": "Calculate average file size in a specific directory.", "code": "await calculate_the_average_file(directory='path/to/directory')"},
    {"description": "Calculate average file size with a maximum depth of 2.", "code": "await calculate_the_average_file(directory='path/to/directory', max_depth=2)"},
    {"description": "Calculate average file size with caching.", "code": "await calculate_the_average_file(cache_file='cache.json')"},
    {"description": "Calculate average file size, ignoring errors.", "code": "await calculate_the_average_file(ignore_errors=True)"},
    {"description": "Calculate average file size in JSON format.", "code":"await calculate_the_average_file(output_format='json')"},
    {"description": "Calculate average file size, ignoring files smaller than 1024 bytes.", "code": "await calculate_the_average_file(min_file_size=1024)"},
    {"description": "Calculate average file size, including hidden files and directories.", "code": "await calculate_the_average_file(include_hidden=True)"},
    {"description": "Calculate average file size and log errors to a file.", "code": "await calculate_the_average_file(log_file='error.log')"},
    {"description": "Report empty directories.", "code": "await calculate_the_average_file(report_empty_dirs=True)"},
    {"description": "Delete the cache file before execution.", "code": "await calculate_the_average_file(cache_file='cache.json', delete_cache=True)"},
    {"description": "Clear the log file before execution.", "code": "await calculate_the_average_file(log_file='error.log', clear_log=True)"},
    {"description": "Include symlinks in the analysis.", "code": "await calculate_the_average_file(include_symlinks=True)"},
    {"description": "Exclude files matching certain patterns.", "code": "await calculate_the_average_file(exclude_patterns='*.txt,*.log')"}
]


async def calculate_the_average_file(**kwargs) -> Dict[str, Any]:
    """Calculates the average file size within each directory of a given directory structure."""
    directory: str = kwargs.get('directory', '.')
    max_depth: int = kwargs.get('max_depth', -1)
    cache_file: Optional[str] = kwargs.get('cache_file')
    ignore_errors: bool = kwargs.get('ignore_errors', False)
    output_format: str = kwargs.get('output_format', 'list')
    min_file_size: int = kwargs.get('min_file_size', 0)
    include_hidden: bool = kwargs.get('include_hidden', False)
    log_file: Optional[str] = kwargs.get('log_file')
    report_empty_dirs: bool = kwargs.get('report_empty_dirs', False)
    delete_cache: bool = kwargs.get('delete_cache', False)
    clear_log: bool = kwargs.get('clear_log', False)
    include_symlinks: bool = kwargs.get('include_symlinks', False)
    exclude_patterns: Optional[str] = kwargs.get('exclude_patterns')

    # Input validation
    if not isinstance(directory, str):
        return {"error": "directory must be a string."}
    if not isinstance(max_depth, int):
        return {"error": "max_depth must be an integer."}
    if cache_file is not None and not isinstance(cache_file, str):
        return {"error": "cache_file must be a string or None."}
    if not isinstance(ignore_errors, bool):
        return {"error": "ignore_errors must be a boolean."}
    if not isinstance(output_format, str):
        return {"error": "output_format must be a string."}
    if not isinstance(min_file_size, int) or min_file_size < 0:
        return {"error": "min_file_size must be a non-negative integer."}
    if not isinstance(include_hidden, bool):
        return {"error": "include_hidden must be a boolean."}
    if log_file is not None and not isinstance(log_file, str):
        return {"error": "log_file must be a string or None."}
    if not isinstance(report_empty_dirs, bool):
        return {"error": "report_empty_dirs must be a boolean."}
    if not isinstance(delete_cache, bool):
        return {"error": "delete_cache must be a boolean."}
    if not isinstance(clear_log, bool):
        return {"error": "clear_log must be a boolean."}
    if not isinstance(include_symlinks, bool):
        return {"error": "include_symlinks must be a boolean."}
    if exclude_patterns is not None and not isinstance(exclude_patterns, str):
        return {"error": "exclude_patterns must be a string or None."}


    if output_format not in ['list','json']:
        return{"error": f"Invalid output format.  Choose 'list' or 'json'."}

    def log_message(message: str, level: str = 'INFO'):
        """Logs a message to the console and/or a log file."""
        log_string = f"[{datetime.now().isoformat()}] [{level}] {message}"
        print(log_string)
        if log_file:
            try:
                with open(log_file, 'a') as f:
                    f.write(log_string + '\n')
            except Exception as e:
                print(f"Error writing to log file: {e}")

    def matches_exclude_pattern(filename: str, exclude_patterns: Optional[str]) -> bool:
        """Checks if a filename matches any of the exclude patterns."""
        if not exclude_patterns:
            return False

        import fnmatch
        patterns = [p.strip() for p in exclude_patterns.split(',')]
        for pattern in patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    try:
        path = Path(directory)
        if not path.exists():
            return {"error": f"Directory not found: {directory}"}

        if not path.is_dir():
            return {"error": f"Not a directory: {directory}"}

        # Caching
        cache_key = {'directory': directory, 'max_depth': max_depth, 'ignore_errors': ignore_errors, 'min_file_size': min_file_size, 'include_hidden':include_hidden, 'report_empty_dirs': report_empty_dirs, 'include_symlinks': include_symlinks, 'exclude_patterns': exclude_patterns}
        if cache_file:
            cache_path = Path(cache_file)

            if delete_cache and cache_path.exists():
                try:
                    os.remove(cache_path)
                    log_message(f"Cache file '{cache_file}' deleted.", 'INFO')
                except OSError as e:
                    log_message(f"Error deleting cache file '{cache_file}': {e}", 'WARNING')


            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        cached_data = json.load(f)
                    #check if the parameters used to calculate the cached data are identical to the ones given now.
                    if cached_data.get('parameters') == cache_key:
                        log_message("Using cached data.")
                        return cached_data.get('results')
                    else:
                        log_message('Cache parameters do not match current parameters. Recalculating...', 'WARNING')
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    log_message(f"Error loading cache: {e}. Recalculating...", 'WARNING')

        # Log Clearing
        if clear_log and log_file:
            log_path = Path(log_file)
            if log_path.exists():
                try:
                    with open(log_file, 'w'):  # Open in write mode to clear the file
                        pass  # Do nothing, just open and close
                    log_message(f"Log file '{log_file}' cleared.", 'INFO')
                except OSError as e:
                    log_message(f"Error clearing log file '{log_file}': {e}", 'WARNING')




        directory_sizes: Dict[str, Dict[str, Any]] = {}
        empty_directories: List[str] = []

        def traverse_directory(current_path: Path, depth: int) -> None:
            """Recursively traverses the directory structure."""
            nonlocal empty_directories  # Allow modification of outer scope variable

            if max_depth != -1 and depth > max_depth:
                return

            files_in_dir: List[int] = []
            try:
                items = list(current_path.iterdir())  # List to handle potential changes during iteration
                for item in items:
                    if not include_hidden and item.name.startswith('.'):
                        continue

                    if item.is_symlink() and not include_symlinks:
                        continue
                    
                    if matches_exclude_pattern(item.name, exclude_patterns):
                        continue


                    if item.is_file():
                        try:
                            file_size = item.stat().st_size
                            if file_size >= min_file_size:
                                files_in_dir.append(file_size)
                        except Exception as e:
                            if ignore_errors:
                                log_message(f"Error getting file size for {item}: {e}. Ignoring...", 'WARNING')
                            else:
                                log_message(f"Error getting file size for {item}: {e}", 'ERROR')
                                raise  # Re-raise the exception if ignore_errors is False
                    elif item.is_dir():
                         traverse_directory(item, depth + 1)  # Recursive call
            except OSError as e:
                log_message(f"Error accessing directory {current_path}: {e}. Ignoring...", 'WARNING')
                if not ignore_errors:
                    raise #Re-raise the exception if ignore_errors is False
                return

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
                if report_empty_dirs:
                    empty_directories.append(str(current_path.absolute()))


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
        if report_empty_dirs:
            summary += f" {len(empty_directories)} empty directories found."

        output = {
            "results": results,
            "summary": summary,
            "analysis_time": datetime.now().isoformat(), #More comprehensive output
            "parameters_used": cache_key, # Include parameters used in the output
            "empty_directories": empty_directories if report_empty_dirs else [],
            "directory_count": len(directory_sizes)
        }

        # Caching (save)
        if cache_file:
            try:
                with open(cache_path, 'w') as f:
                    json.dump({
                        'results': output,
                        'parameters': cache_key,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=4)
                log_message(f"Results cached to {cache_file}")
            except Exception as e:
                log_message(f"Error writing to cache: {e}", 'ERROR')
        if output_format == 'json':
            return json.dumps(output, indent=4)
        else:
            return output

    except Exception as e:
        log_message(f"Error calculating average file sizes: {str(e)}", 'ERROR')
        return {"error": f"Error calculating average file sizes: {str(e)}"}