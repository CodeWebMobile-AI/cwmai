"""
AI-Generated Tool: show_me_files_changed_today
Description: Show me files changed today
Generated: 2025-06-15T12:16:19.345744+00:00
Requirements: list all files modified in the last 24 hours
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from typing import Dict, List
from typing import List

import datetime
import os
import time
import json
import logging
import fnmatch
import shutil
import traceback  # Import traceback

from scripts.state_manager import StateManager


"""File Change Tracker Module

This module provides functionality to list all files modified in the last 24 hours.
It scans the current directory and all subdirectories recursively to find files
with modification timestamps within the last day.
"""


__description__ = "Show me files changed today"
__parameters__ = {
    "directory": {
        "type": "str",
        "required": False,
        "default": ".",
        "description": "Directory path to search (default: current directory)"
    },
    "recursive": {
        "type": "bool",
        "required": False,
        "default": True,
        "description": "Whether to search subdirectories recursively"
    },
    "extensions": {
        "type": "list",
        "required": False,
        "default": [],
        "description": "List of file extensions to filter (e.g., ['.txt', '.py'])"
    },
    "exclude_patterns": {
        "type": "list",
        "required": False,
        "default": [],
        "description": "List of file name patterns to exclude (e.g., ['*temp*', '.*.log'])"
    },
    "include_hidden": {
        "type": "bool",
        "required": False,
        "default": False,
        "description": "Whether to include hidden files and directories (starting with '.')"
    },
    "cache_duration": {
        "type": "int",
        "required": False,
        "default": 60,
        "description": "Duration in seconds to cache results (0 to disable)"
    },
    "output_format":{
        "type": "str",
        "required": False,
        "default": "json",
        "description": "Format of the output ('json' or 'human_readable')"
    },
    "date_format":{
        "type": "str",
        "required": False,
        "default": "%Y-%m-%d %H:%M:%S",
        "description": "Format for displaying the modified date."
    },
    "min_size_kb": {
        "type": "int",
        "required": False,
        "default": 0,
        "description": "Minimum file size in KB to include in the results."
    },
    "max_size_kb": {
        "type": "int",
        "required": False,
        "default": 0,
        "description": "Maximum file size in KB to include in the results. 0 means no limit."
    },
    "sort_by": {
        "type": "str",
        "required": False,
        "default": "path",
        "description": "Sort the results by 'path', 'modified', or 'size'.",
    },
    "sort_order": {
        "type": "str",
        "required": False,
        "default": "ascending",
        "description": "Sort order: 'ascending' or 'descending'.",
    },
    "destination_dir": {
        "type": "str",
        "required": False,
        "default": None,
        "description": "Destination directory to copy the found files to. If None, files are not copied.",
    },
    "hash_algorithm": {
        "type": "str",
        "required": False,
        "default": None,
        "description": "Hash algorithm to use for file hashing (e.g., 'md5', 'sha256'). If None, no hashing is performed."
    },
    "hash_output_format": {
        "type": "str",
        "required": False,
        "default": "hex",
        "description": "Format for the hash output ('hex' or 'base64').  Only applies if hash_algorithm is set."
    },
    "symlink_handling": {
        "type": "str",
        "required": False,
        "default": "ignore",
        "description": "How to handle symbolic links: 'ignore', 'follow', or 'report'.  'ignore' skips symlinks. 'follow' treats them as normal files. 'report' includes them in the results with a flag.",
    },
    "report_empty_dirs": {
        "type": "bool",
        "required": False,
        "default": False,
        "description": "Whether to report empty directories that have been modified in the last 24 hours."
    },
    "size_unit": {
        "type": "str",
        "required": False,
        "default": "KB",
        "description": "Unit for displaying file size ('B', 'KB', 'MB', 'GB').",
    }

}
__examples__ = [
    {"description": "Find changed files in current directory",
     "code": "await show_me_files_changed_today()"},
    {"description": "Find changed files in specific directory non-recursively",
     "code": "await show_me_files_changed_today(directory='/path/to/dir', recursive=False)"},
    {"description": "Find changed python files in directory",
     "code": "await show_me_files_changed_today(directory='/path/to/dir', extensions=['.py'])"},
    {"description": "Find changed files except temp files",
     "code": "await show_me_files_changed_today(directory='/path/to/dir', exclude_patterns=['*temp*'])"},
    {"description": "Find changed files, including hidden files",
     "code": "await show_me_files_changed_today(include_hidden=True)"},
    {"description": "Find changed files and output in human readable format",
     "code": "await show_me_files_changed_today(output_format='human_readable')"},
    {"description": "Find changed files larger than 1MB",
     "code": "await show_me_files_changed_today(min_size_kb=1024)"},
    {"description": "Find changed files smaller than 500KB",
     "code": "await show_me_files_changed_today(max_size_kb=500)"},
    {"description": "Find changed files with a custom date format",
     "code": "await show_me_files_changed_today(date_format='%m/%d/%Y %I:%M %p')"},
    {"description": "Find changed files sorted by size in descending order",
     "code": "await show_me_files_changed_today(sort_by='size', sort_order='descending')"},
    {"description": "Find changed files and copy them to a destination directory",
     "code": "await show_me_files_changed_today(destination_dir='/path/to/destination')"},
    {"description": "Find changed files and calculate their MD5 hash",
     "code": "await show_me_files_changed_today(hash_algorithm='md5')"},
    {"description": "Find changed files and calculate their SHA256 hash in base64 format",
     "code": "await show_me_files_changed_today(hash_algorithm='sha256', hash_output_format='base64')"},
    {"description": "Find changed files and report symbolic links",
     "code": "await show_me_files_changed_today(symlink_handling='report')"},
    {"description": "Find changed files, follow symbolic links",
     "code": "await show_me_files_changed_today(symlink_handling='follow')"},
    {"description": "Find changed files and report empty directories modified in last 24 hours",
     "code": "await show_me_files_changed_today(report_empty_dirs=True)"},
    {"description": "Find changed files and display size in MB",
     "code": "await show_me_files_changed_today(size_unit='MB')"},
]


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import hashlib
import base64


async def show_me_files_changed_today(**kwargs) -> Dict[str, List[Dict[str, str]]]:
    """List all files modified in the last 24 hours.
    
    Args:
        directory (str, optional): Directory to search. Defaults to current directory.
        recursive (bool, optional): Search subdirectories. Defaults to True.
        extensions (list, optional): List of file extensions to filter. Defaults to [].
        exclude_patterns (list, optional): List of filename patterns to exclude. Defaults to [].
        include_hidden (bool, optional): Whether to include hidden files. Defaults to False.
        cache_duration (int, optional): Duration in seconds to cache the results. Defaults to 60.
        output_format (str, optional): Format of the output ('json' or 'human_readable'). Defaults to 'json'.
        date_format (str, optional): Format of the modified date in the output. Defaults to '%Y-%m-%d %H:%M:%S'.
        min_size_kb (int, optional): Minimum file size in KB. Defaults to 0.
        max_size_kb (int, optional): Maximum file size in KB. Defaults to 0 (no limit).
        sort_by (str, optional): Sort results by 'path', 'modified', or 'size'. Defaults to 'path'.
        sort_order (str, optional): Sort order 'ascending' or 'descending'. Defaults to 'ascending'.
        destination_dir (str, optional): Destination directory to copy the found files to. Defaults to None.
        hash_algorithm (str, optional): Algorithm for hashing files (e.g., 'md5', 'sha256'). Defaults to None (no hashing).
        hash_output_format (str, optional): Format for the hash output ('hex' or 'base64'). Defaults to 'hex'.
        symlink_handling (str, optional): How to handle symbolic links ('ignore', 'follow', 'report'). Defaults to 'ignore'.
        report_empty_dirs (bool, optional): Whether to report empty directories. Defaults to False.
        size_unit (str, optional): Unit for file size ('B', 'KB', 'MB', 'GB'). Defaults to 'KB'.

    Returns:
        Dict: Dictionary containing list of files with their paths and modification times.
              Example: {"files": [{"path": "/path/to/file", "modified": "2023-01-01 12:00:00", "size_kb": 1024, "hash": "...", "is_symlink": True}], "count": 1, "copy_results": {"copied": 2, "failed": 0}}
    
    Raises:
        ValueError: If the specified directory doesn't exist or invalid input.
        OSError: If encountering an OS-level error during file system access.
        PermissionError: If lacking permissions to access files or directories.
    """
    state_manager = StateManager()
    state = state_manager.load_state()  # Load persistent state from StateManager

    # Get parameters with defaults
    directory = kwargs.get('directory', '.')
    recursive = kwargs.get('recursive', True)
    extensions = kwargs.get('extensions', [])
    exclude_patterns = kwargs.get('exclude_patterns', [])
    include_hidden = kwargs.get('include_hidden', False)
    cache_duration = kwargs.get('cache_duration', 60)
    output_format = kwargs.get('output_format', 'json')
    date_format = kwargs.get('date_format', '%Y-%m-%d %H:%M:%S')
    min_size_kb = kwargs.get('min_size_kb', 0)
    max_size_kb = kwargs.get('max_size_kb', 0)
    sort_by = kwargs.get('sort_by', 'path')
    sort_order = kwargs.get('sort_order', 'ascending')
    destination_dir = kwargs.get('destination_dir')
    hash_algorithm = kwargs.get('hash_algorithm')
    hash_output_format = kwargs.get('hash_output_format', 'hex')
    symlink_handling = kwargs.get('symlink_handling', 'ignore')
    report_empty_dirs = kwargs.get('report_empty_dirs', False)
    size_unit = kwargs.get('size_unit', 'KB').upper()

    # Input Validation
    if not isinstance(directory, str):
        raise ValueError("Directory must be a string.")
    if not isinstance(recursive, bool):
        raise ValueError("Recursive must be a boolean.")
    if not isinstance(extensions, list):
        raise ValueError("Extensions must be a list.")
    if not isinstance(exclude_patterns, list):
        raise ValueError("Exclude patterns must be a list.")
    if not isinstance(include_hidden, bool):
        raise ValueError("include_hidden must be a boolean.")
    if not isinstance(cache_duration, int):
        raise ValueError("Cache duration must be an integer.")
    if cache_duration < 0:
        raise ValueError("Cache duration must be non-negative.")
    if not isinstance(output_format, str):
        raise ValueError("Output format must be a string.")
    if output_format not in ['json', 'human_readable']:
        raise ValueError("Output format must be 'json' or 'human_readable'.")
    if not isinstance(date_format, str):
        raise ValueError("Date format must be a string.")
    try:
        datetime.datetime.strptime("2024-01-01 00:00:00", date_format)
    except ValueError:
        raise ValueError("Invalid date format.")

    if not isinstance(min_size_kb, int):
        raise ValueError("Min size must be an integer")
    if not isinstance(max_size_kb, int):
        raise ValueError("Max size must be an integer")
    if min_size_kb < 0:
        raise ValueError("Min size cannot be negative.")
    if max_size_kb < 0:
        raise ValueError("Max size cannot be negative.")
    if min_size_kb > max_size_kb and max_size_kb != 0:
        raise ValueError("Min size cannot be greater than max size.")
    if not isinstance(sort_by, str):
        raise ValueError("Sort by must be a string.")
    if sort_by not in ['path', 'modified', 'size']:
        raise ValueError("Sort by must be 'path', 'modified', or 'size'.")
    if not isinstance(sort_order, str):
        raise ValueError("Sort order must be a string.")
    if sort_order not in ['ascending', 'descending']:
        raise ValueError("Sort order must be 'ascending' or 'descending'.")
    if destination_dir is not None and not isinstance(destination_dir, str):
        raise ValueError("Destination directory must be a string or None.")

    if hash_algorithm is not None and not isinstance(hash_algorithm, str):
        raise ValueError("Hash algorithm must be a string or None.")

    if hash_algorithm:
        try:
            hashlib.new(hash_algorithm)  # Test if the algorithm is available
        except ValueError:
            raise ValueError(f"Invalid hash algorithm: {hash_algorithm}")

        if not isinstance(hash_output_format, str):
            raise ValueError("Hash output format must be a string.")
        if hash_output_format not in ["hex", "base64"]:
            raise ValueError("Hash output format must be 'hex' or 'base64'.")
    if not isinstance(symlink_handling, str):
        raise ValueError("Symlink handling must be a string.")
    if symlink_handling not in ["ignore", "follow", "report"]:
        raise ValueError("Symlink handling must be 'ignore', 'follow', or 'report'.")
    if not isinstance(report_empty_dirs, bool):
        raise ValueError("report_empty_dirs must be a boolean.")
    if not isinstance(size_unit, str):
        raise ValueError("size_unit must be a string.")
    if size_unit not in ['B', 'KB', 'MB', 'GB']:
        raise ValueError("size_unit must be 'B', 'KB', 'MB', or 'GB'.")


    # Validate directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    # Cache Management (using StateManager for persistence)
    cache_key = f"file_changes:{directory}:{recursive}:{extensions}:{exclude_patterns}:{include_hidden}:{date_format}:{min_size_kb}:{max_size_kb}:{sort_by}:{sort_order}:{hash_algorithm}:{hash_output_format}:{symlink_handling}:{report_empty_dirs}:{size_unit}"

    cached_data = state.get(cache_key)

    if cache_duration > 0 and cached_data:
        last_cached_time = cached_data.get("timestamp", 0)
        if time.time() - last_cached_time <= cache_duration:
            logging.info("Returning cached results.")
            if output_format == 'human_readable':
                return convert_to_human_readable(cached_data["data"], size_unit)
            else:
                return cached_data["data"]
        else:
            logging.info("Cache expired, refreshing...")


    # Calculate cutoff time (24 hours ago)
    cutoff_time = time.time() - 24 * 60 * 60
    
    # Prepare results container
    changed_files = []
    changed_dirs = []  # List to store changed empty directories
    
    # Walk through directory
    walk = os.walk(directory, followlinks=(symlink_handling == "follow")) if recursive else [next(os.walk(directory, followlinks=(symlink_handling == "follow")))]
    
    for root, dirs, files in walk:
        # Skip hidden directories if not include_hidden
        if not include_hidden and os.path.basename(root).startswith('.'):
            continue

        # Check if the directory itself was modified
        try:
            dir_mod_time = os.path.getmtime(root)
            if dir_mod_time >= cutoff_time and report_empty_dirs and not files and not dirs:
                changed_dirs.append({
                    "path": root,
                    "modified": datetime.datetime.fromtimestamp(dir_mod_time).strftime(date_format),
                })
        except (OSError, PermissionError) as e:
            logging.warning(f"Skipping directory {root}: {e}")
            logging.debug(traceback.format_exc())
        except Exception as e:
            logging.error(f"Unexpected error processing directory {root}: {e}")
            logging.debug(traceback.format_exc())


        for filename in files:
            # Skip hidden files if not include_hidden
            if not include_hidden and filename.startswith('.'):
                continue

            filepath = os.path.join(root, filename)
            try:
                # Apply filters before checking modification time
                if extensions and not any(filepath.endswith(ext) for ext in extensions):
                    continue
                if exclude_patterns and any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns):
                    continue

                # Symlink Handling
                is_symlink = os.path.islink(filepath)
                if is_symlink and symlink_handling == "ignore":
                    continue  # Skip symlinks if set to ignore

                file_size_bytes = os.path.getsize(filepath)
                file_size_kb = file_size_bytes / 1024
                if min_size_kb > 0 and file_size_kb < min_size_kb:
                    continue
                if max_size_kb > 0 and file_size_kb > max_size_kb:
                    continue
                
                mod_time = os.path.getmtime(filepath)
                if mod_time >= cutoff_time:
                    file_info = {
                        "path": filepath,
                        "modified": datetime.datetime.fromtimestamp(mod_time).strftime(date_format),
                        "size_bytes": file_size_bytes, # Store size in bytes for accurate unit conversion
                        "is_symlink": is_symlink
                    }

                    if is_symlink and symlink_handling == "report":
                        file_info["symlink_target"] = os.readlink(filepath)


                    # Calculate hash if requested
                    if hash_algorithm:
                        try:
                            file_hash = calculate_file_hash(filepath, hash_algorithm, hash_output_format)
                            file_info["hash"] = file_hash
                        except Exception as e:
                            logging.warning(f"Failed to calculate hash for {filepath}: {e}")
                            logging.debug(traceback.format_exc())
                            file_info["hash"] = None # Indicate failure to generate the hash

                    changed_files.append(file_info)

            except (OSError, PermissionError) as e:
                logging.warning(f"Skipping {filepath}: {e}")  # Log the error
                logging.debug(traceback.format_exc())  # Log the full traceback
            except Exception as e:
                logging.error(f"Unexpected error processing {filepath}: {e}")
                logging.debug(traceback.format_exc()) # Log the full traceback


    # Sort the results
    changed_files = sort_files(changed_files, sort_by, sort_order, date_format)

    result = {"files": changed_files, "count": len(changed_files)}
    if report_empty_dirs:
        result["empty_directories"] = changed_dirs
        result["empty_directories_count"] = len(changed_dirs)


    # Copy files if destination_dir is provided
    copy_results = None
    if destination_dir:
        try:
            copy_results = copy_files(changed_files, destination_dir)
            result["copy_results"] = copy_results
        except Exception as e:
            logging.error(f"Error during file copying: {e}")
            logging.debug(traceback.format_exc())
            result["copy_results"] = {"copied": 0, "failed": len(changed_files)} # Default if copy fails

    # Cache the results
    if cache_duration > 0:
        state[cache_key] = {"data": result, "timestamp": time.time()}
        try:
            state_manager.save_state(state)  # Persist the updated state
            logging.info("Results cached.")
        except Exception as e:
            logging.error(f"Failed to save state: {e}") # log the save failure but continue
            logging.debug(traceback.format_exc())

    if output_format == 'human_readable':
        return convert_to_human_readable(result, size_unit)
    else:
        return result


def convert_bytes(num, unit="KB"):
    """Convert bytes to KB, MB, GB"""
    if unit == "KB":
        return num / 1024
    elif unit == "MB":
        return num / (1024 * 1024)
    elif unit == "GB":
        return num / (1024 * 1024 * 1024)
    elif unit == "B":
        return num
    else:
        return num / 1024 # Default to KB


def convert_to_human_readable(result: Dict[str, List[Dict[str, str]]], size_unit: str) -> str:
    """Converts the JSON result to a human-readable string."""
    output = f"Found {result['count']} files modified in the last 24 hours:\n"
    for file_info in result['files']:
        output += f"- Path: {file_info['path']}\n"
        output += f"  Modified: {file_info['modified']}\n"
        file_size = convert_bytes(file_info['size_bytes'], size_unit)
        output += f"  Size: {file_size:.2f} {size_unit}\n"

        if 'hash' in file_info:
            output += f"  Hash: {file_info['hash']}\n"
        if file_info.get("is_symlink"):
            output += f"  [Symbolic Link]"
            if 'symlink_target' in file_info:
                output += f" -> {file_info['symlink_target']}"
            output += "\n"
    if "empty_directories" in result and result["empty_directories"]:
        output += f"\nFound {result['empty_directories_count']} empty directories modified in the last 24 hours:\n"
        for dir_info in result["empty_directories"]:
            output += f"- Path: {dir_info['path']}\n"
            output += f"  Modified: {dir_info['modified']}\n"


    if 'copy_results' in result:
        output += f"\nCopy Results:\n"
        output += f"  Successfully copied: {result['copy_results']['copied']}\n"
        output += f"  Failed to copy: {result['copy_results']['failed']}\n"

    return output

def sort_files(files: List[Dict[str, any]], sort_by: str, sort_order: str, date_format: str) -> List[Dict[str, any]]:
    """Sorts the list of files based on the specified criteria."""
    reverse = sort_order == 'descending'
    if sort_by == 'path':
        return sorted(files, key=lambda x: x['path'], reverse=reverse)
    elif sort_by == 'modified':
        try:
            return sorted(files, key=lambda x: datetime.datetime.strptime(x['modified'], date_format), reverse=reverse) # Use datetime object for correct sorting
        except ValueError as e:
            logging.error(f"Error parsing date during sorting: {e}. Returning unsorted data.")
            logging.debug(traceback.format_exc())
            return files #Return unsorted if date parsing fails.
    elif sort_by == 'size':
        return sorted(files, key=lambda x: x['size_bytes'], reverse=reverse)
    else:
        return files  # Return unsorted if sort_by is invalid

def copy_files(files: List[Dict[str, str]], destination_dir: str) -> Dict[str, int]:
    """Copies the found files to the specified destination directory.

    Args:
        files: A list of dictionaries, where each dictionary contains the 'path' key.
        destination_dir: The directory to copy the files to.

    Returns:
        A dictionary containing the count of successfully copied files and the count of failed copies.
    """
    copied_count = 0
    failed_count = 0

    # Create the destination directory if it doesn't exist.
    try:
        os.makedirs(destination_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create destination directory: {e}")
        logging.debug(traceback.format_exc())
        return {"copied": 0, "failed": len(files)}

    for file_info in files:
        source_path = file_info['path']
        destination_path = os.path.join(destination_dir, os.path.basename(source_path))

        try:
            shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
            copied_count += 1
        except (OSError, PermissionError) as e:
            logging.error(f"Failed to copy {source_path} to {destination_path}: {e}")
            logging.debug(traceback.format_exc())
            failed_count += 1
        except Exception as e:
            logging.error(f"Unexpected error copying {source_path} to {destination_path}: {e}")
            logging.debug(traceback.format_exc())
            failed_count += 1
    return {"copied": copied_count, "failed": failed_count}

def calculate_file_hash(filepath: str, algorithm: str, output_format: str = "hex") -> str:
    """Calculates the hash of a file using the specified algorithm.

    Args:
        filepath: The path to the file.
        algorithm: The hashing algorithm to use (e.g., 'md5', 'sha256').
        output_format: The format of the output ('hex' or 'base64').

    Returns:
        The hash of the file as a hexadecimal string or base64 encoded string.
    """
    try:
        hasher = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(4096)  # Read in chunks
                if not chunk:
                    break
                hasher.update(chunk)

        if output_format == "hex":
            return hasher.hexdigest()
        elif output_format == "base64":
            return base64.b64encode(hasher.digest()).decode('utf-8')
        else:
            raise ValueError("Invalid hash output format. Must be 'hex' or 'base64'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except OSError as e:
        raise OSError(f"Error reading file: {e}")
    except Exception as e:
        raise Exception(f"Error calculating hash: {e}")