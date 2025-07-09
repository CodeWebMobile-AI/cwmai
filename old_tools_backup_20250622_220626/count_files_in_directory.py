"""
AI-Generated Tool: count_files_in_directory
Description: Count items for files in directory
Generated: 2025-06-15T12:22:34.615491+00:00
Requirements: 
        Tool Name: count_files_in_directory
        Intent: Count items for files in directory
        Expected Parameters: {}
        Category: analytics
        
        The tool should:
        1. Count all items in the system
2. Return total count with breakdown by status/type
3. Include summary statistics
        
"""

from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Dict, Any, List, Tuple
import os
import os.path
import stat
import asyncio
import hashlib
import logging
from functools import lru_cache
from datetime import datetime

from scripts.state_manager import StateManager  # Assuming this exists

"""
Directory File Counter Tool

This module provides functionality to count files in a directory and return
statistics about the items found, including total count and breakdown by type.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

__description__ = "Count items for files in directory"
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "Directory path to count files in (defaults to current directory)",
        "required": False
    },
    "include_hidden": {
        "type": "boolean",
        "description": "Include hidden files and directories in the count",
        "required": False,
        "default": False
    },
    "calculate_hash": {
        "type": "boolean",
        "description": "Calculate MD5 hash for each file (can be slow for large directories)",
        "required": False,
        "default": False
    },
    "max_depth": {
        "type": "integer",
        "description": "Maximum directory depth to traverse (default: 1, only the specified directory)",
        "required": False,
        "default": 1
    },
    "file_extensions": {
        "type": "list",
        "description": "List of file extensions to include in the count (e.g., ['.txt', '.py'])",
        "required": False,
        "default": None  # Count all file types by default
    },
    "min_size": {
        "type": "integer",
        "description": "Minimum file size in bytes to include in the count.",
        "required": False,
        "default": 0
    },
    "max_size": {
        "type": "integer",
        "description": "Maximum file size in bytes to include in the count.",
        "required": False,
        "default": None  # No maximum size by default
    },
    "include_empty_directories": {
        "type": "boolean",
        "description": "Include empty directories in the count.",
        "required": False,
        "default": False
    },
    "modified_within": {
        "type": "integer",
        "description": "Only count files modified within the last specified number of days (integer).  If zero, only count files modified today. Default is None, meaning no restriction.",
        "required": False,
        "default": None
    },
    "created_within": {
        "type": "integer",
        "description": "Only count files created within the last specified number of days (integer).  If zero, only count files created today. Default is None, meaning no restriction.",
        "required": False,
        "default": None
    },
    "report_top_n_largest": {
        "type": "integer",
        "description": "Report the top N largest files.  Defaults to 0 (disabled)",
        "required": False,
        "default": 0
    }
}
__examples__ = [
    {"description": "Count files in current directory", "code": "await count_files_in_directory()"},
    {"description": "Count files in specific directory", "code": "await count_files_in_directory(directory='/path/to/dir')"},
    {"description": "Count files including hidden files", "code": "await count_files_in_directory(directory='/path/to/dir', include_hidden=True)"},
    {"description": "Count files and calculate hashes", "code": "await count_files_in_directory(directory='/path/to/dir', calculate_hash=True)"},
    {"description": "Count files with max depth", "code": "await count_files_in_directory(directory='/path/to/dir', max_depth=3)"},
    {"description": "Count only python and text files", "code": "await count_files_in_directory(directory='/path/to/dir', file_extensions=['.py', '.txt']"},
    {"description": "Count files larger than 1MB", "code": "await count_files_in_directory(directory='/path/to/dir', min_size=1048576)"},
    {"description": "Count empty directories", "code": "await count_files_in_directory(directory='/path/to/dir', include_empty_directories=True)"},
    {"description": "Count files modified within the last week", "code": "await count_files_in_directory(directory='/path/to/dir', modified_within=7)"},
    {"description": "Report the top 5 largest files", "code": "await count_files_in_directory(directory='/path/to/dir', report_top_n_largest=5)"}

]

def validate_parameters(kwargs: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
    """Validates the input parameters."""
    errors = {}
    directory = kwargs.get('directory')
    include_hidden = kwargs.get('include_hidden', False)
    calculate_hash = kwargs.get('calculate_hash', False)
    max_depth = kwargs.get('max_depth', 1)
    file_extensions = kwargs.get('file_extensions', None)
    min_size = kwargs.get('min_size', 0)
    max_size = kwargs.get('max_size')
    include_empty_directories = kwargs.get('include_empty_directories', False)
    modified_within = kwargs.get('modified_within')
    created_within = kwargs.get('created_within')
    report_top_n_largest = kwargs.get('report_top_n_largest', 0)

    if directory is not None and not isinstance(directory, str):
        errors['directory'] = 'Directory must be a string'
    if not isinstance(include_hidden, bool):
        errors['include_hidden'] = 'include_hidden must be a boolean'
    if not isinstance(calculate_hash, bool):
        errors['calculate_hash'] = 'calculate_hash must be a boolean'
    if not isinstance(max_depth, int):
        errors['max_depth'] = 'max_depth must be an integer'
    if file_extensions is not None and not isinstance(file_extensions, list):
        errors['file_extensions'] = 'file_extensions must be a list'
    if file_extensions is not None:
        for ext in file_extensions:
            if not isinstance(ext, str):
                errors['file_extensions'] = 'file_extensions must be a list of strings'
            if not ext.startswith('.'):
                errors['file_extensions'] = 'file_extensions must start with a dot (e.g., .txt)'

    if not isinstance(min_size, int):
        errors['min_size'] = 'min_size must be an integer'
    elif min_size < 0:
        errors['min_size'] = 'min_size must be non-negative'

    if max_size is not None and not isinstance(max_size, int):
        errors['max_size'] = 'max_size must be an integer'
    elif max_size is not None and min_size > max_size:
        errors['max_size'] = 'max_size must be greater than or equal to min_size'

    if not isinstance(include_empty_directories, bool):
        errors['include_empty_directories'] = 'include_empty_directories must be a boolean'
    
    if modified_within is not None and not isinstance(modified_within, int):
        errors['modified_within'] = 'modified_within must be an integer'
    elif modified_within is not None and modified_within < 0:
        errors['modified_within'] = 'modified_within must be non-negative'
    
    if created_within is not None and not isinstance(created_within, int):
        errors['created_within'] = 'created_within must be an integer'
    elif created_within is not None and created_within < 0:
        errors['created_within'] = 'created_within must be non-negative'

    if not isinstance(report_top_n_largest, int):
        errors['report_top_n_largest'] = 'report_top_n_largest must be an integer'
    elif report_top_n_largest < 0:
        errors['report_top_n_largest'] = 'report_top_n_largest must be non-negative'


    if errors:
        return False, errors
    return True, {}


@lru_cache(maxsize=32)  # Cache results for up to 32 different directories
async def count_files_in_directory(**kwargs) -> Dict[str, Any]:
    """
    Count all items in a directory and return statistics.

    Returns:
        Dictionary containing:
        - total: Total count of all items
        - files: Count of regular files
        - directories: Count of directories
        - symlinks: Count of symbolic links
        - other: Count of other item types
        - summary: Human-readable summary string
        - statistics: Dictionary with size statistics
        - file_hashes: Dictionary with file names and their MD5 hashes (if calculate_hash is True)
        - directory: the directory that was scanned.
    """
    is_valid, errors = validate_parameters(kwargs)
    if not is_valid:
        return {'error': 'Invalid parameters', 'details': errors, 'total': 0}

    try:
        state_manager = StateManager()
        state = state_manager.load_state()
        
        # Get directory from kwargs first, then state, then default to current directory
        directory = kwargs.get('directory')
        if not directory:
            directory = state.get('current_directory', os.getcwd())
        
        include_hidden = kwargs.get('include_hidden', False)
        calculate_hash = kwargs.get('calculate_hash', False)
        max_depth = kwargs.get('max_depth', 1)
        file_extensions = kwargs.get('file_extensions', None)
        min_size = kwargs.get('min_size', 0)
        max_size = kwargs.get('max_size')
        include_empty_directories = kwargs.get('include_empty_directories', False)
        modified_within = kwargs.get('modified_within')
        created_within = kwargs.get('created_within')
        report_top_n_largest = kwargs.get('report_top_n_largest', 0)

        directory = os.path.abspath(directory)  # Normalize the path

        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        counts = defaultdict(int)
        size_stats = {
            'total_size': 0,
            'largest_file': {'name': '', 'size': 0, 'path': ''},
            'smallest_file': {'name': '', 'size': float('inf'), 'path': ''},
            'average_size': 0
        }
        file_hashes = {}
        file_count = 0
        empty_directory_count = 0
        directories_with_errors = 0 # Keep track of directories we can't scan
        files_with_hashing_errors = 0 # Keep track of files with hashing errors
        
        top_n_largest_files: List[Dict[str, Any]] = [] # used if report_top_n_largest > 0

        async def process_directory(current_dir: str, depth: int):
            nonlocal counts, size_stats, file_hashes, file_count, empty_directory_count, directories_with_errors, files_with_hashing_errors, top_n_largest_files

            if depth > max_depth:
                return

            try:
                entries = os.scandir(current_dir)
                entries_list = list(entries) # Consume iterator to check for empty directories before iterating.
            except OSError as e:
                logging.error(f"Error scanning directory {current_dir}: {str(e)}")  # Log error
                directories_with_errors += 1
                return
            
            if not include_empty_directories and not entries_list:  # Check for empty directory
                empty_directory_count +=1
                return

            for entry in entries_list:
                try:
                    if not include_hidden and entry.name.startswith('.'):
                        continue  # Skip hidden files/directories

                    full_path = os.path.join(current_dir, entry.name)

                    if entry.is_file():
                        stat_info = entry.stat()
                        file_size = stat_info.st_size
                        
                        # File Type and Size Filters
                        if file_extensions is not None:
                            _, ext = os.path.splitext(entry.name)
                            if ext.lower() not in file_extensions:
                                continue

                        if file_size < min_size:
                            continue
                        if max_size is not None and file_size > max_size:
                            continue

                        #Date filters
                        now = datetime.now()
                        if modified_within is not None:
                            modified_date = datetime.fromtimestamp(stat_info.st_mtime)
                            days_difference = (now - modified_date).days
                            if days_difference > modified_within:
                                continue

                        if created_within is not None:
                            created_date = datetime.fromtimestamp(stat_info.st_ctime)
                            days_difference = (now - created_date).days
                            if days_difference > created_within:
                                continue


                        counts['files'] += 1
                        
                        size_stats['total_size'] += file_size

                        # Track largest file
                        if file_size > size_stats['largest_file']['size']:
                            size_stats['largest_file'] = {'name': entry.name, 'size': file_size, 'path': full_path}

                        # Track smallest file
                        if file_size < size_stats['smallest_file']['size']:
                            size_stats['smallest_file'] = {'name': entry.name, 'size': file_size, 'path': full_path}
                        
                        file_count += 1

                        if calculate_hash:
                            try:
                                with open(full_path, 'rb') as f:
                                    file_hash = hashlib.md5(f.read()).hexdigest()
                                    file_hashes[entry.name] = file_hash
                            except OSError as hash_err:
                                logging.warning(f"Error calculating hash for {entry.name}: {str(hash_err)}")
                                file_hashes[entry.name] = f"Error calculating hash: {str(hash_err)}"
                                files_with_hashing_errors += 1

                        # Maintain top N largest files, if requested
                        if report_top_n_largest > 0:
                            file_info = {'name': entry.name, 'size': file_size, 'path': full_path}
                            if len(top_n_largest_files) < report_top_n_largest:
                                top_n_largest_files.append(file_info)
                                top_n_largest_files.sort(key=lambda x: x['size'], reverse=True) # Sort descending
                            elif file_size > top_n_largest_files[-1]['size']:
                                top_n_largest_files[-1] = file_info
                                top_n_largest_files.sort(key=lambda x: x['size'], reverse=True)

                    elif entry.is_dir():
                        counts['directories'] += 1
                        await process_directory(full_path, depth + 1)  # Recursive call for subdirectories
                    elif entry.is_symlink():
                        counts['symlinks'] += 1
                    else:
                        counts['other'] += 1
                except OSError as entry_err:
                    logging.error(f"Error processing entry {entry.name}: {entry_err}")  # Log the error, don't stop the whole process
                    continue  # Go to next entry
        
        await process_directory(directory, 1) #start at depth 1
        

        # Calculate average if there are files
        if file_count > 0:
            size_stats['average_size'] = size_stats['total_size'] / file_count
        else:
            size_stats['smallest_file'] = {'name': 'N/A', 'size': 0, 'path': 'N/A'}
            size_stats['largest_file'] = {'name': 'N/A', 'size': 0, 'path': 'N/A'}
        
        total_items = sum(counts.values())
        summary = (
            f"Found {total_items} items in {directory} (max_depth={max_depth}): "
            f"{counts['files']} files, "
            f"{counts['directories']} directories, "
            f"{counts['symlinks']} symlinks, "
            f"{counts['other']} other items."
            f" Empty directories skipped: {empty_directory_count}."
            f" Directories with errors: {directories_with_errors}."
            f" Files with hashing errors: {files_with_hashing_errors}."
        )
        
        result = {
            'total': total_items,
            'files': counts['files'],
            'directories': counts['directories'],
            'symlinks': counts['symlinks'],
            'other': counts['other'],
            'summary': summary,
            'statistics': size_stats,
            'directory': directory,
            'max_depth': max_depth,
            'empty_directories_skipped': empty_directory_count,
            'directories_with_errors': directories_with_errors,
            'files_with_hashing_errors': files_with_hashing_errors
        }

        if calculate_hash:
            result['file_hashes'] = file_hashes
        
        if report_top_n_largest > 0:
            result['top_n_largest_files'] = top_n_largest_files

        return result
        
    except FileNotFoundError as fnf_error:
        logging.error(f"FileNotFoundError: {fnf_error}")
        return {
            'error': str(fnf_error),
            'total': 0,
            'files': 0,
            'directories': 0,
            'symlinks': 0,
            'other': 0,
            'summary': f"Error counting files: {str(fnf_error)}",
            'statistics': {
                'total_size': 0,
                'largest_file': {'name': 'N/A', 'size': 0, 'path': 'N/A'},
                'smallest_file': {'name': 'N/A', 'size': 0, 'path': 'N/A'},
                'average_size': 0
            }
        }
    except NotADirectoryError as nad_error:
        logging.error(f"NotADirectoryError: {nad_error}")
        return {
            'error': str(nad_error),
            'total': 0,
            'files': 0,
            'directories': 0,
            'symlinks': 0,
            'other': 0,
            'summary': f"Error counting files: {str(nad_error)}",
            'statistics': {
                'total_size': 0,
                'largest_file': {'name': 'N/A', 'size': 0, 'path': 'N/A'},
                'smallest_file': {'name': 'N/A', 'size': 0, 'path': 'N/A'},
                'average_size': 0
            }
        }
    except PermissionError as perm_error:
        logging.error(f"PermissionError: {perm_error}")
        return {
            'error': f"Permission error: {str(perm_error)}.  Ensure the tool has read access to the directory.",
            'total': 0,
            'files': 0,
            'directories': 0,
            'symlinks': 0,
            'other': 0,
            'summary': f"Error counting files: {str(perm_error)}",
            'statistics': {
                'total_size': 0,
                'largest_file': {'name': 'N/A', 'size': 0, 'path': 'N/A'},
                'smallest_file': {'name': 'N/A', 'size': 0, 'path': 'N/A'},
                'average_size': 0
            }
        }
    except Exception as e:
        logging.exception("An unexpected error occurred:") # Log full exception including stack trace
        return {
            'error': str(e),
            'total': 0,
            'files': 0,
            'directories': 0,
            'symlinks': 0,
            'other': 0,
            'summary': f"Error counting files: {str(e)}",
            'statistics': {
                'total_size': 0,
                'largest_file': {'name': 'N/A', 'size': 0, 'path': 'N/A'},
                'smallest_file': {'name': 'N/A', 'size': 0, 'path': 'N/A'},
                'average_size': 0
            }
        }