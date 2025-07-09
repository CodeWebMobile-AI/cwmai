"""
AI-Generated Tool: analyze_file_types_and_size
Description: Analyze data for file types and size
Generated: 2025-06-15T11:07:18.115986+00:00
Requirements: 
        Tool Name: analyze_file_types_and_size
        Intent: Analyze data for file types and size
        Expected Parameters: {}
        Category: analytics
        
        The tool should:
        1. Perform comprehensive analysis of items
2. Generate insights and recommendations
3. Return detailed report
        
"""

"""
This module provides functionality to analyze file types and sizes in a given dataset.
It generates insights and recommendations based on the analysis performed.

Tool Name: analyze_file_types_and_size
Description: Analyze data for file types and size
Category: analytics
"""

import os
import hashlib
import json
from collections import defaultdict
from scripts.state_manager import StateManager  # Assuming this exists
import logging
import mimetypes # For more robust file type detection
import shutil # For disk space analysis

# Configure logging (optional, but recommended)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Module-level variables
__description__ = "Analyze data for file types and size"
__parameters__ = {"directory": {"type": "string", "description": "The directory to analyze (defaults to current directory)", "required": False},
                  "max_largest_files": {"type": "integer", "description": "Maximum number of largest files to track", "required": False, "default": 10},
                  "include_hidden": {"type": "boolean", "description": "Include hidden files and directories in the analysis", "required": False, "default": False},
                  "min_access_count": {"type": "integer", "description": "Minimum access count to consider a file frequently accessed.", "required": False, "default": 3},
                  "cache_expiry_hours": {"type": "integer", "description": "Number of hours after which the cache expires.", "required": False, "default": 24}
                 } # Mark as optional and provide defaults.
__examples__ = {
    "example_1": "await analyze_file_types_and_size()",
    "example_2": "await analyze_file_types_and_size(directory='/path/to/my/directory')",
    "example_3": "await analyze_file_types_and_size(directory='/path/to/my/directory', max_largest_files=20)",
    "example_4": "await analyze_file_types_and_size(include_hidden=True)",
    "example_5": "await analyze_file_types_and_size(min_access_count=5, cache_expiry_hours=48)"
}

CACHE_FILE = "file_analysis_cache.json"
DEFAULT_MAX_LARGEST_FILES = 10 #Constant for the default value
DEFAULT_MIN_ACCESS_COUNT = 3
DEFAULT_CACHE_EXPIRY_HOURS = 24


def calculate_directory_hash(directory, include_hidden=False):
    """Calculates a hash of the directory structure and modification times."""
    hasher = hashlib.sha256()
    try:
        for root, dirs, files in os.walk(directory):
            # Sort dirs and files for consistent hashing across different OS
            dirs.sort()
            files.sort()

            if not include_hidden and os.path.basename(root).startswith('.'):
                dirs[:] = [d for d in dirs if not d.startswith('.')]  # Skip hidden directories
                continue

            hasher.update(root.encode('utf-8'))
            for d in dirs:
                if not include_hidden and d.startswith('.'):
                    continue #Skip hidden directory within a visible one.
                hasher.update(d.encode('utf-8'))

            for f in files:
                if not include_hidden and f.startswith('.'):
                    continue
                file_path = os.path.join(root, f)
                try:
                    hasher.update(str(os.path.getmtime(file_path)).encode('utf-8')) #modification time
                    hasher.update(str(os.path.getsize(file_path)).encode('utf-8')) #include size in hash.
                except OSError as e:
                    logging.warning(f"Could not get modification time/size for {file_path}: {e}")
                    # Handle cases where the file might be inaccessible
                    pass
    except OSError as e:
        logging.error(f"Error walking directory {directory}: {e}")
        return None  # Indicate failure to hash

    return hasher.hexdigest()


async def load_from_cache(directory, include_hidden, cache_expiry_hours):
    """Loads analysis results from cache if available and valid."""
    try:
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
            cache_age_hours = (os.path.getmtime(CACHE_FILE) - os.path.getctime(CACHE_FILE))/(60*60) #Approx cache age in hours

            if cache_data.get('directory') == directory and cache_data.get('include_hidden') == include_hidden and cache_data.get('directory_hash') == calculate_directory_hash(directory, include_hidden) and cache_age_hours <= cache_expiry_hours:
                logging.info("Loading from cache.")
                return cache_data.get('report')
            else:
                logging.info("Cache invalid (directory change, include_hidden flag changed, or cache expired).")
                return None
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logging.info(f"Cache file not found, invalid, or inaccessible: {e}")
        return None
    except Exception as e:
         logging.error(f"Error loading from cache: {e}")
         return None # Don't crash if cache loading fails.


async def save_to_cache(directory, report, include_hidden):
    """Saves analysis results to cache."""
    try:
        cache_data = {
            'directory': directory,
            'include_hidden': include_hidden,
            'directory_hash': calculate_directory_hash(directory, include_hidden),
            'report': report
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
        logging.info("Saved to cache.")
    except Exception as e:
        logging.error(f"Error saving to cache: {e}")  # Log the error but don't crash.


def human_readable_size(size_in_bytes):
    """Converts bytes to human-readable format (e.g., KB, MB, GB)."""
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} EB"


def get_file_age_in_days(file_path):
    """Calculates the age of a file in days using the last modified time."""
    try:
        file_modification_time = os.path.getmtime(file_path)
        current_time = os.path.getctime(file_path)  #use last access time instead.
        age_in_seconds = current_time - file_modification_time #difference between last access and modification.
        age_in_days = age_in_seconds / (60 * 60 * 24)
        return age_in_days
    except OSError as e:
        logging.warning(f"Could not get modification time for {file_path}: {e}")
        return None # or some other sensible default.

def is_binary_file(file_path):
    """Checks if a file is likely a binary file by inspecting the first 1024 bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk  # Check for null bytes, a strong indicator of binary files
    except Exception:
        return False  # Assume not binary if we can't read it

def get_file_type_from_mime(file_path):
    """Gets the file type from the mime type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    return None

def get_file_extension(file_path):
    """Safely gets the file extension, returning None if there isn't one."""
    try:
        _, file_extension = os.path.splitext(file_path)
        return file_extension.lower()
    except:
        return None

def get_disk_usage(directory):
    """Gets the total, used, and free disk space for the directory's partition."""
    try:
        total, used, free = shutil.disk_usage(directory)
        return {
            "total": total,
            "used": used,
            "free": free,
            "total_human_readable": human_readable_size(total),
            "used_human_readable": human_readable_size(used),
            "free_human_readable": human_readable_size(free),
        }
    except OSError as e:
        logging.error(f"Could not get disk usage for {directory}: {e}")
        return None

async def analyze_file_types_and_size(**kwargs):
    """
    Perform comprehensive analysis of items to generate insights and recommendations.
    
    This function analyzes data for file types and sizes and returns a detailed report.
    
    Returns:
        dict: A dictionary containing analysis results including total files, size distribution,
              type distribution, and recommendations.
    """
    state_manager = StateManager()  # Instantiate state manager

    # Input Validation
    directory_to_analyze = kwargs.get('directory', '.')
    if not isinstance(directory_to_analyze, str):
        return {"error": "Directory must be a string."}

    if not os.path.exists(directory_to_analyze):
        return {"error": f"The path '{directory_to_analyze}' does not exist."}

    max_largest_files = kwargs.get('max_largest_files', DEFAULT_MAX_LARGEST_FILES)
    if not isinstance(max_largest_files, int) or max_largest_files <= 0:
        max_largest_files = DEFAULT_MAX_LARGEST_FILES
        logging.warning(f"Invalid max_largest_files value.  Using default of {DEFAULT_MAX_LARGEST_FILES}")

    include_hidden = kwargs.get('include_hidden', False)
    if not isinstance(include_hidden, bool):
        include_hidden = False
        logging.warning("Invalid include_hidden value. Using default of False.")

    min_access_count = kwargs.get('min_access_count', DEFAULT_MIN_ACCESS_COUNT)
    if not isinstance(min_access_count, int) or min_access_count <= 0:
        min_access_count = DEFAULT_MIN_ACCESS_COUNT
        logging.warning(f"Invalid min_access_count value. Using default of {DEFAULT_MIN_ACCESS_COUNT}")

    cache_expiry_hours = kwargs.get('cache_expiry_hours', DEFAULT_CACHE_EXPIRY_HOURS)
    if not isinstance(cache_expiry_hours, int) or cache_expiry_hours <= 0:
        cache_expiry_hours = DEFAULT_CACHE_EXPIRY_HOURS
        logging.warning(f"Invalid cache_expiry_hours value. Using default of {DEFAULT_CACHE_EXPIRY_HOURS}")



    try:
        # Load the state, if any specific state handling is required
        state = state_manager.load_state()

        # Attempt to load from cache
        cached_report = await load_from_cache(directory_to_analyze, include_hidden, cache_expiry_hours)
        if cached_report:
            return cached_report

        if not os.path.isdir(directory_to_analyze):
            return {"error": f"The path '{directory_to_analyze}' is not a valid directory."}
        
        # Initialize variables for analysis
        total_files = 0
        total_size = 0
        type_distribution = defaultdict(int)
        size_distribution = defaultdict(int)
        largest_files = []  # List to store information about the largest files
        empty_directories = [] # List to store paths of empty directories.
        oldest_files = [] # New feature: track oldest files.
        file_access_frequency = defaultdict(int) #track how often files are accessed.
        error_files = [] # Track files with access errors

        # Walk through the directory and perform the analysis
        for root, dirs, files in os.walk(directory_to_analyze):
            # Sort dirs and files for consistent results.
            dirs.sort()
            files.sort()

            if not include_hidden and os.path.basename(root).startswith('.'):
                dirs[:] = [d for d in dirs if not d.startswith('.')]  # Skip hidden directories
                continue  # Skip this hidden directory and its contents

            if not dirs and not files:
                empty_directories.append(root)

            for file in files:
                if not include_hidden and file.startswith('.'):
                    continue

                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    file_access_frequency[file_path] += 1

                    # Get file extension for type analysis
                    file_extension = get_file_extension(file_path)
                    if file_extension:
                        type_distribution[file_extension] += 1
                    else:
                        mime_type = get_file_type_from_mime(file_path)
                        if mime_type:
                            type_distribution[f"mime:{mime_type}"] += 1
                        else:
                            type_distribution["no_extension"] += 1



                    size_distribution[file_size] += 1

                    # Track largest files (keep top MAX_LARGEST_FILES)
                    largest_files.append((file_path, file_size))
                    largest_files.sort(key=lambda x: x[1], reverse=True)
                    largest_files = largest_files[:max_largest_files] # Only keep the top MAX_LARGEST_FILES

                    # Track oldest files
                    file_age = get_file_age_in_days(file_path)
                    if file_age is not None:  #Check for error case
                        oldest_files.append((file_path, file_age))
                        oldest_files.sort(key=lambda x: x[1], reverse=True) #Sort by age (largest first)
                        oldest_files = oldest_files[:max_largest_files]

                except OSError as e:
                    logging.warning(f"Error accessing {file_path}: {e}") # Handle errors when getting file size.
                    error_files.append({"file_path": file_path, "error": str(e)})
                    continue #Continue to the next file
        # Example insights and recommendations based on analysis
        insights = {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_human_readable": human_readable_size(total_size),
            "type_distribution": dict(type_distribution),
            "size_distribution": dict(size_distribution),
            "largest_files": [{"path": path, "size_bytes": size, "size_human_readable": human_readable_size(size)} for path, size in largest_files],
            "empty_directories": empty_directories,
            "oldest_files": [{"path": path, "age_days": age} for path, age in oldest_files],
            "file_access_frequency": dict(file_access_frequency),
            "disk_usage": get_disk_usage(directory_to_analyze), # Include disk usage information
            "error_files": error_files
        }

        recommendations = []

        if total_size > 1e9:  # If total size is greater than 1GB
            recommendations.append("Consider cleaning up large files to free up space.")

        if ".tmp" in type_distribution or ".temp" in type_distribution:
            recommendations.append("Temporary files found, consider removing them to reclaim space.")

        if len(empty_directories) > 5:
            recommendations.append("Multiple empty directories found. Consider removing them to improve organization.")

        common_types = sorted(type_distribution.items(), key=lambda item: item[1], reverse=True)[:3] #top 3 file types.
        recommendations.append(f"The most common file types are {', '.join([f'{t[0]} ({t[1]})' for t in common_types])}")

        if oldest_files:
            oldest_file_path, oldest_file_age = oldest_files[0]
            recommendations.append(f"The oldest file is {oldest_file_path} ({oldest_file_age:.2f} days old). Consider archiving or deleting if no longer needed.")

        # Binary File Check Recommendation
        for file_path, _ in largest_files:
            if is_binary_file(file_path):
                recommendations.append(f"{file_path} appears to be a binary file.  Consider specialized tools for analysis.")
                break # Only recommend once.

        # Check for infrequently accessed files
        infrequently_accessed_files = {file: count for file, count in file_access_frequency.items() if count < min_access_count}
        if infrequently_accessed_files:
            recommendations.append(f"Consider archiving files that are accessed infrequently (less than {min_access_count} times).")

        # Disk Space Recommendations
        disk_usage = insights.get("disk_usage")
        if disk_usage:
            free_space = disk_usage.get("free")
            total_space = disk_usage.get("total")
            if free_space is not None and total_space is not None:
                if (free_space / total_space) < 0.1: #Less than 10% free
                    recommendations.append("Disk space is running low. Consider freeing up space.")


        # Error file recommendations
        if error_files:
            recommendations.append(f"Encountered errors accessing {len(error_files)} files. Check permissions or file integrity.")


        # Compile the report
        report = {
            "directory": directory_to_analyze, #Include this in the report
            "include_hidden": include_hidden,
            "insights": insights,
            "recommendations": recommendations
        }

        await save_to_cache(directory_to_analyze, report, include_hidden)

        return report

    except Exception as e:
        logging.exception("An unexpected error occurred.") # Log full exception details
        return {"error": f"An unexpected error occurred: {str(e)}"}