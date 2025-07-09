import os
import re
from typing import Dict, Any, Optional
from scripts.logger import logger
import json  # For caching
import hashlib  # For generating cache keys

__description__ = "Counts the number of Markdown files in a specified directory with caching and improved error handling."
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "The directory to search in. Defaults to the current directory.",
        "required": False
    },
    "use_cache": {
        "type": "boolean",
        "description": "Whether to use the cache for faster results. Defaults to True.",
        "required": False
    },
    "clear_cache": {
        "type": "boolean",
        "description": "Whether to clear the cache before counting. Defaults to False.",
        "required": False
    }
}
__examples__ = [
    {"command": "count_markdown_files", "description": "Counts all Markdown files in the current directory using the cache."},
    {"command": "count_markdown_files directory=/path/to/directory", "description": "Counts Markdown files in the specified directory."},
    {"command": "count_markdown_files use_cache=false", "description": "Counts Markdown files without using the cache."},
    {"command": "count_markdown_files clear_cache=true", "description": "Clears the cache and then counts Markdown files."}
]

CACHE_FILE = ".markdown_count_cache.json"


def _generate_cache_key(directory: str) -> str:
    """Generates a cache key based on the directory path."""
    return hashlib.md5(directory.encode('utf-8')).hexdigest()


def _load_cache() -> Dict[str, int]:
    """Loads the cache from the cache file."""
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        logger.warning("Cache file is corrupted.  Returning empty cache.")
        return {}
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return {}  # Return an empty dictionary if loading fails


def _save_cache(cache_data: Dict[str, int]) -> None:
    """Saves the cache to the cache file."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")


def _validate_directory(directory: str) -> Optional[str]:
    """Validates that the directory exists and is a directory.
    Returns None if valid, an error message string otherwise.
    """
    if not os.path.exists(directory):
        return f"Directory '{directory}' does not exist."
    if not os.path.isdir(directory):
        return f"'{directory}' is not a directory."
    return None


async def count_markdown_files(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Counts the number of Markdown files in a specified directory.
    Uses caching for performance.

    Args:
        **kwargs (Dict[str, Any]): Keyword arguments.
            - directory (str, optional): The directory to search in. Defaults to the current directory.
            - use_cache (bool, optional): Whether to use the cache. Defaults to True.
            - clear_cache (bool, optional): Whether to clear the cache before counting. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing the count of Markdown files and the directory searched.
                       If an error occurs, it will contain an error message.
    """
    directory = kwargs.get('directory', '.')  # Default to current directory
    use_cache = kwargs.get('use_cache', True)  # Default to True
    clear_cache = kwargs.get('clear_cache', False)  # Default to False

    # Input Validation
    validation_error = _validate_directory(directory)
    if validation_error:
        logger.error(validation_error)
        return {"error": validation_error}

    cache_key = _generate_cache_key(directory)
    cache = _load_cache()

    if clear_cache:
        logger.info("Clearing the cache.")
        cache = {}
        _save_cache(cache)

    if use_cache and cache_key in cache:
        logger.info(f"Using cached count for directory: {directory}")
        return {"count": cache[cache_key], "directory": directory, "cache_hit": True}

    try:
        count = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    count += 1

        cache[cache_key] = count
        _save_cache(cache)
        logger.info(f"Counted {count} Markdown files in directory: {directory}")
        return {"count": count, "directory": directory, "cache_hit": False}

    except Exception as e:
        logger.error(f"Error counting Markdown files in {directory}: {e}")
        return {"error": str(e)}