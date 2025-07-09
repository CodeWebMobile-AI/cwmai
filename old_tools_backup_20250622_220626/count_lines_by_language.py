"""
AI-Generated Tool: count_lines_by_language
Description: Count items for lines by language
Generated: 2025-06-15T11:08:05.315927+00:00
Requirements: 
        Tool Name: count_lines_by_language
        Intent: Count items for lines by language
        Expected Parameters: {}
        Category: analytics
        
        The tool should:
        1. Count all items in the system
2. Return total count with breakdown by status/type
3. Include summary statistics
        
"""

"""
Module for counting lines by language in the system.

Provides functionality to:
- Count all items in the system
- Return total count with breakdown by language
- Include summary statistics
"""

from scripts.state_manager import StateManager
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
import os
import time
import re

__description__ = "Count items for lines by language"
__parameters__ = {}
__examples__ = [
    {"description": "Count all lines by language",
     "code": "await count_lines_by_language()"}
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache settings
CACHE_ENABLED = True
CACHE_FILE = "language_counts_cache.json"
CACHE_EXPIRY_SECONDS = 3600  # 1 hour

# Language Alias Mapping (Enhancement: Language Normalization)
LANGUAGE_ALIASES = {
    "javascript": "js",
    "typescript": "ts",
    "python": "py",
    "html": "markup", # Use markup as a general term
    "css": "markup", # Use markup as a general term
    "php": "markup",  # Use markup as a general term
    "c++": "cpp",
    "c#": "csharp"
}


def load_from_cache() -> Optional[Dict[str, Any]]:
    """Loads data from cache if it exists and is not expired."""
    if not CACHE_ENABLED:
        return None

    try:
        if os.path.exists(CACHE_FILE):
            modified_time = os.path.getmtime(CACHE_FILE)
            age_seconds = time.time() - modified_time
            if age_seconds < CACHE_EXPIRY_SECONDS:
                with open(CACHE_FILE, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON from cache: {e}. Invalidating cache.")
                        return None # Invalidate cache if JSON is corrupted

                    logging.info("Data retrieved from cache.")
                    return data
            else:
                logging.info("Cache expired.")
                return None
        else:
            return None
    except (FileNotFoundError, OSError) as e:
        logging.error(f"Error loading from cache: {e}")
        return None


def save_to_cache(data: Dict[str, Any]):
    """Saves data to cache."""
    if not CACHE_ENABLED:
        return

    try:
        # Ensure directory exists (Enhancement: Directory creation)
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)  # Add indentation for readability
        logging.info("Data saved to cache.")
    except (IOError, OSError) as e:
        logging.error(f"Error saving to cache: {e}")


def normalize_language(language: str) -> str:
    """Normalizes language names using the LANGUAGE_ALIASES map."""
    if not isinstance(language, str):
        logging.warning(f"Invalid language type: {type(language)}. Returning 'unknown'")
        return 'unknown'
    language = language.lower()
    return LANGUAGE_ALIASES.get(language, language)


def validate_item(item: Any) -> bool:
    """
    Validates that an item is a dictionary and contains a valid 'language' field.
    """
    if not isinstance(item, dict):
        logging.warning(f"Skipping invalid item: {item} (not a dictionary)")
        return False
    
    if 'language' not in item:
        logging.warning(f"Skipping item without 'language' field: {item}")
        return False

    if not isinstance(item['language'], str):
        logging.warning(f"Skipping item with invalid 'language' type: {item}")
        return False
    
    return True

def calculate_descriptive_statistics(counts: Dict[str, int]) -> Dict[str, Any]:
    """Calculates descriptive statistics from a dictionary of counts."""
    total = sum(counts.values())
    if not counts:
        return {
            "total_items": 0,
            "unique_languages": 0,
            "most_common": None,
            "least_common": None,
            "average_per_language": 0,
            "standard_deviation": 0,  # Enhancement: Add standard deviation
        }

    # Calculate mean
    mean = total / len(counts)

    # Calculate variance
    variance = sum([(count - mean) ** 2 for count in counts.values()]) / len(counts)

    # Calculate standard deviation
    standard_deviation = variance ** 0.5

    most_common = max(counts.items(), key=lambda x: x[1])[0]
    least_common = min(counts.items(), key=lambda x: x[1])[0]

    return {
        "total_items": total,
        "unique_languages": len(counts),
        "most_common": most_common,
        "least_common": least_common,
        "average_per_language": mean,
        "standard_deviation": standard_deviation,
    }


async def count_lines_by_language(**kwargs) -> Dict[str, Any]:
    """
    Count all items in the system by language and return statistics.
    
    Returns:
        Dictionary containing:
        - total: Total count of all items
        - by_language: Breakdown of counts by language
        - language_percentages: Percentage breakdown by language
        - summary: Summary statistics
        - errors: List of any encountered errors during processing
    """

    # Check cache first
    cached_data = load_from_cache()
    if cached_data:
        return cached_data
    
    errors: List[str] = []  # Store errors during processing

    try:
        state_manager = StateManager()
        try:
            state = state_manager.load_state()
        except Exception as e:
            error_message = f"Error loading state from StateManager: {e}"
            logging.error(error_message)
            errors.append(error_message)
            result = {
                "status": "error",
                "error": error_message, # Changed to "error" to better reflect single error scenario
                "total": 0,
                "by_language": {},
                "language_percentages": {},
                "summary": "Error occurred while loading system state",
                "errors": errors
            }
            save_to_cache(result)
            return result
        
        if not state or 'items' not in state:
            warning_message = "No items found in system state."
            logging.warning(warning_message)
            errors.append(warning_message)

            result = {
                "status": "warning",
                "total": 0,
                "by_language": {},
                "language_percentages": {},
                "summary": "No items found in system",
                "errors": errors
            }
            save_to_cache(result)
            return result
        
        items = state['items']

        # Input validation: Check if items is a list
        if not isinstance(items, list):
            error_message = f"Invalid data type for 'items'. Expected list, got {type(items)}"
            logging.error(error_message)
            errors.append(error_message)
            result = {
                "status": "error",
                "error": "Invalid data format: 'items' is not a list", # changed to "error"
                "total": 0,
                "by_language": {},
                "language_percentages": {},
                "summary": "Error occurred due to invalid data format",
                "errors": errors
            }
            save_to_cache(result)
            return result

        language_counts = defaultdict(int)
        valid_items_count = 0  # Count of valid items processed
        invalid_items_count = 0 # Count of invalid items skipped
        
        for item in items:
            try: #Wrap item processing in try-except
                if validate_item(item):
                    normalized_language = normalize_language(item['language'])  # Normalize Language
                    language_counts[normalized_language] += 1
                    valid_items_count += 1
                else:
                    language_counts['unknown'] +=1
                    invalid_items_count += 1 # Increment invalid items count
            except Exception as e:
                error_message = f"Error processing item: {item}. Error: {e}"
                logging.error(error_message)
                errors.append(error_message)  # Collect errors for each item

        total = sum(language_counts.values())
        
        # More comprehensive output, including percentage breakdown
        language_percentages = {lang: (count / total) * 100 for lang, count in language_counts.items()}

        # Calculate statistics
        summary_statistics = calculate_descriptive_statistics(dict(language_counts))

        result = {
            "status": "success",
            "total": total,
            "by_language": dict(language_counts),
            "language_percentages": language_percentages,
            "summary": {
                **summary_statistics, # Unpack existing stats
                "valid_items_count": valid_items_count,
                "invalid_items_count": invalid_items_count
            },
            "errors": errors # Include the list of errors in the result.  Even if empty.
        }

        save_to_cache(result)  # Save to cache before returning
        return result
        
    except FileNotFoundError as e:
        logging.exception("State file not found.")
        error_message = f"State file not found: {e}"
        errors.append(error_message) #append to error list
        result = {
            "status": "error",
            "error": error_message, #Changed to "error"
            "total": 0,
            "by_language": {},
            "language_percentages": {},
            "summary": "Error: State file not found",
            "errors": errors
        }
        save_to_cache(result)
        return result
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        error_message = str(e)
        errors.append(error_message)

        result = {
            "status": "error",
            "error": error_message, #Changed to "error"
            "total": 0,
            "by_language": {},
            "language_percentages": {},
            "summary": "Error occurred while counting items",
            "errors": errors # Include collected errors
        }
        save_to_cache(result)
        return result