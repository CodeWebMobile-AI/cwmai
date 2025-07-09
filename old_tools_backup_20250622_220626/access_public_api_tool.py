"""
Module containing the access_public_api_tool.

This tool accepts a URL of a public API endpoint, fetches data,
handles errors, and returns the data in a structured format (JSON).
"""

import asyncio
import json
import aiohttp
import hashlib
import os
import re  # Import the regular expression module

from scripts.state_manager import StateManager
from scripts.logger import Logger  # Assuming you have a logger module

__description__ = "Accesses a public API endpoint and returns the data.  Supports caching, enhanced error handling, content type detection, request headers, and custom parsing."
__parameters__ = {
    "url": {
        "type": "string",
        "description": "The URL of the public API endpoint.",
        "required": True,
    },
    "cache_expiry": {
        "type": "integer",
        "description": "Cache expiry time in seconds.  Defaults to 3600 (1 hour). Set to 0 to disable caching.",
        "required": False,
        "default": 3600,
    },
    "headers": {
        "type": "string",
        "description": "JSON string of headers to include in the API request (e.g., '{\"Authorization\": \"Bearer token\"}').",
        "required": False,
    },
    "custom_parser": {
        "type": "string",
        "description": "Python code snippet to parse the API response. The response text is available as 'response_text'. Must return a JSON-serializable object.",
        "required": False,
    },
    "max_retries": {
        "type": "integer",
        "description": "Maximum number of retries for failed API requests. Defaults to 1.",
        "required": False,
        "default": 1
    }
}
__examples__ = [
    {
        "url": "https://api.example.com/data",
        "description": "Fetches data from the specified API endpoint.",
    },
    {
        "url": "https://api.example.com/data",
        "cache_expiry": 7200,
        "description": "Fetches data from the specified API endpoint, caching the result for 2 hours.",
    },
    {
        "url": "https://api.example.com/data",
        "cache_expiry": 0,
        "description": "Fetches data from the specified API endpoint, disabling caching.",
    },
    {
        "url": "https://api.example.com/data",
        "headers": '{"Authorization": "Bearer my_token"}',
        "description": "Fetches data with custom headers.",
    },
    {
        "url": "https://api.example.com/data",
        "custom_parser": "response_text.split(',')[0]",
        "description": "Fetches data and parses it using custom Python code.",
    },
    {
        "url": "https://api.example.com/data",
        "max_retries": 3,
        "description": "Fetches data and retries up to 3 times if the request fails."
    }
]


async def access_public_api_tool(**kwargs):
    """
    Accesses a public API endpoint and returns the data.

    Args:
        **kwargs: Keyword arguments containing the 'url' of the API endpoint, 'cache_expiry', 'headers', and 'custom_parser'.

    Returns:
        A dictionary containing the fetched data, or an error message.
    """
    state_manager = StateManager()
    state = state_manager.load_state()
    logger = Logger()  # Initialize the logger

    url = kwargs.get("url")
    cache_expiry = kwargs.get("cache_expiry", 3600)  # Default to 1 hour
    cache_expiry = int(cache_expiry)  # Ensure it's an integer.
    headers_str = kwargs.get("headers")
    custom_parser = kwargs.get("custom_parser")
    max_retries = kwargs.get("max_retries", 1)
    max_retries = int(max_retries)

    # Input validation
    if not url:
        error_message = "Error: URL is required."
        logger.info(error_message, level="ERROR")  # Log the error
        return {"error": error_message}

    if not isinstance(url, str):
        error_message = "Error: URL must be a string."
        logger.info(error_message, level="ERROR")
        return {"error": error_message}

    if not (isinstance(cache_expiry, int) and cache_expiry >= 0):
        error_message = "Error: Cache expiry must be a non-negative integer."
        logger.info(error_message, level="ERROR")
        return {"error": error_message}

    if headers_str:
        try:
            headers = json.loads(headers_str)
            if not isinstance(headers, dict):
                raise ValueError("Headers must be a JSON object (dictionary).")
        except (json.JSONDecodeError, ValueError) as e:
            error_message = f"Error: Invalid headers format.  Must be a JSON string representing a dictionary: {e}"
            logger.info(error_message, level="ERROR")
            return {"error": error_message}
    else:
        headers = {}

    if custom_parser and not isinstance(custom_parser, str):
         error_message = "Error: custom_parser must be a string."
         logger.info(error_message, level="ERROR")
         return {"error": error_message}

    if not (isinstance(max_retries, int) and max_retries >= 0):
        error_message = "Error: Max retries must be a non-negative integer."
        logger.info(error_message, level="ERROR")
        return {"error": error_message}


    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_file = f".api_cache/{cache_key}.json"  # Store cache in .api_cache directory

    # Create the cache directory if it doesn't exist
    if not os.path.exists(".api_cache"):
        os.makedirs(".api_cache")

    if cache_expiry > 0 and os.path.exists(cache_file) and (os.path.getmtime(cache_file) + cache_expiry > asyncio.get_event_loop().time()):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                logger.info(f"Retrieved data from cache for {url}")
                return {"data": data, "source": "cache", "cache_expiry": cache_expiry}  # Indicate data source and expiry
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Error reading or decoding cache file: {e}. Fetching from API.") # Log warning but proceed to fetch

    for attempt in range(max_retries + 1):  # Retry loop
        try:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url, headers=headers, timeout=10) as response: # Adding timeout & headers
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        content_type = response.headers.get('Content-Type', '')
                        try:
                            response_text = await response.text()  # Get the raw response text

                            if custom_parser:
                                try:
                                    # Execute the custom parser code
                                    local_vars = {'response_text': response_text, 'json': json, 're': re}  # Provide access to the response text, json library, and re library
                                    exec(f"parsed_data = {custom_parser}", globals(), local_vars)
                                    data = local_vars['parsed_data']
                                    logger.info(f"Successfully parsed data from {url} using custom parser.")
                                except Exception as e:
                                    error_message = f"Error: Custom parser failed: {e}"
                                    logger.info(error_message, level="ERROR")
                                    return {"error": error_message}


                            elif 'application/json' in content_type:
                                data = json.loads(response_text) # Directly parse from text to avoid double parsing
                            else:
                                try:
                                    data = json.loads(response_text) #attempt to parse if possible
                                except json.JSONDecodeError:
                                    data = response_text # Fallback to raw text if JSON parsing fails

                            logger.info(f"Successfully fetched data from {url}")  # Log success

                            if cache_expiry > 0:
                                try:
                                    with open(cache_file, "w") as f:
                                        json.dump(data, f)
                                    logger.info(f"Cached data for {url}")
                                except OSError as e:
                                    logger.error(f"Failed to write to cache file: {e}", level="ERROR")


                            return {"data": data, "source": "api", "cache_expiry": cache_expiry, "content_type": content_type} # Indicate data source, expiry, and content type
                        except json.JSONDecodeError:
                            error_message = "Error: Could not decode JSON response."
                            logger.info(error_message, level="ERROR")  # Log the error
                            return {"error": error_message}

                except aiohttp.ClientResponseError as e:
                    error_message = f"Error: API request failed with status code {e.status} for URL: {url}.  Reason: {e.message}"
                    logger.info(error_message, level="ERROR")  # Log the error
                    if attempt < max_retries:
                        logger.warning(f"Retrying in 5 seconds (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(5)
                    else:
                        return {"error": error_message, "status_code": e.status} # Include status code in output


                except asyncio.TimeoutError:
                    error_message = f"Error: API request timed out for URL: {url}"
                    logger.info(error_message, level="ERROR")
                    if attempt < max_retries:
                        logger.warning(f"Retrying in 5 seconds (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(5)
                    else:
                        return {"error": error_message}


        except aiohttp.ClientError as e:
            error_message = f"Error: Network issue or invalid URL: {e}"
            logger.info(error_message, level="ERROR")  # Log the error
            if attempt < max_retries:
                logger.warning(f"Retrying in 5 seconds (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(5)
            else:
                return {"error": error_message}
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            logger.info(error_message, level="ERROR")  # Log the error
            return {"error": error_message}

    return {"error": "Max retries exceeded. API request failed."}