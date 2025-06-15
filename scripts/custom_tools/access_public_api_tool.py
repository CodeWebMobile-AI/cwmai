"""
AI-Generated Tool: access_public_api_tool
Description: 
Since there isn't an existing tool for this specific functionality, here's what I suggest:

CREATE_TOOL: access_public_api_tool

This tool should:
- Accept a URL of a public API endpoint.
- Fetch data from the endpoint without using any authentication credentials.
- Handle possible errors such as network issues or invalid URLs.
- Return the fetched data in a structured format (e.g., JSON).

Would you like to proceed with creating this tool?
Generated: 2025-06-15T10:53:32.704484+00:00
Requirements: 
        Based on user request: create a tool that accesses external APIs without credentials
        Create a tool that: 
Since there isn't an existing tool for this specific functionality, here's what I suggest:

CREATE_TOOL: access_public_api_tool

This tool should:
- Accept a URL of a public API endpoint.
- Fetch data from the endpoint without using any authentication credentials.
- Handle possible errors such as network issues or invalid URLs.
- Return the fetched data in a structured format (e.g., JSON).

Would you like to proceed with creating this tool?
        Tool should integrate with existing system components
        
"""

"""
Module containing the access_public_api_tool.

This tool accepts a URL of a public API endpoint, fetches data,
handles errors, and returns the data in a structured format (JSON).
"""

import asyncio
import json
import aiohttp

from scripts.state_manager import StateManager
from scripts.logger import Logger  # Assuming you have a logger module

__description__ = "Accesses a public API endpoint and returns the data."
__parameters__ = {
    "url": {
        "type": "string",
        "description": "The URL of the public API endpoint.",
        "required": True,
    }
}
__examples__ = [
    {
        "url": "https://api.example.com/data",
        "description": "Fetches data from the specified API endpoint.",
    }
]


async def access_public_api_tool(**kwargs):
    """
    Accesses a public API endpoint and returns the data.

    Args:
        **kwargs: Keyword arguments containing the 'url' of the API endpoint.

    Returns:
        A dictionary containing the fetched data, or an error message.
    """
    state_manager = StateManager()
    state = state_manager.load_state()
    logger = Logger()  # Initialize the logger

    url = kwargs.get("url")

    if not url:
        error_message = "Error: URL is required."
        logger.log(error_message, level="ERROR")  # Log the error
        return {"error": error_message}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        logger.log(f"Successfully fetched data from {url}")  # Log success
                        return data
                    except json.JSONDecodeError:
                        error_message = "Error: Could not decode JSON response."
                        logger.log(error_message, level="ERROR")  # Log the error
                        return {"error": error_message}
                else:
                    error_message = f"Error: API request failed with status code {response.status} for URL: {url}"
                    logger.log(error_message, level="ERROR")  # Log the error
                    return {"error": error_message}

    except aiohttp.ClientError as e:
        error_message = f"Error: Network issue or invalid URL: {e}"
        logger.log(error_message, level="ERROR")  # Log the error
        return {"error": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        logger.log(error_message, level="ERROR")  # Log the error
        return {"error": error_message}
