"""
AI-Generated Tool: get_me
Description: Retrieve information for me
Generated: 2025-06-15T09:55:54.772004+00:00
Requirements: 
        Tool Name: get_me
        Intent: Retrieve information for me
        Expected Parameters: {}
        Category: query
        
        The tool should:
        Implement functionality for get_me
        
"""

"""
Module: get_me

Description: Retrieves information about the user based on the current system state.
This tool attempts to access user-specific information and returns relevant details.
"""

import asyncio
import logging
import json
import time
from typing import Optional, Dict, Any

# Module-level variables
__description__ = "Retrieve information for me"
__parameters__ = {}  # No parameters required
__examples__ = [
    {
        "query": "get_me",
        "description": "Retrieves information about the user.",
    }
]

# Define a cache (simple dictionary)
user_information_cache: Optional[dict] = None
cache_expiry = 300  # Cache expiry time in seconds
last_cache_update: Optional[float] = None #Timestamp for last cache update
cache_lock = asyncio.Lock() #Lock to ensure thread-safe cache access



async def get_me(self) -> Dict[str, Any]:
    """
    Retrieves information about the user based on the current system state.
    Includes caching, retry logic and better output formatting.

    Args:
        self: The agent instance.

    Returns:
        A dictionary with 'success' and result data.
        'success' is True if the operation was successful, False otherwise.
        If successful, 'result' contains a detailed description of the user if available,
        or a default message if not.  If unsuccessful, 'result' contains an error message.
    """
    global user_information_cache
    global cache_expiry
    global last_cache_update

    try:
        # Input Validation (example, though no explicit input is expected)
        if not isinstance(self, object):
            raise TypeError("Invalid agent object provided.")

        # Access state manager
        if self.state_manager is None:
            raise ValueError("State manager is not initialized.")

        # Access task manager
        if self.task_manager is None:
            raise ValueError("Task manager is not initialized.")

        # Access repository analyzer
        if self.repo_analyzer is None:
            raise ValueError("Repository analyzer is not initialized.")


        # Check cache validity
        async with cache_lock: #Acquire lock for cache operations
            if (user_information_cache is not None and
                last_cache_update is not None and
                time.time() - last_cache_update < cache_expiry):

                logging.info("Retrieving user information from cache.")
                result_message = f"User information (from cache): {user_information_cache}"
                return {"success": True, "result": result_message, "cached": True} # Indicate data came from cache


        # Retry logic (attempt a few times in case of transient errors)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Attempt to retrieve user information from the state
                user_information = await self.state_manager.get("user_information")
                if user_information is None:
                    raise ValueError("User information is None in State Manager.")
                break #If successfull break retry loop
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed to retrieve user information: {str(e)}. Retrying...")
                await asyncio.sleep(1)  # Wait before retrying
        else:
            raise Exception("Failed to retrieve user information after multiple retries.") #Raise Exception when all retries failed

        if user_information:
            #Enhanced output formatting with json
            try:
                user_information_json = json.dumps(user_information, indent=4)
                result_message = f"User information: \n{user_information_json}"
            except (TypeError, OverflowError) as e:
                result_message = f"User information: {user_information}. Error serializing to JSON: {str(e)}"


            # Update the cache
            async with cache_lock: #Acquire lock for cache operations
                user_information_cache = user_information
                last_cache_update = time.time()
                logging.info("User information cached.")

        else:
            result_message = "No user information available. The system may need to be initialized or user details captured using other tools."


        logging.info(f"get_me tool executed successfully: {result_message}")

        return {"success": True, "result": result_message, "cached": False} #Indicate data didn't came from cache

    except ValueError as ve:
        error_message = f"Validation Error: {str(ve)}"
        logging.error(error_message, exc_info=True)
        return {"success": False, "result": error_message, "cached": False}

    except Exception as e:
        error_message = f"Error retrieving user information: {str(e)}"
        logging.error(error_message, exc_info=True)  # Log the exception with traceback
        return {"success": False, "result": error_message, "cached": False}


# Utility function (example: clear the cache)
async def clear_user_info_cache():
    """
    Clears the cached user information.
    """
    global user_information_cache
    global last_cache_update

    async with cache_lock: #Acquire lock for cache operations
        user_information_cache = None
        last_cache_update = None
        logging.info("User information cache cleared.")

# Utility Function (Set Cache Expiry)
def set_cache_expiry(new_expiry: int):
    """
    Sets the cache expiry time in seconds.

    Args:
        new_expiry: The new expiry time in seconds.  Must be a non-negative integer.
    """
    global cache_expiry

    if not isinstance(new_expiry, int) or new_expiry < 0:
        raise ValueError("Cache expiry must be a non-negative integer.")

    cache_expiry = new_expiry
    logging.info(f"Cache expiry set to {new_expiry} seconds.")

#Utility Function (Inspect Cache Content)
def inspect_cache():
    """
    Returns the content of the cache. Useful for debugging and monitoring.
    """
    global user_information_cache
    return user_information_cache


# Example Usage (This part is for testing purposes only)
if __name__ == "__main__":
    # Create a mock agent class for testing
    class MockAgent:
        def __init__(self):
            self.state_manager = MockStateManager()
            self.task_manager = MockTaskManager()
            self.repo_analyzer = MockRepoAnalyzer()

    class MockStateManager:
        async def get(self, key):
            # Simulate retrieving user information from state
            await asyncio.sleep(0.1) #Simulate latency
            if key == "user_information":
                return {"name": "John Doe", "title": "AI Researcher", "skills": ["Python", "ML", "DL"]}
            else:
                return None

    class MockTaskManager:
        async def create_task(self, *args, **kwargs):
            # Simulate task creation
            return "Task created"

    class MockRepoAnalyzer:
        async def analyze_repository(self, *args, **kwargs):
            return "Repo analyzed"

    async def main():
        agent = MockAgent()
        result1 = await get_me(agent)
        print("First call:", result1)

        result2 = await get_me(agent) # call it again, should be cached
        print("Second call:", result2)

        print("Cache Content:", inspect_cache())

        set_cache_expiry(5) # Set a short cache expiry for testing

        await asyncio.sleep(6) # Wait for the cache to expire

        result3 = await get_me(agent) # call it again after expiry, should not be cached
        print("Third call after expiry:", result3)

        await clear_user_info_cache()

        result4 = await get_me(agent) # call it after clear, should not be cached
        print("Fourth call after cache clear:", result4)


    asyncio.run(main())