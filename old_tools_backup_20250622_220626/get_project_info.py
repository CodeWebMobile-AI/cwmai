"""
Get Project Information Tool
===========================

Returns detailed information about a specific project from system state.
"""

from typing import Dict, Any, List
from scripts.state_manager import StateManager
import logging
import time
import asyncio
import re  # For more complex input validation

__description__ = "Get detailed information about a specific project including health score, description, and metrics"
__parameters__ = {'project_name': 'The name of the project (e.g., eco-track-ai)'}
__examples__ = [{'project_name': 'eco-track-ai'}]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Project info cache (project_name: project_data)
PROJECT_INFO_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRY_SECONDS = 60  # Cache expires after 60 seconds
CACHE_LAST_UPDATED: float = 0.0

# Constants for default values and error messages
DEFAULT_DESCRIPTION = 'No description available'
DEFAULT_LANGUAGE = 'Unknown'
DEFAULT_STATUS = 'Unknown'
PROJECT_NOT_FOUND_ERROR = "Project '{project_name}' not found in system state"
INVALID_PROJECT_NAME_ERROR = "Invalid project name provided. Please provide a valid string."
SHORT_PROJECT_NAME_ERROR = "Project name must be at least 3 characters long."
STATE_LOAD_ERROR = "Error loading state from StateManager: {}"
FETCH_PROJECT_ERROR = "Error fetching project data from state for '{project_name}': {}"
GENERAL_FETCH_ERROR = "An error occurred while fetching project data: {}"
CACHE_CLEAR_ERROR = "Error clearing the cache: {}"


async def _get_all_project_names() -> List[str]:
    """Helper function to retrieve all project names from the system state."""
    try:
        state_manager = StateManager()
        state_manager.load_state()
        state = state_manager.get_state()
        projects = state.get('projects', {})
        if not projects:
            logging.warning("No projects found in the system state.")
            return []
        return list(projects.keys())
    except Exception as e:
        logging.error(STATE_LOAD_ERROR.format(e))
        return []


async def _clear_cache():
    """Helper function to clear the cache."""
    global PROJECT_INFO_CACHE, CACHE_LAST_UPDATED
    try:
        PROJECT_INFO_CACHE = {}
        CACHE_LAST_UPDATED = 0.0
        logging.info("Project info cache cleared.")
    except Exception as e:
        logging.error(CACHE_CLEAR_ERROR.format(e))


async def _is_project_name_valid(project_name: str) -> bool:
    """Helper function to validate a project name.  Can be extended with regex for more complex validation."""
    if not isinstance(project_name, str):
        return False

    if len(project_name) < 3:
        return False

    # Example: Check if project name contains only alphanumeric characters and hyphens
    if not re.match(r"^[a-zA-Z0-9-]+$", project_name):
        return False

    return True


async def _get_project_from_state(project_name: str) -> Dict[str, Any]:
    """Helper function to fetch a project's data from the system state."""
    try:
        state_manager = StateManager()
        state_manager.load_state()
        state = state_manager.get_state()
        projects = state.get('projects', {})

        if not projects:
             return {
            "success": False,
            "error": PROJECT_NOT_FOUND_ERROR.format(project_name=project_name),
            "available_projects": await _get_all_project_names()  # provide a list of available projects.
        }


        for proj_name, proj_data in projects.items():
            if proj_name.lower() == project_name or project_name in proj_name.lower():
                project_data_to_return = {
                    "name": proj_name,
                    "health_score": proj_data.get('health_score', 0),
                    "description": proj_data.get('description', DEFAULT_DESCRIPTION),
                    "language": proj_data.get('language', DEFAULT_LANGUAGE),
                    "status": proj_data.get('status', DEFAULT_STATUS),
                    "metrics": {
                        "open_issues": proj_data.get('metrics', {}).get('issues_open', 0),
                        "stars": proj_data.get('metrics', {}).get('stars', 0),
                        "forks": proj_data.get('metrics', {}).get('forks', 0),
                        "watchers": proj_data.get('metrics', {}).get('watchers', 0)
                    },
                    "recent_activity": {
                        "commits": proj_data.get('recent_activity', {}).get('recent_commits', 0),
                        "last_commit": proj_data.get('recent_activity', {}).get('last_commit_date', 'Unknown'),
                        "contributors": proj_data.get('recent_activity', {}).get('contributors_count', 0)
                    },
                    "repository_url": proj_data.get('url', ''),
                    "clone_url": proj_data.get('clone_url', ''),
                    "topics": proj_data.get('topics', [])
                }
                return {
                    "success": True,
                    "project": project_data_to_return
                }

        return {
            "success": False,
            "error": PROJECT_NOT_FOUND_ERROR.format(project_name=project_name),
            "available_projects": await _get_all_project_names()  # provide a list of available projects.
        }

    except Exception as e:
        logging.error(FETCH_PROJECT_ERROR.format(project_name=project_name, e=e))
        return {"success": False, "error": GENERAL_FETCH_ERROR.format(e)}


async def get_project_info(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed information about a specific project.

    Args:
        project_name (str): The name of the project

    Returns:
        Dict[str, Any]: Detailed project information
    """
    global PROJECT_INFO_CACHE, CACHE_LAST_UPDATED

    project_name = kwargs.get('project_name', '').lower()

    # Input Validation
    if not project_name:
       return {"success": False, "error": INVALID_PROJECT_NAME_ERROR}


    if not await _is_project_name_valid(project_name):
        return {"success": False, "error": SHORT_PROJECT_NAME_ERROR}


    # Caching Logic
    current_time = time.time()
    if project_name in PROJECT_INFO_CACHE and current_time - CACHE_LAST_UPDATED <= CACHE_EXPIRY_SECONDS:
        logging.info(f"Retrieving project '{project_name}' information from cache.")
        return PROJECT_INFO_CACHE[project_name]

    # Fetch project data from the state
    project_data = await _get_project_from_state(project_name)

    if project_data["success"]:
        # Update cache
        PROJECT_INFO_CACHE[project_name] = project_data
        CACHE_LAST_UPDATED = current_time
        logging.info(f"Project '{project_name}' information cached.")
        # Enhance output with cache information
        project_data["cache_hit"] = False  # Indicate it was not a cache hit initially
    else:
        logging.warning(f"Project '{project_name}' not found.  Will not cache.")


    return project_data


# Utility function to get project status counts (Example enhancement)
async def get_project_status_counts() -> Dict[str, int]:
    """
    Retrieves the count of projects for each status.
    """
    try:
        state_manager = StateManager()
        state_manager.load_state()
        state = state_manager.get_state()
        projects = state.get('projects', {})
        if not projects:
            logging.warning("No projects found in the system state.")
            return {}

        status_counts: Dict[str, int] = {}
        for proj_name, proj_data in projects.items():
            status = proj_data.get('status', DEFAULT_STATUS)
            status_counts[status] = status_counts.get(status, 0) + 1

        return status_counts

    except Exception as e:
        logging.error(f"Error getting project status counts: {e}")
        return {}

# Example usage (for testing):
async def main():
    # Test case 1: Valid project name
    result = await get_project_info(project_name="eco-track-ai")
    print(f"Result for 'eco-track-ai': {result}")

    # Test case 2: Project not found
    result = await get_project_info(project_name="nonexistent-project")
    print(f"Result for 'nonexistent-project': {result}")

    # Test case 3: Invalid project name (too short)
    result = await get_project_info(project_name="ec")
    print(f"Result for 'ec': {result}")

    # Test case 4: Empty project name
    result = await get_project_info(project_name="")
    print(f"Result for '': {result}")

    # Test cache clearing
    await _clear_cache()

    #Test cache retrieval after a small delay
    await asyncio.sleep(2) #Wait 2 seconds to ensure cache has time to clear (if it were populated)
    result = await get_project_info(project_name="eco-track-ai") #Run again with valid project name after cache clear.
    print(f"Result for 'eco-track-ai' after cache clear: {result}")

    # Test retrieving project status counts
    status_counts = await get_project_status_counts()
    print(f"Project Status Counts: {status_counts}")

if __name__ == "__main__":
    asyncio.run(main())