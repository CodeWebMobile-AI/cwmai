"""
AI-Generated Tool: tool_task_5
Description: Prepare the final output by combining the repository count and total stars.
Generated: 2025-06-15T12:00:35.727371+00:00
Requirements: Tool to Prepare the final output by combining the repository count and total stars.
"""

from scripts.state_manager import StateManager


"""
Module for preparing the final output by combining repository count and total stars.

This tool aggregates GitHub repository statistics by combining the total count of repositories
with the sum of all stars across those repositories to produce a summary report.
"""


__description__ = "Prepare the final output by combining the repository count and total stars."
__parameters__ = {
    "repository_count": "int: Total number of repositories",
    "total_stars": "int: Sum of stars across all repositories"
}
__examples__ = [
    {
        "description": "Combine repository count and stars",
        "parameters": {
            "repository_count": 5,
            "total_stars": 42
        },
        "result": {
            "total_repositories": 5,
            "total_stars": 42,
            "summary": "5 repositories with 42 total stars"
        }
    }
]

async def tool_task_5(**kwargs):
    """
    Async function to prepare final output by combining repository count and total stars.

    Args:
        **kwargs: Expected keys 'repository_count' (int) and 'total_stars' (int)

    Returns:
        dict: Combined statistics with keys:
            - total_repositories (int)
            - total_stars (int)
            - summary (str)

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    state_manager = StateManager()
    state = state_manager.load_state()

    # Validate input parameters
    if 'repository_count' not in kwargs or 'total_stars' not in kwargs:
        raise ValueError("Both 'repository_count' and 'total_stars' parameters are required")
    
    try:
        repo_count = int(kwargs['repository_count'])
        stars = int(kwargs['total_stars'])
    except (ValueError, TypeError) as e:
        raise ValueError("Parameters must be valid integers") from e

    if repo_count < 0 or stars < 0:
        raise ValueError("Values cannot be negative")

    # Prepare result
    return {
        "total_repositories": repo_count,
        "total_stars": stars,
        "summary": f"{repo_count} repositories with {stars} total stars"
    }
