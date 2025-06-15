"""
AI-Generated Tool: count_repositories_evolved
Description: Count repositories with enhanced error handling and validation
Generated: 2025-06-15
Requirements: Count total repositories being managed with statistics
"""

from typing import Dict, Any
import logging

# Set up logging for this module
logger = logging.getLogger(__name__)

__description__ = "Count repositories with enhanced error handling and validation"
__parameters__ = {}
__examples__ = [
    {"command": "count_repositories_evolved", "description": "Count all managed repositories with statistics"}
]


class CountRepositoriesError(Exception):
    """Custom exception for count_repositories tool."""
    pass


async def count_repositories_evolved(**kwargs) -> Dict[str, Any]:
    """Count total repositories being managed."""
    from scripts.state_manager import StateManager
    
    state_manager = StateManager()
    
    try:
        # Attempt to reload state, gracefully falling back to loading if reload fails
        try:
            if hasattr(state_manager, "force_reload_state"):
                state = state_manager.force_reload_state()
            else:
                state = state_manager.load_state()
        except Exception as e:
            logger.warning(f"Failed to reload state. Falling back to loading. Error: {e}")
            state = state_manager.load_state()
    except Exception as e:
        logger.error(f"Failed to load state. Error: {e}")
        raise CountRepositoriesError("Failed to load state.") from e

    projects = state.get('projects', {})

    # Input validation: Ensure projects is a dictionary
    if not isinstance(projects, dict):
        logger.error(f"Invalid state format: projects is not a dictionary. Got: {type(projects)}")
        raise CountRepositoriesError("Invalid state format: Projects must be a dictionary.")

    # Count by various criteria
    total = len(projects)
    by_language = {}
    by_status = {'active': 0, 'archived': 0, 'unknown': 0}
    total_stars = 0
    total_issues = 0

    for repo_name, repo_data in projects.items():
        # Input validation: Ensure repo_data is a dictionary
        if not isinstance(repo_data, dict):
            logger.warning(f"Invalid repo data format for {repo_name}. Skipping. Got: {type(repo_data)}")
            continue  # Skip invalid repository data

        # Language count
        language = repo_data.get('language', 'Unknown')
        by_language[language] = by_language.get(language, 0) + 1

        # Status count
        is_archived = repo_data.get('archived', False)
        if not isinstance(is_archived, bool): #Type checking
            logger.warning(f"Invalid archived value for {repo_name}. Assuming 'False'. Got: {type(is_archived)}")
            is_archived = False

        if is_archived:
            by_status['archived'] += 1
        else:
            by_status['active'] += 1

        # Aggregate metrics
        stars = repo_data.get('stars', 0)
        open_issues_count = repo_data.get('open_issues_count', 0)
        
        # Type checking stars and open_issues_count
        if not isinstance(stars, (int, float)):
            logger.warning(f"Invalid stars value for {repo_name}. Skipping. Got: {type(stars)}")
            stars = 0
        if not isinstance(open_issues_count, (int, float)):
            logger.warning(f"Invalid open_issues_count value for {repo_name}. Skipping. Got: {type(open_issues_count)}")
            open_issues_count = 0

        total_stars += stars
        total_issues += open_issues_count

    try:
        avg_stars_per_repo = total_stars / max(total, 1)
    except ZeroDivisionError:
        logger.warning("Total is zero, cannot compute avg_stars_per_repo.")
        avg_stars_per_repo = 0.0

    return {
        "total": total,
        "breakdown": {
            "by_status": by_status,
            "by_language": by_language
        },
        "metrics": {
            "total_stars": total_stars,
            "total_open_issues": total_issues,
            "avg_stars_per_repo": avg_stars_per_repo
        },
        "summary": f"Managing {total} repositories ({by_status['active']} active, {by_status['archived']} archived)"
    }