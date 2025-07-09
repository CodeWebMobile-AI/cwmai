"""
AI-Generated Tool: get_repositories_evolved
Description: Get repositories with optional filtering and sorting
Generated: 2025-06-15
Requirements: Get list of repositories with filtering by name/language
"""

from typing import List, Dict, Any, Optional
import asyncio
import time
import logging
import re  # Import the 're' module for regular expressions
from datetime import datetime, timezone
import os
import sys

# Add scripts directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

__description__ = "Get repositories with optional filtering and sorting. To get newest repos, use sort_by='created_at' with sort_order='desc'"
__parameters__ = {
    "filter": {
        "type": "string",
        "description": "Optional filter string to match repository names or languages",
        "required": False
    },
    "limit": {
        "type": "integer",
        "description": "Maximum number of repositories to return",
        "required": False,
        "default": 10
    },
    "sort_by": {
        "type": "string",
        "description": "Field to sort repositories by. Use 'created_at' for newest/oldest repos, 'updated_at' for recently updated, 'stars' for popularity",
        "required": False,
        "default": "stars",
        "enum": ["name", "stars", "issues", "created_at", "updated_at"]
    },
    "sort_order": {
        "type": "string",
        "description": "Sort order (asc or desc)",
        "required": False,
        "default": "desc",
        "enum": ["asc", "desc"]
    },
    "min_stars": {
        "type": "integer",
        "description": "Minimum number of stars for a repository to be included",
        "required": False,
        "default": 0
    },
    "include_forks": {
        "type": "boolean",
        "description": "Include forked repositories in the results",
        "required": False,
        "default": False
    },
    "owner": {
        "type": "string",
        "description": "Filter repositories by owner (username/organization)",
        "required": False,
        "default": None
    },
    "archived": {
        "type": "boolean",
        "description": "Filter repositories by archived status",
        "required": False,
        "default": None
    },
    "exclude_patterns": {
        "type": "array",
        "description": "List of patterns to exclude repositories by name (using regular expressions)",
        "items": {"type": "string"},
        "required": False,
        "default": []
    },
    "language": {
        "type": "string",
        "description": "Filter by a specific programming language.",
        "required": False,
        "default": None
    },
    "pushed_since": {  # New parameter
        "type": "integer",
        "description": "Filter repositories that have been pushed to within the last X days.",
        "required": False,
        "default": None
    },
    "created_since": {
        "type": "integer",
        "description": "FILTER (not sort!) - Only show repos created within the last X days. This is different from sort_by='created_at'",
        "required": False,
        "default": None
    },
    "has_issues": {
        "type": "boolean",
        "description": "Filter repositories that have open issues.",
        "required": False,
        "default": None
    }
}
__examples__ = [
    {
        "command": "get_repositories_evolved",
        "description": "Get all repositories"
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"filter": "python", "limit": 5},
        "description": "Get top 5 Python repositories"
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"sort_by": "name", "sort_order": "asc"},
        "description": "Get all repositories sorted by name in ascending order"
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"min_stars": 100, "limit": 3},
        "description": "Get top 3 repositories with at least 100 stars"
    },
    {
      "command": "get_repositories_evolved",
      "parameters": {"sort_by": "created_at", "sort_order": "desc", "limit": 5},
      "description": "Get the 5 most recently created repositories."
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"owner": "my_org"},
        "description": "Get repositories belonging to the owner 'my_org'."
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"archived": True, "limit": 5},
        "description": "Get the first 5 archived repositories."
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"exclude_patterns": ["temp.*", "archive-.*"]},
        "description": "Exclude repositories with names containing 'temp' or 'archive'."
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"language": "JavaScript", "limit": 5},
        "description": "Get top 5 JavaScript repositories"
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"pushed_since": 30, "limit": 5},
        "description": "Get the 5 most recently updated repositories pushed in the last 30 days."
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"created_since": 7, "limit": 5},
        "description": "Get the 5 most recently created repositories created in the last 7 days."
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"has_issues": True, "limit": 5},
        "description": "Get the 5 repositories with open issues."
    }
]

# Cache for 10 seconds
_repositories_cache: Optional[List[Dict[str, Any]]] = None
_repositories_cache_time: Optional[float] = None
_repositories_cache_ttl = 10
_repositories_cache_args: Optional[Dict[str, Any]] = None

def _matches_exclude_patterns(repo_name: str, exclude_patterns: List[str]) -> bool:
    """
    Checks if the repository name matches any of the exclude patterns using regular expressions.

    Args:
        repo_name: The name of the repository.
        exclude_patterns: A list of regular expression patterns to exclude repositories by name.

    Returns:
        True if the repository name matches any of the exclude patterns, False otherwise.
    """
    try:
        for pattern in exclude_patterns:
            if re.search(pattern, repo_name, re.IGNORECASE):
                return True
        return False
    except re.error as e:
        logging.error(f"Invalid regular expression pattern: {e}")
        return False


async def _load_repositories_from_state() -> List[Dict[str, Any]]:
    """Loads repositories from the state manager or local storage."""
    from scripts.state_manager import StateManager
    from scripts.storage_factory import StorageFactory
    from scripts.local_github_simulator import LocalGitHubSimulator

    storage_factory = StorageFactory()
    storage_mode = storage_factory.storage_mode
    
    if storage_mode == "local":
        # Use local storage
        local_simulator = LocalGitHubSimulator()
        repositories = []
        
        try:
            # List all local repositories
            local_repos = local_simulator.list_repositories()
            
            for repo in local_repos:
                # LocalRepository objects have attributes, not dict items
                repositories.append({
                    'name': repo.name,
                    'description': repo.description or '',
                    'language': '',  # Local repos don't track language
                    'stars': 0,  # Local repos don't have stars
                    'issues': 0,  # Will be updated if needed
                    'created_at': repo.created_at.isoformat() if hasattr(repo, 'created_at') else None,
                    'updated_at': repo.updated_at.isoformat() if hasattr(repo, 'updated_at') else None,
                    'is_fork': False,  # Local repos are not forks
                    'owner': repo.full_name.split('/')[0] if '/' in repo.full_name else 'local_user',
                    'archived': False  # Local repos don't track archived status
                })
            
            logging.info(f"Loaded {len(repositories)} repositories from local storage")
            return repositories
            
        except Exception as e:
            logging.exception(f"Error loading repositories from local storage: {e}")
            return []
    
    # Use GitHub/state manager (existing code)
    state_manager = StateManager()

    try:
        if hasattr(state_manager, "force_reload_state"):
            try:
                # force_reload_state is synchronous, not async
                state = state_manager.force_reload_state()
                logging.info("Successfully reloaded state from state manager.")

            except Exception as e:
                logging.error(f"Error reloading state: {e}. Using cached state, if available, or an empty list.")
                if _repositories_cache:
                    return _repositories_cache
                else:
                    return []


        else:
            try:
                state = await asyncio.wait_for(asyncio.to_thread(state_manager.load_state), timeout=5)
                logging.info("Successfully loaded state from state manager.")
            except asyncio.TimeoutError:
                logging.error("State load timed out. Using cached state, if available, or an empty list.")
                if _repositories_cache:
                    return _repositories_cache
                else:
                    return []

        repositories = []
        for repo_name, repo_data in state.get('projects', {}).items():
            if repo_data is None:
                logging.warning(f"Repository data for '{repo_name}' is None. Skipping.")
                continue
            try:
                repositories.append({
                    'name': repo_name,
                    'description': repo_data.get('description', ''),
                    'language': repo_data.get('language', ''),
                    'stars': repo_data.get('stars', 0),
                    'issues': repo_data.get('open_issues_count', 0),
                    'created_at': repo_data.get('created_at', None),
                    'updated_at': repo_data.get('pushed_at', None),  # Added updated_at
                    'is_fork': repo_data.get('is_fork', False),
                    'owner': repo_data.get('owner', None),
                    'archived': repo_data.get('archived', False)
                })
            except Exception as e:
                logging.error(f"Error processing repository '{repo_name}': {e}")
        return repositories
    except Exception as e:
        logging.exception(f"An error occurred while loading repositories from the state manager: {e}")
        return []

async def _validate_date_format(date_string: str) -> bool:
    """Validates if a date string conforms to the expected format."""
    try:
        datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
        return True
    except (ValueError, TypeError):
        return False


async def get_repositories_evolved(**kwargs) -> Dict[str, Any]:
    """Get repositories with optional filtering and sorting."""
    global _repositories_cache, _repositories_cache_time, _repositories_cache_args

    # Extract parameters with defaults and input validation
    filter_str = kwargs.get('filter', None)
    limit = kwargs.get('limit', 10)
    sort_by = kwargs.get('sort_by', 'stars')
    sort_order = kwargs.get('sort_order', 'desc')
    min_stars = kwargs.get('min_stars', 0)
    include_forks = kwargs.get('include_forks', False)
    owner = kwargs.get('owner', None)
    archived = kwargs.get('archived', None)
    exclude_patterns = kwargs.get('exclude_patterns', [])
    language = kwargs.get('language', None)
    pushed_since = kwargs.get('pushed_since', None) #New Parameter
    created_since = kwargs.get('created_since', None)  # New Parameter
    has_issues = kwargs.get('has_issues', None) # New Parameter

    # Input validation and sanitization

    if not isinstance(limit, int) or limit <= 0:
        limit = 10
        logging.warning(f"Invalid limit value '{kwargs.get('limit')}', using default value of {limit}.")
    else:
         limit = min(limit, 100) #Add upper limit
         logging.info(f"Limit set to {limit}.")


    if not isinstance(min_stars, int) or min_stars < 0:
        min_stars = 0
        logging.warning(f"Invalid min_stars value '{kwargs.get('min_stars')}', using default value of {min_stars}.")
    else:
         logging.info(f"Minimum stars set to {min_stars}.")


    if sort_by not in ['name', 'stars', 'issues', 'created_at', 'updated_at']:
        sort_by = 'stars'
        logging.warning(f"Invalid sort_by value '{kwargs.get('sort_by')}', using default value of '{sort_by}'.")
    else:
         logging.info(f"Sorting by {sort_by}.")

    if sort_order not in ['asc', 'desc']:
        sort_order = 'desc'
        logging.warning(f"Invalid sort_order value '{kwargs.get('sort_order')}', using default value of '{sort_order}'.")
    else:
         logging.info(f"Sort order set to {sort_order}.")

    if not isinstance(include_forks, bool):
        include_forks = False
        logging.warning(f"Invalid include_forks value '{kwargs.get('include_forks')}', using default value of {include_forks}.")
    else:
        logging.info(f"Including forks: {include_forks}.")

    if not isinstance(exclude_patterns, list):
        exclude_patterns = []
        logging.warning(f"Invalid exclude_patterns value '{kwargs.get('exclude_patterns')}', using default value of {exclude_patterns}.")
    elif not all(isinstance(pattern, str) for pattern in exclude_patterns):
        exclude_patterns = []
        logging.warning(f"Invalid exclude_patterns value: all patterns must be strings, using default value of {exclude_patterns}.")
    else:
        logging.info(f"Excluding patterns: {exclude_patterns}.")

    if filter_str:
        filter_str = filter_str.strip()
        logging.info(f"Applying filter: {filter_str}.")

    if language:
        language = language.strip()
        logging.info(f"Applying language filter: {language}.")

    if owner:
        owner = owner.strip()
        logging.info(f"Filtering by owner: {owner}.")

    if archived is not None and not isinstance(archived, bool):
        archived = None
        logging.warning(f"Invalid archived value '{kwargs.get('archived')}', ignoring archived filter.")
    else:
        logging.info(f"Filtering by archived: {archived}.")

    if pushed_since is not None:
        if not isinstance(pushed_since, int) or pushed_since <= 0:
            pushed_since = None
            logging.warning(f"Invalid pushed_since value '{kwargs.get('pushed_since')}', ignoring pushed_since filter.")
        else:
             logging.info(f"Filtering repositories pushed within the last {pushed_since} days.")

    if created_since is not None:
        if not isinstance(created_since, int) or created_since <= 0:
            created_since = None
            logging.warning(f"Invalid created_since value '{kwargs.get('created_since')}', ignoring created_since filter.")
        else:
            logging.info(f"Filtering repositories created within the last {created_since} days.")

    if has_issues is not None and not isinstance(has_issues, bool):
        has_issues = None
        logging.warning(f"Invalid has_issues value '{kwargs.get('has_issues')}', ignoring has_issues filter.")
    else:
        logging.info(f"Filtering by open issues: {has_issues}.")



    # Check cache. Caching only happens if no owner filter is used. Owner filter can change the result dynamically.
    cache_hit = False
    cache_key_args = {k: v for k, v in kwargs.items() if k not in ['exclude_patterns']}  # exclude_patterns from the cache key
    if owner is None and _repositories_cache is not None and _repositories_cache_time is not None and time.time() - _repositories_cache_time < _repositories_cache_ttl and _repositories_cache_args == cache_key_args:
        repositories = _repositories_cache.copy()  # Return a copy to prevent modification of the cache
        logging.info("Using cached repositories.")
        cache_hit = True
    else:
        try:
            repositories = await _load_repositories_from_state()
            # Apply owner filter immediately after loading from state to ensure up-to-date filtering
            if owner:
                repositories = [repo for repo in repositories if repo.get('owner') == owner]
                logging.info(f"Applied owner filter: {owner}. {len(repositories)} repositories remain.")

            # Cache the repositories *before* applying the language filter.  This allows the language filter to be used with caching.
            _repositories_cache = repositories.copy()  # Store a copy in the cache
            _repositories_cache_time = time.time()
            _repositories_cache_args = cache_key_args # Store args except for exclude_patterns
            logging.info("Repositories loaded from state and cached.")
        except Exception as e:
            logging.exception(f"Error loading repositories: {e}")
            return {'total_count': 0, 'repositories': [], 'error': str(e), 'cache_hit': False}

    # Apply filter
    if filter_str:
        filtered_repositories = []
        for repo in repositories:
            if filter_str.lower() in repo['name'].lower() or (repo.get('language') and filter_str.lower() in repo['language'].lower()):
                filtered_repositories.append(repo)
        repositories = filtered_repositories
        logging.info(f"Applied filter: {filter_str}.  {len(repositories)} repositories remain.")

    # Apply language filter
    if language:
        repositories = [repo for repo in repositories if repo.get('language', '').lower() == language.lower()]
        logging.info(f"Applied language filter: {language}.  {len(repositories)} repositories remain.")

    # Apply exclude_patterns filter
    if exclude_patterns:
        repositories = [repo for repo in repositories if not _matches_exclude_patterns(repo['name'], exclude_patterns)]
        logging.info(f"Applied exclude_patterns filter: {exclude_patterns}. {len(repositories)} repositories remain.")

    # Apply min_stars filter
    repositories = [repo for repo in repositories if repo['stars'] >= min_stars]
    logging.info(f"Applied min_stars filter: {min_stars}. {len(repositories)} repositories remain.")

    # Filter out forks if include_forks is False
    if not include_forks:
        repositories = [repo for repo in repositories if not repo['is_fork']]
        logging.info(f"Excluded forks. {len(repositories)} repositories remain.")

    # Apply archived filter
    if archived is not None:
        repositories = [repo for repo in repositories if repo.get('archived') == archived]
        logging.info(f"Applied archived filter: {archived}. {len(repositories)} repositories remain.")

    # Apply pushed_since filter
    if pushed_since is not None:
        cutoff_date = time.time() - (pushed_since * 24 * 3600)  # Calculate cutoff timestamp in seconds
        valid_repos = []
        for repo in repositories:
            updated_at = repo.get('updated_at')
            if updated_at:
                if await _validate_date_format(updated_at):
                    updated_at_timestamp = time.mktime(datetime.strptime(updated_at, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timetuple())
                    if updated_at_timestamp > cutoff_date:
                        valid_repos.append(repo)
                else:
                    logging.warning(f"Invalid date format for updated_at: {updated_at}. Skipping repository.")
            else:
                logging.warning("updated_at is missing for a repository. Skipping.")
        repositories = valid_repos
        logging.info(f"Applied pushed_since filter: {pushed_since} days. {len(repositories)} repositories remain.")

    # Apply created_since filter
    if created_since is not None:
        cutoff_date = time.time() - (created_since * 24 * 3600)
        valid_repos = []
        for repo in repositories:
            created_at = repo.get('created_at')
            if created_at:
                if await _validate_date_format(created_at):
                    created_at_timestamp = time.mktime(datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timetuple())
                    if created_at_timestamp > cutoff_date:
                        valid_repos.append(repo)
                else:
                    logging.warning(f"Invalid date format for created_at: {created_at}. Skipping repository.")
            else:
                logging.warning("created_at is missing for a repository. Skipping.")
        repositories = valid_repos
        logging.info(f"Applied created_since filter: {created_since} days. {len(repositories)} repositories remain.")

    # Apply has_issues filter
    if has_issues is not None:
        if has_issues:
            repositories = [repo for repo in repositories if repo.get('issues', 0) > 0]
        else:
            repositories = [repo for repo in repositories if repo.get('issues', 0) == 0]
        logging.info(f"Applied has_issues filter: {has_issues}. {len(repositories)} repositories remain.")


    # Sort
    reverse_sort = sort_order == 'desc'
    try:
        # Handle missing or None values during sorting
        repositories.sort(
            key=lambda x: (x[sort_by] if x.get(sort_by) is not None else -1) if isinstance(x.get(sort_by), (int, float)) else (str(x.get(sort_by, "")).lower() if x.get(sort_by) is not None else ""),
            reverse=reverse_sort
        )
        logging.info(f"Sorted repositories by {sort_by} in {sort_order} order.")
    except KeyError as e:
        logging.warning(f"Invalid sort_by key '{sort_by}'. Sorting by stars instead. Error: {e}")
        repositories.sort(key=lambda x: x['stars'], reverse=True)  # Default back to stars
    except Exception as e:
        logging.error(f"Error during sorting: {e}. Returning unsorted results.")


    # Limit and prepare output
    limited_repositories = repositories[:limit]

    # Enhance output with total_count and potentially an error message
    output = {
        'total_count': len(repositories),
        'repositories': limited_repositories,
        'cache_hit': cache_hit,
        'applied_filters': {k: v for k, v in kwargs.items() if v is not None} # return applied filters
    }

    return output


# Example usage (for testing purposes):
async def main():
    # Example 1: Get all repositories
    result1 = await get_repositories_evolved()
    print("Example 1:", result1)

    # Example 2: Get top 5 Python repositories
    result2 = await get_repositories_evolved(filter="python", limit=5)
    print("Example 2:", result2)

    # Example 3: Get all repositories sorted by name in ascending order
    result3 = await get_repositories_evolved(sort_by="name", sort_order="asc", limit=3)
    print("Example 3:", result3)

    # Example 4: Test caching
    result4 = await get_repositories_evolved(limit=2)
    print("Example 4 (First run):", result4)
    result5 = await get_repositories_evolved(limit=2)
    print("Example 5 (Cached run):", result5)  # Should be nearly instant due to cache
    await asyncio.sleep(11)  # Wait for cache to expire
    result6 = await get_repositories_evolved(limit=2)
    print("Example 6 (After cache expired):", result6)

    #Example 7: Test min_stars
    result7 = await get_repositories_evolved(min_stars=100)
    print("Example 7 (min_stars=100):", result7)

    #Example 8: Test include_forks
    result8 = await get_repositories_evolved(include_forks=True, limit=3)
    print("Example 8 (include_forks=True):", result8)

    # Example 9: Sorting by created_at
    result9 = await get_repositories_evolved(sort_by="created_at", sort_order="asc", limit=3)
    print("Example 9 (Sorting by created_at):", result9)

    #Example 10:  Test filtering and sorting with caching.
    result10 = await get_repositories_evolved(filter="my_repo", sort_by="name", sort_order="asc", limit=2)
    print("Example 10 (Filtered and Sorted):", result10)

    # Example 11: Test filtering by owner
    result11 = await get_repositories_evolved(owner="my_org", limit=3)
    print("Example 11 (Filtering by owner):", result11)

    # Example 12: Test filtering by archived status
    result12 = await get_repositories_evolved(archived=True, limit=3)
    print("Example 12 (Filtering by archived status):", result12)

    # Example 13: Test exclude_patterns
    result13 = await get_repositories_evolved(exclude_patterns=["temp.*", "archive-.*"], limit=5)
    print("Example 13 (exclude_patterns):", result13)

    # Example 14: Test language filtering
    result14 = await get_repositories_evolved(language="JavaScript", limit=5)
    print("Example 14 (language=JavaScript):", result14)

    #Example 15: Test pushed_since
    result15 = await get_repositories_evolved(pushed_since=30, limit=5)
    print("Example 15 (pushed_since=30):", result15)

    #Example 16: Test created_since
    result16 = await get_repositories_evolved(created_since=7, limit=5)
    print("Example 16 (created_since=7):", result16)

    #Example 17: Test has_issues
    result17 = await get_repositories_evolved(has_issues=True, limit=5)
    print("Example 17 (has_issues=True):", result17)

if __name__ == "__main__":
    asyncio.run(main())