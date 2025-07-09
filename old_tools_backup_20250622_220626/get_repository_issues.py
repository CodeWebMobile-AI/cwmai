"""
AI-Generated Tool: get_repository_issues
Description: Get detailed list of issues for a specific repository
Generated: 2025-06-16T17:36:46.640684+00:00
Requirements: 
        Based on user request: create a tool that can retrieve the exact issue im not taking about the count, i want to know the content of the issue, the tool should be able to work with any repositories that we are currently managing
        Create a tool that: Get detailed list of issues for a specific repository
        Tool should integrate with existing system components
        
"""

from typing import Any
from typing import Dict
from typing import Dict, Any
from typing import List
import asyncio
import re
import json
import os
import logging
from datetime import datetime, timedelta
import traceback  # Import traceback

from scripts.github_issue_creator import GitHubIssueCreator
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.state_manager import StateManager
from scripts.storage_factory import StorageFactory
from scripts.local_github_simulator import LocalGitHubSimulator


"""
Module: get_repository_issues

Description:
    This module provides a tool to retrieve a detailed list of issues for a specific repository.
    It dynamically extracts the repository name from the user's input using pattern matching
    and leverages the MCP GitHub Integration to fetch the issue details.  It handles cases
    where the repository name cannot be determined from the context.
"""


__description__ = "Get detailed list of issues for a specific repository."
__parameters__ = {
    "repository_name": "The name of the repository to retrieve issues from. If not explicitly provided, it attempts to infer it from the user's input.",
    "max_issues": "The maximum number of issues to retrieve (default: 100).",
    "cache_expiry_hours": "The number of hours to cache the issues for (default: 24). Set to 0 to disable caching.",
    "issue_states": "A comma-separated list of issue states to filter by (e.g., open,closed).  Defaults to 'open'.",
    "labels": "A comma-separated list of labels to filter issues by.",
    "sort_by": "The field to sort the issues by (e.g., created_at, updated_at, comments). Defaults to 'created_at'.",
    "sort_order": "The order to sort the issues in (e.g., asc, desc). Defaults to 'desc'.",
    "include_closed_at": "Whether to include the 'closed_at' field in the output, even if null. Defaults to False.",
    "include_reactions": "Whether to include the reactions information (e.g., +1, -1, laugh). Defaults to False."
}
__examples__ = [
    "get repository issues for CodeWebMobile-AI/mindleap-wiki",
    "list issues for repository mindleap-wiki",
    "retrieve issues from mindleap-wiki",
    "get issues for my-repo with max_issues=50",
    "get issues for my-repo with cache_expiry_hours=12",
    "get open issues for my-repo with issue_states=open",
    "get issues for my-repo with labels=bug,enhancement",
    "get issues for my-repo sorted by updated_at",
    "get issues for my-repo sorted by comments in ascending order",
    "get issues for my-repo including the closed_at date",
    "get issues for my-repo including reaction counts"
]

CACHE_DIR = "issue_cache"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def get_repository_issues(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves a detailed list of issues for a specified repository, with caching, error handling, and filtering.

    Args:
        **kwargs: Keyword arguments containing context and potentially the repository name,
                   max_issues, cache_expiry_hours, issue_states and labels.

    Returns:
        A dictionary containing the issues found in the repository.

    Raises:
        ValueError: If the repository name cannot be determined from the context or if invalid input is provided.
        Exception: If any other error occurs during the process.
    """

    # Initialize storage factory to detect mode
    storage_factory = StorageFactory()
    storage_mode = storage_factory.storage_mode
    
    state_manager = StateManager()
    state = state_manager.get_state()
    
    if storage_mode == "local":
        local_simulator = LocalGitHubSimulator()
        logger.info("Using local storage mode")
    else:
        mcp_creator = MCPGitHubIssueCreator()
        logger.info("Using GitHub storage mode")

    user_input = kwargs.get('context', {}).get('user_input', '')
    max_issues = kwargs.get('max_issues', 100)  # Default max issues
    cache_expiry_hours = kwargs.get('cache_expiry_hours', 24)  # Default cache expiry
    issue_states = kwargs.get('issue_states', 'open').split(',')  # Default to 'open' state, allow multiple
    labels = kwargs.get('labels', '').split(',') # Default to no labels.
    sort_by = kwargs.get('sort_by', 'created_at')  # Default sort by created_at
    sort_order = kwargs.get('sort_order', 'desc')  # Default sort order desc
    include_closed_at = kwargs.get('include_closed_at', False)  # Default is False
    include_reactions = kwargs.get('include_reactions', False) # Default is False


    # Input Validation
    if not isinstance(max_issues, int) or max_issues <= 0:
        raise ValueError("max_issues must be a positive integer.")

    if not isinstance(cache_expiry_hours, int) or cache_expiry_hours < 0:
        raise ValueError("cache_expiry_hours must be a non-negative integer.")

    if not isinstance(issue_states, list):
      raise ValueError("issue_states must be a list")

    if not isinstance(labels, list):
      raise ValueError("labels must be a list")

    if sort_by not in ['created_at', 'updated_at', 'comments']:
        raise ValueError("sort_by must be one of 'created_at', 'updated_at', 'comments'.")

    if sort_order not in ['asc', 'desc']:
        raise ValueError("sort_order must be one of 'asc', 'desc'.")

    if not isinstance(include_closed_at, bool):
        raise ValueError("include_closed_at must be a boolean.")

    if not isinstance(include_reactions, bool):
        raise ValueError("include_reactions must be a boolean.")


    repository_name = kwargs.get('repository_name')  # Check if repository_name is passed directly

    if not repository_name:
        # Attempt to extract the repository name from the user input using pattern matching
        match = re.search(r"(?:repository|repo)\s+([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)?)", user_input)
        if match:
            repository_name = match.group(1)
        else:
            match = re.search(r"([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)?)", user_input)
            if match:
                repository_name = match.group(1)

    if not repository_name:
        raise ValueError("Could not determine the repository name from the user input. Please specify the repository name.")

    if not validate_repository_name(repository_name):
        raise ValueError("Invalid repository name format.  Should be in the format owner/repo.")

    # Normalizing inputs
    issue_states = [state.strip().lower() for state in issue_states]
    labels = [label.strip().lower() for label in labels]


    # Caching Logic
    cache_key = f"{repository_name}_issues_{max_issues}_{cache_expiry_hours}_{'_'.join(issue_states)}_{'_'.join(labels)}_{sort_by}_{sort_order}_{include_closed_at}_{include_reactions}.json"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if cache_expiry_hours > 0 and os.path.exists(cache_path):
        try:
            last_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - last_modified_time < timedelta(hours=cache_expiry_hours):
                logger.info(f"Using cached issues for {repository_name} from {last_modified_time}")
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                return {"issues": cached_data, "cache_hit": True, "source": "cache"} # Indicate cache hit
        except (json.JSONDecodeError, OSError) as e: #Catch more specific exceptions
            logger.warning(f"Error reading cache for {repository_name}: {e}. Fetching fresh data.")
            # Attempt to recover by deleting the corrupted cache file
            try:
                os.remove(cache_path)
                logger.info(f"Corrupted cache file {cache_path} deleted.")
            except OSError as e:
                logger.error(f"Failed to delete corrupted cache file {cache_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading cache: {e}.  Fetching fresh data.")

    try:
        if storage_mode == "local":
            # Fetch issues from local storage
            all_issues = local_simulator.get_issues(repository_name)
            
            # Filter by state
            filtered_issues = []
            for issue in all_issues:
                if issue.state in issue_states:
                    filtered_issues.append(issue)
            
            # Limit the number of issues
            issues = filtered_issues[:max_issues]
            
            # Convert LocalIssue objects to dicts
            issues_data = []
            for issue in issues:
                issue_dict = issue.to_dict() if hasattr(issue, 'to_dict') else {
                    'number': issue.number,
                    'title': issue.title,
                    'body': issue.body,
                    'state': issue.state,
                    'labels': issue.labels if hasattr(issue, 'labels') else [],
                    'created_at': issue.created_at.isoformat() if hasattr(issue, 'created_at') else None,
                    'updated_at': issue.updated_at.isoformat() if hasattr(issue, 'updated_at') else None,
                    'html_url': issue.html_url if hasattr(issue, 'html_url') else f"local://issue/{issue.number}",
                    'user': {'login': 'local_user'},
                    'assignees': [],
                    'comments': 0
                }
                issues_data.append(issue_dict)
            
            issues = issues_data
            source = "local"
        else:
            # Fetch from GitHub using MCP
            issues = await mcp_creator.list_recent_issues(repo=repository_name, limit=max_issues, states=issue_states, labels=labels)
            source = "github"

        if not issues:
            logger.info(f"No issues found for repository {repository_name} with the specified criteria.")
            return {"issues": [], "cache_hit": False, "source": source, "message": "No issues found matching the criteria."}


        # Helper function to extract labels
        def extract_labels(labels):
            """Extract label names from labels that could be strings or dicts."""
            result = []
            for label in labels:
                if isinstance(label, str):
                    result.append(label)
                elif isinstance(label, dict):
                    result.append(label.get("name", ""))
            return result
        
        # Enrich Issue Data
        enriched_issues = []
        for issue in issues:
            try:
                #Basic sanitization for display
                body_preview = issue.get("body", "")[:100]  # First 100 characters of body
                body_preview = body_preview.replace("\r", "").replace("\n", " ") # Remove newlines

                # Extract assignees' usernames
                assignees = [assignee.get("login", "Unknown") for assignee in issue.get("assignees", [])]

                enriched_issue = {
                    "title": issue.get("title"),
                    "number": issue.get("number"),
                    "url": issue.get("html_url"),
                    "state": issue.get("state"),
                    "body_preview": body_preview,  # Short snippet of the issue body
                    "created_at": issue.get("created_at"),
                    "updated_at": issue.get("updated_at"),
                    "user": issue.get("user", {}).get("login", "Unknown"),
                    "labels": extract_labels(issue.get("labels", [])),  # List of label names
                    "comments_count": issue.get("comments", 0),  # Add count of comments
                    "assignees": assignees  # List of assignees
                }

                if include_closed_at:
                    enriched_issue["closed_at"] = issue.get("closed_at")

                if include_reactions:
                    # Simplified example:  Just count the +1 reactions
                    reactions = issue.get("reactions", {})
                    enriched_issue["plus_one_reactions"] = reactions.get("+1", 0)


                enriched_issues.append(enriched_issue)
            except Exception as enrichment_error:
                logger.error(f"Error enriching issue {issue.get('number', 'unknown')}: {enrichment_error}", exc_info=True)

        # Sort the issues
        enriched_issues = sort_issues(enriched_issues, sort_by, sort_order)

        # Caching (if enabled)
        if cache_expiry_hours > 0:
            os.makedirs(CACHE_DIR, exist_ok=True)
            try:
                with open(cache_path, "w") as f:
                    json.dump(enriched_issues, f, indent=4) #Add indent for readability
                logger.info(f"Cached issues for {repository_name} to {cache_path}")
            except (OSError, TypeError) as e: # Catch specific exceptions.
                logger.error(f"Error writing cache to {cache_path}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error writing cache: {e}", exc_info=True)


        return {"issues": enriched_issues, "cache_hit": False, "source": source} # Indicate cache miss
    except Exception as e:
        logger.exception(f"An error occurred while fetching issues for {repository_name}: {e}")  # Log full exception info
        # More specific error handling for known GitHub API errors could be added here
        if "API rate limit exceeded" in str(e):
            logger.warning("Consider reducing the frequency of requests or using authentication.")
            # Suggest a recovery action.
            return {"error": "API rate limit exceeded. Please wait a while or authenticate to increase the limit.", "source": source if 'source' in locals() else "unknown", "retry_after": 60} #Added retry after
        elif "Not Found" in str(e):
            return {"error": f"Repository '{repository_name}' not found. Please check the repository name.", "source": source if 'source' in locals() else "unknown"}
        else:
            #Include more details about the error
            return {"error": f"An error occurred: {e}", "traceback": traceback.format_exc(), "source": source if 'source' in locals() else "unknown"}


def validate_repository_name(repo_name: str) -> bool:
    """
    Validates the repository name format.  Should be in the format owner/repo.
    """
    if not isinstance(repo_name, str):
        return False

    parts = repo_name.split("/")
    if len(parts) != 2:
        return False

    owner, repo = parts
    if not (owner and repo):  # Ensure both parts are non-empty
        return False

    # Add more sophisticated validation using regular expression
    pattern = r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$"
    if not re.match(pattern, repo_name):
        return False

    return True

def sort_issues(issues: List[Dict[str, Any]], sort_by: str, sort_order: str) -> List[Dict[str, Any]]:
    """Sorts a list of issues based on a specified field and order."""

    reverse = sort_order == 'desc'  # Determine reverse order

    try:
        return sorted(issues, key=lambda issue: issue.get(sort_by), reverse=reverse)
    except KeyError:
        logger.error(f"Cannot sort by key {sort_by}. Returning unsorted list.")
        return issues
    except Exception as e:
        logger.error(f"An error occurred during sorting: {e}. Returning unsorted list.")
        return issues

def clear_cache(repository_name: str = None) -> None:
    """Clears the issue cache. If repository_name is provided, only clears the cache for that repository."""
    if not os.path.exists(CACHE_DIR):
        logger.info("Cache directory does not exist.")
        return

    files = os.listdir(CACHE_DIR)
    for filename in files:
        if filename.endswith(".json"):
            if repository_name is None or repository_name in filename:
                file_path = os.path.join(CACHE_DIR, filename)
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted cache file: {filename}")
                except OSError as e:
                    logger.error(f"Error deleting cache file {filename}: {e}")
            else:
                logger.debug(f"Skipping cache file: {filename} (not matching repository {repository_name})")

def get_cache_size() -> int:
    """Returns the total size of the cache directory in bytes."""
    total_size = 0
    if os.path.exists(CACHE_DIR):
        for dirpath, dirnames, filenames in os.walk(CACHE_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except OSError as e:
                    logger.error(f"Error getting size of file {fp}: {e}")
    return total_size

async def test_get_repository_issues():
  # Example Usage (Simulate Tool Execution)
  try:
    # Create CACHE_DIR if it doesn't exist
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Test 1: Successful retrieval
    result1 = await get_repository_issues(context={'user_input': 'get issues for my-org/my-repo'})
    print("Test 1 Result:", len(result1['issues']))

    # Test 2: Explicit repository name
    result2 = await get_repository_issues(repository_name='explicit-org/explicit-repo', context={'user_input': 'irrelevant'})
    print("Test 2 Result:", len(result2['issues']))

    # Test 3: Handling no repository found
    try:
      await get_repository_issues(context={'user_input': 'get issues without a repo'})
    except ValueError as e:
      print("Test 3 (Expected Error):", e)

    # Test 4: Using max_issues
    result4 = await get_repository_issues(context={'user_input': 'get issues for small-repo/small-repo'}, max_issues=5)
    print("Test 4 Result:", len(result4['issues']))

    # Test 5: Caching (assuming a few runs with sufficient cache expiry)
    result5 = await get_repository_issues(context={'user_input': 'get issues for cached-repo/cached-repo'}, cache_expiry_hours=1)
    print("Test 5 Result:", len(result5['issues']), "Cache Hit:", result5.get("cache_hit"))

    #Test 6: Bad repository name
    try:
      validate_repository_name("bad-repo")
    except Exception as e:
      print("Test 6 Expected Error", e)

    # Test 7: issue_states filter
    result7 = await get_repository_issues(context={'user_input': 'get issues for state-repo/state-repo'}, issue_states='open,closed')
    print("Test 7 Result:", len(result7['issues']))

    # Test 8: labels filter
    result8 = await get_repository_issues(context={'user_input': 'get issues for label-repo/label-repo'}, labels='bug')
    print("Test 8 Result:", len(result8['issues']))
    
    # Test 9: combined filters and explicit repo name
    result9 = await get_repository_issues(repository_name='combined-repo/combined-repo', issue_states='open', labels='enhancement', max_issues=20)
    print("Test 9 Result:", len(result9['issues']))

    # Test 10: Cache expiry zero, no caching
    await get_repository_issues(context={'user_input': 'get issues for no-cache/no-cache'}, cache_expiry_hours=0)
    # Optionally check the cache directory to confirm no file was created

    # Test 11: Invalid inputs
    try:
      await get_repository_issues(repository_name='invalid-repo', max_issues='string')
    except ValueError as e:
        print(f"Test 11 Expected error: {e}")

    # Test 12: Sorting by created_at
    result12 = await get_repository_issues(context={'user_input': 'get issues for sort-repo/sort-repo'}, sort_by='created_at', sort_order='asc')
    print("Test 12 Result:", len(result12['issues']))

    # Test 13: Clearing cache
    clear_cache("my-org/my-repo")
    print("Test 13: Cache cleared for my-org/my-repo")

    # Test 14: Repository not found
    result14 = await get_repository_issues(repository_name='nonexistent-org/nonexistent-repo')
    if "error" in result14 and "Repository 'nonexistent-org/nonexistent-repo' not found" in result14["error"]:
        print("Test 14: Repository Not Found Handled")
    else:
        print("Test 14: Repository Not Found NOT Handled. Unexpected result:", result14)

    # Test 15: Include closed_at
    result15 = await get_repository_issues(repository_name='closed-at-repo/closed-at-repo', include_closed_at=True)
    print("Test 15 Result: Include Closed At", result15['issues'])

    # Test 16: Include reactions
    result16 = await get_repository_issues(repository_name='reaction-repo/reaction-repo', include_reactions=True)
    print("Test 16 Result: Include Reactions", result16['issues'])

    # Test 17: Get cache size
    cache_size = get_cache_size()
    print(f"Test 17: Cache size: {cache_size} bytes")


  except Exception as e:
    print("Test Failed:", e)


if __name__ == '__main__':
  asyncio.run(test_get_repository_issues())