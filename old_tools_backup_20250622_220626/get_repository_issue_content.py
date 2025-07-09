"""
AI-Generated Tool: get_repository_issue_content
Description: Retrieve information for repository issue content
Generated: 2025-06-16T17:40:07.450935+00:00
Requirements: 
        Tool Name: get_repository_issue_content
        Intent: Retrieve information for repository issue content
        Expected Parameters: {"context": {"user_input": " create a tool that can retrieve the exact issue im not taking about the count, i want to know the content of the issue, the tool should be able to work with any repositories that we are currently managing", "description": "Get the content of a specific issue from a repository. This tool requires the repository name and issue number."}}
        Category: query
        
        The tool should:
        Implement functionality for get_repository_issue_content
        
"""

from typing import Any
from typing import Dict
from typing import Dict, Any
from functools import lru_cache

import asyncio
import re
import logging

from scripts.github_issue_creator import GitHubIssueCreator  # Import even if potentially unused, for flexibility
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.repository_analyzer import RepositoryAnalyzer # Import even if potentially unused, for flexibility
from scripts.state_manager import StateManager
from scripts.storage_factory import StorageFactory
from scripts.local_github_simulator import LocalGitHubSimulator

"""
Module for retrieving information about repository issue content.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


__description__ = "Retrieve information for repository issue content."

__parameters__ = {
    "repository": {"type": "string", "description": "The name of the repository."},
    "issue_number": {"type": "integer", "description": "The issue number."}
}

__examples__ = [
    {"repository": "my_repo", "issue_number": 123},
    {"repository": "another_repo", "issue_number": 42}
]

MAX_CACHE_SIZE = 128 # Configuration option
@lru_cache(maxsize=MAX_CACHE_SIZE)
async def cached_get_issue(mcp_creator, repo, issue_number):
    """Cached function to retrieve issue details, improving performance."""
    try:
        logging.info(f"Attempting to retrieve issue #{issue_number} from {repo} (cached).")
        if hasattr(mcp_creator, 'get_issue'):
            issue = await mcp_creator.get_issue(repo=repo, issue_number=issue_number)
            logging.info(f"Successfully retrieved issue #{issue_number} from {repo} using get_issue.")
            return issue
        else:
            logging.info(f"get_issue method not found. Falling back to list_recent_issues for {repo}.")
            issues = await mcp_creator.list_recent_issues(repo=repo, limit=100)
            for i in issues:
                if i.get('number') == issue_number:
                    logging.info(f"Found issue #{issue_number} in list_recent_issues response for {repo}.")
                    return i
            logging.warning(f"Issue #{issue_number} not found in recent issues for {repo}.")
            return None
    except Exception as e:
        logging.error(f"Error in cached_get_issue: {e}", exc_info=True)
        return None

def is_valid_repository_name(repository: str) -> bool:
    """
    Validates the repository name format using a regular expression.
    Allows alphanumeric characters, hyphens, underscores, and dots.

    Args:
        repository: The repository name string.

    Returns:
        True if the repository name is valid, False otherwise.
    """
    pattern = r"^[\w\-\.]+$"  # Allows alphanumeric, hyphen, underscore, and dot.
    return bool(re.match(pattern, repository))

def validate_input(repository: str, issue_number: int) -> Dict[str, str]:
    """Validates the repository and issue number inputs."""
    errors = {}
    if not isinstance(repository, str) or not repository:
        errors["repository"] = "Repository name must be a non-empty string."
    elif not is_valid_repository_name(repository):
        errors["repository"] = "Invalid repository name format.  Use alphanumeric characters, hyphens, underscores, and dots."
    if not isinstance(issue_number, int):
        errors["issue_number"] = "Issue number must be an integer."
    elif issue_number <= 0:
        errors["issue_number"] = "Issue number must be a positive integer."
    return errors


async def extract_repo_and_issue(user_input: str) -> Dict[str, Any]:
    """Extracts the repository name and issue number from the user input using regex."""
    repository = None
    issue_number = None

    repo_match = re.search(r"(?:repository|repo)\s+([\w\-\.]+)", user_input, re.IGNORECASE)
    if repo_match:
        repository = repo_match.group(1)
        logging.debug(f"Extracted repository name: {repository} from user input.")

    issue_number_match = re.search(r"(?:issue|#)\s*(\d+)", user_input, re.IGNORECASE)
    if issue_number_match:
        try:
            issue_number = int(issue_number_match.group(1))
            logging.debug(f"Extracted issue number: {issue_number} from user input.")
        except ValueError:
            logging.warning("Invalid issue number format in user input.")
            return {"error": "Invalid issue number format in user input."}

    return {"repository": repository, "issue_number": issue_number}

def format_error(error_message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Formats an error message with optional details."""
    error_response = {"error": error_message}
    if details:
        error_response["details"] = details
    logging.error(error_message, extra={"details": details})
    return error_response

def sanitize_string(input_string: str) -> str:
    """
    Sanitizes a string by removing or replacing potentially harmful characters.
    This helps prevent injection attacks and ensures data integrity.
    """
    # Example: Remove HTML tags
    sanitized_string = re.sub('<[^<]+?>', '', input_string)
    # Example: Replace special characters with safe alternatives
    sanitized_string = sanitized_string.replace(";", "").replace("&", "&amp;")
    return sanitized_string

async def get_issue_comments(mcp_creator, repo: str, issue_number: int) -> list:
    """Retrieves comments for a given issue."""
    try:
        comments = await mcp_creator.get_issue_comments(repo=repo, issue_number=issue_number)
        if isinstance(comments, list):
            return [sanitize_string(comment.get('body', '')) if isinstance(comment, dict) else sanitize_string(getattr(comment, 'body', '')) for comment in comments]
        else:
            logging.warning(f"Unexpected comments format: {type(comments)}")
            return []
    except Exception as e:
        logging.error(f"Error retrieving comments for issue #{issue_number} in {repo}: {e}", exc_info=True)
        return []

def truncate_string(input_string: str, max_length: int = 500) -> str:
    """
    Truncates a string to a maximum length, adding an ellipsis if truncated.
    """
    if len(input_string) > max_length:
        return input_string[:max_length] + "..."
    return input_string

async def get_repository_issue_content(**kwargs) -> Dict[str, Any]:
    """
    Retrieves the content of a specific issue from a repository.

    Args:
        **kwargs: Keyword arguments containing context and parameters.

    Returns:
        A dictionary containing the issue details (title, body, number, state, url, reactions, comments) or an error message.
    """
    # Initialize storage factory to detect mode
    storage_factory = StorageFactory()
    storage_mode = storage_factory.storage_mode
    
    state_manager = StateManager()
    state = state_manager.get_state()
    
    if storage_mode == "local":
        local_simulator = LocalGitHubSimulator()
        logging.info("Using local storage mode")
    else:
        mcp_creator = MCPGitHubIssueCreator()
        logging.info("Using GitHub storage mode")
    user_input = kwargs.get('context', {}).get('user_input', '')

    repository = kwargs.get('repository')
    issue_number = kwargs.get('issue_number')
    
    # Input Validation
    if not repository or not issue_number:
        extracted_data = await extract_repo_and_issue(user_input)
        if not repository:
            repository = extracted_data.get("repository")
        if not issue_number:
            issue_number = extracted_data.get("issue_number")
    

    # More Robust Input Handling and Validation
    if not repository:
        return format_error("Repository name not found in parameters or user input. Please specify the repository.")
    if not issue_number:
        return format_error("Issue number not found in parameters or user input. Please specify the issue number.")

    try:
        issue_number = int(issue_number)  # Ensure issue_number is an integer after extraction
    except ValueError:
        return format_error("Invalid issue number format. Please provide a valid integer.")

    validation_errors = validate_input(repository, issue_number)
    if validation_errors:
        return format_error("Input validation failed.", details=validation_errors)

    try:
        logging.info(f"Fetching issue #{issue_number} from repository {repository}.")
        
        if storage_mode == "local":
            # Get issue from local storage
            issues = local_simulator.get_issues(repository)
            issue = None
            for local_issue in issues:
                if local_issue.number == issue_number:
                    # Convert LocalIssue to dict format
                    issue = local_issue.to_dict() if hasattr(local_issue, 'to_dict') else {
                        'number': local_issue.number,
                        'title': local_issue.title,
                        'body': local_issue.body,
                        'state': local_issue.state,
                        'html_url': local_issue.html_url if hasattr(local_issue, 'html_url') else f"local://issue/{local_issue.number}",
                        'user': {'login': 'local_user'},
                        'comments': 0,
                        'reactions': {}
                    }
                    break
        else:
            # Get issue from GitHub
            issue = await cached_get_issue(mcp_creator, repository, issue_number)

        if issue:
            # Enhanced Output: Including reactions (if available)
            reactions = {}
            if isinstance(issue, dict) and 'reactions' in issue:
                reactions = issue['reactions']
            elif hasattr(issue, 'reactions'):
                # Safely handle missing reaction attributes
                reactions = {k: getattr(issue.reactions, k, 0) if hasattr(issue.reactions, k) else 0 for k in ['+1', '-1', 'laugh', 'confused', 'heart', 'rocket', 'eyes']} # Assuming reactions is an object with these attributes

            comments = await get_issue_comments(mcp_creator, repository, issue_number)

            # Handle both object and dict responses
            if isinstance(issue, dict):
                issue_data = {
                    "title": sanitize_string(issue.get('title', '')), # Sanitize the title
                    "body": sanitize_string(issue.get('body', '')),   # Sanitize the body
                    "number": issue.get('number', issue_number),
                    "state": issue.get('state', 'unknown'),
                    "url": issue.get('html_url', '') or issue.get('url', ''),
                    "reactions": reactions,
                    "repository": repository,
                    "author": issue.get('user', {}).get('login', 'unknown'),  # Added issue author
                    "comments": comments,
                    "created_at": issue.get('created_at', None),  # Added issue creation timestamp
                    "updated_at": issue.get('updated_at', None),  # Added issue update timestamp
                    "closed_at": issue.get('closed_at', None),    # Added issue close timestamp
                    "labels": [label.get('name', '') if isinstance(label, dict) else getattr(label, 'name', '') for label in issue.get('labels', [])] # Added issue labels, handling both dict and object responses
                }
            else:
                issue_data = {
                    "title": sanitize_string(getattr(issue, 'title', '')), # Sanitize the title
                    "body": sanitize_string(getattr(issue, 'body', '')),   # Sanitize the body
                    "number": getattr(issue, 'number', issue_number),
                    "state": getattr(issue, 'state', 'unknown'),
                    "url": getattr(issue, 'html_url', '') or getattr(issue, 'url', ''),
                    "reactions": reactions,
                    "repository": repository,
                    "author": getattr(issue.user, 'login', 'unknown') if hasattr(issue, 'user') else 'unknown', #Added issue author
                    "comments": comments,
                     "created_at": getattr(issue, 'created_at', None),  # Added issue creation timestamp
                    "updated_at": getattr(issue, 'updated_at', None),  # Added issue update timestamp
                    "closed_at": getattr(issue, 'closed_at', None),    # Added issue close timestamp
                    "labels": [getattr(label, 'name', '') for label in getattr(issue, 'labels', [])] # Added issue labels, handling potential AttributeError
                }
            logging.info(f"Successfully retrieved issue #{issue_number} from repository {repository}.")
            return issue_data
        else:
            error_message = f"Issue #{issue_number} not found in repository {repository}."
            logging.warning(error_message)
            return format_error(error_message)
    except AttributeError as e:
        # Dynamic method creation should handle this, but provide fallback
        if 'get_issue' in str(e):
            logging.warning(f"get_issue method missing. Attempting dynamic creation.  Original error: {e}")
            try:
                # Create the method dynamically
                from scripts.dynamic_method_creator import dynamic_method_creator
                
                context = {
                    'args': (),
                    'kwargs': {'repo': repository, 'issue_number': issue_number},
                    'user_intent': kwargs.get('context', {}).get('user_input', f'Get issue {issue_number} from {repository}')
                }
                
                # Create and call the method
                get_issue_method = await dynamic_method_creator.create_method(
                    mcp_creator, 'get_issue', context
                )
                
                if get_issue_method:
                    issue = await get_issue_method(repo=repository, issue_number=issue_number)
                    if issue:
                        # Enhanced Output: Including reactions (if available)
                        reactions = {}
                        if isinstance(issue, dict) and 'reactions' in issue:
                            reactions = issue['reactions']
                        elif hasattr(issue, 'reactions'):
                            reactions = {k: getattr(issue.reactions, k, 0) if hasattr(issue.reactions, k) else 0 for k in ['+1', '-1', 'laugh', 'confused', 'heart', 'rocket', 'eyes']} # Assuming reactions is an object with these attributes
                        comments = await get_issue_comments(mcp_creator, repository, issue_number)

                        if isinstance(issue, dict):
                            issue_data = {
                                "title": sanitize_string(issue.get('title', '')), # Sanitize the title
                                "body": sanitize_string(issue.get('body', '')),   # Sanitize the body
                                "number": issue.get('number', issue_number),
                                "state": issue.get('state', 'unknown'),
                                "url": issue.get('html_url', '') or issue.get('url', ''),
                                "reactions": reactions,
                                "repository": repository,
                                "author": issue.get('user', {}).get('login', 'unknown'),  # Added issue author
                                "comments": comments,
                                "created_at": issue.get('created_at', None),  # Added issue creation timestamp
                                "updated_at": issue.get('updated_at', None),  # Added issue update timestamp
                                "closed_at": issue.get('closed_at', None),    # Added issue close timestamp
                                "labels": [label.get('name', '') if isinstance(label, dict) else getattr(label, 'name', '') for label in issue.get('labels', [])] # Added issue labels, handling both dict and object responses
                            }
                        else:
                            issue_data = {
                                "title": sanitize_string(getattr(issue, 'title', '')), # Sanitize the title
                                "body": sanitize_string(getattr(issue, 'body', '')),   # Sanitize the body
                                "number": getattr(issue, 'number', issue_number),
                                "state": getattr(issue, 'state', 'unknown'),
                                "url": getattr(issue, 'html_url', '') or getattr(issue, 'url', ''),
                                "reactions": reactions,
                                "repository": repository,
                                "author": getattr(issue.user, 'login', 'unknown') if hasattr(issue, 'user') else 'unknown',  # Added issue author
                                "comments": comments,
                                "created_at": getattr(issue, 'created_at', None),  # Added issue creation timestamp
                                "updated_at": getattr(issue, 'updated_at', None),  # Added issue update timestamp
                                "closed_at": getattr(issue, 'closed_at', None),    # Added issue close timestamp
                                 "labels": [getattr(label, 'name', '') for label in getattr(issue, 'labels', [])] # Added issue labels, handling potential AttributeError
                            }
                        logging.info(f"Successfully retrieved issue #{issue_number} from repository {repository} after dynamic method creation.")
                        return issue_data
                    else:
                        error_message = f"Issue #{issue_number} not found in repository {repository}."
                        logging.warning(error_message)
                        return format_error(error_message)
            except Exception as create_error:
                logging.error(f"Dynamic method creation failed.  Falling back to list_recent_issues. Error: {create_error}", exc_info=True)
                # If dynamic creation fails, use manual fallback
                try:
                    issues = await mcp_creator.list_recent_issues(repo=repository, limit=100)
                    for i in issues:
                        if i.get('number') == issue_number:
                            # Enhanced Output: Including reactions (if available)
                            reactions = {}
                            if 'reactions' in i:
                                reactions = i['reactions']
                            comments = await get_issue_comments(mcp_creator, repository, issue_number)

                            issue_data = {
                                "title": sanitize_string(i.get('title', '')), # Sanitize the title
                                "body": sanitize_string(i.get('body', '')),   # Sanitize the body
                                "number": i.get('number', issue_number),
                                "state": i.get('state', 'unknown'),
                                "url": i.get('html_url', '') or i.get('url', ''),
                                "reactions": reactions,
                                "repository": repository,
                                "author": i.get('user', {}).get('login', 'unknown'),  # Added issue author
                                "comments": comments,
                                "created_at": i.get('created_at', None),  # Added issue creation timestamp
                                "updated_at": i.get('updated_at', None),  # Added issue update timestamp
                                "closed_at": i.get('closed_at', None),    # Added issue close timestamp
                                "labels": [label.get('name', '') if isinstance(label, dict) else getattr(label, 'name', '') for label in i.get('labels', [])] # Added issue labels, handling both dict and object responses
                            }
                            logging.info(f"Successfully retrieved issue #{issue_number} from repository {repository} using list_recent_issues fallback.")
                            return issue_data
                    error_message = f"Issue #{issue_number} not found in repository {repository}."
                    logging.warning(error_message)
                    return format_error(error_message)

                except Exception as fallback_error:
                    error_message = f"Failed to retrieve issue using list_recent_issues fallback. Error: {fallback_error}"
                    logging.error(error_message, exc_info=True)
                    return format_error(f"Error retrieving issue from repository {repository} after dynamic creation failure: {error_message}")


        else:
            error_message = f"Error retrieving issue from repository {repository}: {str(e)}"
            logging.error(error_message, exc_info=True, extra={"attribute_error": str(e)})
            return format_error(error_message, details={"attribute_error": str(e)})
    except Exception as e:
        error_message = f"Error retrieving issue from repository {repository}: {str(e)}"
        logging.error(error_message, exc_info=True)
        return format_error(error_message)