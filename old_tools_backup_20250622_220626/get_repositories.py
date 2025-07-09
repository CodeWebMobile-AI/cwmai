from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import time
import json  # For caching
import os
from urllib.parse import urlparse, urlunparse

from scripts.logger import get_logger
from scripts.workflow_orchestrator import WorkflowOrchestrator
from scripts.tool_calling_system import ToolCallingSystem
from scripts.error_handler import ErrorHandler
from scripts.state_manager import StateManager
from scripts.github_issue_creator import GitHubIssueCreator
from scripts.storage_factory import StorageFactory
from scripts.local_github_simulator import LocalGitHubSimulator


class GetRepositories:
    """
    A tool to fetch GitHub repositories.
    """

    def __init__(self, github_issue_creator: GitHubIssueCreator,
                 error_handler: ErrorHandler,
                 state_manager: StateManager,
                 cache_dir: str = "cache",
                 github_api_url: str = "https://api.github.com") -> None:
        """
        Initializes the GetRepositories tool.

        Args:
            github_issue_creator: An instance of GitHubIssueCreator for creating GitHub issues.
            error_handler: An instance of ErrorHandler for handling errors.
            state_manager: An instance of StateManager for managing state.
            cache_dir: Directory to store cached repository data. Defaults to "cache".
            github_api_url: The base URL of the GitHub API.  Defaults to "https://api.github.com".
        """
        self.logger = get_logger(__name__)
        self.github_issue_creator = github_issue_creator
        self.error_handler = error_handler
        self.state_manager = state_manager
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure cache dir exists
        self.github_api_url = github_api_url
        self.api_request_timeout = 10  # seconds


    def get_repositories(self, owner: str, use_cache: bool = True, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetches GitHub repositories for a given owner.  Uses caching if enabled.

        Args:
            owner: The GitHub username or organization name.
            use_cache:  Whether to use cached data if available.  Defaults to True.
            force_refresh: If True, bypasses the cache and fetches fresh data.  Defaults to False.

        Returns:
            A dictionary containing either:
            - "repositories": A list of repository names on success,
            - "message": A human-readable success or error message.
            - "status":  "success" or "error"
            - "cached":  True if the data was retrieved from the cache, False otherwise.
            - "owner": The owner for which the repositories were fetched
            - "cache_path": The path to the cache file used (if any)
        """

        # Input validation
        if not isinstance(owner, str) or not owner:
            error_message = "Invalid owner provided. Owner must be a non-empty string."
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message)
            return {"status": "error", "message": error_message, "owner": owner}


        cache_key = f"{owner}_repos.json"
        cache_path = self.cache_dir / cache_key

        if use_cache and not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                print(f"Retrieved repositories for owner: {owner} from cache.")
                return {"status": "success", "message": f"Repositories for {owner} retrieved from cache.",
                        "repositories": cached_data, "cached": True, "owner": owner, "cache_path": str(cache_path)}
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.warning(f"Failed to load cache for owner {owner}: {e}.  Fetching from source.")
                # Consider deleting the corrupted cache file
                try:
                    os.remove(cache_path)
                    print(f"Corrupted cache file {cache_path} deleted.")
                except OSError as oe:
                    self.logger.error(f"Failed to delete corrupted cache file {cache_path}: {oe}")
            except Exception as e:
                self.logger.warning(f"Unexpected error loading cache for owner {owner}: {e}. Fetching from source.")



        try:
            repo_names = self._fetch_repositories(owner)

            if not repo_names:
                message = f"No repositories found for owner: {owner}"
                print(message)
                return {"status": "error", "message": message, "owner": owner, "repositories": [], "cached": False}

            print(f"Successfully fetched repositories for owner: {owner}")

            # Cache the results
            try:
                with open(cache_path, "w") as f:
                    json.dump(repo_names, f, indent=4)  # Store as JSON with indentation for readability
            except Exception as e:
                self.logger.warning(f"Failed to save cache for owner {owner}: {e}")


            return {"status": "success", "message": f"Successfully fetched repositories for {owner}.",
                    "repositories": repo_names, "cached": False, "owner": owner, "cache_path": str(cache_path)}

        except Exception as e:
            error_message = f"Error fetching repositories for owner: {owner}. Error: {str(e)}"
            self.logger.exception(error_message)
            self.error_handler.handle_error(error_message)
            return {"status": "error", "message": error_message, "owner": owner, "repositories": [], "cached": False}

    def _fetch_repositories(self, owner: str) -> List[str]:
        """
        Fetches repositories for a given owner using the GitHub API.

        Includes retry logic with exponential backoff to handle transient errors.
        Leverages the github_issue_creator for API calls.

        Args:
            owner: The GitHub username or organization name.

        Returns:
            A list of repository names.  Returns an empty list if no repositories
            are found or if all retries fail.  Returns None if there is an authentication error.
        """
        max_retries = 3
        retry_delay = 1  # seconds
        repos = []

        for attempt in range(max_retries):
            try:
                # Use github_issue_creator to authenticate and make API requests.
                # Construct the API URL
                api_url = self._build_api_url(f"/orgs/{owner}/repos")

                # Fetch the repositories using a GET request
                response = self.github_issue_creator.make_api_request("GET", api_url, timeout=self.api_request_timeout)

                if response.status_code == 200:
                    # Parse the JSON response
                    repo_data = response.json()
                    # Extract the repository names
                    repos = [repo["name"] for repo in repo_data]
                    print(f"Successfully fetched {len(repos)} repositories for owner: {owner} on attempt {attempt+1}")
                    return repos
                elif response.status_code == 404:
                    self.logger.warning(f"Owner {owner} not found on attempt {attempt+1}.")
                    return []  # Owner not found, return empty list.  Do not retry.
                elif response.status_code == 403:
                    self.logger.error(f"Authentication error or insufficient permissions to access owner {owner} on attempt {attempt+1}. Status code: {response.status_code}")
                    self.error_handler.handle_error(f"Authentication error or insufficient permissions to access owner {owner}.  Check your GitHub token.")
                    return None  # Authentication error, stop retrying.
                elif response.status_code == 429:
                    # Handle rate limiting
                    reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60)) #default 60 seconds
                    wait_time = reset_time - int(time.time())
                    self.logger.warning(f"Rate limited. Waiting {wait_time} seconds before retrying attempt {attempt+1}.")
                    time.sleep(wait_time)

                else:
                    self.logger.error(f"API request failed for owner: {owner} with status code: {response.status_code} on attempt {attempt+1}.  Response: {response.text}")
                    self.error_handler.handle_error(f"API request failed for owner: {owner} with status code: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # Re-raise the exception on the last retry
                        self.logger.error(f"Max retries reached.  Failing to fetch repos for owner: {owner}")
                        return [] # Return empty list after all retries fail.

            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed to fetch repos for owner: {owner}. Error: {e}")
                self.error_handler.handle_error(f"Attempt {attempt + 1} failed to fetch repos for owner: {owner}. Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Re-raise the exception on the last retry
                    self.logger.error(f"Max retries reached.  Failing to fetch repos for owner: {owner}")
                    return [] # Return empty list after all retries fail.

        # Return empty list after all retries fail.
        return []

    def _build_api_url(self, path: str) -> str:
        """
        Builds a full API URL from a path.
        Ensures that the URL has a scheme (e.g., https)
        """
        parsed_url = urlparse(self.github_api_url)
        if not parsed_url.scheme:
            raise ValueError(f"Invalid github_api_url: {self.github_api_url}.  Must include a scheme (e.g., https://)")

        return urlunparse((parsed_url.scheme, parsed_url.netloc, path, parsed_url.params, parsed_url.query, parsed_url.fragment))



    def clear_cache(self, owner: Optional[str] = None) -> None:
        """
        Clears the cache for a specific owner or all owners.

        Args:
            owner: The GitHub username or organization name to clear the cache for.
                   If None, clears the entire cache directory.
        """
        if owner:
            cache_key = f"{owner}_repos.json"
            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                try:
                    os.remove(cache_path)  # Use os.remove for better cross-platform compatibility
                    print(f"Cache cleared for owner: {owner}")
                except OSError as e:
                    self.logger.error(f"Failed to clear cache for owner {owner}: {e}")
                    self.error_handler.handle_error(f"Failed to clear cache for owner {owner}: {e}")

            else:
                print(f"No cache found for owner: {owner}")
        else:
            # Clear entire cache directory
            for file in self.cache_dir.glob("*.json"):  # Only delete json files
                try:
                    os.remove(file) # Use os.remove
                except OSError as e:
                    self.logger.error(f"Failed to delete cache file {file}: {e}")
                    self.error_handler.handle_error(f"Failed to delete cache file {file}: {e}")
            print("Entire cache directory cleared.")

    def get_cache_path(self, owner: str) -> str:
        """
        Returns the path to the cache file for a specific owner.

        Args:
            owner: The GitHub username or organization name.

        Returns:
            The full path to the cache file.
        """
        cache_key = f"{owner}_repos.json"
        return str(self.cache_dir / cache_key)

    def is_cache_valid(self, owner: str, max_age_seconds: int) -> bool:
        """
        Checks if the cache for a specific owner is valid based on its age.

        Args:
            owner: The GitHub username or organization name.
            max_age_seconds: The maximum age of the cache in seconds.

        Returns:
            True if the cache is valid (exists and is not older than max_age_seconds), False otherwise.
        """
        cache_path = self.get_cache_path(owner)
        if not os.path.exists(cache_path):
            return False

        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age <= max_age_seconds