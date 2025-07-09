from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
import os
from scripts.logger import get_logger
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.config_loader import ConfigLoader
from scripts.error_handler import ErrorHandler
import re
import time

class DeleteAllIssues:
    """
    A tool to delete all issues in specified GitHub repositories.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DeleteAllIssues tool.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing necessary parameters.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.error_handler = ErrorHandler()  # Initialize ErrorHandler
        self.github_issue_creator = MCPGitHubIssueCreator(config)  # Updated line
        self.cache_dir = Path(self.config.get("cache_dir", ".cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.github_api_retries = self.config.get("github_api_retries", 3)  # Number of retries for GitHub API calls.
        self.retry_delay = self.config.get("retry_delay", 5) # Delay in seconds between retries

    def _get_cache_file_path(self, owner: str, repo_name: str) -> Path:
        """
        Constructs the cache file path for a given repository.
        """
        return self.cache_dir / f"{owner}_{repo_name}_issues.json"

    def _load_issues_from_cache(self, owner: str, repo_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Loads issues from the cache if available.
        """
        cache_file_path = self._get_cache_file_path(owner, repo_name)
        if cache_file_path.exists():
            try:
                with open(cache_file_path, "r") as f:
                    print(f"Loading issues from cache for {owner}/{repo_name}")
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to load issues from cache for {owner}/{repo_name}: Invalid JSON format. Deleting cache. {e}")
                cache_file_path.unlink() # Delete corrupted cache file
                return None
            except Exception as e:
                self.logger.warning(f"Failed to load issues from cache for {owner}/{repo_name}: {e}")
        return None

    def _save_issues_to_cache(self, owner: str, repo_name: str, issues: List[Dict[str, Any]]):
        """
        Saves issues to the cache.
        """
        cache_file_path = self._get_cache_file_path(owner, repo_name)
        try:
            with open(cache_file_path, "w") as f:
                json.dump(issues, f, indent=4)
                print(f"Saving issues to cache for {owner}/{repo_name}")
        except Exception as e:
            self.logger.error(f"Failed to save issues to cache for {owner}/{repo_name}: {e}")

    def _validate_repo_name(self, repo: str) -> bool:
        """
        Validates the repository name format (owner/repo).
        """
        parts = repo.split("/")
        if len(parts) != 2:
            self.logger.error(f"Invalid repository name format: {repo}. Expected 'owner/repo'.")
            return False

        owner, repo_name = parts
        if not re.match(r"^[a-zA-Z0-9_-]+$", owner) or not re.match(r"^[a-zA-Z0-9_-]+$", repo_name):
            self.logger.error(f"Invalid repository name format: {repo}. Owner and repo should contain only alphanumeric characters, underscores, and hyphens.")
            return False
        return True

    def get_all_issues(self, owner: str, repo_name: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves all issues for a repository, using cache if enabled and available.
        """
        if use_cache:
            cached_issues = self._load_issues_from_cache(owner, repo_name)
            if cached_issues:
                return cached_issues

        for attempt in range(self.github_api_retries):
            try:
                issues = self.github_issue_creator.get_all_issues(owner, repo_name)

                if issues:
                    self._save_issues_to_cache(owner, repo_name, issues)  # Save to cache after fetching
                return issues
            except Exception as e:
                error_message = f"Attempt {attempt + 1} failed to get issues for {owner}/{repo_name}: {str(e)}"
                self.logger.error(error_message)
                self.error_handler.handle_error(error_message)
                if attempt < self.github_api_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise  # Re-raise the exception if all retries failed

    def close_issues_batch(self, owner: str, repo_name: str, issue_numbers: List[int]) -> List[Dict[str, Any]]:
        """
        Closes issues in batches to improve performance and reduce API calls.
        Returns a list of dictionaries, each with success/failure and details of the close operation.
        """
        results = []
        for issue_number in issue_numbers:
            for attempt in range(self.github_api_retries):
                try:
                    self.github_issue_creator.close_issue(owner, repo_name, issue_number)
                    print(f"Successfully closed issue #{issue_number} in {repo_name}")
                    results.append({"issue": issue_number, "status": "success"})
                    break # Break the retry loop on success
                except Exception as e:
                    error_message = f"Attempt {attempt + 1} failed to close issue #{issue_number} in {repo_name}: {str(e)}"
                    self.logger.error(error_message)
                    self.error_handler.handle_error(error_message)
                    if attempt < self.github_api_retries - 1:
                        print(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                    else:
                        results.append({"issue": issue_number, "status": "failure", "error": error_message}) # Append failure only after all retries fail

        return results

    def delete_all_issues(self, repos: List[str], use_cache: bool = True, batch_size: int = 20, dry_run: bool = False) -> Dict[str, Any]:
        """
        Deletes all issues in the specified GitHub repositories.

        Args:
            repos (List[str]): A list of repository names in the format "owner/repo".
            use_cache (bool): Whether to use cached issue data.
            batch_size (int): The number of issues to close in each batch.
            dry_run (bool): If True, only lists issues to be deleted without actually deleting them.

        Returns:
            Dict[str, Any]: A dictionary indicating success or failure with details.
        """
        total_deleted_count = 0
        repo_results = {}
        total_issue_count = 0

        for repo in repos:
            if not self._validate_repo_name(repo):
                repo_results[repo] = {"status": "failure", "error": "Invalid repository name format."}
                continue

            owner, repo_name = repo.split("/")
            try:
                issues = self.get_all_issues(owner, repo_name, use_cache)
                if not issues:
                    print(f"No issues found in {repo}")
                    repo_results[repo] = {"status": "success", "message": "No issues found."}
                    continue

                issue_numbers = [issue['number'] for issue in issues]
                total_issue_count += len(issue_numbers)

                if dry_run:
                    print(f"Dry run: Found {len(issue_numbers)} issues in {repo} to be deleted.")
                    repo_results[repo] = {"status": "success", "message": f"Dry run: {len(issue_numbers)} issues found to be deleted."}
                    continue

                # Batch the closing of issues for efficiency
                for i in range(0, len(issue_numbers), batch_size):
                    batch = issue_numbers[i:i + batch_size]
                    batch_results = self.close_issues_batch(owner, repo_name, batch)
                    total_deleted_count += sum(1 for result in batch_results if result["status"] == "success")

                repo_results[repo] = {"status": "success", "deleted_count": len(issue_numbers)}
                # Invalidate Cache After Deletion
                cache_file_path = self._get_cache_file_path(owner, repo_name)
                if cache_file_path.exists():
                    cache_file_path.unlink()
                    print(f"Invalidated cache for {repo} after issue deletion.")

            except Exception as e:
                error_message = f"Error processing repository {repo}: {str(e)}"
                self.logger.exception(error_message)
                self.error_handler.handle_error(error_message)
                repo_results[repo] = {"status": "failure", "error": error_message}

        overall_status = "success" if all(result.get("status") == "success" for result in repo_results.values() if isinstance(result, dict)) else "partial success"

        result = {
            "overall_status": overall_status,
            "total_deleted_count": total_deleted_count,
            "total_issue_count": total_issue_count,
            "repository_results": repo_results
        }

        return result

    def list_cached_repos(self) -> List[str]:
        """
        Lists repositories with cached issue data.
        """
        cached_repos = []
        for file in self.cache_dir.glob("*_issues.json"):
            filename = file.stem
            owner_repo = filename.replace("_issues", "").replace("_", "/")
            cached_repos.append(owner_repo)
        return cached_repos

    def clear_cache(self) -> None:
        """
        Clears the entire cache directory.  Use with caution!
        """
        try:
            for file in self.cache_dir.glob("*_issues.json"):
                file.unlink()
            print("Successfully cleared the issue cache.")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

def delete_all_issues_entrypoint(config: Dict[str, Any], repos: List[str], use_cache: bool = True, batch_size: int = 20, dry_run: bool = False) -> Dict[str, Any]:
    """
    Entrypoint for the delete_all_issues tool.
    """
    delete_issues_tool = DeleteAllIssues(config)
    result = delete_issues_tool.delete_all_issues(repos, use_cache, batch_size, dry_run)
    return result

if __name__ == '__main__':
    # Example Usage (replace with your actual config and repositories)
    example_config = {
        "github_token": os.environ.get("GITHUB_TOKEN"),  # Get token from environment variable
        "cache_dir": ".cache",
        "github_api_retries": 2,
        "retry_delay": 3
    }

    example_repos = ["your_owner/your_repo", "another_owner/another_repo"]

    # Create dummy files for scripts.logger, scripts.mcp_github_issue_creator,
    # scripts.config_loader, and scripts.error_handler
    # Create dummy scripts.logger.py
    with open("scripts/logger.py", "w") as f:
        f.write("""
import logging

def get_logger(name):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
        """)
    # Create dummy scripts.mcp_github_issue_creator.py
    with open("scripts/mcp_github_issue_creator.py", "w") as f:
        f.write("""
import requests
import os
class MCPGitHubIssueCreator:
    def __init__(self, config):
        self.config = config
        self.github_token = config.get("github_token")
        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def get_all_issues(self, owner, repo_name):
        # Replace with actual GitHub API call in real implementation
        # This is just a dummy implementation
        if owner == "your_owner" and repo_name == "your_repo":
            return [{"number": 1}, {"number": 2}]
        elif owner == "another_owner" and repo_name == "another_repo":
            return [{"number": 3}]
        else:
            return []

    def close_issue(self, owner, repo_name, issue_number):
        # Replace with actual GitHub API call in real implementation
        # This is just a dummy implementation
        print(f"Closing issue #{issue_number} in {owner}/{repo_name}")
        return {}
        """)
    # Create dummy scripts.config_loader.py
    with open("scripts/config_loader.py", "w") as f:
        f.write("""
def load_config(path):
    # Replace with actual config loading logic in real implementation
    return {}
        """)

    # Create dummy scripts.error_handler.py
    with open("scripts/error_handler.py", "w") as f:
        f.write("""
class ErrorHandler:
    def handle_error(self, message):
        print(f"Error: {message}")
        """)
    if not os.path.exists("scripts"):
        os.makedirs("scripts")

    result = delete_all_issues_entrypoint(example_config, example_repos, dry_run=True) #added dry_run example
    print(result)

    # Example to clear the cache:
    # delete_issues_tool = DeleteAllIssues(example_config)
    # delete_issues_tool.clear_cache()