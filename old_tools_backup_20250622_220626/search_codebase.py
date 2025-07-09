from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scripts.logger import get_logger
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.architecture_generator import ArchitectureGenerator
from scripts.ai_code_analyzer import AICodeAnalyzer
from scripts.file_manager import FileManager


class SearchCodebase:
    """
    A tool for fetching GitHub repositories and searching their content.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the SearchCodebase tool.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing necessary parameters.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.github_issue_creator = MCPGitHubIssueCreator(config)  # Assuming this is initialized correctly elsewhere or adjust as needed
        self.file_manager = FileManager()

    def search_github_repository(self, owner: str, repo: str, query: str) -> Dict[str, Any]:
        """
        Fetches a GitHub repository and searches for a specific query within its content.

        Args:
            owner (str): The owner of the GitHub repository.
            repo (str): The name of the GitHub repository.
            query (str): The search query.

        Returns:
            Dict[str, Any]: A dictionary containing the search results or an error message.
        """
        try:
            # 1. Clone the repository
            repo_path = Path(f"./temp_repos/{owner}/{repo}")  # Define a temporary directory
            self.file_manager.remove_directory(repo_path) # Remove existing directory

            clone_url = f"https://github.com/{owner}/{repo}.git"
            self.file_manager.clone_repository(clone_url, repo_path)

            # 2. Search for the query within the repository files
            results = self._search_files(repo_path, query)

            # 3. Clean up the temporary repository directory
            # self.file_manager.remove_directory(repo_path) # Comment out for debugging if needed.

            return {"success": True, "results": results}

        except Exception as e:
            error_message = f"Error searching codebase: {str(e)}"
            self.logger.exception(error_message)

            # Create GitHub issue if specified in the config
            if self.config.get("create_github_issues_on_error", False): # Check config for enabling issue creation
                issue_title = "Error during codebase search"
                issue_body = f"An error occurred while searching the codebase for repository {owner}/{repo}:\n\n{error_message}"
                self.github_issue_creator.create_issue(title=issue_title, body=issue_body) # Use GithubIssueCreator instance.

            return {"success": False, "error": error_message}

    def _search_files(self, repo_path: Path, query: str) -> List[Dict[str, Any]]:
        """
        Searches for a query within the files of a repository.

        Args:
            repo_path (Path): The path to the repository.
            query (str): The search query.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the file path and matching lines.
        """
        results = []
        try:
            for file_path in self.file_manager.get_all_files(repo_path):
                if self.file_manager.is_binary_file(file_path):
                    continue  # Skip binary files

                with open(file_path, "r", encoding="utf-8", errors="ignore") as file: # Handle encoding issues
                    for line_number, line in enumerate(file, 1):
                        if query in line:
                            results.append({
                                "file_path": str(file_path.relative_to(repo_path)),
                                "line_number": line_number,
                                "line": line.strip()
                            })
        except Exception as e:
                self.logger.exception(f"Error during file search in {repo_path}: {e}")
                # Optionally, create a github issue here for specific file search errors.
        return results

    def execute(self, owner: str, repo: str, query: str) -> Dict[str, Any]:
        """
        Executes the codebase search.

        Args:
            owner (str): The owner of the GitHub repository.
            repo (str): The name of the GitHub repository.
            query (str): The search query.

        Returns:
            Dict[str, Any]: A dictionary containing the search results or an error message.
        """
        return self.search_github_repository(owner, repo, query)
