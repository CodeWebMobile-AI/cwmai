from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scripts.logger import get_logger
from scripts.github_issue_creator import GitHubIssueCreator
from scripts.error_handler import ErrorHandler
from scripts.mcp_config import MCPConfigManager


class CheckArchitectureFilesFinal:
    """
    A general tool to fetch GitHub repositories, check architecture files,
    and handle errors with logging.
    """

    def __init__(self, config_path: str):
        """
        Initializes the CheckArchitectureFilesFinal tool.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.logger = get_logger(__name__)
        try:
            self.config_manager = MCPConfigManager(config_path)
            self.config = self.config_manager.get_config()
            self.github_issue_creator = GitHubIssueCreator(
                self.config.github_org, self.config.github_repo, self.config.github_token
            )

        except Exception as e:
            self.logger.error(f"Error initializing CheckArchitectureFilesFinal: {e}")
            self.error_handler = ErrorHandler()
            self.error_handler.handle_error(e, "initialization", self.logger)

    def fetch_repository_files(self, repo_url: str) -> Dict[str, Any]:
        """
        Fetches files from a GitHub repository.

        Args:
            repo_url (str): The URL of the GitHub repository.

        Returns:
            Dict[str, Any]: A dictionary containing the status of the fetch operation and the file contents, if successful.
        """
        try:
            # Placeholder for fetching repository files.  Replace with actual implementation.
            # This example simply returns a dummy dictionary.
            file_contents = {
                "file1.txt": "This is the content of file1.",
                "file2.txt": "This is the content of file2.",
            }
            return {"success": True, "files": file_contents}
        except Exception as e:
            self.logger.error(f"Error fetching repository files from {repo_url}: {e}")
            self.error_handler = ErrorHandler() # moved here to be defined in case of exception at __init__
            self.error_handler.handle_error(e, "fetching_repository_files", self.logger)
            return {"error": str(e)}

    def check_architecture_files(self, files: Dict[str, str]) -> Dict[str, Any]:
        """
        Checks the architecture files for specific criteria.

        Args:
            files (Dict[str, str]): A dictionary of file names and their contents.

        Returns:
            Dict[str, Any]: A dictionary containing the status of the check and any identified issues.
        """
        try:
            issues = []
            for filename, content in files.items():
                # Placeholder for checking architecture files. Replace with actual implementation.
                # This example checks if the file contains the word "TODO".
                if "TODO" in content:
                    issues.append(f"File {filename} contains 'TODO'.")

            if issues:
                return {"success": True, "issues": issues}
            else:
                return {"success": True, "message": "No issues found."}
        except Exception as e:
            self.logger.error(f"Error checking architecture files: {e}")
            self.error_handler = ErrorHandler() # moved here to be defined in case of exception at __init__
            self.error_handler.handle_error(e, "checking_architecture_files", self.logger)
            return {"error": str(e)}

    def create_github_issue(self, title: str, body: str) -> Dict[str, Any]:
        """
        Creates a GitHub issue.

        Args:
            title (str): The title of the issue.
            body (str): The body of the issue.

        Returns:
            Dict[str, Any]: A dictionary containing the status of the issue creation.
        """
        try:
            result = self.github_issue_creator.create_issue(title, body)  # Use github_issue_creator
            if result.get("success"):
                return {"success": True, "issue_url": result.get("issue_url")}
            else:
                return {"error": result.get("error")}
        except Exception as e:
            self.logger.error(f"Error creating GitHub issue: {e}")
            self.error_handler = ErrorHandler() # moved here to be defined in case of exception at __init__
            self.error_handler.handle_error(e, "creating_github_issue", self.logger)
            return {"error": str(e)}

    def run(self, repo_url: str) -> Dict[str, Any]:
        """
        Runs the architecture check process.

        Args:
            repo_url (str): The URL of the GitHub repository to check.

        Returns:
            Dict[str, Any]: A dictionary containing the overall status of the process.
        """
        try:
            files_result = self.fetch_repository_files(repo_url)

            if "error" in files_result:
                return {"error": f"Failed to fetch repository files: {files_result['error']}"}

            files = files_result["files"]
            check_result = self.check_architecture_files(files)

            if "error" in check_result:
                return {"error": f"Failed to check architecture files: {check_result['error']}"}

            if "issues" in check_result:
                title = "Architecture Check Issues Found"
                body = "\n".join(check_result["issues"])
                issue_result = self.create_github_issue(title, body)

                if "error" in issue_result:
                    return {"error": f"Failed to create GitHub issue: {issue_result['error']}"}

                return {"success": True, "message": "Issues found and GitHub issue created.", "issue_url": issue_result["issue_url"]}
            else:
                return {"success": True, "message": "No architecture issues found."}

        except Exception as e:
            self.logger.error(f"Error running architecture check process: {e}")
            self.error_handler = ErrorHandler() # moved here to be defined in case of exception at __init__
            self.error_handler.handle_error(e, "running_architecture_check", self.logger)
            return {"error": str(e)}


if __name__ == "__main__":
    # Example usage:
    config_path = "config.json"  # Replace with your actual config path
    repo_url = "https://github.com/example/repo"  # Replace with the repository URL

    checker = CheckArchitectureFilesFinal(config_path)
    result = checker.run(repo_url)

    if "success" in result:
        print(f"Success: {result['message']}")
        if "issue_url" in result:
            print(f"Issue URL: {result['issue_url']}")
    elif "error" in result:
        print(f"Error: {result['error']}")
