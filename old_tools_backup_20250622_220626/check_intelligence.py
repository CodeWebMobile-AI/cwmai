from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scripts.logger import get_logger
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.ai_code_analyzer import AICodeAnalyzer
from scripts.state_manager import StateManager
from scripts.error_handler import ErrorHandler
from scripts.mcp_config import MCPConfigManager


class CheckIntelligence:
    """
    A tool to fetch GitHub repositories, analyze code, and create GitHub issues.
    """

    def __init__(self, config_path: str):
        """
        Initializes the CheckIntelligence tool.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.logger = get_logger(__name__)
        try:
            self.config_manager = MCPConfigManager(config_path)
            self.config = self.config_manager.get_config()
            self.issue_creator = MCPGitHubIssueCreator(
                self.config.github_org,
                self.config.github_repo,
                self.config.github_token
            )
            self.error_handler = ErrorHandler()  # Initialize ErrorHandler
            self.state_manager = StateManager(
                redis_host=self.config.redis_host,
                redis_port=self.config.redis_port,
                redis_db=self.config.redis_db,
                state_key_prefix="check_intelligence"
            )

        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            raise

    def fetch_github_repositories(self, owner: str, repo_name: str) -> Dict[str, Any]:
        """
        Fetches GitHub repositories using the GitHub API.

        Args:
            owner (str): The owner of the repository.
            repo_name (str): The name of the repository.

        Returns:
            Dict[str, Any]: A dictionary containing the repository data or an error message.
        """
        try:
            # This is a placeholder - replace with actual GitHub API interaction.
            repo_data = {
                "owner": owner,
                "repo_name": repo_name,
                "description": "Example repository",
                "files": ["file1.py", "file2.py"],
            }
            return {"success": repo_data}

        except Exception as e:
            error_message = f"Error fetching repository {owner}/{repo_name}: {e}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message, "github_fetch")
            return {"error": error_message}

    def analyze_code(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the code in the given repository data.  This is a placeholder, and in
        reality you'd want an AICodeAnalyzer or similar to do this.

        Args:
            repo_data (Dict[str, Any]): The repository data.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results or an error message.
        """
        try:
            # Placeholder analysis
            analysis_results = {
                "code_quality": "Good",
                "security_vulnerabilities": "None",
            }
            return {"success": analysis_results}

        except Exception as e:
            error_message = f"Error analyzing code: {e}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message, "code_analysis")
            return {"error": error_message}

    def create_github_issue(self, title: str, body: str) -> Dict[str, Any]:
        """
        Creates a GitHub issue using the GitHub API.

        Args:
            title (str): The title of the issue.
            body (str): The body of the issue.

        Returns:
            Dict[str, Any]: A dictionary containing the issue creation status or an error message.
        """
        try:
            issue_details = self.issue_creator.create_issue(title, body)
            if 'error' in issue_details:
                return issue_details
            return {"success": issue_details}

        except Exception as e:
            error_message = f"Error creating GitHub issue: {e}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message, "github_issue_creation")
            return {"error": error_message}

    def run(self, owner: str, repo_name: str) -> Dict[str, Any]:
        """
        Runs the CheckIntelligence tool.

        Args:
            owner (str): The owner of the repository.
            repo_name (str): The name of the repository.

        Returns:
            Dict[str, Any]: A dictionary containing the overall status of the tool run.
        """
        try:
            repo_data_result = self.fetch_github_repositories(owner, repo_name)
            if 'error' in repo_data_result:
                return repo_data_result

            repo_data = repo_data_result['success']

            analysis_result = self.analyze_code(repo_data)
            if 'error' in analysis_result:
                return analysis_result

            analysis_results = analysis_result['success']

            # Example: Create an issue if code quality is not "Good"
            if analysis_results["code_quality"] != "Good":
                issue_title = f"Code quality issue in {owner}/{repo_name}"
                issue_body = f"Code quality is not good. Details: {analysis_results}"
                issue_creation_result = self.create_github_issue(issue_title, issue_body)
                if 'error' in issue_creation_result:
                    return issue_creation_result

            self.state_manager.state = {"last_run": "success"}
            self.state_manager.save_state({"last_run": "success"})

            return {"success": f"CheckIntelligence run completed for {owner}/{repo_name}"}

        except Exception as e:
            error_message = f"Error running CheckIntelligence: {e}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message, "tool_run")
            return {"error": error_message}


if __name__ == "__main__":
    # Example Usage
    config_path = "config.json"  # Replace with your actual config file path
    try:
        check_intelligence = CheckIntelligence(config_path)
        result = check_intelligence.run("octocat", "Spoon-Knife")
        print(result)

    except Exception as e:
        print(f"Initialization failed: {e}")
