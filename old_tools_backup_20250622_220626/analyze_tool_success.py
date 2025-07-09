from typing import Dict, Any, Optional
import traceback
import re
from scripts.logger import logger
from scripts.state_manager import StateManager

__description__ = "Analyzes tool success based on available data. Extracts repository names from the context to focus the analysis. Can use data from StateManager, GitHub issues, and logs."
__parameters__ = {}
__examples__ = [
    {"user_input": "Analyze tool success in repository my-repo."},
    {"user_input": "Analyze tool success for project example-project-ai."}
]

async def analyze_tool_success(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes tool success based on available data. Extracts repository names from the context 
    to focus the analysis. Can use data from StateManager, GitHub issues, and logs.
    """
    try:
        repo_name = kwargs.get('repo_name') or kwargs.get('repo')

        if not repo_name:
            context = kwargs.get('context', {})
            user_input = context.get('user_input', '')

            repo_patterns = [
                r'(?:in|for|about|within)\s+([a-zA-Z0-9-]+(?:-ai)?)',
                r'([a-zA-Z0-9-]+(?:-ai))\s+(?:repository|repo|project)',
                r'([a-zA-Z0-9-]+(?:-ai))'  # Any word ending in -ai
            ]

            for pattern in repo_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    repo_name = match.group(1)
                    break

        if not repo_name:
            return {"error": "Could not determine repository name from context or parameters."}

        if '/' not in repo_name:
            repo_name = f"CodeWebMobile-AI/{repo_name}"

        state_manager = StateManager()
        state = state_manager.get_state()
        projects = state.get('projects', {})

        project_data = projects.get(repo_name, {})

        analysis_results = {
            "repository": repo_name,
            "state_manager_data": project_data,
            "success_metrics": {},
            "issues": []
        }

        # Placeholder for more sophisticated analysis logic
        # (e.g., analyzing tool performance logs, GitHub issues, etc.)
        # Example: analyzing health scores from StateManager
        if 'health_score' in project_data:
            analysis_results["success_metrics"]["health_score"] = project_data['health_score']

        # Example: Getting open issues (implementation in a separate tool)
        # from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
        # mcp_creator = MCPGitHubIssueCreator()
        # open_issues = await mcp_creator.list_recent_issues(repo=repo_name, limit=5)
        # analysis_results["issues"] = open_issues

        # Example: Analyzing logs (implementation in a separate tool)
        # from scripts.log_viewer import LogViewer
        # log_viewer = LogViewer()
        # recent_errors = await log_viewer.get_recent_errors(search_term=repo_name)
        # analysis_results["recent_errors"] = recent_errors

        return analysis_results

    except Exception as e:
        logger.error(f"Error analyzing tool success: {e}")
        traceback.print_exc()  # Print the traceback for debugging purposes
        return {"error": str(e), "traceback": traceback.format_exc()}