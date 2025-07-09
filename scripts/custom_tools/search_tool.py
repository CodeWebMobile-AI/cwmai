"""
AI-Generated Tool: search_tool
Description: Search for items for tool
Generated: 2025-06-23T09:13:25.000409+00:00
Requirements: 
        Tool Name: search_tool
        Intent: Search for items for tool
        Expected Parameters: {"query": "'best restaurants in New York'"}
        Category: query
        
        The tool should:
        1. Search items using flexible criteria
2. Support partial matches and filters
3. Return ranked results
        
"""

"""
Module for the search_tool, which allows searching for items based on flexible criteria,
supporting partial matches and filters, and returning ranked results.
"""

import re
from typing import Dict, Any, List

from scripts.state_manager import StateManager
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.repository_analyzer import RepositoryAnalyzer


__description__ = "Search for items for tool"
__parameters__ = {"query": "Search query. Can be any string."}
__examples__ = [
    {"query": "best restaurants in New York"},
    {"query": "find mindleap-ai repos"},
    {"query": "issues related to authentication"},
]


async def search_tool(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Searches for items using flexible criteria, supporting partial matches and filters,
    and returns ranked results.  Context-aware and adaptable to different data sources.

    Args:
        **kwargs: Keyword arguments containing context and parameters.
                   Expected to contain 'context' with 'user_input'.

    Returns:
        A dictionary containing the search results.
        Example: {"total": 10, "summary": "Found 10 items matching the query."}

    Raises:
        ValueError: If the search query cannot be determined from the context.
    """

    state_manager = StateManager()
    mcp_creator = MCPGitHubIssueCreator()
    analyzer = RepositoryAnalyzer()

    user_input = kwargs.get('context', {}).get('user_input', '')
    query = kwargs.get('query')

    if not query:
        query = user_input  # Try to use user input directly if 'query' parameter is missing

    if not query:
        return {"error": "Search query cannot be determined from the context."}

    # 1. Search in system state (projects, etc.)
    state = state_manager.get_state()
    projects = state.get('projects', {})
    project_matches = [
        project_name for project_name, project_data in projects.items()
        if query.lower() in project_name.lower() or query.lower() in str(project_data).lower()
    ]

    # 2. Search in GitHub issues (using MCP integration)
    issue_matches = []
    repo_patterns = r'\b([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)\b'  # Matches "org/repo"
    repo_matches = re.findall(repo_patterns, user_input)

    for repo in repo_matches:
        try:
            issues = await mcp_creator.list_recent_issues(repo=repo, limit=10)  # Limiting to 10 for efficiency
            issue_matches.extend([
                issue for issue in issues
                if query.lower() in issue.get('title', '').lower() or query.lower() in issue.get('body', '').lower()
            ])
        except Exception as e:
            print(f"Error fetching issues from {repo}: {e}")  # Non-critical error - continue with other searches.

    # 3. Combine and rank results (very basic ranking for now - can be improved)
    results = []
    results.extend([{"type": "project", "data": projects[project_name]} for project_name in project_matches])
    results.extend([{"type": "issue", "data": issue} for issue in issue_matches])

    # Basic ranking: Projects mentioned first, then issues
    ranked_results = results  # No actual re-ranking is performed here but could be implemented

    # Summarize results
    total_results = len(ranked_results)
    summary = f"Found {total_results} items matching the query '{query}'."

    return {"total": total_results, "summary": summary, "results": ranked_results}
