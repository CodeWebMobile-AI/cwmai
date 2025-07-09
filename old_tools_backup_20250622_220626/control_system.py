from typing import Dict, Any, Optional, List
import traceback
import re
import asyncio
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator  # Assuming this exists and is properly implemented
from scripts.logger import logger  # Assuming this exists and is properly implemented
from scripts.state_manager import StateManager  # Assuming this exists and is properly implemented

__description__ = "Provides an enhanced control_system tool for interacting with GitHub issues and accessing project status. This tool leverages context and parameters to identify the relevant repository and execute actions such as retrieving issues, closing issues, and creating new issues. It also accesses project health scores and issue counts from the state manager."
__parameters__ = {
    "repo": "Repository name (e.g., my-repo). Can be inferred from context.",
    "action": "Action to perform (get_issues, create_issue, close_issue, get_health_score, get_open_issue_count, search_issues, get_issue_details, update_issue, add_comment, list_comments, get_pull_requests, merge_pull_request, get_workflow_status). Defaults to get_issues.",
    "title": "Issue title (required for create_issue, update_issue).",
    "body": "Issue body (required for create_issue, update_issue, add_comment).",
    "labels": "List of issue labels (optional for create_issue, update_issue).",
    "issue_number": "Issue number (required for close_issue, get_issue_details, update_issue, add_comment, list_comments).",
    "search_query": "Search query for issues (required for search_issues).",
    "limit": "Maximum number of issues to retrieve (optional for get_issues, defaults to 20).",
    "state": "State of the issue to filter by (open, closed, all) - applicable to get_issues and search_issues. Defaults to open.",
    "assignees": "List of assignees for the issue (applicable to create_issue and update_issue).",
    "milestone": "Milestone for the issue (applicable to create_issue and update_issue).",
    "comment_body": "Body of the comment to add (required for add_comment).",
    "pull_request_number": "Pull request number (required for merge_pull_request).",
    "workflow_id": "Workflow ID to check status (required for get_workflow_status).",
    "workflow_run_id": "Workflow Run ID to check status (alternative to workflow_id).",
    "merge_method": "Merge method to use (merge, squash, rebase) for merge_pull_request. Defaults to merge.",
}
__examples__ = [
    {'description': 'Get recent issues for a repository.', 'example': "await control_system(repo='my-repo')"},
    {'description': 'Create a new issue in a repository.', 'example': "await control_system(repo='my-repo', action='create_issue', title='Bug Report', body='Detailed description.')"},
    {'description': 'Close an existing issue in a repository.', 'example': "await control_system(repo='my-repo', action='close_issue', issue_number=123)"},
    {'description': 'Get project health score from the state manager.', 'example': "await control_system(repo='my-repo', action='get_health_score')"},
    {'description': 'Get open issue count from the state manager.', 'example': "await control_system(repo='my-repo', action='get_open_issue_count')"},
    {'description': 'Automatically detect repo from context.', 'example': "await control_system(context={'user_input': 'Fix this issue in my-repo.'})"},
    {'description': 'Search issues based on a query.', 'example': "await control_system(repo='my-repo', action='search_issues', search_query='performance issue')"},
    {'description': 'Get details of a specific issue.', 'example': "await control_system(repo='my-repo', action='get_issue_details', issue_number=456)"},
    {'description': 'Update an existing issue.', 'example': "await control_system(repo='my-repo', action='update_issue', issue_number=789, title='Updated Title', body='Updated Description', labels=['enhancement']"},
    {'description': 'Add a comment to an issue.', 'example': "await control_system(repo='my-repo', action='add_comment', issue_number=101, comment_body='This is a helpful comment.')"},
    {'description': 'List comments on an issue.', 'example': "await control_system(repo='my-repo', action='list_comments', issue_number=101)"},
    {'description': 'Get pull requests for a repository.', 'example': "await control_system(repo='my-repo', action='get_pull_requests')"},
    {'description': 'Merge a pull request.', 'example': "await control_system(repo='my-repo', action='merge_pull_request', pull_request_number=22)"},
    {'description': 'Get workflow status by ID.', 'example': "await control_system(repo='my-repo', action='get_workflow_status', workflow_id=12345)"},
    {'description': 'Get workflow status by Run ID.', 'example': "await control_system(repo='my-repo', action='get_workflow_status', workflow_run_id=123456)"}
]

# Define a simple cache (in-memory)
_cache: Dict[str, Any] = {}
_cache_expiry: Dict[str, float] = {}
_cache_ttl = 60  # Cache time-to-live in seconds


async def control_system(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provides an enhanced control_system tool for interacting with GitHub.
    """
    repo_name = kwargs.get('repo_name') or kwargs.get('repo')
    action = kwargs.get('action', 'get_issues')  # Default action is to get issues
    limit = kwargs.get('limit', 20)
    state = kwargs.get('state', 'open')  # Default issue state is open
    merge_method = kwargs.get('merge_method', 'merge') # Default merge method

    try:
        # Input Validation
        if not isinstance(limit, int):
            return {"error": "Limit must be an integer"}

        if state not in ['open', 'closed', 'all']:
            return {"error": "State must be one of 'open', 'closed', or 'all'"}

        if not isinstance(merge_method, str):
            return {"error": "Merge method must be a string."}

        if action in ['create_issue'] and not all(key in kwargs for key in ['title', 'body']):
            return {"error": "Title and body are required for create_issue action."}
        if action in ['close_issue', 'get_issue_details', 'update_issue', 'add_comment', 'list_comments'] and 'issue_number' not in kwargs:
            return {"error": "Issue number is required for close_issue, get_issue_details, update_issue, add_comment, and list_comments actions."}
        if action == 'search_issues' and 'search_query' not in kwargs:
            return {"error": "Search query is required for search_issues action."}
        if action == 'update_issue' and not any(key in kwargs for key in ['title', 'body', 'labels', 'assignees', 'milestone']):
            return {"error": "At least one of title, body, labels, assignees, or milestone must be provided for update_issue action."}
        if action == 'add_comment' and 'comment_body' not in kwargs:
            return {"error": "Comment body is required for add_comment action."}
        if action == 'merge_pull_request' and 'pull_request_number' not in kwargs:
            return {"error": "Pull request number is required for merge_pull_request action."}
        if action == 'get_workflow_status' and not ('workflow_id' in kwargs or 'workflow_run_id' in kwargs):
            return {"error": "Either workflow_id or workflow_run_id is required for get_workflow_status action."}
        if action == 'get_workflow_status' and 'workflow_id' in kwargs and 'workflow_run_id' in kwargs:
            logger.warning("Both workflow_id and workflow_run_id are provided. workflow_run_id will be used.")

        # If not provided, try to extract from context
        if not repo_name:
            context = kwargs.get('context', {})
            user_input = context.get('user_input', '')
            repo_patterns = [
                r'(?:in|for|about|within)\s+([a-zA-Z0-9-]+(?:-ai)?)',
                r'([a-zA-Z0-9-]+(?:-ai))\s+(?:repository|repo|project)',
                r'([a-zA-Z0-9-]+(?:-ai))'
            ]

            for pattern in repo_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    repo_name = match.group(1)
                    break

        if not repo_name:
            return {"error": "Could not determine repository name from context or parameters"}

        if '/' not in repo_name:
            repo_name = f"CodeWebMobile-AI/{repo_name}"

        mcp_creator = MCPGitHubIssueCreator()
        state_manager = StateManager()

        # Caching Implementation
        cache_key = f"{repo_name}_{action}"
        if action == 'get_issues':
            cache_key += f"_{limit}_{state}"
        if action == 'search_issues':
            cache_key += f"_{kwargs.get('search_query', '')}_{state}"
        if action in ['get_health_score', 'get_open_issue_count']:
            cache_key = f"state_{repo_name}_{action}"
        if action == 'get_issue_details':
            cache_key += f"_{kwargs.get('issue_number')}"
        if action == 'list_comments':
            cache_key += f"_{kwargs.get('issue_number')}"  # Include issue number in cache key
        if action == 'get_pull_requests':
            cache_key += f"_{limit}_{state}"
        if action == 'get_workflow_status':
            cache_key += f"_{kwargs.get('workflow_id') or kwargs.get('workflow_run_id')}"

        if cache_key in _cache and asyncio.get_event_loop().time() < _cache_expiry.get(cache_key, 0):
            logger.info(f"Returning {action} for {repo_name} from cache.")
            return _cache[cache_key]

        if action == 'get_issues':
            issues = await mcp_creator.list_recent_issues(repo=repo_name, limit=limit, state=state)
            result = {"repository": repo_name, "issues": issues, "total": len(issues), "state": state, "limit": limit}

        elif action == 'create_issue':
            title = kwargs.get('title', 'New Issue')
            body = kwargs.get('body', 'Issue Description')
            labels = kwargs.get('labels', []) or []  # Ensure it's a list
            assignees = kwargs.get('assignees', []) or []
            milestone = kwargs.get('milestone')

            if not isinstance(labels, list):
                return {"error": "Labels must be a list."}
            if not isinstance(assignees, list):
                return {"error": "Assignees must be a list."}

            try:
                issue = await mcp_creator.create_issue(repo=repo_name, title=title, body=body, labels=labels, assignees=assignees, milestone=milestone)
                result = {"repository": repo_name, "issue": issue}
            except Exception as e:
                logger.error(f"Failed to create issue: {e}")
                return {"repository": repo_name, "error": f"Failed to create issue: {e}", "traceback": traceback.format_exc()}

        elif action == 'close_issue':
            issue_number = kwargs.get('issue_number')
            try:
                issue_number = int(issue_number)  # Ensure issue_number is an integer
            except (ValueError, TypeError):
                return {"error": "Issue number must be an integer."}

            try:
                await mcp_creator.close_issue(repo=repo_name, issue_number=issue_number)
                result = {"repository": repo_name, "issue_number": issue_number, "status": "closed"}
            except Exception as e:
                logger.error(f"Failed to close issue: {e}")
                return {"repository": repo_name, "issue_number": issue_number, "status": "failed", "error": str(e)}

        elif action == 'get_health_score':
            projects = state_manager.get_state().get('projects', {})  # Safe access
            if repo_name in projects:
                health_score = projects[repo_name].get('health_score', 'N/A')
                result = {"repository": repo_name, "health_score": health_score}
            else:
                result = {"repository": repo_name, "health_score": "Repository not found in state"}

        elif action == 'get_open_issue_count':
            projects = state_manager.get_state().get('projects', {})  # Safe access
            if repo_name in projects:
                open_issues_count = projects[repo_name].get('open_issues_count', 0)
                result = {"repository": repo_name, "open_issues_count": open_issues_count}
            else:
                result = {"repository": repo_name, "open_issues_count": "Repository not found in state"}

        elif action == 'search_issues':
            search_query = kwargs.get('search_query')
            if not search_query:
                return {"error": "Search query is required for search_issues action."}
            issues = await mcp_creator.search_issues(repo=repo_name, query=search_query, state=state)
            result = {"repository": repo_name, "search_query": search_query, "issues": issues, "total": len(issues), "state": state}

        elif action == 'get_issue_details':
            issue_number = kwargs.get('issue_number')
            try:
                issue_number = int(issue_number)
            except (ValueError, TypeError):
                return {"error": "Issue number must be an integer."}

            try:
                issue = await mcp_creator.get_issue(repo=repo_name, issue_number=issue_number)
                result = {"repository": repo_name, "issue": issue}
            except Exception as e:
                logger.error(f"Failed to get issue details: {e}")
                return {"repository": repo_name, "issue_number": issue_number, "error": str(e), "traceback": traceback.format_exc()}

        elif action == 'update_issue':
            issue_number = kwargs.get('issue_number')
            title = kwargs.get('title')
            body = kwargs.get('body')
            labels = kwargs.get('labels')
            assignees = kwargs.get('assignees')
            milestone = kwargs.get('milestone')

            try:
                issue_number = int(issue_number)
            except (ValueError, TypeError):
                return {"error": "Issue number must be an integer."}

            try:
                updated_issue = await mcp_creator.update_issue(repo=repo_name, issue_number=issue_number, title=title, body=body, labels=labels, assignees=assignees, milestone=milestone)
                result = {"repository": repo_name, "issue_number": issue_number, "updated_issue": updated_issue}
            except Exception as e:
                logger.error(f"Failed to update issue: {e}")
                return {"repository": repo_name, "issue_number": issue_number, "error": str(e), "traceback": traceback.format_exc()}

        elif action == 'add_comment':
            issue_number = kwargs.get('issue_number')
            comment_body = kwargs.get('comment_body')

            try:
                issue_number = int(issue_number)
            except (ValueError, TypeError):
                return {"error": "Issue number must be an integer."}

            try:
                comment = await mcp_creator.add_comment(repo=repo_name, issue_number=issue_number, body=comment_body)
                result = {"repository": repo_name, "issue_number": issue_number, "comment": comment}
            except Exception as e:
                logger.error(f"Failed to add comment to issue: {e}")
                return {"repository": repo_name, "issue_number": issue_number, "error": str(e), "traceback": traceback.format_exc()}

        elif action == 'list_comments':
            issue_number = kwargs.get('issue_number')
            try:
                issue_number = int(issue_number)
            except (ValueError, TypeError):
                return {"error": "Issue number must be an integer."}

            try:
                comments = await mcp_creator.list_comments(repo=repo_name, issue_number=issue_number)
                result = {"repository": repo_name, "issue_number": issue_number, "comments": comments, "total": len(comments)}
            except Exception as e:
                logger.error(f"Failed to list comments for issue: {e}")
                return {"repository": repo_name, "issue_number": issue_number, "error": str(e), "traceback": traceback.format_exc()}

        elif action == 'get_pull_requests':
            try:
                pull_requests = await mcp_creator.list_pull_requests(repo=repo_name, limit=limit, state=state)
                result = {"repository": repo_name, "pull_requests": pull_requests, "total": len(pull_requests), "state": state, "limit": limit}
            except Exception as e:
                logger.error(f"Failed to get pull requests: {e}")
                return {"repository": repo_name, "error": str(e), "traceback": traceback.format_exc()}

        elif action == 'merge_pull_request':
            pull_request_number = kwargs.get('pull_request_number')
            try:
                pull_request_number = int(pull_request_number)
            except (ValueError, TypeError):
                return {"error": "Pull request number must be an integer."}
            try:
                await mcp_creator.merge_pull_request(repo=repo_name, pull_number=pull_request_number, merge_method=merge_method)
                result = {"repository": repo_name, "pull_request_number": pull_request_number, "status": "merged"}
            except Exception as e:
                logger.error(f"Failed to merge pull request: {e}")
                return {"repository": repo_name, "pull_request_number": pull_request_number, "status": "failed", "error": str(e)}

        elif action == 'get_workflow_status':
            workflow_id = kwargs.get('workflow_id')
            workflow_run_id = kwargs.get('workflow_run_id')

            try:
                if workflow_run_id:
                    workflow_run_id = int(workflow_run_id)
                    status = await mcp_creator.get_workflow_status_by_run_id(repo=repo_name, run_id=workflow_run_id)
                    result = {"repository": repo_name, "workflow_run_id": workflow_run_id, "status": status}
                else:
                    workflow_id = int(workflow_id)
                    status = await mcp_creator.get_workflow_status(repo=repo_name, workflow_id=workflow_id)
                    result = {"repository": repo_name, "workflow_id": workflow_id, "status": status}
            except (ValueError, TypeError) as e:
                return {"error": "Workflow ID or Run ID must be an integer."}
            except Exception as e:
                logger.error(f"Failed to get workflow status: {e}")
                return {"repository": repo_name, "error": str(e), "traceback": traceback.format_exc()}

        else:
            return {"error": f"Unknown action: {action}"}

        # Store the result in the cache
        if action not in ['create_issue', 'close_issue', 'update_issue', 'add_comment', 'merge_pull_request']:  # Don't cache actions that modify state
            _cache[cache_key] = result
            _cache_expiry[cache_key] = asyncio.get_event_loop().time() + _cache_ttl
        return result


    except Exception as e:
        error_message = f"Error in control_system for action {action} on repo {repo_name}: {e}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        return {"error": error_message, "traceback": traceback.format_exc()}