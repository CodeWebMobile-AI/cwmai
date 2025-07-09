import logging
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.state_manager import StateManager

__description__ = "Analyzes repository health using existing tools"

def safe_get(data, *keys, default=None):
    """Safely navigate nested dictionaries."""
    result = data
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result

async def analyze_repository_health(repo_name: str):
    """
    Analyzes repository health using the RepositoryAnalyzer and StateManager.

    Args:
        repo_name (str): The name of the repository to analyze.

    Returns:
        dict: A dictionary containing the analysis result.
    """
    logger = logging.getLogger(__name__)
    try:
        # 1. Analyze the repository using RepositoryAnalyzer
        analyzer = RepositoryAnalyzer()
        analysis_result = await analyzer.analyze_repository(repo_name)

        if not analysis_result:
            return {"success": False, "result": None, "error": f"Failed to analyze repository: {repo_name}"}

        health_score = safe_get(analysis_result, 'health_metrics', 'health_score', default=0)
        open_issues_count = safe_get(analysis_result, 'issue_summary', 'open_issues', default=0)
        code_quality = safe_get(analysis_result, 'code_quality', 'overall_score', default=0)
        recent_activity_score = safe_get(analysis_result, 'recent_activity', 'activity_score', default=0)
        description = safe_get(analysis_result, 'basic_info', 'description', default="No description available.")

        health_summary = (
            f"Health Score: {health_score}\n"
            f"Open Issues: {open_issues_count}\n"
            f"Code Quality: {code_quality}\n"
            f"Recent Activity: {recent_activity_score}\n"
            f"Description: {description}"
        )

        result = {
            "repo_name": repo_name,
            "health_score": health_score,
            "open_issues_count": open_issues_count,
            "code_quality": code_quality,
            "recent_activity_score": recent_activity_score,
            "description": description,
            "health_summary": health_summary
        }

        return {"success": True, "result": result, "error": None}

    except Exception as e:
        logger.exception(f"Error analyzing repository health for {repo_name}: {e}")
        return {"success": False, "result": None, "error": str(e)}