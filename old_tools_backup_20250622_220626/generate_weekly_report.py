"""
Module for generating weekly development reports.

This module provides a tool that aggregates data from multiple sources
(tasks, repositories, system activity, and code quality) and formats it
into a standardized report template.
"""

import asyncio
import json
from typing import Optional, Dict, Any, List

from datetime import datetime, timedelta
import logging

# Assuming these modules exist in your system
# Replace with actual import paths if needed
# from scripts.state_manager import StateManager  # Example
# from scripts.data_fetcher import get_tasks, count_repositories, get_tool_usage_stats, repository_health_check # Example

__description__ = "Generates a weekly development report aggregating data from various sources."

__parameters__ = {
    "include_graphs": {
        "type": "boolean",
        "description": "Whether to include graphs in the report.",
        "default": True,
    },
    "format": {
        "type": "string",
        "description": "The format of the report (e.g., 'markdown').",
        "default": "markdown",
        "enum": ["markdown", "json", "html"],  # Add supported formats
    },
    "cache_expiration_seconds": {
        "type": "integer",
        "description": "Cache expiration time in seconds. Set to 0 to disable caching.",
        "default": 3600,  # 1 hour default
        "minimum": 0,
    },
    "error_retries": {
        "type": "integer",
        "description": "Number of retries for data fetching on failure.",
        "default": 3,
        "minimum": 0,
    },
}

__examples__ = [
    {
        "command": "generate_weekly_report(include_graphs=True, format='markdown')",
        "description": "Generates a weekly report in markdown format with graphs.",
    },
    {
        "command": "generate_weekly_report(include_graphs=False, format='json')",
        "description": "Generates a weekly report in JSON format without graphs.",
    },
    {
        "command": "generate_weekly_report(include_graphs=True, format='markdown', cache_expiration_seconds=7200)",
        "description": "Generates a weekly report in markdown format with graphs and a 2 hour cache expiration.",
    },
    {
        "command": "generate_weekly_report(include_graphs=True, format='markdown', cache_expiration_seconds=0)",
        "description": "Generates a weekly report in markdown format with graphs and disables caching.",
    },
    {
        "command": "generate_weekly_report(include_graphs=True, format='html', error_retries=5)",
        "description": "Generates a weekly report in HTML format with graphs and retries data fetching 5 times on failure.",
    },
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simple in-memory cache (replace with a more robust solution in production)
_report_cache: Dict[str, Dict[str, Any]] = {}
_report_cache_timestamps: Dict[str, datetime] = {}


async def get_tasks(start_date, end_date):
    """Placeholder for fetching task data.  Replace with actual implementation."""
    # Simulate fetching tasks - in reality, query a database or API
    # Simulate a potential failure for demonstration purposes.

    if datetime.now().second % 10 == 0:  # Simulate intermittent failure
        raise ConnectionError("Simulated task fetching failure.")

    tasks = [
        {"id": 1, "name": "Task A", "status": "completed", "created_at": start_date + timedelta(days=1)},
        {"id": 2, "name": "Task B", "status": "active", "created_at": start_date + timedelta(days=2)},
        {"id": 3, "name": "Task C", "status": "completed", "created_at": start_date + timedelta(days=3)},
        {"id": 4, "name": "Task D", "status": "active", "created_at": start_date + timedelta(days=4)},
    ]

    #Filter tasks within date range:
    filtered_tasks = [task for task in tasks if start_date <= task['created_at'] <= end_date]

    return filtered_tasks


async def count_repositories():
    """Placeholder for counting repositories. Replace with actual implementation."""

    if datetime.now().second % 15 == 0:
        raise ValueError("Simulated repository counting failure.")

    # Simulate repository count retrieval
    return {"total_repositories": 15, "active_repositories": 12}


async def get_tool_usage_stats(start_date, end_date):
    """Placeholder for fetching tool usage statistics. Replace with actual implementation."""
    # Simulate tool usage stats

    if datetime.now().second % 20 == 0:
        raise TimeoutError("Simulated tool usage stats timeout.")

    return {
        "api_calls": 1200,
        "user_sessions": 500,
        "average_response_time": 0.25,
        "start_date": start_date.isoformat(),  #Ensuring ISO format for consistency
        "end_date": end_date.isoformat(),
    }


async def repository_health_check():
    """Placeholder for repository health check. Replace with actual implementation."""

    if datetime.now().second % 25 == 0:
        raise RuntimeError("Simulated repository health check error.")

    # Simulate repository health check results
    return {
        "critical_issues": 3,
        "major_issues": 10,
        "average_code_coverage": 85.5,
    }


def _validate_format(format_str: str) -> None:
    """Validates the report format."""
    if format_str not in ["markdown", "json", "html"]:
        raise ValueError(f"Invalid format: {format_str}. Supported formats are 'markdown', 'json', and 'html'.")


def _validate_cache_expiration(cache_expiration_seconds: int) -> None:
    """Validates the cache expiration time."""
    if not isinstance(cache_expiration_seconds, int):
        raise TypeError("cache_expiration_seconds must be an integer.")
    if cache_expiration_seconds < 0:
        raise ValueError("cache_expiration_seconds must be a non-negative integer.")


def _generate_cache_key(include_graphs: bool, format: str, start_date: datetime, end_date: datetime) -> str:
    """Generates a unique cache key based on the input parameters."""
    return f"report_graphs_{include_graphs}_format_{format}_start_{start_date.isoformat()}_end_{end_date.isoformat()}"


async def _fetch_data_with_retry(data_fetcher, start_date, end_date, max_retries: int = 3):
    """Fetches data with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            if data_fetcher.__name__ in ["get_tasks"]:
                return await data_fetcher(start_date, end_date)
            elif data_fetcher.__name__ in ["get_tool_usage_stats"]:
                return await data_fetcher(start_date, end_date)
            else:
                return await data_fetcher()  # Other data fetchers take no arguments

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {data_fetcher.__name__}: {e}")
            if attempt == max_retries:
                logging.error(f"Max retries reached for {data_fetcher.__name__}.  Failing.")
                raise  # Re-raise the exception after all retries have failed
            await asyncio.sleep(2 ** attempt)  # Exponential backoff


def _format_as_html(report_data: Dict[str, Any]) -> str:
    """Formats the report data as HTML."""
    start_date = datetime.fromisoformat(report_data['report_period']['start_date'])
    end_date = datetime.fromisoformat(report_data['report_period']['end_date'])

    html_string = f"""
<!DOCTYPE html>
<html>
<head>
<title>Weekly Development Report</title>
</head>
<body>
<h1>Weekly Development Report ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})</h1>

<h2>Tasks:</h2>
<ul>
    <li>Completed: {report_data['tasks']['completed']}</li>
    <li>Active: {report_data['tasks']['active']}</li>
    <li>Total: {report_data['tasks']['total_tasks']}</li>
</ul>

<h2>Repositories:</h2>
<ul>
    <li>Total Repositories: {report_data['repositories']['total_repositories']}</li>
    <li>Active Repositories: {report_data['repositories']['active_repositories']}</li>
</ul>

<h2>System Activity:</h2>
<ul>
    <li>API Calls: {report_data['system_activity']['api_calls']}</li>
    <li>User Sessions: {report_data['system_activity']['user_sessions']}</li>
    <li>Average Response Time: {report_data['system_activity']['average_response_time']} s</li>
</ul>

<h2>Code Quality:</h2>
<ul>
    <li>Critical Issues: {report_data['code_quality']['critical_issues']}</li>
    <li>Major Issues: {report_data['code_quality']['major_issues']}</li>
    <li>Average Code Coverage: {report_data['code_quality']['average_code_coverage']}%</li>
</ul>

<p><b>Generated At:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>
"""
    return html_string


async def generate_weekly_report(include_graphs: bool = True, format: str = "markdown", cache_expiration_seconds: int = 3600, error_retries: int = 3) -> Dict[str, Any]:
    """
    Generates a weekly development report.

    Args:
        include_graphs: Whether to include graphs in the report.
        format: The format of the report (e.g., 'markdown', 'json').
        cache_expiration_seconds: Cache expiration time in seconds. Set to 0 to disable caching.
        error_retries: Number of retries for data fetching on failure.

    Returns:
        A dictionary containing the generated report data.
    """

    try:
        _validate_format(format)
        _validate_cache_expiration(cache_expiration_seconds)

        # Determine the start and end dates for the report (last week)
        today = datetime.now()
        end_date = today - timedelta(days=today.weekday() + 1)  # Last Sunday
        start_date = end_date - timedelta(days=6)  # Last Monday

        cache_key = _generate_cache_key(include_graphs, format, start_date, end_date)

        # Check cache
        if cache_expiration_seconds > 0 and cache_key in _report_cache:
            if (datetime.now() - _report_cache_timestamps[cache_key]).total_seconds() < cache_expiration_seconds:
                logging.info("Retrieving report from cache.")
                return _report_cache[cache_key]
            else:
                logging.info("Cache expired. Generating new report.")
        else:
            logging.info("Cache disabled or no cache entry found. Generating new report.")


        # Fetch data from various sources concurrently with retry logic
        try:
            tasks_data, repo_stats, usage_stats, health_data = await asyncio.gather(
                _fetch_data_with_retry(get_tasks, start_date, end_date, error_retries),
                _fetch_data_with_retry(count_repositories, start_date, end_date, error_retries),
                _fetch_data_with_retry(get_tool_usage_stats, start_date, end_date, error_retries),
                _fetch_data_with_retry(repository_health_check, start_date, end_date, error_retries),
            )
        except Exception as e:
            logging.error(f"Failed to fetch data after multiple retries: {e}")
            return {"error": f"Failed to fetch data: {e}"}

        # Calculate derived metrics
        completed_tasks = sum(1 for task in tasks_data if task["status"] == "completed")
        active_tasks = sum(1 for task in tasks_data if task["status"] == "active")

        # Create the report data
        report_data = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "tasks": {
                "completed": completed_tasks,
                "active": active_tasks,
                "total_tasks": len(tasks_data),
                "tasks_list": tasks_data  # Include the list of tasks
            },
            "repositories": repo_stats,
            "system_activity": usage_stats,
            "code_quality": health_data,
            "include_graphs": include_graphs,
            "format": format,
            "generated_at": datetime.now().isoformat()  # Add generation timestamp
        }

        # Format the report
        if format == "markdown":
            report_string = f"""
## Weekly Development Report ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})

**Tasks:**
- Completed: {completed_tasks}
- Active: {active_tasks}
- Total: {len(tasks_data)}

**Repositories:**
- Total Repositories: {repo_stats['total_repositories']}
- Active Repositories: {repo_stats['active_repositories']}

**System Activity:**
- API Calls: {usage_stats['api_calls']}
- User Sessions: {usage_stats['user_sessions']}
- Average Response Time: {usage_stats['average_response_time']} s

**Code Quality:**
- Critical Issues: {health_data['critical_issues']}
- Major Issues: {health_data['major_issues']}
- Average Code Coverage: {health_data['average_code_coverage']}%

{'**(Graphs are included)**' if include_graphs else ''}

**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            report_data["report_string"] = report_string
        elif format == "json":
            report_data["report_string"] = json.dumps(report_data, indent=4)
        elif format == "html":
            report_data["report_string"] = _format_as_html(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Store in cache
        if cache_expiration_seconds > 0:
            _report_cache[cache_key] = report_data
            _report_cache_timestamps[cache_key] = datetime.now()
            logging.info("Report stored in cache.")

        return report_data

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        error_message = f"Error generating weekly report: {e}"
        logging.exception(error_message)  # Log the error with traceback

        # Enhanced error reporting - include the traceback (optional)
        import traceback
        traceback_str = traceback.format_exc()
        error_report = {"error": error_message, "traceback": traceback_str}

        return error_report