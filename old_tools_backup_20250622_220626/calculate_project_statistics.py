import os
import re
import asyncio
import logging
import traceback
from typing import Dict, Any

from scripts.code_analyzer import CodeAnalyzer
from scripts.complexity_analyzer import ComplexityAnalyzer
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.state_manager import StateManager
from scripts.logger import logger
from scripts.deduplicating_formatter import DeduplicatingFormatter

# Setup logger with DeduplicatingFormatter
handler = logging.StreamHandler()
handler.setFormatter(DeduplicatingFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Set the desired logging level

__description__ = "Analyzes a project to provide size and complexity metrics, calculating total lines of code, cyclomatic complexity, and potential code duplication. A summary report is generated."
__parameters__ = {}
__examples__ = []

async def calculate_project_statistics(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes a project to provide size and complexity metrics, calculating total lines of code,
    cyclomatic complexity, and potential code duplication. A summary report is generated.
    """
    try:
        # Extract repository name from context
        repo_name = kwargs.get('repo_name') or kwargs.get('repo')
        if not repo_name:
            context = kwargs.get('context', {})
            user_input = context.get('user_input', '')

            # Extract repository name from user input
            repo_patterns = [
                r'(?:in|for|about|within)\s+([a-zA-Z0-9-]+(?:-ai)?)',
                r'([a-zA-Z0-9-]+(?:-ai))\s+(?:repository|repo|project)',
                r'([a-zA-Z0-9-]+(?:-ai))'  # Any word ending in -ai
            ]

            for pattern in repo_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    repo_name = match.group(1)
                    break

        if not repo_name:
            return {"error": "Could not determine repository name from context or parameters"}
        
        # Auto-complete repo name
        if '/' not in repo_name:
            repo_name = f"CodeWebMobile-AI/{repo_name}"

        # Initialize analyzers
        code_analyzer = CodeAnalyzer()
        complexity_analyzer = ComplexityAnalyzer()

        # Placeholder for project directory.  In reality, you would need to clone the repository
        # or have access to the files locally.  For this example, we will just analyze the current directory.
        project_directory = "."

        # Calculate total lines of code
        total_lines_of_code = await code_analyzer.calculate_lines_of_code(project_directory)

        # Calculate cyclomatic complexity
        cyclomatic_complexity = await complexity_analyzer.calculate_average_complexity(project_directory)

        # Calculate code duplication (basic example)
        code_duplication_percentage = await code_analyzer.calculate_code_duplication(project_directory)

        # Generate summary report
        report = {
            "repository": repo_name,
            "total_lines_of_code": total_lines_of_code,
            "cyclomatic_complexity": cyclomatic_complexity,
            "code_duplication_percentage": code_duplication_percentage,
        }

        # Log the report
        logger.info(f"Project Statistics Report: {report}")

        return report

    except Exception as e:
        error_message = f"Error analyzing project: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return {"error": error_message}