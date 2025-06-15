"""
AI-Generated Tool: tool_task_1
Description: Retrieve all error messages from the log data.
Generated: 2025-06-15T12:43:38.949473+00:00
Requirements: Tool to Retrieve all error messages from the log data.
"""

from typing import Any
from typing import Dict
from typing import Dict, Any

import json
import re

from scripts.state_manager import StateManager


"""
Module for tool_task_1: Retrieve all error messages from log data.
"""



__description__ = "Retrieve all error messages from the log data."

__parameters__ = {
    "log_file_path": {
        "type": "string",
        "description": "Path to the log file.",
        "required": True,
    },
    "error_regex": {
        "type": "string",
        "description": "Regular expression to match error messages.",
        "default": "ERROR:.*",
    },
}

__examples__ = [
    {
        "description": "Retrieve error messages from a specific log file with the default regex.",
        "parameters": {"log_file_path": "/path/to/your/log_file.txt"},
    },
    {
        "description": "Retrieve error messages from a specific log file with a custom regex.",
        "parameters": {
            "log_file_path": "/path/to/your/log_file.txt",
            "error_regex": "CRITICAL:.*",
        },
    },
]


async def tool_task_1(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves all error messages from the log data based on a regular expression.

    Args:
        **kwargs: Keyword arguments containing tool parameters.
            - log_file_path (str): Path to the log file.
            - error_regex (str, optional): Regular expression to match error messages. Defaults to "ERROR:.*".

    Returns:
        Dict[str, Any]: A dictionary containing the extracted error messages and a summary.
            Example: {"errors": ["ERROR: Something went wrong", "ERROR: Another error occurred"], "total": 2}
    """
    state_manager = StateManager()

    try:
        # Validate input parameters
        if "log_file_path" not in kwargs:
            return {"error": "Missing required parameter: log_file_path"}

        log_file_path = kwargs["log_file_path"]
        error_regex = kwargs.get("error_regex", "ERROR:.*")

        # Load the log data
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_data = f.read()
        except FileNotFoundError:
            return {"error": f"Log file not found: {log_file_path}"}
        except Exception as e:
            return {"error": f"Failed to load log data: {str(e)}"}

        # Extract error messages
        try:
            errors = re.findall(error_regex, log_data)
        except re.error as e:
            return {"error": f"Invalid regular expression: {str(e)}"}

        # Persist state (example, adjust as needed)
        state = state_manager.load_state()
        state["last_error_count"] = len(errors)
        state_manager.save_state(state)

        # Return the results
        return {"errors": errors, "total": len(errors)}

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        return {"error": error_message}
