"""
AI-Generated Tool: tool_task_4
Description: Identify and remove comments from each file's content.
Generated: 2025-06-15T12:42:32.107447+00:00
Requirements: Tool to Identify and remove comments from each file's content.
"""

import asyncio
import re

from scripts.state_manager import StateManager


"""
Module: tool_task_4

This module provides a tool to identify and remove comments from the content of files.
"""


__description__ = "Identify and remove comments from each file's content."
__parameters__ = ["files_content: List[str]"]
__examples__ = {
    "Example 1": {
        "files_content": ["# Comment\nprint('Hello, World!')", "print('No comments here')"],
        "result": ["print('Hello, World!')", "print('No comments here')"]
    }
}

async def tool_task_4(files_content):
    """
    Asynchronously identifies and removes comments from the content of given files.

    Args:
        files_content (list of str): List containing the content of each file as a string.

    Returns:
        dict: A dictionary containing the cleaned content of each file, without comments.
    """
    # Initialize state manager
    state_manager = StateManager()
    
    try:
        # Validate input
        if not isinstance(files_content, list):
            raise ValueError("files_content should be a list of strings.")

        cleaned_contents = []

        # Regex pattern to match comments (both single line and inline)
        comment_pattern = re.compile(r"(?:^|\s)#.*?$", re.MULTILINE)

        for content in files_content:
            if not isinstance(content, str):
                raise ValueError("Each file content should be a string.")
            # Remove comments using regex
            cleaned_content = re.sub(comment_pattern, '', content)
            cleaned_contents.append(cleaned_content.strip())
        
        # Return cleaned contents
        return {"cleaned_contents": cleaned_contents}

    except Exception as e:
        # Handle and log exceptions
        state_manager.log_error(f"Error in tool_task_4: {e}")
        return {"error": str(e)}

# Note: This module does not include any top-level testing code or __main__ entry point.
