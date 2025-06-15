"""
AI-Generated Tool: find_todo_comments_in_python_files
Description: 
The tool should search through all Python files in the project and return the lines containing TODO comments. This tool will help identify areas of code that require further work or documentation.
Generated: 2025-06-15T11:07:23.926796+00:00
Requirements: 
        Based on user request: find all TODO comments in Python files
        Create a tool that: 
The tool should search through all Python files in the project and return the lines containing TODO comments. This tool will help identify areas of code that require further work or documentation.
        Tool should integrate with existing system components
        
"""

"""
Module: find_todo_comments_in_python_files

Description:
This module provides a tool to search through all Python files in the project and return the lines containing TODO comments.
This tool will help identify areas of code that require further work or documentation.
"""

import os
import asyncio
import glob
from typing import Dict, Any, List
from scripts.state_manager import StateManager  # Assuming this exists
from scripts.file_manager import FileManager    # Assuming this exists
from scripts.project_manager import ProjectManager  # Assuming this exists


__description__ = "Finds all TODO comments in Python files within the project."
__parameters__ = {}
__examples__ = [
    {
        "description": "Find TODO comments in all Python files.",
        "input": {},
        "output": {
            "todo_comments": [
                {"file": "example.py", "line_number": 10, "comment": "TODO: Implement this function."},
                {"file": "another_example.py", "line_number": 5, "comment": "TODO: Add better error handling."},
            ]
        }
    }
]


async def find_todo_comments_in_python_files(**kwargs) -> Dict[str, Any]:
    """
    Searches through all Python files in the project and returns the lines containing TODO comments.
    """
    state_manager = StateManager()
    file_manager = FileManager()
    project_manager = ProjectManager()

    try:
        # Get the project root directory from the state.  Handle missing state.
        state = state_manager.load_state()
        project_root = state.get("project_root")
        if not project_root:
            return {"error": "Project root not found in state.  Please initialize project first."}

        todo_comments: List[Dict[str, Any]] = []
        python_files = glob.glob(os.path.join(project_root, "**/*.py"), recursive=True)

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if "TODO" in line:  # Simple check for "TODO" in the line
                            todo_comments.append({
                                "file": file_path,
                                "line_number": i + 1,
                                "comment": line.strip()
                            })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                # Log the error with file_manager
                await file_manager.write_to_file("error_log.txt", f"Error reading file {file_path}: {e}\n", append=True)


        return {"todo_comments": todo_comments}

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        # Log the error using the file_manager
        await file_manager.write_to_file("error_log.txt", error_message + "\n", append=True)
        return {"error": error_message}
