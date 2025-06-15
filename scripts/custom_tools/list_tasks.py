"""
AI-Generated Tool: list_tasks
Description: List items for tasks
Generated: 2025-06-15T10:37:35.999733+00:00
Requirements: 
        Tool Name: list_tasks
        Intent: List items for tasks
        Expected Parameters: {}
        Category: query
        
        The tool should:
        1. List all tasks with relevant details
2. Support filtering and pagination
3. Return structured data
        
"""

"""
Module for listing tasks with filtering and pagination support.

This module provides an asynchronous function `list_tasks` that retrieves
and returns a structured list of tasks with relevant details. The function
supports filtering and pagination, making it convenient for querying tasks
based on specific criteria.

Requirements:
- Tool Name: list_tasks
- Intent: List items for tasks
- Expected Parameters: {}
- Category: query
"""

from scripts.state_manager import StateManager
from scripts.task_manager import TaskManager
import asyncio

__description__ = "List items for tasks"
__parameters__ = {}
__examples__ = """
Example usage:

tasks = await list_tasks(page=1, limit=10, filter_by='status:completed')
"""

async def list_tasks(page: int = 1, limit: int = 10, filter_by: str = None) -> dict:
    """
    List all tasks with relevant details, supporting filtering and pagination.

    :param page: The page number for pagination.
    :param limit: The number of items per page.
    :param filter_by: Optional filter criteria (e.g., 'status:completed').
    :return: A dictionary with the total number of tasks and the list of tasks.
    """
    try:
        # Initialize state and task managers
        state_manager = StateManager()
        task_manager = TaskManager()
        
        # Load current state (if needed for further operations)
        state = state_manager.load_state()
        
        # Fetch tasks using TaskManager with pagination and filtering
        tasks, total = await task_manager.get_tasks(page=page, limit=limit, filter_by=filter_by)
        
        # Return structured data
        return {
            "total": total,
            "tasks": tasks
        }
        
    except Exception as e:
        # Handle any exceptions and return an error message
        return {
            "error": str(e)
        }

# Example of how the TaskManager might be implemented
class TaskManager:
    async def get_tasks(self, page: int, limit: int, filter_by: str = None) -> tuple:
        # This is a mocked implementation for demonstration purposes.
        # In a real implementation, this would query a database or external API.
        
        # Sample data
        all_tasks = [
            {"id": 1, "name": "Task 1", "status": "completed"},
            {"id": 2, "name": "Task 2", "status": "pending"},
            {"id": 3, "name": "Task 3", "status": "completed"},
            # More tasks...
        ]
        
        # Apply filtering if specified
        if filter_by:
            key, value = filter_by.split(':')
            filtered_tasks = [task for task in all_tasks if task.get(key) == value]
        else:
            filtered_tasks = all_tasks
        
        # Calculate pagination
        start = (page - 1) * limit
        end = start + limit
        paginated_tasks = filtered_tasks[start:end]
        
        # Return paginated and filtered tasks
        return paginated_tasks, len(filtered_tasks)
