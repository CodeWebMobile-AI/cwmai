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

from scripts.state_manager import StateManager  # Assuming StateManager exists
from scripts.task_manager import TaskManager  # Assuming TaskManager exists
import asyncio
import json
from typing import Optional, List, Dict, Any
import logging
import time

__description__ = "List items for tasks"
__parameters__ = {}
__examples__ = """
Example usage:

tasks = await list_tasks(page=1, limit=10, filter_by='status:completed')
tasks = await list_tasks(sort_by='name', sort_order='asc') # Sort by task name in ascending order.
tasks = await list_tasks(search='keyword') # Search for tasks containing 'keyword' in name or description
tasks = await list_tasks(priority='high') # Filter by priority
tasks = await list_tasks(due_date_start='2024-01-01', due_date_end='2024-01-31') # Filter tasks within a date range
tasks = await list_tasks(tags='important,urgent') # Filter tasks by multiple tags
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simple in-memory cache
_task_cache = {}
_cache_expiry = 60  # Cache expiry in seconds
_cache_enabled = True  # Enable or disable caching
_cache_max_size = 100 #Maximum number of entries in the cache.

def _is_valid_filter(filter_by: str) -> bool:
    """
    Validates the filter_by string.  Ensures it's in the correct format
    and only allows filtering by allowed fields.
    """
    if not filter_by:
        return True

    try:
        key, value = filter_by.split(':')
        allowed_fields = ["status", "name", "priority"]  # Example allowed fields
        if key not in allowed_fields:
            return False
        return True
    except ValueError:
        return False


def _generate_cache_key(page: int, limit: int, filter_by: str, sort_by: str, sort_order: str, search: str,
                        priority: str, due_date_start: str, due_date_end: str, tags: str) -> str:
    """Generates a unique cache key based on the input parameters."""
    return f"tasks:page={page}:limit={limit}:filter={filter_by}:sort_by={sort_by}:sort_order={sort_order}:search={search}:priority={priority}:due_start={due_date_start}:due_end={due_date_end}:tags={tags}"


def _clear_cache():
    """Utility function to clear the cache.  Useful for testing or when data changes."""
    global _task_cache
    _task_cache = {}
    logging.info("Task cache cleared.")

def _trim_cache():
    """Trims the cache to maintain a maximum size.  Removes the oldest entries."""
    global _task_cache
    if len(_task_cache) > _cache_max_size:
        # Sort cache entries by their creation time (expiry time is a proxy)
        sorted_cache = sorted(_task_cache.items(), key=lambda item: item[1]['expiry'])
        num_to_remove = len(_task_cache) - _cache_max_size
        for i in range(num_to_remove):
            del _task_cache[sorted_cache[i][0]]
        logging.info(f"Task cache trimmed. Removed {num_to_remove} entries.")

def _validate_date(date_str: str) -> bool:
    """Validates that a string is a valid date in YYYY-MM-DD format."""
    if not date_str:
        return True  # Allow empty dates (no filtering)
    try:
        time.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


async def list_tasks(page: int = 1, limit: int = 10, filter_by: str = None, sort_by: str = None,
                       sort_order: str = 'asc', search: str = None, priority: str = None,
                       due_date_start: str = None, due_date_end: str = None, tags: str = None) -> dict:
    """
    List all tasks with relevant details, supporting filtering and pagination.

    :param page: The page number for pagination.
    :param limit: The number of items per page.
    :param filter_by: Optional filter criteria (e.g., 'status:completed').
    :param sort_by: Optional field to sort by (e.g., 'name', 'status').
    :param sort_order: Optional sort order ('asc' or 'desc'). Defaults to 'asc'.
    :param search: Optional search term to filter by name or description.
    :param priority: Optional priority level to filter by (e.g., 'high', 'medium', 'low').
    :param due_date_start: Optional start date (YYYY-MM-DD) to filter by due date.
    :param due_date_end: Optional end date (YYYY-MM-DD) to filter by due date.
    :param tags: Optional comma-separated list of tags to filter by.
    :return: A dictionary with the total number of tasks and the list of tasks.
    """

    # Input Validation
    if not isinstance(page, int) or page < 1:
        return {"error": "Invalid page number. Must be an integer greater than 0.", "status": "failed"}
    if not isinstance(limit, int) or limit < 1 or limit > 100:  # Reasonable limit
        return {"error": "Invalid limit. Must be an integer between 1 and 100.", "status": "failed"}
    if filter_by and not _is_valid_filter(filter_by):
        return {"error": "Invalid filter format or filter field not allowed. Use format 'field:value' and allowed fields are status, name, and priority.", "status": "failed"}
    if sort_by and sort_by not in ["name", "status", "id", "priority", "due_date"]:  # Allowed sorting fields
        return {"error": "Invalid sort_by field. Allowed fields are name, status, id, priority, and due_date.", "status": "failed"}
    if sort_order not in ["asc", "desc"]:
        return {"error": "Invalid sort_order. Must be 'asc' or 'desc'.", "status": "failed"}
    if priority and priority not in ["high", "medium", "low"]:
        return {"error": "Invalid priority. Must be 'high', 'medium', or 'low'.", "status": "failed"}
    if due_date_start and not _validate_date(due_date_start):
        return {"error": "Invalid due_date_start format. Must be YYYY-MM-DD.", "status": "failed"}
    if due_date_end and not _validate_date(due_date_end):
        return {"error": "Invalid due_date_end format. Must be YYYY-MM-DD.", "status": "failed"}

    cache_key = _generate_cache_key(page, limit, filter_by, sort_by, sort_order, search, priority, due_date_start, due_date_end, tags)

    # Check cache
    if _cache_enabled and cache_key in _task_cache and _task_cache[cache_key]["expiry"] > asyncio.get_event_loop().time():
        logging.info("Cache hit!")
        return _task_cache[cache_key]["data"]

    try:
        # Initialize state and task managers
        state_manager = StateManager()  # Instantiate StateManager
        task_manager = TaskManager()

        # Load current state (if needed for further operations)
        try:
            state = state_manager.load_state()
        except Exception as e:
            logging.warning(f"Could not load state: {e}")
            state = None  # Or handle the lack of state appropriately

        # Fetch tasks using TaskManager with pagination and filtering
        tasks, total = await task_manager.get_tasks(page=page, limit=limit, filter_by=filter_by, sort_by=sort_by,
                                                      sort_order=sort_order, search=search, priority=priority,
                                                      due_date_start=due_date_start, due_date_end=due_date_end, tags=tags)

        # Prepare comprehensive output
        output = {
            "total": total,
            "page": page,
            "limit": limit,
            "filter": filter_by,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "search": search,
            "priority": priority,
            "due_date_start": due_date_start,
            "due_date_end": due_date_end,
            "tags": tags,
            "tasks": tasks,
            "metadata": {
                "timestamp": asyncio.get_event_loop().time(),
                "data_source": "TaskManager"
            },
            "status": "success"  # Indicate success
        }

        # Store in cache
        if _cache_enabled:
            # Trim cache if it exceeds maximum size before adding new entry
            _trim_cache()
            _task_cache[cache_key] = {
                "data": output,
                "expiry": asyncio.get_event_loop().time() + _cache_expiry
            }

        return output

    except Exception as e:
        # Handle any exceptions and return an error message
        error_message = f"Error processing request: {str(e)}"
        logging.exception(error_message)  # Log the error with traceback
        return {
            "error": error_message,
            "status": "failed",
            "timestamp": asyncio.get_event_loop().time()
        }

def get_task_by_id(task_id: int) -> Dict[str, Any]:
    """
    Retrieves a single task by its ID.  This is a utility function
    that might be needed by other tools.
    """
    try:
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        if task:
            return {"task": task, "status": "success"}
        else:
            return {"error": f"Task with ID {task_id} not found.", "status": "failed"}
    except Exception as e:
        error_message = f"Error retrieving task: {str(e)}"
        logging.exception(error_message)
        return {"error": error_message, "status": "failed"}


# Example of how the TaskManager might be implemented
class TaskManager:
    async def get_tasks(self, page: int, limit: int, filter_by: str = None, sort_by: str = None,
                       sort_order: str = 'asc', search: str = None, priority: str = None,
                       due_date_start: str = None, due_date_end: str = None, tags: str = None) -> tuple[List[Dict], int]:
        """
        Retrieves tasks, applying filtering, sorting, and pagination.

        :param page: The page number for pagination.
        :param page: The page number for pagination.
        :param limit: The number of items per page.
        :param filter_by: Optional filter criteria (e.g., 'status:completed').
        :param sort_by: Optional field to sort by (e.g., 'name', 'status').
        :param sort_order: Optional sort order ('asc' or 'desc'). Defaults to 'asc'.
        :param search: Optional search term to filter by name or description.
        :param priority: Optional priority level to filter by (e.g., 'high', 'medium', 'low').
        :param due_date_start: Optional start date (YYYY-MM-DD) to filter by due date.
        :param due_date_end: Optional end date (YYYY-MM-DD) to filter by due date.
        :param tags: Optional comma-separated list of tags to filter by.
        :return: A tuple containing the list of tasks and the total number of tasks.
        """
        # This is a mocked implementation for demonstration purposes.
        # In a real implementation, this would query a database or external API.

        # Sample data
        all_tasks = [
            {"id": 1, "name": "Task 1", "status": "completed", "description": "Complete Task 1", "priority": "low", "due_date": "2024-01-15", "tags": ["development"]},
            {"id": 2, "name": "Task 2", "status": "pending", "description": "Do Task 2", "priority": "medium", "due_date": "2024-02-20", "tags": ["design", "urgent"]},
            {"id": 3, "name": "Task 3", "status": "completed", "description": "Task 3 is finished", "priority": "high", "due_date": "2024-01-10", "tags": ["testing"]},
            {"id": 4, "name": "Task 4", "status": "in_progress", "description": "Currently working on Task 4", "priority": "medium", "due_date": "2024-03-01", "tags": ["development", "important"]},
            {"id": 5, "name": "Task 5", "status": "pending", "description": "Task 5 needs to be started", "priority": "low", "due_date": "2024-02-15", "tags": ["documentation"]},
            {"id": 6, "name": "Urgent Task", "status": "pending", "description": "This task needs immediate attention", "priority": "high", "due_date": "2024-01-25", "tags": ["urgent", "important"]},
            {"id": 7, "name": "Another Task", "status": "in_progress", "description": "Working on another task", "priority": "medium", "due_date": "2024-02-10", "tags": ["design"]},

        ]

        filtered_tasks = all_tasks.copy()  # Create a copy to avoid modifying the original list.

        # Apply filtering if specified
        if filter_by:
            try:
                key, value = filter_by.split(':')
                filtered_tasks = [task for task in filtered_tasks if task.get(key) == value]
            except ValueError:
                # Handle badly formatted filter (should be caught in validation, but handle anyway)
                logging.warning("Invalid filter format in TaskManager")
                # In a production environment, you might raise an exception or return an error.
                pass

        # Apply priority filtering if specified
        if priority:
            filtered_tasks = [task for task in filtered_tasks if task.get('priority') == priority]

        # Apply search if specified
        if search:
            search = search.lower()
            filtered_tasks = [task for task in filtered_tasks if
                              search in task.get('name', '').lower() or search in task.get('description', '').lower()]

        # Apply date range filtering
        if due_date_start:
            filtered_tasks = [task for task in filtered_tasks if task.get('due_date') and task.get('due_date') >= due_date_start]
        if due_date_end:
            filtered_tasks = [task for task in filtered_tasks if task.get('due_date') and task.get('due_date') <= due_date_end]

        # Apply tags filtering
        if tags:
            tag_list = [tag.strip().lower() for tag in tags.split(',')]
            filtered_tasks = [task for task in filtered_tasks if task.get('tags') and any(tag in [t.lower() for t in task.get('tags')] for tag in tag_list)]


        # Apply sorting if specified
        if sort_by:
            reverse = sort_order == 'desc'
            try:
                filtered_tasks.sort(key=lambda task: task.get(sort_by), reverse=reverse)
            except Exception as e:
                logging.error(f"Error during sorting: {e}")  # Log the error
                # In a production environment, you may want to return an error message

        # Calculate pagination
        start = (page - 1) * limit
        end = start + limit
        paginated_tasks = filtered_tasks[start:end]

        # Return paginated and filtered tasks
        return paginated_tasks, len(filtered_tasks)

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single task by its ID."""
        # In a real implementation, this would query a database.
        for task in all_tasks:  # Use the all_tasks defined in get_tasks for this example
            if task["id"] == task_id:
                return task
        return None


# Example Usage (within an asyncio event loop):
async def main():
    # Example 1: List tasks with pagination
    tasks_page_1 = await list_tasks(page=1, limit=3)
    print("Tasks Page 1:", json.dumps(tasks_page_1, indent=2))

    # Example 2: List tasks with filtering
    completed_tasks = await list_tasks(filter_by='status:completed')
    print("Completed Tasks:", json.dumps(completed_tasks, indent=2))

    # Example 3: List tasks with sorting
    sorted_tasks = await list_tasks(sort_by='name', sort_order='asc')
    print("Sorted Tasks (by name, ascending):", json.dumps(sorted_tasks, indent=2))

    # Example 4: List tasks with search
    searched_tasks = await list_tasks(search='Urgent')
    print("Searched Tasks (for 'Urgent'):", json.dumps(searched_tasks, indent=2))

    # Example 5: List tasks with invalid input
    invalid_tasks = await list_tasks(page='invalid', limit=0)
    print("Invalid Tasks:", json.dumps(invalid_tasks, indent=2))

    # Example 6: List tasks filtered by priority
    high_priority_tasks = await list_tasks(priority='high')
    print("High Priority Tasks:", json.dumps(high_priority_tasks, indent=2))

    # Example 7: Clear the cache
    _clear_cache()

    # Example 8: List tasks with due date range
    date_range_tasks = await list_tasks(due_date_start='2024-01-01', due_date_end='2024-01-31')
    print("Tasks within date range:", json.dumps(date_range_tasks, indent=2))

    # Example 9: List tasks filtered by tags
    tagged_tasks = await list_tasks(tags='urgent,important')
    print("Tasks with tags:", json.dumps(tagged_tasks, indent=2))

    # Example 10: Get a specific task by ID
    task_by_id = get_task_by_id(2)
    print("Task with ID 2:", json.dumps(task_by_id, indent=2))

    #Example 11: Cache test
    tasks_page_1_again = await list_tasks(page=1, limit=3) #This should be a cache hit.
    print("Tasks Page 1 again (cache test):", json.dumps(tasks_page_1_again, indent=2))

if __name__ == "__main__":
    asyncio.run(main())