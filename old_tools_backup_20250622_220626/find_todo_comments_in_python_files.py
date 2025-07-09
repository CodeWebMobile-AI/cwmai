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
import hashlib
import json
from typing import Dict, Any, List, Optional
from scripts.state_manager import StateManager  # Assuming this exists
from scripts.file_manager import FileManager    # Assuming this exists
from scripts.project_manager import ProjectManager  # Assuming this exists


__description__ = "Finds all TODO comments in Python files within the project.  Supports filtering by priority (TODO, FIXME, etc.)"
__parameters__ = {
    "priority_keywords": {
        "type": "list",
        "description": "List of priority keywords to search for (e.g., ['TODO', 'FIXME', 'HACK']).  Defaults to ['TODO'] if empty.",
        "default": ["TODO"]
    },
    "exclude_patterns": {
        "type": "list",
        "description": "List of glob patterns to exclude files from the search (e.g., ['**/test_*.py', '**/migrations/*.py']). Defaults to empty list.",
        "default": []
    },
    "max_line_length": {
        "type": "int",
        "description": "Maximum length of the comment to include in the output.  Longer comments will be truncated. Defaults to 200 characters.",
        "default": 200
    },
    "include_line_context": {
        "type": "int",
        "description": "Number of lines of code to include before and after the TODO comment. Defaults to 0 (no context).",
        "default": 0
    },
    "output_format": {
        "type": "string",
        "description": "Format of the output. Can be 'json' or 'text'. Defaults to 'json'.",
        "default": "json"
    },
    "report_file": {
        "type": "string",
        "description": "Path to a file where the report will be written, regardless of the output_format. If empty, no report file will be written.",
        "default": ""
    },
    "fail_on_error": {
      "type": "boolean",
      "description": "If True, the tool will raise an exception if any error occurs during processing.  Defaults to False.",
      "default": False
    }

}
__examples__ = [
    {
        "description": "Find TODO and FIXME comments in all Python files, excluding test files.",
        "input": {"priority_keywords": ["TODO", "FIXME"], "exclude_patterns": ["**/test_*.py"]},
        "output": {
            "todo_comments": [
                {"file": "example.py", "line_number": 10, "comment": "TODO: Implement this function."},
                {"file": "another_example.py", "line_number": 5, "comment": "FIXME: Add better error handling."},
            ],
            "cache_hit_rate": 50.0,
            "files_analyzed": 2,
            "files_skipped": 1
        }
    },
    {
        "description": "Find TODO comments with default parameters.",
        "input": {},
        "output": {
            "todo_comments": [
                {"file": "example.py", "line_number": 10, "comment": "TODO: Implement this function."},
                {"file": "another_example.py", "line_number": 5, "comment": "TODO: Add better error handling."},
            ],
            "cache_hit_rate": 75.0,
            "files_analyzed": 4,
            "files_skipped": 0
        }
    },
    {
        "description": "Find TODO comments with 2 lines of context before and after the comment.",
        "input": {"include_line_context": 2},
        "output": {
            "todo_comments": [
                {"file": "example.py", "line_number": 10, "comment": "TODO: Implement this function.", "context": ["line 8", "line 9", "line 10: TODO: Implement this function.", "line 11", "line 12"]},
                {"file": "another_example.py", "line_number": 5, "comment": "TODO: Add better error handling.", "context": ["line 3", "line 4", "line 5: TODO: Add better error handling.", "line 6", "line 7"]},
            ],
            "cache_hit_rate": 60.0,
            "files_analyzed": 5,
            "files_skipped": 0
        }
    }
]

CACHE_FILE = "todo_cache.json"
ERROR_LOG_FILE = "todo_error_log.txt" # separate error log file for this tool


async def _calculate_file_hash(file_path: str) -> Optional[str]:
    """Calculates the SHA256 hash of a file."""
    try:
        with open(file_path, "rb") as f:
            file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")  # More specific warning
        return None
    except PermissionError as e:
        print(f"Warning: Permission error accessing {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


async def _load_cache() -> Dict[str, Any]:
    """Loads the cache from the CACHE_FILE."""
    try:
        file_manager = FileManager()
        cache_content = await file_manager.read_file(CACHE_FILE)
        if cache_content:
            return json.loads(cache_content)
        else:
            return {}
    except FileNotFoundError:
        print(f"Cache file not found: {CACHE_FILE}. Starting with an empty cache.")
        return {}
    except json.JSONDecodeError:
        print("Warning: Cache file is corrupted.  Starting with an empty cache.")
        return {}
    except Exception as e:
        print(f"Error loading cache: {e}")
        return {}


async def _save_cache(cache: Dict[str, Any]) -> None:
    """Saves the cache to the CACHE_FILE."""
    try:
        file_manager = FileManager()
        await file_manager.write_to_file(CACHE_FILE, json.dumps(cache, indent=4), append=False) # Add indent for readability
    except Exception as e:
        print(f"Error saving cache: {e}")


async def _matches_exclude_pattern(file_path: str, exclude_patterns: List[str]) -> bool:
    """Checks if the file path matches any of the exclude patterns."""
    for pattern in exclude_patterns:
        if glob.has_magic(pattern):
            if glob.fnmatch.fnmatch(file_path, pattern):
                return True
        elif pattern == file_path:
            return True

    return False


async def _get_line_context(file_path: str, line_number: int, context_lines: int) -> List[str]:
    """Retrieves the specified number of lines before and after the target line."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            start = max(0, line_number - context_lines - 1) # line_number is 1-indexed, list is 0-indexed
            end = min(len(lines), line_number + context_lines)
            context = [lines[i].strip() for i in range(start, end)]
            return context
    except FileNotFoundError:
        print(f"Warning: File not found while retrieving context for {file_path}")
        return []
    except PermissionError as e:
        print(f"Warning: Permission error accessing {file_path} while retrieving context: {e}")
        return []
    except Exception as e:
        print(f"Error getting line context for {file_path} line {line_number}: {e}")
        return []


def _format_output(output_data: Dict[str, Any], output_format: str) -> str:
    """Formats the output based on the specified format."""
    if output_format == "json":
        return json.dumps(output_data, indent=4)
    elif output_format == "text":
        text_output = ""
        for comment in output_data["todo_comments"]:
            text_output += f"File: {comment['file']}, Line: {comment['line_number']}, Comment: {comment['comment']}\n"
            if "context" in comment:
                text_output += "Context:\n"
                for line in comment["context"]:
                    text_output += f"  {line}\n"
            text_output += "\n"

        text_output += f"Cache Hit Rate: {output_data['cache_hit_rate']:.2f}%\n"
        text_output += f"Files Analyzed: {output_data['files_analyzed']}\n"
        text_output += f"Files Skipped: {output_data['files_skipped']}\n"
        text_output += f"Errors Encountered: {output_data.get('errors_encountered', 0)}\n" #Show Errors
        return text_output
    else:
        return f"Error: Invalid output format: {output_format}"


async def _write_report_file(report_file: str, content: str) -> None:
    """Writes the report to the specified file."""
    if report_file:
        file_manager = FileManager()
        try:
            await file_manager.write_to_file(report_file, content, append=False)
            print(f"Report written to {report_file}")
        except Exception as e:
            print(f"Error writing report file: {e}")



async def find_todo_comments_in_python_files(**kwargs) -> Dict[str, Any]:
    """
    Searches through all Python files in the project and returns the lines containing TODO comments.
    Uses caching to improve performance.  Allows filtering by priority keywords and excluding files.
    """
    state_manager = StateManager()
    file_manager = FileManager()
    project_manager = ProjectManager()

    priority_keywords = kwargs.get("priority_keywords", ["TODO"])
    exclude_patterns = kwargs.get("exclude_patterns", [])
    max_line_length = kwargs.get("max_line_length", 200)  # Default to 200
    include_line_context = kwargs.get("include_line_context", 0) # Default to 0
    output_format = kwargs.get("output_format", "json").lower() # Default to JSON
    report_file = kwargs.get("report_file", "") # Default to empty string
    fail_on_error = kwargs.get("fail_on_error", False)

    # Input Validation
    if not isinstance(priority_keywords, list):
        return {"error": "priority_keywords must be a list."}
    if not all(isinstance(keyword, str) for keyword in priority_keywords):
        return {"error": "All elements in priority_keywords must be strings."}
    if not isinstance(exclude_patterns, list):
        return {"error": "exclude_patterns must be a list."}
    if not all(isinstance(pattern, str) for pattern in exclude_patterns):
        return {"error": "All elements in exclude_patterns must be strings."}
    if not isinstance(max_line_length, int):
        return {"error": "max_line_length must be an integer."}
    if max_line_length <= 0:
        return {"error": "max_line_length must be a positive integer."}
    if not isinstance(include_line_context, int):
        return {"error": "include_line_context must be an integer."}
    if include_line_context < 0:
        return {"error": "include_line_context must be a non-negative integer."}
    if output_format not in ["json", "text"]:
        return {"error": "output_format must be 'json' or 'text'."}
    if not isinstance(report_file, str):
        return {"error": "report_file must be a string."}
    if not isinstance(fail_on_error, bool):
        return {"error": "fail_on_error must be a boolean."}


    try:
        # Get the project root directory from the state.  Handle missing state.
        state = state_manager.load_state()
        project_root = state.get("project_root")
        if not project_root:
            return {"error": "Project root not found in state.  Please initialize project first."}

        todo_comments: List[Dict[str, Any]] = []
        python_files = glob.glob(os.path.join(project_root, "**/*.py"), recursive=True)
        files_analyzed: int = 0
        files_skipped: int = 0
        errors_encountered: int = 0

        cache = await _load_cache()

        for file_path in python_files:
            if await _matches_exclude_pattern(file_path, exclude_patterns):
                print(f"Skipping file {file_path} due to exclude pattern.")
                files_skipped += 1
                continue

            file_hash = await _calculate_file_hash(file_path)
            if not file_hash:
                files_skipped += 1
                errors_encountered += 1 # Count as an error
                print(f"Skipping file {file_path} due to hash calculation failure.")

                if fail_on_error:
                    raise Exception(f"Failed to calculate hash for {file_path}")
                continue  # Skip to the next file if hash calculation fails

            cache_key = f"{file_path}_{include_line_context}"

            if cache_key in cache and cache[cache_key].get("hash") == file_hash:
                # Use cached results
                cached_todos = cache[cache_key].get("todos", [])
                #Correct cached filepaths
                for todo in cached_todos:
                  todo["file"] = file_path
                todo_comments.extend(cached_todos)

                print(f"Using cached TODOs for {file_path}")
            else:
                # Analyze the file and update the cache
                file_todos: List[Dict[str, Any]] = []
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines() # Read all lines at once for context retrieval
                        for i, line in enumerate(lines):
                            for keyword in priority_keywords:
                                if keyword in line:
                                    comment_start = line.find(keyword)
                                    comment_text = line[comment_start:].strip()
                                    if len(comment_text) > max_line_length:
                                        comment_text = comment_text[:max_line_length] + "..."

                                    todo_entry = {
                                        "file": file_path,
                                        "line_number": i + 1,
                                        "comment": comment_text
                                    }

                                    if include_line_context > 0:
                                        context = await _get_line_context(file_path, i + 1, include_line_context)
                                        todo_entry["context"] = context

                                    file_todos.append(todo_entry)

                    todo_comments.extend(file_todos)
                    cache[cache_key] = {"hash": file_hash, "todos": file_todos}
                    print(f"Analyzed {file_path} and updated cache.")
                except FileNotFoundError:
                    print(f"Error: File not found {file_path}")
                    await file_manager.write_to_file(ERROR_LOG_FILE, f"Error: File not found {file_path}\n", append=True)
                    files_skipped += 1
                    errors_encountered += 1
                    cache.pop(cache_key, None)
                    if fail_on_error:
                       raise
                    continue # move to the next file
                except PermissionError as e:
                    print(f"Error: Permission denied reading {file_path}: {e}")
                    await file_manager.write_to_file(ERROR_LOG_FILE, f"Error: Permission denied reading {file_path}: {e}\n", append=True)
                    files_skipped += 1
                    errors_encountered += 1
                    cache.pop(cache_key, None)
                    if fail_on_error:
                       raise
                    continue
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    # Log the error with file_manager
                    await file_manager.write_to_file(ERROR_LOG_FILE, f"Error reading file {file_path}: {e}\n", append=True)
                    cache.pop(cache_key, None) # Remove from cache in case of persistent error.
                    files_skipped += 1
                    errors_encountered += 1
                    if fail_on_error:
                       raise
                    continue # move to the next file after logging the error
                files_analyzed += 1


        await _save_cache(cache)  # Save the updated cache

        # Comprehensive Output
        output_data = {
            "todo_comments": todo_comments,
            "cache_hit_rate": _calculate_cache_hit_rate(len(python_files) - files_skipped, cache),
            "files_analyzed": files_analyzed,
            "files_skipped": files_skipped,
            "errors_encountered": errors_encountered # Add Errors to output
        }

        formatted_output = _format_output(output_data, output_format)
        await _write_report_file(report_file, formatted_output)
        if "error" in formatted_output:
           return {"error": formatted_output}  # Preserve explicit errors
        if output_format == "json":
            return output_data # Ensure returns dictionary for json format
        else:
            return {"output": formatted_output} # return the full formatted output in a dictionary
    except FileNotFoundError as e:
        error_message = "Error: Project root directory not found. Please make sure the project is initialized."
        await file_manager.write_to_file(ERROR_LOG_FILE, error_message + "\n", append=True)
        if fail_on_error:
           raise e
        return {"error": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        # Log the error using the file_manager
        await file_manager.write_to_file(ERROR_LOG_FILE, error_message + "\n", append=True)
        if fail_on_error:
           raise e
        return {"error": error_message}


def _calculate_cache_hit_rate(total_files: int, cache: Dict[str, Any]) -> float:
    """Calculates the cache hit rate."""
    if total_files == 0:
        return 0.0
    cache_hits = sum(1 for cache_key in cache if cache[cache_key].get("todos") is not None) # Account for include_line_context in cache hit
    return (cache_hits / total_files) * 100 if total_files > 0 else 0.0