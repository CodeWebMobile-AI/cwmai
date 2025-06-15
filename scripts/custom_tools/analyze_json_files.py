"""
AI-Generated Tool: analyze_json_files
Description: Analyze JSON files and extract statistics
Generated: 2025-06-15T11:44:03.578009+00:00
Requirements: 
        1. Find all JSON files in a directory
        2. Load and parse each file
        3. Count keys, values, and nested structures
        4. Return statistics using pandas DataFrame
        
"""

from typing import Any
from typing import Dict
from typing import Dict, Any, List
from typing import List

import asyncio
import json
import os
import pandas as pd


"""
Module: analyze_json_files
Description: Analyze JSON files in a directory and extract statistics using pandas DataFrame.
"""


__description__ = "Analyze JSON files and extract statistics."

__parameters__ = {
    "directory_path": {
        "type": "string",
        "description": "Path to the directory containing JSON files.",
        "required": True,
    }
}

__examples__ = [
    {
        "description": "Analyze JSON files in the 'data' directory.",
        "arguments": {"directory_path": "data"},
    }
]


async def analyze_json_files(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes JSON files in a directory and extracts statistics.

    Args:
        **kwargs: Keyword arguments containing tool parameters.  Must include 'directory_path'.

    Returns:
        A dictionary containing statistics about the JSON files.
        Returns:
        A dictionary with statistics on the JSON files, including counts of keys, values, and nested structures.  Returns an error message if the directory does not exist, or if no JSON files are found.
    """

    directory_path = kwargs.get("directory_path")

    if not directory_path:
        return {"error": "Directory path is required."}

    if not os.path.isdir(directory_path):
        return {"error": f"Directory not found: {directory_path}"}

    json_files: List[str] = [
        f for f in os.listdir(directory_path) if f.endswith(".json")
    ]

    if not json_files:
        return {"error": f"No JSON files found in: {directory_path}"}

    all_data: List[Dict[str, Any]] = []

    for filename in json_files:
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                all_data.append(data)
        except json.JSONDecodeError as e:
            return {"error": f"Error decoding JSON in {filename}: {e}"}
        except Exception as e:
            return {"error": f"Error reading file {filename}: {e}"}

    total_keys = 0
    total_values = 0
    total_nested = 0

    for data in all_data:
        total_keys += len(data.keys())
        total_values += sum(1 for _ in data.values())

        def count_nested(obj: Any) -> int:
            count = 0
            if isinstance(obj, dict):
                count += 1
                for value in obj.values():
                    count += count_nested(value)
            elif isinstance(obj, list):
                count += 1
                for item in obj:
                    count += count_nested(item)
            return count

        total_nested += count_nested(data)

    df = pd.DataFrame(
        {
            "Metric": ["Total Keys", "Total Values", "Total Nested Structures"],
            "Count": [total_keys, total_values, total_nested],
        }
    )

    result_dict = df.to_dict(orient="list")
    return result_dict
