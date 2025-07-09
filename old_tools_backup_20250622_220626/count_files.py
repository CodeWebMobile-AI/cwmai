import os
import glob
import asyncio
from typing import Dict, Any, List

__description__ = "Counts the number of files in a specified directory, optionally filtering by file pattern."

__parameters__ = {
    "directory": {
        "type": "string",
        "description": "The directory to count files in.",
        "required": True,
    },
    "pattern": {
        "type": "string",
        "description": "A file pattern to filter by (e.g., '*.txt'). If not provided, all files are counted.",
        "required": False,
    },
}

__examples__ = [
    {
        "description": "Count all files in the current directory.",
        "code": 'count_files(directory=".")',
    },
    {
        "description": "Count all text files in the 'data' directory.",
        "code": 'count_files(directory="data", pattern="*.txt")',
    },
]


async def count_files(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Counts the number of files in a specified directory, optionally filtering by file pattern.

    Args:
        **kwargs: Keyword arguments containing 'directory' (required) and optionally 'pattern'.

    Returns:
        A dictionary containing the file count or an error message.
    """
    directory = kwargs.get("directory")
    pattern = kwargs.get("pattern")

    if not directory:
        return {"error": "Directory must be specified."}

    try:
        if not os.path.isdir(directory):
            return {"error": f"Directory '{directory}' does not exist."}

        if pattern:
            files = glob.glob(os.path.join(directory, pattern))
        else:
            files = [
                f
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]

        file_count = len(files)
        return {"file_count": file_count}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}