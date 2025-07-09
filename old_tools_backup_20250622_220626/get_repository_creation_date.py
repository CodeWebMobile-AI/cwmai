import asyncio
import json
import subprocess
from typing import Any, Dict

async def get_repository_creation_date(**kwargs: Any) -> Dict[str, Any]:
    """
    Retrieves the creation date of a specified repository on GitHub.
    """
    __description__ = "Retrieves the creation date of a specified repository on GitHub."
    __parameters__ = {
        "repository_name": {
            "type": "string",
            "description": "The name of the repository (e.g., 'owner/repo').",
            "required": True,
        }
    }
    __examples__ = [
        {
            "description": "Get the creation date of the 'octocat/Spoon-Knife' repository.",
            "code": "get_repository_creation_date(repository_name='octocat/Spoon-Knife')",
        }
    ]

    repository_name = kwargs.get("repository_name")
    if not repository_name:
        return {"error": "Repository name is required."}

    try:
        command = [
            "gh",
            "repo",
            "view",
            repository_name,
            "--json",
            "createdAt",
            "--jq",
            ".createdAt",
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_message = stderr.decode("utf-8").strip()
            return {"error": f"Failed to retrieve creation date: {error_message}"}

        creation_date = stdout.decode("utf-8").strip().replace('"', '')
        return {"creation_date": creation_date}

    except FileNotFoundError:
        return {"error": "GitHub CLI (gh) not found. Please ensure it is installed and in your PATH."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}