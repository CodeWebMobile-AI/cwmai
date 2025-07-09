from typing import Dict, Any, List
import os
import re
import asyncio
from scripts.logger import logger
from scripts.mcp_client import MCPClient
from functools import lru_cache

__description__ = "Checks if a repository has an ARCHITECTURE.md file, optionally retrieves its contents, and supports multiple architecture file names."
__parameters__ = {
    "repo_name": "The name of the repository (e.g., my-repo or org/my-repo).",
    "include_content": "Optional. If True, the content of the ARCHITECTURE.md file is included in the output. Defaults to False.",
    "architecture_file_names": "Optional. A list of possible architecture file names (e.g., ['ARCHITECTURE.md', 'Architecture.md', 'docs/architecture.md']). Defaults to ['ARCHITECTURE.md'].",
    "max_content_length": "Optional. If include_content is True, limits the content returned to this many characters.  Defaults to 10000.",
    "check_file_permissions": "Optional. If True, check file permissions before reading. Defaults to True.",
    "enable_content_hashing": "Optional. If True, a hash of the file content is returned. Defaults to False." # New Feature: content hashing
}
__examples__ = [
    {"repo_name": "my-repo"},
    {"repo_name": "my-repo", "include_content": True},
    {"repo_name": "my-repo", "architecture_file_names": ["Architecture.md", "docs/architecture.md"]},
    {"context": {"user_input": "Check for ARCHITECTURE.md in my-repo"}},
    {"context": {"user_input": "What is the architecture of my-repo?", "include_content": True}},
    {"repo_name": "my-repo", "include_content": True, "max_content_length": 500},
    {"repo_name": "my-repo", "check_file_permissions": False},
    {"repo_name": "my-repo", "include_content": True, "enable_content_hashing": True} #Example with hashing
]



@lru_cache(maxsize=32)  # Cache results for up to 32 different repos
async def check_for_architecture_file(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks if a repository has an architecture file (ARCHITECTURE.md by default),
    optionally retrieves its content, and handles various error conditions.
    """
    repo_name = None  # Initialize repo_name outside the try block for wider scope
    try:
        repo_name = kwargs.get('repo_name') or kwargs.get('repo')

        if not repo_name:
            context = kwargs.get('context', {})
            user_input = context.get('user_input', '')

            repo_patterns = [
                r'(?:in|for|about|within)\s+([a-zA-Z0-9-]+(?:-ai)?)',
                r'([a-zA-Z0-9-]+(?:-ai))\s+(?:repository|repo|project)',
                r'([a-zA-Z0-9-]+(?:-ai))'  # Any word ending in -ai
            ]

            for pattern in repo_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    repo_name = match.group(1)
                    break

        if not repo_name:
            return {"error": "Could not determine repository name from context or parameters.", "repository": None, "architecture_file_exists": False, "details": "Please provide the 'repo_name' parameter or include it in the user input."}

        include_content = kwargs.get('include_content', False)
        architecture_file_names = kwargs.get('architecture_file_names', ['ARCHITECTURE.md'])
        max_content_length = kwargs.get('max_content_length', 10000) # Default max content length
        check_file_permissions = kwargs.get('check_file_permissions', True) # Default to checking permissions
        enable_content_hashing = kwargs.get('enable_content_hashing', False)

        # Input Validation
        if not isinstance(include_content, bool):
            logger.warning(f"Invalid type for 'include_content'.  Expected bool, got {type(include_content)}.  Using default value of False.")
            include_content = False

        if not isinstance(architecture_file_names, list):
            logger.warning(f"Invalid type for 'architecture_file_names'.  Expected list, got {type(architecture_file_names)}. Using default value of ['ARCHITECTURE.md'].")
            architecture_file_names = ['ARCHITECTURE.md']
        else:
            validated_filenames = []
            for filename in architecture_file_names:
                if not isinstance(filename, str):
                    logger.warning(f"Invalid type in 'architecture_file_names' list.  Expected str, got {type(filename)}. Skipping this filename.")
                elif not filename: # Check for empty string filenames
                    logger.warning(f"Empty filename found in 'architecture_file_names' list. Skipping.")
                else:
                    validated_filenames.append(filename) # Only add valid filenames

            if not validated_filenames:
                logger.warning("No valid filenames provided in 'architecture_file_names'. Using default value of ['ARCHITECTURE.md'].")
                architecture_file_names = ['ARCHITECTURE.md']
            else:
                architecture_file_names = validated_filenames

        if not isinstance(max_content_length, int):
            logger.warning(f"Invalid type for 'max_content_length'. Expected int, got {type(max_content_length)}. Using default value of 10000.")
            max_content_length = 10000
        elif max_content_length <= 0:
            logger.warning(f"'max_content_length' must be a positive integer. Using default value of 10000.")
            max_content_length = 10000

        if not isinstance(check_file_permissions, bool): #Validation for check_file_permissions
            logger.warning(f"Invalid type for 'check_file_permissions'. Expected bool, got {type(check_file_permissions)}. Using default value of True.")
            check_file_permissions = True

        if not isinstance(enable_content_hashing, bool):
            logger.warning(f"Invalid type for 'enable_content_hashing'. Expected bool, got {type(enable_content_hashing)}. Using default value of False.")
            enable_content_hashing = False


        if '/' not in repo_name:
            repo_name = f"CodeWebMobile-AI/{repo_name}"

        # Use MCP client to check for files
        mcp_client = MCPClient()
        
        architecture_file_path = None
        file_exists = False
        architecture_file_content = None
        content_hash = None

        # Check each potential architecture file
        for filename in architecture_file_names:
            try:
                # Use MCP to check if file exists in the repository
                file_info = await mcp_client.call_tool(
                    server_name="github",
                    tool_name="get_file_contents", 
                    arguments={
                        "owner": repo_name.split('/')[0],
                        "repo": repo_name.split('/')[1],
                        "path": filename
                    }
                )
                
                if file_info and not isinstance(file_info, dict) or (isinstance(file_info, dict) and 'error' not in file_info):
                    architecture_file_path = filename
                    file_exists = True
                    
                    if include_content:
                        # Extract content from the response
                        if isinstance(file_info, dict) and 'content' in file_info:
                            import base64
                            # GitHub API returns base64 encoded content
                            architecture_file_content = base64.b64decode(file_info['content']).decode('utf-8')
                        elif isinstance(file_info, str):
                            architecture_file_content = file_info
                        else:
                            architecture_file_content = str(file_info)
                            
                        if len(architecture_file_content) > max_content_length:
                            architecture_file_content = architecture_file_content[:max_content_length] + " ... (truncated due to max_content_length)"
                            
                        if enable_content_hashing:
                            import hashlib
                            content_hash = hashlib.sha256(architecture_file_content.encode()).hexdigest()
                    
                    break  # Found the file, stop searching
                    
            except Exception as e:
                # File doesn't exist or error accessing it, try next filename
                logger.debug(f"Could not access {filename} in {repo_name}: {e}")
                continue


        result = {
            "repository": repo_name,
            "architecture_file_exists": file_exists,
            "architecture_file_path": architecture_file_path,
        }

        if include_content:
            result["architecture_file_content"] = architecture_file_content
            result["architecture_file_content_truncated"] = len(architecture_file_content) > max_content_length

            if enable_content_hashing:
                result["architecture_file_content_hash"] = content_hash


        result["cache_hit"] = check_for_architecture_file.cache_info().hits > 0  # Add cache hit information
        result["number_of_filenames_checked"] = len(architecture_file_names) #Enhanced output
        result["first_filename_checked"] = architecture_file_names[0] if architecture_file_names else None # Enhanced output
        return result

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}", "repository": repo_name if repo_name else None, "architecture_file_exists": False, "details": str(e), "unexpected_error": True} # Added error type for unexpected errors