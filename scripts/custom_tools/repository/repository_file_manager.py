"""
AI-Generated Tool: repository_file_manager
Description: Manage files within repositories - create, read, update, delete, search
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import re
from datetime import datetime

__description__ = "Manage files within repositories - CRUD operations, search, bulk operations"
__parameters__ = {
    "operation": {
        "type": "string",
        "required": True,
        "enum": ["create", "read", "update", "delete", "list", "search", "replace", "copy", "move"],
        "description": "File operation to perform"
    },
    "repo_name": {
        "type": "string",
        "required": True,
        "description": "Repository name"
    },
    "file_path": {
        "type": "string",
        "required": False,
        "description": "File path relative to repository root"
    },
    "content": {
        "type": "string",
        "required": False,
        "description": "File content for create/update operations"
    },
    "search_pattern": {
        "type": "string",
        "required": False,
        "description": "Pattern to search for (regex supported)"
    },
    "replace_with": {
        "type": "string",
        "required": False,
        "description": "Replacement text for search/replace"
    },
    "destination": {
        "type": "string",
        "required": False,
        "description": "Destination path for copy/move operations"
    },
    "recursive": {
        "type": "boolean",
        "required": False,
        "default": False,
        "description": "Apply operation recursively"
    }
}
__examples__ = [
    "repository_file_manager(operation='create', repo_name='my-app', file_path='src/config.js', content='export default {};')",
    "repository_file_manager(operation='read', repo_name='my-app', file_path='README.md')",
    "repository_file_manager(operation='list', repo_name='my-app', recursive=True)",
    "repository_file_manager(operation='search', repo_name='my-app', search_pattern='TODO', recursive=True)",
    "repository_file_manager(operation='replace', repo_name='my-app', search_pattern='old_api', replace_with='new_api')"
]

# Base path
REPOSITORIES_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "repositories" / "CodeWebMobile-AI"

async def repository_file_manager(**kwargs) -> Dict[str, Any]:
    """
    Manage files within repositories with comprehensive operations.
    """
    operation = kwargs.get('operation')
    repo_name = kwargs.get('repo_name')
    
    if not operation:
        return {
            "status": "error",
            "message": "operation parameter is required"
        }
    
    if not repo_name:
        return {
            "status": "error",
            "message": "repo_name parameter is required"
        }
    
    # Check if repository exists
    repo_path = REPOSITORIES_PATH / repo_name
    if not repo_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' not found"
        }
    
    # Route to appropriate operation
    operations = {
        'create': create_file,
        'read': read_file,
        'update': update_file,
        'delete': delete_file,
        'list': list_files,
        'search': search_files,
        'replace': replace_in_files,
        'copy': copy_file,
        'move': move_file
    }
    
    if operation not in operations:
        return {
            "status": "error",
            "message": f"Unknown operation: {operation}",
            "available_operations": list(operations.keys())
        }
    
    try:
        kwargs['repo_path'] = repo_path
        return await operations[operation](kwargs)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error performing {operation}: {str(e)}"
        }

async def create_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new file in the repository."""
    repo_path = params['repo_path']
    file_path = params.get('file_path')
    content = params.get('content', '')
    
    if not file_path:
        return {
            "status": "error",
            "message": "file_path is required for create operation"
        }
    
    full_path = repo_path / file_path
    
    # Check if file already exists
    if full_path.exists():
        return {
            "status": "error",
            "message": f"File '{file_path}' already exists",
            "suggestion": "Use 'update' operation to modify existing file"
        }
    
    # Create parent directories if needed
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write content
    full_path.write_text(content)
    
    return {
        "status": "success",
        "message": f"File created successfully",
        "file": {
            "path": file_path,
            "full_path": str(full_path),
            "size": len(content),
            "created_at": datetime.now().isoformat()
        }
    }

async def read_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Read a file from the repository."""
    repo_path = params['repo_path']
    file_path = params.get('file_path')
    
    if not file_path:
        return {
            "status": "error",
            "message": "file_path is required for read operation"
        }
    
    full_path = repo_path / file_path
    
    if not full_path.exists():
        return {
            "status": "error",
            "message": f"File '{file_path}' not found"
        }
    
    if not full_path.is_file():
        return {
            "status": "error",
            "message": f"'{file_path}' is not a file"
        }
    
    try:
        content = full_path.read_text()
        
        # Get file stats
        stats = full_path.stat()
        
        return {
            "status": "success",
            "file": {
                "path": file_path,
                "content": content,
                "size": stats.st_size,
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "lines": len(content.splitlines())
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading file: {str(e)}"
        }

async def update_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing file."""
    repo_path = params['repo_path']
    file_path = params.get('file_path')
    content = params.get('content')
    
    if not file_path:
        return {
            "status": "error",
            "message": "file_path is required for update operation"
        }
    
    if content is None:
        return {
            "status": "error",
            "message": "content is required for update operation"
        }
    
    full_path = repo_path / file_path
    
    if not full_path.exists():
        return {
            "status": "error",
            "message": f"File '{file_path}' not found",
            "suggestion": "Use 'create' operation to create new file"
        }
    
    # Backup original content
    original_content = full_path.read_text()
    
    # Write new content
    full_path.write_text(content)
    
    return {
        "status": "success",
        "message": f"File updated successfully",
        "file": {
            "path": file_path,
            "previous_size": len(original_content),
            "new_size": len(content),
            "updated_at": datetime.now().isoformat()
        }
    }

async def delete_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a file from the repository."""
    repo_path = params['repo_path']
    file_path = params.get('file_path')
    
    if not file_path:
        return {
            "status": "error",
            "message": "file_path is required for delete operation"
        }
    
    full_path = repo_path / file_path
    
    if not full_path.exists():
        return {
            "status": "error",
            "message": f"File '{file_path}' not found"
        }
    
    if full_path.is_dir():
        return {
            "status": "error",
            "message": f"'{file_path}' is a directory, not a file"
        }
    
    # Delete the file
    full_path.unlink()
    
    return {
        "status": "success",
        "message": f"File deleted successfully",
        "deleted_file": file_path
    }

async def list_files(params: Dict[str, Any]) -> Dict[str, Any]:
    """List files in the repository."""
    repo_path = params['repo_path']
    recursive = params.get('recursive', False)
    
    files = []
    
    if recursive:
        # Walk through all directories
        for root, dirs, filenames in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in filenames:
                if not filename.startswith('.'):
                    file_path = Path(root) / filename
                    relative_path = file_path.relative_to(repo_path)
                    
                    files.append({
                        "path": str(relative_path),
                        "name": filename,
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
    else:
        # List only root directory
        for item in repo_path.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                files.append({
                    "path": item.name,
                    "name": item.name,
                    "size": item.stat().st_size,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
    
    # Sort by path
    files.sort(key=lambda x: x['path'])
    
    # Group by directory if recursive
    if recursive:
        by_directory = {}
        for file in files:
            dir_path = str(Path(file['path']).parent)
            if dir_path not in by_directory:
                by_directory[dir_path] = []
            by_directory[dir_path].append(file)
        
        return {
            "status": "success",
            "total_files": len(files),
            "recursive": recursive,
            "files": files[:50],  # Limit to first 50 for readability
            "by_directory": by_directory if len(files) <= 50 else "Too many files to group"
        }
    
    return {
        "status": "success",
        "total_files": len(files),
        "recursive": recursive,
        "files": files
    }

async def search_files(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search for content in files."""
    repo_path = params['repo_path']
    search_pattern = params.get('search_pattern')
    recursive = params.get('recursive', True)
    
    if not search_pattern:
        return {
            "status": "error",
            "message": "search_pattern is required for search operation"
        }
    
    matches = []
    files_searched = 0
    
    try:
        # Compile regex pattern
        pattern = re.compile(search_pattern, re.IGNORECASE)
    except re.error as e:
        return {
            "status": "error",
            "message": f"Invalid regex pattern: {str(e)}"
        }
    
    # Define text file extensions
    text_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.php', '.java', '.c', '.cpp',
                      '.h', '.css', '.scss', '.html', '.xml', '.json', '.md', '.txt',
                      '.yml', '.yaml', '.env', '.gitignore', '.sql', '.sh', '.bat'}
    
    if recursive:
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                
                file_path = Path(root) / filename
                
                # Only search text files
                if file_path.suffix.lower() not in text_extensions:
                    continue
                
                files_searched += 1
                relative_path = file_path.relative_to(repo_path)
                
                try:
                    content = file_path.read_text()
                    
                    # Find all matches with line numbers
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if pattern.search(line):
                            matches.append({
                                "file": str(relative_path),
                                "line": line_num,
                                "content": line.strip(),
                                "match": pattern.search(line).group()
                            })
                except:
                    # Skip files that can't be read as text
                    pass
    else:
        # Search only in root directory
        for item in repo_path.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                if item.suffix.lower() not in text_extensions:
                    continue
                
                files_searched += 1
                
                try:
                    content = item.read_text()
                    
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if pattern.search(line):
                            matches.append({
                                "file": item.name,
                                "line": line_num,
                                "content": line.strip(),
                                "match": pattern.search(line).group()
                            })
                except:
                    pass
    
    return {
        "status": "success",
        "search_pattern": search_pattern,
        "files_searched": files_searched,
        "total_matches": len(matches),
        "matches": matches[:100],  # Limit to first 100 matches
        "truncated": len(matches) > 100
    }

async def replace_in_files(params: Dict[str, Any]) -> Dict[str, Any]:
    """Replace text in files."""
    repo_path = params['repo_path']
    search_pattern = params.get('search_pattern')
    replace_with = params.get('replace_with', '')
    recursive = params.get('recursive', False)
    
    if not search_pattern:
        return {
            "status": "error",
            "message": "search_pattern is required for replace operation"
        }
    
    files_modified = []
    total_replacements = 0
    
    try:
        pattern = re.compile(search_pattern)
    except re.error as e:
        return {
            "status": "error",
            "message": f"Invalid regex pattern: {str(e)}"
        }
    
    text_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.php', '.java', '.c', '.cpp',
                      '.h', '.css', '.scss', '.html', '.xml', '.json', '.md', '.txt',
                      '.yml', '.yaml', '.env', '.gitignore', '.sql', '.sh', '.bat'}
    
    files_to_process = []
    
    if recursive:
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in filenames:
                if not filename.startswith('.'):
                    file_path = Path(root) / filename
                    if file_path.suffix.lower() in text_extensions:
                        files_to_process.append(file_path)
    else:
        for item in repo_path.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                if item.suffix.lower() in text_extensions:
                    files_to_process.append(item)
    
    for file_path in files_to_process:
        try:
            content = file_path.read_text()
            new_content, replacements = pattern.subn(replace_with, content)
            
            if replacements > 0:
                file_path.write_text(new_content)
                relative_path = file_path.relative_to(repo_path)
                
                files_modified.append({
                    "file": str(relative_path),
                    "replacements": replacements
                })
                total_replacements += replacements
        except:
            # Skip files that can't be processed
            pass
    
    return {
        "status": "success",
        "search_pattern": search_pattern,
        "replace_with": replace_with,
        "files_modified": len(files_modified),
        "total_replacements": total_replacements,
        "modified_files": files_modified
    }

async def copy_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a file within the repository."""
    repo_path = params['repo_path']
    file_path = params.get('file_path')
    destination = params.get('destination')
    
    if not file_path:
        return {
            "status": "error",
            "message": "file_path is required for copy operation"
        }
    
    if not destination:
        return {
            "status": "error",
            "message": "destination is required for copy operation"
        }
    
    source_path = repo_path / file_path
    dest_path = repo_path / destination
    
    if not source_path.exists():
        return {
            "status": "error",
            "message": f"Source file '{file_path}' not found"
        }
    
    if dest_path.exists():
        return {
            "status": "error",
            "message": f"Destination '{destination}' already exists"
        }
    
    # Create parent directories if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy file
    shutil.copy2(source_path, dest_path)
    
    return {
        "status": "success",
        "message": "File copied successfully",
        "source": file_path,
        "destination": destination
    }

async def move_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Move a file within the repository."""
    repo_path = params['repo_path']
    file_path = params.get('file_path')
    destination = params.get('destination')
    
    if not file_path:
        return {
            "status": "error",
            "message": "file_path is required for move operation"
        }
    
    if not destination:
        return {
            "status": "error",
            "message": "destination is required for move operation"
        }
    
    source_path = repo_path / file_path
    dest_path = repo_path / destination
    
    if not source_path.exists():
        return {
            "status": "error",
            "message": f"Source file '{file_path}' not found"
        }
    
    if dest_path.exists():
        return {
            "status": "error",
            "message": f"Destination '{destination}' already exists"
        }
    
    # Create parent directories if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Move file
    source_path.rename(dest_path)
    
    return {
        "status": "success",
        "message": "File moved successfully",
        "source": file_path,
        "destination": destination
    }

# Test function
async def main():
    # Test listing files
    result = await repository_file_manager(
        operation='list',
        repo_name='brand-guardian-ai',
        recursive=True
    )
    print(f"Files in brand-guardian-ai: {result.get('total_files', 0)}")
    
    # Test searching
    search_result = await repository_file_manager(
        operation='search',
        repo_name='brand-guardian-ai',
        search_pattern='TODO',
        recursive=True
    )
    print(f"TODO matches: {search_result.get('total_matches', 0)}")

if __name__ == "__main__":
    asyncio.run(main())