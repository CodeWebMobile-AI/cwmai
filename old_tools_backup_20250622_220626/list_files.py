"""
List Files Tool
===============

Lists files and directories in a given path, similar to Claude Code's ls functionality.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

__description__ = "List files and directories in a given path with details"
__parameters__ = {
    'path': {'type': 'string', 'required': False, 'description': 'Path to list (default: current directory)'},
    'pattern': {'type': 'string', 'required': False, 'description': 'Filter files by pattern (e.g., *.py)'},
    'show_hidden': {'type': 'boolean', 'required': False, 'description': 'Show hidden files (default: False)'},
    'recursive': {'type': 'boolean', 'required': False, 'description': 'List files recursively (default: False)'},
    'sort_by': {'type': 'string', 'required': False, 'description': 'Sort by: name, size, date (default: name)'}
}
__examples__ = [
    {'path': 'scripts/custom_tools/', 'pattern': '*.py'},
    {'path': '.', 'show_hidden': True},
    {'path': 'scripts/', 'recursive': True, 'pattern': '*.py'}
]

async def list_files(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    List files and directories with various options.
    """
    path = kwargs.get('path', '.')
    pattern = kwargs.get('pattern')
    show_hidden = kwargs.get('show_hidden', False)
    recursive = kwargs.get('recursive', False)
    sort_by = kwargs.get('sort_by', 'name')
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Handle both absolute and relative paths
    if os.path.isabs(path):
        target_path = Path(path)
    else:
        target_path = project_root / path
    
    try:
        # Check if path exists
        if not target_path.exists():
            return {
                "success": False,
                "error": f"Path not found: {path}"
            }
        
        items = []
        
        if recursive:
            # Recursive listing
            for root, dirs, files in os.walk(target_path):
                # Filter hidden directories
                if not show_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                root_path = Path(root)
                
                # Add directories
                for dir_name in dirs:
                    if not show_hidden and dir_name.startswith('.'):
                        continue
                        
                    dir_path = root_path / dir_name
                    rel_path = dir_path.relative_to(target_path)
                    
                    items.append({
                        'name': str(rel_path),
                        'type': 'directory',
                        'size': 0,
                        'modified': datetime.fromtimestamp(dir_path.stat().st_mtime).isoformat()
                    })
                
                # Add files
                for file_name in files:
                    if not show_hidden and file_name.startswith('.'):
                        continue
                    
                    if pattern:
                        from fnmatch import fnmatch
                        if not fnmatch(file_name, pattern):
                            continue
                    
                    file_path = root_path / file_name
                    rel_path = file_path.relative_to(target_path)
                    stat = file_path.stat()
                    
                    items.append({
                        'name': str(rel_path),
                        'type': 'file',
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        else:
            # Non-recursive listing
            for item in target_path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                if pattern and item.is_file():
                    from fnmatch import fnmatch
                    if not fnmatch(item.name, pattern):
                        continue
                
                stat = item.stat()
                items.append({
                    'name': item.name,
                    'type': 'directory' if item.is_dir() else 'file',
                    'size': stat.st_size if item.is_file() else 0,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort items
        if sort_by == 'size':
            items.sort(key=lambda x: x['size'], reverse=True)
        elif sort_by == 'date':
            items.sort(key=lambda x: x['modified'], reverse=True)
        else:  # name
            items.sort(key=lambda x: (x['type'] == 'file', x['name'].lower()))
        
        # Format for display
        formatted_items = []
        for item in items:
            if item['type'] == 'directory':
                formatted_items.append(f"[DIR]  {item['name']}/")
            else:
                size_str = _format_size(item['size'])
                formatted_items.append(f"[FILE] {item['name']} ({size_str})")
        
        rel_path = target_path.relative_to(project_root) if target_path.is_relative_to(project_root) else target_path
        
        return {
            "success": True,
            "path": str(rel_path),
            "items": items,
            "formatted_items": formatted_items,
            "total_files": sum(1 for i in items if i['type'] == 'file'),
            "total_directories": sum(1 for i in items if i['type'] == 'directory'),
            "total_size": sum(i['size'] for i in items if i['type'] == 'file')
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error listing files: {str(e)}"
        }

def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

# Export for tool system
list_files_tool = list_files

if __name__ == "__main__":
    # Test listing files
    async def test():
        # Test basic listing
        result = await list_files(path="scripts/custom_tools/", pattern="*.py")
        print(f"Result: {result['success']}")
        if result['success']:
            print(f"Path: {result['path']}")
            print(f"Files: {result['total_files']}, Directories: {result['total_directories']}")
            print("\nItems:")
            for item in result['formatted_items'][:10]:
                print(f"  {item}")
    
    asyncio.run(test())