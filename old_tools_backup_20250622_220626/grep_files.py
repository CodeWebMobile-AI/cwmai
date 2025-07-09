"""
Grep Files Tool
===============

Provides grep-like functionality to search through local files for patterns,
similar to Claude Code's grep capabilities.
"""

import os
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import fnmatch

__description__ = "Search through files for patterns using grep-like functionality"
__parameters__ = {
    'pattern': {'type': 'string', 'required': True, 'description': 'Pattern to search for (supports regex)'},
    'file_pattern': {'type': 'string', 'required': False, 'description': 'File pattern to search in (e.g., *.py, *.md)'},
    'directory': {'type': 'string', 'required': False, 'description': 'Directory to search in (default: entire project)'},
    'case_sensitive': {'type': 'boolean', 'required': False, 'description': 'Case sensitive search (default: False)'},
    'max_results': {'type': 'integer', 'required': False, 'description': 'Maximum results to return (default: 50)'}
}
__examples__ = [
    {'pattern': 'architecture', 'file_pattern': '*.py'},
    {'pattern': 'class.*Generator', 'directory': 'scripts/'},
    {'pattern': 'TODO|FIXME', 'case_sensitive': True}
]

async def grep_files(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search through files for patterns, similar to grep command.
    """
    pattern = kwargs.get('pattern')
    file_pattern = kwargs.get('file_pattern', '*')
    directory = kwargs.get('directory', '.')
    case_sensitive = kwargs.get('case_sensitive', False)
    max_results = kwargs.get('max_results', 50)
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    search_dir = project_root / directory
    
    # Compile regex pattern
    regex_flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled_pattern = re.compile(pattern, regex_flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}"
        }
    
    results = []
    files_searched = 0
    matches_found = 0
    
    # Directories to skip
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'logs', '.idea'}
    
    try:
        for root, dirs, files in os.walk(search_dir):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if fnmatch.fnmatch(file, file_pattern):
                    file_path = Path(root) / file
                    files_searched += 1
                    
                    try:
                        # Skip binary files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Search for pattern in file
                        for line_no, line in enumerate(content.splitlines(), 1):
                            if compiled_pattern.search(line):
                                matches_found += 1
                                
                                # Get relative path from project root
                                rel_path = file_path.relative_to(project_root)
                                
                                results.append({
                                    'file': str(rel_path),
                                    'line': line_no,
                                    'content': line.strip(),
                                    'match': compiled_pattern.search(line).group(0)
                                })
                                
                                if len(results) >= max_results:
                                    break
                    
                    except (UnicodeDecodeError, PermissionError):
                        # Skip binary or inaccessible files
                        continue
                
                if len(results) >= max_results:
                    break
            
            if len(results) >= max_results:
                break
        
        # Format results for display
        formatted_results = []
        for r in results:
            formatted_results.append(f"{r['file']}:{r['line']}: {r['content']}")
        
        return {
            "success": True,
            "results": results,
            "formatted_results": formatted_results,
            "files_searched": files_searched,
            "matches_found": matches_found,
            "truncated": matches_found > max_results,
            "summary": f"Found {min(matches_found, max_results)} matches in {files_searched} files"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Search error: {str(e)}"
        }

# Export for tool system
grep_files_tool = grep_files

if __name__ == "__main__":
    # Test the search
    async def test():
        result = await grep_files(
            pattern="architecture",
            file_pattern="*.py",
            max_results=10
        )
        print(f"Search results: {result}")
    
    asyncio.run(test())