"""
Read File Tool
==============

Provides direct file reading capabilities similar to Claude Code's Read tool.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

__description__ = "Read the contents of a file from the filesystem"
__parameters__ = {
    'file_path': {'type': 'string', 'required': True, 'description': 'Path to the file to read'},
    'start_line': {'type': 'integer', 'required': False, 'description': 'Start reading from this line number'},
    'end_line': {'type': 'integer', 'required': False, 'description': 'Stop reading at this line number'},
    'encoding': {'type': 'string', 'required': False, 'description': 'File encoding (default: utf-8)'}
}
__examples__ = [
    {'file_path': 'scripts/conversational_ai_assistant.py'},
    {'file_path': 'README.md', 'start_line': 1, 'end_line': 50},
    {'file_path': 'scripts/custom_tools/some_tool.py'}
]

async def read_file(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the contents of a file, with optional line range.
    """
    file_path = kwargs.get('file_path')
    start_line = kwargs.get('start_line')
    end_line = kwargs.get('end_line')
    encoding = kwargs.get('encoding', 'utf-8')
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Handle both absolute and relative paths
    if os.path.isabs(file_path):
        full_path = Path(file_path)
    else:
        full_path = project_root / file_path
    
    try:
        # Check if file exists
        if not full_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        # Check if it's a file
        if not full_path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {file_path}"
            }
        
        # Read the file
        with open(full_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        
        # Apply line range if specified
        if start_line is not None or end_line is not None:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)
            lines = lines[start_idx:end_idx]
            line_numbers = range(start_idx + 1, start_idx + len(lines) + 1)
        else:
            line_numbers = range(1, len(lines) + 1)
        
        # Format with line numbers
        numbered_lines = []
        for line_no, line in zip(line_numbers, lines):
            numbered_lines.append(f"{line_no:4d}→ {line.rstrip()}")
        
        content = '\n'.join(numbered_lines)
        
        # Get file info
        stat = full_path.stat()
        rel_path = full_path.relative_to(project_root) if full_path.is_relative_to(project_root) else full_path
        
        return {
            "success": True,
            "file_path": str(rel_path),
            "content": content,
            "lines": len(lines),
            "total_lines": len(open(full_path, 'r', encoding=encoding).readlines()),
            "size_bytes": stat.st_size,
            "encoding": encoding
        }
        
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": f"Unable to decode file with {encoding} encoding. Try a different encoding."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {str(e)}"
        }

# Export for tool system
read_file_tool = read_file

if __name__ == "__main__":
    # Test reading a file
    async def test():
        result = await read_file(
            file_path="scripts/conversational_ai_assistant.py",
            start_line=1,
            end_line=50
        )
        print(f"Read result: {result['success']}")
        if result['success']:
            print(f"File: {result['file_path']}")
            print(f"Lines: {result['lines']} of {result['total_lines']}")
            print("Content preview:")
            print(result['content'][:500] + "...")
    
    asyncio.run(test())