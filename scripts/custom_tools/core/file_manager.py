"""
File Manager

Unified tool for all file operations including search, read, write, grep,
and content management. Consolidates 12 file-related tools into one.

Capabilities:
- List, read, write files
- Search files by pattern or content
- Grep and find operations
- Smart content analysis
- API file operations
- File type analysis
"""

import os
import re
import json
import shutil
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timezone
from collections import defaultdict
import hashlib

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager


class FileManager:
    """Comprehensive file management system."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.state_manager = StateManager()
        self.project_root = Path(__file__).parent.parent.parent
        
        # Common code file extensions
        self.code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r'
        }
        
        # Document extensions
        self.doc_extensions = {
            '.md', '.txt', '.rst', '.doc', '.docx', '.pdf', '.html', '.xml'
        }
        
    async def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute file operation.
        
        Operations:
        - list: List files in directory
        - read: Read file contents
        - write: Write content to file
        - search: Search files by pattern
        - grep: Search content in files
        - find: Find files by criteria
        - analyze: Analyze file types and sizes
        - smart_search: AI-powered content search
        """
        operations = {
            'list': self.list_files,
            'read': self.read_file,
            'write': self.write_file,
            'search': self.search_files,
            'grep': self.grep_files,
            'find': self.find_files,
            'analyze': self.analyze_files,
            'smart_search': self.smart_search,
            'copy': self.copy_file,
            'move': self.move_file,
            'delete': self.delete_file,
            'create_directory': self.create_directory,
            'get_info': self.get_file_info,
            'compare': self.compare_files
        }
        
        handler = operations.get(operation)
        if not handler:
            # Fuzzy match
            for op, func in operations.items():
                if op in operation or operation in op:
                    handler = func
                    break
                    
        if not handler:
            # Use AI to understand intent
            return await self._intelligent_file_operation(operation, **kwargs)
            
        return await handler(**kwargs)
        
    async def list_files(self, path: str = '.', pattern: str = '*', 
                        recursive: bool = False, **kwargs) -> Dict[str, Any]:
        """List files in directory."""
        list_path = Path(path) if path != '.' else self.project_root
        
        if not list_path.exists():
            return {"error": f"Path {path} does not exist"}
            
        if not list_path.is_dir():
            return {"error": f"{path} is not a directory"}
            
        files = []
        dirs = []
        
        if recursive:
            for item in list_path.rglob(pattern):
                if not any(part.startswith('.') for part in item.parts):
                    if item.is_file():
                        files.append(self._get_file_details(item, list_path))
                    elif item.is_dir():
                        dirs.append(str(item.relative_to(list_path)))
        else:
            for item in list_path.glob(pattern):
                if not item.name.startswith('.'):
                    if item.is_file():
                        files.append(self._get_file_details(item, list_path))
                    elif item.is_dir():
                        dirs.append(item.name)
                        
        # Sort by name
        files.sort(key=lambda x: x['name'])
        dirs.sort()
        
        return {
            "success": True,
            "path": str(list_path),
            "pattern": pattern,
            "total_files": len(files),
            "total_dirs": len(dirs),
            "files": files,
            "directories": dirs
        }
        
    async def read_file(self, file_path: str, encoding: str = 'utf-8', 
                       lines: int = None, **kwargs) -> Dict[str, Any]:
        """Read file contents."""
        file = Path(file_path)
        
        if not file.is_absolute():
            file = self.project_root / file
            
        if not file.exists():
            return {"error": f"File {file_path} not found"}
            
        if not file.is_file():
            return {"error": f"{file_path} is not a file"}
            
        try:
            # Detect file type
            mime_type = mimetypes.guess_type(str(file))[0] or 'text/plain'
            
            # For binary files
            if not mime_type.startswith('text') and mime_type != 'application/json':
                return {
                    "success": True,
                    "file": str(file),
                    "type": "binary",
                    "mime_type": mime_type,
                    "size": file.stat().st_size,
                    "message": "Binary file - content not displayed"
                }
                
            # Read text file
            with open(file, 'r', encoding=encoding) as f:
                if lines:
                    content = ''.join(f.readline() for _ in range(lines))
                else:
                    content = f.read()
                    
            return {
                "success": True,
                "file": str(file),
                "content": content,
                "size": len(content),
                "lines": content.count('\n') + 1,
                "encoding": encoding
            }
            
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
            
    async def write_file(self, file_path: str, content: str, 
                        mode: str = 'w', encoding: str = 'utf-8', **kwargs) -> Dict[str, Any]:
        """Write content to file."""
        file = Path(file_path)
        
        if not file.is_absolute():
            file = self.project_root / file
            
        try:
            # Create directory if needed
            file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(file, mode, encoding=encoding) as f:
                f.write(content)
                
            return {
                "success": True,
                "file": str(file),
                "size": len(content),
                "mode": mode,
                "action": "created" if mode == 'w' else "appended"
            }
            
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}
            
    async def search_files(self, pattern: str, path: str = '.', 
                          file_type: str = None, **kwargs) -> Dict[str, Any]:
        """Search for files by name pattern."""
        search_path = Path(path) if path != '.' else self.project_root
        
        if not search_path.exists():
            return {"error": f"Path {path} does not exist"}
            
        matches = []
        
        # Convert pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        pattern_re = re.compile(regex_pattern, re.IGNORECASE)
        
        for file_path in search_path.rglob('*'):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                # Check name match
                if pattern_re.search(file_path.name):
                    # Check file type if specified
                    if file_type:
                        if file_type == 'code' and file_path.suffix not in self.code_extensions:
                            continue
                        elif file_type == 'doc' and file_path.suffix not in self.doc_extensions:
                            continue
                        elif file_type not in ['code', 'doc'] and file_path.suffix != f'.{file_type}':
                            continue
                            
                    matches.append(self._get_file_details(file_path, search_path))
                    
        # Sort by relevance (exact matches first)
        matches.sort(key=lambda x: (pattern.lower() != x['name'].lower(), x['name']))
        
        return {
            "success": True,
            "pattern": pattern,
            "path": str(search_path),
            "matches": len(matches),
            "files": matches[:100]  # Limit results
        }
        
    async def grep_files(self, pattern: str, path: str = '.', 
                        file_pattern: str = '*', case_sensitive: bool = True,
                        context_lines: int = 0, **kwargs) -> Dict[str, Any]:
        """Search content in files."""
        grep_path = Path(path) if path != '.' else self.project_root
        
        if not grep_path.exists():
            return {"error": f"Path {path} does not exist"}
            
        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern_re = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex pattern: {str(e)}"}
            
        matches = []
        files_searched = 0
        
        # Search files
        for file_path in grep_path.rglob(file_pattern):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                # Skip binary files
                if file_path.suffix in ['.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe']:
                    continue
                    
                files_searched += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    for i, line in enumerate(lines):
                        if pattern_re.search(line):
                            match_info = {
                                'file': str(file_path.relative_to(grep_path)),
                                'line_number': i + 1,
                                'line': line.rstrip(),
                                'match': pattern_re.search(line).group(0)
                            }
                            
                            # Add context if requested
                            if context_lines > 0:
                                start = max(0, i - context_lines)
                                end = min(len(lines), i + context_lines + 1)
                                match_info['context'] = [
                                    {'line': j + 1, 'text': lines[j].rstrip()} 
                                    for j in range(start, end)
                                ]
                                
                            matches.append(match_info)
                            
                except Exception:
                    # Skip files that can't be read as text
                    pass
                    
        return {
            "success": True,
            "pattern": pattern,
            "files_searched": files_searched,
            "total_matches": len(matches),
            "matches": matches[:100],  # Limit results
            "truncated": len(matches) > 100
        }
        
    async def find_files(self, criteria: Dict[str, Any], path: str = '.', **kwargs) -> Dict[str, Any]:
        """Find files by multiple criteria."""
        find_path = Path(path) if path != '.' else self.project_root
        
        if not find_path.exists():
            return {"error": f"Path {path} does not exist"}
            
        matches = []
        
        # Extract criteria
        name_pattern = criteria.get('name')
        min_size = criteria.get('min_size', 0)
        max_size = criteria.get('max_size', float('inf'))
        file_type = criteria.get('type')
        modified_after = criteria.get('modified_after')
        modified_before = criteria.get('modified_before')
        
        for file_path in find_path.rglob('*'):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                # Check all criteria
                stat = file_path.stat()
                
                # Name pattern
                if name_pattern and not re.search(name_pattern, file_path.name, re.IGNORECASE):
                    continue
                    
                # Size
                if not (min_size <= stat.st_size <= max_size):
                    continue
                    
                # Type
                if file_type:
                    if file_type == 'code' and file_path.suffix not in self.code_extensions:
                        continue
                    elif file_type == 'doc' and file_path.suffix not in self.doc_extensions:
                        continue
                        
                # Modified time
                if modified_after or modified_before:
                    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    if modified_after and mtime < modified_after:
                        continue
                    if modified_before and mtime > modified_before:
                        continue
                        
                matches.append(self._get_file_details(file_path, find_path))
                
        # Sort by modification time (newest first)
        matches.sort(key=lambda x: x['modified'], reverse=True)
        
        return {
            "success": True,
            "criteria": criteria,
            "path": str(find_path),
            "matches": len(matches),
            "files": matches[:100]  # Limit results
        }
        
    async def analyze_files(self, path: str = '.', **kwargs) -> Dict[str, Any]:
        """Analyze file types and sizes in directory."""
        analyze_path = Path(path) if path != '.' else self.project_root
        
        if not analyze_path.exists():
            return {"error": f"Path {path} does not exist"}
            
        stats = {
            'total_files': 0,
            'total_size': 0,
            'by_extension': defaultdict(lambda: {'count': 0, 'size': 0}),
            'by_type': defaultdict(lambda: {'count': 0, 'size': 0}),
            'largest_files': [],
            'oldest_files': [],
            'newest_files': []
        }
        
        all_files = []
        
        for file_path in analyze_path.rglob('*'):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                stat = file_path.stat()
                stats['total_files'] += 1
                stats['total_size'] += stat.st_size
                
                # By extension
                ext = file_path.suffix or 'no_extension'
                stats['by_extension'][ext]['count'] += 1
                stats['by_extension'][ext]['size'] += stat.st_size
                
                # By type
                file_type = self._get_file_type(file_path)
                stats['by_type'][file_type]['count'] += 1
                stats['by_type'][file_type]['size'] += stat.st_size
                
                # Store for sorting
                all_files.append({
                    'path': str(file_path.relative_to(analyze_path)),
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
                
        # Get top files
        all_files_by_size = sorted(all_files, key=lambda x: x['size'], reverse=True)
        all_files_by_time = sorted(all_files, key=lambda x: x['modified'])
        
        stats['largest_files'] = all_files_by_size[:10]
        stats['oldest_files'] = all_files_by_time[:10]
        stats['newest_files'] = all_files_by_time[-10:][::-1]
        
        # Convert defaultdicts to regular dicts
        stats['by_extension'] = dict(stats['by_extension'])
        stats['by_type'] = dict(stats['by_type'])
        
        # Add summary
        stats['summary'] = {
            'average_file_size': round(stats['total_size'] / max(stats['total_files'], 1), 2),
            'total_size_mb': round(stats['total_size'] / 1024 / 1024, 2),
            'unique_extensions': len(stats['by_extension']),
            'primary_type': max(stats['by_type'].items(), key=lambda x: x[1]['count'])[0] if stats['by_type'] else 'unknown'
        }
        
        return {
            "success": True,
            "path": str(analyze_path),
            "analysis": stats
        }
        
    async def smart_search(self, query: str, path: str = '.', **kwargs) -> Dict[str, Any]:
        """AI-powered content search."""
        search_path = Path(path) if path != '.' else self.project_root
        
        if not search_path.exists():
            return {"error": f"Path {path} does not exist"}
            
        # First, get AI to understand the query
        prompt = f"""Analyze this search query and extract key concepts:
Query: {query}

Provide:
1. Key terms to search for
2. File types most likely to contain this information
3. Regex patterns that might help"""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if not response.get('success'):
            # Fallback to simple search
            return await self.grep_files(query, path)
            
        # Extract search parameters from AI response
        ai_analysis = response.get('content', '')
        
        # Perform smart search
        results = []
        relevance_scores = {}
        
        # Search through files
        for file_path in search_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.code_extensions.union(self.doc_extensions):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Calculate relevance
                    relevance = 0
                    
                    # Direct query match
                    if query.lower() in content.lower():
                        relevance += 10
                        
                    # Check for related terms
                    for term in query.lower().split():
                        if term in content.lower():
                            relevance += 1
                            
                    if relevance > 0:
                        results.append({
                            'file': str(file_path.relative_to(search_path)),
                            'relevance': relevance,
                            'preview': self._get_content_preview(content, query),
                            'type': self._get_file_type(file_path)
                        })
                        
                except Exception:
                    pass
                    
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return {
            "success": True,
            "query": query,
            "ai_analysis": ai_analysis,
            "total_results": len(results),
            "results": results[:20]  # Top 20 results
        }
        
    async def copy_file(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        """Copy file to new location."""
        src = Path(source)
        dst = Path(destination)
        
        if not src.is_absolute():
            src = self.project_root / src
        if not dst.is_absolute():
            dst = self.project_root / dst
            
        if not src.exists():
            return {"error": f"Source file {source} not found"}
            
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            
            return {
                "success": True,
                "source": str(src),
                "destination": str(dst),
                "size": src.stat().st_size
            }
            
        except Exception as e:
            return {"error": f"Failed to copy file: {str(e)}"}
            
    async def move_file(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        """Move file to new location."""
        src = Path(source)
        dst = Path(destination)
        
        if not src.is_absolute():
            src = self.project_root / src
        if not dst.is_absolute():
            dst = self.project_root / dst
            
        if not src.exists():
            return {"error": f"Source file {source} not found"}
            
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            
            return {
                "success": True,
                "source": str(src),
                "destination": str(dst)
            }
            
        except Exception as e:
            return {"error": f"Failed to move file: {str(e)}"}
            
    async def delete_file(self, file_path: str, confirm: bool = False, **kwargs) -> Dict[str, Any]:
        """Delete file (with safety check)."""
        file = Path(file_path)
        
        if not file.is_absolute():
            file = self.project_root / file
            
        if not file.exists():
            return {"error": f"File {file_path} not found"}
            
        if not confirm:
            return {
                "warning": "Deletion requires confirmation",
                "file": str(file),
                "size": file.stat().st_size,
                "would_delete": True
            }
            
        try:
            if file.is_file():
                file.unlink()
            else:
                shutil.rmtree(file)
                
            return {
                "success": True,
                "deleted": str(file)
            }
            
        except Exception as e:
            return {"error": f"Failed to delete file: {str(e)}"}
            
    async def create_directory(self, path: str, **kwargs) -> Dict[str, Any]:
        """Create directory."""
        dir_path = Path(path)
        
        if not dir_path.is_absolute():
            dir_path = self.project_root / dir_path
            
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            
            return {
                "success": True,
                "created": str(dir_path),
                "existed": dir_path.exists()
            }
            
        except Exception as e:
            return {"error": f"Failed to create directory: {str(e)}"}
            
    async def get_file_info(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Get detailed file information."""
        file = Path(file_path)
        
        if not file.is_absolute():
            file = self.project_root / file
            
        if not file.exists():
            return {"error": f"File {file_path} not found"}
            
        stat = file.stat()
        
        # Calculate hash for files
        file_hash = None
        if file.is_file() and stat.st_size < 10 * 1024 * 1024:  # Only hash files < 10MB
            try:
                with open(file, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            except:
                pass
                
        return {
            "success": True,
            "path": str(file),
            "name": file.name,
            "type": "directory" if file.is_dir() else self._get_file_type(file),
            "size": stat.st_size,
            "size_human": self._human_readable_size(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime, tz=timezone.utc).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
            "mime_type": mimetypes.guess_type(str(file))[0],
            "hash": file_hash
        }
        
    async def compare_files(self, file1: str, file2: str, **kwargs) -> Dict[str, Any]:
        """Compare two files."""
        f1 = Path(file1)
        f2 = Path(file2)
        
        if not f1.is_absolute():
            f1 = self.project_root / f1
        if not f2.is_absolute():
            f2 = self.project_root / f2
            
        if not f1.exists():
            return {"error": f"File {file1} not found"}
        if not f2.exists():
            return {"error": f"File {file2} not found"}
            
        try:
            # Compare basic properties
            stat1 = f1.stat()
            stat2 = f2.stat()
            
            result = {
                "file1": str(f1),
                "file2": str(f2),
                "same_size": stat1.st_size == stat2.st_size,
                "size_diff": stat1.st_size - stat2.st_size,
                "modified_diff": stat1.st_mtime - stat2.st_mtime
            }
            
            # For text files, compare content
            if f1.suffix in self.code_extensions.union(self.doc_extensions):
                with open(f1, 'r', encoding='utf-8') as file1:
                    content1 = file1.readlines()
                with open(f2, 'r', encoding='utf-8') as file2:
                    content2 = file2.readlines()
                    
                result['line_diff'] = len(content1) - len(content2)
                result['identical_content'] = content1 == content2
                
                # Find differences
                if not result['identical_content']:
                    differences = []
                    for i, (line1, line2) in enumerate(zip(content1, content2)):
                        if line1 != line2:
                            differences.append({
                                'line': i + 1,
                                'file1': line1.rstrip(),
                                'file2': line2.rstrip()
                            })
                            if len(differences) >= 10:  # Limit differences
                                break
                                
                    result['differences'] = differences
                    
            return {"success": True, "comparison": result}
            
        except Exception as e:
            return {"error": f"Failed to compare files: {str(e)}"}
            
    def _get_file_details(self, file_path: Path, base_path: Path) -> Dict[str, Any]:
        """Get file details for listing."""
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'path': str(file_path.relative_to(base_path)),
            'size': stat.st_size,
            'size_human': self._human_readable_size(stat.st_size),
            'modified': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            'type': self._get_file_type(file_path)
        }
        
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type."""
        if file_path.suffix in self.code_extensions:
            return 'code'
        elif file_path.suffix in self.doc_extensions:
            return 'document'
        elif file_path.suffix in ['.jpg', '.png', '.gif', '.svg']:
            return 'image'
        elif file_path.suffix in ['.zip', '.tar', '.gz', '.rar']:
            return 'archive'
        elif file_path.suffix in ['.json', '.xml', '.yaml', '.yml']:
            return 'config'
        else:
            return 'other'
            
    def _human_readable_size(self, size: int) -> str:
        """Convert size to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
        
    def _get_content_preview(self, content: str, query: str) -> str:
        """Get content preview around query match."""
        lines = content.split('\n')
        query_lower = query.lower()
        
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                preview_lines = lines[start:end]
                return '\n'.join(f"{start + j + 1}: {line}" for j, line in enumerate(preview_lines))
                
        # Return first few lines if no match
        return '\n'.join(f"{i + 1}: {line}" for i, line in enumerate(lines[:3]))
        
    async def _intelligent_file_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Use AI to understand file operation intent."""
        prompt = f"""User wants to perform this file operation: {operation}
Arguments: {json.dumps(kwargs)}

Determine the correct operation from:
- list, read, write, search, grep, find, analyze, smart_search
- copy, move, delete, create_directory, get_info, compare

Return the operation name and any missing parameters."""

        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model="gemini"
        )
        
        if response.get('success'):
            # Try to extract operation from response
            content = response.get('content', '').lower()
            
            for op in ['list', 'read', 'write', 'search', 'grep', 'find', 
                      'analyze', 'smart_search', 'copy', 'move', 'delete',
                      'create_directory', 'get_info', 'compare']:
                if op in content:
                    return await self.execute(op, **kwargs)
                    
        return {
            "error": f"Could not understand operation: {operation}",
            "available_operations": list(self.execute.__annotations__['operations'])
        }


async def main(operation: str = None, **kwargs):
    """Entry point for file manager."""
    manager = FileManager()
    
    if operation:
        return await manager.execute(operation, **kwargs)
    else:
        # Default to list current directory
        return await manager.list_files(**kwargs)
        

if __name__ == "__main__":
    # Test the manager
    import asyncio
    
    async def test():
        manager = FileManager()
        
        # Test operations
        print("List:", await manager.list_files(pattern="*.py", recursive=False))
        print("\nSearch:", await manager.search_files("test", file_type="code"))
        print("\nAnalyze:", await manager.analyze_files())
        
    asyncio.run(test())