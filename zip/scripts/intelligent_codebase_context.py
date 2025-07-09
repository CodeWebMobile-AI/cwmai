"""
Intelligent Codebase Context Module

Provides smart, query-aware code analysis that only loads what's needed,
with aggressive caching and progressive enhancement.
"""

import ast
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from datetime import datetime, timedelta
import re
from functools import lru_cache
import asyncio


class IntelligentCodebaseContext:
    """Provides query-aware codebase understanding with smart caching."""
    
    def __init__(self, cache_dir: str = ".codebase_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory caches
        self._file_index = None
        self._import_graph = None
        self._last_index_time = None
        
        # Query patterns that need code analysis
        self.code_query_patterns = [
            r'\b(how|where|what|which)\s+(does|is|are)\s+\w+\s+(implemented|defined|work)',
            r'\b(show|find|search)\s+(me\s+)?(the\s+)?(code|function|class|method)',
            r'\b(analyze|review|check|debug|fix)\s+(the\s+)?(code|function|bug)',
            r'\bimport\s+error|module\s+not\s+found',
            r'\b(refactor|optimize|improve)\s+(this|the)\s+code',
            r'\bwhere\s+is\s+\w+\s+used',
            r'\bwhat\s+calls\s+\w+',
            r'\b(create|generate|write)\s+.*\s+(like|similar\s+to)\s+\w+',
        ]
        
    async def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """Get intelligent context based on the query."""
        context = {
            'query_type': self._classify_query(query),
            'timestamp': datetime.now().isoformat()
        }
        
        # Always include basic structure (lightweight)
        context['structure'] = await self._get_file_structure()
        
        # Determine what additional context is needed
        if context['query_type'] == 'code_analysis':
            # Extract specific entities mentioned in the query
            entities = self._extract_code_entities(query)
            
            if entities['files']:
                context['file_analysis'] = await self._analyze_specific_files(entities['files'])
            
            if entities['functions']:
                context['function_analysis'] = await self._analyze_functions(entities['functions'])
                
            if entities['classes']:
                context['class_analysis'] = await self._analyze_classes(entities['classes'])
            
            # If no specific entities, provide high-level overview
            if not any(entities.values()):
                context['overview'] = await self._get_codebase_overview()
                
        elif context['query_type'] == 'tool_related':
            context['tools'] = await self._get_tool_context()
            
        elif context['query_type'] == 'error_debugging':
            context['error_context'] = await self._get_debugging_context(query)
            
        elif context['query_type'] == 'architecture':
            context['architecture'] = await self._get_architecture_context()
        
        # Add usage hints for the AI
        context['usage_hints'] = self._generate_usage_hints(context)
        
        return context
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query to determine needed context."""
        query_lower = query.lower()
        
        # Check for code analysis patterns
        for pattern in self.code_query_patterns:
            if re.search(pattern, query_lower):
                return 'code_analysis'
        
        # Tool-related queries
        if any(word in query_lower for word in ['tool', 'custom_tools', 'create tool']):
            return 'tool_related'
            
        # Error/debugging queries
        if any(word in query_lower for word in ['error', 'bug', 'fix', 'broken', 'failing']):
            return 'error_debugging'
            
        # Architecture queries
        if any(word in query_lower for word in ['architecture', 'structure', 'design', 'pattern']):
            return 'architecture'
            
        # Default - minimal context needed
        return 'general'
    
    def _extract_code_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract specific code entities mentioned in the query."""
        entities = {
            'files': [],
            'functions': [],
            'classes': [],
            'modules': []
        }
        
        # Look for file paths
        file_pattern = r'[\w/]+\.py'
        entities['files'] = re.findall(file_pattern, query)
        
        # Look for function names (usually snake_case)
        func_pattern = r'\b[a-z_][a-z0-9_]*(?:_[a-z0-9_]+)*\b'
        potential_funcs = re.findall(func_pattern, query.lower())
        
        # Filter out common words
        common_words = {'the', 'in', 'is', 'at', 'to', 'for', 'and', 'or', 'but', 'how', 'what', 'where', 'when', 'why'}
        entities['functions'] = [f for f in potential_funcs if f not in common_words and len(f) > 3]
        
        # Look for class names (usually CamelCase)
        class_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*\b'
        entities['classes'] = re.findall(class_pattern, query)
        
        # Look for module references
        module_pattern = r'(?:from|import)\s+([\w\.]+)'
        entities['modules'] = re.findall(module_pattern, query)
        
        return entities
    
    async def _get_file_structure(self) -> Dict[str, Any]:
        """Get lightweight file structure (cached)."""
        cache_file = self.cache_dir / "file_structure.json"
        
        # Check if cache is fresh (less than 1 hour old)
        if cache_file.exists():
            stat = cache_file.stat()
            if datetime.now() - datetime.fromtimestamp(stat.st_mtime) < timedelta(hours=1):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Build file structure
        structure = {
            'directories': {},
            'key_files': {},
            'total_files': 0
        }
        
        exclude = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', '.venv'}
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in exclude]
            
            for file in files:
                if file.endswith('.py'):
                    rel_path = os.path.relpath(os.path.join(root, file))
                    
                    # Categorize key files
                    if 'test_' in file:
                        category = 'tests'
                    elif file in ['main.py', 'app.py', '__main__.py']:
                        category = 'entry_points'
                    elif 'config' in file:
                        category = 'configuration'
                    elif file.endswith('_manager.py') or file.endswith('_controller.py'):
                        category = 'managers'
                    else:
                        category = 'modules'
                    
                    if category not in structure['key_files']:
                        structure['key_files'][category] = []
                    
                    structure['key_files'][category].append(rel_path)
                    structure['total_files'] += 1
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(structure, f)
        
        return structure
    
    async def _analyze_specific_files(self, files: List[str]) -> Dict[str, Any]:
        """Analyze specific files mentioned in the query."""
        analysis = {}
        
        for file_path in files:
            if not os.path.exists(file_path):
                # Try to find the file
                found_files = list(Path('.').rglob(f"*{file_path}*"))
                if found_files:
                    file_path = str(found_files[0])
                else:
                    continue
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Basic analysis
                tree = ast.parse(content)
                
                file_info = {
                    'path': file_path,
                    'size': len(content),
                    'lines': content.count('\n'),
                    'classes': [],
                    'functions': [],
                    'imports': []
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        file_info['classes'].append({
                            'name': node.name,
                            'line': node.lineno,
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        })
                    elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                        file_info['functions'].append({
                            'name': node.name,
                            'line': node.lineno,
                            'is_async': isinstance(node, ast.AsyncFunctionDef)
                        })
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            file_info['imports'].extend([n.name for n in node.names])
                        else:
                            file_info['imports'].append(node.module)
                
                analysis[file_path] = file_info
                
            except Exception as e:
                analysis[file_path] = {'error': str(e)}
        
        return analysis
    
    async def _analyze_functions(self, function_names: List[str]) -> Dict[str, Any]:
        """Find and analyze specific functions."""
        results = {}
        
        # Build function index if not cached
        if not hasattr(self, '_function_index'):
            self._function_index = await self._build_function_index()
        
        for func_name in function_names:
            if func_name in self._function_index:
                results[func_name] = self._function_index[func_name]
            else:
                # Try fuzzy matching
                similar = [name for name in self._function_index.keys() 
                          if func_name in name or name in func_name]
                if similar:
                    results[f"{func_name}_similar"] = similar[:5]
        
        return results
    
    async def _build_function_index(self) -> Dict[str, Any]:
        """Build an index of all functions (cached)."""
        cache_file = self.cache_dir / "function_index.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        index = {}
        
        for py_file in Path('.').rglob('*.py'):
            if any(part in str(py_file) for part in ['.git', '__pycache__', 'venv']):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_info = {
                            'file': str(py_file),
                            'line': node.lineno,
                            'is_async': isinstance(node, ast.AsyncFunctionDef),
                            'docstring': ast.get_docstring(node),
                            'args': [arg.arg for arg in node.args.args]
                        }
                        
                        # Include the first few lines of the function
                        lines = content.split('\n')
                        start = node.lineno - 1
                        end = min(start + 10, len(lines))
                        func_info['preview'] = '\n'.join(lines[start:end])
                        
                        index[node.name] = func_info
                        
            except:
                pass
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(index, f)
        
        return index
    
    async def _get_tool_context(self) -> Dict[str, Any]:
        """Get context about tools."""
        tools = {
            'custom_tools': [],
            'tool_patterns': {}
        }
        
        tools_dir = Path('scripts/custom_tools')
        if tools_dir.exists():
            for tool_file in tools_dir.glob('*.py'):
                if tool_file.name == '__init__.py':
                    continue
                    
                try:
                    with open(tool_file, 'r') as f:
                        content = f.read()
                    
                    # Extract key information
                    tree = ast.parse(content)
                    docstring = ast.get_docstring(tree)
                    
                    tool_info = {
                        'name': tool_file.stem,
                        'file': str(tool_file),
                        'description': docstring.split('\n')[0] if docstring else 'No description',
                        'has_main': 'def main(' in content or 'async def main(' in content
                    }
                    
                    tools['custom_tools'].append(tool_info)
                    
                except:
                    pass
        
        return tools
    
    async def _get_debugging_context(self, query: str) -> Dict[str, Any]:
        """Get context for debugging errors."""
        context = {
            'error_type': 'unknown',
            'related_files': [],
            'suggestions': []
        }
        
        # Identify error type
        if 'import' in query.lower() or 'module not found' in query.lower():
            context['error_type'] = 'import_error'
            
            # Extract module name
            module_match = re.search(r"module\s+['\"]?(\w+)['\"]?", query, re.I)
            if module_match:
                module_name = module_match.group(1)
                
                # Find where this module might be
                possible_locations = []
                for py_file in Path('.').rglob('*.py'):
                    if module_name in py_file.stem:
                        possible_locations.append(str(py_file))
                
                context['suggestions'] = [
                    f"Found possible module at: {loc}" for loc in possible_locations[:5]
                ]
        
        elif 'attribute' in query.lower():
            context['error_type'] = 'attribute_error'
            context['suggestions'] = [
                "Check if the object has the expected type",
                "Verify the import statements",
                "Look for typos in attribute names"
            ]
        
        return context
    
    async def _get_architecture_context(self) -> Dict[str, Any]:
        """Get high-level architecture context."""
        return {
            'main_components': {
                'scripts/': 'Core functionality modules',
                'scripts/custom_tools/': 'Generated and custom tools',
                'scripts/redis_integration/': 'Redis connectivity',
                'tests/': 'Test files',
                'logs/': 'System logs'
            },
            'key_patterns': [
                "Manager classes for state management",
                "Async/await for concurrent operations",
                "Tool generation and execution system",
                "AI client abstraction for multiple providers"
            ],
            'entry_points': [
                "cwmai - Main CLI entry point",
                "run_continuous_ai.py - Autonomous system",
                "scripts/conversational_ai_assistant.py - Chat interface"
            ]
        }
    
    async def _get_codebase_overview(self) -> Dict[str, Any]:
        """Get a high-level overview of the codebase."""
        overview = {
            'statistics': {},
            'main_modules': [],
            'complexity': 'medium'
        }
        
        # Count files by type
        py_files = list(Path('.').rglob('*.py'))
        overview['statistics'] = {
            'python_files': len(py_files),
            'test_files': len([f for f in py_files if 'test_' in f.name]),
            'tool_files': len(list(Path('scripts/custom_tools').glob('*.py')))
        }
        
        # Identify main modules
        for py_file in Path('scripts').glob('*.py'):
            if py_file.stat().st_size > 5000:  # Significant modules
                overview['main_modules'].append({
                    'name': py_file.stem,
                    'size': py_file.stat().st_size
                })
        
        return overview
    
    def _generate_usage_hints(self, context: Dict[str, Any]) -> List[str]:
        """Generate hints for the AI on how to use the provided context."""
        hints = []
        
        if 'file_analysis' in context:
            hints.append("File analysis includes class/function locations - reference with file:line format")
        
        if 'function_analysis' in context:
            hints.append("Function previews show implementation - use for understanding logic")
            
        if 'tools' in context:
            hints.append("Tool context shows available custom tools - suggest relevant ones")
            
        if 'error_context' in context:
            hints.append("Error context includes suggestions - use them to guide debugging")
            
        if context.get('query_type') == 'general':
            hints.append("Query seems general - provided minimal context to stay focused")
        
        return hints