#!/usr/bin/env python3
"""
Tool Generation Templates and Context Providers
Provides rich context and templates for AI tool generation
Enhanced with AI capabilities for smarter generation
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import ast
import logging
import asyncio
from datetime import datetime

# Try to import intelligent system, fall back to basic if not available
try:
    from scripts.intelligent_tool_generation_templates import (
        IntelligentToolGenerationTemplates,
        ToolRequirementAnalysis
    )
    INTELLIGENT_MODE = True
except ImportError:
    INTELLIGENT_MODE = False


class ToolGenerationTemplates:
    """Manages templates and context for tool generation."""
    
    def __init__(self, use_ai=True):
        self.templates = self._load_templates()
        self.common_patterns = self._load_common_patterns()
        self.error_fixes = self._load_error_fixes()
        self.discovered_scripts = self._discover_available_scripts()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize intelligent system if available and requested
        self.intelligent_system = None
        self.use_ai = use_ai
        if INTELLIGENT_MODE and use_ai:
            try:
                self.intelligent_system = IntelligentToolGenerationTemplates()
                self.logger.info("Intelligent tool generation system initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize intelligent system: {e}")
                self.intelligent_system = None
    
    def _load_templates(self) -> Dict[str, str]:
        """Load tool templates by category."""
        return {
            "file_operations": '''"""
AI-Generated Tool: {name}
Description: {description}
Generated: {timestamp}
Requirements: {requirements}
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

__description__ = "{description}"
__parameters__ = {{
    "path": {{
        "type": "string",
        "description": "File or directory path",
        "required": False,
        "default": "."
    }}
}}
__examples__ = [
    {{"description": "Process current directory", "code": "await {name}()"}},
    {{"description": "Process specific path", "code": "await {name}(path='/path/to/target')"}}
]


async def {name}(**kwargs) -> Dict[str, Any]:
    """{description}"""
    path = kwargs.get('path', '.')
    
    try:
        target_path = Path(path)
        
        if not target_path.exists():
            return {{"error": f"Path not found: {{path}}"}}
        
        # TODO: Implement file operation logic
        result = {{
            "path": str(target_path.absolute()),
            "processed": 0,
            "summary": "File operation completed"
        }}
        
        return result
        
    except PermissionError:
        return {{"error": f"Permission denied accessing: {{path}}"}}
    except Exception as e:
        return {{"error": f"Error processing files: {{str(e)}}"}}
''',

            "data_analysis": '''"""
AI-Generated Tool: {name}
Description: {description}
Generated: {timestamp}
Requirements: {requirements}
"""

from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict
from datetime import datetime
import json

from scripts.state_manager import StateManager

__description__ = "{description}"
__parameters__ = {{
    "filter": {{
        "type": "string",
        "description": "Optional filter criteria",
        "required": False
    }},
    "limit": {{
        "type": "integer",
        "description": "Maximum results to return",
        "required": False,
        "default": 100
    }}
}}
__examples__ = [
    {{"description": "Analyze all data", "code": "await {name}()"}},
    {{"description": "Analyze with filter", "code": "await {name}(filter='active', limit=10)"}}
]


async def {name}(**kwargs) -> Dict[str, Any]:
    """{description}"""
    filter_criteria = kwargs.get('filter', None)
    limit = kwargs.get('limit', 100)
    
    try:
        # Initialize state manager
        state_manager = StateManager()
        state = state_manager.load_state()
        
        # TODO: Implement analysis logic
        results = []
        
        # Apply filter if provided
        if filter_criteria:
            # Filter logic here
            pass
        
        # Limit results
        results = results[:limit]
        
        return {{
            "total": len(results),
            "filtered": filter_criteria is not None,
            "results": results,
            "summary": f"Analyzed {{len(results)}} items"
        }}
        
    except Exception as e:
        return {{"error": f"Analysis error: {{str(e)}}"}}
''',

            "system_operations": '''"""
AI-Generated Tool: {name}
Description: {description}
Generated: {timestamp}
Requirements: {requirements}
"""

import subprocess
import platform
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime

__description__ = "{description}"
__parameters__ = {{}}
__examples__ = [
    {{"description": "Execute system operation", "code": "await {name}()"}}
]


async def {name}(**kwargs) -> Dict[str, Any]:
    """{description}"""
    try:
        # TODO: Implement system operation
        result = {{
            "timestamp": datetime.now().isoformat(),
            "platform": platform.system(),
            "status": "success",
            "summary": "System operation completed"
        }}
        
        return result
        
    except subprocess.CalledProcessError as e:
        return {{"error": f"Command failed: {{e.cmd}} - {{e.stderr}}"}}
    except Exception as e:
        return {{"error": f"System operation error: {{str(e)}}"}}
''',

            "git_operations": '''"""
AI-Generated Tool: {name}
Description: {description}
Generated: {timestamp}
Requirements: {requirements}
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

__description__ = "{description}"
__parameters__ = {{
    "repo_path": {{
        "type": "string",
        "description": "Repository path",
        "required": False,
        "default": "."
    }}
}}
__examples__ = [
    {{"description": "Operate on current repo", "code": "await {name}()"}},
    {{"description": "Operate on specific repo", "code": "await {name}(repo_path='/path/to/repo')"}}
]


async def {name}(**kwargs) -> Dict[str, Any]:
    """{description}"""
    repo_path = kwargs.get('repo_path', '.')
    
    try:
        # Verify git repository
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {{"error": f"Not a git repository: {{repo_path}}"}}
        
        # TODO: Implement git operation
        
        return {{
            "repo_path": repo_path,
            "status": "success",
            "summary": "Git operation completed"
        }}
        
    except FileNotFoundError:
        return {{"error": "Git is not installed"}}
    except Exception as e:
        return {{"error": f"Git operation error: {{str(e)}}"}}
'''
        }
    
    def _load_common_patterns(self) -> Dict[str, str]:
        """Load common code patterns."""
        return {
            "validate_directory": '''
        if not os.path.isdir(directory):
            return {"error": f"Not a directory: {directory}"}
''',
            "validate_file": '''
        if not os.path.isfile(file_path):
            return {"error": f"File not found: {file_path}"}
''',
            "parse_json_safe": '''
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {str(e)}"}
''',
            "execute_command": '''
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return {"error": f"Command failed: {result.stderr}"}
''',
            "async_file_read": '''
        async def read_file_async(path: str) -> str:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, Path(path).read_text)
'''
        }
    
    def _load_error_fixes(self) -> Dict[str, str]:
        """Load common error fixes."""
        return {
            "missing_state_manager_import": "from scripts.state_manager import StateManager",
            "missing_path_import": "from pathlib import Path",
            "missing_typing_import": "from typing import Dict, Any, List, Optional",
            "missing_json_import": "import json",
            "missing_os_import": "import os",
            "missing_datetime_import": "from datetime import datetime, timedelta",
            "missing_asyncio_import": "import asyncio",
            "undefined_timezone": "from datetime import timezone",
            "undefined_counter": "from collections import Counter",
            "undefined_defaultdict": "from collections import defaultdict"
        }
    
    def _discover_available_scripts(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available scripts in the scripts directory."""
        discovered = {}
        scripts_dir = Path(__file__).parent
        
        # Skip these files/patterns
        skip_patterns = [
            '__pycache__', '__init__.py', '.pyc', 
            'test_', 'demo_', 'example_'
        ]
        
        # Special handling for custom tools - don't skip test_ prefix
        custom_tools_skip = [
            '__pycache__', '__init__.py', '.pyc'
        ]
        
        try:
            for script_path in scripts_dir.glob('*.py'):
                # Skip files matching skip patterns
                if any(pattern in script_path.name for pattern in skip_patterns):
                    continue
                
                module_name = script_path.stem
                
                # Try to extract information from the script
                script_info = self._extract_script_info(script_path)
                if script_info:
                    discovered[module_name] = script_info
            
            # Also check specialized directories
            specialized_dirs = ['custom_tools', 'specialized_agents', 'redis_integration']
            for dir_name in specialized_dirs:
                sub_dir = scripts_dir / dir_name
                if sub_dir.exists() and sub_dir.is_dir():
                    for script_path in sub_dir.glob('*.py'):
                        # Use different skip patterns for custom_tools
                        patterns_to_use = custom_tools_skip if dir_name == 'custom_tools' else skip_patterns
                        if any(pattern in script_path.name for pattern in patterns_to_use):
                            continue
                        
                        module_name = f"{dir_name}.{script_path.stem}"
                        script_info = self._extract_script_info(script_path)
                        if script_info:
                            discovered[module_name] = script_info
        
        except Exception as e:
            self.logger.error(f"Error discovering scripts: {e}")
        
        return discovered
    
    def _extract_script_info(self, script_path: Path) -> Optional[Dict[str, Any]]:
        """Extract information from a script file."""
        try:
            content = script_path.read_text(encoding='utf-8')
            
            # Parse the AST to get classes and functions
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                # For files with syntax errors, still try to extract basic info
                docstring = ""
                for line in content.split('\n')[:10]:
                    if line.strip().startswith('"""'):
                        docstring = line.strip().strip('"""')
                        break
                
                return {
                    'path': str(script_path),
                    'docstring': docstring or f"Tool: {script_path.stem}",
                    'classes': [],
                    'functions': [script_path.stem],  # Assume function name matches file
                    'category': self._categorize_script(str(script_path), docstring, [], []),
                    'has_async': 'async def' in content
                }
            
            # Extract module docstring
            docstring = ast.get_docstring(tree) or ""
            
            # Extract main classes and functions
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and node.name[0] != '_':
                    # Only public functions
                    functions.append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef) and node.name[0] != '_':
                    # Also include async functions
                    functions.append(node.name)
            
            # Extract imports to understand dependencies
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Categorize the script based on its name and content
            category = self._categorize_script(str(script_path), docstring, classes, functions)
            
            # For custom tools, try to get __description__ if available
            if 'custom_tools' in str(script_path) and '__description__' in content:
                # Extract __description__ value
                for line in content.split('\n'):
                    if line.strip().startswith('__description__'):
                        try:
                            desc_value = line.split('=', 1)[1].strip().strip('"\'')
                            if desc_value:
                                docstring = desc_value
                        except:
                            pass
                        break
            
            return {
                'path': str(script_path),
                'docstring': docstring.split('\n')[0] if docstring else f"Tool: {script_path.stem}",
                'classes': classes[:5],  # Limit for context size
                'functions': functions[:5],
                'category': category,
                'has_async': any('async' in content for func in functions if f'async def {func}' in content)
            }
            
        except Exception as e:
            self.logger.debug(f"Could not extract info from {script_path}: {e}")
            # Still return basic info for the file
            return {
                'path': str(script_path),
                'docstring': f"Tool: {script_path.stem}",
                'classes': [],
                'functions': [script_path.stem],
                'category': self._categorize_script(str(script_path), "", [], []),
                'has_async': False
            }
    
    def _categorize_script(self, filename: str, docstring: str, classes: List[str], functions: List[str]) -> str:
        """Categorize a script based on its content."""
        content = (filename + docstring + ' '.join(classes) + ' '.join(functions)).lower()
        
        # Check if it's a custom tool first
        if 'custom_tools' in filename or 'tool_task' in filename:
            return 'custom_tools'
        elif any(word in content for word in ['ai', 'llm', 'client', 'anthropic', 'openai']):
            return 'ai_integration'
        elif any(word in content for word in ['redis', 'cache', 'queue', 'stream']):
            return 'redis_integration'
        elif any(word in content for word in ['task', 'job', 'work', 'queue']):
            return 'task_management'
        elif any(word in content for word in ['state', 'manager', 'persist']):
            return 'state_management'
        elif any(word in content for word in ['tool', 'generate', 'create']):
            return 'tool_generation'
        elif any(word in content for word in ['swarm', 'agent', 'coordinator']):
            return 'agent_system'
        elif any(word in content for word in ['monitor', 'metric', 'log']):
            return 'monitoring'
        elif any(word in content for word in ['git', 'github', 'repo']):
            return 'version_control'
        else:
            return 'utility'
    
    def get_template(self, category: str) -> str:
        """Get template for a category."""
        return self.templates.get(category, self.templates['file_operations'])
    
    def get_context_prompt(self, tool_type: str) -> str:
        """Get context-specific prompt additions."""
        contexts = {
            "file": """
This tool will work with files and directories. Consider:
- Path validation and existence checks
- Permission errors
- Cross-platform path handling (use pathlib)
- Large file handling
- Encoding issues
""",
            "analysis": """
This tool will analyze data. Consider:
- Input validation and type checking
- Empty data handling
- Statistical calculations
- Memory efficiency for large datasets
- Clear summaries and insights
""",
            "system": """
This tool will interact with the system. Consider:
- Platform compatibility
- Permission requirements
- Resource usage
- Error recovery
- Security implications
""",
            "git": """
This tool will interact with git. Consider:
- Repository validation
- Git command availability
- Working directory state
- Remote operations
- Branch handling
"""
        }
        return contexts.get(tool_type, "")
    
    def get_import_context(self) -> str:
        """Get comprehensive import context including dynamically discovered scripts."""
        # Build dynamic project imports section
        project_imports = []
        
        # Group discovered scripts by category
        categorized_scripts = {}
        for module_name, info in self.discovered_scripts.items():
            category = info.get('category', 'utility')
            if category not in categorized_scripts:
                categorized_scripts[category] = []
            categorized_scripts[category].append((module_name, info))
        
        # Build import strings by category
        import_sections = []
        
        # Core imports (always available)
        import_sections.append("""Core Project Modules:
- from scripts.state_manager import StateManager
- from scripts.task_manager import TaskManager
- from scripts.http_ai_client import HTTPAIClient
- from scripts.repository_analyzer import RepositoryAnalyzer""")
        
        # Add discovered scripts by category
        category_names = {
            'custom_tools': 'Custom Tools (Ready to Use)',
            'ai_integration': 'AI Integration',
            'redis_integration': 'Redis/Caching',
            'task_management': 'Task Management',
            'state_management': 'State Management',
            'tool_generation': 'Tool Generation',
            'agent_system': 'Agent Systems',
            'monitoring': 'Monitoring/Logging',
            'version_control': 'Version Control',
            'utility': 'Utilities'
        }
        
        for category, scripts in sorted(categorized_scripts.items()):
            if scripts:
                section_lines = [f"\n{category_names.get(category, category.title())}:"]
                for module_name, info in scripts[:10]:  # Limit per category
                    # Build import statement
                    if '.' in module_name:  # Submodule
                        import_stmt = f"from scripts.{module_name.replace('.', '.')} import ..."
                    else:
                        if info['classes']:
                            import_stmt = f"from scripts.{module_name} import {', '.join(info['classes'][:3])}"
                        elif info['functions']:
                            import_stmt = f"from scripts.{module_name} import {', '.join(info['functions'][:3])}"
                        else:
                            import_stmt = f"import scripts.{module_name}"
                    
                    # Add description
                    desc = info['docstring'][:60] + "..." if len(info['docstring']) > 60 else info['docstring']
                    section_lines.append(f"- {import_stmt}  # {desc}")
                
                import_sections.append('\n'.join(section_lines))
        
        # Add special section for custom tools
        custom_tools_section = self._get_custom_tools_section()
        
        return f"""
AVAILABLE IMPORTS:

Standard Library:
- import os, sys, json, re, ast
- import asyncio, subprocess, platform
- from pathlib import Path
- from datetime import datetime, timedelta, timezone
- from collections import Counter, defaultdict, namedtuple, deque
- from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Awaitable, TypeVar, Generic
- import logging, traceback
- import time, random, uuid
- import hashlib, base64
- import urllib.parse, urllib.request
- import tempfile, shutil, glob
- import threading, multiprocessing
- from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
- import sqlite3, csv
- import xml.etree.ElementTree as ET
- import argparse, configparser
- from contextlib import contextmanager, asynccontextmanager
- from functools import lru_cache, partial, wraps
- from itertools import chain, groupby, combinations
- from dataclasses import dataclass, field
- from enum import Enum, auto
- import inspect, importlib

Third-party (if needed):
- import requests
- import psutil
- import aiohttp
- import redis
- import yaml
- import toml
- from bs4 import BeautifulSoup
- import pandas as pd (for data analysis)
- import numpy as np (for numerical operations)

Project-specific Modules:
{chr(10).join(import_sections)}

{custom_tools_section}

Total available project modules: {len(self.discovered_scripts)}

DESIGN PATTERNS & BEST PRACTICES:

1. **Result Type Pattern** - For clean error handling:
```python
@dataclass
class Result:
    value: Any = None
    error: str = None
    
    @property
    def is_ok(self) -> bool:
        return self.error is None
```

2. **Async Context Manager** - For resource management:
```python
@asynccontextmanager
async def managed_resource(path: str):
    resource = await acquire_resource(path)
    try:
        yield resource
    finally:
        await release_resource(resource)
```

3. **Retry Decorator** - For resilient operations:
```python
def retry(max_attempts=3, delay=1.0, backoff=2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    await asyncio.sleep(delay * (backoff ** attempt))
        return wrapper
    return decorator
```

4. **Batch Processing** - For performance:
```python
async def process_batch(items: List[Any], batch_size: int = 100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        results = await asyncio.gather(*[process_item(item) for item in batch])
        yield from results
```

CWMAI-SPECIFIC PATTERNS:

1. **Standard Return Format**:
```python
# Success
return {{
    "result": data,
    "summary": "Human-readable summary",
    "metadata": {{"count": len(data), "timestamp": datetime.now().isoformat()}}
}}

# Error
return {{
    "error": "Clear error message",
    "details": {{"context": "helpful debugging info"}}
}}
```

2. **Parameter Validation**:
```python
def validate_params(**kwargs):
    required = ['param1', 'param2']
    missing = [p for p in required if p not in kwargs]
    if missing:
        return {{"error": f"Missing required parameters: {{', '.join(missing)}}"}}
    return None
```

3. **Progress Tracking**:
```python
async def long_operation(items: List[Any], callback=None):
    total = len(items)
    for i, item in enumerate(items):
        result = await process_item(item)
        if callback:
            await callback({{"progress": i+1, "total": total, "percent": (i+1)/total*100}})
        yield result
```

PERFORMANCE TIPS:
1. Use asyncio.gather() for concurrent operations
2. Implement caching with @lru_cache for expensive computations
3. Use generators/async generators for large datasets
4. Batch database/API operations
5. Stream large files instead of loading into memory

SECURITY BEST PRACTICES:
1. Path traversal prevention: Path(user_input).resolve()
2. Input sanitization
3. Resource limits (file size, memory usage)
4. Never execute user input directly
5. Clean up resources in finally blocks

IMPORTANT: Only import what you actually use! Each import adds to startup time and memory usage.
"""
    
    def _get_custom_tools_section(self) -> str:
        """Get a special section highlighting available custom tools."""
        custom_tools = []
        
        for module_name, info in self.discovered_scripts.items():
            if info.get('category') == 'custom_tools':
                # Extract the main function name (usually same as module name)
                func_name = module_name.split('.')[-1]
                desc = info['docstring'][:80] + "..." if len(info['docstring']) > 80 else info['docstring']
                custom_tools.append(f"- {func_name}: {desc}")
        
        if custom_tools:
            return f"""
EXISTING CUSTOM TOOLS (Can be called directly or used as examples):
{chr(10).join(custom_tools[:15])}  # Showing first 15 tools

These tools are already implemented and can be:
1. Called directly from your generated tool using: from scripts.custom_tools.{'{tool_name}'} import {'{function_name}'}
2. Used as examples of working tool implementations
3. Combined to create more complex tools
"""
        return ""
    
    def get_module_details(self, module_name: str) -> Optional[str]:
        """Get detailed information about a specific module."""
        if module_name in self.discovered_scripts:
            info = self.discovered_scripts[module_name]
            details = [f"Module: {module_name}"]
            details.append(f"Description: {info['docstring']}")
            
            if info['classes']:
                details.append(f"Classes: {', '.join(info['classes'])}")
            
            if info['functions']:
                details.append(f"Functions: {', '.join(info['functions'])}")
            
            if info['has_async']:
                details.append("Contains async functions")
            
            return '\n'.join(details)
        return None
    
    def get_available_modules_summary(self) -> str:
        """Get a summary of all available modules for context."""
        summary = []
        summary.append(f"Total modules discovered: {len(self.discovered_scripts)}")
        
        # Count by category
        category_counts = {}
        for info in self.discovered_scripts.values():
            category = info.get('category', 'utility')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        summary.append("\nModules by category:")
        for category, count in sorted(category_counts.items()):
            summary.append(f"  - {category}: {count} modules")
        
        return '\n'.join(summary)
    
    def create_enhanced_prompt(self, name: str, description: str, 
                             requirements: str, category: str = None) -> str:
        """Create an enhanced prompt with rich context."""
        
        # Determine category from name/description if not provided
        if not category:
            if any(word in name.lower() + description.lower() 
                   for word in ['file', 'directory', 'path']):
                category = 'file_operations'
            elif any(word in name.lower() + description.lower() 
                     for word in ['analyze', 'count', 'statistics', 'report']):
                category = 'data_analysis'
            elif any(word in name.lower() + description.lower() 
                     for word in ['git', 'commit', 'branch', 'repository']):
                category = 'git_operations'
            else:
                category = 'system_operations'
        
        template = self.get_template(category)
        context = self.get_context_prompt(category.split('_')[0])
        
        prompt = f"""You are an expert Python developer creating a tool for the CWMAI system.

TASK: Generate a complete, working Python tool module.

TOOL SPECIFICATION:
- Name: {name}
- Description: {description}
- Requirements: {requirements}
- Category: {category}

{context}

{self.get_import_context()}

TEMPLATE TO FOLLOW:
{template}

COMMON PATTERNS YOU CAN USE:
{json.dumps(self.common_patterns, indent=2)}

QUALITY CHECKLIST:
✓ Proper async function definition
✓ Comprehensive error handling
✓ Input validation
✓ Type hints on all parameters
✓ Meaningful return values
✓ Clear error messages
✓ No hardcoded paths
✓ Cross-platform compatibility
✓ Memory efficient
✓ Proper logging (if needed)
✓ Idempotent operations where possible
✓ Graceful degradation on partial failures
✓ Proper resource cleanup
✓ Timeout handling for external calls
✓ Rate limiting awareness

AVOID THESE MISTAKES:
✗ Missing imports
✗ Syntax errors in strings
✗ Undefined variables
✗ Using 'self' (tools are functions, not methods)
✗ Returning None instead of dict
✗ Catching exceptions without handling
✗ Hardcoded values that should be parameters
✗ Platform-specific code without checks

Generate ONLY the Python code. Make it production-ready.
"""
        
        return prompt
    
    def generate_smart_tool(self, name: str, description: str, 
                           requirements: str) -> Dict[str, Any]:
        """Generate a tool using AI if available, otherwise fall back to templates."""
        if self.intelligent_system:
            self.logger.info(f"Using AI to generate tool: {name}")
            return self.intelligent_system.generate_intelligent_tool(
                name, description, requirements
            )
        else:
            self.logger.info(f"Using template system for tool: {name}")
            # Fall back to template-based generation
            prompt = self.create_enhanced_prompt(name, description, requirements)
            return {
                'success': True,
                'code': f"# Generated using template system\n# Prompt:\n{prompt[:200]}...",
                'analysis': None,
                'confidence': 0.5,
                'message': 'Generated using template system (AI not available)'
            }
    
    def analyze_tool_requirements(self, name: str, description: str,
                                requirements: str) -> Dict[str, Any]:
        """Analyze requirements using AI if available."""
        if self.intelligent_system:
            analysis = self.intelligent_system.analyze_requirements(
                name, description, requirements
            )
            return analysis.__dict__
        else:
            # Basic analysis without AI
            detected_ops = []
            text = f"{name} {description} {requirements}".lower()
            
            if 'file' in text or 'read' in text or 'write' in text:
                detected_ops.append('file_operations')
            if 'analyze' in text or 'data' in text:
                detected_ops.append('data_analysis')
            if 'git' in text or 'commit' in text:
                detected_ops.append('git_operations')
            
            return {
                'primary_category': self._guess_category(name, description),
                'confidence': 0.3,
                'detected_operations': detected_ops,
                'suggested_imports': ['pathlib', 'json', 'logging'],
                'similar_tools': [],
                'complexity_score': 0.5,
                'security_considerations': ['Validate all inputs'],
                'performance_requirements': {'needs_async': True}
            }
    
    def _guess_category(self, name: str, description: str) -> str:
        """Guess category from name and description."""
        text = f"{name} {description}".lower()
        
        if any(word in text for word in ['file', 'directory', 'path']):
            return 'file_operations'
        elif any(word in text for word in ['analyze', 'count', 'statistics']):
            return 'data_analysis'
        elif any(word in text for word in ['git', 'commit', 'branch']):
            return 'git_operations'
        else:
            return 'system_operations'
    
    def get_generation_insights(self) -> Dict[str, Any]:
        """Get insights about tool generation performance."""
        if self.intelligent_system:
            return self.intelligent_system.get_generation_report()
        else:
            return {
                'mode': 'template-based',
                'total_discovered_scripts': len(self.discovered_scripts),
                'available_templates': list(self.templates.keys()),
                'ai_available': False
            }
    
    def suggest_improvements(self, code: str) -> List[Dict[str, Any]]:
        """Suggest improvements for generated code."""
        suggestions = []
        
        # Basic pattern checking
        if 'try:' not in code:
            suggestions.append({
                'type': 'error_handling',
                'message': 'Add try-except blocks for error handling',
                'priority': 'high'
            })
        
        if 'async def' in code and 'await' not in code:
            suggestions.append({
                'type': 'async_usage',
                'message': 'Async function defined but no await used',
                'priority': 'medium'
            })
        
        if '__doc__' not in code and '"""' not in code:
            suggestions.append({
                'type': 'documentation',
                'message': 'Add docstrings for better documentation',
                'priority': 'medium'
            })
        
        # Use AI suggestions if available
        if self.intelligent_system and hasattr(self.intelligent_system, '_validate_and_improve'):
            try:
                # This would use the AI system's validation
                pass
            except Exception as e:
                self.logger.debug(f"Could not get AI suggestions: {e}")
        
        return suggestions


def demonstrate_template_system():
    """Demonstrate the enhanced template system with AI capabilities."""
    print("=== Smart Tool Generation Templates Demo ===\n")
    
    # Initialize with AI support
    templates = ToolGenerationTemplates(use_ai=True)
    
    # Check if AI mode is active
    if templates.intelligent_system:
        print("✓ AI-Enhanced mode active")
    else:
        print("⚠ Running in template-only mode (AI not available)")
    
    print("\n1. Analyzing Tool Requirements:")
    print("-" * 40)
    
    # Example tool requirements
    name = "smart_log_analyzer"
    description = "Analyze log files for patterns and anomalies"
    requirements = "Parse multiple formats, detect anomalies, real-time monitoring, generate insights"
    
    # Analyze requirements
    analysis = templates.analyze_tool_requirements(name, description, requirements)
    
    print(f"Tool: {name}")
    print(f"Category: {analysis['primary_category']} (confidence: {analysis['confidence']:.2f})")
    print(f"Detected Operations: {', '.join(analysis['detected_operations'])}")
    print(f"Complexity Score: {analysis['complexity_score']:.2f}")
    print(f"Suggested Imports: {', '.join(analysis['suggested_imports'][:5])}")
    
    if analysis['security_considerations']:
        print(f"Security Considerations:")
        for consideration in analysis['security_considerations']:
            print(f"  - {consideration}")
    
    print("\n2. Generating Smart Tool:")
    print("-" * 40)
    
    # Generate the tool
    result = templates.generate_smart_tool(name, description, requirements)
    
    if result['success']:
        print("✓ Tool generated successfully!")
        if result.get('confidence'):
            print(f"  Confidence: {result['confidence']:.2f}")
        if result.get('test_results'):
            print(f"  Tests: {'Passed' if result['test_results']['success'] else 'Failed'}")
        
        # Show code preview
        code = result['code']
        if len(code) > 500:
            print(f"\nGenerated Code Preview:")
            print("```python")
            print(code[:500] + "...")
            print("```")
    else:
        print(f"✗ Generation failed: {result.get('error', 'Unknown error')}")
    
    print("\n3. Tool Generation Insights:")
    print("-" * 40)
    
    insights = templates.get_generation_insights()
    
    if 'total_generations' in insights:
        print(f"Total Generations: {insights['total_generations']}")
        print(f"Success Rate: {insights.get('overall_success_rate', 0):.2%}")
        
        if 'category_performance' in insights:
            print("\nCategory Performance:")
            for category, perf in insights['category_performance'].items():
                print(f"  - {category}: {perf['success_rate']:.2%} ({perf['total_generated']} generated)")
    else:
        print(f"Mode: {insights.get('mode', 'unknown')}")
        print(f"Available Templates: {', '.join(insights.get('available_templates', []))}")
        print(f"Discovered Scripts: {insights.get('total_discovered_scripts', 0)}")
    
    print("\n4. Code Improvement Suggestions:")
    print("-" * 40)
    
    # Get improvement suggestions for sample code
    sample_code = '''
async def process_data(data):
    result = data * 2
    return result
'''
    
    suggestions = templates.suggest_improvements(sample_code)
    
    if suggestions:
        print("Suggested improvements for sample code:")
        for suggestion in suggestions:
            print(f"  [{suggestion['priority']}] {suggestion['type']}: {suggestion['message']}")
    else:
        print("No improvements suggested.")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_template_system()