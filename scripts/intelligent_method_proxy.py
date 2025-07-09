"""
Intelligent Method Proxy

Wraps objects to automatically create missing methods by analyzing
the class, related files, and understanding patterns.
"""

import inspect
import types
import asyncio
import re
from typing import Any, Dict, Optional, List
from pathlib import Path
import ast

from scripts.http_ai_client import HTTPAIClient
from scripts.logger import logger


class IntelligentMethodProxy:
    """Proxy that creates methods intelligently based on context."""
    
    def __init__(self, wrapped_obj, tool_context=None):
        self._wrapped = wrapped_obj
        self._created_methods = {}
        self._ai_client = HTTPAIClient()
        self._tool_context = tool_context or {}
        self._class_context = self._analyze_class_context()
        
    def __getattr__(self, name):
        # First try to get from wrapped object
        try:
            return getattr(self._wrapped, name)
        except AttributeError:
            # Method doesn't exist, create it intelligently
            return self._create_method_intelligently(name)
            
    def _analyze_class_context(self) -> Dict[str, Any]:
        """Analyze the wrapped class to understand its patterns."""
        context = {
            'class_name': self._wrapped.__class__.__name__,
            'module': self._wrapped.__class__.__module__,
            'existing_methods': {},
            'imports': [],
            'patterns': {}
        }
        
        # Get all methods
        for method_name in dir(self._wrapped):
            if not method_name.startswith('_'):
                try:
                    method = getattr(self._wrapped, method_name)
                    if callable(method):
                        sig = inspect.signature(method) if hasattr(inspect, 'signature') else None
                        context['existing_methods'][method_name] = {
                            'signature': str(sig) if sig else 'Unknown',
                            'is_async': asyncio.iscoroutinefunction(method),
                            'doc': inspect.getdoc(method) or ''
                        }
                except:
                    pass
                    
        # Analyze patterns
        context['patterns'] = self._analyze_method_patterns(context['existing_methods'])
        
        # Read the source file to get imports and implementation details
        try:
            source_file = inspect.getfile(self._wrapped.__class__)
            context['source_file'] = source_file
            context['source_analysis'] = self._analyze_source_file(source_file)
        except:
            pass
            
        return context
        
    def _analyze_method_patterns(self, methods: Dict) -> Dict:
        """Analyze patterns in existing methods."""
        patterns = {
            'list_pattern': None,
            'get_pattern': None,
            'create_pattern': None,
            'common_returns': [],
            'common_params': []
        }
        
        # Find list methods
        list_methods = [m for m in methods if m.startswith('list_')]
        if list_methods:
            # Analyze how list methods work
            patterns['list_pattern'] = {
                'methods': list_methods,
                'typically_returns': 'List[Dict]',
                'common_params': ['repo', 'limit']
            }
            
        # Find get methods
        get_methods = [m for m in methods if m.startswith('get_')]
        if get_methods:
            patterns['get_pattern'] = {
                'methods': get_methods,
                'typically_returns': 'Optional[Dict]',
                'common_params': ['repo', 'id']
            }
            
        return patterns
        
    def _analyze_source_file(self, filepath: str) -> Dict:
        """Analyze the source file for implementation patterns."""
        analysis = {
            'imports': [],
            'method_implementations': {},
            'api_calls': []
        }
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Get imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
                        
            # Look for API patterns
            if 'mcp' in filepath.lower():
                analysis['api_calls'].append('MCP GitHub API')
            if 'github' in filepath.lower():
                analysis['api_calls'].append('GitHub API')
                
        except:
            pass
            
        return analysis
        
    def _create_method_intelligently(self, method_name: str):
        """Create a method based on analysis of the class and patterns."""
        logger.info(f"Intelligently creating method {method_name} for {self._class_context['class_name']}")
        
        # Create an async wrapper that will generate and execute the method
        async def method_wrapper(*args, **kwargs):
            # Check if we already created this method
            if method_name in self._created_methods:
                return await self._created_methods[method_name](*args, **kwargs)
                
            # Generate the method implementation
            implementation = await self._generate_method_implementation(
                method_name, args, kwargs
            )
            
            if implementation:
                # Create the method in a namespace with access to self
                namespace = {
                    'self': self._wrapped,
                    'asyncio': asyncio,
                    'logger': logger,
                    'Dict': Dict,
                    'List': List,
                    'Optional': Optional,
                }
                
                # Execute the implementation
                exec(implementation, namespace)
                
                if method_name in namespace:
                    # Store the created method
                    method = namespace[method_name]
                    bound_method = types.MethodType(method, self._wrapped)
                    setattr(self._wrapped, method_name, bound_method)
                    self._created_methods[method_name] = bound_method
                    
                    # Call it
                    return await bound_method(*args, **kwargs)
                    
            raise AttributeError(f"Could not create method {method_name}")
            
        return method_wrapper
        
    async def _generate_method_implementation(self, method_name: str, 
                                           args: tuple, kwargs: dict) -> Optional[str]:
        """Generate method implementation using AI with full context."""
        
        # Extract tool context from kwargs
        tool_name = self._tool_context.get('tool_name', 'Unknown Tool')
        tool_purpose = self._tool_context.get('tool_purpose', '')
        user_intent = kwargs.get('context', {}).get('user_input', '')
        
        # Build comprehensive prompt
        prompt = f"""
You need to create a method implementation for {self._class_context['class_name']}.

TOOL CONTEXT:
- This method is being called by tool: {tool_name}
- Tool purpose: {tool_purpose}
- User's original request: {user_intent}

METHOD TO CREATE: {method_name}
CALLED WITH: args={args}, kwargs={kwargs}

CLASS CONTEXT:
- Class: {self._class_context['class_name']}
- Module: {self._class_context['module']}

EXISTING METHODS:
{self._format_methods(self._class_context['existing_methods'])}

PATTERNS DETECTED:
{self._format_patterns(self._class_context['patterns'])}

SOURCE FILE ANALYSIS:
- Uses: {', '.join(self._class_context.get('source_analysis', {}).get('api_calls', []))}
- Key imports: {', '.join(self._class_context.get('source_analysis', {}).get('imports', [])[:10])}

SPECIFIC REQUIREMENTS:
1. If creating get_issue and list_recent_issues exists:
   - Call list_recent_issues and filter/return first result
   - Handle issue_number parameter if provided
2. Follow the same async/sync pattern as existing methods
3. Use the same return types and error handling
4. Reuse existing methods where possible

Generate ONLY the method implementation (no class definition, just the method).
"""
        
        response = await self._ai_client.generate_enhanced_response(prompt)
        
        if response.get('content'):
            code = response['content']
            
            # Extract just the method
            if 'async def' in code or 'def' in code:
                # Clean up
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0]
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0]
                    
                return code.strip()
                
        return None
        
    def _format_methods(self, methods: Dict) -> str:
        """Format methods for prompt."""
        lines = []
        for name, info in methods.items():
            async_prefix = "async " if info['is_async'] else ""
            lines.append(f"- {async_prefix}{name}{info['signature']}")
            if info['doc']:
                lines.append(f"  {info['doc'].split(chr(10))[0]}")
        return '\n'.join(lines)
        
    def _format_patterns(self, patterns: Dict) -> str:
        """Format patterns for prompt."""
        lines = []
        if patterns.get('list_pattern'):
            lines.append(f"- List methods: {', '.join(patterns['list_pattern']['methods'])}")
            lines.append(f"  Typically return: {patterns['list_pattern']['typically_returns']}")
        if patterns.get('get_pattern'):
            lines.append(f"- Get methods: {', '.join(patterns['get_pattern']['methods'])}")
            lines.append(f"  Typically return: {patterns['get_pattern']['typically_returns']}")
        return '\n'.join(lines) if lines else "No clear patterns detected"


def wrap_with_intelligent_proxy(obj: Any, tool_context: Optional[Dict] = None) -> Any:
    """Wrap an object with intelligent method creation proxy.
    
    Args:
        obj: The object to wrap
        tool_context: Optional context about the calling tool
            - tool_name: Name of the tool creating this object
            - tool_purpose: What the tool is trying to accomplish
            - user_intent: Original user request
    """
    return IntelligentMethodProxy(obj, tool_context)


# Generic factory for any class that needs intelligent method creation
class IntelligentProxyFactory:
    """Factory that creates wrapped instances with intelligent method creation."""
    
    @staticmethod
    def create(class_path: str, tool_context: Optional[Dict] = None, *args, **kwargs):
        """Create an instance of any class wrapped with intelligent proxy.
        
        Args:
            class_path: Full module path to the class (e.g., 'scripts.mcp_github_issue_creator.MCPGitHubIssueCreator')
            tool_context: Context about the calling tool
            *args, **kwargs: Arguments to pass to the class constructor
        """
        # Import the class dynamically
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        
        # Create instance
        instance = cls(*args, **kwargs)
        
        # Wrap with intelligent proxy
        proxied = IntelligentMethodProxy(instance, tool_context)
        
        logger.info(f"Created proxied {class_name} instance with intelligent method creation")
        return proxied


# Convenience function for commonly used classes
def create_intelligent_proxy(obj_or_class, tool_context: Optional[Dict] = None, *args, **kwargs):
    """Create or wrap an object with intelligent method creation.
    
    Args:
        obj_or_class: Either an existing object to wrap, or a class to instantiate
        tool_context: Context about the calling tool
        *args, **kwargs: If obj_or_class is a class, these are passed to constructor
    """
    if isinstance(obj_or_class, type):
        # It's a class, instantiate it
        instance = obj_or_class(*args, **kwargs)
        return IntelligentMethodProxy(instance, tool_context)
    else:
        # It's an object, wrap it
        return IntelligentMethodProxy(obj_or_class, tool_context)