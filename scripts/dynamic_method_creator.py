"""
Dynamic Method Creator

Automatically creates missing methods on classes based on usage context.
"""

import types
import inspect
import asyncio
from typing import Any, Dict, Optional, Callable
import logging

from scripts.http_ai_client import HTTPAIClient

logger = logging.getLogger(__name__)


class DynamicMethodCreator:
    """Creates methods dynamically when they don't exist."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.created_methods = {}
        
    async def create_method(self, obj: Any, method_name: str, 
                          call_context: Dict[str, Any]) -> Optional[Callable]:
        """
        Create a method dynamically based on usage context.
        
        Args:
            obj: The object that needs the method
            method_name: Name of the missing method
            call_context: Context including arguments passed, user intent, etc.
            
        Returns:
            The created method or None if creation failed
        """
        class_name = obj.__class__.__name__
        logger.info(f"Creating method {method_name} for {class_name}")
        
        # Analyze existing methods to understand patterns
        existing_methods = self._analyze_class_methods(obj)
        
        # Generate method implementation
        method_code = await self._generate_method_code(
            class_name, method_name, existing_methods, call_context
        )
        
        if not method_code:
            return None
            
        # Create the method
        try:
            # Create a namespace with necessary imports
            namespace = {
                'asyncio': asyncio,
                'logging': logging,
                'logger': logger,
                'Dict': Dict,
                'Any': Any,
                'Optional': Optional,
            }
            
            # Add any imports the object's module uses
            if hasattr(obj, '__module__'):
                module = __import__(obj.__module__, fromlist=['*'])
                for name in dir(module):
                    if not name.startswith('_'):
                        namespace[name] = getattr(module, name)
            
            # Add common imports that might be needed
            namespace['aiohttp'] = __import__('aiohttp')
            namespace['os'] = __import__('os')
            namespace['ssl'] = __import__('ssl')
            namespace['json'] = __import__('json')
            
            # Execute the method code in the namespace
            exec(method_code, namespace)
            
            # Get the created method
            if method_name in namespace:
                method = namespace[method_name]
                
                # Bind the method to the instance
                bound_method = types.MethodType(method, obj)
                
                # Add to the instance
                setattr(obj, method_name, bound_method)
                
                # Store for future reference
                self.created_methods[f"{class_name}.{method_name}"] = method_code
                
                logger.info(f"Successfully created method {method_name}")
                return bound_method
                
        except Exception as e:
            logger.error(f"Failed to create method: {e}")
            return None
    
    def _analyze_class_methods(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        """Analyze existing methods to understand patterns."""
        methods = {}
        
        for name in dir(obj):
            if not name.startswith('_'):
                try:
                    attr = getattr(obj, name)
                    if callable(attr):
                        sig = None
                        try:
                            sig = inspect.signature(attr)
                        except:
                            sig = "Unknown"
                            
                        methods[name] = {
                            'signature': str(sig),
                            'is_async': inspect.iscoroutinefunction(attr),
                            'doc': inspect.getdoc(attr) or ""
                        }
                except:
                    continue
                    
        return methods
    
    async def _generate_method_code(self, class_name: str, method_name: str,
                                   existing_methods: Dict[str, Dict[str, Any]],
                                   call_context: Dict[str, Any]) -> Optional[str]:
        """Generate method code using AI."""
        
        # Extract information from call context
        args = call_context.get('args', ())
        kwargs = call_context.get('kwargs', {})
        user_intent = call_context.get('user_intent', '')
        
        prompt = f"""
You are creating a missing method for {class_name}. Be smart and figure out what it should do.

METHOD: {method_name}
CALLED WITH: args={args}, kwargs={kwargs}  
USER'S GOAL: {user_intent}

CLASS CONTEXT:
{self._format_existing_methods(existing_methods)}

ANALYZE AND IMPLEMENT:
1. What does '{method_name}' mean? {self._infer_purpose(method_name)}
2. What parameters were passed? {self._infer_from_params(kwargs)}
3. What is the user trying to achieve? {user_intent}

CRITICAL RULES:
- NEVER mock, fake, or pretend - ALWAYS perform the real operation
- DO NOT create work items or task objects - make direct API calls
- DO NOT call execute_work_item or create_task - those create mocks
- USE the actual GitHub API or MCP client methods directly
- If closing an issue, use GitHub API PATCH to update state to 'closed'
- Research the class for mcp_hub or github client attributes

For close_issue specifically:
1. Use aiohttp to make a PATCH request to GitHub API
2. URL pattern: https://api.github.com/repos/OWNER/REPO/issues/ISSUE_NUMBER
3. Headers: Authorization with GitHub token from environment
4. Body: JSON with state = closed
5. Handle SSL issues appropriately
6. Return the actual GitHub response

For delete_issue:
1. GitHub doesn't support deleting issues via API
2. Just call self.close_issue(repo, issue_number) instead
3. Return a message explaining issues were closed not deleted

Generate ONLY the method code (async def {method_name}...) that:
- Actually performs the requested operation
- Returns real results
- Handles errors appropriately

EXAMPLE of a REAL implementation (for close_issue):
```python
async def close_issue(self, repo: str, issue_number: int, **kwargs):
    import aiohttp
    import ssl
    import os
    
    # Parse owner/repo
    if '/' in repo:
        owner, repo_name = repo.split('/', 1)
    else:
        owner = 'CodeWebMobile-AI'
        repo_name = repo
    
    url = f'https://api.github.com/repos/{{owner}}/{{repo_name}}/issues/{{issue_number}}'
    headers = {{
        'Authorization': f'token {{os.getenv("GITHUB_TOKEN")}}',
        'Accept': 'application/vnd.github.v3+json'
    }}
    
    # SSL context for HTTPS
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        data = {{'state': 'closed'}}
        async with session.patch(url, json=data, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f'Successfully closed issue #{{issue_number}} in {{repo}}')
                return result
            else:
                error = await response.text()
                raise Exception(f'Failed to close issue: {{response.status}} - {{error}}')
```

Generate similar REAL implementations, not mock/fake ones!
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        if not response:
            logger.error(f"No response from AI for {method_name}")
            return None
            
        if response.get('error'):
            logger.error(f"AI error for {method_name}: {response.get('error')}")
            return None
            
        if response.get('content'):
            # Extract just the method code
            code = response['content']
            
            # Clean up the code
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0]
            elif '```' in code:
                code = code.split('```')[1].split('```')[0]
            
            # Ensure proper formatting
            code = code.strip()
            
            # Log the generated code for debugging
            logger.info(f"Generated method code for {method_name}:\n{code}")
            
            # Basic validation
            if not code or 'def ' not in code:
                logger.error("Generated code doesn't contain a method definition")
                return None
                
            return code
            
        return None
    
    def _format_existing_methods(self, methods: Dict[str, Dict[str, Any]]) -> str:
        """Format existing methods for the prompt."""
        lines = []
        for name, info in methods.items():
            async_prefix = "async " if info['is_async'] else ""
            lines.append(f"- {async_prefix}{name}{info['signature']}")
            if info['doc']:
                lines.append(f"  {info['doc'].split(chr(10))[0]}")
        return '\n'.join(lines)
    
    def _infer_purpose(self, method_name: str) -> str:
        """Infer the purpose of a method from its name."""
        parts = method_name.split('_')
        
        if method_name.startswith('get_'):
            item = '_'.join(parts[1:])
            return f"Retrieve a single {item}"
        elif method_name.startswith('list_'):
            items = '_'.join(parts[1:])
            return f"Retrieve multiple {items}"
        elif method_name.startswith('create_'):
            item = '_'.join(parts[1:])
            return f"Create a new {item}"
        elif method_name.startswith('update_'):
            item = '_'.join(parts[1:])
            return f"Update an existing {item}"
        elif method_name.startswith('delete_'):
            item = '_'.join(parts[1:])
            return f"Delete an {item}"
        elif method_name.startswith('find_'):
            item = '_'.join(parts[1:])
            return f"Find/search for {item}"
        elif method_name.startswith('fetch_'):
            item = '_'.join(parts[1:])
            return f"Fetch/retrieve {item} data"
        else:
            return f"Perform action: {method_name.replace('_', ' ')}"
    
    def _infer_from_params(self, params: dict) -> str:
        """Infer intent from parameters."""
        clues = []
        
        if 'id' in params or 'number' in params or 'name' in params:
            clues.append("Looking for a specific item")
        if 'repo' in params or 'repository' in params:
            clues.append("Working with repository data")
        if 'issue_number' in params:
            clues.append("Working with a specific issue")
        if 'limit' in params:
            clues.append("Retrieving multiple items with limit")
        if 'filter' in params or 'query' in params:
            clues.append("Filtering or searching data")
            
        return ", ".join(clues) if clues else "General operation"
    
    def enable_dynamic_methods(self, obj: Any) -> Any:
        """
        Wrap an object to automatically create missing methods.
        
        Returns a proxy that intercepts attribute access and creates
        methods on demand.
        """
        creator = self
        
        class DynamicMethodProxy:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self._creator = creator
                
            def __getattr__(self, name):
                # First try to get from wrapped object
                try:
                    return getattr(self._wrapped, name)
                except AttributeError:
                    # Method doesn't exist, create a placeholder that will
                    # create the real method when called
                    async def method_placeholder(*args, **kwargs):
                        # Create context
                        context = {
                            'args': args,
                            'kwargs': kwargs,
                            'user_intent': kwargs.get('context', {}).get('user_input', ''),
                        }
                        
                        # Create the method
                        created_method = await self._creator.create_method(
                            self._wrapped, name, context
                        )
                        
                        if created_method:
                            # Call the newly created method
                            return await created_method(*args, **kwargs)
                        else:
                            raise AttributeError(
                                f"Could not create method '{name}' for {type(self._wrapped).__name__}"
                            )
                    
                    return method_placeholder
            
            def __setattr__(self, name, value):
                if name.startswith('_'):
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._wrapped, name, value)
                    
        return DynamicMethodProxy(obj)


# Global instance
dynamic_method_creator = DynamicMethodCreator()