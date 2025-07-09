"""
Method Discovery and Self-Healing System

Automatically discovers available methods and suggests corrections
when tools try to use non-existent methods.
"""

import inspect
import asyncio
from typing import Dict, List, Any, Optional, Callable
from difflib import get_close_matches
import re

class MethodDiscovery:
    """Discovers and suggests methods for self-healing."""
    
    def __init__(self):
        self.method_cache = {}
        
    def discover_methods(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        """Discover all methods of an object with their signatures."""
        methods = {}
        
        for name in dir(obj):
            if not name.startswith('_'):  # Skip private methods
                try:
                    attr = getattr(obj, name)
                    if callable(attr):
                        # Get method signature
                        sig = None
                        try:
                            sig = inspect.signature(attr)
                        except:
                            sig = "(*args, **kwargs)"
                            
                        methods[name] = {
                            'signature': str(sig),
                            'is_async': inspect.iscoroutinefunction(attr),
                            'doc': inspect.getdoc(attr) or "No documentation"
                        }
                except:
                    continue
                    
        return methods
    
    def find_similar_method(self, obj: Any, attempted_method: str) -> Optional[str]:
        """Find the most similar existing method to the attempted one."""
        available_methods = list(self.discover_methods(obj).keys())
        
        # Try exact match with different cases
        for method in available_methods:
            if method.lower() == attempted_method.lower():
                return method
                
        # Try close matches
        close_matches = get_close_matches(attempted_method, available_methods, n=1, cutoff=0.6)
        if close_matches:
            return close_matches[0]
            
        # Try semantic matching
        # If looking for "get_issue", suggest "list_recent_issues"
        if 'get' in attempted_method and 'issue' in attempted_method:
            for method in available_methods:
                if 'issue' in method and ('list' in method or 'fetch' in method):
                    return method
                    
        return None
    
    def generate_healing_code(self, obj_name: str, attempted_method: str, 
                            correct_method: str, original_args: Dict) -> str:
        """Generate code to use the correct method."""
        # Map common patterns
        if attempted_method.startswith('get_') and correct_method.startswith('list_'):
            # Trying to get single item but only list is available
            return f"""
# Self-healed: {attempted_method} doesn't exist, using {correct_method}
issues = await {obj_name}.{correct_method}(repo=repo, limit=10)
if issues:
    issue = issues[0]  # Get first issue
else:
    issue = None
"""
        else:
            # Generic replacement
            return f"""
# Self-healed: {attempted_method} replaced with {correct_method}
result = await {obj_name}.{correct_method}(**{original_args})
"""


# Global instance
method_discovery = MethodDiscovery()