"""
API Context Provider

Provides comprehensive context about available APIs and methods
to prevent non-existent method errors during tool creation.
"""

import inspect
import importlib
from typing import Dict, List, Any, Set
from pathlib import Path

class APIContextProvider:
    """Provides rich context about available APIs."""
    
    def __init__(self):
        self.api_registry = {}
        self._scan_available_apis()
        
    def _scan_available_apis(self):
        """Scan and register all available APIs."""
        # Common modules with useful APIs
        modules_to_scan = [
            'scripts.mcp_github_issue_creator',
            'scripts.state_manager',
            'scripts.repository_analyzer',
            'scripts.task_manager',
            'scripts.workflow_orchestrator',
            'scripts.github_issue_creator',
        ]
        
        for module_name in modules_to_scan:
            try:
                module = importlib.import_module(module_name)
                self._register_module_apis(module_name, module)
            except:
                pass
                
    def _register_module_apis(self, module_name: str, module: Any):
        """Register APIs from a module."""
        for name in dir(module):
            if name.startswith('_'):
                continue
                
            obj = getattr(module, name)
            if inspect.isclass(obj):
                # Register class and its methods
                methods = {}
                for method_name in dir(obj):
                    if not method_name.startswith('_') or method_name == '__init__':
                        try:
                            method = getattr(obj, method_name)
                            if callable(method):
                                sig = inspect.signature(method) if hasattr(method, '__func__') else None
                                methods[method_name] = {
                                    'signature': str(sig) if sig else 'Unknown',
                                    'is_async': inspect.iscoroutinefunction(method)
                                }
                        except:
                            pass
                            
                self.api_registry[f"{module_name}.{name}"] = {
                    'type': 'class',
                    'methods': methods
                }
                
    def get_context_for_tool_creation(self) -> str:
        """Get comprehensive context for tool creation."""
        lines = ["COMPLETE API REFERENCE:\n"]
        
        # Group by module
        by_module = {}
        for full_name, info in self.api_registry.items():
            module = full_name.split('.')[1]  # Extract module name
            if module not in by_module:
                by_module[module] = []
            by_module[module].append((full_name, info))
            
        # Format each module
        for module, classes in by_module.items():
            lines.append(f"\n=== {module.upper()} MODULE ===")
            
            for full_name, info in classes:
                class_name = full_name.split('.')[-1]
                lines.append(f"\n{class_name}:")
                
                # Show important methods
                important_methods = []
                for method_name, method_info in info['methods'].items():
                    if method_name == '__init__':
                        continue
                    # Prioritize certain method patterns
                    if any(pattern in method_name for pattern in ['get', 'list', 'create', 'update', 'delete']):
                        async_prefix = "await " if method_info['is_async'] else ""
                        important_methods.append(f"  - {async_prefix}{method_name}{method_info['signature']}")
                        
                # Show first 10 important methods
                lines.extend(important_methods[:10])
                if len(important_methods) > 10:
                    lines.append(f"  ... and {len(important_methods) - 10} more methods")
                    
        # Add specific examples
        lines.append("\n\nCOMMON PATTERNS:")
        lines.append("- To get a single issue when only list_recent_issues exists:")
        lines.append("  issues = await mcp_creator.list_recent_issues(repo=repo, limit=1)")
        lines.append("  issue = issues[0] if issues else None")
        lines.append("\n- To get project info:")
        lines.append("  state = state_manager.get_state()")
        lines.append("  project = state.get('projects', {}).get(project_name)")
        
        return '\n'.join(lines)
        
    def get_method_alternatives(self, class_name: str, missing_method: str) -> List[str]:
        """Get alternative methods when one doesn't exist."""
        alternatives = []
        
        # Find the class in registry
        for full_name, info in self.api_registry.items():
            if full_name.endswith(class_name):
                methods = info['methods']
                
                # Look for similar methods
                if 'get_' in missing_method and any('list_' in m for m in methods):
                    list_methods = [m for m in methods if 'list_' in m]
                    alternatives.extend(list_methods)
                    
                # Look for semantically similar
                for method in methods:
                    if any(word in method for word in missing_method.split('_') if len(word) > 2):
                        alternatives.append(method)
                        
        return list(set(alternatives))


# Global instance
api_context_provider = APIContextProvider()