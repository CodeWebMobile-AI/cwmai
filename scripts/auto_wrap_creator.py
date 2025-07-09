"""
Auto-Wrap Creator

Automatically wraps object creation in tools to enable intelligent method creation.
This module intercepts common object instantiation patterns and wraps them with
the intelligent proxy.
"""

import ast
import inspect
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from scripts.intelligent_method_proxy import IntelligentMethodProxy, create_intelligent_proxy

logger = logging.getLogger(__name__)


class AutoWrapCreator:
    """Automatically wraps object creation with intelligent proxies."""
    
    # Classes that should be automatically wrapped
    WRAPPED_CLASSES = {
        'MCPGitHubIssueCreator': 'scripts.mcp_github_issue_creator',
        'GitHubIssueCreator': 'scripts.github_issue_creator',
        'StateManager': 'scripts.state_manager',
        'RepositoryAnalyzer': 'scripts.repository_analyzer',
        'TaskManager': 'scripts.task_manager',
        'WorkflowOrchestrator': 'scripts.workflow_orchestrator',
    }
    
    @classmethod
    def wrap_in_tool_context(cls, obj: Any, tool_name: str, tool_file: str, user_input: str = '') -> Any:
        """Wrap an object with tool context if it's a wrappable class.
        
        Args:
            obj: The object to potentially wrap
            tool_name: Name of the tool creating this object
            tool_file: Path to the tool file
            user_input: User's original request
            
        Returns:
            Wrapped object if wrappable, otherwise the original object
        """
        class_name = obj.__class__.__name__
        
        if class_name in cls.WRAPPED_CLASSES:
            # Extract tool purpose from file
            tool_purpose = cls._extract_tool_purpose(tool_file)
            
            # Create tool context
            tool_context = {
                'tool_name': tool_name,
                'tool_purpose': tool_purpose,
                'tool_file': tool_file,
                'user_intent': user_input,
                'wrapped_class': class_name,
            }
            
            logger.info(f"Auto-wrapping {class_name} created by {tool_name}")
            return create_intelligent_proxy(obj, tool_context)
            
        return obj
    
    @classmethod
    def _extract_tool_purpose(cls, tool_file: str) -> str:
        """Extract tool purpose from docstring or description."""
        try:
            path = Path(tool_file)
            if path.exists():
                content = path.read_text()
                
                # Look for __description__ variable
                if '__description__' in content:
                    import re
                    match = re.search(r'__description__\s*=\s*["\']([^"\']+)["\']', content)
                    if match:
                        return match.group(1)
                
                # Fall back to module docstring
                tree = ast.parse(content)
                docstring = ast.get_docstring(tree)
                if docstring:
                    # Get first line
                    return docstring.split('\n')[0].strip()
                    
        except Exception as e:
            logger.debug(f"Could not extract purpose from {tool_file}: {e}")
            
        return "Unknown purpose"
    
    @classmethod
    def patch_module_before_load(cls, module, tool_name: str, tool_file: str):
        """Patch a module object before it's executed to enable auto-wrapping.
        
        This modifies the module's __dict__ to inject wrapper factories.
        """
        try:
            # Inject our wrapping logic into the module
            module.__dict__['_intelligent_proxy_enabled'] = True
            module.__dict__['_tool_name'] = tool_name
            module.__dict__['_tool_file'] = tool_file
            
            # Inject the wrapper imports
            exec("from scripts.intelligent_method_proxy import create_intelligent_proxy", module.__dict__)
            exec("from scripts.auto_wrap_creator import AutoWrapCreator", module.__dict__)
            
            # For each wrappable class, inject a factory
            for class_name, class_module_path in cls.WRAPPED_CLASSES.items():
                # Create factory code that will wrap instances
                factory_code = f'''
# Original import
from {class_module_path} import {class_name} as _{class_name}_Original

# Wrapper factory
class {class_name}:
    """Auto-wrapped {class_name} with intelligent method creation."""
    
    def __new__(cls, *args, **kwargs):
        # Create original instance
        instance = _{class_name}_Original(*args, **kwargs)
        
        # Get context from kwargs if available
        context = kwargs.get('context', {{}})
        user_input = context.get('user_input', '')
        
        # Create tool context
        tool_context = {{
            'tool_name': _tool_name,
            'tool_file': _tool_file,
            'tool_purpose': __description__ if '__description__' in globals() else 'Unknown',
            'user_intent': user_input,
            'wrapped_class': '{class_name}'
        }}
        
        # Wrap with intelligent proxy
        return create_intelligent_proxy(instance, tool_context)
'''
                
                # Execute the factory code in the module's namespace
                exec(factory_code, module.__dict__)
                
            logger.info(f"Patched module {tool_name} for intelligent method creation")
                
        except Exception as e:
            logger.warning(f"Could not patch module {tool_name}: {e}")


# Global instance
auto_wrap_creator = AutoWrapCreator()


# Alternative approach: Monkey-patch the actual classes globally
def enable_global_intelligent_wrapping():
    """Enable intelligent wrapping for all instances globally."""
    try:
        # Import and patch MCPGitHubIssueCreator
        from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
        original_init = MCPGitHubIssueCreator.__init__
        
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Mark for wrapping
            self._needs_intelligent_wrap = True
            
        MCPGitHubIssueCreator.__init__ = wrapped_init
        
        # Override __getattribute__ to wrap on first method access
        original_getattr = MCPGitHubIssueCreator.__getattribute__
        
        def wrapped_getattr(self, name):
            if name == '_needs_intelligent_wrap' or not hasattr(self, '_needs_intelligent_wrap'):
                return original_getattr(self, name)
                
            if getattr(self, '_needs_intelligent_wrap', False):
                # Remove the flag and wrap
                object.__setattr__(self, '_needs_intelligent_wrap', False)
                
                # Get context from current frame if possible
                import inspect
                frame = inspect.currentframe()
                tool_context = {
                    'tool_name': 'Unknown',
                    'tool_purpose': 'Unknown',
                    'wrapped_class': 'MCPGitHubIssueCreator'
                }
                
                # Try to find tool context from call stack
                while frame:
                    if '_tool_name' in frame.f_locals:
                        tool_context['tool_name'] = frame.f_locals['_tool_name']
                        break
                    frame = frame.f_back
                
                # Create proxy
                from scripts.intelligent_method_proxy import IntelligentMethodProxy
                proxy = IntelligentMethodProxy(self, tool_context)
                
                # Replace method access to use proxy
                return getattr(proxy, name)
                
            return original_getattr(self, name)
            
        MCPGitHubIssueCreator.__getattribute__ = wrapped_getattr
        
        logger.info("Enabled global intelligent wrapping for MCPGitHubIssueCreator")
        
    except Exception as e:
        logger.warning(f"Could not enable global intelligent wrapping: {e}")