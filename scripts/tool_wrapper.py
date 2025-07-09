"""
Tool Wrapper for Self-Healing (DEPRECATED)

This module is deprecated and no longer used.
"""

import functools
import asyncio
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def self_healing_tool_deprecated(func: Callable) -> Callable:
    """
    Decorator that ensures AttributeErrors propagate for self-healing.
    
    This wrapper:
    1. Executes the tool function
    2. Checks if the result contains an AttributeError message
    3. Re-raises the error to trigger self-healing
    """
    
    @functools.wraps(func)
    async def async_wrapper(**kwargs):
        try:
            result = await func(**kwargs)
            
            # Check if result contains an AttributeError that was caught
            if isinstance(result, dict) and 'error' in result:
                error_msg = result['error']
                
                # Check for dict attribute access errors
                if "'dict' object has no attribute" in error_msg:
                    # Extract attribute name
                    import re
                    match = re.search(r"'dict' object has no attribute '(\w+)'", error_msg)
                    if match:
                        attr_name = match.group(1)
                        # Re-raise to trigger self-healing
                        raise AttributeError(f"'dict' object has no attribute '{attr_name}'")
                        
                # Check for other attribute errors
                elif "object has no attribute" in error_msg:
                    # Re-raise to trigger self-healing
                    raise AttributeError(error_msg)
                    
            return result
            
        except AttributeError:
            # Let AttributeErrors propagate
            raise
        except Exception as e:
            # Other exceptions can be handled normally
            return {"error": f"An error occurred: {str(e)}"}
            
    @functools.wraps(func)
    def sync_wrapper(**kwargs):
        try:
            result = func(**kwargs)
            
            # Same checking logic for sync functions
            if isinstance(result, dict) and 'error' in result:
                error_msg = result['error']
                
                if "'dict' object has no attribute" in error_msg:
                    import re
                    match = re.search(r"'dict' object has no attribute '(\w+)'", error_msg)
                    if match:
                        attr_name = match.group(1)
                        raise AttributeError(f"'dict' object has no attribute '{attr_name}'")
                elif "object has no attribute" in error_msg:
                    raise AttributeError(error_msg)
                    
            return result
            
        except AttributeError:
            raise
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}
            
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def wrap_tool_module(module):
    """
    Wrap all tool functions in a module with self-healing capability.
    
    Args:
        module: The module containing tool functions
    """
    # TEMPORARILY DISABLED to debug issue
    return
    
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        
        # Find the main tool function (usually matches module name)
        if callable(attr) and not attr_name.startswith('_'):
            # Check if it's likely a tool function
            if attr_name in module.__name__ or attr_name == module.__name__.split('.')[-1]:
                wrapped = self_healing_tool_deprecated(attr)
                setattr(module, attr_name, wrapped)
                logger.info(f"Wrapped {attr_name} for self-healing")