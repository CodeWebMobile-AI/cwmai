"""
Error Handler - Stub module for custom tools
"""

import logging
from typing import Any, Callable, Optional
import traceback


class ErrorHandler:
    """Simple error handler for custom tools"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: str = ""):
        """Handle an error with logging"""
        error_msg = f"Error in {context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg)
        self.logger.debug(traceback.format_exc())
        return {"error": error_msg}
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function safely with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self.handle_error(e, context=func.__name__)


# Default instance
error_handler = ErrorHandler()