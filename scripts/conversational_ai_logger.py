"""
Custom logging configuration for the Conversational AI Assistant.

This module sets up a clean logging environment that:
- Keeps the console output clean for chat interaction
- Redirects system warnings/errors to a dedicated log file
- Filters out non-essential messages from various modules
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Set


class ConversationalAIFilter(logging.Filter):
    """
    Custom filter to control which messages appear in console vs log file.
    """
    
    # Modules whose messages should be suppressed or redirected
    SYSTEM_MODULES: Set[str] = {
        'scripts.task_manager',
        'scripts.tool_calling_system',
        'scripts.ai_response_cache',
        'scripts.redis_ai_response_cache',
        'scripts.state_manager',
        'scripts.redis_async_state_manager',
        'scripts.mcp_integration',
        'scripts.mcp_client',
        'scripts.enhanced_tool_generator_integration',
        'scripts.improved_tool_generator',
        'scripts.semantic_tool_matcher',
        'scripts.tool_generator',
        'scripts.redis_connection_monitor',
        'scripts.redis_lockfree_state_manager',
        'scripts.conversational_ai_system',
        'scripts.conversational_ai_assistant',
        'scripts.dynamic_context_collector',
        'scripts.ai_brain',
        'scripts.ai_task_content_generator',
        'scripts.task_decomposition_engine',
        'scripts.hierarchical_task_manager',
        'scripts.intelligent_task_generator',
        'scripts.enhanced_work_generator',
        'urllib3.connectionpool',
        'asyncio',
        'redis',
        'redis.asyncio',
        'redis.connection',
    }
    
    # Message patterns to suppress from console
    SUPPRESSED_PATTERNS: Set[str] = {
        'AI response cache not available',
        'Error loading custom tool',
        'Redis connection error',
        'Connection pool',
        'Retry attempt',
        'Circuit breaker',
        'No custom tools directory found',
        'Failed to load tool',
        'Cache miss',
        'State not found',
        'Queue empty',
        'Worker not found',
        'No active workers',
        'Timeout waiting for',
        'Connection refused',
        'Connection reset',
        'Broken pipe',
        'Processing user input:',
        'Error reading response:',
        'TaskManager initialization:',
        'AI content generator initialized',
        'Decomposition system initialized',
        'AI executing tool:',
        'Managing',
        'GitHub token exists:',
        'Repository name:',
        'GitHub client created:',
        'Repository object created:',
        'CANCELLED:',
        'Future cancelled',
    }
    
    def __init__(self, is_console: bool = False):
        """
        Initialize the filter.
        
        Args:
            is_console: If True, this filter is for console output
        """
        super().__init__()
        self.is_console = is_console
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determine if a log record should be processed.
        
        Args:
            record: The log record to filter
            
        Returns:
            True if the record should be processed, False otherwise
        """
        # Check if this is from a system module
        is_system_module = any(
            record.name.startswith(module) 
            for module in self.SYSTEM_MODULES
        )
        
        # Check if message contains suppressed patterns
        message = record.getMessage()
        contains_suppressed = any(
            pattern in message 
            for pattern in self.SUPPRESSED_PATTERNS
        )
        
        if self.is_console:
            # For console: only show user-facing messages
            # Suppress system modules and certain patterns
            if is_system_module or contains_suppressed:
                return False
            
            # Only show WARNING and above for non-conversational modules
            if not record.name.startswith('scripts.conversational_ai'):
                return record.levelno >= logging.WARNING
            
            # Show all messages from the conversational AI module
            return True
        else:
            # For file: log everything
            return True


def setup_conversational_ai_logging(log_dir: str = "logs"):
    """
    Set up the logging configuration for the conversational AI assistant.
    
    Args:
        log_dir: Directory to store log files
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(message)s'  # Clean format for console
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create handlers
    # Console handler - clean output for user interaction
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ConversationalAIFilter(is_console=True))
    
    # File handler - detailed system logs
    log_file = os.path.join(log_dir, 'conversational_ai_system.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(ConversationalAIFilter(is_console=False))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific log levels for noisy modules
    noisy_modules = [
        'urllib3',
        'asyncio',
        'redis',
        'redis.asyncio',
        'redis.connection',
        'concurrent.futures',
        'multiprocessing',
    ]
    
    for module in noisy_modules:
        logging.getLogger(module).setLevel(logging.WARNING)
    
    # Create a logger for important user-facing messages
    user_logger = logging.getLogger('conversational_ai.user')
    user_logger.setLevel(logging.INFO)
    
    # Log initialization
    system_logger = logging.getLogger('scripts.conversational_ai_logger')
    system_logger.info(f"Conversational AI logging initialized. System logs: {log_file}")
    
    return user_logger


def get_user_logger():
    """
    Get the logger for user-facing messages.
    
    Returns:
        Logger configured for user messages
    """
    return logging.getLogger('conversational_ai.user')


def log_system_message(message: str, level = None):
    """
    Log a system message (will go to file only).
    
    Args:
        message: The message to log
        level: The logging level (can be int or string)
    """
    logger = logging.getLogger('scripts.conversational_ai_system')
    
    # Handle both string and integer levels
    if level is None:
        level = logging.INFO
    elif isinstance(level, str):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        level = level_map.get(level.upper(), logging.INFO)
    
    logger.log(level, message)


def log_user_message(message: str):
    """
    Log a user-facing message (will appear in console).
    
    Args:
        message: The message to log
    """
    logger = get_user_logger()
    logger.info(message)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    setup_conversational_ai_logging()
    
    # Test different types of messages
    print("\n=== Testing Conversational AI Logger ===\n")
    
    # User-facing messages (should appear in console)
    log_user_message("Welcome to the Conversational AI Assistant!")
    log_user_message("Type 'help' for available commands.")
    
    # System messages (should only go to file)
    log_system_message("System initialized successfully")
    log_system_message("Redis connection established")
    
    # Test suppressed messages
    task_logger = logging.getLogger('scripts.task_manager')
    task_logger.warning("AI response cache not available")  # Should be suppressed
    
    tool_logger = logging.getLogger('scripts.tool_calling_system')
    tool_logger.error("Error loading custom tool: test_tool")  # Should be suppressed
    
    # Test important warning that should show
    app_logger = logging.getLogger('scripts.conversational_ai_assistant')
    app_logger.warning("Important: API rate limit approaching")  # Should show
    
    print("\n=== Logger Test Complete ===")
    print(f"Check logs/conversational_ai_system.log for system messages")