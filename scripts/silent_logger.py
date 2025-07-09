"""
Silent logger configuration that suppresses ALL console output.
Only AI responses should appear in the terminal.
"""

import sys
import os
import logging
import logging.handlers
from contextlib import contextmanager

class SilentFilter(logging.Filter):
    """Filter that blocks ALL log records from console."""
    def filter(self, record):
        return False

class LogCapture:
    """Captures all stdout/stderr output to log file."""
    
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        
    def write(self, message):
        # Write to log file only, not terminal
        if hasattr(self.log_file, 'write'):
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        if hasattr(self.log_file, 'flush'):
            self.log_file.flush()
    
    def isatty(self):
        """Return False since this is not a terminal."""
        return False
    
    def fileno(self):
        """Return the file descriptor if available."""
        if hasattr(self.log_file, 'fileno'):
            return self.log_file.fileno()
        return 0

def setup_silent_logging():
    """Set up completely silent logging - ALL output goes to file only."""
    
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Open log file for all output
    log_file_path = os.path.join(log_dir, "conversational_ai_system.log")
    log_file = open(log_file_path, 'a', encoding='utf-8')
    
    # Redirect stdout and stderr to log file
    sys.stdout = LogCapture(log_file)
    sys.stderr = LogCapture(log_file)
    
    # Configure logging to suppress ALL console output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove ALL existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add only file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Suppress specific noisy loggers
    for logger_name in ['urllib3', 'asyncio', 'redis', 'concurrent', 'multiprocessing']:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)
    
    return log_file

@contextmanager
def temporary_stdout():
    """Temporarily restore stdout for AI responses."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Get the real terminal stdout
    if hasattr(original_stdout, 'terminal'):
        sys.stdout = original_stdout.terminal
    else:
        sys.stdout = sys.__stdout__
    
    sys.stderr = sys.__stderr__
    
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr