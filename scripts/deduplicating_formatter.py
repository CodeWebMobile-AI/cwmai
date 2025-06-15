"""
Custom logging formatter with built-in deduplication.
"""
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

from scripts.log_deduplicator import LogDeduplicator


class DeduplicatingFormatter(logging.Formatter):
    """Logging formatter that deduplicates repeated messages."""
    
    def __init__(self, *args, dedup_window: int = 5, **kwargs):
        """
        Initialize the formatter.
        
        Args:
            dedup_window: Time window in seconds for deduplication
        """
        super().__init__(*args, **kwargs)
        self.deduplicator = LogDeduplicator(time_window_seconds=dedup_window)
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # Cleanup every minute
        
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with deduplication."""
        # First format normally
        original_msg = super().format(record)
        
        # Convert record time to datetime
        timestamp = datetime.fromtimestamp(record.created)
        
        # Periodic cleanup
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.deduplicator.cleanup_old_messages(timestamp)
            self.last_cleanup = current_time
        
        # Process through deduplicator
        deduped_msg = self.deduplicator.add_message(timestamp, original_msg)
        
        # Return empty string if message should be suppressed
        return deduped_msg if deduped_msg else ""


class SmartDeduplicatingHandler(logging.Handler):
    """Handler that only emits non-empty messages from DeduplicatingFormatter."""
    
    def __init__(self, base_handler: logging.Handler, dedup_window: int = 5):
        """
        Initialize the smart handler.
        
        Args:
            base_handler: The underlying handler to wrap
            dedup_window: Time window for deduplication
        """
        super().__init__()
        self.base_handler = base_handler
        self.setLevel(base_handler.level)
        
        # Create deduplicating formatter
        if base_handler.formatter:
            fmt = base_handler.formatter._fmt if hasattr(base_handler.formatter, '_fmt') else None
            datefmt = base_handler.formatter.datefmt
        else:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            datefmt = None
            
        self.formatter = DeduplicatingFormatter(fmt=fmt, datefmt=datefmt, dedup_window=dedup_window)
        
    def emit(self, record: logging.LogRecord):
        """Emit the record only if formatter returns non-empty string."""
        try:
            msg = self.formatter.format(record)
            if msg:  # Only emit non-empty messages
                # Create a copy of the record with formatted message
                new_record = logging.LogRecord(
                    record.name, record.levelno, record.pathname,
                    record.lineno, msg, (), None, record.funcName
                )
                new_record.created = record.created
                new_record.msecs = record.msecs
                new_record.relativeCreated = record.relativeCreated
                
                # Use a simple formatter for the base handler
                original_formatter = self.base_handler.formatter
                self.base_handler.formatter = logging.Formatter('%(message)s')
                self.base_handler.emit(new_record)
                self.base_handler.formatter = original_formatter
        except Exception:
            self.handleError(record)


def configure_deduplicating_logs(logger_name: Optional[str] = None, 
                                dedup_window: int = 5,
                                level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger with deduplication.
    
    Args:
        logger_name: Name of logger to configure (None for root)
        dedup_window: Time window for deduplication
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Replace existing handlers with deduplicating versions
    new_handlers = []
    for handler in logger.handlers[:]:
        smart_handler = SmartDeduplicatingHandler(handler, dedup_window=dedup_window)
        new_handlers.append(smart_handler)
        logger.removeHandler(handler)
    
    for handler in new_handlers:
        logger.addHandler(handler)
        
    return logger