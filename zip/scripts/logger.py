"""
Logger - Stub module for custom tools
"""

import logging
import sys


def get_logger(name: str = __name__) -> logging.Logger:
    """Get a configured logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


# Default logger instance
logger = get_logger("cwmai")

# For compatibility with custom tools expecting Logger class
class Logger:
    """Logger class for backward compatibility"""
    
    def __init__(self, name: str = "cwmai"):
        self.logger = get_logger(name)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def critical(self, msg: str):
        self.logger.critical(msg)