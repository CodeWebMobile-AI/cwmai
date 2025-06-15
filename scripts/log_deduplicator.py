"""
Log deduplication utility for grouping repeated log messages.
"""
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class LogDeduplicator:
    """Groups and deduplicates repeated log messages."""
    
    def __init__(self, time_window_seconds: int = 5):
        """
        Initialize the log deduplicator.
        
        Args:
            time_window_seconds: Time window for grouping similar messages
        """
        self.time_window = timedelta(seconds=time_window_seconds)
        self.message_groups: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self.last_output_time: Dict[str, datetime] = {}
        
    def normalize_message(self, message: str) -> str:
        """
        Normalize a log message by removing variable parts.
        
        Args:
            message: Original log message
            
        Returns:
            Normalized message key
        """
        # Remove timestamps
        message = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', '', message)
        
        # Remove specific IDs and UUIDs
        message = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', 'UUID', message)
        message = re.sub(r'worker_\d+', 'worker_N', message)
        
        # Remove numbers that might be counters or measurements
        message = re.sub(r'\b\d+\.\d+\b', 'X.X', message)
        message = re.sub(r'\b\d+\b', 'N', message)
        
        # Remove extra whitespace
        message = ' '.join(message.split())
        
        return message
    
    def add_message(self, timestamp: datetime, raw_message: str) -> Optional[str]:
        """
        Add a message and return deduplicated output if needed.
        
        Args:
            timestamp: Message timestamp
            raw_message: Raw log message
            
        Returns:
            Formatted output if message should be displayed, None otherwise
        """
        normalized = self.normalize_message(raw_message)
        
        # Add to group
        self.message_groups[normalized].append((timestamp, raw_message))
        
        # Check if we should output
        last_output = self.last_output_time.get(normalized)
        if last_output is None or timestamp - last_output > self.time_window:
            # Count recent occurrences
            recent_messages = [
                (ts, msg) for ts, msg in self.message_groups[normalized]
                if timestamp - ts <= self.time_window
            ]
            
            if len(recent_messages) > 1:
                # Output with count
                self.last_output_time[normalized] = timestamp
                first_msg = recent_messages[0][1]
                return f"{first_msg} [×{len(recent_messages)}]"
            else:
                # Single occurrence
                self.last_output_time[normalized] = timestamp
                return raw_message
                
        return None
    
    def cleanup_old_messages(self, current_time: datetime):
        """Remove old messages outside the time window."""
        for key in list(self.message_groups.keys()):
            self.message_groups[key] = [
                (ts, msg) for ts, msg in self.message_groups[key]
                if current_time - ts <= self.time_window * 2
            ]
            if not self.message_groups[key]:
                del self.message_groups[key]
                if key in self.last_output_time:
                    del self.last_output_time[key]