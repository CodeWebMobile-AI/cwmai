"""
Hybrid State Manager

Combines SQLite for persistent state storage with in-memory queues for
high-performance work management. Drop-in replacement for Redis.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone
import threading

from scripts.sqlite_state_manager import SQLiteStateManager
from scripts.in_memory_work_queue import InMemoryWorkQueue, WorkItem, TaskPriority


class HybridStateManager:
    """
    Hybrid state manager combining SQLite and in-memory components.
    
    Features:
    - SQLite for persistent key-value, hash, set, and list operations
    - In-memory work queue for high-performance task management
    - Redis-compatible API
    - No connection limits
    - Automatic persistence
    """
    
    def __init__(self, db_path: str = "cwmai_state.db", 
                 queue_persistence_file: str = "work_queue_backup.json",
                 enable_queue_persistence: bool = True):
        """Initialize hybrid state manager.
        
        Args:
            db_path: Path to SQLite database
            queue_persistence_file: File for queue persistence
            enable_queue_persistence: Enable queue persistence
        """
        self.logger = logging.getLogger(__name__)
        
        # SQLite for persistent storage
        self.sqlite = SQLiteStateManager(db_path)
        
        # In-memory work queue
        self.work_queue = InMemoryWorkQueue(
            enable_persistence=enable_queue_persistence,
            persistence_file=queue_persistence_file
        )
        
        # Track active components
        self.components = {
            'sqlite': True,
            'work_queue': True
        }
        
        self.logger.info("Hybrid state manager initialized")
    
    # ========== Work Queue Methods ==========
    
    def enqueue_work(self, work_item: Dict[str, Any]) -> bool:
        """Add work item to queue.
        
        Args:
            work_item: Work item dictionary
            
        Returns:
            True if successful
        """
        return self.work_queue.enqueue(work_item)
    
    def dequeue_work(self, timeout: Optional[float] = None, worker_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get next work item from queue.
        
        Args:
            timeout: Timeout in seconds
            worker_id: ID of the worker claiming this task
            
        Returns:
            Work item or None
        """
        return self.work_queue.dequeue(timeout, worker_id)
    
    def complete_work(self, item_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark work item as completed.
        
        Args:
            item_id: Work item ID
            result: Optional result data
            
        Returns:
            True if successful
        """
        return self.work_queue.complete(item_id, result)
    
    def fail_work(self, item_id: str, error: str, retry: bool = True) -> bool:
        """Mark work item as failed.
        
        Args:
            item_id: Work item ID
            error: Error message
            retry: Whether to retry
            
        Returns:
            True if successful
        """
        return self.work_queue.fail(item_id, error, retry)
    
    def get_work_stats(self) -> Dict[str, Any]:
        """Get work queue statistics."""
        return self.work_queue.get_stats()
    
    def get_processing_work(self) -> List[Dict[str, Any]]:
        """Get list of work being processed."""
        return self.work_queue.get_processing()
    
    def record_in_progress_task(self, task_id: str, metadata: Dict[str, Any]) -> bool:
        """Record a task that is in progress (e.g., queued for Claude).
        
        Args:
            task_id: Task ID
            metadata: Task metadata (claude_task_id, issue_number, etc.)
            
        Returns:
            True if successful
        """
        try:
            # Store in-progress task metadata
            self.hset('in_progress_tasks', task_id, json.dumps(metadata))
            
            # Also store by Claude task ID for reverse lookup
            if metadata.get('claude_task_id'):
                self.hset('claude_to_work_item', metadata['claude_task_id'], task_id)
            
            self.logger.info(f"Recorded in-progress task: {task_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error recording in-progress task: {e}")
            return False
    
    def complete_in_progress_task(self, task_id: str) -> bool:
        """Mark an in-progress task as completed.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if successful
        """
        try:
            # Get metadata before removing
            metadata_str = self.hget('in_progress_tasks', task_id)
            if metadata_str:
                metadata = json.loads(metadata_str)
                
                # Remove from in-progress
                self.hdel('in_progress_tasks', task_id)
                
                # Remove Claude task mapping
                if metadata.get('claude_task_id'):
                    self.hdel('claude_to_work_item', metadata['claude_task_id'])
                
                self.logger.info(f"Completed in-progress task: {task_id}")
                return True
            else:
                self.logger.warning(f"No in-progress task found: {task_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error completing in-progress task: {e}")
            return False
    
    def get_work_item_by_claude_task(self, claude_task_id: str) -> Optional[str]:
        """Get work item ID by Claude task ID.
        
        Args:
            claude_task_id: Claude task ID
            
        Returns:
            Work item ID or None
        """
        return self.hget('claude_to_work_item', claude_task_id)
    
    # ========== Redis-Compatible String Operations ==========
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        return self.sqlite.get(key)
    
    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set key-value pair with optional expiry."""
        return self.sqlite.set(key, value, ex)
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        return self.sqlite.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.sqlite.exists(key)
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on key."""
        return self.sqlite.expire(key, seconds)
    
    # ========== Redis-Compatible Hash Operations ==========
    
    def hset(self, name: str, key: str, value: str) -> int:
        """Set hash field."""
        return self.sqlite.hset(name, key, value)
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field."""
        return self.sqlite.hget(name, key)
    
    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        return self.sqlite.hgetall(name)
    
    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        return self.sqlite.hdel(name, *keys)
    
    # ========== Redis-Compatible Set Operations ==========
    
    def sadd(self, name: str, *values: str) -> int:
        """Add members to set."""
        return self.sqlite.sadd(name, *values)
    
    def srem(self, name: str, *values: str) -> int:
        """Remove members from set."""
        return self.sqlite.srem(name, *values)
    
    def smembers(self, name: str) -> Set[str]:
        """Get all set members."""
        return self.sqlite.smembers(name)
    
    def sismember(self, name: str, value: str) -> bool:
        """Check if value is in set."""
        return self.sqlite.sismember(name, value)
    
    # ========== Redis-Compatible List Operations ==========
    
    def lpush(self, name: str, *values: str) -> int:
        """Push values to list head."""
        return self.sqlite.lpush(name, *values)
    
    def rpush(self, name: str, *values: str) -> int:
        """Push values to list tail."""
        return self.sqlite.rpush(name, *values)
    
    def lpop(self, name: str) -> Optional[str]:
        """Pop from list head."""
        return self.sqlite.lpop(name)
    
    def llen(self, name: str) -> int:
        """Get list length."""
        return self.sqlite.llen(name)
    
    # ========== Utility Methods ==========
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        return self.sqlite.keys(pattern)
    
    def flushdb(self) -> bool:
        """Clear all data."""
        sqlite_result = self.sqlite.flushdb()
        queue_result = self.work_queue.clear()
        return sqlite_result and queue_result
    
    # ========== JSON Convenience Methods ==========
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get and parse JSON value."""
        return self.sqlite.get_json(key)
    
    def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set value as JSON."""
        return self.sqlite.set_json(key, value, ex)
    
    # ========== State Manager Specific Methods ==========
    
    def save_state(self, state: Dict[str, Any]) -> bool:
        """Save complete system state.
        
        Args:
            state: State dictionary
            
        Returns:
            True if successful
        """
        try:
            # Save each component separately for better organization
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Core state
            self.set_json('system_state', state, ex=None)
            self.set('system_state_updated', timestamp)
            
            # Save components
            for key, value in state.items():
                if isinstance(value, (dict, list)):
                    self.set_json(f'state:{key}', value)
                else:
                    self.set(f'state:{key}', str(value))
            
            self.logger.info("System state saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False
    
    def load_state(self) -> Dict[str, Any]:
        """Load complete system state.
        
        Returns:
            State dictionary
        """
        try:
            # Try to load complete state first
            state = self.get_json('system_state')
            if state:
                return state
            
            # Otherwise reconstruct from components
            state = {}
            
            # Get all state: keys
            state_keys = [k for k in self.keys('state:*')]
            
            for key in state_keys:
                # Extract component name
                component = key.replace('state:', '')
                
                # Try JSON first
                value = self.get_json(key)
                if value is None:
                    value = self.get(key)
                
                if value is not None:
                    state[component] = value
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of hybrid system.
        
        Returns:
            Status dictionary
        """
        status = {
            'type': 'hybrid',
            'components': self.components,
            'sqlite': {
                'db_path': self.sqlite.db_path,
                'key_count': len(self.sqlite.keys())
            },
            'work_queue': self.work_queue.get_stats(),
            'last_updated': self.get('system_state_updated')
        }
        
        return status
    
    def close(self):
        """Close all components gracefully."""
        try:
            # Close SQLite
            self.sqlite.close()
            
            # Save work queue state
            self.work_queue.close()
            
            self.logger.info("Hybrid state manager closed")
            
        except Exception as e:
            self.logger.error(f"Error closing hybrid state manager: {e}")


# ========== Drop-in Redis Replacement ==========

class RedisReplacement(HybridStateManager):
    """
    Drop-in replacement for Redis client with same API.
    
    This allows existing code using redis-py to work without modification.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract relevant kwargs
        db_path = kwargs.pop('db', 'cwmai_state.db')
        super().__init__(db_path=db_path)
        
        # Mimic Redis connection info
        self.connection_pool = type('MockPool', (), {'max_connections': 9999})()
    
    # Redis compatibility methods
    
    def ping(self) -> bool:
        """Redis PING command."""
        return True
    
    def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Redis INFO command."""
        return {
            'server': {
                'redis_version': 'SQLite-Hybrid-1.0',
                'uptime_in_seconds': 0
            },
            'clients': {
                'connected_clients': 1,
                'maxclients': 9999
            },
            'memory': {
                'used_memory': 0,
                'used_memory_human': '0B'
            }
        }
    
    def ttl(self, key: str) -> int:
        """Get TTL of key (not implemented, returns -1)."""
        return -1
    
    def mget(self, keys: List[str]) -> List[Optional[str]]:
        """Get multiple keys."""
        return [self.get(k) for k in keys]
    
    def mset(self, mapping: Dict[str, str]) -> bool:
        """Set multiple keys."""
        for k, v in mapping.items():
            self.set(k, v)
        return True


def create_redis_replacement(*args, **kwargs) -> RedisReplacement:
    """
    Factory function to create Redis replacement.
    
    Can be used as drop-in for redis.Redis:
    
    Example:
        # Before:
        import redis
        client = redis.Redis()
        
        # After:
        from scripts.hybrid_state_manager import create_redis_replacement
        client = create_redis_replacement()
    """
    return RedisReplacement(*args, **kwargs)