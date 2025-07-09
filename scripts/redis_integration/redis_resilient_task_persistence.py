"""
Resilient Redis Task Persistence with Automatic Fallback

Enhanced task persistence that automatically handles Redis failures
by falling back to SQLite, ensuring the system continues to function
even when Redis is unavailable.
"""

import asyncio
import json
import time
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.redis_integration.redis_connection_manager import (
    RedisConnectionManager, get_connection_manager, ConnectionState
)
from scripts.task_persistence import TaskPersistence  # Import file-based fallback


@dataclass
class ResilientCompletedTask:
    """Resilient completed task record."""
    work_item_id: str
    title: str
    task_type: str
    repository: Optional[str]
    description_hash: str
    completed_at: str  # ISO format string
    issue_number: Optional[int] = None
    value_created: float = 0.0
    execution_result: Optional[str] = None  # JSON string

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        data = {
            'work_item_id': self.work_item_id,
            'title': self.title,
            'task_type': self.task_type,
            'description_hash': self.description_hash,
            'completed_at': self.completed_at,
            'value_created': str(self.value_created)
        }
        
        if self.repository:
            data['repository'] = self.repository
        if self.issue_number is not None:
            data['issue_number'] = str(self.issue_number)
        if self.execution_result:
            data['execution_result'] = self.execution_result
            
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResilientCompletedTask':
        """Create from dictionary."""
        return cls(
            work_item_id=data['work_item_id'],
            title=data['title'],
            task_type=data['task_type'],
            repository=data.get('repository'),
            description_hash=data['description_hash'],
            completed_at=data['completed_at'],
            issue_number=int(data['issue_number']) if 'issue_number' in data else None,
            value_created=float(data.get('value_created', 0.0)),
            execution_result=data.get('execution_result')
        )


class ResilientRedisTaskPersistence:
    """Resilient Redis-based task persistence with automatic fallback."""
    
    def __init__(self, namespace: str = "task_persistence", enable_file_fallback: bool = True):
        """Initialize resilient task persistence.
        
        Args:
            namespace: Redis key namespace
            enable_file_fallback: Whether to enable file-based fallback as last resort
        """
        self.namespace = namespace
        self.enable_file_fallback = enable_file_fallback
        self.logger = logging.getLogger(__name__)
        
        # Connection manager handles Redis with SQLite fallback
        self.connection_manager: Optional[RedisConnectionManager] = None
        
        # File-based fallback as last resort
        self.file_fallback: Optional[TaskPersistence] = None
        if enable_file_fallback:
            self.file_fallback = TaskPersistence()
        
        # Redis key patterns
        self.completed_tasks_key = f"{namespace}:completed"
        self.title_hashes_key = f"{namespace}:title_hashes"
        self.description_hashes_key = f"{namespace}:desc_hashes"
        self.skip_stats_key = f"{namespace}:skip_stats"
        self.skip_cooldowns_key = f"{namespace}:skip_cooldowns"
        self.task_history_key = f"{namespace}:history"
        self.stats_key = f"{namespace}:stats"
        
        # Task-specific cooldown periods (in hours)
        self.task_cooldowns = {
            'TESTING': 12,
            'FEATURE': 72,
            'BUG_FIX': 48,
            'DOCUMENTATION': 48,
            'RESEARCH': 168,
            'SYSTEM_IMPROVEMENT': 168,
            'MAINTENANCE': 72,
            'NEW_PROJECT': 720,
            'INTEGRATION': 168,
            'REPOSITORY_HEALTH': 24
        }
        
        # Configuration
        self.default_cooldown_hours = 24
        self.base_skip_cooldown = 300  # 5 minutes
        self.max_stored_tasks = 10000
        self.history_ttl = 30 * 24 * 3600  # 30 days
        
        self._initialized = False
        
        # Performance metrics
        self.metrics = {
            'redis_operations': 0,
            'fallback_operations': 0,
            'file_fallback_operations': 0,
            'total_operations': 0
        }
    
    async def _ensure_connection(self):
        """Ensure connection manager is available."""
        if not self.connection_manager:
            self.connection_manager = await get_connection_manager(enable_fallback=True)
            self._initialized = True
            
            # Log current state
            status = self.connection_manager.get_status()
            print(
                f"Task persistence initialized - State: {status['state']}, "
                f"Redis: {status['redis_connected']}, "
                f"Fallback: {status['fallback_enabled']}"
            )
    
    async def record_completed_task(self, work_item: WorkItem, execution_result: Dict[str, Any]) -> bool:
        """Record a completed task with resilient storage.
        
        Args:
            work_item: The completed work item
            execution_result: Result of task execution
            
        Returns:
            True if recorded successfully
        """
        try:
            await self._ensure_connection()
            self.metrics['total_operations'] += 1
            
            # Create task record
            completed_task = ResilientCompletedTask(
                work_item_id=work_item.id,
                title=work_item.title,
                task_type=work_item.task_type,
                repository=work_item.repository,
                description_hash=self._hash_description(work_item.description),
                completed_at=datetime.now(timezone.utc).isoformat(),
                issue_number=execution_result.get('issue_number'),
                value_created=execution_result.get('value_created', 0.0),
                execution_result=json.dumps(execution_result) if execution_result else None
            )
            
            # Check connection state
            state = self.connection_manager.state
            
            if state == ConnectionState.HEALTHY:
                self.metrics['redis_operations'] += 1
            elif state in [ConnectionState.DEGRADED, ConnectionState.FAILED]:
                self.metrics['fallback_operations'] += 1
            
            # Store completed task
            task_key = f"{self.completed_tasks_key}:{work_item.id}"
            await self.connection_manager.hset(task_key, mapping=completed_task.to_dict())
            await self.connection_manager.expire(task_key, self.history_ttl)
            
            # Add to indexes
            await self.connection_manager.sadd(self.title_hashes_key, self._hash_title(work_item.title))
            await self.connection_manager.sadd(self.description_hashes_key, completed_task.description_hash)
            
            # Add to history sorted set (score = timestamp)
            timestamp = time.time()
            await self.connection_manager.zadd(self.task_history_key, {work_item.id: timestamp})
            
            # Update stats
            await self.connection_manager.hincrby(self.stats_key, 'total_completed', 1)
            await self.connection_manager.hincrby(self.stats_key, f'type_{work_item.task_type}', 1)
            if work_item.repository:
                await self.connection_manager.hincrby(self.stats_key, f'repo_{work_item.repository}', 1)
            
            print(
                f"✅ Recorded completed task: {work_item.title} "
                f"(storage: {'Redis' if state == ConnectionState.HEALTHY else 'Fallback'})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error recording task with connection manager: {e}")
            
            # Try file-based fallback as last resort
            if self.enable_file_fallback and self.file_fallback:
                try:
                    self.metrics['file_fallback_operations'] += 1
                    result = self.file_fallback.record_completed_task(work_item, execution_result)
                    if result:
                        print(f"✅ Recorded task using file fallback: {work_item.title}")
                    return result
                except Exception as fe:
                    self.logger.error(f"❌ File fallback also failed: {fe}")
            
            return False
    
    async def record_skipped_task(self, task_title: str, reason: str = "duplicate") -> None:
        """Record a skipped task with resilient storage.
        
        Args:
            task_title: Title of the skipped task
            reason: Reason for skipping
        """
        try:
            await self._ensure_connection()
            
            skip_key = f"{self.skip_stats_key}:{self._hash_title(task_title)}"
            cooldown_key = f"{self.skip_cooldowns_key}:{self._hash_title(task_title)}"
            
            # Get current skip count
            skip_count_str = await self.connection_manager.hget(skip_key, 'count')
            skip_count = int(skip_count_str) if skip_count_str else 0
            skip_count += 1
            
            # Update skip stats
            await self.connection_manager.hset(skip_key, mapping={
                'count': str(skip_count),
                'last_skip': datetime.now(timezone.utc).isoformat(),
                'reason': reason
            })
            
            # Calculate cooldown with exponential backoff
            if skip_count > 10:
                cooldown = min(3600, self.base_skip_cooldown * (2 ** ((skip_count - 10) // 5)))
            else:
                cooldown = self.base_skip_cooldown
            
            # Set cooldown
            await self.connection_manager.set(cooldown_key, "1", ex=cooldown)
            
            if skip_count > 10:
                self.logger.warning(
                    f"⚠️ Task '{task_title}' skipped {skip_count} times. "
                    f"Cooldown increased to {cooldown // 60} minutes"
                )
            
        except Exception as e:
            self.logger.error(f"Error recording skipped task: {e}")
            
            # Use file fallback if available
            if self.enable_file_fallback and self.file_fallback:
                self.file_fallback.record_skipped_task(task_title, reason)
    
    async def is_duplicate_task(self, work_item: WorkItem) -> bool:
        """Check if a work item is a duplicate with resilient checking.
        
        Args:
            work_item: Work item to check
            
        Returns:
            True if this is a duplicate task
        """
        try:
            await self._ensure_connection()
            
            # Check skip cooldown
            cooldown_key = f"{self.skip_cooldowns_key}:{self._hash_title(work_item.title)}"
            if await self.connection_manager.exists(cooldown_key):
                self.logger.debug(f"🔄 Task in skip cooldown: {work_item.title}")
                return True
            
            # Check title duplicates
            title_hash = self._hash_title(work_item.title)
            title_exists = await self.connection_manager.exists(f"{self.title_hashes_key}:{title_hash}")
            if title_exists:
                # Check if still in cooldown by looking at recent history
                recent_tasks = await self._get_recent_tasks_by_title(work_item.title)
                for task in recent_tasks:
                    if self._is_in_cooldown(task, work_item.task_type):
                        self.logger.debug(f"🔄 Duplicate title found in cooldown: {work_item.title}")
                        return True
            
            # Check description duplicates
            description_hash = self._hash_description(work_item.description)
            desc_exists = await self.connection_manager.exists(f"{self.description_hashes_key}:{description_hash}")
            if desc_exists:
                self.logger.debug(f"🔄 Duplicate description found: {work_item.title}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking duplicate: {e}")
            
            # Use file fallback if available
            if self.enable_file_fallback and self.file_fallback:
                return self.file_fallback.is_duplicate_task(work_item)
            
            # If all else fails, assume not duplicate to avoid blocking work
            return False
    
    async def _get_recent_tasks_by_title(self, title: str) -> List[ResilientCompletedTask]:
        """Get recent tasks with similar title."""
        tasks = []
        
        # Get recent task IDs from history
        recent_ids = await self.connection_manager.zrange(
            self.task_history_key, -100, -1  # Last 100 tasks
        )
        
        for task_id in recent_ids:
            task_key = f"{self.completed_tasks_key}:{task_id}"
            task_data = await self.connection_manager.hgetall(task_key)
            
            if task_data and task_data.get('title') == title:
                tasks.append(ResilientCompletedTask.from_dict(task_data))
        
        return tasks
    
    def _is_in_cooldown(self, completed_task: ResilientCompletedTask, new_task_type: str) -> bool:
        """Check if a completed task is still in its cooldown period."""
        cooldown_hours = self.task_cooldowns.get(new_task_type, self.default_cooldown_hours)
        cooldown_time = timedelta(hours=cooldown_hours)
        
        completed_at = datetime.fromisoformat(completed_task.completed_at.replace('Z', '+00:00'))
        time_since_completion = datetime.now(timezone.utc) - completed_at
        
        return time_since_completion < cooldown_time
    
    def _hash_title(self, title: str) -> str:
        """Create a hash of a task title."""
        normalized = title.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _hash_description(self, description: str) -> str:
        """Create a hash of a task description."""
        normalized = description.lower().strip()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'for', 'to', 'of', 'in', 'on', 'at'}
        words = [word for word in normalized.split() if word not in common_words]
        normalized = ' '.join(sorted(words))
        return hashlib.md5(normalized.encode()).hexdigest()
    
    async def get_completion_stats(self) -> Dict[str, Any]:
        """Get statistics about completed tasks."""
        try:
            await self._ensure_connection()
            
            stats = await self.connection_manager.hgetall(self.stats_key)
            
            # Convert string values to appropriate types
            result = {
                'total_tasks': int(stats.get('total_completed', 0)),
                'task_types': {},
                'repositories': {},
                'connection_state': self.connection_manager.state.value,
                'storage_metrics': self.metrics
            }
            
            # Parse task type counts
            for key, value in stats.items():
                if key.startswith('type_'):
                    task_type = key.replace('type_', '')
                    result['task_types'][task_type] = int(value)
                elif key.startswith('repo_'):
                    repo = key.replace('repo_', '')
                    result['repositories'][repo] = int(value)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            
            # Use file fallback if available
            if self.enable_file_fallback and self.file_fallback:
                return self.file_fallback.get_completion_stats()
            
            return {
                'total_tasks': 0,
                'task_types': {},
                'repositories': {},
                'error': str(e)
            }
    
    async def cleanup_old_tasks(self, max_age_days: int = 30):
        """Clean up old completed tasks."""
        try:
            await self._ensure_connection()
            
            # Remove old entries from sorted set
            cutoff_timestamp = time.time() - (max_age_days * 24 * 3600)
            removed = await self.connection_manager.zremrangebyscore(
                self.task_history_key, 0, cutoff_timestamp
            )
            
            if removed:
                print(f"🧹 Cleaned up {removed} old tasks from Redis")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old tasks: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get persistence layer metrics."""
        return {
            'total_operations': self.metrics['total_operations'],
            'redis_operations': self.metrics['redis_operations'],
            'sqlite_fallback_operations': self.metrics['fallback_operations'],
            'file_fallback_operations': self.metrics['file_fallback_operations'],
            'redis_percentage': (
                (self.metrics['redis_operations'] / self.metrics['total_operations'] * 100)
                if self.metrics['total_operations'] > 0 else 0
            )
        }