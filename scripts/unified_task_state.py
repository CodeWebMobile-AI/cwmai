"""
Unified Task State Management System

Provides centralized task state management to prevent duplicates and ensure
consistency across system restarts. Integrates with all task-related components.
"""

import json
import os
import threading
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import logging

from scripts.work_item_types import WorkItem, TaskPriority


class UnifiedTaskState:
    """Centralized task state management system."""
    
    def __init__(self, state_dir: str = "local_repos/state"):
        """Initialize unified task state manager.
        
        Args:
            state_dir: Directory for state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # State files
        self.active_tasks_file = self.state_dir / "active_tasks.json"
        self.completed_tasks_file = self.state_dir / "completed_tasks.json"
        self.task_history_file = self.state_dir / "task_history.json"
        self.task_hashes_file = self.state_dir / "task_hashes.json"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory state
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_hashes: Dict[str, str] = {}  # hash -> task_id
        self.cooldown_periods: Dict[str, datetime] = {}  # task_id -> cooldown_until
        
        # Configuration
        self.default_cooldown_hours = {
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
        
        # Load existing state
        self._load_state()
    
    def calculate_task_hash(self, work_item: WorkItem) -> str:
        """Calculate a hash for task deduplication.
        
        Args:
            work_item: Work item to hash
            
        Returns:
            Hash string
        """
        # Create hash based on key attributes
        hash_parts = [
            work_item.task_type,
            work_item.repository or '',
            work_item.title.lower().strip(),
            str(work_item.metadata.get('issue_number', ''))
        ]
        
        hash_string = '|'.join(hash_parts)
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def is_duplicate_task(self, work_item: WorkItem) -> Tuple[bool, Optional[str]]:
        """Check if a task is a duplicate.
        
        Args:
            work_item: Work item to check
            
        Returns:
            Tuple of (is_duplicate, reason)
        """
        with self._lock:
            # Calculate hash
            task_hash = self.calculate_task_hash(work_item)
            
            # Check if hash exists
            if task_hash in self.task_hashes:
                existing_id = self.task_hashes[task_hash]
                
                # Check if task is active
                if existing_id in self.active_tasks:
                    return True, f"Task already active: {existing_id}"
                
                # Check if task is in cooldown
                if existing_id in self.cooldown_periods:
                    cooldown_until = self.cooldown_periods[existing_id]
                    if datetime.now(timezone.utc) < cooldown_until:
                        remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 3600
                        return True, f"Task in cooldown for {remaining:.1f} hours"
                
                # Check completed tasks
                if existing_id in self.completed_tasks:
                    completed_task = self.completed_tasks[existing_id]
                    completed_at = datetime.fromisoformat(completed_task['completed_at'])
                    
                    # Apply task-specific cooldown
                    cooldown_hours = self.default_cooldown_hours.get(
                        work_item.task_type, 
                        24  # Default 24 hours
                    )
                    
                    time_since = datetime.now(timezone.utc) - completed_at
                    if time_since < timedelta(hours=cooldown_hours):
                        remaining = cooldown_hours - (time_since.total_seconds() / 3600)
                        return True, f"Task completed {time_since.total_seconds()/3600:.1f} hours ago, cooldown: {remaining:.1f} hours"
            
            # Check for semantic duplicates in active tasks
            for task_id, task_data in self.active_tasks.items():
                if self._is_semantically_similar(work_item, task_data):
                    return True, f"Similar active task exists: {task_data['title']}"
            
            return False, None
    
    def register_task(self, work_item: WorkItem, stage: str = "pending") -> bool:
        """Register a new task in the system.
        
        Args:
            work_item: Work item to register
            stage: Initial stage (pending, queued, executing, etc.)
            
        Returns:
            True if registered successfully
        """
        with self._lock:
            # Check for duplicates first
            is_dup, reason = self.is_duplicate_task(work_item)
            if is_dup:
                self.logger.warning(f"Duplicate task rejected: {reason}")
                return False
            
            # Calculate hash
            task_hash = self.calculate_task_hash(work_item)
            
            # Create task record
            task_data = {
                'id': work_item.id,
                'task_type': work_item.task_type,
                'title': work_item.title,
                'description': work_item.description,
                'repository': work_item.repository,
                'priority': work_item.priority.value if hasattr(work_item.priority, 'value') else work_item.priority,
                'stage': stage,
                'registered_at': datetime.now(timezone.utc).isoformat(),
                'metadata': work_item.metadata
            }
            
            # Register task
            self.active_tasks[work_item.id] = task_data
            self.task_hashes[task_hash] = work_item.id
            
            # Save state
            self._save_state()
            
            self.logger.info(f"Registered task: {work_item.id} - {work_item.title}")
            return True
    
    def update_task_stage(self, task_id: str, stage: str, metadata: Optional[Dict] = None) -> bool:
        """Update task stage/status.
        
        Args:
            task_id: Task ID
            stage: New stage
            metadata: Optional metadata updates
            
        Returns:
            True if updated successfully
        """
        with self._lock:
            if task_id not in self.active_tasks:
                self.logger.warning(f"Task not found: {task_id}")
                return False
            
            self.active_tasks[task_id]['stage'] = stage
            self.active_tasks[task_id]['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            if metadata:
                self.active_tasks[task_id]['metadata'].update(metadata)
            
            # Save state
            self._save_state()
            
            return True
    
    def complete_task(self, task_id: str, result: Optional[Dict] = None) -> bool:
        """Mark a task as completed.
        
        Args:
            task_id: Task ID
            result: Completion result data
            
        Returns:
            True if completed successfully
        """
        with self._lock:
            if task_id not in self.active_tasks:
                self.logger.warning(f"Task not found: {task_id}")
                return False
            
            task_data = self.active_tasks[task_id]
            
            # Create completion record
            completion_data = {
                **task_data,
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'result': result or {}
            }
            
            # Move to completed
            self.completed_tasks[task_id] = completion_data
            del self.active_tasks[task_id]
            
            # Set cooldown
            cooldown_hours = self.default_cooldown_hours.get(task_data['task_type'], 24)
            self.cooldown_periods[task_id] = datetime.now(timezone.utc) + timedelta(hours=cooldown_hours)
            
            # Append to history
            self._append_to_history(completion_data)
            
            # Save state
            self._save_state()
            
            self.logger.info(f"Completed task: {task_id} - {task_data['title']}")
            return True
    
    def fail_task(self, task_id: str, error: str, retry: bool = True) -> bool:
        """Mark a task as failed.
        
        Args:
            task_id: Task ID
            error: Error message
            retry: Whether to allow retry
            
        Returns:
            True if marked as failed
        """
        with self._lock:
            if task_id not in self.active_tasks:
                self.logger.warning(f"Task not found: {task_id}")
                return False
            
            task_data = self.active_tasks[task_id]
            task_data['last_error'] = error
            task_data['fail_count'] = task_data.get('fail_count', 0) + 1
            
            if not retry or task_data['fail_count'] >= 3:
                # Move to completed with failed status
                completion_data = {
                    **task_data,
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'failed',
                    'error': error
                }
                
                self.completed_tasks[task_id] = completion_data
                del self.active_tasks[task_id]
                
                # Set shorter cooldown for failed tasks
                self.cooldown_periods[task_id] = datetime.now(timezone.utc) + timedelta(hours=6)
            else:
                # Keep in active but update stage
                task_data['stage'] = 'retry_pending'
                task_data['retry_after'] = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
            
            # Save state
            self._save_state()
            
            return True
    
    def get_active_tasks(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of active tasks.
        
        Args:
            task_type: Optional filter by task type
            
        Returns:
            List of active tasks
        """
        with self._lock:
            tasks = list(self.active_tasks.values())
            
            if task_type:
                tasks = [t for t in tasks if t['task_type'] == task_type]
            
            return tasks
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            # Count by stage
            stages = {}
            for task in self.active_tasks.values():
                stage = task['stage']
                stages[stage] = stages.get(stage, 0) + 1
            
            # Count by type
            types = {}
            for task in self.active_tasks.values():
                task_type = task['task_type']
                types[task_type] = types.get(task_type, 0) + 1
            
            return {
                'total_active': len(self.active_tasks),
                'total_completed': len(self.completed_tasks),
                'by_stage': stages,
                'by_type': types,
                'in_cooldown': len(self.cooldown_periods)
            }
    
    def cleanup_old_state(self, days: int = 30):
        """Clean up old completed tasks and expired cooldowns.
        
        Args:
            days: Number of days to keep
        """
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Clean completed tasks
            tasks_to_remove = []
            for task_id, task_data in self.completed_tasks.items():
                completed_at = datetime.fromisoformat(task_data['completed_at'])
                if completed_at < cutoff:
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.completed_tasks[task_id]
            
            # Clean expired cooldowns
            current_time = datetime.now(timezone.utc)
            expired_cooldowns = []
            for task_id, cooldown_until in self.cooldown_periods.items():
                if cooldown_until < current_time:
                    expired_cooldowns.append(task_id)
            
            for task_id in expired_cooldowns:
                del self.cooldown_periods[task_id]
            
            # Save state
            self._save_state()
            
            self.logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks and {len(expired_cooldowns)} expired cooldowns")
    
    def _is_semantically_similar(self, work_item: WorkItem, task_data: Dict[str, Any]) -> bool:
        """Check if two tasks are semantically similar.
        
        Args:
            work_item: New work item
            task_data: Existing task data
            
        Returns:
            True if similar
        """
        # Same type and repository
        if (work_item.task_type == task_data['task_type'] and 
            work_item.repository == task_data.get('repository')):
            
            # Calculate title similarity
            words1 = set(work_item.title.lower().split())
            words2 = set(task_data['title'].lower().split())
            
            if not words1 or not words2:
                return False
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            similarity = len(intersection) / len(union)
            return similarity > 0.7
        
        return False
    
    def _load_state(self):
        """Load state from files."""
        # Load active tasks
        if self.active_tasks_file.exists():
            try:
                with open(self.active_tasks_file, 'r') as f:
                    data = json.load(f)
                    # Handle different formats
                    if isinstance(data, dict) and 'tasks' in data:
                        # Convert list to dict
                        self.active_tasks = {}
                        for task in data['tasks']:
                            if isinstance(task, dict) and 'work_item_id' in task:
                                # Use work_item_id as the key
                                self.active_tasks[task['work_item_id']] = {
                                    'id': task['work_item_id'],
                                    'task_type': task.get('status', 'assigned'),
                                    'title': task.get('title', ''),
                                    'repository': task.get('repository', ''),
                                    'issue_number': task.get('issue_number'),
                                    'created_at': task.get('created_at'),
                                    'status': task.get('status', 'assigned')
                                }
                    else:
                        self.active_tasks = data
            except Exception as e:
                self.logger.error(f"Error loading active tasks: {e}")
        
        # Load completed tasks
        if self.completed_tasks_file.exists():
            try:
                with open(self.completed_tasks_file, 'r') as f:
                    data = json.load(f)
                    # Handle both formats
                    if isinstance(data, dict) and 'completed_tasks' in data:
                        # Convert list to dict
                        for task in data['completed_tasks']:
                            if isinstance(task, dict) and 'id' in task:
                                self.completed_tasks[task['id']] = task
                    else:
                        self.completed_tasks = data
            except Exception as e:
                self.logger.error(f"Error loading completed tasks: {e}")
        
        # Load task hashes
        if self.task_hashes_file.exists():
            try:
                with open(self.task_hashes_file, 'r') as f:
                    self.task_hashes = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading task hashes: {e}")
        
        # Rebuild hashes if needed
        self._rebuild_hashes()
    
    def _save_state(self):
        """Save state to files."""
        try:
            # Save active tasks
            with open(self.active_tasks_file, 'w') as f:
                json.dump(self.active_tasks, f, indent=2)
            
            # Save completed tasks
            with open(self.completed_tasks_file, 'w') as f:
                json.dump(self.completed_tasks, f, indent=2)
            
            # Save task hashes
            with open(self.task_hashes_file, 'w') as f:
                json.dump(self.task_hashes, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def _rebuild_hashes(self):
        """Rebuild task hashes from active and completed tasks."""
        for task_data in self.active_tasks.values():
            # Create a temporary WorkItem to calculate hash
            work_item = WorkItem(
                id=task_data['id'],
                task_type=task_data['task_type'],
                title=task_data['title'],
                description=task_data.get('description', ''),
                repository=task_data.get('repository'),
                priority=TaskPriority.MEDIUM  # Not used in hash
            )
            task_hash = self.calculate_task_hash(work_item)
            self.task_hashes[task_hash] = task_data['id']
    
    def _append_to_history(self, task_data: Dict[str, Any]):
        """Append task to history file."""
        try:
            history = []
            if self.task_history_file.exists():
                with open(self.task_history_file, 'r') as f:
                    history = json.load(f)
            
            history.append(task_data)
            
            # Keep only last 1000 entries
            if len(history) > 1000:
                history = history[-1000:]
            
            with open(self.task_history_file, 'w') as f:
                json.dump(history, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error appending to history: {e}")