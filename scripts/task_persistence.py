"""
Task Persistence System for Continuous AI

Tracks completed tasks to prevent infinite loops and duplicate work.
Implements smart cooldown periods and semantic deduplication.
"""

import json
import logging
import time
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

from scripts.work_item_types import WorkItem, TaskPriority


@dataclass
class CompletedTask:
    """Record of a completed task."""
    work_item_id: str
    title: str
    task_type: str
    repository: Optional[str]
    description_hash: str  # Hash of description for semantic matching
    completed_at: datetime
    issue_number: Optional[int] = None
    value_created: float = 0.0
    execution_result: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompletedTask':
        """Create from dictionary."""
        data = data.copy()
        data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


class TaskPersistence:
    """Manages persistence and deduplication of completed tasks."""
    
    def __init__(self, storage_file: str = "completed_tasks.json"):
        """Initialize task persistence.
        
        Args:
            storage_file: File to store completed tasks
        """
        self.storage_file = storage_file
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache
        self.completed_tasks: Dict[str, CompletedTask] = {}
        self.title_hashes: Set[str] = set()  # For quick title duplicate checking
        self.description_hashes: Set[str] = set()  # For semantic duplicate checking
        
        # Configuration
        self.default_cooldown_hours = 24  # Don't repeat same task for 24 hours
        self.semantic_similarity_threshold = 0.8  # Threshold for semantic duplicates
        self.max_stored_tasks = 1000  # Limit stored tasks to prevent unbounded growth
        
        # Task-specific cooldown periods (in hours)
        self.task_cooldowns = {
            'TESTING': 12,  # Testing tasks - 12 hour cooldown
            'FEATURE': 72,  # Feature tasks - 3 day cooldown  
            'BUG_FIX': 48,  # Bug fixes - 2 day cooldown
            'DOCUMENTATION': 48,  # Documentation - 2 day cooldown
            'RESEARCH': 168,  # Research - 1 week cooldown
            'SYSTEM_IMPROVEMENT': 168,  # System improvements - 1 week cooldown
            'MAINTENANCE': 72,  # Maintenance - 3 day cooldown
            'NEW_PROJECT': 720,  # New projects - 30 day cooldown
            'INTEGRATION': 168,  # Integration - 1 week cooldown
            'REPOSITORY_HEALTH': 24  # Repository health - 1 day cooldown
        }
        
        # Skip tracking for problematic tasks
        self.skip_stats: Dict[str, Dict[str, Any]] = {}
        self.title_cooldown: Dict[str, float] = {}  # Title -> last seen timestamp
        self.cooldown_period = 300  # Base 5 minute cooldown
        
        # Track problematic tasks that can't generate alternatives
        self.problematic_tasks: Dict[str, Dict[str, Any]] = {}
        self.problematic_task_cooldown = 86400  # 24 hour cooldown for problematic tasks
        
        # Load existing data
        self._load_completed_tasks()
        self._load_problematic_tasks()
    
    def record_completed_task(self, work_item: WorkItem, execution_result: Dict[str, Any]) -> bool:
        """Record a completed task.
        
        Args:
            work_item: The completed work item
            execution_result: Result of task execution
            
        Returns:
            True if recorded successfully
        """
        try:
            # Create completed task record
            completed_task = CompletedTask(
                work_item_id=work_item.id,
                title=work_item.title,
                task_type=work_item.task_type,
                repository=work_item.repository,
                description_hash=self._hash_description(work_item.description),
                completed_at=datetime.now(timezone.utc),
                issue_number=execution_result.get('issue_number'),
                value_created=execution_result.get('value_created', 0.0),
                execution_result=execution_result
            )
            
            # Store in memory
            self.completed_tasks[work_item.id] = completed_task
            self.title_hashes.add(self._hash_title(work_item.title))
            self.description_hashes.add(completed_task.description_hash)
            
            # Persist to disk
            self._save_completed_tasks()
            
            self.logger.info(f"✅ Recorded completed task: {work_item.title}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error recording completed task: {e}")
            return False
    
    def record_skipped_task(self, task_title: str, reason: str = "duplicate") -> None:
        """Record a skipped task to extend cooldown period and track problematic tasks.
        
        Args:
            task_title: Title of the skipped task
            reason: Reason for skipping (e.g., "duplicate", "error")
        """
        # Update the cooldown timestamp for skipped tasks
        self.title_cooldown[task_title] = time.time()
        
        # Track skip statistics
        if task_title not in self.skip_stats:
            self.skip_stats[task_title] = {
                'count': 0,
                'first_skip': datetime.now(timezone.utc).isoformat(),
                'reasons': []
            }
        
        self.skip_stats[task_title]['count'] += 1
        self.skip_stats[task_title]['last_skip'] = datetime.now(timezone.utc).isoformat()
        self.skip_stats[task_title]['reasons'].append(reason)
        
        # If a task is skipped too many times, exponentially increase its cooldown
        skip_count = self.skip_stats[task_title]['count']
        if skip_count > 10:
            # Exponential backoff: 5 min, 10 min, 20 min, 40 min, ..., max 1 hour
            self.cooldown_period = min(3600, 300 * (2 ** ((skip_count - 10) // 5)))
            self.logger.warning(
                f"⚠️ Task '{task_title}' skipped {skip_count} times. "
                f"Cooldown increased to {self.cooldown_period // 60} minutes"
            )
    
    def is_duplicate_task(self, work_item: WorkItem) -> bool:
        """Check if a work item is a duplicate of recent work.
        
        Args:
            work_item: Work item to check
            
        Returns:
            True if this is a duplicate task
        """
        # Check if this is a problematic task
        task_key = f"{work_item.task_type}:{work_item.repository}:{work_item.title.lower()}"
        if task_key in self.problematic_tasks:
            task_data = self.problematic_tasks[task_key]
            cooldown_until = datetime.fromisoformat(task_data['cooldown_until'].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) < cooldown_until:
                self.logger.debug(
                    f"🚫 Problematic task still in cooldown: {work_item.title} "
                    f"(attempts: {task_data['attempt_count']})"
                )
                return True
        
        # Check if title is in recent skip cooldown
        if work_item.title in self.title_cooldown:
            cooldown_elapsed = time.time() - self.title_cooldown[work_item.title]
            skip_info = self.skip_stats.get(work_item.title, {})
            skip_count = skip_info.get('count', 0)
            
            # Use exponential backoff for frequently skipped tasks
            if skip_count > 10:
                cooldown = min(3600, 300 * (2 ** ((skip_count - 10) // 5)))
            else:
                cooldown = self.cooldown_period
                
            if cooldown_elapsed < cooldown:
                self.logger.debug(
                    f"🔄 Task in skip cooldown: {work_item.title} "
                    f"({cooldown_elapsed:.0f}s < {cooldown}s)"
                )
                return True
        
        # Check exact title duplicates (completed tasks)
        title_hash = self._hash_title(work_item.title)
        if title_hash in self.title_hashes:
            # Check if any matching title is still in cooldown
            for task in self.completed_tasks.values():
                if (self._hash_title(task.title) == title_hash and 
                    self._is_in_cooldown(task, work_item.task_type)):
                    self.logger.debug(f"🔄 Duplicate title found in cooldown: {work_item.title}")
                    return True
        
        # Check semantic duplicates (description similarity)
        description_hash = self._hash_description(work_item.description)
        if description_hash in self.description_hashes:
            # Check if any matching description is still in cooldown
            for task in self.completed_tasks.values():
                if (task.description_hash == description_hash and
                    self._is_in_cooldown(task, work_item.task_type)):
                    self.logger.debug(f"🔄 Duplicate description found in cooldown: {work_item.title}")
                    return True
        
        # Check repository-specific duplicates
        if work_item.repository:
            for task in self.completed_tasks.values():
                if (task.repository == work_item.repository and
                    task.task_type == work_item.task_type and
                    self._is_similar_repository_task(task, work_item) and
                    self._is_in_cooldown(task, work_item.task_type)):
                    self.logger.debug(f"🔄 Similar repository task found in cooldown: {work_item.title}")
                    return True
        
        return False
    
    def get_task_history(self, repository: Optional[str] = None, 
                        task_type: Optional[str] = None,
                        hours_back: int = 168) -> List[CompletedTask]:
        """Get history of completed tasks.
        
        Args:
            repository: Filter by repository (optional)
            task_type: Filter by task type (optional)
            hours_back: How many hours of history to include
            
        Returns:
            List of completed tasks matching criteria
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        filtered_tasks = []
        for task in self.completed_tasks.values():
            if task.completed_at < cutoff_time:
                continue
                
            if repository and task.repository != repository:
                continue
                
            if task_type and task.task_type != task_type:
                continue
                
            filtered_tasks.append(task)
        
        # Sort by completion time (most recent first)
        filtered_tasks.sort(key=lambda t: t.completed_at, reverse=True)
        return filtered_tasks
    
    def get_completion_stats(self) -> Dict[str, Any]:
        """Get statistics about completed tasks.
        
        Returns:
            Statistics dictionary
        """
        if not self.completed_tasks:
            return {
                'total_tasks': 0,
                'total_value_created': 0.0,
                'task_types': {},
                'repositories': {},
                'avg_value_per_task': 0.0
            }
        
        task_types = {}
        repositories = {}
        total_value = 0.0
        
        for task in self.completed_tasks.values():
            # Count by task type
            task_types[task.task_type] = task_types.get(task.task_type, 0) + 1
            
            # Count by repository
            if task.repository:
                repositories[task.repository] = repositories.get(task.repository, 0) + 1
            
            # Sum value
            total_value += task.value_created
        
        return {
            'total_tasks': len(self.completed_tasks),
            'total_value_created': round(total_value, 2),
            'task_types': task_types,
            'repositories': repositories,
            'avg_value_per_task': round(total_value / len(self.completed_tasks), 2),
            'last_task_completed': max(task.completed_at for task in self.completed_tasks.values()).isoformat()
        }
    
    def cleanup_old_tasks(self, max_age_days: int = 30):
        """Clean up old completed tasks to prevent unbounded growth.
        
        Args:
            max_age_days: Maximum age of tasks to keep
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        
        tasks_to_remove = []
        for task_id, task in self.completed_tasks.items():
            if task.completed_at < cutoff_time:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            task = self.completed_tasks[task_id]
            del self.completed_tasks[task_id]
            
            # Remove from hash sets (might remove others with same hash, but that's OK)
            self.title_hashes.discard(self._hash_title(task.title))
            self.description_hashes.discard(task.description_hash)
        
        if tasks_to_remove:
            self.logger.info(f"🧹 Cleaned up {len(tasks_to_remove)} old completed tasks")
            self._save_completed_tasks()
    
    def _is_in_cooldown(self, completed_task: CompletedTask, new_task_type: str) -> bool:
        """Check if a completed task is still in its cooldown period.
        
        Args:
            completed_task: Previously completed task
            new_task_type: Type of new task being considered
            
        Returns:
            True if still in cooldown period
        """
        cooldown_hours = self.task_cooldowns.get(new_task_type, self.default_cooldown_hours)
        cooldown_time = timedelta(hours=cooldown_hours)
        
        time_since_completion = datetime.now(timezone.utc) - completed_task.completed_at
        return time_since_completion < cooldown_time
    
    def _is_similar_repository_task(self, completed_task: CompletedTask, new_work_item: WorkItem) -> bool:
        """Check if two repository tasks are similar enough to be considered duplicates.
        
        Args:
            completed_task: Previously completed task
            new_work_item: New work item being considered
            
        Returns:
            True if tasks are similar
        """
        # Same task type and repository
        if completed_task.task_type != new_work_item.task_type:
            return False
        
        if completed_task.repository != new_work_item.repository:
            return False
        
        # For testing tasks, check if they're for the same general purpose
        if new_work_item.task_type == 'TESTING':
            # If both are testing tasks for the same repo, they're likely similar
            return True
        
        # For other task types, use title similarity
        return self._calculate_title_similarity(completed_task.title, new_work_item.title) > 0.7
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple word-based similarity
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _hash_title(self, title: str) -> str:
        """Create a hash of a task title for quick duplicate checking.
        
        Args:
            title: Task title
            
        Returns:
            Hash string
        """
        # Normalize title for hashing
        normalized = title.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _hash_description(self, description: str) -> str:
        """Create a hash of a task description for semantic duplicate checking.
        
        Args:
            description: Task description
            
        Returns:
            Hash string
        """
        # Normalize description for hashing
        normalized = description.lower().strip()
        # Remove common words that don't affect semantic meaning
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'for', 'to', 'of', 'in', 'on', 'at'}
        words = [word for word in normalized.split() if word not in common_words]
        normalized = ' '.join(sorted(words))  # Sort for consistent hashing
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _load_completed_tasks(self):
        """Load completed tasks from storage."""
        try:
            if Path(self.storage_file).exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                
                for task_data in data.get('completed_tasks', []):
                    try:
                        task = CompletedTask.from_dict(task_data)
                        self.completed_tasks[task.work_item_id] = task
                        self.title_hashes.add(self._hash_title(task.title))
                        self.description_hashes.add(task.description_hash)
                    except Exception as e:
                        self.logger.warning(f"Error loading task: {e}")
                
                self.logger.info(f"📋 Loaded {len(self.completed_tasks)} completed tasks from storage")
            else:
                self.logger.info("📋 No previous completed tasks found - starting fresh")
                
        except Exception as e:
            self.logger.error(f"❌ Error loading completed tasks: {e}")
    
    def _save_completed_tasks(self):
        """Save completed tasks to storage."""
        try:
            # Limit the number of tasks we store
            tasks_to_save = list(self.completed_tasks.values())
            
            # Sort by completion time and keep only the most recent
            tasks_to_save.sort(key=lambda t: t.completed_at, reverse=True)
            tasks_to_save = tasks_to_save[:self.max_stored_tasks]
            
            data = {
                'completed_tasks': [task.to_dict() for task in tasks_to_save],
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'total_tasks_recorded': len(self.completed_tasks)
            }
            
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Error saving completed tasks: {e}")
    
    def record_problematic_task(self, task_title: str, task_type: str, 
                               repository: Optional[str] = None) -> None:
        """Record a task that couldn't generate alternatives, applying extended cooldown.
        
        Args:
            task_title: Title of the problematic task
            task_type: Type of the task
            repository: Repository name (optional)
        """
        current_time = datetime.now(timezone.utc)
        
        # Create unique key for this task
        task_key = f"{task_type}:{repository}:{task_title.lower()}"
        
        # Record the problematic task with extended cooldown
        self.problematic_tasks[task_key] = {
            'title': task_title,
            'task_type': task_type,
            'repository': repository,
            'first_seen': current_time.isoformat(),
            'last_seen': current_time.isoformat(),
            'attempt_count': self.problematic_tasks.get(task_key, {}).get('attempt_count', 0) + 1,
            'cooldown_until': (current_time + timedelta(seconds=self.problematic_task_cooldown)).isoformat()
        }
        
        # Also extend the regular cooldown
        self.title_cooldown[task_title] = time.time() + self.problematic_task_cooldown
        
        # Update skip stats with "problematic" reason
        self.record_skipped_task(task_title, "problematic_no_alternatives")
        
        self.logger.warning(
            f"⚠️ Recorded problematic task: {task_title} "
            f"(attempt #{self.problematic_tasks[task_key]['attempt_count']}) "
            f"- Extended cooldown until {self.problematic_tasks[task_key]['cooldown_until']}"
        )
        
        # Persist problematic tasks
        self._save_problematic_tasks()
    
    def _save_problematic_tasks(self):
        """Save problematic tasks to disk for persistence across sessions."""
        try:
            # Save to a separate file
            problematic_file = Path(self.storage_file).parent / "problematic_tasks.json"
            
            data = {
                'problematic_tasks': self.problematic_tasks,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            with open(problematic_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Error saving problematic tasks: {e}")
    
    def _load_problematic_tasks(self):
        """Load problematic tasks from disk."""
        try:
            problematic_file = Path(self.storage_file).parent / "problematic_tasks.json"
            
            if problematic_file.exists():
                with open(problematic_file, 'r') as f:
                    data = json.load(f)
                
                # Clean up expired cooldowns
                current_time = datetime.now(timezone.utc)
                for task_key, task_data in data.get('problematic_tasks', {}).items():
                    cooldown_until = datetime.fromisoformat(task_data['cooldown_until'].replace('Z', '+00:00'))
                    if cooldown_until > current_time:
                        self.problematic_tasks[task_key] = task_data
                        
                self.logger.info(f"📋 Loaded {len(self.problematic_tasks)} problematic tasks")
                
        except Exception as e:
            self.logger.warning(f"❌ Error loading problematic tasks: {e}")