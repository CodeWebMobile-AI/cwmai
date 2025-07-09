"""
Claude Execution Queue System

Manages a queue of Claude execution tasks that can be processed
in parallel by multiple Claude executor workers.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
import hashlib
from scripts.repository_exclusion import RepositoryExclusion


class ClaudeTaskStatus(Enum):
    """Status of a Claude execution task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ClaudeTask:
    """Represents a task for Claude execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issue_path: str = ""  # Path to the issue JSON file
    repository: str = ""
    issue_number: int = 0
    prompt: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ClaudeTaskStatus = ClaudeTaskStatus.PENDING
    executor_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'issue_path': self.issue_path,
            'repository': self.repository,
            'issue_number': self.issue_number,
            'prompt': self.prompt,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'executor_id': self.executor_id,
            'result': self.result,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClaudeTask':
        """Create from dictionary."""
        task = cls()
        task.id = data.get('id', str(uuid.uuid4()))
        task.issue_path = data.get('issue_path', '')
        task.repository = data.get('repository', '')
        task.issue_number = data.get('issue_number', 0)
        task.prompt = data.get('prompt', '')
        
        # Parse dates
        if data.get('created_at'):
            task.created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        if data.get('started_at'):
            task.started_at = datetime.fromisoformat(data['started_at'].replace('Z', '+00:00'))
        if data.get('completed_at'):
            task.completed_at = datetime.fromisoformat(data['completed_at'].replace('Z', '+00:00'))
        
        # Parse status
        status_value = data.get('status', 'pending')
        task.status = ClaudeTaskStatus(status_value)
        
        task.executor_id = data.get('executor_id')
        task.result = data.get('result')
        task.error = data.get('error')
        
        return task


class ClaudeExecutionQueue:
    """Manages a queue of Claude execution tasks."""
    
    def __init__(self, queue_file: str = "claude_execution_queue.json"):
        """Initialize the queue.
        
        Args:
            queue_file: Path to the queue persistence file
        """
        self.queue_file = queue_file
        self.logger = logging.getLogger(__name__)
        self._queue: List[ClaudeTask] = []
        self._processing: Dict[str, ClaudeTask] = {}  # task_id -> task
        self._completed: List[ClaudeTask] = []
        self._lock = asyncio.Lock()
        self._last_wait_time = 0.0  # Track last wait time for metrics
        self._wait_times: List[float] = []  # Track recent wait times
        
        # Load existing queue
        self._load_queue()
    
    def _load_queue(self):
        """Load queue from disk."""
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                
                # Restore queue state and remove excluded repositories
                pending_raw = data.get('pending', [])
                orig_pending = len(pending_raw)
                filtered_pending = [t for t in pending_raw if not RepositoryExclusion.is_excluded_repo(t.get('repository', ''))]
                self._queue = [ClaudeTask.from_dict(t) for t in filtered_pending]
                
                # Handle processing - filter out excluded repos; support list or dict formats
                processing_data = data.get('processing', [])
                orig_processing = len(processing_data) if isinstance(processing_data, dict) else len(processing_data or [])
                if isinstance(processing_data, list):
                    proc_list = [t for t in processing_data if not RepositoryExclusion.is_excluded_repo(t.get('repository', ''))]
                    self._processing = {
                        t.get('id', str(i)): ClaudeTask.from_dict(t)
                        for i, t in enumerate(proc_list)
                    }
                else:
                    proc_dict = {
                        k: v for k, v in processing_data.items()
                        if not RepositoryExclusion.is_excluded_repo(v.get('repository', ''))
                    }
                    self._processing = {
                        k: ClaudeTask.from_dict(v)
                        for k, v in proc_dict.items()
                    }
                # Persist removal if any excluded tasks were dropped
                if orig_pending != len(self._queue) or orig_processing != len(self._processing):
                    self._save_queue()
                    self.logger.info(
                        f"Removed tasks for excluded repositories: "
                        f"pending {orig_pending - len(self._queue)}, "
                        f"processing {orig_processing - len(self._processing)}"
                    )
                
                self._completed = [
                    ClaudeTask.from_dict(t) 
                    for t in data.get('completed', [])[-100:]  # Keep last 100
                ]
                
                self.logger.info(f"Loaded Claude queue: {len(self._queue)} pending, "
                               f"{len(self._processing)} processing, {len(self._completed)} completed")
            except Exception as e:
                self.logger.error(f"Error loading queue: {e}")
    
    def _save_queue(self):
        """Save queue to disk."""
        try:
            data = {
                'pending': [t.to_dict() for t in self._queue],
                'processing': {k: v.to_dict() for k, v in self._processing.items()},
                'completed': [t.to_dict() for t in self._completed[-100:]]  # Keep last 100
            }
            
            with open(self.queue_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving queue: {e}")
    
    async def add_task(self, issue_path: str, repository: str, issue_number: int, prompt: str) -> ClaudeTask:
        """Add a new Claude execution task to the queue.
        
        Args:
            issue_path: Path to the issue JSON file
            repository: Repository name
            issue_number: Issue number
            prompt: Prompt for Claude
            
        Returns:
            The created task
        """
        # Check if repository is excluded
        if RepositoryExclusion.is_excluded_repo(repository):
            self.logger.warning(f"Repository {repository} is excluded from processing, skipping task")
            return None
        
        # Check for duplicates before creating task
        if self.has_pending_task_for_issue(repository, issue_number):
            self.logger.warning(f"Task already pending for {repository} issue #{issue_number}, skipping")
            return None
            
        if self.has_completed_task_for_issue(repository, issue_number):
            self.logger.info(f"Task already completed for {repository} issue #{issue_number}, skipping")
            return None
        
        task = ClaudeTask(
            issue_path=issue_path,
            repository=repository,
            issue_number=issue_number,
            prompt=prompt
        )
        
        async with self._lock:
            self._queue.append(task)
            self._save_queue()
            
        self.logger.info(f"Added Claude task {task.id} for {repository} issue #{issue_number}")
        return task
    
    def has_pending_task_for_issue(self, repository: str, issue_number: int) -> bool:
        """Check if there's already a pending or processing task for this issue.
        
        Args:
            repository: Repository name
            issue_number: Issue number
            
        Returns:
            True if a pending or processing task exists
        """
        # Check in queue
        for task in self._queue:
            if task.repository == repository and task.issue_number == issue_number:
                return True
        
        # Check in processing
        for task in self._processing.values():
            if task.repository == repository and task.issue_number == issue_number:
                return True
        
        return False
    
    def has_completed_task_for_issue(self, repository: str, issue_number: int) -> bool:
        """Check if there's already a completed task for this issue.
        
        Args:
            repository: Repository name
            issue_number: Issue number
            
        Returns:
            True if a completed task exists
        """
        # Check in completed tasks
        for task in self._completed:
            if task.repository == repository and task.issue_number == issue_number and task.status == ClaudeTaskStatus.COMPLETED:
                return True
        
        return False
    
    def has_active_task_for_repository(self, repository: str, max_per_repo: int = None) -> bool:
        """Check if repository has reached max active tasks limit.
        
        Args:
            repository: Repository name
            max_per_repo: Maximum concurrent tasks per repository (default: dynamic based on executors)
            
        Returns:
            True if repository has reached the max active tasks limit
        """
        # Dynamic limit based on number of executors and repositories
        if max_per_repo is None:
            # Count unique repositories with VALID tasks (exclude deleted repos)
            active_repos = set()
            
            # Count repos with processing tasks
            for task in self._processing.values():
                if task.status == ClaudeTaskStatus.RUNNING:
                    # Skip excluded repositories
                    if not RepositoryExclusion.is_excluded_repo(task.repository):
                        active_repos.add(task.repository)
            
            # Count repos with queued tasks
            for task in self._queue:
                # Skip excluded repositories
                if not RepositoryExclusion.is_excluded_repo(task.repository):
                    active_repos.add(task.repository)
            
            # Calculate dynamic limit based on actual active repos
            # With 10 workers and fewer repos, we can allow more tasks per repo
            num_repos = max(1, len(active_repos))
            num_executors = 10  # We know we have 10 workers
            
            # Better distribution: allow at least 1 task per repo, 
            # but if we have excess capacity, allow more
            if num_repos <= num_executors:
                # We have more workers than repos, so distribute evenly
                # But cap at 5 to prevent overloading a single repo
                max_per_repo = min(5, max(1, num_executors // num_repos))
            else:
                # More repos than workers, so limit to 1 per repo
                max_per_repo = 1
            
            self.logger.debug(f"Dynamic limit: {max_per_repo} tasks per repo "
                            f"(repos: {num_repos}, executors: {num_executors})")
        
        # Count active tasks for this repository
        active_count = 0
        for task in self._processing.values():
            if task.repository == repository and task.status == ClaudeTaskStatus.RUNNING:
                active_count += 1
                if active_count >= max_per_repo:
                    return True
        
        return False
    
    def get_repository_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all repositories with tasks.
        
        Returns:
            Dictionary mapping repository names to their status
        """
        repo_status = {}
        
        # Count pending tasks per repository
        for task in self._queue:
            if task.repository not in repo_status:
                repo_status[task.repository] = {
                    'pending': 0,
                    'processing': 0,
                    'completed': 0,
                    'active': False
                }
            repo_status[task.repository]['pending'] += 1
        
        # Count processing tasks per repository
        for task in self._processing.values():
            if task.repository not in repo_status:
                repo_status[task.repository] = {
                    'pending': 0,
                    'processing': 0,
                    'completed': 0,
                    'active': False
                }
            repo_status[task.repository]['processing'] += 1
            repo_status[task.repository]['active'] = True
        
        # Count completed tasks per repository
        for task in self._completed:
            if task.repository not in repo_status:
                repo_status[task.repository] = {
                    'pending': 0,
                    'processing': 0,
                    'completed': 0,
                    'active': False
                }
            repo_status[task.repository]['completed'] += 1
        
        return repo_status
    
    async def get_next_task(self, executor_id: str) -> Optional[ClaudeTask]:
        """Get the next task for an executor to process.
        
        Args:
            executor_id: ID of the executor requesting work
            
        Returns:
            Next task to process, or None if queue is empty
        """
        async with self._lock:
            if not self._queue:
                return None
            
            # Find the first task for a repository that doesn't have active tasks
            selected_task = None
            selected_index = -1
            
            for i, task in enumerate(self._queue):
                # Check if this repository already has an active task
                if not self.has_active_task_for_repository(task.repository):
                    selected_task = task
                    selected_index = i
                    break
            
            if selected_task is None:
                # All repositories have reached their task limit
                repo_counts = {}
                for task in self._processing.values():
                    if task.status == ClaudeTaskStatus.RUNNING:
                        # Skip excluded repositories in count
                        if not RepositoryExclusion.is_excluded_repo(task.repository):
                            repo_counts[task.repository] = repo_counts.get(task.repository, 0) + 1
                
                # Count unique active repos (excluding deleted ones)
                active_repos = set()
                for task in self._processing.values():
                    if task.status == ClaudeTaskStatus.RUNNING:
                        if not RepositoryExclusion.is_excluded_repo(task.repository):
                            active_repos.add(task.repository)
                for task in self._queue:
                    if not RepositoryExclusion.is_excluded_repo(task.repository):
                        active_repos.add(task.repository)
                
                # Calculate current dynamic limit with proper logic
                num_repos = max(1, len(active_repos))
                num_executors = 10
                if num_repos <= num_executors:
                    current_limit = min(5, max(1, num_executors // num_repos))
                else:
                    current_limit = 1
                
                self.logger.info(f"All repositories at capacity (dynamic limit: {current_limit} per repo), "
                               f"no work for executor {executor_id}. Active repos: {num_repos}, "
                               f"Active tasks by repo: {repo_counts}")
                return None
            
            # Remove the selected task from queue
            task = self._queue.pop(selected_index)
            task.status = ClaudeTaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)
            task.executor_id = executor_id
            
            # Calculate wait time for metrics
            if task.created_at:
                wait_time = (task.started_at - task.created_at).total_seconds()
                # Store wait time for stats (will be used by dynamic pool)
                self._last_wait_time = wait_time
                self._wait_times.append(wait_time)
                # Keep only last 100 wait times
                self._wait_times = self._wait_times[-100:]
            
            # Move to processing
            self._processing[task.id] = task
            self._save_queue()
            
        self.logger.info(f"Executor {executor_id} claimed task {task.id} for repository {task.repository}")
        return task
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as completed.
        
        Args:
            task_id: ID of the completed task
            result: Execution result
        """
        async with self._lock:
            if task_id not in self._processing:
                self.logger.warning(f"Task {task_id} not found in processing")
                return
            
            task = self._processing.pop(task_id)
            task.status = ClaudeTaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.result = result
            
            self._completed.append(task)
            self._save_queue()
            
        duration = (task.completed_at - task.started_at).total_seconds()
        self.logger.info(f"Task {task_id} completed in {duration:.1f}s")
        
        # Notify task persistence system
        try:
            from scripts.hybrid_state_manager import HybridStateManager
            hybrid_state = HybridStateManager()
            
            # Find the work item ID associated with this Claude task
            work_item_id = hybrid_state.get_work_item_by_claude_task(task_id)
            if work_item_id:
                # Mark the work item as completed
                hybrid_state.complete_in_progress_task(work_item_id)
                hybrid_state.complete_work(work_item_id, {
                    'success': result.get('success', False),
                    'claude_task_id': task_id,
                    'issue_number': task.issue_number,
                    'repository': task.repository,
                    'claude_result': result,
                    'completed_at': task.completed_at.isoformat()
                })
                self.logger.info(f"✅ Updated task persistence for work item {work_item_id}")
            else:
                self.logger.warning(f"No work item found for Claude task {task_id}")
        except Exception as e:
            self.logger.error(f"Error updating task persistence: {e}")
        
        # Mark the issue as completed if the task was successful
        if result.get('success', False):
            try:
                from scripts.local_issue_creator import LocalIssueCreator
                issue_creator = LocalIssueCreator(enable_claude_execution=False)
                
                # Mark issue as completed
                success = await issue_creator.mark_issue_completed(
                    repository=task.repository,
                    issue_number=task.issue_number
                )
                
                if success:
                    self.logger.info(f"✅ Marked issue #{task.issue_number} as completed in {task.repository}")
                else:
                    self.logger.warning(f"Failed to mark issue #{task.issue_number} as completed")
                    
            except Exception as e:
                self.logger.error(f"Error marking issue as completed: {e}")
    
    async def fail_task(self, task_id: str, error: str, retry: bool = True):
        """Mark a task as failed.
        
        Args:
            task_id: ID of the failed task
            error: Error message
            retry: Whether to retry the task
        """
        async with self._lock:
            if task_id not in self._processing:
                self.logger.warning(f"Task {task_id} not found in processing")
                return
            
            task = self._processing.pop(task_id)
            task.error = error
            
            if retry:
                # Reset for retry
                task.status = ClaudeTaskStatus.PENDING
                task.started_at = None
                task.executor_id = None
                self._queue.append(task)
                self.logger.info(f"Task {task_id} failed, requeueing for retry: {error}")
            else:
                task.status = ClaudeTaskStatus.FAILED
                task.completed_at = datetime.now(timezone.utc)
                self._completed.append(task)
                self.logger.error(f"Task {task_id} failed permanently: {error}")
                
                # Notify task persistence system of failure
                try:
                    from scripts.hybrid_state_manager import HybridStateManager
                    hybrid_state = HybridStateManager()
                    
                    # Find the work item ID associated with this Claude task
                    work_item_id = hybrid_state.get_work_item_by_claude_task(task_id)
                    if work_item_id:
                        # Mark the work item as failed
                        hybrid_state.fail_work(work_item_id, error, retry=False)
                        self.logger.info(f"❌ Updated task persistence for failed work item {work_item_id}")
                except Exception as e:
                    self.logger.error(f"Error updating task persistence for failure: {e}")
            
            self._save_queue()
    
    async def timeout_task(self, task_id: str):
        """Mark a task as timed out and requeue it.
        
        Args:
            task_id: ID of the timed out task
        """
        await self.fail_task(task_id, "Task execution timeout", retry=True)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Queue statistics
        """
        async with self._lock:
            stats = {
                'pending': len(self._queue),
                'processing': len(self._processing),
                'completed': len(self._completed),
                'total_processed': len(self._completed),
                'executors': list(set(t.executor_id for t in self._processing.values() if t.executor_id))
            }
            
            # Calculate average execution time
            completed_with_time = [
                t for t in self._completed 
                if t.completed_at and t.started_at
            ]
            if completed_with_time:
                total_time = sum(
                    (t.completed_at - t.started_at).total_seconds()
                    for t in completed_with_time
                )
                stats['avg_execution_time'] = total_time / len(completed_with_time)
            else:
                stats['avg_execution_time'] = 0
            
            # Add wait time statistics
            if self._wait_times:
                stats['avg_wait_time'] = sum(self._wait_times) / len(self._wait_times)
                stats['max_wait_time'] = max(self._wait_times)
                stats['last_wait_time'] = self._last_wait_time
            else:
                stats['avg_wait_time'] = 0
                stats['max_wait_time'] = 0
                stats['last_wait_time'] = 0
            
            return stats
    
    async def recover_stale_tasks(self, timeout_seconds: Optional[int] = None):
        """Recover tasks that have been processing for too long.
        
        NOTE: By default, this does NOT timeout tasks as Claude executions
        can take a very long time (30+ minutes) and should not be interrupted.
        Only use timeout if you have a specific reason.
        
        Args:
            timeout_seconds: Optional timeout in seconds (None = no timeout)
        """
        if timeout_seconds is None:
            # Don't timeout tasks by default
            return
            
        current_time = datetime.now(timezone.utc)
        stale_tasks = []
        
        async with self._lock:
            for task_id, task in self._processing.items():
                if task.started_at:
                    processing_time = (current_time - task.started_at).total_seconds()
                    if processing_time > timeout_seconds:
                        stale_tasks.append(task_id)
                        # Log details about stuck task
                        self.logger.warning(
                            f"Task {task_id} has been processing for {processing_time/3600:.1f} hours "
                            f"(started: {task.started_at.isoformat()}, executor: {task.executor_id})"
                        )
        
        # Timeout stale tasks (outside lock to avoid deadlock)
        for task_id in stale_tasks:
            await self.timeout_task(task_id)
            
        if stale_tasks:
            self.logger.warning(f"Recovered {len(stale_tasks)} stale tasks")
    
    async def cleanup_stuck_tasks(self, force: bool = False):
        """Clean up tasks that are stuck in processing state.
        
        This is more aggressive than recover_stale_tasks and should be used
        when the queue is known to be in a bad state.
        
        Args:
            force: If True, recover all processing tasks regardless of time
        """
        current_time = datetime.now(timezone.utc)
        tasks_to_recover = []
        
        async with self._lock:
            for task_id, task in list(self._processing.items()):
                should_recover = False
                
                if force:
                    should_recover = True
                    reason = "forced cleanup"
                elif not task.started_at:
                    should_recover = True
                    reason = "no start time"
                elif task.error and "timeout" in task.error.lower():
                    should_recover = True
                    reason = "previous timeout"
                else:
                    # Check if task has been processing for more than 2 hours
                    processing_time = (current_time - task.started_at).total_seconds()
                    if processing_time > 7200:  # 2 hours
                        should_recover = True
                        reason = f"processing for {processing_time/3600:.1f} hours"
                
                if should_recover:
                    tasks_to_recover.append((task_id, task, reason))
        
        # Recover tasks
        recovered = 0
        for task_id, task, reason in tasks_to_recover:
            self.logger.info(f"Recovering task {task_id} ({reason})")
            
            # Mark with detailed error
            error_msg = f"Task recovered due to: {reason}"
            if task.error:
                error_msg = f"{task.error} | {error_msg}"
            
            await self.fail_task(task_id, error_msg, retry=True)
            recovered += 1
        
        if recovered > 0:
            self.logger.info(f"Cleaned up {recovered} stuck tasks")
        
        return recovered
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get list of pending tasks (synchronous for monitoring).
        
        Returns:
            List of pending tasks
        """
        return [t.to_dict() for t in self._queue]
    
    def get_processing_tasks(self) -> List[Dict[str, Any]]:
        """Get list of processing tasks (synchronous for monitoring).
        
        Returns:
            List of processing tasks
        """
        return [t.to_dict() for t in self._processing.values()]