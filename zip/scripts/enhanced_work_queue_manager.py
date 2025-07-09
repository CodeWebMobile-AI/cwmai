"""
Enhanced Work Queue Manager with Deduplication and Locking

Prevents duplicate task assignment and coordinates work discovery.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
import uuid
import time

from scripts.work_item_types import WorkItem, TaskPriority


@dataclass
class WorkAssignment:
    """Track work assignments to prevent duplicates."""
    work_item: WorkItem
    assigned_to: str
    assigned_at: datetime
    completed: bool = False


class EnhancedWorkQueueManager:
    """Manages work queue with deduplication and locking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Main work queue
        self.work_queue: List[WorkItem] = []
        self.queue_lock = asyncio.Lock()
        
        # Assignment tracking
        self.active_assignments: Dict[str, WorkAssignment] = {}  # task_id -> assignment
        self.worker_assignments: Dict[str, List[str]] = {}  # worker_id -> [task_ids]
        
        # Deduplication tracking
        self.seen_tasks: Set[str] = set()  # Set of task signatures
        self.task_cooldown: Dict[str, float] = {}  # task_signature -> expiry_time
        
        # Work discovery coordination
        self.discovery_lock = asyncio.Lock()
        self.last_discovery_time = 0
        self.discovery_interval = 30  # seconds between discoveries
        self.discovery_in_progress = False
        
    def _get_task_signature(self, work_item: WorkItem) -> str:
        """Get unique signature for a task to detect duplicates."""
        # Create signature from repository + title
        return f"{work_item.repository or 'system'}::{work_item.title.lower().strip()}"
        
    async def add_work(self, work_items: List[WorkItem]) -> int:
        """Add work items to queue with deduplication.
        
        Returns:
            Number of items actually added (after deduplication)
        """
        added = 0
        
        async with self.queue_lock:
            for item in work_items:
                signature = self._get_task_signature(item)
                
                # Check if task is duplicate or in cooldown
                if signature in self.seen_tasks:
                    self.logger.debug(f"Skipping duplicate task: {item.title}")
                    continue
                    
                # Check cooldown
                if signature in self.task_cooldown:
                    if time.time() < self.task_cooldown[signature]:
                        self.logger.debug(f"Task in cooldown: {item.title}")
                        continue
                    else:
                        # Cooldown expired
                        del self.task_cooldown[signature]
                
                # Add to queue
                self.work_queue.append(item)
                self.seen_tasks.add(signature)
                added += 1
                
        if added > 0:
            self.logger.info(f"Added {added} new work items to queue (filtered {len(work_items) - added} duplicates)")
            
        return added
        
    async def get_work_for_worker(self, worker_id: str, specialization: Optional[str] = None) -> Optional[WorkItem]:
        """Get work for a specific worker with assignment tracking."""
        
        async with self.queue_lock:
            # Find suitable work
            selected_item = None
            selected_index = None
            
            # First pass: Look for specialized work
            if specialization and specialization != "general":
                for i, item in enumerate(self.work_queue):
                    # Skip if already assigned
                    if item.id in self.active_assignments:
                        continue
                        
                    # Check specialization match
                    if specialization == "system_tasks" and item.repository is None:
                        selected_item = item
                        selected_index = i
                        break
                    elif specialization == item.repository:
                        selected_item = item
                        selected_index = i
                        break
                        
            # Second pass: If no specialized work, take any work
            if selected_item is None:
                for i, item in enumerate(self.work_queue):
                    if item.id not in self.active_assignments:
                        selected_item = item
                        selected_index = i
                        break
                        
            # Assign work if found
            if selected_item is not None:
                # Remove from queue
                self.work_queue.pop(selected_index)
                
                # Track assignment
                assignment = WorkAssignment(
                    work_item=selected_item,
                    assigned_to=worker_id,
                    assigned_at=datetime.now(timezone.utc)
                )
                self.active_assignments[selected_item.id] = assignment
                
                # Track by worker
                if worker_id not in self.worker_assignments:
                    self.worker_assignments[worker_id] = []
                self.worker_assignments[worker_id].append(selected_item.id)
                
                self.logger.info(f"Assigned task '{selected_item.title}' to worker {worker_id}")
                return selected_item
                
        return None
        
    async def complete_work(self, worker_id: str, task_id: str, success: bool = True):
        """Mark work as completed and handle cleanup."""
        
        async with self.queue_lock:
            if task_id in self.active_assignments:
                assignment = self.active_assignments[task_id]
                assignment.completed = True
                
                # Add to cooldown if failed
                if not success:
                    signature = self._get_task_signature(assignment.work_item)
                    # 5 minute cooldown for failed tasks
                    self.task_cooldown[signature] = time.time() + 300
                    self.logger.info(f"Added failed task to cooldown: {assignment.work_item.title}")
                
                # Remove from active assignments
                del self.active_assignments[task_id]
                
                # Remove from worker assignments
                if worker_id in self.worker_assignments:
                    self.worker_assignments[worker_id].remove(task_id)
                    
    async def should_discover_work(self) -> bool:
        """Check if work discovery should run (with coordination)."""
        
        current_time = time.time()
        
        # Quick check without lock
        if self.discovery_in_progress:
            return False
            
        if current_time - self.last_discovery_time < self.discovery_interval:
            return False
            
        # Try to acquire discovery lock
        if self.discovery_lock.locked():
            return False
            
        try:
            # Try to acquire with timeout
            async with asyncio.timeout(0.1):
                await self.discovery_lock.acquire()
                
            # Double-check timing with lock held
            if current_time - self.last_discovery_time >= self.discovery_interval:
                self.discovery_in_progress = True
                self.last_discovery_time = current_time
                return True
            else:
                return False
                
        except asyncio.TimeoutError:
            return False
        finally:
            if self.discovery_lock.locked():
                self.discovery_lock.release()
                
    async def end_discovery(self):
        """Mark discovery as complete."""
        self.discovery_in_progress = False
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'queue_size': len(self.work_queue),
            'active_assignments': len(self.active_assignments),
            'seen_tasks': len(self.seen_tasks),
            'tasks_in_cooldown': len(self.task_cooldown),
            'workers_with_tasks': len([w for w in self.worker_assignments if self.worker_assignments[w]]),
            'discovery_in_progress': self.discovery_in_progress,
            'time_until_next_discovery': max(0, self.discovery_interval - (time.time() - self.last_discovery_time))
        }
        
    async def cleanup_stale_assignments(self, timeout_minutes: int = 30):
        """Clean up assignments that have been active too long."""
        
        current_time = datetime.now(timezone.utc)
        stale_assignments = []
        
        async with self.queue_lock:
            for task_id, assignment in self.active_assignments.items():
                if (current_time - assignment.assigned_at).total_seconds() > timeout_minutes * 60:
                    stale_assignments.append((task_id, assignment))
                    
            # Return stale work to queue
            for task_id, assignment in stale_assignments:
                self.logger.warning(f"Returning stale task to queue: {assignment.work_item.title}")
                
                # Remove from assignments
                del self.active_assignments[task_id]
                
                # Remove from worker assignments
                if assignment.assigned_to in self.worker_assignments:
                    self.worker_assignments[assignment.assigned_to].remove(task_id)
                    
                # Add back to queue
                self.work_queue.insert(0, assignment.work_item)  # Add to front
                
        return len(stale_assignments)
