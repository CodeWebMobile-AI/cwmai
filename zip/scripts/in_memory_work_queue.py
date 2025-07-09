"""
In-Memory Work Queue

Drop-in replacement for Redis work queue using Python's built-in queue module.
Provides the same interface but runs entirely in memory with thread safety.
"""

import queue
import threading
import json
from scripts.custom_json_encoder import safe_json_dump
import time
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import heapq

# Import TaskPriority from shared work_item_types to avoid conflicts
from scripts.work_item_types import TaskPriority


@dataclass
class WorkItem:
    """Work item with priority and metadata."""
    id: str
    task_type: str
    title: str
    description: str
    repository: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    attempts: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'task_type': self.task_type,
            'title': self.title,
            'description': self.description,
            'repository': self.repository,
            'priority': self.priority.value if isinstance(self.priority, TaskPriority) else self.priority,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'attempts': self.attempts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkItem':
        """Create from dictionary."""
        # Handle priority
        priority = data.get('priority', TaskPriority.MEDIUM)
        if isinstance(priority, int):
            # Map integer values to TaskPriority enum members
            try:
                # Find the enum member with this value
                for member in TaskPriority:
                    if member.value == priority:
                        priority = member
                        break
                else:
                    # If no match found, use default
                    priority = TaskPriority.MEDIUM
            except Exception:
                priority = TaskPriority.MEDIUM
        elif isinstance(priority, str):
            priority = TaskPriority[priority.upper()]
        
        # Handle created_at
        created_at = data.get('created_at')
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            task_type=data.get('task_type', data.get('type', 'unknown')),  # Support both for compatibility
            title=data['title'],
            description=data.get('description', ''),
            repository=data.get('repository'),
            priority=priority,
            metadata=data.get('metadata', {}),
            created_at=created_at,
            attempts=data.get('attempts', 0)
        )


class InMemoryWorkQueue:
    """Thread-safe in-memory work queue with Redis-compatible interface."""
    
    def __init__(self, max_size: int = 10000, enable_persistence: bool = True,
                 persistence_file: str = "work_queue_backup.json",
                 processing_timeout: int = 0):  # 0 means no timeout
        """Initialize in-memory work queue.
        
        Args:
            max_size: Maximum queue size
            enable_persistence: Enable periodic persistence to disk
            persistence_file: File for queue persistence
            processing_timeout: Timeout in seconds for processing items (0 = no timeout)
        """
        self.logger = logging.getLogger(__name__)
        self.max_size = max_size
        self.enable_persistence = enable_persistence
        self.persistence_file = persistence_file
        self.processing_timeout = processing_timeout
        
        # Priority queue with thread safety
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self._lock = threading.RLock()
        
        # Track work items by ID for quick lookup
        self._items = {}
        self._item_lock = threading.RLock()
        
        # Track claimed items to prevent duplicates
        self._claimed_items = set()
        self._claimed_lock = threading.RLock()
        
        # Processing items (claimed but not completed)
        self._processing = {}
        self._processing_lock = threading.RLock()
        
        # Completed items (recent history)
        self._completed = []
        self._completed_max = 1000
        self._completed_lock = threading.RLock()
        
        # Failed items with retry tracking
        self._failed = {}
        self._failed_lock = threading.RLock()
        
        # Work item hashes for deduplication
        self._work_item_hashes = {}  # hash -> work_item_id
        self._hash_lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'total_enqueued': 0,
            'total_dequeued': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_retried': 0,
            'total_duplicates_rejected': 0
        }
        self._stats_lock = threading.RLock()
        
        # Load persisted queue if enabled
        if enable_persistence:
            self._load_from_disk()
            # Start persistence thread
            self._start_persistence_thread()
    
    def _calculate_work_hash(self, work_item: WorkItem) -> str:
        """Calculate a hash for work item deduplication.
        
        Hash is based on:
        - Task type
        - Repository
        - Title
        - Issue number (if applicable)
        """
        hash_parts = [
            work_item.task_type,
            work_item.repository or '',
            work_item.title,
            str(work_item.metadata.get('issue_number', ''))
        ]
        
        hash_string = '|'.join(hash_parts).lower()
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def enqueue(self, work_item: Dict[str, Any]) -> bool:
        """Add work item to queue (Redis LPUSH equivalent).
        
        Args:
            work_item: Work item dictionary
            
        Returns:
            True if successful
        """
        try:
            # Convert to WorkItem if needed
            if isinstance(work_item, dict):
                item = WorkItem.from_dict(work_item)
            else:
                item = work_item
            
            # Generate ID if missing
            if not item.id:
                item.id = str(uuid.uuid4())
            
            with self._lock:
                # Calculate hash for deduplication
                work_hash = self._calculate_work_hash(item)
                
                # Check if duplicate based on hash
                with self._hash_lock:
                    if work_hash in self._work_item_hashes:
                        existing_id = self._work_item_hashes[work_hash]
                        # Check if the existing item is still active
                        if (existing_id in self._items or 
                            existing_id in self._claimed_items or 
                            existing_id in self._processing):
                            self.logger.debug(f"Duplicate work item detected: {item.title}")
                            with self._stats_lock:
                                self._stats['total_duplicates_rejected'] += 1
                            return False
                        else:
                            # Old item no longer active, remove from hash map
                            del self._work_item_hashes[work_hash]
                
                # Check if already exists by ID
                with self._item_lock:
                    if item.id in self._items:
                        self.logger.warning(f"Work item {item.id} already in queue")
                        return False
                
                # Check if already claimed/processing
                with self._claimed_lock:
                    if item.id in self._claimed_items:
                        self.logger.warning(f"Work item {item.id} already claimed")
                        return False
                
                with self._processing_lock:
                    if item.id in self._processing:
                        self.logger.warning(f"Work item {item.id} already being processed")
                        return False
                
                # Create priority tuple (lower number = higher priority)
                priority_tuple = (
                    item.priority.value if isinstance(item.priority, TaskPriority) else item.priority,
                    item.created_at.timestamp() if item.created_at else time.time(),
                    item.id
                )
                
                # Add to queue
                self._queue.put((priority_tuple, item.id), block=False)
                
                # Track item
                with self._item_lock:
                    self._items[item.id] = item
                
                # Track hash for deduplication
                with self._hash_lock:
                    self._work_item_hashes[work_hash] = item.id
                
                # Update stats
                with self._stats_lock:
                    self._stats['total_enqueued'] += 1
                
                self.logger.debug(f"Enqueued work item: {item.id} ({item.task_type})")
                return True
                
        except queue.Full:
            self.logger.error("Work queue is full")
            return False
        except Exception as e:
            self.logger.error(f"Error enqueuing work: {e}")
            return False
    
    def dequeue(self, timeout: Optional[float] = None, worker_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get next work item from queue (Redis BRPOP equivalent).
        
        Args:
            timeout: Timeout in seconds (None for blocking)
            worker_id: ID of the worker claiming this task
            
        Returns:
            Work item dictionary or None
        """
        try:
            # Get from priority queue
            priority_tuple, item_id = self._queue.get(timeout=timeout)
            
            # Get actual item
            with self._item_lock:
                item = self._items.pop(item_id, None)
            
            if not item:
                self.logger.warning(f"Work item {item_id} not found in items dict")
                return None
            
            # Move to processing with worker assignment
            with self._processing_lock:
                # Check if already being processed (shouldn't happen but be safe)
                if item.id in self._processing:
                    self.logger.warning(f"Work item {item.id} already in processing!")
                    # Put it back in the queue
                    self.enqueue(item.to_dict())
                    return None
                    
                self._processing[item.id] = {
                    'item': item,
                    'started_at': datetime.now(timezone.utc),
                    'worker_id': worker_id or threading.current_thread().name,
                    'claimed': True
                }
            
            # Update stats
            with self._stats_lock:
                self._stats['total_dequeued'] += 1
            
            self.logger.info(f"Worker {worker_id or threading.current_thread().name} claimed work item: {item.id} ({item.task_type})")
            return item.to_dict()
            
        except queue.Empty:
            return None
        except Exception as e:
            self.logger.error(f"Error dequeuing work: {e}")
            return None
    
    def complete(self, item_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark work item as completed.
        
        Args:
            item_id: Work item ID
            result: Optional result data
            
        Returns:
            True if successful
        """
        try:
            # Remove from processing
            with self._processing_lock:
                processing_info = self._processing.pop(item_id, None)
            
            # Remove from claimed items
            with self._claimed_lock:
                self._claimed_items.discard(item_id)
            
            if not processing_info:
                # This is expected for work items created directly by the orchestrator
                # that bypass the queue's enqueue/dequeue mechanism
                self.logger.debug(f"Work item {item_id} not in processing (direct execution)")
                # Still update stats for these items
                with self._stats_lock:
                    self._stats['total_completed'] += 1
                return True
            
            # Add to completed history
            with self._completed_lock:
                self._completed.append({
                    'item': processing_info['item'].to_dict(),
                    'started_at': processing_info['started_at'].isoformat(),
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'worker_id': processing_info['worker_id'],
                    'result': result
                })
                
                # Trim history
                if len(self._completed) > self._completed_max:
                    self._completed = self._completed[-self._completed_max:]
            
            # Update stats
            with self._stats_lock:
                self._stats['total_completed'] += 1
            
            self.logger.debug(f"Completed work item: {item_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error completing work item {item_id}: {e}")
            return False
    
    def fail(self, item_id: str, error: str, retry: bool = True) -> bool:
        """Mark work item as failed.
        
        Args:
            item_id: Work item ID
            error: Error message
            retry: Whether to retry the item
            
        Returns:
            True if successful
        """
        try:
            # Remove from processing
            with self._processing_lock:
                processing_info = self._processing.pop(item_id, None)
            
            # Remove from claimed items
            with self._claimed_lock:
                self._claimed_items.discard(item_id)
            
            if not processing_info:
                # This is expected for work items created directly by the orchestrator
                self.logger.debug(f"Work item {item_id} not in processing (direct execution)")
                # Still update stats for these items
                with self._stats_lock:
                    self._stats['total_failed'] += 1
                return True
            
            item = processing_info['item']
            item.attempts += 1
            
            # Add to failed tracking
            with self._failed_lock:
                self._failed[item_id] = {
                    'item': item,
                    'error': error,
                    'failed_at': datetime.now(timezone.utc),
                    'attempts': item.attempts
                }
            
            # Update stats
            with self._stats_lock:
                self._stats['total_failed'] += 1
            
            # Retry if requested and under limit
            if retry and item.attempts < 3:
                # Add back to queue with lower priority
                new_priority = min(item.priority.value + 1, TaskPriority.LOW.value)
                item.priority = TaskPriority(new_priority)
                
                if self.enqueue(item.to_dict()):
                    with self._stats_lock:
                        self._stats['total_retried'] += 1
                    self.logger.info(f"Retrying work item {item_id} (attempt {item.attempts})")
            
            self.logger.debug(f"Failed work item: {item_id} - {error}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error failing work item {item_id}: {e}")
            return False
    
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def processing_count(self) -> int:
        """Get number of items being processed."""
        with self._processing_lock:
            return len(self._processing)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
        
        stats.update({
            'queue_size': self.size(),
            'processing': self.processing_count(),
            'completed_history': len(self._completed),
            'failed_items': len(self._failed)
        })
        
        return stats
    
    def get_processing(self) -> List[Dict[str, Any]]:
        """Get list of items being processed."""
        with self._processing_lock:
            return [
                {
                    'item': info['item'].to_dict(),
                    'started_at': info['started_at'].isoformat(),
                    'worker_id': info['worker_id']
                }
                for info in self._processing.values()
            ]
    
    def get_completed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent completed items."""
        with self._completed_lock:
            return self._completed[-limit:]
    
    def clear(self) -> bool:
        """Clear all queues."""
        try:
            with self._lock:
                # Clear main queue
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Clear tracking
                with self._item_lock:
                    self._items.clear()
                
                with self._processing_lock:
                    self._processing.clear()
                
                with self._claimed_lock:
                    self._claimed_items.clear()
                
                with self._completed_lock:
                    self._completed.clear()
                
                with self._failed_lock:
                    self._failed.clear()
            
            self.logger.info("Cleared all work queues")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing queues: {e}")
            return False
    
    def _save_to_disk(self):
        """Save queue state to disk."""
        if not self.enable_persistence:
            return
        
        try:
            state = {
                'items': [],
                'processing': [],
                'stats': self.get_stats(),
                'saved_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Get all items from queue (without removing them)
            temp_items = []
            with self._lock:
                # Temporarily drain queue
                while not self._queue.empty():
                    try:
                        item = self._queue.get_nowait()
                        temp_items.append(item)
                    except queue.Empty:
                        break
                
                # Put items back
                for item in temp_items:
                    self._queue.put(item)
                
                # Save items
                with self._item_lock:
                    for _, item_id in temp_items:
                        if item_id in self._items:
                            state['items'].append(self._items[item_id].to_dict())
            
            # Save processing items
            with self._processing_lock:
                for item_id, info in self._processing.items():
                    state['processing'].append({
                        'item': info['item'].to_dict(),
                        'started_at': info['started_at'].isoformat()
                    })
            
            # Write to file
            with open(self.persistence_file, 'w') as f:
                safe_json_dump(state, f, indent=2)
            
            self.logger.debug(f"Saved {len(state['items'])} items to disk")
            
        except Exception as e:
            self.logger.error(f"Error saving queue to disk: {e}")
    
    def _load_from_disk(self):
        """Load queue state from disk."""
        try:
            with open(self.persistence_file, 'r') as f:
                state = json.load(f)
            
            # Restore items
            for item_data in state.get('items', []):
                self.enqueue(item_data)
            
            # Restore processing items (re-queue them)
            for proc_data in state.get('processing', []):
                self.enqueue(proc_data['item'])
            
            self.logger.info(f"Loaded {len(state.get('items', []))} items from disk")
            
        except FileNotFoundError:
            self.logger.debug("No persistence file found, starting with empty queue")
        except Exception as e:
            self.logger.error(f"Error loading queue from disk: {e}")
    
    def _start_persistence_thread(self):
        """Start background thread for periodic persistence."""
        def persist_loop():
            while True:
                time.sleep(60)  # Save every minute
                self._save_to_disk()
        
        thread = threading.Thread(target=persist_loop, daemon=True)
        thread.start()
    
    def recover_stale_processing(self) -> int:
        """Recover work items that have been processing too long.
        
        Returns:
            Number of items recovered
        """
        # Skip recovery if timeout is disabled (0)
        if self.processing_timeout == 0:
            return 0
            
        recovered = 0
        current_time = datetime.now(timezone.utc)
        
        with self._processing_lock:
            stale_items = []
            
            # Find stale items
            for item_id, info in self._processing.items():
                processing_time = (current_time - info['started_at']).total_seconds()
                if processing_time > self.processing_timeout:
                    stale_items.append((item_id, info))
            
            # Recover stale items
            for item_id, info in stale_items:
                item = info['item']
                worker_id = info.get('worker_id', 'unknown')
                
                # Remove from processing
                del self._processing[item_id]
                
                # Remove from claimed
                with self._claimed_lock:
                    self._claimed_items.discard(item_id)
                
                # Re-queue with lower priority
                item.attempts += 1
                new_priority = min(item.priority.value + 1, TaskPriority.LOW.value)
                item.priority = TaskPriority(new_priority)
                
                self.logger.warning(
                    f"Recovering stale item {item_id} from worker {worker_id} "
                    f"(processing for {processing_time:.1f}s)"
                )
                
                # Re-enqueue
                if self.enqueue(item.to_dict()):
                    recovered += 1
        
        if recovered > 0:
            self.logger.info(f"Recovered {recovered} stale processing items")
        
        return recovered
    
    def close(self):
        """Save state and cleanup."""
        if self.enable_persistence:
            self._save_to_disk()
    
    async def is_in_queue(self, work_id: str) -> bool:
        """Check if a work item is already in the queue.
        
        Args:
            work_id: ID of the work item to check
            
        Returns:
            True if the item is in the ready queue
        """
        with self._lock:
            # Check if in items tracking
            with self._item_lock:
                return work_id in self._items