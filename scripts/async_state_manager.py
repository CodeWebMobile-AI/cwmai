"""
Async State Manager Module

High-performance async state management system with non-blocking operations.
Provides atomic updates, concurrent access protection, and intelligent sync strategies.
"""

import asyncio
import aiofiles
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import pickle
from contextlib import asynccontextmanager


@dataclass
class StateOperation:
    """Represents a state operation in the queue."""
    operation_type: str  # 'update', 'get', 'delete', 'backup'
    key_path: str
    value: Any = None
    timestamp: datetime = None
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class StateMetrics:
    """State manager performance metrics."""
    total_operations: int = 0
    read_operations: int = 0
    write_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time: float = 0.0
    last_sync_time: Optional[datetime] = None
    pending_operations: int = 0


class AsyncStateManager:
    """High-performance async state manager with intelligent caching and batching."""
    
    def __init__(self, 
                 state_file: str = "system_state.json",
                 backup_dir: str = "state_backups",
                 cache_size: int = 1000,
                 sync_interval: int = 30,
                 batch_size: int = 10):
        """Initialize async state manager.
        
        Args:
            state_file: Primary state file path
            backup_dir: Directory for state backups
            cache_size: Maximum cache entries
            sync_interval: Automatic sync interval in seconds
            batch_size: Number of operations to batch together
        """
        self.state_file = Path(state_file)
        self.backup_dir = Path(backup_dir)
        self.cache_size = cache_size
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        
        # State storage
        self._state: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._dirty_keys: set = set()
        
        # Async coordination  
        self._lock = asyncio.Lock()
        self._operation_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = StateMetrics()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.AsyncStateManager")
        
        # Ensure directories exist
        self.backup_dir.mkdir(exist_ok=True)
        
        # Default state structure
        self.default_state = {
            "charter": {
                "primary_goal": "innovation",
                "secondary_goal": "community_engagement",
                "constraints": ["maintain_quality", "ensure_security"]
            },
            "projects": {},
            "system_performance": {
                "total_cycles": 0,
                "successful_actions": 0,
                "failed_actions": 0,
                "learning_metrics": {
                    "decision_accuracy": 0.0,
                    "resource_efficiency": 0.0,
                    "goal_achievement": 0.0
                }
            },
            "task_queue": [],
            "external_context": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "market_trends": [],
                "security_alerts": [],
                "technology_updates": []
            },
            "intelligence_hub": {
                "events": [],
                "insights": [],
                "learning_data": {}
            },
            "version": "2.0.0",
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    async def initialize(self):
        """Initialize state manager and start background tasks."""
        self.logger.info("Initializing Async State Manager")
        
        # Load initial state
        await self._load_state()
        
        # Start background workers
        self._worker_task = asyncio.create_task(self._operation_worker())
        self._sync_task = asyncio.create_task(self._periodic_sync())
        
        self.logger.info("Async State Manager initialized successfully")
    
    async def shutdown(self):
        """Shutdown state manager gracefully."""
        self.logger.info("Shutting down Async State Manager")
        
        # Cancel background tasks
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        
        # Process remaining operations
        await self._flush_operations()
        
        # Final state save
        await self._sync_to_disk()
        
        self.logger.info("Async State Manager shutdown complete")
    
    async def _load_state(self):
        """Load state from disk asynchronously."""
        try:
            if self.state_file.exists():
                async with aiofiles.open(self.state_file, 'r') as f:
                    content = await f.read()
                    self._state = json.loads(content)
                    self.logger.info(f"Loaded state from {self.state_file}")
            else:
                self._state = self.default_state.copy()
                await self._sync_to_disk()
                self.logger.info("Created new state file with defaults")
                
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            self._state = self.default_state.copy()
    
    async def _sync_to_disk(self):
        """Sync current state to disk."""
        try:
            # Update timestamp
            self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Write atomically using temporary file
            temp_file = self.state_file.with_suffix('.tmp')
            
            async with aiofiles.open(temp_file, 'w') as f:
                content = json.dumps(self._state, indent=2, default=str)
                await f.write(content)
            
            # Atomic rename
            temp_file.replace(self.state_file)
            
            self.metrics.last_sync_time = datetime.now(timezone.utc)
            self._dirty_keys.clear()
            
            self.logger.debug(f"State synced to {self.state_file}")
            
        except Exception as e:
            self.logger.error(f"Error syncing state to disk: {e}")
            raise
    
    async def _operation_worker(self):
        """Background worker to process state operations."""
        batch = []
        
        while True:
            try:
                # Collect operations for batching
                try:
                    # Wait for first operation
                    operation = await asyncio.wait_for(self._operation_queue.get(), timeout=1.0)
                    batch.append(operation)
                    
                    # Collect additional operations for batching
                    while len(batch) < self.batch_size:
                        try:
                            operation = await asyncio.wait_for(self._operation_queue.get(), timeout=0.1)
                            batch.append(operation)
                        except asyncio.TimeoutError:
                            break
                    
                    # Process batch
                    await self._process_operation_batch(batch)
                    batch.clear()
                    
                except asyncio.TimeoutError:
                    # Timeout is normal, continue loop
                    continue
                    
            except asyncio.CancelledError:
                # Process remaining batch before shutting down
                if batch:
                    await self._process_operation_batch(batch)
                break
            except Exception as e:
                self.logger.error(f"Error in operation worker: {e}")
    
    async def _process_operation_batch(self, operations: List[StateOperation]):
        """Process a batch of state operations."""
        async with self._lock:
            for operation in operations:
                try:
                    start_time = time.time()
                    
                    if operation.operation_type == 'update':
                        await self._process_update(operation)
                    elif operation.operation_type == 'get':
                        await self._process_get(operation)
                    elif operation.operation_type == 'delete':
                        await self._process_delete(operation)
                    elif operation.operation_type == 'backup':
                        await self._process_backup(operation)
                    
                    # Update metrics
                    response_time = time.time() - start_time
                    self._update_metrics(operation.operation_type, response_time)
                    
                    # Call callback if provided
                    if operation.callback:
                        await operation.callback()
                        
                except Exception as e:
                    self.logger.error(f"Error processing operation {operation.operation_type}: {e}")
    
    async def _process_update(self, operation: StateOperation):
        """Process update operation."""
        key_path = operation.key_path
        value = operation.value
        
        # Navigate to nested key
        keys = key_path.split('.')
        current = self._state
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value
        current[keys[-1]] = value
        self._dirty_keys.add(key_path)
        
        # Update cache
        self._cache[key_path] = value
        
        # Enforce cache size limit
        if len(self._cache) > self.cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
    
    async def _process_get(self, operation: StateOperation):
        """Process get operation."""
        key_path = operation.key_path
        
        # Check cache first
        if key_path in self._cache:
            self.metrics.cache_hits += 1
            return self._cache[key_path]
        
        self.metrics.cache_misses += 1
        
        # Navigate to nested key
        keys = key_path.split('.')
        current = self._state
        
        try:
            for key in keys:
                current = current[key]
            
            # Cache the result
            self._cache[key_path] = current
            return current
            
        except (KeyError, TypeError):
            return None
    
    async def _process_delete(self, operation: StateOperation):
        """Process delete operation."""
        key_path = operation.key_path
        keys = key_path.split('.')
        current = self._state
        
        try:
            for key in keys[:-1]:
                current = current[key]
            
            if keys[-1] in current:
                del current[keys[-1]]
                self._dirty_keys.add(key_path)
                
                # Remove from cache
                if key_path in self._cache:
                    del self._cache[key_path]
                    
        except (KeyError, TypeError):
            pass  # Key doesn't exist, nothing to delete
    
    async def _process_backup(self, operation: StateOperation):
        """Process backup operation."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"state_backup_{timestamp}.json"
        
        try:
            async with aiofiles.open(backup_file, 'w') as f:
                content = json.dumps(self._state, indent=2, default=str)
                await f.write(content)
            
            self.logger.info(f"State backup created: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
    
    async def _periodic_sync(self):
        """Periodically sync dirty state to disk."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                
                if self._dirty_keys:
                    await self._sync_to_disk()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic sync: {e}")
    
    def _update_metrics(self, operation_type: str, response_time: float):
        """Update performance metrics."""
        self.metrics.total_operations += 1
        
        if operation_type == 'get':
            self.metrics.read_operations += 1
        else:
            self.metrics.write_operations += 1
        
        # Update average response time
        total_time = self.metrics.avg_response_time * (self.metrics.total_operations - 1)
        self.metrics.avg_response_time = (total_time + response_time) / self.metrics.total_operations
    
    async def _flush_operations(self):
        """Process all remaining operations in queue."""
        remaining_ops = []
        
        while not self._operation_queue.empty():
            try:
                op = self._operation_queue.get_nowait()
                remaining_ops.append(op)
            except asyncio.QueueEmpty:
                break
        
        if remaining_ops:
            await self._process_operation_batch(remaining_ops)
    
    # Public API methods
    
    async def update(self, key_path: str, value: Any, callback: Optional[Callable] = None):
        """Update state value asynchronously.
        
        Args:
            key_path: Dot-separated path to the value (e.g., 'projects.myproject.health_score')
            value: New value to set
            callback: Optional callback to execute after update
        """
        operation = StateOperation(
            operation_type='update',
            key_path=key_path,
            value=value,
            callback=callback
        )
        
        await self._operation_queue.put(operation)
        self.metrics.pending_operations += 1
    
    async def get(self, key_path: str, default: Any = None) -> Any:
        """Get state value asynchronously.
        
        Args:
            key_path: Dot-separated path to the value
            default: Default value if key doesn't exist
            
        Returns:
            Value at key_path or default
        """
        # For reads, we can access directly with cache
        async with self._lock:
            result = await self._process_get(StateOperation('get', key_path))
            return result if result is not None else default
    
    async def delete(self, key_path: str):
        """Delete state value asynchronously.
        
        Args:
            key_path: Dot-separated path to the value to delete
        """
        operation = StateOperation(
            operation_type='delete',
            key_path=key_path
        )
        
        await self._operation_queue.put(operation)
    
    async def backup(self):
        """Create a backup of current state."""
        operation = StateOperation(operation_type='backup', key_path='')
        await self._operation_queue.put(operation)
    
    async def get_full_state(self) -> Dict[str, Any]:
        """Get complete state dictionary."""
        async with self._lock:
            return self._state.copy()
    
    async def update_full_state(self, new_state: Dict[str, Any]):
        """Replace entire state (use carefully)."""
        async with self._lock:
            self._state = new_state.copy()
            self._cache.clear()
            self._dirty_keys.add('*')  # Mark everything as dirty
            await self._sync_to_disk()
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for atomic state transactions."""
        async with self._lock:
            # Create snapshot
            snapshot = self._state.copy()
            try:
                yield self
                # If we reach here, transaction succeeded
                await self._sync_to_disk()
            except Exception:
                # Rollback on error
                self._state = snapshot
                self._cache.clear()
                raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        self.metrics.pending_operations = self._operation_queue.qsize()
        return asdict(self.metrics)
    
    async def optimize_cache(self):
        """Optimize cache by removing unused entries."""
        # Simple optimization: keep only frequently accessed keys
        # In a real implementation, you'd track access patterns
        if len(self._cache) > self.cache_size * 0.8:
            # Remove half the cache entries
            keys_to_remove = list(self._cache.keys())[::2]
            for key in keys_to_remove:
                del self._cache[key]
            
            self.logger.debug(f"Cache optimized: removed {len(keys_to_remove)} entries")


# Global instance
_global_async_state_manager: Optional[AsyncStateManager] = None


async def get_async_state_manager() -> AsyncStateManager:
    """Get or create global async state manager."""
    global _global_async_state_manager
    if _global_async_state_manager is None:
        _global_async_state_manager = AsyncStateManager()
        await _global_async_state_manager.initialize()
    return _global_async_state_manager


# Convenience functions
async def async_state_update(key_path: str, value: Any):
    """Convenience function for state updates."""
    manager = await get_async_state_manager()
    await manager.update(key_path, value)


async def async_state_get(key_path: str, default: Any = None) -> Any:
    """Convenience function for state retrieval."""
    manager = await get_async_state_manager()
    return await manager.get(key_path, default)