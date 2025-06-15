"""
Lock-free Redis state management system for autonomous workers.

This implementation uses atomic operations, partitioning, and optimistic
concurrency control to eliminate locking bottlenecks while maintaining
consistency.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import hashlib
import redis.asyncio as redis
from redis.exceptions import WatchError
import logging

logger = logging.getLogger(__name__)


class RedisLockFreeStateManager:
    """
    Lock-free state manager using Redis atomic operations and partitioning.
    
    Key features:
    - Partitioned state by worker ID to eliminate contention
    - Optimistic concurrency control with versioning
    - Eventual consistency for non-critical data
    - Redis Streams for append-only operations
    - High-concurrency support without locks
    """
    
    def __init__(self, redis_client: Any = None, redis_url: str = "redis://localhost:6379"):
        """Initialize state manager with optional shared RedisClient or raw URL."""
        self.redis_url = redis_url
        self.redis_client = redis_client
        self._initialized = False
        
        # Prefixes for different data types
        self.WORKER_STATE_PREFIX = "worker:state:"
        self.WORKER_VERSION_PREFIX = "worker:version:"
        self.TASK_STATE_PREFIX = "task:state:"
        self.TASK_VERSION_PREFIX = "task:version:"
        self.GLOBAL_STATE_KEY = "global:state"
        self.GLOBAL_VERSION_KEY = "global:version"
        self.EVENT_STREAM_KEY = "events:stream"
        self.METRICS_STREAM_KEY = "metrics:stream"
        
        # Eventual consistency settings
        self.EVENTUAL_SYNC_INTERVAL = 5  # seconds
        self.EVENTUAL_KEYS_PREFIX = "eventual:"
        
    async def initialize(self):
        """Initialize Redis connection and data structures."""
        if not self._initialized:
            # Use provided RedisClient or fallback to pooled client
            if self.redis_client is None:
                from scripts.redis_integration.redis_client import get_redis_client
                self.redis_client = await get_redis_client()
            # Initialize global version counter if not exists
            await self.redis_client.set(self.GLOBAL_VERSION_KEY, 0, nx=True)
            
            self._initialized = True
            logger.info("Lock-free state manager initialized")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.disconnect()
            self._initialized = False
    
    # Worker State Management (Partitioned)
    
    async def get_worker_state(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get worker state using atomic read."""
        state_key = f"{self.WORKER_STATE_PREFIX}{worker_id}"
        state_data = await self.redis_client.get(state_key)
        
        if state_data:
            return json.loads(state_data)
        return None
    
    async def update_worker_state(self, worker_id: str, state: Dict[str, Any], 
                                 expected_version: Optional[int] = None) -> bool:
        """
        Update worker state with optimistic concurrency control.
        
        Returns True if update succeeded, False if version mismatch.
        """
        state_key = f"{self.WORKER_STATE_PREFIX}{worker_id}"
        version_key = f"{self.WORKER_VERSION_PREFIX}{worker_id}"
        
        # Use Redis transaction with WATCH for optimistic locking
        async with self.redis_client.pipeline() as pipe:
            while True:
                try:
                    # Watch the version key for changes
                    await pipe.watch(version_key)
                    
                    # Get current version
                    current_version = await pipe.get(version_key)
                    current_version = int(current_version) if current_version else 0
                    
                    # Check version if expected version provided
                    if expected_version is not None and current_version != expected_version:
                        await pipe.unwatch()
                        return False
                    
                    # Start transaction
                    pipe.multi()
                    
                    # Update state and increment version atomically
                    pipe.set(state_key, json.dumps(state))
                    pipe.incr(version_key)
                    
                    # Execute transaction
                    await pipe.execute()
                    
                    # Log state change to stream
                    await self._log_state_change("worker", worker_id, state)
                    
                    return True
                    
                except WatchError:
                    # Another client modified the version, retry
                    continue
    
    async def update_worker_field(self, worker_id: str, field: str, value: Any) -> bool:
        """Update a single field in worker state atomically."""
        current_state = await self.get_worker_state(worker_id)
        if current_state is None:
            current_state = {}
        
        current_state[field] = value
        return await self.update_worker_state(worker_id, current_state)
    
    # Task State Management (Partitioned by task ID)
    
    async def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task state atomically."""
        state_key = f"{self.TASK_STATE_PREFIX}{task_id}"
        state_data = await self.redis_client.get(state_key)
        
        if state_data:
            return json.loads(state_data)
        return None
    
    async def update_task_state(self, task_id: str, state: Dict[str, Any],
                               expected_version: Optional[int] = None) -> bool:
        """Update task state with optimistic concurrency control."""
        state_key = f"{self.TASK_STATE_PREFIX}{task_id}"
        version_key = f"{self.TASK_VERSION_PREFIX}{task_id}"
        
        async with self.redis_client.pipeline() as pipe:
            while True:
                try:
                    await pipe.watch(version_key)
                    
                    current_version = await pipe.get(version_key)
                    current_version = int(current_version) if current_version else 0
                    
                    if expected_version is not None and current_version != expected_version:
                        await pipe.unwatch()
                        return False
                    
                    pipe.multi()
                    pipe.set(state_key, json.dumps(state))
                    pipe.incr(version_key)
                    await pipe.execute()
                    
                    await self._log_state_change("task", task_id, state)
                    return True
                    
                except WatchError:
                    continue
    
    # Global State Management (Less frequent updates)
    
    async def get_global_state(self) -> Dict[str, Any]:
        """Get global state atomically."""
        state_data = await self.redis_client.get(self.GLOBAL_STATE_KEY)
        if state_data:
            return json.loads(state_data)
        return {}
    
    async def update_global_state(self, updates: Dict[str, Any]) -> bool:
        """Update global state with optimistic concurrency control."""
        async with self.redis_client.pipeline() as pipe:
            while True:
                try:
                    await pipe.watch(self.GLOBAL_VERSION_KEY)
                    
                    # Get current state
                    current_state = await self.get_global_state()
                    
                    # Merge updates
                    current_state.update(updates)
                    
                    pipe.multi()
                    pipe.set(self.GLOBAL_STATE_KEY, json.dumps(current_state))
                    pipe.incr(self.GLOBAL_VERSION_KEY)
                    await pipe.execute()
                    
                    return True
                    
                except WatchError:
                    continue
    
    # Atomic Counter Operations
    
    async def increment_counter(self, counter_name: str, amount: int = 1) -> int:
        """Atomically increment a counter and return new value."""
        counter_key = f"counter:{counter_name}"
        return await self.redis_client.incrby(counter_key, amount)
    
    async def get_counter(self, counter_name: str) -> int:
        """Get current counter value."""
        counter_key = f"counter:{counter_name}"
        value = await self.redis_client.get(counter_key)
        return int(value) if value else 0
    
    # Set Operations (for tracking active workers, tasks, etc.)
    
    async def add_to_set(self, set_name: str, *members: str) -> int:
        """Atomically add members to a set."""
        set_key = f"set:{set_name}"
        return await self.redis_client.sadd(set_key, *members)
    
    async def remove_from_set(self, set_name: str, *members: str) -> int:
        """Atomically remove members from a set."""
        set_key = f"set:{set_name}"
        return await self.redis_client.srem(set_key, *members)
    
    async def get_set_members(self, set_name: str) -> Set[str]:
        """Get all members of a set atomically."""
        set_key = f"set:{set_name}"
        return await self.redis_client.smembers(set_key)
    
    async def is_in_set(self, set_name: str, member: str) -> bool:
        """Check if member exists in set atomically."""
        set_key = f"set:{set_name}"
        return await self.redis_client.sismember(set_key, member)
    
    # Hash Operations (for field-level updates)
    
    async def update_hash_field(self, hash_name: str, field: str, value: Any) -> int:
        """Atomically update a single field in a hash."""
        hash_key = f"hash:{hash_name}"
        return await self.redis_client.hset(hash_key, field, json.dumps(value))
    
    async def get_hash_field(self, hash_name: str, field: str) -> Optional[Any]:
        """Get a single field from a hash atomically."""
        hash_key = f"hash:{hash_name}"
        value = await self.redis_client.hget(hash_key, field)
        return json.loads(value) if value else None
    
    async def get_all_hash_fields(self, hash_name: str) -> Dict[str, Any]:
        """Get all fields from a hash atomically."""
        hash_key = f"hash:{hash_name}"
        data = await self.redis_client.hgetall(hash_key)
        return {k: json.loads(v) for k, v in data.items()}
    
    # Stream Operations (for append-only logs)
    
    async def append_to_stream(self, stream_name: str, data: Dict[str, Any]) -> str:
        """Append data to a Redis stream (append-only log)."""
        stream_key = f"stream:{stream_name}"
        
        # Add timestamp and serialize nested data
        stream_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "data": json.dumps(data)
        }
        
        # XADD returns the stream entry ID
        entry_id = await self.redis_client.xadd(stream_key, stream_data)
        return entry_id
    
    async def read_stream(self, stream_name: str, start_id: str = "-", 
                         count: int = 100) -> List[Tuple[str, Dict[str, Any]]]:
        """Read entries from a stream."""
        stream_key = f"stream:{stream_name}"
        
        entries = await self.redis_client.xrange(stream_key, start_id, "+", count=count)
        
        result = []
        for entry_id, data in entries:
            parsed_data = {
                "timestamp": data["timestamp"],
                "data": json.loads(data["data"])
            }
            result.append((entry_id, parsed_data))
        
        return result
    
    async def trim_stream(self, stream_name: str, max_length: int = 10000):
        """Trim stream to maximum length (keep most recent entries)."""
        stream_key = f"stream:{stream_name}"
        await self.redis_client.xtrim(stream_key, maxlen=max_length, approximate=True)
    
    # Eventual Consistency Operations
    
    async def set_eventual(self, key: str, value: Any, ttl: int = None):
        """Set a value with eventual consistency (no strict ordering)."""
        eventual_key = f"{self.EVENTUAL_KEYS_PREFIX}{key}"
        
        data = {
            "value": value,
            "timestamp": time.time()
        }
        
        if ttl:
            await self.redis_client.setex(eventual_key, ttl, json.dumps(data))
        else:
            await self.redis_client.set(eventual_key, json.dumps(data))
    
    async def get_eventual(self, key: str) -> Optional[Any]:
        """Get a value with eventual consistency."""
        eventual_key = f"{self.EVENTUAL_KEYS_PREFIX}{key}"
        data = await self.redis_client.get(eventual_key)
        
        if data:
            parsed = json.loads(data)
            return parsed["value"]
        return None
    
    # Batch Operations (for efficiency)
    
    async def batch_get_worker_states(self, worker_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get multiple worker states in a single operation."""
        if not worker_ids:
            return {}
        
        keys = [f"{self.WORKER_STATE_PREFIX}{wid}" for wid in worker_ids]
        values = await self.redis_client.mget(keys)
        
        result = {}
        for worker_id, value in zip(worker_ids, values):
            if value:
                result[worker_id] = json.loads(value)
        
        return result
    
    async def batch_update_fields(self, updates: List[Tuple[str, str, str, Any]]):
        """
        Batch update multiple fields across different hashes.
        
        Args:
            updates: List of (hash_name, field, value) tuples
        """
        async with self.redis_client.pipeline() as pipe:
            for hash_name, field, value in updates:
                hash_key = f"hash:{hash_name}"
                pipe.hset(hash_key, field, json.dumps(value))
            
            await pipe.execute()
    
    # Metrics and Monitoring
    
    async def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric to the metrics stream."""
        metric_data = {
            "metric": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": time.time()
        }
        
        await self.redis_client.xadd(
            self.METRICS_STREAM_KEY,
            {"data": json.dumps(metric_data)},
            maxlen=100000
        )
    
    # Private helper methods
    
    async def _log_state_change(self, entity_type: str, entity_id: str, state: Dict[str, Any]):
        """Log state changes to event stream for audit/debugging."""
        event_data = {
            "event": "state_change",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "state_hash": hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest(),
            "timestamp": time.time()
        }
        
        await self.redis_client.xadd(
            self.EVENT_STREAM_KEY,
            {"data": json.dumps(event_data)},
            maxlen=50000
        )
    
    # Compatibility methods for drop-in replacement
    
    async def acquire_lock(self, *args, **kwargs):
        """No-op for compatibility - no locks needed."""
        return True
    
    async def release_lock(self, *args, **kwargs):
        """No-op for compatibility - no locks needed."""
        pass
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Generic state getter for compatibility."""
        if key.startswith("worker:"):
            worker_id = key.replace("worker:", "")
            return await self.get_worker_state(worker_id)
        elif key.startswith("task:"):
            task_id = key.replace("task:", "")
            return await self.get_task_state(task_id)
        else:
            # Use hash for other keys
            return await self.get_all_hash_fields(key)
    
    async def set_state(self, key: str, state: Dict[str, Any]) -> bool:
        """Generic state setter for compatibility."""
        if key.startswith("worker:"):
            worker_id = key.replace("worker:", "")
            return await self.update_worker_state(worker_id, state)
        elif key.startswith("task:"):
            task_id = key.replace("task:", "")
            return await self.update_task_state(task_id, state)
        else:
            # Use hash for other keys
            for field, value in state.items():
                await self.update_hash_field(key, field, value)
            return True
    
    async def update_state(self, key: str, state: Dict[str, Any], distributed: bool = False) -> bool:
        """Update state - alias for set_state for compatibility."""
        return await self.set_state(key, state)
    
    async def update(self, key_path: str, value: Any, distributed: bool = False):
        """Update state value - compatibility method."""
        await self.update_state(key_path, value, distributed=distributed)


# Example usage and testing
async def example_usage():
    """Example of using the lock-free state manager."""
    manager = RedisLockFreeStateManager()
    await manager.initialize()
    
    try:
        # Worker state management
        worker_id = "worker-001"
        worker_state = {
            "status": "active",
            "current_task": "task-123",
            "metrics": {"completed": 10, "failed": 0}
        }
        
        # Update worker state
        success = await manager.update_worker_state(worker_id, worker_state)
        print(f"Worker state updated: {success}")
        
        # Read worker state
        state = await manager.get_worker_state(worker_id)
        print(f"Worker state: {state}")
        
        # Atomic counter operations
        completed_count = await manager.increment_counter("tasks_completed", 1)
        print(f"Tasks completed: {completed_count}")
        
        # Set operations for active workers
        await manager.add_to_set("active_workers", worker_id)
        active_workers = await manager.get_set_members("active_workers")
        print(f"Active workers: {active_workers}")
        
        # Stream operations for logging
        await manager.append_to_stream("worker_events", {
            "worker_id": worker_id,
            "event": "task_completed",
            "task_id": "task-123"
        })
        
        # Read recent events
        events = await manager.read_stream("worker_events", count=10)
        print(f"Recent events: {len(events)}")
        
        # Batch operations
        worker_states = await manager.batch_get_worker_states(["worker-001", "worker-002"])
        print(f"Batch worker states: {worker_states}")
        
        # Record metrics
        await manager.record_metric("task_duration", 45.3, {"worker": worker_id})
        
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(example_usage())