"""
Redis-Backed Async State Manager

Enhanced async state management with Redis backend, distributed coordination,
conflict resolution, and seamless migration from file-based storage.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import asynccontextmanager

from scripts.redis_integration import (
    RedisClient,
    RedisStateManager,
    RedisPubSubManager,
    RedisLocksManager,
    get_redis_client
)
try:
    from scripts.async_state_manager import AsyncStateManager, StateOperation, StateMetrics
except ImportError:
    from async_state_manager import AsyncStateManager, StateOperation, StateMetrics


@dataclass
class RedisStateOperation(StateOperation):
    """Enhanced state operation with Redis-specific features."""
    distributed: bool = False
    conflict_resolution: str = "last_write_wins"  # last_write_wins, merge, abort
    lock_required: bool = False
    component_id: str = "default"
    version: Optional[int] = None


@dataclass
class RedisEnhancedStateMetrics(StateMetrics):
    """Enhanced state metrics with Redis-specific tracking."""
    redis_operations: int = 0
    redis_conflicts: int = 0
    redis_locks_acquired: int = 0
    redis_pubsub_events: int = 0
    distributed_sync_operations: int = 0
    conflict_resolutions: int = 0
    background_sync_count: int = 0
    network_latency_ms: float = 0.0
    
    @property
    def redis_success_rate(self) -> float:
        """Calculate Redis operation success rate."""
        total_redis = self.redis_operations
        failed_redis = self.redis_conflicts
        return (total_redis - failed_redis) / max(total_redis, 1)


class RedisAsyncStateManager:
    """Advanced Redis-backed async state manager with distributed features."""
    
    def __init__(self,
                 redis_client: Optional[RedisClient] = None,
                 fallback_manager: Optional[AsyncStateManager] = None,
                 component_id: str = "default",
                 namespace: str = "state",
                 dual_write_mode: bool = True,
                 migration_mode: str = "gradual",  # gradual, immediate, readonly
                 enable_pubsub: bool = True,
                 enable_locking: bool = True,
                 conflict_resolution: str = "last_write_wins",
                 sync_interval: int = 30,
                 batch_size: int = 10):
        """Initialize Redis async state manager.
        
        Args:
            redis_client: Redis client instance
            fallback_manager: Fallback file-based state manager
            component_id: Unique component identifier
            namespace: Redis namespace for state isolation
            dual_write_mode: Write to both Redis and fallback
            migration_mode: Migration strategy
            enable_pubsub: Enable real-time state synchronization
            enable_locking: Enable distributed locking
            conflict_resolution: Default conflict resolution strategy
            sync_interval: Sync interval in seconds
            batch_size: Operation batch size
        """
        self.redis_client = redis_client
        self.fallback_manager = fallback_manager
        self.component_id = component_id
        self.namespace = namespace
        self.dual_write_mode = dual_write_mode
        self.migration_mode = migration_mode
        self.enable_pubsub = enable_pubsub
        self.enable_locking = enable_locking
        self.conflict_resolution = conflict_resolution
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        
        # Redis components
        self.redis_state: Optional[RedisStateManager] = None
        self.pubsub_manager: Optional[RedisPubSubManager] = None
        self.locks_manager: Optional[RedisLocksManager] = None
        
        # Enhanced state management
        self._local_state: Dict[str, Any] = {}
        self._pending_operations: asyncio.Queue = asyncio.Queue()
        self._state_lock = asyncio.Lock()
        self._shutdown = False
        
        # Enhanced metrics
        self.metrics = RedisEnhancedStateMetrics()
        
        # Background tasks
        self._worker_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._migration_task: Optional[asyncio.Task] = None
        self._pubsub_task: Optional[asyncio.Task] = None
        
        # Change listeners
        self._change_listeners: List[Callable] = []
        
        # Migration tracking
        self._migration_progress = {
            'total_keys': 0,
            'migrated_keys': 0,
            'failed_migrations': 0,
            'start_time': None,
            'completed': False
        }
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.RedisAsyncStateManager")
    
    async def initialize(self, shared_redis: RedisClient = None):
        """Initialize Redis state manager and start background tasks."""
        try:
            # Use shared Redis client if provided
            if self.redis_client is None:
                self.redis_client = shared_redis or await get_redis_client()
            
            # Initialize Redis components
            self.redis_state = RedisStateManager(
                self.redis_client,
                component_id=self.component_id
            )
            await self.redis_state.start()
            
            if self.enable_pubsub:
                self.pubsub_manager = RedisPubSubManager(
                    self.redis_client,
                    instance_id=self.component_id
                )
                await self.pubsub_manager.start()
                
                # Subscribe to state change notifications
                await self.pubsub_manager.subscribe(
                    f"state_changes:{self.namespace}",
                    self._handle_remote_state_change
                )
            
            if self.enable_locking:
                self.locks_manager = RedisLocksManager(
                    self.redis_client,
                    namespace=f"{self.namespace}_locks",
                    instance_id=self.component_id
                )
                await self.locks_manager.start()
            
            # Load initial state
            await self._load_initial_state()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info(f"Redis Async State Manager initialized (component: {self.component_id}, mode: {self.migration_mode})")
            
        except Exception as e:
            self.logger.error(f"Error initializing Redis state manager: {e}")
            raise
    
    async def _load_initial_state(self):
        """Load initial state from Redis and fallback sources."""
        try:
            # Load from Redis first (if not in readonly mode)
            if self.migration_mode != "readonly" and self.redis_state:
                redis_state = await self.redis_state.get_state(self.component_id)
                if redis_state:
                    self._local_state = redis_state.copy()
                    self.logger.info("Loaded initial state from Redis")
                    return
            
            # Load from fallback manager
            if self.fallback_manager:
                fallback_state = await self.fallback_manager.get_full_state()
                if fallback_state:
                    self._local_state = fallback_state.copy()
                    self.logger.info("Loaded initial state from fallback manager")
                    
                    # Migrate to Redis if dual-write enabled
                    if self.dual_write_mode and self.redis_state:
                        await self._migrate_state_to_redis(fallback_state)
            
        except Exception as e:
            self.logger.error(f"Error loading initial state: {e}")
    
    async def _start_background_tasks(self):
        """Start background processing tasks."""
        self._worker_task = asyncio.create_task(self._operation_worker())
        self._sync_task = asyncio.create_task(self._periodic_sync())
        
        if self.migration_mode == "gradual" and self.fallback_manager:
            self._migration_task = asyncio.create_task(self._gradual_migration_worker())
        
        if self.enable_pubsub:
            self._pubsub_task = asyncio.create_task(self._pubsub_monitor())
    
    async def update(self, key_path: str, value: Any, 
                    distributed: bool = False,
                    conflict_resolution: str = None,
                    lock_required: bool = None,
                    callback: Optional[Callable] = None):
        """Update state value with Redis backend and conflict resolution.
        
        Args:
            key_path: Dot-separated path to the value
            value: New value to set
            distributed: Whether to coordinate across instances
            conflict_resolution: Conflict resolution strategy
            lock_required: Whether to acquire distributed lock
            callback: Optional callback to execute after update
        """
        operation = RedisStateOperation(
            operation_type='update',
            key_path=key_path,
            value=value,
            distributed=distributed if distributed is not None else False,
            conflict_resolution=conflict_resolution or self.conflict_resolution,
            lock_required=lock_required if lock_required is not None else self.enable_locking,
            component_id=self.component_id,
            callback=callback
        )
        
        await self._pending_operations.put(operation)
        self.metrics.pending_operations += 1
    
    async def get(self, key_path: str, default: Any = None, 
                 use_cache: bool = True) -> Any:
        """Get state value with intelligent caching and fallback.
        
        Args:
            key_path: Dot-separated path to the value
            default: Default value if key doesn't exist
            use_cache: Whether to use local cache
            
        Returns:
            Value at key_path or default
        """
        start_time = time.time()
        self.metrics.read_operations += 1
        
        # Try local cache first
        if use_cache:
            async with self._state_lock:
                local_value = self._get_nested_value(self._local_state, key_path)
                if local_value is not None:
                    self.metrics.cache_hits += 1
                    return local_value
                self.metrics.cache_misses += 1
        
        # Try Redis
        if self.redis_state and self.migration_mode != "readonly":
            try:
                redis_value = await self.redis_state.get_state(
                    self.component_id, 
                    key_path.split('.'),
                    use_cache=False
                )
                
                if redis_value is not None:
                    # Update local cache
                    async with self._state_lock:
                        self._set_nested_value(self._local_state, key_path, redis_value)
                    
                    self.metrics.redis_operations += 1
                    response_time = (time.time() - start_time) * 1000
                    self.metrics.network_latency_ms = (
                        (self.metrics.network_latency_ms * self.metrics.redis_operations + response_time) /
                        (self.metrics.redis_operations + 1)
                    )
                    
                    return redis_value
                    
            except Exception as e:
                self.logger.warning(f"Redis get error, trying fallback: {e}")
        
        # Try fallback manager
        if self.fallback_manager:
            try:
                fallback_value = await self.fallback_manager.get(key_path, default)
                
                # Update local cache
                if fallback_value is not None:
                    async with self._state_lock:
                        self._set_nested_value(self._local_state, key_path, fallback_value)
                
                return fallback_value
                
            except Exception as e:
                self.logger.error(f"Fallback get error: {e}")
        
        return default
    
    async def delete(self, key_path: str, distributed: bool = False):
        """Delete state value with distributed coordination.
        
        Args:
            key_path: Dot-separated path to the value to delete
            distributed: Whether to coordinate across instances
        """
        operation = RedisStateOperation(
            operation_type='delete',
            key_path=key_path,
            distributed=distributed,
            component_id=self.component_id
        )
        
        await self._pending_operations.put(operation)
    
    async def _operation_worker(self):
        """Background worker to process state operations with batching."""
        batch = []
        
        while not self._shutdown:
            try:
                # Collect operations for batching
                try:
                    operation = await asyncio.wait_for(
                        self._pending_operations.get(), 
                        timeout=1.0
                    )
                    batch.append(operation)
                    
                    # Collect additional operations
                    while len(batch) < self.batch_size:
                        try:
                            operation = await asyncio.wait_for(
                                self._pending_operations.get(), 
                                timeout=0.1
                            )
                            batch.append(operation)
                        except asyncio.TimeoutError:
                            break
                    
                    # Process batch
                    await self._process_operation_batch(batch)
                    batch.clear()
                    
                except asyncio.TimeoutError:
                    continue
                    
            except asyncio.CancelledError:
                if batch:
                    await self._process_operation_batch(batch)
                break
            except Exception as e:
                self.logger.error(f"Error in operation worker: {e}")
    
    async def _process_operation_batch(self, operations: List[RedisStateOperation]):
        """Process batch of state operations with conflict resolution."""
        for operation in operations:
            try:
                start_time = time.time()
                
                if operation.operation_type == 'update':
                    await self._process_redis_update(operation)
                elif operation.operation_type == 'delete':
                    await self._process_redis_delete(operation)
                
                # Update metrics
                response_time = time.time() - start_time
                self._update_metrics(operation.operation_type, response_time)
                
                # Execute callback
                if operation.callback:
                    await operation.callback()
                
                self.metrics.pending_operations = max(0, self.metrics.pending_operations - 1)
                
            except Exception as e:
                self.logger.error(f"Error processing operation: {e}")
    
    async def _process_redis_update(self, operation: RedisStateOperation):
        """Process update operation with Redis backend and locking."""
        key_path = operation.key_path
        value = operation.value
        
        try:
            # Acquire distributed lock if required
            lock_key = None
            if operation.lock_required and self.locks_manager:
                lock_key = f"state:{self.component_id}:{key_path}"
                
                async with self.locks_manager.lock(lock_key, timeout=30.0):
                    await self._execute_redis_update(operation)
                    self.metrics.redis_locks_acquired += 1
            else:
                await self._execute_redis_update(operation)
                
        except Exception as e:
            self.logger.error(f"Error in Redis update: {e}")
            self.metrics.redis_conflicts += 1
            
            # Try fallback on Redis failure
            if self.fallback_manager and self.dual_write_mode:
                await self.fallback_manager.update(key_path, value)
    
    async def _execute_redis_update(self, operation: RedisStateOperation):
        """Execute the actual Redis update with conflict resolution."""
        key_path = operation.key_path
        value = operation.value
        
        # Update Redis (if not readonly mode)
        if self.migration_mode != "readonly" and self.redis_state:
            success = await self.redis_state.set_state(
                value,
                self.component_id,
                key_path.split('.'),
                merge=(operation.conflict_resolution == "merge"),
                atomic=True
            )
            
            if success:
                self.metrics.redis_operations += 1
            else:
                self.metrics.redis_conflicts += 1
                self.logger.warning(f"Redis update conflict for {key_path}")
        
        # Update fallback (if dual-write enabled)
        if self.fallback_manager and self.dual_write_mode:
            await self.fallback_manager.update(key_path, value)
        
        # Update local state
        async with self._state_lock:
            self._set_nested_value(self._local_state, key_path, value)
        
        # Notify via pub/sub if distributed
        if operation.distributed and self.pubsub_manager:
            await self._broadcast_state_change(operation)
    
    async def _process_redis_delete(self, operation: RedisStateOperation):
        """Process delete operation with Redis backend."""
        key_path = operation.key_path
        
        try:
            # Delete from Redis
            if self.migration_mode != "readonly" and self.redis_state:
                await self.redis_state.delete_state(
                    self.component_id,
                    key_path.split('.')
                )
                self.metrics.redis_operations += 1
            
            # Delete from fallback
            if self.fallback_manager and self.dual_write_mode:
                await self.fallback_manager.delete(key_path)
            
            # Delete from local state
            async with self._state_lock:
                self._delete_nested_value(self._local_state, key_path)
            
            # Notify via pub/sub if distributed
            if operation.distributed and self.pubsub_manager:
                await self._broadcast_state_change(operation)
                
        except Exception as e:
            self.logger.error(f"Error in Redis delete: {e}")
    
    async def _broadcast_state_change(self, operation: RedisStateOperation):
        """Broadcast state change via pub/sub."""
        try:
            if self.pubsub_manager:
                change_data = {
                    'operation': operation.operation_type,
                    'key_path': operation.key_path,
                    'value': operation.value,
                    'component_id': operation.component_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                await self.pubsub_manager.publish(
                    f"state_changes:{self.namespace}",
                    change_data,
                    message_type="state_change"
                )
                
                self.metrics.redis_pubsub_events += 1
                
        except Exception as e:
            self.logger.error(f"Error broadcasting state change: {e}")
    
    async def _handle_remote_state_change(self, message):
        """Handle state change notification from other instances."""
        try:
            if message.sender_id == self.component_id:
                return  # Ignore our own messages
            
            change_data = message.data
            key_path = change_data.get('key_path')
            value = change_data.get('value')
            operation_type = change_data.get('operation')
            
            # Apply remote change to local state
            async with self._state_lock:
                if operation_type == 'update':
                    self._set_nested_value(self._local_state, key_path, value)
                elif operation_type == 'delete':
                    self._delete_nested_value(self._local_state, key_path)
            
            # Notify local change listeners
            for listener in self._change_listeners:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(change_data)
                    else:
                        listener(change_data)
                except Exception as e:
                    self.logger.error(f"Error in change listener: {e}")
            
            self.logger.debug(f"Applied remote state change: {key_path}")
            
        except Exception as e:
            self.logger.error(f"Error handling remote state change: {e}")
    
    async def _periodic_sync(self):
        """Periodically sync with Redis and handle conflicts."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.sync_interval)
                
                if self.redis_state:
                    await self._sync_with_redis()
                    self.metrics.background_sync_count += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic sync: {e}")
    
    async def _sync_with_redis(self):
        """Synchronize local state with Redis."""
        try:
            if not self.redis_state:
                return
            
            # Get current Redis state
            redis_state = await self.redis_state.get_state(self.component_id)
            
            if redis_state:
                async with self._state_lock:
                    # Simple last-write-wins merge for now
                    # In production, you'd implement sophisticated conflict resolution
                    self._local_state.update(redis_state)
                
                self.metrics.distributed_sync_operations += 1
                
        except Exception as e:
            self.logger.error(f"Error syncing with Redis: {e}")
    
    async def _gradual_migration_worker(self):
        """Background worker for gradual migration from fallback to Redis."""
        if not self.fallback_manager or not self.redis_state:
            return
        
        self.logger.info("Starting gradual state migration to Redis")
        self._migration_progress['start_time'] = datetime.now(timezone.utc)
        
        try:
            # Get full state from fallback
            fallback_state = await self.fallback_manager.get_full_state()
            self._migration_progress['total_keys'] = len(self._flatten_dict(fallback_state))
            
            # Migrate keys gradually
            await self._migrate_state_to_redis(fallback_state)
            
            self._migration_progress['completed'] = True
            migration_time = (datetime.now(timezone.utc) - self._migration_progress['start_time']).total_seconds()
            
            self.logger.info(f"State migration completed in {migration_time:.1f}s. "
                           f"Migrated: {self._migration_progress['migrated_keys']}, "
                           f"Failed: {self._migration_progress['failed_migrations']}")
            
        except Exception as e:
            self.logger.error(f"Error in migration worker: {e}")
    
    async def _migrate_state_to_redis(self, state: Dict[str, Any]):
        """Migrate state dictionary to Redis."""
        try:
            if not self.redis_state:
                return
            
            # Flatten state for migration tracking
            flattened = self._flatten_dict(state)
            
            for key_path, value in flattened.items():
                try:
                    await self.redis_state.set_state(
                        value,
                        self.component_id,
                        key_path.split('.'),
                        atomic=True
                    )
                    
                    self._migration_progress['migrated_keys'] += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    self._migration_progress['failed_migrations'] += 1
                    self.logger.error(f"Error migrating key {key_path}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in state migration: {e}")
    
    async def _pubsub_monitor(self):
        """Monitor pub/sub system health."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.pubsub_manager:
                    stats = self.pubsub_manager.get_statistics()
                    self.logger.debug(f"Pub/Sub stats: {stats}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in pub/sub monitor: {e}")
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get nested value from data structure."""
        keys = key_path.split('.')
        current = data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return None
    
    def _set_nested_value(self, data: Dict[str, Any], key_path: str, value: Any):
        """Set nested value in data structure."""
        keys = key_path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _delete_nested_value(self, data: Dict[str, Any], key_path: str):
        """Delete nested value from data structure."""
        keys = key_path.split('.')
        current = data
        
        try:
            for key in keys[:-1]:
                current = current[key]
            
            if keys[-1] in current:
                del current[keys[-1]]
        except (KeyError, TypeError):
            pass
    
    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
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
    
    # Public API methods
    
    def add_change_listener(self, listener: Callable):
        """Add state change listener."""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable):
        """Remove state change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    async def get_full_state(self) -> Dict[str, Any]:
        """Get complete state dictionary."""
        async with self._state_lock:
            return self._local_state.copy()
    
    async def update_full_state(self, new_state: Dict[str, Any]):
        """Replace entire state (use carefully)."""
        async with self._state_lock:
            self._local_state = new_state.copy()
        
        # Update Redis
        if self.redis_state and self.migration_mode != "readonly":
            await self.redis_state.set_state(
                new_state,
                self.component_id,
                atomic=True
            )
        
        # Update fallback
        if self.fallback_manager and self.dual_write_mode:
            await self.fallback_manager.update_full_state(new_state)
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for atomic state transactions with distributed locking."""
        if self.locks_manager:
            transaction_lock = f"transaction:{self.component_id}:{int(time.time())}"
            
            async with self.locks_manager.lock(transaction_lock, timeout=60.0):
                async with self._state_lock:
                    snapshot = self._local_state.copy()
                    try:
                        yield self
                        # Transaction succeeded - changes already applied
                    except Exception:
                        # Rollback
                        self._local_state = snapshot
                        raise
        else:
            # Fallback to local transaction
            async with self._state_lock:
                snapshot = self._local_state.copy()
                try:
                    yield self
                except Exception:
                    self._local_state = snapshot
                    raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = asdict(self.metrics)
        base_metrics['pending_operations'] = self._pending_operations.qsize()
        
        # Add migration progress
        base_metrics['migration'] = self._migration_progress.copy()
        
        # Add Redis component metrics
        if self.redis_state:
            redis_stats = asyncio.create_task(self.redis_state.get_statistics())
            base_metrics['redis_state'] = redis_stats
        
        if self.locks_manager:
            base_metrics['locks'] = self.locks_manager.get_statistics()
        
        if self.pubsub_manager:
            base_metrics['pubsub'] = self.pubsub_manager.get_statistics()
        
        return base_metrics
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get detailed migration status."""
        return {
            'progress': self._migration_progress.copy(),
            'mode': self.migration_mode,
            'dual_write_enabled': self.dual_write_mode,
            'redis_available': self.redis_state is not None,
            'fallback_available': self.fallback_manager is not None,
            'pubsub_enabled': self.enable_pubsub,
            'locking_enabled': self.enable_locking
        }
    
    async def backup(self):
        """Create backup of current state."""
        if self.redis_state:
            # Redis backup is handled by Redis infrastructure
            pass
        
        if self.fallback_manager:
            await self.fallback_manager.backup()
    
    async def shutdown(self):
        """Shutdown state manager gracefully."""
        self.logger.info("Shutting down Redis Async State Manager")
        self._shutdown = True
        
        # Stop background tasks
        for task in [self._worker_task, self._sync_task, self._migration_task, self._pubsub_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Process remaining operations
        remaining_ops = []
        while not self._pending_operations.empty():
            try:
                op = self._pending_operations.get_nowait()
                remaining_ops.append(op)
            except asyncio.QueueEmpty:
                break
        
        if remaining_ops:
            await self._process_operation_batch(remaining_ops)
        
        # Shutdown Redis components
        if self.redis_state:
            await self.redis_state.stop()
        
        if self.pubsub_manager:
            await self.pubsub_manager.stop()
        
        if self.locks_manager:
            await self.locks_manager.stop()
        
        # Shutdown fallback
        if self.fallback_manager:
            await self.fallback_manager.shutdown()
        
        self.logger.info("Redis Async State Manager shutdown complete")


# Global instance with migration support
_global_redis_state_manager: Optional[RedisAsyncStateManager] = None


async def get_redis_async_state_manager(migration_mode: str = "gradual") -> RedisAsyncStateManager:
    """Get or create global Redis async state manager with migration support."""
    global _global_redis_state_manager
    
    if _global_redis_state_manager is None:
        # Import existing state manager for migration
        from scripts.async_state_manager import get_async_state_manager
        fallback_manager = await get_async_state_manager()
        
        # Create Redis-backed state manager
        _global_redis_state_manager = RedisAsyncStateManager(
            fallback_manager=fallback_manager,
            migration_mode=migration_mode,
            dual_write_mode=True
        )
        
        await _global_redis_state_manager.initialize()
    
    return _global_redis_state_manager


# Convenience functions with Redis backend
async def redis_state_update(key_path: str, value: Any, distributed: bool = False):
    """Convenience function for state updates with Redis backend."""
    manager = await get_redis_async_state_manager()
    await manager.update(key_path, value, distributed=distributed)


async def redis_state_get(key_path: str, default: Any = None) -> Any:
    """Convenience function for state retrieval with Redis backend."""
    manager = await get_redis_async_state_manager()
    return await manager.get(key_path, default)