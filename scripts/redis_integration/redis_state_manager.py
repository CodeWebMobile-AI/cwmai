"""
Redis State Manager

Distributed state management with Redis backend featuring real-time synchronization,
version control, atomic updates, and multi-component state coordination.
"""

import asyncio
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import hashlib
from .redis_client import RedisClient
from .redis_pubsub_manager import RedisPubSubManager


class StateChangeType(Enum):
    """Types of state changes."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    REPLACE = "replace"


@dataclass
class StateChange:
    """State change event."""
    id: str
    component: str
    change_type: StateChangeType
    path: List[str]
    old_value: Any
    new_value: Any
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'component': self.component,
            'change_type': self.change_type.value,
            'path': self.path,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class StateVersion:
    """State version information."""
    version: int
    timestamp: datetime
    component: str
    change_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateConflictResolver:
    """Resolves state conflicts in distributed environment."""
    
    def __init__(self):
        self.strategies = {
            'last_write_wins': self._last_write_wins,
            'merge_recursive': self._merge_recursive,
            'custom': self._custom_resolve
        }
        self.logger = logging.getLogger(__name__)
    
    def resolve(self, strategy: str, local_state: Any, remote_state: Any,
               local_version: StateVersion, remote_version: StateVersion) -> Any:
        """Resolve state conflict using specified strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown conflict resolution strategy: {strategy}")
        
        return self.strategies[strategy](local_state, remote_state, local_version, remote_version)
    
    def _last_write_wins(self, local_state: Any, remote_state: Any,
                        local_version: StateVersion, remote_version: StateVersion) -> Any:
        """Use the most recently written state."""
        if remote_version.timestamp > local_version.timestamp:
            return remote_state
        return local_state
    
    def _merge_recursive(self, local_state: Any, remote_state: Any,
                        local_version: StateVersion, remote_version: StateVersion) -> Any:
        """Recursively merge state objects."""
        if not isinstance(local_state, dict) or not isinstance(remote_state, dict):
            return self._last_write_wins(local_state, remote_state, local_version, remote_version)
        
        merged = local_state.copy()
        for key, value in remote_state.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_recursive(merged[key], value, local_version, remote_version)
            else:
                merged[key] = value
        
        return merged
    
    def _custom_resolve(self, local_state: Any, remote_state: Any,
                       local_version: StateVersion, remote_version: StateVersion) -> Any:
        """Custom resolution logic - override in subclass."""
        return self._last_write_wins(local_state, remote_state, local_version, remote_version)


class RedisStateManager:
    """Distributed state management with Redis backend."""
    
    def __init__(self, redis_client: RedisClient, component_id: str = "default",
                 pubsub_manager: RedisPubSubManager = None):
        """Initialize Redis state manager.
        
        Args:
            redis_client: Redis client instance
            component_id: Unique component identifier
            pubsub_manager: Optional pub/sub manager for real-time sync
        """
        self.redis = redis_client
        self.component_id = component_id
        self.pubsub = pubsub_manager
        self.conflict_resolver = StateConflictResolver()
        self.logger = logging.getLogger(__name__)
        
        # State configuration
        self.namespace = "state"
        self.version_namespace = "state_versions"
        self.changes_namespace = "state_changes"
        self.locks_namespace = "state_locks"
        
        # Local state cache
        self._local_cache: Dict[str, Any] = {}
        self._cache_versions: Dict[str, StateVersion] = {}
        self._cache_lock = threading.RLock()
        
        # Change tracking
        self._change_listeners: List[Callable] = []
        self._change_history: List[StateChange] = []
        self._max_history = 1000
        
        # Synchronization settings
        self.auto_sync = True
        self.sync_interval = 30.0  # seconds
        self.lock_timeout = 30  # seconds
        self.enable_versioning = True
        self.enable_change_tracking = True
        
        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    def _make_key(self, component: str, key_path: List[str] = None) -> str:
        """Create Redis key for state component."""
        path_str = ":".join(key_path) if key_path else ""
        if path_str:
            return f"{self.namespace}:{component}:{path_str}"
        return f"{self.namespace}:{component}"
    
    def _version_key(self, component: str) -> str:
        """Create version key for component."""
        return f"{self.version_namespace}:{component}"
    
    def _changes_key(self, component: str) -> str:
        """Create changes key for component."""
        return f"{self.changes_namespace}:{component}"
    
    def _lock_key(self, component: str) -> str:
        """Create lock key for component."""
        return f"{self.locks_namespace}:{component}"
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate hash of state data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    async def start(self):
        """Start state manager background tasks."""
        if self.auto_sync and not self._sync_task:
            self._sync_task = asyncio.create_task(self._sync_loop())
            self.logger.info(f"State manager started for component {self.component_id}")
        
        if self.pubsub and not self._pubsub_task:
            self._pubsub_task = asyncio.create_task(self._listen_changes())
            self.logger.info("State change listener started")
    
    async def stop(self):
        """Stop state manager background tasks."""
        self._shutdown = True
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
        
        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass
            self._pubsub_task = None
        
        self.logger.info(f"State manager stopped for component {self.component_id}")
    
    async def _sync_loop(self):
        """Background synchronization loop."""
        while not self._shutdown:
            try:
                await self._sync_all_components()
                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def _listen_changes(self):
        """Listen for state changes via pub/sub."""
        channel = f"state_changes:{self.component_id}"
        
        async def handle_change(message):
            try:
                change_data = json.loads(message)
                change = StateChange(
                    id=change_data['id'],
                    component=change_data['component'],
                    change_type=StateChangeType(change_data['change_type']),
                    path=change_data['path'],
                    old_value=change_data['old_value'],
                    new_value=change_data['new_value'],
                    timestamp=datetime.fromisoformat(change_data['timestamp']),
                    metadata=change_data.get('metadata', {})
                )
                
                await self._handle_remote_change(change)
                
            except Exception as e:
                self.logger.error(f"Error handling state change: {e}")
        
        if self.pubsub:
            await self.pubsub.subscribe(channel, handle_change)
    
    async def get_state(self, component: str = None, key_path: List[str] = None,
                       use_cache: bool = True) -> Optional[Any]:
        """Get state value.
        
        Args:
            component: Component name (defaults to self.component_id)
            key_path: Path to nested state value
            use_cache: Whether to use local cache
            
        Returns:
            State value or None if not found
        """
        component = component or self.component_id
        
        # Check local cache first
        if use_cache:
            with self._cache_lock:
                if component in self._local_cache:
                    cached_state = self._local_cache[component]
                    if key_path:
                        return self._get_nested_value(cached_state, key_path)
                    return cached_state
        
        # Get from Redis
        try:
            redis_key = self._make_key(component, key_path)
            data = await self.redis.get(redis_key)
            
            if data is None:
                return None
            
            # Ensure data is properly decoded
            if hasattr(data, '__await__'):
                self.logger.error(f"WARNING: data is a coroutine, not awaited properly")
                return None
            
            state_value = json.loads(data)
            
            # Update local cache
            if use_cache and not key_path:  # Only cache full component state
                with self._cache_lock:
                    self._local_cache[component] = state_value
                    
                    # Get version info
                    version_data = await self.redis.get(self._version_key(component))
                    if version_data:
                        version_info = json.loads(version_data)
                        self._cache_versions[component] = StateVersion(
                            version=version_info['version'],
                            timestamp=datetime.fromisoformat(version_info['timestamp']),
                            component=version_info['component'],
                            change_hash=version_info['change_hash'],
                            metadata=version_info.get('metadata', {})
                        )
            
            return state_value
            
        except Exception as e:
            self.logger.error(f"Error getting state for {component}: {e}")
            return None
    
    async def set_state(self, value: Any, component: str = None, key_path: List[str] = None,
                       merge: bool = False, atomic: bool = True) -> bool:
        """Set state value.
        
        Args:
            value: Value to set
            component: Component name (defaults to self.component_id)
            key_path: Path to nested state value
            merge: Whether to merge with existing state
            atomic: Whether to use atomic operations
            
        Returns:
            True if successful
        """
        component = component or self.component_id
        
        if atomic:
            return await self._set_state_atomic(value, component, key_path, merge)
        else:
            return await self._set_state_simple(value, component, key_path, merge)
    
    async def _set_state_atomic(self, value: Any, component: str, key_path: List[str] = None,
                               merge: bool = False) -> bool:
        """Set state atomically with locking."""
        lock_key = self._lock_key(component)
        
        try:
            # Acquire distributed lock
            lock_value = f"{self.component_id}:{time.time()}"
            locked = await self.redis.set(lock_key, lock_value, ex=self.lock_timeout, nx=True)
            
            if not locked:
                self.logger.warning(f"Could not acquire lock for component {component}")
                return False
            
            try:
                # Get current state
                current_state = await self.get_state(component, use_cache=False)
                old_value = current_state
                
                # Prepare new state
                if key_path:
                    if current_state is None:
                        current_state = {}
                    new_state = self._set_nested_value(current_state, key_path, value, merge)
                else:
                    if merge and isinstance(current_state, dict) and isinstance(value, dict):
                        new_state = {**current_state, **value}
                    else:
                        new_state = value
                
                # Calculate version
                change_hash = self._calculate_hash(new_state)
                current_version = await self._get_version(component)
                new_version = StateVersion(
                    version=(current_version.version + 1) if current_version else 1,
                    timestamp=datetime.now(timezone.utc),
                    component=component,
                    change_hash=change_hash
                )
                
                # Store state and version
                redis_key = self._make_key(component, key_path if key_path else None)
                
                # Use pipeline for atomic operation
                async with self.redis.pipeline() as pipe:
                    pipe.set(redis_key, json.dumps(new_state))
                    
                    if self.enable_versioning:
                        pipe.set(self._version_key(component), json.dumps({
                            'version': new_version.version,
                            'timestamp': new_version.timestamp.isoformat(),
                            'component': new_version.component,
                            'change_hash': new_version.change_hash,
                            'metadata': new_version.metadata
                        }))
                    
                    await pipe.execute()
                
                # Update local cache
                with self._cache_lock:
                    if key_path:
                        if component not in self._local_cache:
                            self._local_cache[component] = {}
                        self._set_nested_value(self._local_cache[component], key_path, value, merge)
                    else:
                        self._local_cache[component] = new_state
                    self._cache_versions[component] = new_version
                
                # Track change
                if self.enable_change_tracking:
                    await self._track_change(component, key_path or [], old_value, value, StateChangeType.UPDATE)
                
                return True
                
            finally:
                # Release lock
                await self.redis.delete(lock_key)
        
        except Exception as e:
            self.logger.error(f"Error setting state for {component}: {e}")
            return False
    
    async def _set_state_simple(self, value: Any, component: str, key_path: List[str] = None,
                               merge: bool = False) -> bool:
        """Set state without locking (faster but less safe)."""
        try:
            redis_key = self._make_key(component, key_path)
            
            if merge and key_path:
                # Get current state for merging
                current_state = await self.get_state(component, use_cache=False)
                if current_state:
                    new_state = self._set_nested_value(current_state, key_path, value, merge)
                    await self.redis.set(redis_key, json.dumps(new_state))
                else:
                    await self.redis.set(redis_key, json.dumps(value))
            else:
                await self.redis.set(redis_key, json.dumps(value))
            
            # Update local cache
            with self._cache_lock:
                if key_path:
                    if component not in self._local_cache:
                        self._local_cache[component] = {}
                    self._set_nested_value(self._local_cache[component], key_path, value, merge)
                else:
                    self._local_cache[component] = value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting state for {component}: {e}")
            return False
    
    async def delete_state(self, component: str = None, key_path: List[str] = None) -> bool:
        """Delete state value."""
        component = component or self.component_id
        
        try:
            if key_path:
                # Delete nested value
                current_state = await self.get_state(component, use_cache=False)
                if current_state:
                    new_state = self._delete_nested_value(current_state, key_path)
                    return await self.set_state(new_state, component, atomic=True)
            else:
                # Delete entire component state
                redis_key = self._make_key(component)
                version_key = self._version_key(component)
                
                deleted = await self.redis.delete(redis_key, version_key)
                
                # Update local cache
                with self._cache_lock:
                    self._local_cache.pop(component, None)
                    self._cache_versions.pop(component, None)
                
                return deleted > 0
        
        except Exception as e:
            self.logger.error(f"Error deleting state for {component}: {e}")
            return False
    
    async def sync_component(self, component: str) -> bool:
        """Synchronize component state with Redis."""
        try:
            # Get remote state and version
            remote_state = await self.get_state(component, use_cache=False)
            remote_version = await self._get_version(component)
            
            with self._cache_lock:
                local_version = self._cache_versions.get(component)
                
                if remote_version and local_version:
                    if remote_version.version > local_version.version:
                        # Remote is newer, update local
                        self._local_cache[component] = remote_state
                        self._cache_versions[component] = remote_version
                        return True
                    elif local_version.version > remote_version.version:
                        # Local is newer, update remote
                        return await self.set_state(self._local_cache[component], component, atomic=True)
                    else:
                        # Same version, check hash
                        if remote_version.change_hash != local_version.change_hash:
                            # Conflict - resolve it
                            resolved_state = self.conflict_resolver.resolve(
                                'last_write_wins',
                                self._local_cache[component],
                                remote_state,
                                local_version,
                                remote_version
                            )
                            return await self.set_state(resolved_state, component, atomic=True)
                elif remote_version:
                    # Only remote exists
                    self._local_cache[component] = remote_state
                    self._cache_versions[component] = remote_version
                    return True
                elif local_version:
                    # Only local exists
                    return await self.set_state(self._local_cache[component], component, atomic=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing component {component}: {e}")
            return False
    
    async def _sync_all_components(self):
        """Synchronize all cached components."""
        with self._cache_lock:
            components = list(self._local_cache.keys())
        
        for component in components:
            await self.sync_component(component)
    
    async def _get_version(self, component: str) -> Optional[StateVersion]:
        """Get version info for component."""
        try:
            version_key = self._version_key(component)
            data = await self.redis.get(version_key)
            
            if data:
                version_info = json.loads(data)
                return StateVersion(
                    version=version_info['version'],
                    timestamp=datetime.fromisoformat(version_info['timestamp']),
                    component=version_info['component'],
                    change_hash=version_info['change_hash'],
                    metadata=version_info.get('metadata', {})
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting version for {component}: {e}")
            return None
    
    async def get_version(self, component: str) -> Optional[StateVersion]:
        """Get version info for component (public method)."""
        return await self._get_version(component)
    
    async def _track_change(self, component: str, path: List[str], old_value: Any,
                           new_value: Any, change_type: StateChangeType):
        """Track state change for history and notifications."""
        try:
            change = StateChange(
                id=f"{component}:{int(time.time() * 1000)}",
                component=component,
                change_type=change_type,
                path=path,
                old_value=old_value,
                new_value=new_value,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Add to local history
            self._change_history.append(change)
            if len(self._change_history) > self._max_history:
                self._change_history.pop(0)
            
            # Store in Redis
            changes_key = self._changes_key(component)
            await self.redis.lpush(changes_key, json.dumps(change.to_dict()))
            await self.redis.ltrim(changes_key, 0, self._max_history - 1)
            
            # Notify listeners
            for listener in self._change_listeners:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(change)
                    else:
                        listener(change)
                except Exception as e:
                    self.logger.error(f"Error in change listener: {e}")
            
            # Publish change via pub/sub
            if self.pubsub:
                channel = f"state_changes:{component}"
                await self.pubsub.publish(channel, change.to_dict())
        
        except Exception as e:
            self.logger.error(f"Error tracking change: {e}")
    
    async def _handle_remote_change(self, change: StateChange):
        """Handle state change from remote component."""
        try:
            # Update local cache if this component
            if change.component == self.component_id:
                with self._cache_lock:
                    if change.component in self._local_cache:
                        if change.path:
                            self._set_nested_value(
                                self._local_cache[change.component],
                                change.path,
                                change.new_value
                            )
                        else:
                            self._local_cache[change.component] = change.new_value
            
            # Add to change history
            self._change_history.append(change)
            if len(self._change_history) > self._max_history:
                self._change_history.pop(0)
            
            # Notify listeners
            for listener in self._change_listeners:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(change)
                    else:
                        listener(change)
                except Exception as e:
                    self.logger.error(f"Error in change listener: {e}")
        
        except Exception as e:
            self.logger.error(f"Error handling remote change: {e}")
    
    def add_change_listener(self, listener: Callable[[StateChange], None]):
        """Add state change listener."""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[StateChange], None]):
        """Remove state change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def get_change_history(self, component: str = None, limit: int = 100) -> List[StateChange]:
        """Get change history."""
        if component:
            return [c for c in self._change_history if c.component == component][-limit:]
        return self._change_history[-limit:]
    
    def _get_nested_value(self, data: Any, path: List[str]) -> Any:
        """Get nested value from data structure."""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], path: List[str], value: Any, merge: bool = False) -> Dict[str, Any]:
        """Set nested value in data structure."""
        if not path:
            return value
        
        result = data.copy()
        current = result
        
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        final_key = path[-1]
        if merge and isinstance(current.get(final_key), dict) and isinstance(value, dict):
            current[final_key] = {**current[final_key], **value}
        else:
            current[final_key] = value
        
        return result
    
    def _delete_nested_value(self, data: Dict[str, Any], path: List[str]) -> Dict[str, Any]:
        """Delete nested value from data structure."""
        if not path:
            return {}
        
        result = data.copy()
        current = result
        
        for key in path[:-1]:
            if key not in current or not isinstance(current[key], dict):
                return result  # Path doesn't exist
            current = current[key]
        
        final_key = path[-1]
        current.pop(final_key, None)
        
        return result
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        with self._cache_lock:
            cached_components = len(self._local_cache)
            total_changes = len(self._change_history)
        
        return {
            'component_id': self.component_id,
            'cached_components': cached_components,
            'total_changes': total_changes,
            'auto_sync': self.auto_sync,
            'sync_interval': self.sync_interval,
            'enable_versioning': self.enable_versioning,
            'enable_change_tracking': self.enable_change_tracking,
            'listeners': len(self._change_listeners)
        }