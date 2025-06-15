"""
Adapter for RedisLockFreeStateManager to match the existing state manager API.

This adapter provides compatibility between the lock-free state manager's
partitioned approach and the existing code's expectations.
"""

import json
from typing import Any, Dict, Optional, Callable
from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager


class RedisLockFreeAdapter:
    """Adapter to make RedisLockFreeStateManager compatible with existing API."""
    
    def __init__(self, component_id: str = "default"):
        self.component_id = component_id
        self.state_manager = RedisLockFreeStateManager()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the lock-free state manager."""
        if not self._initialized:
            await self.state_manager.initialize()
            self._initialized = True
    
    async def set(self, key_path: str, value: Any, distributed: bool = False):
        """
        Set state value - alias for update for compatibility.
        """
        return await self.update(key_path, value, distributed=distributed)
    
    async def update(self, key_path: str, value: Any, 
                    distributed: bool = False,
                    conflict_resolution: str = None,
                    lock_required: bool = None,
                    callback: Optional[Callable] = None):
        """
        Update state value using the lock-free state manager.
        
        Maps the hierarchical key path to the appropriate lock-free storage.
        """
        # Parse the key path to determine storage type
        parts = key_path.split('.')
        
        if len(parts) == 0:
            return False
        
        # Route based on key pattern
        if parts[0] == "intelligence" and len(parts) >= 3:
            if parts[1] == "workers" and len(parts) >= 3:
                # Worker-specific state: intelligence.workers.{worker_id}...
                worker_id = parts[2]
                
                if len(parts) == 3:
                    # Full worker state update
                    return await self.state_manager.update_worker_state(worker_id, value)
                else:
                    # Nested field update - store as hash
                    field_path = '.'.join(parts[3:])
                    return await self.state_manager.update_hash_field(
                        f"worker:{worker_id}:data",
                        field_path,
                        value
                    )
            elif parts[1] == "tasks" and len(parts) >= 3:
                # Task-specific state: intelligence.tasks.{task_id}...
                task_id = parts[2]
                
                if len(parts) == 3:
                    # Full task state update
                    return await self.state_manager.update_task_state(task_id, value)
                else:
                    # Nested field update
                    field_path = '.'.join(parts[3:])
                    return await self.state_manager.update_hash_field(
                        f"task:{task_id}:data",
                        field_path,
                        value
                    )
            else:
                # Other intelligence data - use hash storage
                hash_name = f"{self.component_id}:{parts[0]}:{parts[1]}"
                field_name = '.'.join(parts[2:]) if len(parts) > 2 else "data"
                return await self.state_manager.update_hash_field(hash_name, field_name, value)
        
        elif parts[0] == "worker" and len(parts) >= 2:
            # Direct worker state: worker.{worker_id}...
            worker_id = parts[1]
            if len(parts) == 2:
                return await self.state_manager.update_worker_state(worker_id, value)
            else:
                field_name = '.'.join(parts[2:])
                return await self.state_manager.update_worker_field(worker_id, field_name, value)
        
        elif parts[0] == "task" and len(parts) >= 2:
            # Direct task state: task.{task_id}...
            task_id = parts[1]
            if len(parts) == 2:
                return await self.state_manager.update_task_state(task_id, value)
            else:
                field_path = '.'.join(parts[2:])
                return await self.state_manager.update_hash_field(
                    f"task:{task_id}:data",
                    field_path,
                    value
                )
        
        else:
            # Generic storage - use hash with component namespace
            hash_name = f"{self.component_id}:{parts[0]}"
            field_name = '.'.join(parts[1:]) if len(parts) > 1 else "data"
            await self.state_manager.update_hash_field(hash_name, field_name, value)
            
            # For backwards compatibility, return True
            return True
        
        # Execute callback if provided
        if callback:
            if hasattr(callback, '__call__'):
                await callback()
        
        return True
    
    async def get(self, key_path: str, default: Any = None, 
                 use_cache: bool = True) -> Any:
        """
        Get state value using the lock-free state manager.
        """
        parts = key_path.split('.')
        
        if len(parts) == 0:
            return default
        
        # Route based on key pattern (similar to update)
        if parts[0] == "intelligence" and len(parts) >= 3:
            if parts[1] == "workers" and len(parts) >= 3:
                worker_id = parts[2]
                
                if len(parts) == 3:
                    # Get full worker state
                    state = await self.state_manager.get_worker_state(worker_id)
                    return state if state is not None else default
                else:
                    # Get nested field
                    field_path = '.'.join(parts[3:])
                    value = await self.state_manager.get_hash_field(
                        f"worker:{worker_id}:data",
                        field_path
                    )
                    return value if value is not None else default
            
            elif parts[1] == "tasks" and len(parts) >= 3:
                task_id = parts[2]
                
                if len(parts) == 3:
                    state = await self.state_manager.get_task_state(task_id)
                    return state if state is not None else default
                else:
                    field_path = '.'.join(parts[3:])
                    value = await self.state_manager.get_hash_field(
                        f"task:{task_id}:data",
                        field_path
                    )
                    return value if value is not None else default
            else:
                # Other intelligence data
                hash_name = f"{self.component_id}:{parts[0]}:{parts[1]}"
                field_name = '.'.join(parts[2:]) if len(parts) > 2 else "data"
                value = await self.state_manager.get_hash_field(hash_name, field_name)
                return value if value is not None else default
        
        elif parts[0] == "worker" and len(parts) >= 2:
            worker_id = parts[1]
            if len(parts) == 2:
                state = await self.state_manager.get_worker_state(worker_id)
                return state if state is not None else default
            else:
                # For nested fields, we need to get the full state and extract
                state = await self.state_manager.get_worker_state(worker_id)
                if state:
                    # Navigate through nested structure
                    current = state
                    for part in parts[2:]:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            return default
                    return current
                return default
        
        elif parts[0] == "task" and len(parts) >= 2:
            task_id = parts[1]
            if len(parts) == 2:
                state = await self.state_manager.get_task_state(task_id)
                return state if state is not None else default
            else:
                field_path = '.'.join(parts[2:])
                value = await self.state_manager.get_hash_field(
                    f"task:{task_id}:data",
                    field_path
                )
                return value if value is not None else default
        
        else:
            # Generic storage
            hash_name = f"{self.component_id}:{parts[0]}"
            field_name = '.'.join(parts[1:]) if len(parts) > 1 else "data"
            value = await self.state_manager.get_hash_field(hash_name, field_name)
            return value if value is not None else default
    
    async def delete(self, key_path: str, distributed: bool = False):
        """Delete is not directly supported in lock-free design - use TTL or set to None."""
        # For compatibility, we'll set the value to None
        await self.update(key_path, None, distributed=distributed)
    
    async def get_full_state(self) -> Dict[str, Any]:
        """Get complete state - returns global state for this component."""
        return await self.state_manager.get_global_state()
    
    async def shutdown(self):
        """Shutdown the state manager."""
        if self._initialized:
            await self.state_manager.close()
            self._initialized = False
    
    # Additional compatibility methods
    async def increment_counter(self, counter_name: str, amount: int = 1) -> int:
        """Increment a counter atomically."""
        return await self.state_manager.increment_counter(counter_name, amount)
    
    async def get_counter(self, counter_name: str) -> int:
        """Get counter value."""
        return await self.state_manager.get_counter(counter_name)
    
    async def add_to_set(self, set_name: str, *members: str) -> int:
        """Add members to a set."""
        return await self.state_manager.add_to_set(set_name, *members)
    
    async def remove_from_set(self, set_name: str, *members: str) -> int:
        """Remove members from a set."""
        return await self.state_manager.remove_from_set(set_name, *members)
    
    async def get_set_members(self, set_name: str) -> set:
        """Get all members of a set."""
        return await self.state_manager.get_set_members(set_name)
    
    async def update_worker_field(self, worker_id: str, field: str, value: Any) -> bool:
        """Update a single field in worker state."""
        return await self.state_manager.update_worker_field(worker_id, field, value)
    
    async def update_task_state(self, task_id: str, state: Dict[str, Any]) -> bool:
        """Update task state."""
        return await self.state_manager.update_task_state(task_id, state)
    
    async def close(self):
        """Close the state manager."""
        await self.shutdown()


# Factory function for creating adapted state managers with unique component IDs
def create_lockfree_state_manager(component_id: str) -> RedisLockFreeAdapter:
    """Create a lock-free state manager with a unique component ID."""
    return RedisLockFreeAdapter(component_id=component_id)