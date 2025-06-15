"""
Redis State Management Adapter

Adapts the existing StateManager to use Redis backend for distributed state management
while maintaining backward compatibility with the current system.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from scripts.redis_integration.redis_client import RedisClient, get_redis_client
from scripts.redis_integration.redis_state_manager import RedisStateManager
from scripts.redis_integration.redis_pubsub_manager import RedisPubSubManager
from scripts.state_manager import StateManager


class RedisStateAdapter:
    """Adapter to use Redis for state management while maintaining StateManager interface."""
    
    def __init__(self, state_manager: StateManager, redis_client: RedisClient = None,
                 component_id: str = "cwmai_orchestrator"):
        """Initialize Redis state adapter.
        
        Args:
            state_manager: Existing StateManager instance
            redis_client: Redis client instance
            component_id: Component identifier for Redis keys
        """
        self.state_manager = state_manager
        self.redis_client = redis_client
        self.component_id = component_id
        self.logger = logging.getLogger(__name__)
        
        # Redis managers
        self.redis_state: Optional[RedisStateManager] = None
        self.pubsub: Optional[RedisPubSubManager] = None
        
        # Sync settings
        self.sync_to_redis = True
        self.sync_from_redis = True
        self.sync_interval = 30  # seconds
        self.enable_real_time_sync = True
        
        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._initialized = False
    
    async def initialize(self, shared_redis: RedisClient = None):
        """Initialize Redis connection and state sync."""
        if self._initialized:
            return

        try:
            # Use shared Redis client if provided
            if not self.redis_client:
                self.redis_client = shared_redis or await get_redis_client()
            
            # Initialize pub/sub manager for real-time sync
            if self.enable_real_time_sync:
                self.pubsub = RedisPubSubManager(self.redis_client)
                await self.pubsub.start()
            
            # Initialize Redis state manager
            self.redis_state = RedisStateManager(
                self.redis_client,
                self.component_id,
                self.pubsub
            )
            
            # Subscribe to state changes
            if self.pubsub:
                await self.pubsub.subscribe(
                    f"state_changes:{self.component_id}",
                    self._handle_remote_state_change
                )
            
            # Start Redis state manager
            await self.redis_state.start()
            
            # Initial sync from Redis (if exists) or to Redis
            await self._initial_sync()
            
            # Start background sync
            if self.sync_to_redis or self.sync_from_redis:
                self._sync_task = asyncio.create_task(self._sync_loop())
            
            self._initialized = True
            self.logger.info(f"Redis state adapter initialized for component {self.component_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis state adapter: {e}")
            raise
    
    async def _initial_sync(self):
        """Perform initial state synchronization."""
        try:
            # Try to load state from Redis first
            redis_state = await self.redis_state.get_state(self.component_id)
            
            if redis_state:
                # Redis has state - check if it's newer
                redis_updated = redis_state.get('last_updated', '')
                local_state = self.state_manager.get_state()
                local_updated = local_state.get('last_updated', '')
                
                if redis_updated > local_updated:
                    self.logger.info("Loading state from Redis (newer than local)")
                    # Update local state with Redis state
                    self.state_manager.state = redis_state
                    self.state_manager.save_state()
                else:
                    self.logger.info("Pushing local state to Redis (newer than Redis)")
                    # Push local state to Redis
                    await self.redis_state.set_state(local_state, component=self.component_id)
            else:
                # No state in Redis - push local state
                self.logger.info("No state in Redis, pushing local state")
                local_state = self.state_manager.get_state()
                await self.redis_state.set_state(local_state, component=self.component_id)
                
        except Exception as e:
            self.logger.error(f"Error during initial sync: {e}")
    
    async def _sync_loop(self):
        """Background synchronization loop."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                
                if self.sync_to_redis:
                    await self._sync_to_redis()
                
                if self.sync_from_redis:
                    await self._sync_from_redis()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def _sync_to_redis(self):
        """Sync local state changes to Redis."""
        try:
            local_state = self.state_manager.get_state()
            local_updated = local_state.get('last_updated', '')
            
            # Get current Redis state metadata
            redis_version = await self.redis_state.get_version(self.component_id)
            
            if not redis_version or local_updated > redis_version.timestamp.isoformat():
                # Local state is newer - push to Redis
                await self.redis_state.set_state(local_state, component=self.component_id)
                self.logger.debug("Synced local state to Redis")
                
        except Exception as e:
            self.logger.error(f"Error syncing to Redis: {e}")
    
    async def _sync_from_redis(self):
        """Sync Redis state changes to local."""
        try:
            redis_state = await self.redis_state.get_state(self.component_id)
            if not redis_state:
                return
            
            redis_updated = redis_state.get('last_updated', '')
            local_state = self.state_manager.get_state()
            local_updated = local_state.get('last_updated', '')
            
            if redis_updated > local_updated:
                # Log what we're syncing
                self.logger.info(f"Redis state is newer ({redis_updated} > {local_updated}), syncing to local")
                
                # Log repository information
                if "projects" in redis_state:
                    repo_names = list(redis_state["projects"].keys())
                    self.logger.info(f"Redis state contains {len(repo_names)} repositories: {repo_names}")
                else:
                    self.logger.info("Redis state contains no projects")
                
                # Redis state is newer - update local
                self.state_manager.state = redis_state
                self.state_manager.save_state()
                self.logger.info("Synced Redis state to local - this is where deleted repos might be coming from!")
                
        except Exception as e:
            self.logger.error(f"Error syncing from Redis: {e}")
    
    async def _handle_remote_state_change(self, message: Dict[str, Any]):
        """Handle state change notification from Redis pub/sub."""
        try:
            change_component = message.get('component')
            if change_component != self.component_id:
                return  # Not for us
            
            # Sync from Redis on remote change
            if self.sync_from_redis:
                await self._sync_from_redis()
                self.logger.debug(f"Handled remote state change for {self.component_id}")
                
        except Exception as e:
            self.logger.error(f"Error handling remote state change: {e}")
    
    # Proxy methods to maintain StateManager interface
    def get_state(self) -> Dict[str, Any]:
        """Get current state (from local StateManager)."""
        return self.state_manager.get_state()
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update state and sync to Redis."""
        self.state_manager.update_state(updates)
        
        # Queue immediate sync to Redis
        if self.sync_to_redis and self._initialized:
            asyncio.create_task(self._sync_to_redis())
    
    def save_state(self) -> None:
        """Save state locally and sync to Redis."""
        self.state_manager.save_state()
        
        # Queue immediate sync to Redis
        if self.sync_to_redis and self._initialized:
            asyncio.create_task(self._sync_to_redis())
    
    async def get_distributed_state(self, component: str = None) -> Optional[Dict[str, Any]]:
        """Get state from Redis for any component.
        
        Args:
            component: Component ID, defaults to current component
            
        Returns:
            State dictionary or None
        """
        if not self._initialized:
            await self.initialize()
        
        component = component or self.component_id
        return await self.redis_state.get_state(component)
    
    async def get_all_component_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states for all components in the system.
        
        Returns:
            Dictionary mapping component IDs to their states
        """
        if not self._initialized:
            await self.initialize()
        
        states = {}
        
        # Get all state keys
        cursor = 0
        pattern = f"{self.redis_state.namespace}:*"
        
        while True:
            cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
            
            for key in keys:
                # Extract component ID from key
                parts = key.decode().split(':')
                if len(parts) >= 2:
                    component_id = parts[1]
                    state = await self.redis_state.get_state(component_id)
                    if state:
                        states[component_id] = state
            
            if cursor == 0:
                break
        
        return states
    
    async def lock_state(self, timeout: int = 30) -> bool:
        """Acquire distributed lock for state updates.
        
        Args:
            timeout: Lock timeout in seconds
            
        Returns:
            True if lock acquired
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.redis_state.acquire_lock(self.component_id, timeout)
    
    async def unlock_state(self):
        """Release distributed lock."""
        if not self._initialized:
            await self.initialize()
        
        await self.redis_state.release_lock(self.component_id)
    
    async def get_state_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get state change history.
        
        Args:
            limit: Maximum number of changes to retrieve
            
        Returns:
            List of state changes
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.redis_state.get_change_history(self.component_id, limit)
    
    async def cleanup(self):
        """Clean up Redis connections and tasks."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_state:
            await self.redis_state.stop()
        
        if self.pubsub:
            await self.pubsub.stop()
        
        self._initialized = False
        self.logger.info("Redis state adapter cleaned up")


class RedisEnabledStateManager(StateManager):
    """StateManager with Redis backend support."""
    
    def __init__(self, local_path: str = "system_state.json", use_redis: bool = True):
        """Initialize Redis-enabled state manager.
        
        Args:
            local_path: Local file path for state storage
            use_redis: Whether to enable Redis backend
        """
        super().__init__(local_path)
        self.use_redis = use_redis
        self.redis_adapter: Optional[RedisStateAdapter] = None
        
        if use_redis:
            # Create Redis adapter
            self.redis_adapter = RedisStateAdapter(self)
    
    async def initialize_redis(self):
        """Initialize Redis backend."""
        if self.redis_adapter:
            await self.redis_adapter.initialize()
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update state with Redis sync."""
        super().update_state(updates)
        
        # Redis adapter handles sync automatically
        if self.redis_adapter:
            # Adapter's update_state is called via proxy
            pass
    
    def save_state(self) -> None:
        """Save state with Redis sync."""
        super().save_state()
        
        # Redis adapter handles sync automatically
        if self.redis_adapter:
            # Adapter's save_state is called via proxy
            pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state."""
        return super().get_state()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.redis_adapter:
            await self.redis_adapter.cleanup()