"""
Redis Integration Adapters

Seamless integration adapters that provide drop-in replacements for existing
AI cache and state management systems with Redis backends and migration support.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime, timezone

from scripts.redis_ai_response_cache import RedisAIResponseCache, get_redis_ai_cache
from scripts.redis_lockfree_adapter import create_lockfree_state_manager
from scripts.redis_migration_coordinator import RedisMigrationCoordinator, MigrationConfig
from scripts.ai_response_cache import AIResponseCache
from scripts.async_state_manager import AsyncStateManager


class EnhancedAIResponseCacheAdapter:
    """Drop-in replacement adapter for AIResponseCache with Redis backend."""
    
    def __init__(self, 
                 enable_redis: bool = True,
                 migration_mode: str = "gradual",
                 auto_migrate: bool = True):
        """Initialize enhanced AI response cache adapter.
        
        Args:
            enable_redis: Whether to enable Redis backend
            migration_mode: Migration strategy (gradual/immediate/readonly)
            auto_migrate: Whether to automatically start migration
        """
        self.enable_redis = enable_redis
        self.migration_mode = migration_mode
        self.auto_migrate = auto_migrate
        
        self._redis_cache: Optional[RedisAIResponseCache] = None
        self._legacy_cache: Optional[AIResponseCache] = None
        self._initialized = False
        self._migration_started = False
        
        self.logger = logging.getLogger(f"{__name__}.EnhancedAIResponseCacheAdapter")
    
    async def _ensure_initialized(self):
        """Ensure the adapter is properly initialized."""
        if self._initialized:
            return
        
        try:
            if self.enable_redis:
                # Initialize Redis-backed cache
                self._redis_cache = await get_redis_ai_cache(migration_mode=self.migration_mode)
                
                if self.auto_migrate and not self._migration_started:
                    # Start gradual migration in background
                    asyncio.create_task(self._background_migration())
                    self._migration_started = True
            else:
                # Fall back to legacy cache
                from scripts.ai_response_cache import get_global_cache
                self._legacy_cache = get_global_cache()
            
            self._initialized = True
            self.logger.info(f"Enhanced AI cache adapter initialized (Redis: {self.enable_redis})")
            
        except Exception as e:
            self.logger.error(f"Error initializing cache adapter: {e}")
            # Fall back to legacy cache
            from scripts.ai_response_cache import get_global_cache
            self._legacy_cache = get_global_cache()
            self._initialized = True
    
    async def _background_migration(self):
        """Background migration process."""
        try:
            self.logger.info("Starting background cache migration")
            # Migration is handled by the Redis cache itself
            # This is just for monitoring and logging
            
            while self._redis_cache and not self._redis_cache._shutdown:
                status = await self._redis_cache.get_migration_status()
                progress = status['progress']
                
                if progress['completed']:
                    self.logger.info("Cache migration completed successfully")
                    break
                
                # Log progress periodically
                total = progress['total_entries']
                migrated = progress['migrated_entries']
                if total > 0:
                    progress_pct = (migrated / total) * 100
                    self.logger.debug(f"Cache migration progress: {progress_pct:.1f}% ({migrated}/{total})")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            self.logger.error(f"Error in background migration: {e}")
    
    async def get(self, prompt: str, provider: str, model: str) -> Optional[str]:
        """Get cached response with Redis backend and fallback."""
        await self._ensure_initialized()
        
        try:
            if self._redis_cache:
                return await self._redis_cache.get(prompt, provider, model)
            elif self._legacy_cache:
                return await self._legacy_cache.get(prompt, provider, model)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error getting cached response: {e}")
            return None
    
    async def put(self, 
                  prompt: str, 
                  response: str, 
                  provider: str, 
                  model: str,
                  ttl_seconds: Optional[int] = None,
                  cost_estimate: float = 0.0) -> None:
        """Store response in cache with Redis backend and fallback."""
        await self._ensure_initialized()
        
        try:
            if self._redis_cache:
                await self._redis_cache.put(prompt, response, provider, model, ttl_seconds, cost_estimate)
            elif self._legacy_cache:
                await self._legacy_cache.put(prompt, response, provider, model, ttl_seconds, cost_estimate)
        except Exception as e:
            self.logger.error(f"Error storing cached response: {e}")
    
    async def warm_cache(self, historical_data: List[Dict[str, Any]]) -> int:
        """Warm cache with historical data."""
        await self._ensure_initialized()
        
        try:
            if self._redis_cache:
                return await self._redis_cache.warm_cache(historical_data)
            elif self._legacy_cache:
                return await self._legacy_cache.warm_cache(historical_data)
            else:
                return 0
        except Exception as e:
            self.logger.error(f"Error warming cache: {e}")
            return 0
    
    async def clear(self):
        """Clear all cache entries."""
        await self._ensure_initialized()
        
        try:
            if self._redis_cache:
                await self._redis_cache.clear()
            elif self._legacy_cache:
                await self._legacy_cache.clear()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        try:
            if self._redis_cache:
                return self._redis_cache.get_stats()
            elif self._legacy_cache:
                return self._legacy_cache.get_stats()
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status (Redis-specific)."""
        if self._redis_cache:
            return await self._redis_cache.get_migration_status()
        else:
            return {
                'redis_enabled': False,
                'migration_mode': 'legacy_only'
            }
    
    async def shutdown(self):
        """Shutdown cache adapter."""
        try:
            if self._redis_cache:
                await self._redis_cache.shutdown()
            
            if self._legacy_cache:
                await self._legacy_cache.shutdown()
                
            self.logger.info("Enhanced AI cache adapter shutdown complete")
        except Exception as e:
            self.logger.error(f"Error shutting down cache adapter: {e}")


class EnhancedAsyncStateManagerAdapter:
    """Drop-in replacement adapter for AsyncStateManager with Redis backend."""
    
    def __init__(self,
                 enable_redis: bool = True,
                 migration_mode: str = "gradual",
                 auto_migrate: bool = True,
                 component_id: str = "default"):
        """Initialize enhanced async state manager adapter.
        
        Args:
            enable_redis: Whether to enable Redis backend
            migration_mode: Migration strategy (gradual/immediate/readonly)
            auto_migrate: Whether to automatically start migration
            component_id: Unique component identifier
        """
        self.enable_redis = enable_redis
        self.migration_mode = migration_mode
        self.auto_migrate = auto_migrate
        self.component_id = component_id
        
        self._redis_state: Optional[RedisAsyncStateManager] = None
        self._legacy_state: Optional[AsyncStateManager] = None
        self._initialized = False
        self._migration_started = False
        
        self.logger = logging.getLogger(f"{__name__}.EnhancedAsyncStateManagerAdapter")
    
    async def _ensure_initialized(self):
        """Ensure the adapter is properly initialized."""
        if self._initialized:
            return
        
        try:
            if self.enable_redis:
                # Initialize Redis-backed state manager
                self._redis_state = create_lockfree_state_manager("state_adapter")
                await self._redis_state.initialize()
                
                if self.auto_migrate and not self._migration_started:
                    # Start gradual migration in background
                    asyncio.create_task(self._background_migration())
                    self._migration_started = True
            else:
                # Fall back to legacy state manager
                from scripts.async_state_manager import get_async_state_manager
                self._legacy_state = await get_async_state_manager()
            
            self._initialized = True
            self.logger.info(f"Enhanced state manager adapter initialized (Redis: {self.enable_redis})")
            
        except Exception as e:
            self.logger.error(f"Error initializing state manager adapter: {e}")
            # Fall back to legacy state manager
            from scripts.async_state_manager import get_async_state_manager
            self._legacy_state = await get_async_state_manager()
            self._initialized = True
    
    async def _background_migration(self):
        """Background migration process."""
        try:
            self.logger.info("Starting background state migration")
            
            while self._redis_state and not self._redis_state._shutdown:
                status = await self._redis_state.get_migration_status()
                progress = status['progress']
                
                if progress['completed']:
                    self.logger.info("State migration completed successfully")
                    break
                
                # Log progress periodically
                total = progress['total_keys']
                migrated = progress['migrated_keys']
                if total > 0:
                    progress_pct = (migrated / total) * 100
                    self.logger.debug(f"State migration progress: {progress_pct:.1f}% ({migrated}/{total})")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            self.logger.error(f"Error in background migration: {e}")
    
    async def update(self, key_path: str, value: Any, 
                    distributed: bool = False,
                    callback: Optional[Callable] = None):
        """Update state value with Redis backend and fallback."""
        await self._ensure_initialized()
        
        try:
            if self._redis_state:
                await self._redis_state.update(key_path, value, distributed=distributed, callback=callback)
            elif self._legacy_state:
                await self._legacy_state.update(key_path, value, callback=callback)
        except Exception as e:
            self.logger.error(f"Error updating state: {e}")
    
    async def get(self, key_path: str, default: Any = None, use_cache: bool = True) -> Any:
        """Get state value with Redis backend and fallback."""
        await self._ensure_initialized()
        
        try:
            if self._redis_state:
                return await self._redis_state.get(key_path, default, use_cache)
            elif self._legacy_state:
                return await self._legacy_state.get(key_path, default)
            else:
                return default
        except Exception as e:
            self.logger.error(f"Error getting state: {e}")
            return default
    
    async def delete(self, key_path: str, distributed: bool = False):
        """Delete state value with Redis backend and fallback."""
        await self._ensure_initialized()
        
        try:
            if self._redis_state:
                await self._redis_state.delete(key_path, distributed=distributed)
            elif self._legacy_state:
                await self._legacy_state.delete(key_path)
        except Exception as e:
            self.logger.error(f"Error deleting state: {e}")
    
    async def get_full_state(self) -> Dict[str, Any]:
        """Get complete state dictionary."""
        await self._ensure_initialized()
        
        try:
            if self._redis_state:
                return await self._redis_state.get_full_state()
            elif self._legacy_state:
                return await self._legacy_state.get_full_state()
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error getting full state: {e}")
            return {}
    
    async def update_full_state(self, new_state: Dict[str, Any]):
        """Replace entire state."""
        await self._ensure_initialized()
        
        try:
            if self._redis_state:
                await self._redis_state.update_full_state(new_state)
            elif self._legacy_state:
                await self._legacy_state.update_full_state(new_state)
        except Exception as e:
            self.logger.error(f"Error updating full state: {e}")
    
    async def backup(self):
        """Create backup of current state."""
        await self._ensure_initialized()
        
        try:
            if self._redis_state:
                await self._redis_state.backup()
            elif self._legacy_state:
                await self._legacy_state.backup()
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
    
    def add_change_listener(self, listener: Callable):
        """Add state change listener."""
        if self._redis_state:
            self._redis_state.add_change_listener(listener)
        # Legacy state manager doesn't have change listeners
    
    def remove_change_listener(self, listener: Callable):
        """Remove state change listener."""
        if self._redis_state:
            self._redis_state.remove_change_listener(listener)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            if self._redis_state:
                return self._redis_state.get_metrics()
            elif self._legacy_state:
                return self._legacy_state.get_metrics()
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {}
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status (Redis-specific)."""
        if self._redis_state:
            return await self._redis_state.get_migration_status()
        else:
            return {
                'redis_enabled': False,
                'migration_mode': 'legacy_only'
            }
    
    async def transaction(self):
        """Context manager for atomic state transactions."""
        await self._ensure_initialized()
        
        if self._redis_state:
            return self._redis_state.transaction()
        elif self._legacy_state:
            return self._legacy_state.transaction()
        else:
            # Return a dummy context manager
            return self._dummy_transaction()
    
    async def _dummy_transaction(self):
        """Dummy transaction context manager."""
        class DummyTransaction:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return DummyTransaction()
    
    async def shutdown(self):
        """Shutdown state manager adapter."""
        try:
            if self._redis_state:
                await self._redis_state.shutdown()
            
            if self._legacy_state:
                await self._legacy_state.shutdown()
                
            self.logger.info("Enhanced state manager adapter shutdown complete")
        except Exception as e:
            self.logger.error(f"Error shutting down state manager adapter: {e}")


class SystemMigrationManager:
    """High-level system migration manager for coordinated Redis migration."""
    
    def __init__(self, config: Optional[MigrationConfig] = None):
        """Initialize system migration manager.
        
        Args:
            config: Migration configuration settings
        """
        self.config = config or MigrationConfig()
        self.logger = logging.getLogger(f"{__name__}.SystemMigrationManager")
        
        # Migration coordinator
        self._coordinator: Optional[RedisMigrationCoordinator] = None
        
        # Component adapters
        self._cache_adapter: Optional[EnhancedAIResponseCacheAdapter] = None
        self._state_adapter: Optional[EnhancedAsyncStateManagerAdapter] = None
        
        # Migration state
        self._migration_active = False
        self._adapters_initialized = False
    
    async def initialize_adapters(self, 
                                 enable_redis: bool = True,
                                 migration_mode: str = "gradual") -> Tuple[EnhancedAIResponseCacheAdapter, EnhancedAsyncStateManagerAdapter]:
        """Initialize system adapters with Redis backends.
        
        Args:
            enable_redis: Whether to enable Redis backends
            migration_mode: Migration strategy
            
        Returns:
            Tuple of (cache_adapter, state_adapter)
        """
        try:
            self.logger.info("Initializing system adapters")
            
            # Initialize cache adapter
            self._cache_adapter = EnhancedAIResponseCacheAdapter(
                enable_redis=enable_redis,
                migration_mode=migration_mode,
                auto_migrate=False  # We'll control migration centrally
            )
            
            # Initialize state adapter
            self._state_adapter = EnhancedAsyncStateManagerAdapter(
                enable_redis=enable_redis,
                migration_mode=migration_mode,
                auto_migrate=False,
                component_id="system"
            )
            
            # Ensure adapters are initialized
            await self._cache_adapter._ensure_initialized()
            await self._state_adapter._ensure_initialized()
            
            self._adapters_initialized = True
            self.logger.info("System adapters initialized successfully")
            
            return self._cache_adapter, self._state_adapter
            
        except Exception as e:
            self.logger.error(f"Error initializing adapters: {e}")
            raise
    
    async def start_coordinated_migration(self) -> bool:
        """Start coordinated system migration to Redis.
        
        Returns:
            True if migration started successfully
        """
        if self._migration_active:
            self.logger.warning("Migration already active")
            return False
        
        try:
            self.logger.info("Starting coordinated system migration")
            
            # Initialize migration coordinator
            from redis_migration_coordinator import create_migration_coordinator
            self._coordinator = await create_migration_coordinator(self.config)
            
            # Start migration
            self._migration_active = True
            success = await self._coordinator.start_migration()
            
            if success:
                self.logger.info("Coordinated migration completed successfully")
            else:
                self.logger.error("Coordinated migration failed")
            
            self._migration_active = False
            return success
            
        except Exception as e:
            self.logger.error(f"Error in coordinated migration: {e}")
            self._migration_active = False
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system migration status.
        
        Returns:
            System status dictionary
        """
        status = {
            'adapters_initialized': self._adapters_initialized,
            'migration_active': self._migration_active,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Get coordinator status
        if self._coordinator:
            status['migration'] = self._coordinator.get_migration_status()
        
        # Get adapter statuses
        if self._cache_adapter:
            try:
                status['cache'] = await self._cache_adapter.get_migration_status()
            except Exception as e:
                status['cache'] = {'error': str(e)}
        
        if self._state_adapter:
            try:
                status['state'] = await self._state_adapter.get_migration_status()
            except Exception as e:
                status['state'] = {'error': str(e)}
        
        return status
    
    async def rollback_system(self) -> bool:
        """Rollback entire system to legacy mode.
        
        Returns:
            True if rollback successful
        """
        try:
            self.logger.warning("Starting system rollback")
            
            if self._coordinator:
                success = await self._coordinator.rollback_migration()
                if success:
                    self.logger.info("System rollback completed successfully")
                else:
                    self.logger.error("System rollback failed")
                return success
            else:
                self.logger.warning("No coordinator available for rollback")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in system rollback: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown system migration manager."""
        try:
            self.logger.info("Shutting down system migration manager")
            
            # Shutdown coordinator
            if self._coordinator:
                await self._coordinator.shutdown()
            
            # Shutdown adapters
            if self._cache_adapter:
                await self._cache_adapter.shutdown()
            
            if self._state_adapter:
                await self._state_adapter.shutdown()
            
            self.logger.info("System migration manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error shutting down system migration manager: {e}")


# Global instances for easy access
_global_cache_adapter: Optional[EnhancedAIResponseCacheAdapter] = None
_global_state_adapter: Optional[EnhancedAsyncStateManagerAdapter] = None
_global_migration_manager: Optional[SystemMigrationManager] = None


async def get_enhanced_cache(enable_redis: bool = True, migration_mode: str = "gradual") -> EnhancedAIResponseCacheAdapter:
    """Get global enhanced AI cache adapter."""
    global _global_cache_adapter
    
    if _global_cache_adapter is None:
        _global_cache_adapter = EnhancedAIResponseCacheAdapter(
            enable_redis=enable_redis,
            migration_mode=migration_mode
        )
        await _global_cache_adapter._ensure_initialized()
    
    return _global_cache_adapter


async def get_enhanced_state_manager(enable_redis: bool = True, migration_mode: str = "gradual") -> EnhancedAsyncStateManagerAdapter:
    """Get global enhanced async state manager adapter."""
    global _global_state_adapter
    
    if _global_state_adapter is None:
        _global_state_adapter = EnhancedAsyncStateManagerAdapter(
            enable_redis=enable_redis,
            migration_mode=migration_mode
        )
        await _global_state_adapter._ensure_initialized()
    
    return _global_state_adapter


async def get_migration_manager(config: Optional[MigrationConfig] = None) -> SystemMigrationManager:
    """Get global system migration manager."""
    global _global_migration_manager
    
    if _global_migration_manager is None:
        _global_migration_manager = SystemMigrationManager(config)
    
    return _global_migration_manager


# Convenience functions that can replace existing imports
async def enhanced_cache_ai_response(prompt: str, response: str, provider: str, model: str, cost_estimate: float = 0.0):
    """Enhanced cache AI response function."""
    cache = await get_enhanced_cache()
    await cache.put(prompt, response, provider, model, cost_estimate=cost_estimate)


async def get_enhanced_cached_response(prompt: str, provider: str, model: str) -> Optional[str]:
    """Enhanced get cached AI response function."""
    cache = await get_enhanced_cache()
    return await cache.get(prompt, provider, model)


async def enhanced_state_update(key_path: str, value: Any, distributed: bool = False):
    """Enhanced state update function."""
    state_manager = await get_enhanced_state_manager()
    await state_manager.update(key_path, value, distributed=distributed)


async def enhanced_state_get(key_path: str, default: Any = None) -> Any:
    """Enhanced state get function."""
    state_manager = await get_enhanced_state_manager()
    return await state_manager.get(key_path, default)