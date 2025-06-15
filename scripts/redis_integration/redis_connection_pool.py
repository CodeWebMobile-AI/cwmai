"""
Redis Connection Pool Manager

Singleton connection pool manager that ensures proper connection sharing
across all components with connection limits and cleanup.
"""

import asyncio
import logging
import time
import weakref
from typing import Optional, Dict, Any, Set
from contextlib import asynccontextmanager
import redis.asyncio as redis
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.sentinel import Sentinel
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from .redis_config import RedisConfig, RedisMode, get_redis_config


class SingletonConnectionPool:
    """Singleton connection pool manager for Redis."""
    
    _instance: Optional['SingletonConnectionPool'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize connection pool manager."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = logging.getLogger(__name__)
        self.config = get_redis_config()
        
        # Connection pools by mode
        self._pools: Dict[str, Any] = {}
        self._connections: Dict[str, Any] = {}
        
        # Pub/Sub management: single shared Pub/Sub connection
        self._shared_pubsub: Optional[Any] = None
        self._pubsub_connections: Dict[str, Any] = {}
        self._pubsub_refs: Dict[str, weakref.ReferenceType] = {}
        
        # Connection tracking
        self._active_connections = 0
        self._connection_limit = self.config.connection_config.max_connections
        self._connection_semaphore = asyncio.Semaphore(self._connection_limit)
        
        # Reconnection tracking
        self._reconnect_attempts: Dict[str, int] = {}
        self._max_reconnect_attempts = 5
        self._reconnect_backoff_base = 2
        self._reconnect_backoff_max = 60
        
        # Cleanup tracking
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
    @classmethod
    async def get_instance(cls) -> 'SingletonConnectionPool':
        """Get singleton instance with async initialization."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    async def start(self):
        """Start connection pool manager."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info("Connection pool manager started")
    
    async def stop(self):
        """Stop connection pool manager and cleanup."""
        self._shutdown = True
        
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        await self._close_all_connections()
        
        self.logger.info("Connection pool manager stopped")
    
    async def get_connection(self, connection_id: str = "default") -> redis.Redis:
        """Get or create a Redis connection.
        
        Args:
            connection_id: Unique identifier for the connection
            
        Returns:
            Redis connection instance
        """
        async with self._connection_semaphore:
            if connection_id in self._connections:
                conn = self._connections[connection_id]
                # Verify connection is still alive
                try:
                    await conn.ping()
                    return conn
                except (ConnectionError, TimeoutError):
                    self.logger.warning(f"Connection {connection_id} lost, reconnecting...")
                    await self._remove_connection(connection_id)
            
            # Create new connection
            return await self._create_connection(connection_id)
    
    async def get_pubsub(self) -> Any:
        """Get or create a single shared Pub/Sub connection for all subscriptions."""
        if self._shared_pubsub:
            return self._shared_pubsub

        async with self._connection_semaphore:
            conn = await self.get_connection("shared_pubsub")
            self._shared_pubsub = conn.pubsub()
            self.logger.debug("Created shared Pub/Sub connection")
            return self._shared_pubsub
    
    async def _create_connection(self, connection_id: str) -> redis.Redis:
        """Create a new Redis connection."""
        try:
            self.logger.info(f"Creating Redis connection: {connection_id}")
            
            if self.config.mode == RedisMode.CLUSTER:
                conn = await self._create_cluster_connection()
            elif self.config.mode == RedisMode.SENTINEL:
                conn = await self._create_sentinel_connection()
            else:
                conn = await self._create_standalone_connection()
            
            # Test connection
            await conn.ping()
            
            # Store connection
            self._connections[connection_id] = conn
            self._active_connections += 1
            self._reconnect_attempts[connection_id] = 0
            
            self.logger.info(f"Redis connection established: {connection_id} (active: {self._active_connections})")
            return conn
            
        except Exception as e:
            self.logger.error(f"Failed to create connection {connection_id}: {e}")
            await self._handle_connection_failure(connection_id)
            raise
    
    async def _create_standalone_connection(self) -> redis.Redis:
        """Create standalone Redis connection with shared pool."""
        pool_key = "standalone"
        
        if pool_key not in self._pools:
            # Create connection pool
            connection_kwargs = self.config.get_connection_kwargs()
            pool_kwargs = connection_kwargs.copy()
            pool_kwargs['max_connections'] = self._connection_limit
            
            self._pools[pool_key] = redis.ConnectionPool(**pool_kwargs)
            self.logger.info(f"Created connection pool with limit: {self._connection_limit}")
        
        return redis.Redis(connection_pool=self._pools[pool_key])
    
    async def _create_cluster_connection(self) -> RedisCluster:
        """Create Redis cluster connection."""
        cluster_kwargs = self.config.get_cluster_kwargs()
        return RedisCluster(**cluster_kwargs)
    
    async def _create_sentinel_connection(self) -> redis.Redis:
        """Create Redis sentinel connection."""
        sentinel_kwargs = self.config.get_sentinel_kwargs()
        sentinel = Sentinel(
            sentinel_kwargs['sentinels'],
            **sentinel_kwargs.get('sentinel_kwargs', {})
        )
        return sentinel.master_for(sentinel_kwargs['service_name'])
    
    async def _remove_connection(self, connection_id: str):
        """Remove and close a connection."""
        if connection_id in self._connections:
            try:
                conn = self._connections[connection_id]
                await conn.close()
            except Exception as e:
                self.logger.error(f"Error closing connection {connection_id}: {e}")
            finally:
                del self._connections[connection_id]
                self._active_connections = max(0, self._active_connections - 1)
                self.logger.debug(f"Removed connection {connection_id} (active: {self._active_connections})")
    
    async def _remove_pubsub(self, channel_pattern: str):
        """Remove and close a pubsub connection."""
        if channel_pattern in self._pubsub_connections:
            try:
                pubsub = self._pubsub_connections[channel_pattern]
                await pubsub.close()
            except Exception as e:
                self.logger.error(f"Error closing pubsub {channel_pattern}: {e}")
            finally:
                del self._pubsub_connections[channel_pattern]
                self._pubsub_refs.pop(channel_pattern, None)
                self.logger.debug(f"Removed pubsub connection for {channel_pattern}")
    
    async def _handle_connection_failure(self, connection_id: str):
        """Handle connection failure with exponential backoff."""
        attempts = self._reconnect_attempts.get(connection_id, 0) + 1
        self._reconnect_attempts[connection_id] = attempts
        
        if attempts <= self._max_reconnect_attempts:
            backoff_time = min(
                self._reconnect_backoff_base ** attempts,
                self._reconnect_backoff_max
            )
            self.logger.warning(
                f"Connection {connection_id} failed, retrying in {backoff_time}s "
                f"(attempt {attempts}/{self._max_reconnect_attempts})"
            )
            await asyncio.sleep(backoff_time)
        else:
            self.logger.error(f"Max reconnection attempts reached for {connection_id}")
            raise ConnectionError(f"Failed to establish connection after {attempts} attempts")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of idle connections."""
        cleanup_interval = 60  # 1 minute
        
        while not self._shutdown:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._cleanup_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_idle_connections(self):
        """Clean up idle connections."""
        # For now, just log status
        pubsub_count = len(self._pubsub_connections) if hasattr(self, '_pubsub_connections') else 0
        self.logger.debug(
            f"Connection pool status - Active: {self._active_connections}, "
            f"Limit: {self._connection_limit}, "
            f"Connections: {len(self._connections)}, "
            f"PubSubs: {pubsub_count}"
        )
        
        # Clean up dead pubsub weak references
        dead_patterns = []
        for pattern, ref in self._pubsub_refs.items():
            if ref() is None:
                dead_patterns.append(pattern)
        
        for pattern in dead_patterns:
            await self._remove_pubsub(pattern)
    
    async def cleanup_stale_connections(self) -> int:
        """Clean up stale connections from the pool.
        
        Returns:
            Number of connections cleaned up
        """
        cleaned = 0
        
        # Check all connections for health
        connection_ids = list(self._connections.keys())
        for conn_id in connection_ids:
            try:
                conn = self._connections.get(conn_id)
                if conn:
                    # Try to ping the connection
                    await asyncio.wait_for(conn.ping(), timeout=1.0)
            except (asyncio.TimeoutError, ConnectionError, RedisError) as e:
                # Connection is stale, remove it
                self.logger.warning(f"Removing stale connection {conn_id}: {e}")
                await self._remove_connection(conn_id)
                cleaned += 1
            except Exception as e:
                self.logger.error(f"Error checking connection {conn_id}: {e}")
        
        # Clean up dead pubsub connections
        patterns = list(self._pubsub_connections.keys())
        for pattern in patterns:
            try:
                pubsub = self._pubsub_connections.get(pattern)
                ref = self._pubsub_refs.get(pattern)
                
                # Remove if weak reference is dead
                if ref and ref() is None:
                    await self._remove_pubsub(pattern)
                    cleaned += 1
                    continue
                
                # Try to check if pubsub is healthy
                if pubsub and hasattr(pubsub, 'connection') and pubsub.connection:
                    await asyncio.wait_for(pubsub.connection.ping(), timeout=1.0)
            except (asyncio.TimeoutError, ConnectionError, RedisError, AttributeError) as e:
                # PubSub connection is stale
                self.logger.warning(f"Removing stale pubsub {pattern}: {e}")
                await self._remove_pubsub(pattern)
                cleaned += 1
            except Exception as e:
                self.logger.error(f"Error checking pubsub {pattern}: {e}")
        
        if cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned} stale connections")
        
        return cleaned
    
    async def _close_all_connections(self):
        """Close all connections."""
        # Close regular connections
        connection_ids = list(self._connections.keys())
        for conn_id in connection_ids:
            await self._remove_connection(conn_id)
        
        # Close pubsub connections
        patterns = list(self._pubsub_connections.keys())
        for pattern in patterns:
            await self._remove_pubsub(pattern)
        
        # Close connection pools
        for pool in self._pools.values():
            try:
                await pool.disconnect()
            except Exception as e:
                self.logger.error(f"Error closing connection pool: {e}")
        
        self._pools.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'active_connections': self._active_connections,
            'connection_limit': self._connection_limit,
            'regular_connections': len(self._connections),
            'pubsub_connections': len(self._pubsub_connections),
            'pools': len(self._pools),
            'reconnect_attempts': dict(self._reconnect_attempts)
        }


# Global connection pool instance
_connection_pool: Optional[SingletonConnectionPool] = None


async def get_connection_pool() -> SingletonConnectionPool:
    """Get global connection pool instance."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = await SingletonConnectionPool.get_instance()
        await _connection_pool.start()
    return _connection_pool


async def close_connection_pool():
    """Close global connection pool."""
    global _connection_pool
    if _connection_pool:
        await _connection_pool.stop()
        _connection_pool = None