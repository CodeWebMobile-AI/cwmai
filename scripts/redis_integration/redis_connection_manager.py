"""
Enhanced Redis Connection Manager with Failover and Recovery

Provides robust Redis connection management with:
- Automatic health checks and recovery
- Circuit breaker pattern for failure isolation
- SQLite fallback for persistence when Redis is down
- Connection pooling with proper resource management
"""

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from enum import Enum

from scripts.redis_integration.redis_client import RedisClient, RedisConnectionError, RedisCircuitBreakerError
from scripts.redis_integration.redis_config import get_redis_config


class ConnectionState(Enum):
    """Connection state enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


class SQLiteFallback:
    """SQLite fallback storage for Redis operations."""
    
    def __init__(self, db_path: str = "redis_fallback.db"):
        """Initialize SQLite fallback.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expire_at REAL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hash_store (
                    hash_key TEXT,
                    field TEXT,
                    value TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    PRIMARY KEY (hash_key, field)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS set_store (
                    set_key TEXT,
                    member TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    PRIMARY KEY (set_key, member)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sorted_set_store (
                    set_key TEXT,
                    member TEXT,
                    score REAL,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    PRIMARY KEY (set_key, member)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS list_store (
                    list_key TEXT,
                    position INTEGER,
                    value TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    PRIMARY KEY (list_key, position)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expire ON kv_store(expire_at)
            """)
            
            conn.commit()
    
    def _cleanup_expired(self):
        """Remove expired keys."""
        current_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM kv_store WHERE expire_at IS NOT NULL AND expire_at < ?", (current_time,))
            conn.commit()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        self._cleanup_expired()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM kv_store WHERE key = ? AND (expire_at IS NULL OR expire_at > ?)",
                (key, time.time())
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set key-value pair."""
        expire_at = time.time() + ex if ex else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value, expire_at) VALUES (?, ?, ?)",
                (key, str(value), expire_at)
            )
            conn.commit()
        return True
    
    async def delete(self, *keys: str) -> int:
        """Delete keys."""
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' * len(keys))
            cursor = conn.execute(f"DELETE FROM kv_store WHERE key IN ({placeholders})", keys)
            conn.commit()
            return cursor.rowcount
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        self._cleanup_expired()
        
        with sqlite3.connect(self.db_path) as conn:
            count = 0
            for key in keys:
                cursor = conn.execute(
                    "SELECT 1 FROM kv_store WHERE key = ? AND (expire_at IS NULL OR expire_at > ?)",
                    (key, time.time())
                )
                if cursor.fetchone():
                    count += 1
            return count
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM hash_store WHERE hash_key = ? AND field = ?",
                (name, key)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields."""
        with sqlite3.connect(self.db_path) as conn:
            for field, value in mapping.items():
                conn.execute(
                    "INSERT OR REPLACE INTO hash_store (hash_key, field, value) VALUES (?, ?, ?)",
                    (name, field, str(value))
                )
            conn.commit()
            return len(mapping)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT field, value FROM hash_store WHERE hash_key = ?",
                (name,)
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    async def sadd(self, name: str, *values: Any) -> int:
        """Add members to set."""
        with sqlite3.connect(self.db_path) as conn:
            added = 0
            for value in values:
                try:
                    conn.execute(
                        "INSERT INTO set_store (set_key, member) VALUES (?, ?)",
                        (name, str(value))
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    # Already exists
                    pass
            conn.commit()
            return added
    
    async def smembers(self, name: str) -> set:
        """Get all set members."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT member FROM set_store WHERE set_key = ?",
                (name,)
            )
            return {row[0] for row in cursor.fetchall()}
    
    async def zadd(self, name: str, mapping: Dict[Any, float]) -> int:
        """Add members to sorted set."""
        with sqlite3.connect(self.db_path) as conn:
            for member, score in mapping.items():
                conn.execute(
                    "INSERT OR REPLACE INTO sorted_set_store (set_key, member, score) VALUES (?, ?, ?)",
                    (name, str(member), score)
                )
            conn.commit()
            return len(mapping)
    
    async def zrange(self, name: str, start: int, end: int, withscores: bool = False) -> List:
        """Get sorted set range."""
        with sqlite3.connect(self.db_path) as conn:
            # SQLite doesn't have LIMIT with negative offsets, so we need to handle this
            if end == -1:
                cursor = conn.execute(
                    "SELECT member, score FROM sorted_set_store WHERE set_key = ? ORDER BY score LIMIT -1 OFFSET ?",
                    (name, start)
                )
            else:
                limit = end - start + 1
                cursor = conn.execute(
                    "SELECT member, score FROM sorted_set_store WHERE set_key = ? ORDER BY score LIMIT ? OFFSET ?",
                    (name, limit, start)
                )
            
            if withscores:
                return [(row[0], row[1]) for row in cursor.fetchall()]
            else:
                return [row[0] for row in cursor.fetchall()]
    
    async def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        """Increment hash field value."""
        with sqlite3.connect(self.db_path) as conn:
            # Get current value
            cursor = conn.execute(
                "SELECT value FROM hash_store WHERE hash_key = ? AND field = ?",
                (name, key)
            )
            row = cursor.fetchone()
            current = int(row[0]) if row else 0
            
            # Update with new value
            new_value = current + amount
            conn.execute(
                "INSERT OR REPLACE INTO hash_store (hash_key, field, value) VALUES (?, ?, ?)",
                (name, key, str(new_value))
            )
            conn.commit()
            return new_value
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration."""
        expire_at = time.time() + seconds
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE kv_store SET expire_at = ? WHERE key = ?",
                (expire_at, key)
            )
            conn.commit()
            return cursor.rowcount > 0


class RedisConnectionManager:
    """Enhanced Redis connection manager with failover support."""
    
    def __init__(self, enable_fallback: bool = True):
        """Initialize connection manager.
        
        Args:
            enable_fallback: Whether to enable SQLite fallback
        """
        self.redis_client: Optional[RedisClient] = None
        self.sqlite_fallback: Optional[SQLiteFallback] = None
        self.enable_fallback = enable_fallback
        
        self.state = ConnectionState.FAILED
        self.last_health_check = 0
        self.health_check_interval = 5  # seconds
        self.recovery_attempts = 0
        self.max_recovery_attempts = 5
        
        self.logger = logging.getLogger(__name__)
        self._health_monitor_task = None
        self._shutdown = False
        
        # Metrics
        self.metrics = {
            'redis_operations': 0,
            'fallback_operations': 0,
            'recovery_attempts': 0,
            'circuit_breaker_trips': 0,
            'last_state_change': time.time()
        }
        
        if enable_fallback:
            self.sqlite_fallback = SQLiteFallback()
            print("SQLite fallback enabled for Redis operations")
    
    async def initialize(self):
        """Initialize the connection manager."""
        try:
            # Try to establish Redis connection
            self.redis_client = RedisClient(use_connection_pool=True)
            await self.redis_client.connect()
            
            # Start health monitoring
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            self.state = ConnectionState.HEALTHY
            print("Redis connection manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis connection: {e}")
            self.state = ConnectionState.FAILED
            
            if not self.enable_fallback:
                raise
            
            self.logger.warning("Redis unavailable, using SQLite fallback")
    
    async def shutdown(self):
        """Shutdown the connection manager."""
        self._shutdown = True
        
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.disconnect()
        
        print("Redis connection manager shutdown complete")
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while not self._shutdown:
            try:
                await self._check_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_health(self):
        """Perform health check and recovery if needed."""
        try:
            if self.redis_client and self.redis_client._connected:
                # Test Redis connection
                await asyncio.wait_for(self.redis_client.ping(), timeout=2.0)
                
                if self.state != ConnectionState.HEALTHY:
                    print("Redis connection recovered")
                    self.state = ConnectionState.HEALTHY
                    self.recovery_attempts = 0
                    self.metrics['last_state_change'] = time.time()
                
            else:
                raise RedisConnectionError("Redis client not connected")
                
        except Exception as e:
            if self.state == ConnectionState.HEALTHY:
                self.logger.error(f"Redis health check failed: {e}")
                self.state = ConnectionState.DEGRADED
                self.metrics['last_state_change'] = time.time()
            
            # Attempt recovery
            if self.recovery_attempts < self.max_recovery_attempts:
                await self._attempt_recovery()
            else:
                if self.state != ConnectionState.FAILED:
                    self.logger.error("Max recovery attempts reached, marking connection as failed")
                    self.state = ConnectionState.FAILED
                    self.metrics['last_state_change'] = time.time()
    
    async def _attempt_recovery(self):
        """Attempt to recover Redis connection."""
        self.recovery_attempts += 1
        self.metrics['recovery_attempts'] += 1
        
        print(f"Attempting Redis recovery (attempt {self.recovery_attempts}/{self.max_recovery_attempts})")
        
        try:
            if self.redis_client:
                await self.redis_client.disconnect()
            
            self.redis_client = RedisClient(use_connection_pool=True)
            await self.redis_client.connect()
            
            # Test the connection
            await self.redis_client.ping()
            
            print("Redis recovery successful")
            self.state = ConnectionState.HEALTHY
            self.recovery_attempts = 0
            
        except Exception as e:
            self.logger.error(f"Redis recovery failed: {e}")
            self.state = ConnectionState.RECOVERING
    
    async def _execute_with_fallback(self, operation: str, *args, **kwargs):
        """Execute operation with automatic fallback to SQLite.
        
        Args:
            operation: Operation name (e.g., 'get', 'set', 'hget')
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        # Try Redis first if healthy
        if self.state in [ConnectionState.HEALTHY, ConnectionState.DEGRADED]:
            try:
                if self.redis_client and self.redis_client._connected:
                    method = getattr(self.redis_client, operation)
                    result = await method(*args, **kwargs)
                    self.metrics['redis_operations'] += 1
                    return result
            except (RedisConnectionError, RedisCircuitBreakerError) as e:
                self.logger.warning(f"Redis operation failed, using fallback: {e}")
                self.metrics['circuit_breaker_trips'] += 1
            except Exception as e:
                self.logger.error(f"Unexpected Redis error: {e}")
        
        # Use SQLite fallback if enabled
        if self.enable_fallback and self.sqlite_fallback:
            self.logger.debug(f"Using SQLite fallback for operation: {operation}")
            method = getattr(self.sqlite_fallback, operation, None)
            
            if method:
                result = await method(*args, **kwargs)
                self.metrics['fallback_operations'] += 1
                return result
            else:
                self.logger.warning(f"Operation '{operation}' not supported in SQLite fallback")
                return None
        
        # No fallback available
        raise RedisConnectionError(f"Redis unavailable and no fallback for operation: {operation}")
    
    # Proxy all Redis operations
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        return await self._execute_with_fallback('get', key)
    
    async def set(self, key: str, value: Any, ex: Optional[int] = None, nx: bool = False) -> bool:
        """Set key-value pair."""
        # SQLite doesn't support nx flag, so we handle it separately
        if nx and self.state == ConnectionState.FAILED and self.enable_fallback:
            exists = await self.exists(key)
            if exists:
                return False
        return await self._execute_with_fallback('set', key, value, ex=ex)
    
    async def delete(self, *keys: str) -> int:
        """Delete keys."""
        return await self._execute_with_fallback('delete', *keys)
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        return await self._execute_with_fallback('exists', *keys)
    
    async def expire(self, key: str, time: int) -> bool:
        """Set key expiration."""
        return await self._execute_with_fallback('expire', key, time)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value."""
        return await self._execute_with_fallback('hget', name, key)
    
    async def hset(self, name: str, key: str = None, value: Any = None, mapping: Dict[str, Any] = None) -> int:
        """Set hash field value."""
        # Normalize arguments for SQLite fallback
        if self.state == ConnectionState.FAILED and self.enable_fallback:
            if mapping is None and key is not None and value is not None:
                mapping = {key: value}
            return await self._execute_with_fallback('hset', name, mapping=mapping)
        return await self._execute_with_fallback('hset', name, key=key, value=value, mapping=mapping)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        return await self._execute_with_fallback('hgetall', name)
    
    async def sadd(self, name: str, *values: Any) -> int:
        """Add members to set."""
        return await self._execute_with_fallback('sadd', name, *values)
    
    async def smembers(self, name: str) -> set:
        """Get all set members."""
        return await self._execute_with_fallback('smembers', name)
    
    async def zadd(self, name: str, mapping: Dict[Any, float], nx: bool = False, xx: bool = False) -> int:
        """Add members to sorted set."""
        # SQLite doesn't support nx/xx flags
        if self.state == ConnectionState.FAILED and self.enable_fallback:
            return await self._execute_with_fallback('zadd', name, mapping)
        return await self._execute_with_fallback('zadd', name, mapping, nx=nx, xx=xx)
    
    async def zrange(self, name: str, start: int, end: int, withscores: bool = False) -> List:
        """Get sorted set range."""
        return await self._execute_with_fallback('zrange', name, start, end, withscores=withscores)
    
    async def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        """Increment hash field value."""
        return await self._execute_with_fallback('hincrby', name, key, amount)
    
    async def ping(self) -> bool:
        """Ping Redis server."""
        if self.state in [ConnectionState.HEALTHY, ConnectionState.DEGRADED]:
            try:
                if self.redis_client:
                    return await self.redis_client.ping()
            except:
                pass
        return self.state != ConnectionState.FAILED
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection manager status."""
        return {
            'state': self.state.value,
            'redis_connected': self.redis_client._connected if self.redis_client else False,
            'fallback_enabled': self.enable_fallback,
            'recovery_attempts': self.recovery_attempts,
            'metrics': self.metrics,
            'health_stats': self.redis_client.get_connection_stats() if self.redis_client else None
        }
    
    @asynccontextmanager
    async def get_connection(self):
        """Get managed connection context."""
        if self.state in [ConnectionState.HEALTHY, ConnectionState.DEGRADED] and self.redis_client:
            async with self.redis_client.get_connection() as conn:
                yield conn
        else:
            # Yield the manager itself for fallback operations
            yield self


# Global instance
_connection_manager: Optional[RedisConnectionManager] = None
_manager_lock = asyncio.Lock()


async def get_connection_manager(enable_fallback: bool = True) -> RedisConnectionManager:
    """Get global connection manager instance.
    
    Args:
        enable_fallback: Whether to enable SQLite fallback
        
    Returns:
        Connection manager instance
    """
    global _connection_manager
    
    async with _manager_lock:
        if _connection_manager is None:
            _connection_manager = RedisConnectionManager(enable_fallback=enable_fallback)
            await _connection_manager.initialize()
        return _connection_manager


async def close_connection_manager():
    """Close global connection manager instance."""
    global _connection_manager
    
    async with _manager_lock:
        if _connection_manager:
            await _connection_manager.shutdown()
            _connection_manager = None