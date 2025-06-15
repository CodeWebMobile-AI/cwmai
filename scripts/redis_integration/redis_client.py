"""
Redis Client with Advanced Connection Management

Enterprise-grade Redis client with connection pooling, failover, health monitoring,
and automatic reconnection. Supports standalone, cluster, and sentinel deployments.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from contextlib import asynccontextmanager
import redis.asyncio as redis
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.sentinel import Sentinel
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from scripts.redis_integration.redis_config import RedisConfig, RedisMode, get_redis_config
from scripts.redis_integration.redis_connection_pool import get_connection_pool, SingletonConnectionPool


class RedisConnectionError(Exception):
    """Redis connection related errors."""
    pass


class RedisCircuitBreakerError(Exception):
    """Circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for Redis operations."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if time.time() - self.last_failure_time < self.timeout:
                self.logger.debug(f"Circuit breaker is open, will retry in {self.timeout - (time.time() - self.last_failure_time):.1f}s")
                raise RedisCircuitBreakerError("Circuit breaker is open")
            else:
                self.logger.info("Circuit breaker entering half-open state for testing")
                self.state = 'half-open'
        
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                self.logger.info("Circuit breaker closed - connection restored")
            elif self.failure_count > 0:
                self.logger.debug(f"Operation succeeded, failure count: {self.failure_count}")
            return result
        except Exception as e:
            error_msg = str(e).lower()
            error_type = type(e).__name__
            
            # List of transient errors that shouldn't count towards circuit breaker
            transient_errors = [
                "no such key",
                "key does not exist",
                "nil",
                "busygroup",  # Consumer group already exists
                "wrongtype",  # Wrong Redis data type
                "moved",  # Cluster slot moved
                "ask",  # Cluster redirect
                "setinfo",  # CLIENT SETINFO not supported in older Redis
                "unknown subcommand"  # Generic for unsupported commands
            ]
            
            # Check if this is a transient error
            is_transient = any(err in error_msg for err in transient_errors)
            
            if not is_transient:
                # Log the actual error that's causing the failure
                self.logger.warning(f"Circuit breaker detected failure #{self.failure_count + 1}: {error_type}: {str(e)}")
                
                # Special handling for connection errors
                if "too many connections" in error_msg:
                    self.logger.error("Connection limit reached! Consider increasing REDIS_MAX_CONNECTIONS")
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Log the specific error causing the failure
                self.logger.warning(f"Circuit breaker failure #{self.failure_count}: {error_type}: {e}")
                
                # Special handling for connection limit errors
                if "too many connections" in error_msg:
                    self.logger.error("Connection limit reached - this indicates a connection leak or pool exhaustion")
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    self.logger.error(f"Circuit breaker opened after {self.failure_count} failures. Last error: {error_type}: {e}")
            else:
                self.logger.debug(f"Ignoring transient error for circuit breaker: {error_type}: {e}")
            
            raise e


class RedisHealthMonitor:
    """Redis connection health monitoring."""
    
    def __init__(self, redis_client, check_interval: float = 30.0):
        """Initialize health monitor.
        
        Args:
            redis_client: Redis client instance
            check_interval: Health check interval in seconds
        """
        self.redis_client = redis_client
        self.check_interval = check_interval
        self.is_healthy = True
        self.last_check = 0
        self.check_count = 0
        self.failure_count = 0
        self.consecutive_failures = 0
        self.logger = logging.getLogger(__name__)
        self._monitor_task = None
        self._shutdown = False
        self._last_health_change = time.time()
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            self.logger.info("Redis health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._shutdown = True
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            self.logger.info("Redis health monitoring stopped")
    
    async def _monitor_loop(self):
        """Health monitoring loop."""
        while not self._shutdown:
            try:
                await self.check_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_health(self) -> bool:
        """Perform health check."""
        try:
            self.check_count += 1
            start_time = time.time()
            
            # Simple ping test
            await self.redis_client.ping()
            
            response_time = time.time() - start_time
            self.last_check = time.time()
            
            # Reset consecutive failures on success
            self.consecutive_failures = 0
            
            # Only log state changes to avoid spam
            if not self.is_healthy:
                self.logger.info("Redis connection restored")
                self.is_healthy = True
                self._last_health_change = time.time()
            
            # Only log detailed health checks at debug level
            if self.check_count % 10 == 0:  # Log every 10th check
                self.logger.debug(f"Redis health check passed in {response_time:.3f}s")
            
            return True
            
        except Exception as e:
            self.failure_count += 1
            self.consecutive_failures += 1
            
            # Only mark unhealthy after multiple consecutive failures
            if self.consecutive_failures >= 3 and self.is_healthy:
                self.is_healthy = False
                self._last_health_change = time.time()
                self.logger.error(f"Redis health check failed after {self.consecutive_failures} attempts: {e}")
            elif self.consecutive_failures < 3:
                self.logger.warning(f"Redis health check warning ({self.consecutive_failures}/3): {e}")
            
            return False
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health monitoring statistics."""
        return {
            'is_healthy': self.is_healthy,
            'last_check': self.last_check,
            'check_count': self.check_count,
            'failure_count': self.failure_count,
            'failure_rate': self.failure_count / max(self.check_count, 1),
            'uptime_percentage': (self.check_count - self.failure_count) / max(self.check_count, 1) * 100
        }


class RedisClient:
    """Enterprise Redis client with advanced features."""
    
    def __init__(self, config: RedisConfig = None, use_connection_pool: bool = True):
        """Initialize Redis client.
        
        Args:
            config: Redis configuration
            use_connection_pool: Whether to use singleton connection pool
        """
        self.config = config or get_redis_config()
        self.redis: Optional[Union[redis.Redis, RedisCluster]] = None
        self.connection_pool: Optional[SingletonConnectionPool] = None
        self.use_connection_pool = use_connection_pool
        
        # Initialize circuit breaker with config
        if self.config.connection_config.circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.connection_config.circuit_breaker_failure_threshold,
                timeout=self.config.connection_config.circuit_breaker_timeout
            )
        else:
            self.circuit_breaker = None
            
        self.health_monitor = None
        self.connection_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        self._connection_lock = asyncio.Lock()
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._health_check_enabled = True
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        async with self._connection_lock:
            if self._connected:
                return
            
            try:
                self.logger.info(f"Connecting to Redis ({self.config.mode.value}) at {self.config.connection_config.host}:{self.config.connection_config.port}")
                
                if self.use_connection_pool:
                    # Use singleton connection pool
                    self.connection_pool = await get_connection_pool()
                    self.redis = await self.connection_pool.get_connection(self.connection_id)
                else:
                    # Create dedicated connection
                    if self.config.mode == RedisMode.CLUSTER:
                        await self._connect_cluster()
                    elif self.config.mode == RedisMode.SENTINEL:
                        await self._connect_sentinel()
                    else:
                        await self._connect_standalone()
                
                # Test connection
                await self.redis.ping()
                
                # Start health monitoring only if enabled
                if self._health_check_enabled:
                    self.health_monitor = RedisHealthMonitor(self.redis, self.config.connection_config.health_check_interval)
                    await self.health_monitor.start_monitoring()
                
                self._connected = True
                self._reconnect_attempts = 0
                self.logger.info(f"Redis connection established (ID: {self.connection_id}, pooled: {self.use_connection_pool})")
                
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                await self._handle_connection_failure()
                raise RedisConnectionError(f"Failed to connect to Redis: {e}")
    
    async def _connect_standalone(self):
        """Connect to standalone Redis instance."""
        if self.use_connection_pool:
            # Connection will be managed by singleton pool
            return
        
        # Create dedicated connection pool for this client
        connection_kwargs = self.config.get_connection_kwargs()
        pool_kwargs = connection_kwargs.copy()
        pool_kwargs['max_connections'] = self.config.connection_config.max_connections
        
        self._local_pool = redis.ConnectionPool(**pool_kwargs)
        self.redis = redis.Redis(connection_pool=self._local_pool)
    
    async def _connect_cluster(self):
        """Connect to Redis cluster."""
        cluster_kwargs = self.config.get_cluster_kwargs()
        self.redis = RedisCluster(**cluster_kwargs)
    
    async def _connect_sentinel(self):
        """Connect to Redis via Sentinel."""
        sentinel_kwargs = self.config.get_sentinel_kwargs()
        sentinel = Sentinel(sentinel_kwargs['sentinels'], **sentinel_kwargs.get('sentinel_kwargs', {}))
        self.redis = sentinel.master_for(sentinel_kwargs['service_name'])
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        async with self._connection_lock:
            if not self._connected:
                return
            
            try:
                # Stop health monitoring
                if self.health_monitor:
                    await self.health_monitor.stop_monitoring()
                    self.health_monitor = None
                
                # If using connection pool, we don't close the shared connection
                if not self.use_connection_pool and self.redis:
                    await self.redis.close()
                    
                    # Close local connection pool if exists
                    if hasattr(self, '_local_pool') and self._local_pool:
                        await self._local_pool.disconnect()
                        self._local_pool = None
                
                self.redis = None
                self._connected = False
                self.logger.info(f"Redis connection closed (ID: {self.connection_id})")
                
            except Exception as e:
                self.logger.error(f"Error closing Redis connection: {e}")
    
    async def _handle_connection_failure(self):
        """Handle connection failure with exponential backoff."""
        self._reconnect_attempts += 1
        if self._reconnect_attempts <= self._max_reconnect_attempts:
            backoff_time = min(2 ** self._reconnect_attempts, 60)
            self.logger.warning(f"Redis connection failed, retrying in {backoff_time}s (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})")
            await asyncio.sleep(backoff_time)
        else:
            self.logger.error("Max reconnection attempts reached")
    
    async def ensure_connected(self):
        """Ensure Redis connection is established."""
        if not self._connected or not self.redis:
            await self.connect()
            return
        
        # Only check connection health if not using connection pool
        # (pool manages its own connection health)
        if not self.use_connection_pool:
            try:
                await self.redis.ping()
            except (ConnectionError, TimeoutError):
                self.logger.warning("Redis connection lost, reconnecting...")
                self._connected = False
                await self.connect()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get Redis connection with automatic management."""
        await self.ensure_connected()
        try:
            yield self.redis
        except (ConnectionError, TimeoutError) as e:
            self.logger.error(f"Redis operation failed: {e}")
            self._connected = False
            raise RedisConnectionError(f"Redis operation failed: {e}")
    
    async def execute_with_retry(self, operation: Callable, *args, max_retries: int = 3, **kwargs):
        """Execute Redis operation with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                async with self.get_connection() as conn:
                    if self.circuit_breaker:
                        return await self.circuit_breaker.call(operation, conn, *args, **kwargs)
                    else:
                        # Execute directly without circuit breaker
                        if asyncio.iscoroutinefunction(operation):
                            return await operation(conn, *args, **kwargs)
                        else:
                            return operation(conn, *args, **kwargs)
            except (RedisCircuitBreakerError, RedisConnectionError) as e:
                # Log connection pool stats on connection errors
                if "too many connections" in str(e).lower() or isinstance(e, RedisConnectionError):
                    await self._log_connection_diagnostics()
                
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                # Handle specific Redis errors more gracefully
                error_msg = str(e).lower()
                
                # Don't log "no such key" as an error - it's expected behavior
                if "no such key" in error_msg:
                    return None
                
                # Handle timeout errors with more context
                if "timeout" in error_msg:
                    self.logger.warning(f"Redis timeout on attempt {attempt + 1}/{max_retries + 1}")
                
                # Log diagnostics for connection-related errors
                if "connection" in error_msg or "too many" in error_msg:
                    await self._log_connection_diagnostics()
                
                if attempt == max_retries:
                    # Only log as error if it's not an expected condition
                    if "no such key" not in error_msg:
                        self.logger.error(f"Redis operation failed after {max_retries} retries: {e}")
                    raise
                    
                await asyncio.sleep(2 ** attempt)
    
    async def _log_connection_diagnostics(self):
        """Log detailed connection diagnostics for debugging."""
        try:
            diagnostics = {
                'connected': self._connected,
                'reconnect_attempts': self._reconnect_attempts
            }
            
            # Add circuit breaker info if enabled
            if self.circuit_breaker:
                diagnostics.update({
                    'circuit_breaker_state': self.circuit_breaker.state,
                    'circuit_breaker_failures': self.circuit_breaker.failure_count,
                })
            else:
                diagnostics['circuit_breaker'] = 'disabled'
            
            # Get connection pool stats if using pool
            if self.use_connection_pool and self.connection_pool:
                pool_stats = self.connection_pool.get_stats()
                diagnostics['pool_stats'] = {
                    'active_connections': pool_stats.get('active_connections', 0),
                    'connection_limit': pool_stats.get('connection_limit', 0),
                    'total_connections_created': pool_stats.get('total_connections_created', 0),
                    'failed_connections': pool_stats.get('failed_connections', 0),
                    'pubsub_connections': pool_stats.get('pubsub_connections', 0)
                }
                
                # Warn if approaching connection limit
                if pool_stats['active_connections'] >= pool_stats['connection_limit'] * 0.8:
                    self.logger.warning(
                        f"Connection pool near limit: {pool_stats['active_connections']}/{pool_stats['connection_limit']} "
                        f"(PubSub: {pool_stats['pubsub_connections']})"
                    )
            
            # Try to get Redis server info
            try:
                if self.redis and self._connected:
                    info = await self.redis.info('clients')
                    diagnostics['redis_server'] = {
                        'connected_clients': info.get('connected_clients', 'unknown'),
                        'client_recent_max_input_buffer': info.get('client_recent_max_input_buffer', 'unknown'),
                        'client_recent_max_output_buffer': info.get('client_recent_max_output_buffer', 'unknown')
                    }
            except Exception as e:
                diagnostics['redis_server'] = f"Unable to get server info: {e}"
            
            self.logger.warning(f"Connection diagnostics: {json.dumps(diagnostics, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error collecting connection diagnostics: {e}")
    
    # Basic Redis operations with error handling
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            async def _get(conn, k):
                return await conn.get(k)
            result = await self.execute_with_retry(_get, key)
            return result.decode('utf-8') if result and isinstance(result, bytes) else result
        except Exception as e:
            # Silently return None for "no such key" errors
            if "no such key" in str(e).lower():
                return None
            raise
    
    async def set(self, key: str, value: Any, ex: Optional[int] = None, nx: bool = False) -> bool:
        """Set key-value pair."""
        async def _set(conn, k, v, **kwargs):
            return await conn.set(k, v, **kwargs)
        return await self.execute_with_retry(_set, key, value, ex=ex, nx=nx)
    
    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        """Set key with expiration."""
        async def _setex(conn, k, s, v):
            return await conn.setex(k, s, v)
        return await self.execute_with_retry(_setex, key, seconds, value)
    
    async def delete(self, *keys: str) -> int:
        """Delete keys."""
        async def _delete(conn, *k):
            return await conn.delete(*k)
        return await self.execute_with_retry(_delete, *keys)
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        async def _exists(conn, *k):
            return await conn.exists(*k)
        return await self.execute_with_retry(_exists, *keys)
    
    async def expire(self, key: str, time: int) -> bool:
        """Set key expiration."""
        async def _expire(conn, k, t):
            return await conn.expire(k, t)
        return await self.execute_with_retry(_expire, key, time)
    
    async def ttl(self, key: str) -> int:
        """Get key time to live."""
        async def _ttl(conn, k):
            return await conn.ttl(k)
        return await self.execute_with_retry(_ttl, key)
    
    async def incrby(self, key: str, amount: int = 1) -> int:
        """Increment the value of a key by a given amount."""
        async def _incrby(conn, k, a):
            return await conn.incrby(k, a)
        return await self.execute_with_retry(_incrby, key, amount)
    
    async def incr(self, key: str) -> int:
        """Increment the value of a key by 1."""
        async def _incr(conn, k):
            return await conn.incr(k)
        return await self.execute_with_retry(_incr, key)
    
    async def decr(self, key: str) -> int:
        """Decrement the value of a key by 1."""
        async def _decr(conn, k):
            return await conn.decr(k)
        return await self.execute_with_retry(_decr, key)
    
    async def decrby(self, key: str, amount: int = 1) -> int:
        """Decrement the value of a key by a given amount."""
        async def _decrby(conn, k, a):
            return await conn.decrby(k, a)
        return await self.execute_with_retry(_decrby, key, amount)
    
    # Hash operations
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value."""
        async def _hget(conn, n, k):
            return await conn.hget(n, k)
        return await self.execute_with_retry(_hget, name, key)
    
    async def hset(self, name: str, key: str = None, value: Any = None, mapping: Dict[str, Any] = None) -> int:
        """Set hash field value."""
        async def _hset(conn, n, **kwargs):
            return await conn.hset(n, **kwargs)
        return await self.execute_with_retry(_hset, name, key=key, value=value, mapping=mapping)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        async def _hgetall(conn, n):
            return await conn.hgetall(n)
        return await self.execute_with_retry(_hgetall, name)
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        async def _hdel(conn, n, *k):
            return await conn.hdel(n, *k)
        return await self.execute_with_retry(_hdel, name, *keys)
    
    # List operations
    async def lpush(self, name: str, *values: Any) -> int:
        """Push values to list head."""
        async def _lpush(conn, n, *v):
            return await conn.lpush(n, *v)
        return await self.execute_with_retry(_lpush, name, *values)
    
    async def rpush(self, name: str, *values: Any) -> int:
        """Push values to list tail."""
        async def _rpush(conn, n, *v):
            return await conn.rpush(n, *v)
        return await self.execute_with_retry(_rpush, name, *values)
    
    async def lpop(self, name: str, count: int = None) -> Union[str, List[str], None]:
        """Pop value from list head."""
        async def _lpop(conn, n, c):
            return await conn.lpop(n, c)
        return await self.execute_with_retry(_lpop, name, count)
    
    async def rpop(self, name: str, count: int = None) -> Union[str, List[str], None]:
        """Pop value from list tail."""
        async def _rpop(conn, n, c):
            return await conn.rpop(n, c)
        return await self.execute_with_retry(_rpop, name, count)
    
    async def llen(self, name: str) -> int:
        """Get list length."""
        async def _llen(conn, n):
            return await conn.llen(n)
        return await self.execute_with_retry(_llen, name)
    
    async def lrange(self, name: str, start: int, end: int) -> List[str]:
        """Get list range."""
        async def _lrange(conn, n, s, e):
            return await conn.lrange(n, s, e)
        return await self.execute_with_retry(_lrange, name, start, end)
    
    # Set operations
    async def sadd(self, name: str, *values: Any) -> int:
        """Add members to set."""
        async def _sadd(conn, n, *v):
            return await conn.sadd(n, *v)
        return await self.execute_with_retry(_sadd, name, *values)
    
    async def srem(self, name: str, *values: Any) -> int:
        """Remove members from set."""
        async def _srem(conn, n, *v):
            return await conn.srem(n, *v)
        return await self.execute_with_retry(_srem, name, *values)
    
    async def smembers(self, name: str) -> set:
        """Get all set members."""
        async def _smembers(conn, n):
            return await conn.smembers(n)
        return await self.execute_with_retry(_smembers, name)
    
    async def scard(self, name: str) -> int:
        """Get set cardinality."""
        async def _scard(conn, n):
            return await conn.scard(n)
        return await self.execute_with_retry(_scard, name)
    
    async def sismember(self, name: str, value: Any) -> bool:
        """Check if value is member of set."""
        async def _sismember(conn, n, v):
            return await conn.sismember(n, v)
        return await self.execute_with_retry(_sismember, name, value)
    
    # Sorted set operations
    async def zadd(self, name: str, mapping: Dict[Any, float], nx: bool = False, xx: bool = False) -> int:
        """Add members to sorted set."""
        async def _zadd(conn, n, m, **kwargs):
            return await conn.zadd(n, m, **kwargs)
        return await self.execute_with_retry(_zadd, name, mapping, nx=nx, xx=xx)
    
    async def zrem(self, name: str, *values: Any) -> int:
        """Remove members from sorted set."""
        async def _zrem(conn, n, *v):
            return await conn.zrem(n, *v)
        return await self.execute_with_retry(_zrem, name, *values)
    
    async def zrange(self, name: str, start: int, end: int, withscores: bool = False) -> List:
        """Get sorted set range."""
        async def _zrange(conn, n, s, e, w):
            return await conn.zrange(n, s, e, withscores=w)
        return await self.execute_with_retry(_zrange, name, start, end, withscores)
    
    async def zcard(self, name: str) -> int:
        """Get sorted set cardinality."""
        async def _zcard(conn, n):
            return await conn.zcard(n)
        return await self.execute_with_retry(_zcard, name)
    
    async def zrangebyscore(self, name: str, min: Union[int, float, str], max: Union[int, float, str], 
                           start: int = None, num: int = None, withscores: bool = False) -> List:
        """Get sorted set range by score."""
        async def _zrangebyscore(conn, n, mn, mx, s, nm, w):
            return await conn.zrangebyscore(n, mn, mx, start=s, num=nm, withscores=w)
        return await self.execute_with_retry(
            _zrangebyscore,
            name, min, max, start, num, withscores
        )
    
    async def zremrangebyrank(self, name: str, min: int, max: int) -> int:
        """Remove members from sorted set by rank range."""
        async def _zremrangebyrank(conn, n, mn, mx):
            return await conn.zremrangebyrank(n, mn, mx)
        return await self.execute_with_retry(_zremrangebyrank, name, min, max)
    
    # Pipeline operations
    @asynccontextmanager
    async def pipeline(self):
        """Create Redis pipeline as async context manager."""
        await self.ensure_connected()
        pipe = self.redis.pipeline()
        try:
            yield pipe
        finally:
            # Pipeline cleanup is handled by the pipeline itself
            pass
    
    # Pub/Sub operations
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        async def _publish(conn, c, m):
            return await conn.publish(c, m)
        return await self.execute_with_retry(_publish, channel, message)
    
    async def pubsub(self):
        """Create pub/sub instance."""
        await self.ensure_connected()
        
        # If using connection pool, get a dedicated pubsub connection
        if self.use_connection_pool and self.connection_pool:
            # Get the shared pubsub connection from the pool
            return await self.connection_pool.get_pubsub()
        
        return self.redis.pubsub()
    
    # Stream operations (Redis 5.0+)
    async def xadd(self, name: str, fields: Dict[str, Any], id: str = '*', maxlen: int = None) -> str:
        """Add entry to stream."""
        async def _xadd(conn, n, f, **kwargs):
            return await conn.xadd(n, f, **kwargs)
        return await self.execute_with_retry(_xadd, name, fields, id=id, maxlen=maxlen)
    
    async def xreadgroup(self, group: str, consumer: str, streams: Dict[str, str], count: int = None, block: int = None) -> List:
        """Read from stream consumer group."""
        async def _xreadgroup(conn, g, c, s, **kwargs):
            return await conn.xreadgroup(g, c, s, **kwargs)
        return await self.execute_with_retry(_xreadgroup, group, consumer, streams, count=count, block=block)
    
    async def xack(self, name: str, group: str, *ids: str) -> int:
        """Acknowledge messages in a consumer group."""
        async def _xack(conn, n, g, *i):
            return await conn.xack(n, g, *i)
        return await self.execute_with_retry(_xack, name, group, *ids)
    
    async def xpending_range(self, name: str, group: str, start: str = '-', end: str = '+', 
                           count: int = None, consumer: str = None) -> List[Dict[str, Any]]:
        """Get detailed information about pending messages."""
        # Convert response to list of dicts for consistency
        async def _xpending_range(conn, n, g, s, e, c, cons):
            if cons:
                result = await conn.xpending_range(n, g, min=s, max=e, count=c, consumername=cons)
            else:
                result = await conn.xpending_range(n, g, min=s, max=e, count=c)
            # Convert to consistent format
            pending_list = []
            for item in result:
                pending_list.append({
                    'message_id': item['message_id'],
                    'consumer': item['consumer'],
                    'time_since_delivered': item['time_since_delivered'],
                    'times_delivered': item['times_delivered']
                })
            return pending_list
        
        return await self.execute_with_retry(
            _xpending_range, name, group, start, end, count or 10, consumer
        )
    
    async def xclaim(self, name: str, group: str, consumer: str, min_idle_time: int, 
                    ids: List[str], **kwargs) -> List[Tuple[bytes, Dict[bytes, bytes]]]:
        """Claim ownership of pending messages."""
        async def _xclaim(conn, n, g, c, m, i, **kw):
            return await conn.xclaim(n, g, c, m, i, **kw)
        return await self.execute_with_retry(
            _xclaim,
            name, group, consumer, min_idle_time, ids, **kwargs
        )
    
    async def xgroup_create(self, name: str, group: str, id: str = '0', mkstream: bool = False) -> bool:
        """Create consumer group."""
        try:
            async with self.get_connection() as conn:
                async def _xgroup_create(c, n, g, i, m):
                    return await c.xgroup_create(n, g, i, mkstream=m)
                return await self.circuit_breaker.call(
                    _xgroup_create,
                    conn, name, group, id, mkstream
                )
        except Exception as e:
            # BUSYGROUP error is expected if group already exists
            if "BUSYGROUP" in str(e):
                self.logger.debug(f"Consumer group {group} already exists for stream {name}")
                return False
            raise
    
    # Lua script operations
    async def eval(self, script: str, numkeys: int, *keys_and_args) -> Any:
        """Execute Lua script."""
        async def _eval(conn, s, n, *ka):
            return await conn.eval(s, n, *ka)
        return await self.execute_with_retry(_eval, script, numkeys, *keys_and_args)
    
    async def evalsha(self, sha: str, numkeys: int, *keys_and_args) -> Any:
        """Execute Lua script by SHA."""
        async def _evalsha(conn, s, n, *ka):
            return await conn.evalsha(s, n, *ka)
        return await self.execute_with_retry(_evalsha, sha, numkeys, *keys_and_args)
    
    async def script_load(self, script: str) -> str:
        """Load Lua script and return SHA."""
        async def _script_load(conn, s):
            return await conn.script_load(s)
        return await self.execute_with_retry(_script_load, script)
    
    async def scan(self, cursor: int = 0, match: str = None, count: int = None) -> tuple:
        """Scan keys."""
        async def _scan(conn, c, m, ct):
            return await conn.scan(c, match=m, count=ct)
        return await self.execute_with_retry(_scan, cursor, match, count)
    
    async def script_flush(self) -> bool:
        """Flush all Lua scripts."""
        async def _script_flush(conn):
            return await conn.script_flush()
        return await self.execute_with_retry(_script_flush)
    
    async def zremrangebyscore(self, name: str, min_score: Union[int, float], max_score: Union[int, float]) -> int:
        """Remove members from sorted set by score range."""
        async def _zremrangebyscore(conn, n, min_s, max_s):
            return await conn.zremrangebyscore(n, min_s, max_s)
        return await self.execute_with_retry(_zremrangebyscore, name, min_score, max_score)
    
    async def ltrim(self, name: str, start: int, end: int) -> bool:
        """Trim list to specified range."""
        async def _ltrim(conn, n, s, e):
            return await conn.ltrim(n, s, e)
        return await self.execute_with_retry(_ltrim, name, start, end)
    
    # Info and monitoring
    async def info(self, section: str = None) -> Dict[str, Any]:
        """Get Redis server info."""
        async def _info(conn, s):
            return await conn.info(s)
        return await self.execute_with_retry(_info, section)
    
    async def ping(self) -> bool:
        """Ping Redis server."""
        async def _ping(conn):
            return await conn.ping()
        return await self.execute_with_retry(_ping)
    
    async def get_memory_usage(self, key: str) -> int:
        """Get memory usage of key."""
        async def _memory_usage(conn, k):
            return await conn.memory_usage(k)
        return await self.execute_with_retry(_memory_usage, key)
    
    async def xinfo_stream(self, name: str) -> Dict[str, Any]:
        """Get information about a stream."""
        async def _xinfo_stream(conn, n):
            return await conn.xinfo_stream(n)
        return await self.execute_with_retry(_xinfo_stream, name)
    
    async def xrevrange(self, name: str, max: str = '+', min: str = '-', count: int = None) -> List:
        """Read stream entries in reverse order."""
        async def _xrevrange(conn, n, mx, mn, c):
            return await conn.xrevrange(n, mx, mn, count=c)
        return await self.execute_with_retry(_xrevrange, name, max, min, count)
    
    async def xtrim(self, name: str, maxlen: int = None, minid: str = None, approximate: bool = True) -> int:
        """Trim stream to a maximum length or minimum ID."""
        if maxlen is not None:
            async def _xtrim_maxlen(conn, n, ml, a):
                return await conn.xtrim(n, maxlen=ml, approximate=a)
            return await self.execute_with_retry(
                _xtrim_maxlen, 
                name, maxlen, approximate
            )
        elif minid is not None:
            async def _xtrim_minid(conn, n, mi):
                return await conn.xtrim(n, minid=mi)
            return await self.execute_with_retry(
                _xtrim_minid, 
                name, minid
            )
        else:
            raise ValueError("Either maxlen or minid must be specified")
    
    async def xpending(self, name: str, group: str, start: str = None, end: str = None, count: int = None, consumer: str = None) -> Any:
        """Get information about pending messages in a consumer group."""
        # If no range specified, get summary
        if start is None and end is None and count is None:
            async def _xpending(conn, n, g):
                return await conn.xpending(n, g)
            return await self.execute_with_retry(_xpending, name, group)
        
        # Get detailed pending messages
        if consumer:
            async def _xpending_range_with_consumer(conn, n, g, s, e, c, cons):
                return await conn.xpending_range(n, g, s, e, c, consumername=cons)
            return await self.execute_with_retry(
                _xpending_range_with_consumer, 
                name, group, start or '-', end or '+', count or 10, consumer
            )
        else:
            async def _xpending_range(conn, n, g, s, e, c):
                return await conn.xpending_range(n, g, s, e, c)
            return await self.execute_with_retry(
                _xpending_range, 
                name, group, start or '-', end or '+', count or 10
            )
    
    # Statistics and monitoring
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        health_stats = self.health_monitor.get_health_stats() if self.health_monitor else {}
        
        stats = {
            'connection_id': self.connection_id,
            'connected': self._connected,
            'use_connection_pool': self.use_connection_pool,
            'reconnect_attempts': self._reconnect_attempts,
            'config': {
                'mode': self.config.mode.value,
                'environment': self.config.environment.value,
                'host': self.config.connection_config.host,
                'port': self.config.connection_config.port,
                'max_connections': self.config.connection_config.max_connections
            },
            'health': health_stats
        }
        
        # Add circuit breaker stats if enabled
        if self.circuit_breaker:
            stats['circuit_breaker'] = {
                'enabled': True,
                'state': self.circuit_breaker.state,
                'failure_count': self.circuit_breaker.failure_count,
                'failure_threshold': self.circuit_breaker.failure_threshold,
                'last_failure_time': self.circuit_breaker.last_failure_time
            }
        else:
            stats['circuit_breaker'] = {'enabled': False}
        
        # Add connection pool stats if using pool
        if self.use_connection_pool and self.connection_pool:
            stats['connection_pool'] = self.connection_pool.get_stats()
        
        return stats
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Global Redis client instance
_redis_client = None
_client_lock = asyncio.Lock()


async def get_redis_client(config: RedisConfig = None, use_connection_pool: bool = True) -> RedisClient:
    """Get global Redis client instance.
    
    Args:
        config: Redis configuration
        use_connection_pool: Whether to use singleton connection pool
        
    Returns:
        Redis client instance
    """
    global _redis_client
    
    async with _client_lock:
        if _redis_client is None:
            _redis_client = RedisClient(config, use_connection_pool=use_connection_pool)
            await _redis_client.connect()
        return _redis_client


async def close_redis_client():
    """Close global Redis client instance."""
    global _redis_client
    
    async with _client_lock:
        if _redis_client:
            await _redis_client.disconnect()
            _redis_client = None
        
        # Also close the connection pool if it exists
        from scripts.redis_integration.redis_connection_pool import close_connection_pool
        await close_connection_pool()