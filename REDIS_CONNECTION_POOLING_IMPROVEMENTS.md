# Redis Connection Pooling Improvements

## Overview

This document describes the improvements made to the Redis client implementation to fix connection leak issues and implement proper connection pooling.

## Key Improvements

### 1. Singleton Connection Pool (`redis_connection_pool.py`)

- **Single Source of Truth**: All connections are managed through a singleton connection pool
- **Connection Limits**: Enforced connection limits with semaphore-based control
- **Weak References**: Pub/Sub connections use weak references for automatic cleanup
- **Connection Tracking**: Detailed tracking of active connections, pubsub connections, and reconnect attempts

### 2. Enhanced Redis Client (`redis_client.py`)

- **Pool Integration**: Option to use singleton connection pool (default) or dedicated connections
- **Smart Health Checks**: Reduced health check spam with consecutive failure tracking
- **Async Circuit Breaker**: Properly handles async operations in circuit breaker pattern
- **Exponential Backoff**: Configurable exponential backoff for reconnection attempts

### 3. Pub/Sub Management Improvements

- **Dedicated Connections**: Each pub/sub instance gets a unique connection ID
- **Proper Cleanup**: Pub/sub connections are properly closed when no longer needed
- **Resource Tracking**: Monitor active pub/sub connections to prevent leaks

## Usage Examples

### Basic Usage with Connection Pool (Default)

```python
from scripts.redis_integration import get_redis_client

# Uses singleton connection pool by default
client = await get_redis_client()
await client.set("key", "value")
value = await client.get("key")
```

### Dedicated Connection (No Pool)

```python
from scripts.redis_integration import RedisClient

# Create client with dedicated connection
client = RedisClient(use_connection_pool=False)
await client.connect()
await client.set("key", "value")
await client.disconnect()
```

### Monitoring Connections

```python
# Run connection monitor
python scripts/redis_connection_monitor.py

# Test for connection leaks
python scripts/redis_connection_monitor.py test
```

### Cleaning Up Connections

```python
# Clean up idle connections
python scripts/redis_connection_cleanup.py

# Reset Redis connection limits
python scripts/redis_connection_cleanup.py reset

# Force kill all connections
python scripts/redis_connection_cleanup.py force
```

## Configuration

### Environment Variables

- `REDIS_MAX_CONNECTIONS`: Maximum connections in the pool (default: 50)
- `REDIS_HEALTH_CHECK_INTERVAL`: Health check interval in seconds (default: 30)
- `REDIS_SOCKET_TIMEOUT`: Socket timeout in seconds (default: 30)

### Connection Pool Settings

```python
# In redis_connection_pool.py
self._connection_limit = self.config.connection_config.max_connections
self._max_reconnect_attempts = 5
self._reconnect_backoff_base = 2
self._reconnect_backoff_max = 60
```

## Benefits

1. **No More Connection Leaks**: Singleton pool ensures connections are reused
2. **Better Resource Management**: Enforced limits prevent connection exhaustion
3. **Improved Reliability**: Exponential backoff prevents reconnection storms
4. **Reduced Log Spam**: Smart health checks only log state changes
5. **Easy Monitoring**: Built-in tools to monitor and debug connections

## Testing

Run the comprehensive test suite:

```bash
python test_redis_connection_pool.py
```

The test suite verifies:
- Singleton pool behavior
- Connection reuse
- Pub/Sub management
- Connection limits
- Reconnection handling
- Health monitoring
- Stress testing

## Migration Guide

For existing code:

1. **No changes required** for basic usage - `get_redis_client()` now uses the pool by default
2. **Pub/Sub code** benefits automatically from improved connection management
3. **Health checks** are less noisy but still effective
4. **Connection stats** now include pool information

## Troubleshooting

### High Connection Count

If you see high connection counts:
1. Run `python scripts/redis_connection_monitor.py` to identify the source
2. Check for pub/sub subscriptions that aren't being cleaned up
3. Use `python scripts/redis_connection_cleanup.py` to clean up idle connections

### Circuit Breaker Open

If the circuit breaker is open:
1. Check Redis server health
2. Review network connectivity
3. Check for persistent errors in logs
4. Circuit breaker will reset after timeout period

### Connection Pool Exhausted

If connection pool is exhausted:
1. Increase `REDIS_MAX_CONNECTIONS` environment variable
2. Review code for connection leaks
3. Ensure proper cleanup in finally blocks
4. Monitor with connection monitor tool