# Redis Resilience Solution

## Overview

This solution addresses Redis connection problems by implementing a multi-layered resilience strategy:

1. **Automatic Health Checks & Recovery** - Continuous monitoring with automatic reconnection
2. **SQLite Fallback** - Local persistence when Redis is unavailable
3. **Circuit Breaker Pattern** - Prevents cascading failures
4. **Enhanced Connection Pooling** - Proper resource management

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                          │
│                  (Your code using Redis operations)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  Redis Connection Manager                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Health Monitor   │  │Circuit Breaker│  │ Recovery Logic   │   │
│  │ (5s intervals)  │  │(5 failures)   │  │ (Exponential)    │   │
│  └─────────────────┘  └──────────────┘  └──────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
┌───────▼──────────┐                    ┌────────▼─────────┐
│  Redis Client    │                    │ SQLite Fallback  │
│ (Primary Store)  │                    │ (Local Backup)   │
└──────────────────┘                    └──────────────────┘
```

## Key Components

### 1. RedisConnectionManager (`redis_connection_manager.py`)

The central component that manages all Redis connections with automatic failover:

```python
# Usage
from scripts.redis_integration.redis_connection_manager import get_connection_manager

# Get the connection manager (singleton)
manager = await get_connection_manager(enable_fallback=True)

# Use it like a regular Redis client
await manager.set("key", "value")
value = await manager.get("key")
```

**Features:**
- Automatic health monitoring every 5 seconds
- Circuit breaker opens after 5 consecutive failures
- Automatic recovery with exponential backoff
- Seamless fallback to SQLite when Redis is down
- Connection state tracking (HEALTHY, DEGRADED, FAILED, RECOVERING)

### 2. SQLiteFallback

Local SQLite database that mirrors Redis operations:

**Supported Operations:**
- Key-Value: get, set, delete, exists, expire
- Hashes: hget, hset, hgetall
- Sets: sadd, smembers
- Sorted Sets: zadd, zrange
- Counters: hincrby

**Limitations:**
- No pub/sub support
- No Lua scripting
- Simplified transaction support
- Local to each instance (not distributed)

### 3. ResilientRedisTaskPersistence (`redis_resilient_task_persistence.py`)

Enhanced task persistence with three-tier fallback:

```python
# Usage
from scripts.redis_integration.redis_resilient_task_persistence import ResilientRedisTaskPersistence

persistence = ResilientRedisTaskPersistence(enable_file_fallback=True)

# Record completed task
await persistence.record_completed_task(work_item, execution_result)

# Check for duplicates
is_duplicate = await persistence.is_duplicate_task(work_item)
```

**Fallback Hierarchy:**
1. Redis (when healthy)
2. SQLite (when Redis is down)
3. File-based storage (last resort)

## Connection States

### HEALTHY
- Redis is connected and responsive
- All operations use Redis directly
- Health checks passing

### DEGRADED
- Redis experiencing intermittent failures
- Circuit breaker may be half-open
- System attempting to use Redis but ready to fallback

### FAILED
- Redis is completely unavailable
- All operations use SQLite fallback
- System not attempting Redis operations

### RECOVERING
- System detected Redis might be available
- Actively attempting to reconnect
- Operations still use fallback during recovery

## Configuration

### Environment Variables

```bash
# Redis connection settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password  # Optional
REDIS_DB=0

# Connection pool settings
REDIS_MAX_CONNECTIONS=5000
REDIS_MIN_IDLE_TIME=0
REDIS_CONNECTION_TIMEOUT=20
REDIS_SOCKET_TIMEOUT=20

# Health check settings
REDIS_HEALTH_CHECK_INTERVAL=30  # seconds
REDIS_CIRCUIT_BREAKER_ENABLED=true
REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
REDIS_CIRCUIT_BREAKER_TIMEOUT=60  # seconds
```

### Tuning Parameters

1. **Health Check Interval**: Default 5s, increase for less aggressive monitoring
2. **Circuit Breaker Threshold**: Default 5 failures, decrease for faster failover
3. **Recovery Attempts**: Default 5, increase for more persistent recovery
4. **Connection Pool Size**: Default 5000, adjust based on load

## Monitoring

### Real-time Monitor

```bash
python monitor_redis_resilience.py
```

Displays:
- Current connection state
- Operation counts and rates
- Health statistics
- Circuit breaker status
- Persistence metrics

### Testing

```bash
# Run comprehensive tests
python test_redis_resilience.py

# Test includes:
# - Basic operations
# - Task persistence
# - Failure simulation
# - Recovery testing
# - Continuous operations
```

## Best Practices

### 1. Always Use the Connection Manager

```python
# Good
manager = await get_connection_manager()
await manager.set("key", "value")

# Bad - Direct Redis client
client = RedisClient()
await client.set("key", "value")  # No fallback!
```

### 2. Handle State Changes Gracefully

```python
status = manager.get_status()
if status['state'] == 'failed':
    logger.warning("Running on SQLite fallback")
```

### 3. Monitor Metrics

```python
metrics = manager.get_status()['metrics']
if metrics['fallback_operations'] > metrics['redis_operations']:
    # Alert: Mostly running on fallback!
    send_alert("Redis unhealthy - high fallback usage")
```

### 4. Clean Up Resources

```python
from scripts.redis_integration.redis_connection_manager import close_connection_manager

# Always clean up when shutting down
await close_connection_manager()
```

## Troubleshooting

### Redis Won't Connect

1. Check Redis is running: `redis-cli ping`
2. Verify connection settings in `.env.local`
3. Check firewall/network settings
4. Review logs for specific errors

### High Fallback Usage

1. Check Redis server health: `redis-cli info`
2. Monitor connection pool: May be exhausted
3. Check for connection leaks in code
4. Increase circuit breaker threshold if too sensitive

### SQLite Performance Issues

1. Ensure adequate disk space
2. Consider periodic cleanup of old data
3. Monitor SQLite file size
4. Use file-based fallback sparingly

### Recovery Not Working

1. Check max recovery attempts setting
2. Verify Redis is actually accessible
3. Check for persistent network issues
4. Review circuit breaker timeout

## Migration Guide

### From Direct Redis Client

```python
# Old code
redis_client = RedisClient()
await redis_client.connect()
await redis_client.set("key", "value")

# New code
manager = await get_connection_manager()
await manager.set("key", "value")
```

### From RedisTaskPersistence

```python
# Old code
from scripts.redis_task_persistence import RedisTaskPersistence
persistence = RedisTaskPersistence()

# New code
from scripts.redis_integration.redis_resilient_task_persistence import ResilientRedisTaskPersistence
persistence = ResilientRedisTaskPersistence()
```

## Performance Considerations

1. **SQLite Fallback**: ~10-50ms per operation (vs ~1-5ms for Redis)
2. **Health Checks**: Minimal overhead (<1% CPU)
3. **Circuit Breaker**: Prevents unnecessary connection attempts
4. **Connection Pooling**: Reuses connections efficiently

## Future Enhancements

1. **Redis Sentinel Support**: For automatic master failover
2. **Cluster Mode**: For horizontal scaling
3. **Metrics Export**: Prometheus/Grafana integration
4. **Advanced Fallback**: Support for more Redis operations
5. **Distributed SQLite**: Using Litestream for replication