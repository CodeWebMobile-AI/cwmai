# Redis "Event Loop is Closed" Fix

## Problem
The automatic Redis connection cleanup was too aggressive (60 seconds) and was killing connections still in use by async tasks, causing "Event loop is closed" errors when tasks tried to record completion.

## Root Cause
1. Cleanup script was killing connections after just 60 seconds of idle time
2. Task persistence connections were being closed while tasks were still running
3. When tasks completed and tried to record results, the event loop was already closed

## Solution Applied

### 1. Less Aggressive Cleanup (cleanup_redis_now.py)
- **Increased idle timeout from 60 seconds to 4 minutes (240 seconds)**
- Added checks to skip connections with:
  - Active commands (except 'client', 'ping')
  - Transaction flags (M flag)
  - Blocked state (b flag)
  - Named connection pools

### 2. Graceful Error Handling (redis_task_persistence.py)
- Added connection health checking before operations
- Automatic reconnection when connection is unhealthy
- Graceful handling of "event loop is closed" errors
- Task completion is not blocked by recording failures
- Warning logs instead of error logs for shutdown scenarios

### 3. Non-Propagating Errors
- Recording failures don't fail the task
- Tasks complete successfully even if Redis recording fails
- Clear warning messages indicate task completed but wasn't recorded

## Configuration Changes

### Environment Variables (.env.local)
```bash
REDIS_MAX_CONNECTIONS=100        # Was 5000
REDIS_SOCKET_TIMEOUT=5.0         # Was 30.0
REDIS_SOCKET_CONNECT_TIMEOUT=5.0 # Was 30.0
REDIS_CIRCUIT_BREAKER_THRESHOLD=5 # Was 50
```

### Cleanup Script Settings
```python
# Skip connections that are:
idle_time > 240  # 4 minutes (was 1 minute)
# AND not executing commands
# AND not in transaction
# AND not blocked
# AND not named pool connections
```

## Testing

Run the test script to verify the fix:
```bash
python3 test_redis_event_loop_fix.py
```

The test verifies:
1. Normal task recording works
2. Recording works after connection close (auto-reconnect)
3. Duplicate detection works after reconnection
4. Statistics remain accurate
5. Graceful handling during shutdown

## Monitoring

Check Redis connection health:
```bash
./redis_health.sh check
```

Watch connections live:
```bash
./redis_health.sh watch
```

## Best Practices

1. **Use the wrapper scripts** for all Redis operations
2. **Monitor regularly** to catch connection leaks early
3. **Let failed recordings fail gracefully** - task completion is more important
4. **Use connection pools** for long-lived connections
5. **Set reasonable timeouts** based on your workload

## If Issues Persist

1. Increase cleanup timeout further:
   ```python
   if idle_time > 600:  # 10 minutes
   ```

2. Add more exemptions in cleanup script:
   ```python
   # Skip connections from specific components
   if 'persistence' in client_name.lower():
       continue
   ```

3. Disable automatic cleanup temporarily:
   ```bash
   # Comment out in run_cwmai.sh
   # python cleanup_redis_now.py
   ```

4. Check for connection leaks:
   ```bash
   ./redis_health.sh monitor
   ```

---
Fixed: 2024-01-16