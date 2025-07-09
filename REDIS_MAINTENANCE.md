# Redis Maintenance Guide for CWMAI

## Overview
This guide helps you maintain healthy Redis connections and prevent connection leak issues.

## Automatic Setup (Recommended)

### 1. Startup Cleanup
Already configured! Redis connections are automatically cleaned when you run:
```bash
./run_cwmai.sh
```

### 2. Scheduled Cleanup
Set up automatic cleanup every 30 minutes:
```bash
./setup_redis_scheduler.sh
```

## Manual Monitoring & Maintenance

### Quick Health Check
```bash
./redis_health.sh check
# or just
./redis_health.sh
```

### Live Monitoring
Watch Redis connections in real-time:
```bash
./redis_health.sh watch
```

### Detailed Analysis
Run a 5-minute monitoring session:
```bash
./redis_health.sh monitor
```

### Manual Cleanup
Clean idle connections immediately:
```bash
./redis_health.sh cleanup
```

## Individual Scripts

### Debug Current State
```bash
python3 debug_redis_connections.py
```
Shows:
- Redis server status
- Current connections
- Idle connection count
- Connection test results

### Monitor Over Time
```bash
python3 monitor_redis_health.py
```
Runs for 5 minutes and provides:
- Success/failure rates
- Response time analysis
- Failure patterns
- Recommendations

### Force Cleanup
```bash
python3 cleanup_redis_now.py
```
Immediately kills all connections idle >60 seconds

## Configuration

Your Redis is now configured with:
- Max connections: 100 (was 5000)
- Socket timeout: 5s (was 30s)
- Circuit breaker threshold: 5 (was 50)

These settings are in `.env.local`

## Troubleshooting

### High Connection Count
If you see >50 connections:
1. Run cleanup: `./redis_health.sh cleanup`
2. Check for stuck processes: `ps aux | grep cwmai`
3. Restart Redis if needed: `redis-cli SHUTDOWN; redis-server`

### Connection Failures
If connections are failing:
1. Check Redis is running: `redis-cli ping`
2. Check server limits: `redis-cli CONFIG GET maxclients`
3. Run monitoring: `./redis_health.sh monitor`

### View Scheduler Logs
```bash
./redis_health.sh logs
# or
tail -f /tmp/cwmai-redis-cleanup.log
```

## Best Practices

1. **Always use context managers** for Redis connections
2. **Clean up PubSub** connections after use
3. **Monitor regularly** - at least weekly
4. **Set connection timeouts** appropriately
5. **Use connection pooling** (already configured)

## Emergency Reset

If Redis is completely stuck:
```bash
# Stop all Redis connections
redis-cli CLIENT KILL TYPE normal

# Or restart Redis completely
redis-cli SHUTDOWN
redis-server

# Then clean up
./redis_health.sh cleanup
```

## Metrics to Watch

- **Idle connections**: Should be <10
- **Total connections**: Should be <50
- **Response time**: Should be <10ms
- **Success rate**: Should be >99%

## Automated Alerts (Optional)

Add to your monitoring system:
```bash
# Check connection count
CONNECTIONS=$(redis-cli client list | wc -l)
if [ $CONNECTIONS -gt 80 ]; then
    echo "WARNING: Redis connections high: $CONNECTIONS"
fi
```

---
Updated: 2024-01-16