#!/usr/bin/env python3
"""
Redis Connection Diagnostics Tool

Helps diagnose Redis connection issues including circuit breaker state,
connection pool usage, and server connection limits.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.redis_integration.redis_client import RedisClient
from scripts.redis_integration.redis_connection_pool import get_connection_pool

async def diagnose_connections():
    """Run comprehensive Redis connection diagnostics."""
    print("🔍 Redis Connection Diagnostics")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Create Redis client
    redis_client = RedisClient()
    
    try:
        # 1. Test basic connection
        print("1️⃣ Testing Basic Connection...")
        await redis_client.connect()
        print("✅ Connected successfully")
        
        # 2. Get connection stats
        print("\n2️⃣ Connection Statistics:")
        stats = redis_client.get_connection_stats()
        print(json.dumps(stats, indent=2))
        
        # 3. Check circuit breaker
        print("\n3️⃣ Circuit Breaker Status:")
        cb_stats = stats.get('circuit_breaker', {})
        if cb_stats.get('enabled', True):
            print(f"  State: {cb_stats.get('state', 'unknown')}")
            print(f"  Failures: {cb_stats.get('failure_count', 0)}/{cb_stats.get('failure_threshold', 5)}")
            if cb_stats.get('state') == 'open':
                print("  ⚠️  WARNING: Circuit breaker is OPEN!")
        else:
            print("  Circuit breaker is DISABLED")
        
        # 4. Check connection pool
        print("\n4️⃣ Connection Pool Status:")
        if redis_client.use_connection_pool and redis_client.connection_pool:
            pool_stats = redis_client.connection_pool.get_stats()
            print(f"  Active connections: {pool_stats.get('active_connections', 0)}")
            print(f"  Connection limit: {pool_stats.get('connection_limit', 0)}")
            print(f"  Total created: {pool_stats.get('total_connections_created', 'N/A')}")
            print(f"  Failed connections: {pool_stats.get('failed_connections', 'N/A')}")
            print(f"  PubSub connections: {pool_stats.get('pubsub_connections', 0)}")
            
            # Check if near limit
            active = pool_stats.get('active_connections', 0)
            limit = pool_stats.get('connection_limit', 1)
            usage_pct = (active / limit) * 100 if limit > 0 else 0
            print(f"  Usage: {usage_pct:.1f}%")
            
            if usage_pct >= 80:
                print("  ⚠️  WARNING: Connection pool usage is HIGH!")
            elif usage_pct >= 50:
                print("  ⚡ Connection pool usage is moderate")
            else:
                print("  ✅ Connection pool usage is healthy")
        else:
            print("  Not using connection pool")
        
        # 5. Check Redis server
        print("\n5️⃣ Redis Server Status:")
        try:
            # Get server info
            info = await redis_client.redis.info('clients')
            print(f"  Connected clients: {info.get('connected_clients', 'unknown')}")
            print(f"  Max clients: {info.get('maxclients', 'unknown')}")
            
            # Get server memory info
            mem_info = await redis_client.redis.info('memory')
            used_memory = mem_info.get('used_memory_human', 'unknown')
            print(f"  Memory usage: {used_memory}")
            
            # Check server limits
            config = await redis_client.redis.config_get('maxclients')
            max_clients = config.get('maxclients', 'unknown')
            print(f"  Server max clients config: {max_clients}")
            
        except Exception as e:
            print(f"  ❌ Could not get server info: {e}")
        
        # 6. Test operations
        print("\n6️⃣ Testing Redis Operations:")
        
        # Test SET/GET
        try:
            await redis_client.set("diag:test:key", "test_value")
            value = await redis_client.get("diag:test:key")
            print(f"  SET/GET: {'✅ Pass' if value == 'test_value' else '❌ Fail'}")
            await redis_client.delete("diag:test:key")
        except Exception as e:
            print(f"  SET/GET: ❌ Failed - {e}")
        
        # Test streams
        try:
            stream_id = await redis_client.xadd("diag:test:stream", {"test": "message"})
            print(f"  XADD: {'✅ Pass' if stream_id else '❌ Fail'}")
            await redis_client.delete("diag:test:stream")
        except Exception as e:
            print(f"  XADD: ❌ Failed - {e}")
        
        print("\n7️⃣ Recommendations:")
        
        # Provide recommendations based on diagnostics
        recommendations = []
        
        if cb_stats.get('state') == 'open':
            recommendations.append("🔧 Reset circuit breaker with: python reset_redis_circuit_breaker.py")
        
        if redis_client.use_connection_pool and redis_client.connection_pool:
            pool_stats = redis_client.connection_pool.get_stats()
            active = pool_stats.get('active_connections', 0)
            limit = pool_stats.get('connection_limit', 1)
            usage_pct = (active / limit) * 100 if limit > 0 else 0
            if usage_pct >= 80:
                recommendations.append(f"🔧 Increase connection pool limit (current: {limit})")
                recommendations.append("   Set REDIS_MAX_CONNECTIONS=500 in environment")
            
            if pool_stats.get('pubsub_connections', 0) > 10:
                recommendations.append(f"⚠️  High PubSub connections ({pool_stats['pubsub_connections']}), check for leaks")
        
        if cb_stats.get('failure_count', 0) > 0:
            recommendations.append("📊 Check logs for specific errors causing failures")
        
        if recommendations:
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print("  ✅ No immediate issues detected")
        
    except Exception as e:
        print(f"\n❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await redis_client.disconnect()
        print("\n" + "=" * 60)
        print("Diagnostics complete")


async def reset_circuit_breaker():
    """Reset the circuit breaker to closed state."""
    print("🔧 Resetting Circuit Breaker...")
    
    redis_client = RedisClient()
    
    try:
        await redis_client.connect()
        
        if redis_client.circuit_breaker:
            old_state = redis_client.circuit_breaker.state
            old_failures = redis_client.circuit_breaker.failure_count
            
            # Reset circuit breaker
            redis_client.circuit_breaker.state = 'closed'
            redis_client.circuit_breaker.failure_count = 0
            redis_client.circuit_breaker.last_failure_time = 0
            
            print(f"✅ Circuit breaker reset:")
            print(f"   State: {old_state} → closed")
            print(f"   Failures: {old_failures} → 0")
            
            # Test connection
            await redis_client.ping()
            print("✅ Connection test successful")
        else:
            print("ℹ️  Circuit breaker is disabled")
            
    except Exception as e:
        print(f"❌ Reset failed: {e}")
    finally:
        await redis_client.disconnect()


async def cleanup_connections():
    """Clean up stale connections from the pool."""
    print("🧹 Cleaning up stale connections...")
    
    pool = await get_connection_pool()
    
    try:
        cleaned = await pool.cleanup_stale_connections()
        print(f"✅ Cleaned up {cleaned} stale connections")
        
        # Show current stats
        stats = pool.get_stats()
        print(f"\nCurrent pool status:")
        print(f"  Active connections: {stats['active_connections']}")
        print(f"  Connection limit: {stats['connection_limit']}")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'reset':
            await reset_circuit_breaker()
        elif command == 'cleanup':
            await cleanup_connections()
        elif command == 'diagnose':
            await diagnose_connections()
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python diagnose_redis_connections.py [command]")
            print("\nCommands:")
            print("  diagnose  - Run full diagnostics (default)")
            print("  reset     - Reset circuit breaker")
            print("  cleanup   - Clean up stale connections")
    else:
        # Default to diagnose
        await diagnose_connections()


if __name__ == "__main__":
    asyncio.run(main())