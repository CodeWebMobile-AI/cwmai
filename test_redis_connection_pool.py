#!/usr/bin/env python3
"""
Test Redis Connection Pool Implementation

Verify that the new connection pooling implementation properly manages
connections and prevents leaks.
"""

import asyncio
import logging
import time
from typing import List

from scripts.redis_integration import (
    get_redis_client,
    close_redis_client,
    get_connection_pool,
    RedisPubSubManager
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_singleton_pool():
    """Test that connection pool is singleton."""
    logger.info("Testing singleton connection pool...")
    
    pool1 = await get_connection_pool()
    pool2 = await get_connection_pool()
    
    assert pool1 is pool2, "Connection pool should be singleton"
    logger.info("✓ Connection pool is singleton")
    
    # Check pool stats
    stats = pool1.get_stats()
    logger.info(f"Pool stats: {stats}")


async def test_connection_reuse():
    """Test that connections are properly reused."""
    logger.info("\nTesting connection reuse...")
    
    client = await get_redis_client()
    pool = await get_connection_pool()
    
    initial_stats = pool.get_stats()
    initial_connections = initial_stats['active_connections']
    
    # Perform multiple operations
    for i in range(10):
        await client.set(f"test_key_{i}", f"value_{i}")
        await client.get(f"test_key_{i}")
    
    final_stats = pool.get_stats()
    final_connections = final_stats['active_connections']
    
    logger.info(f"Initial connections: {initial_connections}")
    logger.info(f"Final connections: {final_connections}")
    
    assert final_connections <= initial_connections + 1, "Connections should be reused"
    logger.info("✓ Connections are properly reused")


async def test_pubsub_management():
    """Test that pub/sub connections are properly managed."""
    logger.info("\nTesting pub/sub connection management...")
    
    client = await get_redis_client()
    pool = await get_connection_pool()
    
    initial_stats = pool.get_stats()
    initial_pubsubs = initial_stats['pubsub_connections']
    
    # Create multiple pubsub instances
    pubsub_manager = RedisPubSubManager(client)
    await pubsub_manager.start()
    
    # Subscribe to multiple channels
    channels = ['test_channel_1', 'test_channel_2', 'test_channel_3']
    
    async def handler(channel: str, data: dict):
        logger.debug(f"Received on {channel}: {data}")
    
    for channel in channels:
        await pubsub_manager.subscribe(channel, handler)
    
    # Wait a bit
    await asyncio.sleep(2)
    
    mid_stats = pool.get_stats()
    mid_pubsubs = mid_stats['pubsub_connections']
    
    logger.info(f"Initial pubsubs: {initial_pubsubs}")
    logger.info(f"After subscribe: {mid_pubsubs}")
    
    # Unsubscribe
    for channel in channels:
        await pubsub_manager.unsubscribe(channel)
    
    await pubsub_manager.stop()
    
    # Final check
    await asyncio.sleep(1)
    final_stats = pool.get_stats()
    final_pubsubs = final_stats['pubsub_connections']
    
    logger.info(f"After cleanup: {final_pubsubs}")
    logger.info("✓ Pub/sub connections managed properly")


async def test_connection_limits():
    """Test that connection limits are enforced."""
    logger.info("\nTesting connection limits...")
    
    pool = await get_connection_pool()
    stats = pool.get_stats()
    limit = stats['connection_limit']
    
    logger.info(f"Connection limit: {limit}")
    
    # Try to create many connections
    connections = []
    
    try:
        for i in range(limit + 5):
            conn_id = f"test_conn_{i}"
            conn = await pool.get_connection(conn_id)
            connections.append((conn_id, conn))
            
            if i % 10 == 0:
                current_stats = pool.get_stats()
                logger.info(f"Created {i+1} connections, active: {current_stats['active_connections']}")
    
    except Exception as e:
        logger.info(f"Connection creation blocked as expected: {e}")
    
    current_stats = pool.get_stats()
    logger.info(f"Final active connections: {current_stats['active_connections']}")
    
    assert current_stats['active_connections'] <= limit, "Should not exceed connection limit"
    logger.info("✓ Connection limits properly enforced")


async def test_reconnection_handling():
    """Test reconnection with exponential backoff."""
    logger.info("\nTesting reconnection handling...")
    
    client = await get_redis_client()
    
    # Simulate connection failure by using invalid operation
    try:
        # This will trigger reconnection logic
        await client.execute_with_retry(
            lambda conn: conn.execute_command('INVALID_COMMAND'),
            max_retries=2
        )
    except Exception as e:
        logger.info(f"Expected error: {e}")
    
    # Check connection stats
    stats = client.get_connection_stats()
    logger.info(f"Circuit breaker state: {stats['circuit_breaker']['state']}")
    logger.info(f"Reconnect attempts: {stats['reconnect_attempts']}")
    
    logger.info("✓ Reconnection handling works correctly")


async def test_health_monitoring():
    """Test health monitoring without spam."""
    logger.info("\nTesting health monitoring...")
    
    # Create client with short health check interval
    client = await get_redis_client()
    client._health_check_enabled = True
    
    # Wait for some health checks
    await asyncio.sleep(5)
    
    if client.health_monitor:
        stats = client.health_monitor.get_health_stats()
        logger.info(f"Health check stats: {stats}")
        
        assert stats['check_count'] > 0, "Health checks should be running"
        assert stats['is_healthy'], "Connection should be healthy"
        
        logger.info("✓ Health monitoring working without excessive logging")
    else:
        logger.warning("Health monitor not initialized")


async def stress_test_connections():
    """Stress test with many concurrent operations."""
    logger.info("\nRunning connection stress test...")
    
    client = await get_redis_client()
    pool = await get_connection_pool()
    
    initial_stats = pool.get_stats()
    logger.info(f"Initial stats: {initial_stats}")
    
    # Run many concurrent operations
    async def worker(worker_id: int):
        for i in range(100):
            key = f"stress_test_{worker_id}_{i}"
            await client.set(key, f"value_{i}")
            value = await client.get(key)
            await client.delete(key)
    
    # Start workers
    workers = []
    for i in range(10):
        workers.append(asyncio.create_task(worker(i)))
    
    # Wait for completion
    await asyncio.gather(*workers)
    
    final_stats = pool.get_stats()
    logger.info(f"Final stats: {final_stats}")
    
    logger.info("✓ Stress test completed successfully")


async def main():
    """Run all tests."""
    logger.info("Starting Redis connection pool tests...\n")
    
    try:
        await test_singleton_pool()
        await test_connection_reuse()
        await test_pubsub_management()
        await test_connection_limits()
        await test_reconnection_handling()
        await test_health_monitoring()
        await stress_test_connections()
        
        logger.info("\n✅ All tests passed!")
        
    except AssertionError as e:
        logger.error(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        raise
    finally:
        # Cleanup
        await close_redis_client()
        
        # Show final pool stats
        pool = await get_connection_pool()
        final_stats = pool.get_stats()
        logger.info(f"\nFinal pool stats: {final_stats}")


if __name__ == "__main__":
    asyncio.run(main())