#!/usr/bin/env python3
"""
Redis Connection Monitor

Monitor and debug Redis connections to identify and fix connection leaks.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

from scripts.redis_integration import (
    get_redis_client,
    close_redis_client,
    get_connection_pool,
    close_connection_pool
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisConnectionMonitor:
    """Monitor Redis connections and identify issues."""
    
    def __init__(self):
        """Initialize connection monitor."""
        self.start_time = time.time()
        self.samples: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
    async def monitor_connections(self, duration: int = 60, interval: int = 5):
        """Monitor connections for a specified duration.
        
        Args:
            duration: Total monitoring duration in seconds
            interval: Sampling interval in seconds
        """
        logger.info(f"Starting Redis connection monitoring for {duration} seconds...")
        
        end_time = time.time() + duration
        sample_count = 0
        
        try:
            # Get clients
            redis_client = await get_redis_client()
            connection_pool = await get_connection_pool()
            
            while time.time() < end_time:
                sample_count += 1
                
                # Collect metrics
                sample = await self._collect_sample(redis_client, connection_pool, sample_count)
                self.samples.append(sample)
                
                # Check for issues
                self._check_for_issues(sample)
                
                # Display current status
                self._display_status(sample)
                
                await asyncio.sleep(interval)
            
            # Generate report
            self._generate_report()
            
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
        finally:
            # Cleanup
            await close_redis_client()
            await close_connection_pool()
    
    async def _collect_sample(self, redis_client, connection_pool, sample_num: int) -> Dict[str, Any]:
        """Collect a sample of connection metrics."""
        sample = {
            'timestamp': datetime.now().isoformat(),
            'sample_number': sample_num,
            'uptime_seconds': time.time() - self.start_time
        }
        
        try:
            # Get client stats
            client_stats = redis_client.get_connection_stats()
            sample['client_stats'] = client_stats
            
            # Get pool stats
            pool_stats = connection_pool.get_stats()
            sample['pool_stats'] = pool_stats
            
            # Get Redis server info
            info = await redis_client.info()
            sample['redis_info'] = {
                'connected_clients': info.get('connected_clients', 0),
                'client_recent_max_input_buffer': info.get('client_recent_max_input_buffer', 0),
                'client_recent_max_output_buffer': info.get('client_recent_max_output_buffer', 0),
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'used_memory_peak_human': info.get('used_memory_peak_human', 'N/A'),
                'total_connections_received': info.get('total_connections_received', 0),
                'rejected_connections': info.get('rejected_connections', 0)
            }
            
        except Exception as e:
            logger.error(f"Error collecting sample: {e}")
            sample['error'] = str(e)
        
        return sample
    
    def _check_for_issues(self, sample: Dict[str, Any]):
        """Check for connection issues."""
        alerts = []
        
        # Check pool stats
        if 'pool_stats' in sample:
            pool = sample['pool_stats']
            
            # Connection limit approaching
            if pool['active_connections'] > pool['connection_limit'] * 0.8:
                alerts.append({
                    'level': 'WARNING',
                    'message': f"Connection pool near limit: {pool['active_connections']}/{pool['connection_limit']}"
                })
            
            # Too many reconnect attempts
            if any(attempts > 3 for attempts in pool['reconnect_attempts'].values()):
                alerts.append({
                    'level': 'ERROR',
                    'message': f"High reconnect attempts detected: {pool['reconnect_attempts']}"
                })
        
        # Check client stats
        if 'client_stats' in sample:
            client = sample['client_stats']
            
            # Circuit breaker open
            if client.get('circuit_breaker', {}).get('state') == 'open':
                alerts.append({
                    'level': 'ERROR',
                    'message': "Circuit breaker is OPEN - connections failing"
                })
            
            # Health check failures
            health = client.get('health', {})
            if health.get('failure_rate', 0) > 0.1:  # >10% failure rate
                alerts.append({
                    'level': 'WARNING',
                    'message': f"High health check failure rate: {health['failure_rate']:.2%}"
                })
        
        # Check Redis server stats
        if 'redis_info' in sample:
            info = sample['redis_info']
            
            # Rejected connections
            if info['rejected_connections'] > 0:
                alerts.append({
                    'level': 'ERROR',
                    'message': f"Redis rejected {info['rejected_connections']} connections"
                })
        
        # Store alerts
        for alert in alerts:
            alert['timestamp'] = sample['timestamp']
            self.alerts.append(alert)
            logger.warning(f"[{alert['level']}] {alert['message']}")
    
    def _display_status(self, sample: Dict[str, Any]):
        """Display current connection status."""
        if 'pool_stats' in sample and 'redis_info' in sample:
            pool = sample['pool_stats']
            info = sample['redis_info']
            
            logger.info(
                f"Sample #{sample['sample_number']} - "
                f"Pool: {pool['active_connections']}/{pool['connection_limit']} active, "
                f"{pool['pubsub_connections']} pubsub | "
                f"Redis: {info['connected_clients']} clients, "
                f"Memory: {info['used_memory_human']}"
            )
    
    def _generate_report(self):
        """Generate monitoring report."""
        logger.info("\n" + "="*60)
        logger.info("Redis Connection Monitoring Report")
        logger.info("="*60)
        
        # Summary
        logger.info(f"\nMonitoring Duration: {time.time() - self.start_time:.1f} seconds")
        logger.info(f"Total Samples: {len(self.samples)}")
        logger.info(f"Total Alerts: {len(self.alerts)}")
        
        # Connection trends
        if self.samples:
            first_sample = self.samples[0]
            last_sample = self.samples[-1]
            
            if 'pool_stats' in first_sample and 'pool_stats' in last_sample:
                first_pool = first_sample['pool_stats']
                last_pool = last_sample['pool_stats']
                
                logger.info("\nConnection Pool Trends:")
                logger.info(f"  Active connections: {first_pool['active_connections']} -> {last_pool['active_connections']}")
                logger.info(f"  PubSub connections: {first_pool['pubsub_connections']} -> {last_pool['pubsub_connections']}")
            
            if 'redis_info' in first_sample and 'redis_info' in last_sample:
                first_info = first_sample['redis_info']
                last_info = last_sample['redis_info']
                
                logger.info("\nRedis Server Trends:")
                logger.info(f"  Connected clients: {first_info['connected_clients']} -> {last_info['connected_clients']}")
                logger.info(f"  Total connections: {first_info['total_connections_received']} -> {last_info['total_connections_received']}")
        
        # Alert summary
        if self.alerts:
            logger.info("\nAlerts Summary:")
            alert_counts = {}
            for alert in self.alerts:
                key = f"{alert['level']}: {alert['message']}"
                alert_counts[key] = alert_counts.get(key, 0) + 1
            
            for alert_type, count in sorted(alert_counts.items()):
                logger.info(f"  {alert_type} (occurred {count} times)")
        
        # Recommendations
        logger.info("\nRecommendations:")
        if any('Connection pool near limit' in alert['message'] for alert in self.alerts):
            logger.info("  - Consider increasing connection pool limit")
        if any('reconnect attempts' in alert['message'] for alert in self.alerts):
            logger.info("  - Check network stability and Redis server health")
        if any('Circuit breaker' in alert['message'] for alert in self.alerts):
            logger.info("  - Circuit breaker triggered - check for persistent connection issues")
        
        logger.info("\n" + "="*60)


async def test_connection_leaks():
    """Test for connection leaks by creating multiple clients."""
    logger.info("Testing for connection leaks...")
    
    monitor = RedisConnectionMonitor()
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(monitor.monitor_connections(duration=30, interval=2))
    
    # Simulate multiple client connections
    logger.info("Creating test connections...")
    
    try:
        # Test 1: Multiple pubsub subscriptions
        client = await get_redis_client()
        pubsubs = []
        
        for i in range(5):
            pubsub = await client.pubsub()
            await pubsub.subscribe(f"test_channel_{i}")
            pubsubs.append(pubsub)
            await asyncio.sleep(1)
        
        # Test 2: Rapid connect/disconnect
        for i in range(3):
            temp_client = await get_redis_client()
            await temp_client.ping()
            await asyncio.sleep(1)
        
        # Wait for monitoring to complete
        await monitor_task
        
    except Exception as e:
        logger.error(f"Error during leak test: {e}")
    finally:
        # Cleanup
        for pubsub in pubsubs:
            try:
                await pubsub.close()
            except:
                pass


async def main():
    """Main monitoring function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        await test_connection_leaks()
    else:
        monitor = RedisConnectionMonitor()
        await monitor.monitor_connections(duration=60, interval=5)


if __name__ == "__main__":
    asyncio.run(main())