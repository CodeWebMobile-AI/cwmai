#!/usr/bin/env python3
"""
Redis Connection Health Monitor
Helps diagnose intermittent Redis connection failures
"""

import asyncio
import json
import time
from datetime import datetime
from collections import deque
import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError

class RedisHealthMonitor:
    def __init__(self):
        self.connection_attempts = deque(maxlen=1000)
        self.failure_log = deque(maxlen=100)
        self.stats = {
            'total_attempts': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'timeouts': 0,
            'connection_errors': 0,
            'other_errors': 0,
            'average_response_time': 0,
            'max_response_time': 0,
            'min_response_time': float('inf'),
            'last_failure': None,
            'consecutive_failures': 0,
            'max_consecutive_failures': 0
        }
        
    async def test_connection(self, client):
        """Test a single Redis connection"""
        start_time = time.time()
        attempt = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'response_time': 0,
            'error': None,
            'error_type': None
        }
        
        try:
            # Simple ping test
            await client.ping()
            
            # Test basic operations
            test_key = f'health_check_{int(time.time())}'
            await client.set(test_key, 'test_value', ex=5)
            value = await client.get(test_key)
            await client.delete(test_key)
            
            # Success
            response_time = (time.time() - start_time) * 1000  # ms
            attempt['success'] = True
            attempt['response_time'] = response_time
            
            self.stats['successful_connections'] += 1
            self.stats['consecutive_failures'] = 0
            
            # Update response time stats
            if response_time < self.stats['min_response_time']:
                self.stats['min_response_time'] = response_time
            if response_time > self.stats['max_response_time']:
                self.stats['max_response_time'] = response_time
                
        except TimeoutError as e:
            attempt['error'] = str(e)
            attempt['error_type'] = 'timeout'
            self.stats['timeouts'] += 1
            self.stats['failed_connections'] += 1
            self.stats['consecutive_failures'] += 1
            self.stats['last_failure'] = datetime.now().isoformat()
            self.failure_log.append(attempt)
            
        except ConnectionError as e:
            attempt['error'] = str(e)
            attempt['error_type'] = 'connection'
            self.stats['connection_errors'] += 1
            self.stats['failed_connections'] += 1
            self.stats['consecutive_failures'] += 1
            self.stats['last_failure'] = datetime.now().isoformat()
            self.failure_log.append(attempt)
            
        except Exception as e:
            attempt['error'] = str(e)
            attempt['error_type'] = 'other'
            self.stats['other_errors'] += 1
            self.stats['failed_connections'] += 1
            self.stats['consecutive_failures'] += 1
            self.stats['last_failure'] = datetime.now().isoformat()
            self.failure_log.append(attempt)
        
        # Update max consecutive failures
        if self.stats['consecutive_failures'] > self.stats['max_consecutive_failures']:
            self.stats['max_consecutive_failures'] = self.stats['consecutive_failures']
        
        self.stats['total_attempts'] += 1
        self.connection_attempts.append(attempt)
        
        return attempt
    
    async def monitor_connections(self, duration_seconds=300, interval_seconds=1):
        """Monitor Redis connections for specified duration"""
        print(f"Starting Redis health monitor for {duration_seconds} seconds...")
        print(f"Testing every {interval_seconds} second(s)\n")
        
        # Create Redis client with current settings
        client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            max_connections=100,
            health_check_interval=30,
            retry_on_timeout=True,
            retry_on_error=[ConnectionError, TimeoutError]
        )
        
        start_time = time.time()
        last_report_time = start_time
        
        try:
            # Get Redis server info first
            try:
                info = await client.info('clients')
                print(f"Redis Server Info:")
                print(f"  Connected clients: {info.get('connected_clients', 'unknown')}")
                print(f"  Max clients: {info.get('maxclients', 'default (10000)')}")
                print(f"  Blocked clients: {info.get('blocked_clients', 0)}")
                print()
            except Exception as e:
                print(f"Warning: Could not get Redis server info: {e}\n")
            
            while time.time() - start_time < duration_seconds:
                # Run test
                await self.test_connection(client)
                
                # Report every 10 seconds
                if time.time() - last_report_time >= 10:
                    self.print_current_stats()
                    last_report_time = time.time()
                
                # Wait for next test
                await asyncio.sleep(interval_seconds)
                
        finally:
            await client.close()
            
        # Final report
        print("\n" + "="*60)
        print("FINAL REPORT")
        print("="*60)
        self.print_detailed_report()
        
        # Save detailed logs
        self.save_logs()
        
    def print_current_stats(self):
        """Print current statistics"""
        if self.stats['total_attempts'] == 0:
            return
            
        success_rate = (self.stats['successful_connections'] / self.stats['total_attempts']) * 100
        avg_response_time = self.calculate_average_response_time()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Current Stats:")
        print(f"  Total attempts: {self.stats['total_attempts']}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Failed connections: {self.stats['failed_connections']}")
        print(f"  Consecutive failures: {self.stats['consecutive_failures']}")
        print(f"  Avg response time: {avg_response_time:.1f}ms")
        
        if self.stats['failed_connections'] > 0:
            print(f"  Failure breakdown:")
            print(f"    - Timeouts: {self.stats['timeouts']}")
            print(f"    - Connection errors: {self.stats['connection_errors']}")
            print(f"    - Other errors: {self.stats['other_errors']}")
            
    def calculate_average_response_time(self):
        """Calculate average response time for successful connections"""
        successful_attempts = [a for a in self.connection_attempts if a['success']]
        if not successful_attempts:
            return 0
        return sum(a['response_time'] for a in successful_attempts) / len(successful_attempts)
    
    def print_detailed_report(self):
        """Print detailed final report"""
        if self.stats['total_attempts'] == 0:
            print("No connection attempts were made.")
            return
            
        success_rate = (self.stats['successful_connections'] / self.stats['total_attempts']) * 100
        failure_rate = 100 - success_rate
        avg_response_time = self.calculate_average_response_time()
        
        print(f"\nConnection Statistics:")
        print(f"  Total attempts: {self.stats['total_attempts']}")
        print(f"  Successful: {self.stats['successful_connections']} ({success_rate:.1f}%)")
        print(f"  Failed: {self.stats['failed_connections']} ({failure_rate:.1f}%)")
        
        print(f"\nResponse Times (successful connections):")
        print(f"  Average: {avg_response_time:.1f}ms")
        print(f"  Min: {self.stats['min_response_time']:.1f}ms")
        print(f"  Max: {self.stats['max_response_time']:.1f}ms")
        
        if self.stats['failed_connections'] > 0:
            print(f"\nFailure Analysis:")
            print(f"  Timeouts: {self.stats['timeouts']} ({(self.stats['timeouts']/self.stats['failed_connections']*100):.1f}%)")
            print(f"  Connection errors: {self.stats['connection_errors']} ({(self.stats['connection_errors']/self.stats['failed_connections']*100):.1f}%)")
            print(f"  Other errors: {self.stats['other_errors']} ({(self.stats['other_errors']/self.stats['failed_connections']*100):.1f}%)")
            print(f"  Max consecutive failures: {self.stats['max_consecutive_failures']}")
            print(f"  Last failure: {self.stats['last_failure']}")
            
            # Show recent failures
            print(f"\nRecent Failures (last 5):")
            for failure in list(self.failure_log)[-5:]:
                print(f"  [{failure['timestamp']}] {failure['error_type']}: {failure['error']}")
        
        # Analyze patterns
        self.analyze_failure_patterns()
        
    def analyze_failure_patterns(self):
        """Analyze failure patterns for insights"""
        if not self.failure_log:
            return
            
        print("\nFailure Pattern Analysis:")
        
        # Time-based clustering
        failure_times = [datetime.fromisoformat(f['timestamp']) for f in self.failure_log]
        if len(failure_times) >= 2:
            time_diffs = [(failure_times[i+1] - failure_times[i]).total_seconds() 
                         for i in range(len(failure_times)-1)]
            avg_time_between_failures = sum(time_diffs) / len(time_diffs)
            
            print(f"  Average time between failures: {avg_time_between_failures:.1f}s")
            
            # Check for bursts
            burst_threshold = 5  # failures within 5 seconds
            bursts = sum(1 for diff in time_diffs if diff < burst_threshold)
            if bursts > 0:
                print(f"  Detected {bursts} failure bursts (multiple failures within {burst_threshold}s)")
        
        # Error type patterns
        error_types = {}
        for failure in self.failure_log:
            error_type = failure['error_type']
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(failure)
        
        print("\nRecommendations based on analysis:")
        
        if self.stats['timeouts'] > self.stats['connection_errors']:
            print("  ✗ High timeout rate suggests:")
            print("    - Network latency issues")
            print("    - Redis server under heavy load")
            print("    - Consider increasing socket timeout slightly")
            
        if self.stats['connection_errors'] > self.stats['timeouts']:
            print("  ✗ High connection error rate suggests:")
            print("    - Connection pool exhaustion")
            print("    - Network connectivity issues")
            print("    - Redis server rejecting connections")
            
        if self.stats['max_consecutive_failures'] > 10:
            print("  ✗ Long failure streaks suggest:")
            print("    - Periodic network interruptions")
            print("    - Redis server restarts")
            print("    - Consider implementing exponential backoff")
            
        if avg_response_time > 100:
            print("  ✗ High average response time suggests:")
            print("    - Redis server performance issues")
            print("    - Network latency")
            print("    - Consider Redis server optimization")
    
    def save_logs(self):
        """Save detailed logs to file"""
        log_data = {
            'summary': self.stats,
            'connection_attempts': list(self.connection_attempts),
            'failure_log': list(self.failure_log),
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"redis_health_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nDetailed logs saved to: {filename}")

async def main():
    monitor = RedisHealthMonitor()
    
    # Run for 5 minutes by default, test every second
    await monitor.monitor_connections(duration_seconds=300, interval_seconds=1)

if __name__ == "__main__":
    asyncio.run(main())