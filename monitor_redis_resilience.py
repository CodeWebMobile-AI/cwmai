#!/usr/bin/env python3
"""
Real-time Redis Resilience Monitor

Provides real-time monitoring of the Redis connection system,
showing health status, metrics, and automatic failover behavior.
"""

import asyncio
import logging
import time
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any
import curses

from scripts.redis_integration.redis_connection_manager import (
    get_connection_manager, ConnectionState
)
from scripts.redis_integration.redis_resilient_task_persistence import ResilientRedisTaskPersistence


class RedisResilienceMonitor:
    """Real-time monitor for Redis resilience."""
    
    def __init__(self):
        self.connection_manager = None
        self.task_persistence = None
        self.running = True
        self.start_time = time.time()
        self.update_interval = 1.0  # seconds
        
        # Metrics history
        self.history = {
            'timestamps': [],
            'states': [],
            'redis_ops': [],
            'fallback_ops': [],
            'recovery_attempts': [],
            'errors': []
        }
        
        # Configure logging to file
        log_file = f"redis_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize monitoring components."""
        self.logger.info("Initializing Redis Resilience Monitor...")
        
        # Get connection manager
        self.connection_manager = await get_connection_manager(enable_fallback=True)
        
        # Initialize task persistence
        self.task_persistence = ResilientRedisTaskPersistence(enable_file_fallback=True)
        
        self.logger.info("Monitor initialized")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics."""
        status = self.connection_manager.get_status()
        persistence_metrics = self.task_persistence.get_metrics()
        
        # Calculate rates
        elapsed = time.time() - self.start_time
        redis_rate = status['metrics']['redis_operations'] / elapsed if elapsed > 0 else 0
        fallback_rate = status['metrics']['fallback_operations'] / elapsed if elapsed > 0 else 0
        
        metrics = {
            'timestamp': datetime.now(timezone.utc),
            'elapsed_time': elapsed,
            'connection_state': status['state'],
            'redis_connected': status['redis_connected'],
            'fallback_enabled': status['fallback_enabled'],
            'recovery_attempts': status['recovery_attempts'],
            'total_operations': {
                'redis': status['metrics']['redis_operations'],
                'fallback': status['metrics']['fallback_operations'],
                'circuit_breaker_trips': status['metrics']['circuit_breaker_trips']
            },
            'operation_rates': {
                'redis_per_second': redis_rate,
                'fallback_per_second': fallback_rate
            },
            'persistence': {
                'total': persistence_metrics['total_operations'],
                'redis': persistence_metrics['redis_operations'],
                'sqlite_fallback': persistence_metrics['sqlite_fallback_operations'],
                'file_fallback': persistence_metrics['file_fallback_operations'],
                'redis_percentage': persistence_metrics['redis_percentage']
            }
        }
        
        # Add health stats if available
        if status.get('health_stats'):
            health = status['health_stats']['health']
            metrics['health'] = {
                'is_healthy': health.get('is_healthy', False),
                'check_count': health.get('check_count', 0),
                'failure_count': health.get('failure_count', 0),
                'uptime_percentage': health.get('uptime_percentage', 0)
            }
        
        # Add circuit breaker info
        if status['health_stats'] and status['health_stats'].get('circuit_breaker'):
            cb = status['health_stats']['circuit_breaker']
            metrics['circuit_breaker'] = {
                'enabled': cb.get('enabled', False),
                'state': cb.get('state', 'unknown'),
                'failure_count': cb.get('failure_count', 0),
                'failure_threshold': cb.get('failure_threshold', 0)
            }
        
        return metrics
    
    def display_metrics_console(self, metrics: Dict[str, Any]):
        """Display metrics in console format."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print(f"Redis Resilience Monitor - {metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 80)
        
        # Connection Status
        state = metrics['connection_state']
        state_symbol = {
            'healthy': '✅',
            'degraded': '⚠️',
            'failed': '❌',
            'recovering': '🔄'
        }.get(state, '❓')
        
        print(f"\nConnection Status: {state_symbol} {state.upper()}")
        print(f"Redis Connected: {'Yes' if metrics['redis_connected'] else 'No'}")
        print(f"Fallback Enabled: {'Yes' if metrics['fallback_enabled'] else 'No'}")
        print(f"Recovery Attempts: {metrics['recovery_attempts']}")
        
        # Operation Statistics
        print(f"\nOperation Statistics (Total):")
        print(f"  Redis Operations: {metrics['total_operations']['redis']:,}")
        print(f"  Fallback Operations: {metrics['total_operations']['fallback']:,}")
        print(f"  Circuit Breaker Trips: {metrics['total_operations']['circuit_breaker_trips']:,}")
        
        # Operation Rates
        print(f"\nOperation Rates:")
        print(f"  Redis: {metrics['operation_rates']['redis_per_second']:.2f} ops/sec")
        print(f"  Fallback: {metrics['operation_rates']['fallback_per_second']:.2f} ops/sec")
        
        # Task Persistence
        print(f"\nTask Persistence:")
        print(f"  Total Operations: {metrics['persistence']['total']:,}")
        print(f"  Redis: {metrics['persistence']['redis']:,} ({metrics['persistence']['redis_percentage']:.1f}%)")
        print(f"  SQLite Fallback: {metrics['persistence']['sqlite_fallback']:,}")
        print(f"  File Fallback: {metrics['persistence']['file_fallback']:,}")
        
        # Health Monitoring
        if 'health' in metrics:
            health = metrics['health']
            print(f"\nHealth Monitoring:")
            print(f"  Status: {'Healthy' if health['is_healthy'] else 'Unhealthy'}")
            print(f"  Checks: {health['check_count']:,}")
            print(f"  Failures: {health['failure_count']:,}")
            print(f"  Uptime: {health['uptime_percentage']:.1f}%")
        
        # Circuit Breaker
        if 'circuit_breaker' in metrics:
            cb = metrics['circuit_breaker']
            if cb['enabled']:
                print(f"\nCircuit Breaker:")
                print(f"  State: {cb['state'].upper()}")
                print(f"  Failures: {cb['failure_count']}/{cb['failure_threshold']}")
        
        # Runtime
        print(f"\nRuntime: {self._format_duration(metrics['elapsed_time'])}")
        print("\nPress Ctrl+C to stop monitoring...")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    async def monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Display metrics
                self.display_metrics_console(metrics)
                
                # Store history (keep last 1000 points)
                self.history['timestamps'].append(metrics['timestamp'])
                self.history['states'].append(metrics['connection_state'])
                self.history['redis_ops'].append(metrics['total_operations']['redis'])
                self.history['fallback_ops'].append(metrics['total_operations']['fallback'])
                self.history['recovery_attempts'].append(metrics['recovery_attempts'])
                
                if len(self.history['timestamps']) > 1000:
                    for key in self.history:
                        self.history[key] = self.history[key][-1000:]
                
                # Log significant events
                if len(self.history['states']) > 1:
                    if self.history['states'][-1] != self.history['states'][-2]:
                        self.logger.warning(
                            f"State change: {self.history['states'][-2]} -> {self.history['states'][-1]}"
                        )
                
                await asyncio.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                self.history['errors'].append({
                    'timestamp': datetime.now(timezone.utc),
                    'error': str(e)
                })
                await asyncio.sleep(self.update_interval)
    
    async def run(self):
        """Run the monitor."""
        try:
            await self.initialize()
            self.logger.info("Starting monitoring loop...")
            await self.monitor_loop()
        except KeyboardInterrupt:
            self.logger.info("Monitor stopped by user")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up monitor...")
        
        # Save final report
        if self.history['timestamps']:
            report = {
                'start_time': self.start_time,
                'end_time': time.time(),
                'total_duration': time.time() - self.start_time,
                'states_encountered': list(set(self.history['states'])),
                'total_redis_ops': self.history['redis_ops'][-1] if self.history['redis_ops'] else 0,
                'total_fallback_ops': self.history['fallback_ops'][-1] if self.history['fallback_ops'] else 0,
                'max_recovery_attempts': max(self.history['recovery_attempts']) if self.history['recovery_attempts'] else 0,
                'errors_count': len(self.history['errors'])
            }
            
            self.logger.info(f"Final report: {report}")
        
        # Close connection manager
        from scripts.redis_integration.redis_connection_manager import close_connection_manager
        await close_connection_manager()


async def main():
    """Main entry point."""
    monitor = RedisResilienceMonitor()
    await monitor.run()


if __name__ == "__main__":
    print("Starting Redis Resilience Monitor...")
    print("This will display real-time metrics about the Redis connection system.")
    print("Logs are saved to redis_monitor_*.log")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitor stopped.")