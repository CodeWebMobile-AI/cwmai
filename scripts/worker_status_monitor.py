#!/usr/bin/env python3
"""
Worker Status Monitor

Real-time monitoring tool for checking worker status, progress, and current state.
Provides both CLI and programmatic access to worker information.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import argparse
from tabulate import tabulate
import time
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append('/workspaces/cwmai')

from scripts.continuous_orchestrator import ContinuousOrchestrator, WorkerStatus
from scripts.redis_work_queue import RedisWorkQueue
from scripts.redis_lockfree_state_manager import RedisLockFreeStateManager
from scripts.redis_worker_coordinator import RedisWorkerCoordinator, WorkerEvent
from scripts.redis_integration.redis_client import RedisClient, get_redis_client

# Configure logging
log_file_path = Path('/workspaces/cwmai/worker_monitor.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('WorkerMonitor')


class WorkerStatusMonitor:
    """Monitor and display worker status information."""
    
    # Class-level shared resources
    _shared_redis_client = None
    _shared_pubsub_manager = None
    
    def __init__(self):
        """Initialize worker status monitor."""
        self.orchestrator = None
        self.redis_work_queue = None
        self.redis_state_manager = None
        self.worker_coordinator = None
        logger.info("Worker Status Monitor initialized")
        
    async def initialize(self):
        """Initialize monitoring components."""
        logger.info("Initializing monitoring components...")
        try:
            # Get or create shared Redis client
            if WorkerStatusMonitor._shared_redis_client is None:
                WorkerStatusMonitor._shared_redis_client = await get_redis_client()
                logger.info("Created shared Redis client")
            
            # Initialize Redis components with shared client
            self.redis_work_queue = RedisWorkQueue(redis_client=WorkerStatusMonitor._shared_redis_client)
            await self.redis_work_queue.initialize()
            
            self.redis_state_manager = RedisLockFreeStateManager()
            await self.redis_state_manager.initialize()
            logger.info("Redis components initialized successfully")

            # Initialize worker coordinator to receive live worker events via Pub/Sub
            self.worker_coordinator = RedisWorkerCoordinator(
                redis_client=WorkerStatusMonitor._shared_redis_client,
                worker_id="monitor"
            )
            await self.worker_coordinator.initialize()
            logger.info("Redis worker coordinator initialized")
            
        except Exception as e:
            logger.error(f"Error initializing monitor: {e}")
            print(f"Error initializing monitor: {e}")
            raise
    
    async def get_worker_status(self) -> Dict[str, Any]:
        """Get comprehensive worker status information."""
        logger.debug("Getting worker status...")
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'workers': {},
            'queue_status': {},
            'system_health': {},
            'active_tasks': []
        }
        
        try:
            # Check if redis_state_manager is initialized
            if not self.redis_state_manager:
                logger.error("Redis state manager not initialized")
                return status
                
            # Get worker information from Redis
            active_workers = await self.redis_state_manager.get_set_members("active_workers")
            logger.info(f"Found {len(active_workers)} active workers")
            
            for worker_id in active_workers:
                worker_info = await self._get_worker_details(worker_id)
                if worker_info:
                    status['workers'][worker_id] = worker_info
            
            # Get queue status
            status['queue_status'] = await self._get_queue_status()
            
            # Get system health
            status['system_health'] = await self._get_system_health()
            
            # Get active tasks
            status['active_tasks'] = await self._get_active_tasks()
            
        except Exception as e:
            logger.error(f"Error getting worker status: {e}")
            print(f"Error getting worker status: {e}")
            
        return status
    
    async def _get_worker_details(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific worker."""
        try:
            # Get worker state from Redis using the correct method
            worker_data = await self.redis_state_manager.get_worker_state(worker_id)
            
            if not worker_data:
                return None
            
            # Get worker metrics
            completed = await self.redis_state_manager.get_counter(f"worker:{worker_id}:completed")
            errors = await self.redis_state_manager.get_counter(f"worker:{worker_id}:errors")
            
            # Calculate uptime
            start_time = worker_data.get('started_at')
            if start_time:
                try:
                    start_dt = datetime.fromisoformat(start_time)
                    uptime = (datetime.now(timezone.utc) - start_dt).total_seconds()
                except:
                    uptime = 0
            else:
                uptime = 0
            
            return {
                'id': worker_id,
                'status': worker_data.get('status', 'unknown'),
                'current_task': worker_data.get('current_task'),
                'specialization': worker_data.get('specialization', 'general'),
                'total_completed': completed,
                'total_errors': errors,
                'success_rate': (completed / (completed + errors) * 100) if (completed + errors) > 0 else 0,
                'uptime_seconds': uptime,
                'last_activity': worker_data.get('last_activity', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting worker {worker_id} details: {e}")
            print(f"Error getting worker {worker_id} details: {e}")
            return None
    
    async def _get_queue_status(self) -> Dict[str, Any]:
        """Get work queue status."""
        try:
            if not self.redis_work_queue:
                logger.error("Redis work queue not initialized")
                return {}
                
            stats = await self.redis_work_queue.get_queue_stats()
            
            # Get queue breakdown by priority (use pending count rather than raw stream length)
            priority_breakdown = {}
            for priority, info in stats.get('priority_queues', {}).items():
                priority_breakdown[priority] = info.get('pending', 0)
            
            # Get queue breakdown by type
            type_breakdown = {}
            for task_type, count in stats.get('by_type', {}).items():
                type_breakdown[task_type] = count
            
            return {
                'total_queued': stats.get('total_queued', 0),
                'total_processing': stats.get('total_processing', 0),
                'total_completed': stats.get('total_completed', 0),
                'total_failed': stats.get('total_failed', 0),
                'by_priority': priority_breakdown,
                'by_type': type_breakdown,
                'oldest_task_age': stats.get('oldest_task_age_seconds', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            print(f"Error getting queue status: {e}")
            return {}
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        try:
            # Get health scores from Redis
            health_data = await self.redis_state_manager.get_state("system:health")
            
            if not health_data:
                health_data = {}
            
            # Calculate overall system health
            worker_count = 0
            queue_size = 0
            
            if self.redis_state_manager:
                worker_count = await self.redis_state_manager.get_counter("active_worker_count")
            
            if self.redis_work_queue:
                queue_size = self.redis_work_queue.get_queue_size_sync()
            
            # Simple health calculation
            worker_health = min(100, (worker_count / 10) * 100) if worker_count > 0 else 0
            queue_health = 100 if queue_size > 5 else (queue_size / 5) * 100
            
            return {
                'worker_health': worker_health,
                'queue_health': queue_health,
                'system_health': (worker_health + queue_health) / 2,
                'active_workers': worker_count,
                'queued_tasks': queue_size,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            print(f"Error getting system health: {e}")
            return {}
    
    async def _get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of active tasks being processed."""
        active_tasks = []
        
        try:
            # Check if redis_state_manager is initialized
            if not self.redis_state_manager:
                logger.error("Redis state manager not initialized")
                return active_tasks
                
            # Get all active workers
            active_workers = await self.redis_state_manager.get_set_members("active_workers")
            
            for worker_id in active_workers:
                # Get current task for each worker
                worker_data = await self.redis_state_manager.get_state(f"workers:{worker_id}")
                
                if worker_data and worker_data.get('current_task'):
                    task_info = worker_data['current_task']
                    
                    # Calculate duration
                    start_time = task_info.get('started_at')
                    if start_time:
                        try:
                            start_dt = datetime.fromisoformat(start_time)
                            duration = (datetime.now(timezone.utc) - start_dt).total_seconds()
                        except:
                            duration = 0
                    else:
                        duration = 0
                    
                    active_tasks.append({
                        'task_id': task_info.get('id', 'unknown'),
                        'task_type': task_info.get('task_type', 'unknown'),
                        'title': task_info.get('title', 'Unknown Task'),
                        'worker_id': worker_id,
                        'repository': task_info.get('repository'),
                        'duration_seconds': duration,
                        'priority': task_info.get('priority', 'medium')
                    })
            
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            print(f"Error getting active tasks: {e}")
            
        return active_tasks
    
    def format_worker_table(self, workers: Dict[str, Any]) -> str:
        """Format worker information as a table."""
        if not workers:
            return "No active workers found."
        
        headers = ['Worker ID', 'Status', 'Specialization', 'Current Task', 
                   'Completed', 'Errors', 'Success Rate', 'Uptime']
        rows = []
        
        for worker_id, info in workers.items():
            current_task = info.get('current_task', {})
            task_display = current_task.get('title', 'Idle')[:50] + '...' if current_task else 'Idle'
            
            uptime = info.get('uptime_seconds', 0)
            uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
            
            rows.append([
                worker_id[:20],
                info.get('status', 'unknown'),
                info.get('specialization', 'general'),
                task_display,
                info.get('total_completed', 0),
                info.get('total_errors', 0),
                f"{info.get('success_rate', 0):.1f}%",
                uptime_str
            ])
        
        return tabulate(rows, headers=headers, tablefmt='grid')
    
    def format_queue_table(self, queue_status: Dict[str, Any]) -> str:
        """Format queue status as a table."""
        headers = ['Metric', 'Value']
        rows = [
            ['Total Queued', queue_status.get('total_queued', 0)],
            ['Processing', queue_status.get('total_processing', 0)],
            ['Completed', queue_status.get('total_completed', 0)],
            ['Failed', queue_status.get('total_failed', 0)],
            ['Oldest Task Age', f"{queue_status.get('oldest_task_age', 0):.1f}s"]
        ]
        
        # Add priority breakdown
        priority_data = queue_status.get('by_priority', {})
        if priority_data:
            rows.append(['', ''])  # Empty row
            rows.append(['By Priority:', ''])
            for priority, count in priority_data.items():
                rows.append([f"  {priority}", count])
        
        # Add type breakdown
        type_data = queue_status.get('by_type', {})
        if type_data:
            rows.append(['', ''])  # Empty row
            rows.append(['By Type:', ''])
            for task_type, count in type_data.items():
                rows.append([f"  {task_type}", count])
        
        return tabulate(rows, headers=headers, tablefmt='grid')
    
    def format_active_tasks_table(self, tasks: List[Dict[str, Any]]) -> str:
        """Format active tasks as a table."""
        if not tasks:
            return "No active tasks."
        
        headers = ['Task ID', 'Type', 'Title', 'Worker', 'Repository', 'Duration', 'Priority']
        rows = []
        
        for task in tasks:
            duration = task.get('duration_seconds', 0)
            duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
            
            rows.append([
                task.get('task_id', '')[:12] + '...',
                task.get('task_type', 'unknown'),
                task.get('title', '')[:30] + '...',
                task.get('worker_id', '')[:15],
                task.get('repository', 'N/A')[:20],
                duration_str,
                task.get('priority', 'medium')
            ])
        
        return tabulate(rows, headers=headers, tablefmt='grid')
    
    async def display_status(self, continuous: bool = False, interval: int = 5):
        """Display worker status in the console."""
        logger.info(f"Starting status display (continuous={continuous}, interval={interval}s)")
        try:
            while True:
                # Clear screen (works on Unix-like systems)
                print("\033[2J\033[H")
                
                # Get current status
                status = await self.get_worker_status()
                logger.info(f"Status update - Workers: {len(status['workers'])}, Queue: {status['queue_status'].get('total_queued', 0)}, Active tasks: {len(status['active_tasks'])}")
                
                # Display header
                print("=" * 80)
                print(f"WORKER STATUS MONITOR - {status['timestamp']}")
                print("=" * 80)
                
                # Display system health
                health = status['system_health']
                print(f"\nSystem Health:")
                print(f"  Overall: {health.get('system_health', 0):.1f}%")
                print(f"  Worker Health: {health.get('worker_health', 0):.1f}%")
                print(f"  Queue Health: {health.get('queue_health', 0):.1f}%")
                print(f"  Active Workers: {health.get('active_workers', 0)}")
                print(f"  Queued Tasks: {health.get('queued_tasks', 0)}")
                
                # Display workers
                print(f"\n\nWorker Status:")
                print(self.format_worker_table(status['workers']))
                
                # Display queue status
                print(f"\n\nQueue Status:")
                print(self.format_queue_table(status['queue_status']))
                
                # Display active tasks
                print(f"\n\nActive Tasks:")
                print(self.format_active_tasks_table(status['active_tasks']))
                
                if not continuous:
                    break
                
                # Wait for next update
                print(f"\n\nRefreshing in {interval} seconds... (Press Ctrl+C to exit)")
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            print("\n\nMonitoring stopped.")
        except Exception as e:
            logger.error(f"Error displaying status: {e}")
            print(f"\nError displaying status: {e}")
    
    async def get_json_status(self) -> str:
        """Get worker status as JSON string."""
        status = await self.get_worker_status()
        return json.dumps(status, indent=2, default=str)
    
    async def close(self):
        """Close monitoring connections."""
        logger.info("Closing monitoring connections...")
        if self.redis_state_manager:
            await self.redis_state_manager.close()
        # Note: We don't close the shared Redis client here as it may be used by other monitors
        logger.info("Worker Status Monitor closed")


async def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description='Monitor worker status and progress')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Continuously monitor (refresh every N seconds)')
    parser.add_argument('--interval', '-i', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')
    parser.add_argument('--json', '-j', action='store_true',
                       help='Output as JSON instead of tables')
    parser.add_argument('--worker', '-w', type=str,
                       help='Show details for specific worker')
    
    args = parser.parse_args()
    
    logger.info(f"Worker Status Monitor started with args: continuous={args.continuous}, interval={args.interval}, json={args.json}")
    
    # Initialize monitor
    monitor = WorkerStatusMonitor()
    await monitor.initialize()
    
    try:
        if args.json:
            # Output JSON
            status_json = await monitor.get_json_status()
            print(status_json)
        else:
            # Display formatted tables
            await monitor.display_status(
                continuous=args.continuous,
                interval=args.interval
            )
    finally:
        await monitor.close()
        # Close shared resources on final exit
        if WorkerStatusMonitor._shared_redis_client:
            await WorkerStatusMonitor._shared_redis_client.disconnect()
            WorkerStatusMonitor._shared_redis_client = None
            logger.info("Closed shared Redis client")


if __name__ == "__main__":
    asyncio.run(main())