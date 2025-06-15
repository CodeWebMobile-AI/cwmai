#!/usr/bin/env python3
"""
Simple Worker Status Monitor

A lightweight monitoring tool that uses only direct Redis queries without Pub/Sub.
This avoids all connection exhaustion issues.
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
import redis.asyncio as redis

# Add parent directory to path for imports
sys.path.append('/workspaces/cwmai')

# Configure logging
log_file_path = Path('/workspaces/cwmai/worker_monitor_simple.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SimpleWorkerMonitor')


class SimpleWorkerMonitor:
    """Simple worker monitor using only direct Redis queries."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", redis_client: Any = None):
        """Initialize simple worker monitor."""
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = redis_client
        logger.info("Simple Worker Monitor initialized")
    
    async def connect(self, shared_client: Any = None):
        """Connect to Redis, reusing a shared client if provided."""
        if not self.redis_client:
            if shared_client:
                self.redis_client = shared_client
            else:
                from scripts.redis_integration.redis_client import get_redis_client
                self.redis_client = await get_redis_client()
            logger.info("Connected to Redis")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.aclose()
            self.redis_client = None
            logger.info("Disconnected from Redis")
    
    async def get_worker_status(self) -> Dict[str, Any]:
        """Get worker status using direct Redis queries."""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'workers': {},
            'queue_status': {},
            'system_health': {},
            'active_tasks': []
        }
        
        try:
            # Get active workers
            active_workers = await self.redis_client.smembers("active_workers")
            logger.debug(f"Found {len(active_workers)} active workers")
            
            # Get details for each worker
            for worker_id in active_workers:
                worker_info = await self._get_worker_details(worker_id)
                if worker_info:
                    status['workers'][worker_id] = worker_info
            
            # Get queue status
            status['queue_status'] = await self._get_queue_status()
            
            # Get system health
            status['system_health'] = await self._get_system_health(len(active_workers))
            
            # Get active tasks
            status['active_tasks'] = await self._get_active_tasks(active_workers)
            
        except Exception as e:
            logger.error(f"Error getting worker status: {e}")
        
        return status
    
    async def _get_worker_details(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get worker details from Redis."""
        try:
            # Get worker state
            state_key = f"worker:state:{worker_id}"
            state_data = await self.redis_client.get(state_key)
            
            if not state_data:
                return None
            
            worker_data = json.loads(state_data)
            
            # Get metrics
            completed = await self.redis_client.get(f"worker:{worker_id}:completed") or "0"
            errors = await self.redis_client.get(f"worker:{worker_id}:errors") or "0"
            
            completed = int(completed)
            errors = int(errors)
            
            # Calculate uptime
            start_time = worker_data.get('started_at')
            uptime = 0
            if start_time:
                try:
                    start_dt = datetime.fromisoformat(start_time)
                    uptime = (datetime.now(timezone.utc) - start_dt).total_seconds()
                except:
                    pass
            
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
            return None
    
    async def _get_queue_status(self) -> Dict[str, Any]:
        """Get queue status from Redis streams."""
        queue_status = {
            'total_queued': 0,
            'total_processing': 0,
            'total_completed': 0,
            'total_failed': 0,
            'by_priority': {},
            'by_type': {}
        }
        
        try:
            # Priority streams
            priorities = ['critical', 'high', 'medium', 'low', 'background']
            
            for priority in priorities:
                stream_key = f"cwmai:work_queue:{priority}"
                try:
                    info = await self.redis_client.xinfo_stream(stream_key)
                    count = info.get('length', 0)
                    queue_status['by_priority'][priority] = count
                    queue_status['total_queued'] += count
                except:
                    queue_status['by_priority'][priority] = 0
            
            # Get processing count
            processing_count = await self.redis_client.get("queue:processing_count") or "0"
            queue_status['total_processing'] = int(processing_count)
            
            # Get completed/failed counts
            completed_count = await self.redis_client.get("queue:completed_count") or "0"
            failed_count = await self.redis_client.get("queue:failed_count") or "0"
            queue_status['total_completed'] = int(completed_count)
            queue_status['total_failed'] = int(failed_count)
            
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
        
        return queue_status
    
    async def _get_system_health(self, worker_count: int) -> Dict[str, Any]:
        """Calculate system health metrics."""
        try:
            # Get queue size
            total_queued = 0
            priorities = ['critical', 'high', 'medium', 'low', 'background']
            
            for priority in priorities:
                stream_key = f"cwmai:work_queue:{priority}"
                try:
                    info = await self.redis_client.xinfo_stream(stream_key)
                    total_queued += info.get('length', 0)
                except:
                    pass
            
            # Simple health calculations
            worker_health = min(100, (worker_count / 10) * 100) if worker_count > 0 else 0
            queue_health = 100 if total_queued > 5 else (total_queued / 5) * 100
            
            return {
                'worker_health': worker_health,
                'queue_health': queue_health,
                'system_health': (worker_health + queue_health) / 2,
                'active_workers': worker_count,
                'queued_tasks': total_queued,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return {
                'worker_health': 0,
                'queue_health': 0,
                'system_health': 0,
                'active_workers': worker_count,
                'queued_tasks': 0,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
    
    async def _get_active_tasks(self, active_workers: set) -> List[Dict[str, Any]]:
        """Get active tasks from worker states."""
        active_tasks = []
        
        try:
            for worker_id in active_workers:
                # Get worker state
                state_key = f"worker:state:{worker_id}"
                state_data = await self.redis_client.get(state_key)
                
                if state_data:
                    worker_data = json.loads(state_data)
                    current_task = worker_data.get('current_task')
                    
                    if current_task:
                        # Calculate duration
                        duration = 0
                        start_time = current_task.get('started_at')
                        if start_time:
                            try:
                                start_dt = datetime.fromisoformat(start_time)
                                duration = (datetime.now(timezone.utc) - start_dt).total_seconds()
                            except:
                                pass
                        
                        active_tasks.append({
                            'task_id': current_task.get('id', 'unknown'),
                            'task_type': current_task.get('task_type', 'unknown'),
                            'title': current_task.get('title', 'Unknown Task'),
                            'worker_id': worker_id,
                            'repository': current_task.get('repository'),
                            'duration_seconds': duration,
                            'priority': current_task.get('priority', 'medium')
                        })
            
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
        
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
            if isinstance(current_task, dict):
                task_display = current_task.get('title', 'Idle')[:50]
                if len(current_task.get('title', '')) > 50:
                    task_display += '...'
            else:
                task_display = 'Idle'
            
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
            ['Failed', queue_status.get('total_failed', 0)]
        ]
        
        # Add priority breakdown
        priority_data = queue_status.get('by_priority', {})
        if priority_data:
            rows.append(['', ''])  # Empty row
            rows.append(['By Priority:', ''])
            for priority, count in priority_data.items():
                rows.append([f"  {priority}", count])
        
        return tabulate(rows, headers=headers, tablefmt='grid')
    
    def format_active_tasks_table(self, tasks: List[Dict[str, Any]]) -> str:
        """Format active tasks as a table."""
        if not tasks:
            return "No active tasks."
        
        headers = ['Task ID', 'Type', 'Title', 'Worker', 'Duration', 'Priority']
        rows = []
        
        for task in tasks:
            duration = task.get('duration_seconds', 0)
            duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
            
            title = task.get('title', '')[:30]
            if len(task.get('title', '')) > 30:
                title += '...'
            
            task_id = task.get('task_id', '')[:12]
            if len(task.get('task_id', '')) > 12:
                task_id += '...'
            
            rows.append([
                task_id,
                task.get('task_type', 'unknown'),
                title,
                task.get('worker_id', '')[:15],
                duration_str,
                task.get('priority', 'medium')
            ])
        
        return tabulate(rows, headers=headers, tablefmt='grid')
    
    async def display_status(self, continuous: bool = False, interval: int = 5):
        """Display worker status in the console."""
        logger.info(f"Starting status display (continuous={continuous}, interval={interval}s)")
        
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H")
                
                # Get current status
                status = await self.get_worker_status()
                
                # Display header
                print("=" * 80)
                print(f"SIMPLE WORKER STATUS MONITOR - {status['timestamp']}")
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


async def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description='Simple Worker Monitor - No Pub/Sub')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Continuously monitor (refresh every N seconds)')
    parser.add_argument('--interval', '-i', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')
    parser.add_argument('--json', '-j', action='store_true',
                       help='Output as JSON instead of tables')
    parser.add_argument('--redis-url', '-r', type=str, default='redis://localhost:6379',
                       help='Redis connection URL')
    
    args = parser.parse_args()
    
    logger.info(f"Simple Worker Monitor started")
    
    # Initialize monitor
    monitor = SimpleWorkerMonitor(redis_url=args.redis_url)
    await monitor.connect()
    
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


if __name__ == "__main__":
    asyncio.run(main())