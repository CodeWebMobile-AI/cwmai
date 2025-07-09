#!/usr/bin/env python3
"""
Worker Status Monitor (No-Redis Version)

Real-time monitoring tool for checking worker status, progress, and current state.
This version uses in-memory state management instead of Redis.
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
import os
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from scripts.continuous_orchestrator_no_redis import ContinuousOrchestratorNoRedis, WorkerStatus
from scripts.hybrid_state_manager import HybridStateManager

# Configure logging
log_dir = current_dir / 'logs'
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / 'worker_monitor.log'

# Create a specific logger for worker monitoring
logger = logging.getLogger('WorkerMonitor')
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler(sys.stdout)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent propagation to root logger
logger.propagate = False


class WorkerStatusMonitor:
    """Monitor and display worker status information using in-memory state."""
    
    def __init__(self):
        self.hybrid_state = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the monitor with hybrid state manager."""
        if self.initialized:
            return
            
        try:
            # Initialize hybrid state manager
            self.hybrid_state = HybridStateManager(
                db_path="cwmai_state.db",
                queue_persistence_file="work_queue_backup.json",
                enable_queue_persistence=True
            )
            await self.hybrid_state.initialize()
            self.initialized = True
            logger.info("Worker monitor initialized with hybrid state manager")
        except Exception as e:
            logger.error(f"Failed to initialize worker monitor: {e}")
            raise
    
    async def get_worker_status(self) -> Dict[str, Any]:
        """Get comprehensive worker status information."""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Get workers from state
            workers = await self.hybrid_state.get_workers()
            
            # Get queue status
            queue_size = await self.hybrid_state.get_queue_size()
            queue_items = await self.hybrid_state.get_queue_items(limit=10)
            
            # Get completed tasks count
            completed_count = await self.hybrid_state.get_completed_count()
            
            # Calculate system health
            active_workers = sum(1 for w in workers.values() if w and w.get('status') == 'working')
            idle_workers = sum(1 for w in workers.values() if w and w.get('status') == 'idle')
            total_workers = len(workers)
            
            worker_health = 100.0
            if total_workers > 0:
                worker_health = (active_workers / total_workers) * 100
                
            queue_health = 100.0
            if queue_size > 100:
                queue_health = max(0, 100 - (queue_size - 100))
                
            system_health = (worker_health + queue_health) / 2
            
            # Build priority breakdown
            priority_breakdown = {'high': 0, 'medium': 0, 'low': 0}
            for item in queue_items:
                priority = item.get('priority', 'medium')
                priority_breakdown[priority] = priority_breakdown.get(priority, 0) + 1
            
            # Build active tasks list
            active_tasks = []
            for worker_id, worker_data in workers.items():
                if worker_data and worker_data.get('current_task'):
                    task = worker_data['current_task']
                    if isinstance(task, dict):
                        active_tasks.append({
                            'task_id': task.get('id', 'unknown'),
                            'title': task.get('title', 'Unknown task'),
                            'worker_id': worker_id,
                            'started_at': worker_data.get('last_activity')
                        })
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'workers': workers,
                'queue_status': {
                    'total_queued': queue_size,
                    'by_priority': priority_breakdown,
                    'total_completed': completed_count
                },
                'active_tasks': active_tasks,
                'system_health': {
                    'system_health': system_health,
                    'worker_health': worker_health,
                    'queue_health': queue_health,
                    'active_workers': active_workers,
                    'idle_workers': idle_workers,
                    'total_workers': total_workers
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting worker status: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'workers': {},
                'queue_status': {'total_queued': 0, 'by_priority': {}, 'total_completed': 0},
                'active_tasks': [],
                'system_health': {
                    'system_health': 0,
                    'worker_health': 0,
                    'queue_health': 0,
                    'active_workers': 0,
                    'idle_workers': 0,
                    'total_workers': 0
                }
            }
    
    async def display_status(self, continuous: bool = False, interval: int = 2):
        """Display worker status in a formatted way."""
        try:
            while True:
                status = await self.get_worker_status()
                
                # Clear screen for continuous mode
                if continuous:
                    os.system('clear' if os.name == 'posix' else 'cls')
                
                print("\n" + "="*80)
                print(f"WORKER STATUS MONITOR - {status['timestamp']}")
                print("="*80)
                
                # System Health
                health = status['system_health']
                print(f"\nSystem Health: {health['system_health']:.1f}%")
                print(f"  Worker Health: {health['worker_health']:.1f}%")
                print(f"  Queue Health: {health['queue_health']:.1f}%")
                print(f"  Active Workers: {health['active_workers']}/{health['total_workers']}")
                print(f"  Idle Workers: {health['idle_workers']}/{health['total_workers']}")
                
                # Queue Status
                queue = status['queue_status']
                print(f"\nQueue Status:")
                print(f"  Total Queued: {queue['total_queued']}")
                print(f"  Total Completed: {queue['total_completed']}")
                if queue['by_priority']:
                    print("  By Priority:")
                    for priority, count in queue['by_priority'].items():
                        print(f"    {priority}: {count}")
                
                # Worker Details
                print("\nWorker Details:")
                if status['workers']:
                    worker_data = []
                    for worker_id, worker in status['workers'].items():
                        if not worker:
                            continue
                        worker_status = worker.get('status', 'unknown')
                        current_task = worker.get('current_task', 'None')
                        if isinstance(current_task, dict):
                            current_task = current_task.get('title', 'Unknown')[:50] + '...'
                        elif current_task != 'None':
                            current_task = str(current_task)[:50] + '...'
                        
                        worker_data.append([
                            worker_id[:20],
                            worker_status,
                            worker.get('specialization', 'general'),
                            worker.get('total_completed', 0),
                            current_task
                        ])
                    
                    if worker_data:
                        headers = ['Worker ID', 'Status', 'Specialization', 'Completed', 'Current Task']
                        print(tabulate(worker_data, headers=headers, tablefmt='grid'))
                else:
                    print("  No active workers")
                
                # Active Tasks
                if status['active_tasks']:
                    print(f"\nActive Tasks ({len(status['active_tasks'])}):")
                    for task in status['active_tasks'][:5]:  # Show first 5
                        print(f"  - {task['task_id']}: {task['title'][:60]}...")
                    if len(status['active_tasks']) > 5:
                        print(f"  ... and {len(status['active_tasks']) - 5} more tasks")
                
                if not continuous:
                    break
                    
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            logger.error(f"Error displaying status: {e}")
            print(f"\nError: {e}")
    
    async def close(self):
        """Close connections and cleanup."""
        if self.hybrid_state:
            await self.hybrid_state.close()
        self.initialized = False
        logger.info("Worker monitor closed")


async def main():
    """Main entry point for the worker status monitor."""
    parser = argparse.ArgumentParser(description='Monitor worker status and progress')
    parser.add_argument('--continuous', '-c', action='store_true',
                      help='Continuously monitor (refresh every few seconds)')
    parser.add_argument('--interval', '-i', type=int, default=2,
                      help='Update interval in seconds (default: 2)')
    parser.add_argument('--json', '-j', action='store_true',
                      help='Output raw JSON status')
    
    args = parser.parse_args()
    
    monitor = WorkerStatusMonitor()
    try:
        await monitor.initialize()
        
        if args.json:
            # Output JSON and exit
            status = await monitor.get_worker_status()
            print(json.dumps(status, indent=2))
        else:
            # Display formatted status
            await monitor.display_status(
                continuous=args.continuous,
                interval=args.interval
            )
            
    finally:
        await monitor.close()


if __name__ == "__main__":
    asyncio.run(main())