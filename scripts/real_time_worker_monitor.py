#!/usr/bin/env python3
"""
Real-Time Worker Activity Monitor

Provides detailed, live monitoring of all worker activities including:
- Current task execution
- Task progress and timing
- Resource usage per worker
- Error tracking
- Performance metrics
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import os
import sys
from dataclasses import dataclass, asdict
from collections import defaultdict
import curses
from tabulate import tabulate
import argparse

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.redis_integration.redis_client import RedisClient
from scripts.redis_integration.redis_pubsub_manager import RedisPubSubManager
from scripts.redis_integration.redis_streams_manager import RedisStreamsManager


@dataclass
class WorkerActivity:
    """Represents a worker's current activity."""
    worker_id: str
    status: str  # idle, working, error, shutdown
    current_task: Optional[Dict[str, Any]] = None
    task_start_time: Optional[float] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: float = 0
    specialization: str = "general"
    error_count: int = 0
    last_error: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {
                'avg_task_time': 0.0,
                'success_rate': 0.0,
                'tasks_per_minute': 0.0
            }


class RealTimeWorkerMonitor:
    """Real-time monitoring of worker activities."""
    
    def __init__(self):
        """Initialize the real-time monitor."""
        self.redis_client = RedisClient()
        self.pubsub_manager = RedisPubSubManager(self.redis_client)
        self.streams_manager = RedisStreamsManager(self.redis_client)
        
        # Worker tracking
        self.workers: Dict[str, WorkerActivity] = {}
        self.task_history: List[Dict[str, Any]] = []
        
        # Event tracking
        self.event_counts = defaultdict(int)
        self.error_log = []
        
        # Display mode
        self.display_mode = "dashboard"  # dashboard, tasks, errors, events
        self.auto_refresh = True
        self.refresh_interval = 1.0  # seconds
        
        # Subscribe to worker events
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Set up Redis subscriptions for worker events."""
        # Subscribe to worker channels
        self.pubsub_manager.subscribe("cwmai:workers:global")
        self.pubsub_manager.subscribe_pattern("cwmai:workers:*")
        
        # Subscribe to task events
        self.pubsub_manager.subscribe("cwmai:tasks:*")
        
        # Subscribe to error events
        self.pubsub_manager.subscribe("cwmai:errors:*")
    
    async def start_monitoring(self):
        """Start monitoring workers in real-time."""
        print("🔍 Starting Real-Time Worker Monitor...")
        print("Press 'q' to quit, 'd' for dashboard, 't' for tasks, 'e' for errors, 'v' for events")
        print("-" * 80)
        
        # Start background tasks
        monitor_task = asyncio.create_task(self._monitor_loop())
        display_task = asyncio.create_task(self._display_loop())
        
        try:
            await asyncio.gather(monitor_task, display_task)
        except KeyboardInterrupt:
            print("\n👋 Stopping monitor...")
        finally:
            monitor_task.cancel()
            display_task.cancel()
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Process Redis messages
                await self._process_messages()
                
                # Update worker states from Redis
                await self._update_worker_states()
                
                # Clean up stale workers
                self._cleanup_stale_workers()
                
                await asyncio.sleep(0.1)  # Fast polling for real-time updates
                
            except Exception as e:
                self.error_log.append({
                    'time': datetime.now(timezone.utc).isoformat(),
                    'error': str(e),
                    'type': 'monitor_loop'
                })
    
    async def _process_messages(self):
        """Process incoming Redis messages."""
        messages = self.pubsub_manager.get_messages()
        
        for msg in messages:
            if msg['type'] in ['message', 'pmessage']:
                try:
                    data = json.loads(msg['data'])
                    await self._handle_event(msg['channel'], data)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    self.error_log.append({
                        'time': datetime.now(timezone.utc).isoformat(),
                        'error': str(e),
                        'type': 'message_processing'
                    })
    
    async def _handle_event(self, channel: str, data: Dict[str, Any]):
        """Handle different types of events."""
        self.event_counts[data.get('event_type', 'unknown')] += 1
        
        event_type = data.get('event_type', '')
        worker_id = data.get('worker_id', '')
        
        if event_type == 'worker_started':
            self._handle_worker_started(worker_id, data)
        elif event_type == 'task_claimed':
            self._handle_task_claimed(worker_id, data)
        elif event_type == 'task_completed':
            self._handle_task_completed(worker_id, data)
        elif event_type == 'task_failed':
            self._handle_task_failed(worker_id, data)
        elif event_type == 'worker_heartbeat':
            self._handle_heartbeat(worker_id, data)
        elif event_type == 'worker_error':
            self._handle_worker_error(worker_id, data)
        elif event_type == 'worker_shutdown':
            self._handle_worker_shutdown(worker_id, data)
    
    def _handle_worker_started(self, worker_id: str, data: Dict[str, Any]):
        """Handle worker start event."""
        self.workers[worker_id] = WorkerActivity(
            worker_id=worker_id,
            status='idle',
            specialization=data.get('specialization', 'general'),
            last_heartbeat=time.time()
        )
    
    def _handle_task_claimed(self, worker_id: str, data: Dict[str, Any]):
        """Handle task claim event."""
        if worker_id not in self.workers:
            self.workers[worker_id] = WorkerActivity(worker_id=worker_id, status='working')
        
        worker = self.workers[worker_id]
        worker.status = 'working'
        worker.current_task = data.get('task', {})
        worker.task_start_time = time.time()
        worker.last_heartbeat = time.time()
    
    def _handle_task_completed(self, worker_id: str, data: Dict[str, Any]):
        """Handle task completion event."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.status = 'idle'
            worker.tasks_completed += 1
            
            # Record task completion
            if worker.task_start_time:
                task_time = time.time() - worker.task_start_time
                self.task_history.append({
                    'worker_id': worker_id,
                    'task': worker.current_task,
                    'duration': task_time,
                    'status': 'completed',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Update performance metrics
                self._update_performance_metrics(worker, task_time)
            
            worker.current_task = None
            worker.task_start_time = None
            worker.last_heartbeat = time.time()
    
    def _handle_task_failed(self, worker_id: str, data: Dict[str, Any]):
        """Handle task failure event."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.status = 'idle'
            worker.tasks_failed += 1
            worker.error_count += 1
            worker.last_error = data.get('error', 'Unknown error')
            
            # Record task failure
            if worker.current_task:
                self.task_history.append({
                    'worker_id': worker_id,
                    'task': worker.current_task,
                    'error': worker.last_error,
                    'status': 'failed',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            worker.current_task = None
            worker.task_start_time = None
            worker.last_heartbeat = time.time()
    
    def _handle_heartbeat(self, worker_id: str, data: Dict[str, Any]):
        """Handle worker heartbeat."""
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()
    
    def _handle_worker_error(self, worker_id: str, data: Dict[str, Any]):
        """Handle worker error event."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.error_count += 1
            worker.last_error = data.get('error', 'Unknown error')
            worker.status = 'error'
            
        self.error_log.append({
            'time': datetime.now(timezone.utc).isoformat(),
            'worker_id': worker_id,
            'error': data.get('error', 'Unknown error'),
            'type': 'worker_error'
        })
    
    def _handle_worker_shutdown(self, worker_id: str, data: Dict[str, Any]):
        """Handle worker shutdown event."""
        if worker_id in self.workers:
            self.workers[worker_id].status = 'shutdown'
    
    def _update_performance_metrics(self, worker: WorkerActivity, task_time: float):
        """Update worker performance metrics."""
        total_tasks = worker.tasks_completed + worker.tasks_failed
        if total_tasks > 0:
            worker.performance_metrics['success_rate'] = (
                worker.tasks_completed / total_tasks * 100
            )
        
        # Update average task time
        if worker.tasks_completed > 1:
            avg_time = worker.performance_metrics['avg_task_time']
            worker.performance_metrics['avg_task_time'] = (
                (avg_time * (worker.tasks_completed - 1) + task_time) / 
                worker.tasks_completed
            )
        else:
            worker.performance_metrics['avg_task_time'] = task_time
        
        # Calculate tasks per minute
        # This is simplified - in production you'd track over a time window
        if worker.tasks_completed > 0:
            worker.performance_metrics['tasks_per_minute'] = (
                worker.tasks_completed / (time.time() - worker.last_heartbeat) * 60
            )
    
    async def _update_worker_states(self):
        """Update worker states from Redis."""
        try:
            # Get worker states from Redis
            worker_keys = self.redis_client.keys("cwmai:workers:status:*")
            
            for key in worker_keys:
                worker_data = self.redis_client.hgetall(key)
                if worker_data:
                    worker_id = key.split(":")[-1]
                    
                    if worker_id not in self.workers:
                        self.workers[worker_id] = WorkerActivity(
                            worker_id=worker_id,
                            status=worker_data.get('status', 'unknown')
                        )
                    
                    # Update worker info
                    worker = self.workers[worker_id]
                    worker.specialization = worker_data.get('specialization', 'general')
                    
                    # Parse current task if exists
                    if worker_data.get('current_task'):
                        try:
                            worker.current_task = json.loads(worker_data['current_task'])
                        except json.JSONDecodeError:
                            pass
                            
        except Exception as e:
            self.error_log.append({
                'time': datetime.now(timezone.utc).isoformat(),
                'error': str(e),
                'type': 'state_update'
            })
    
    def _cleanup_stale_workers(self):
        """Remove workers that haven't sent heartbeat in 30 seconds."""
        current_time = time.time()
        stale_threshold = 30  # seconds
        
        stale_workers = []
        for worker_id, worker in self.workers.items():
            if current_time - worker.last_heartbeat > stale_threshold:
                if worker.status not in ['shutdown', 'error']:
                    worker.status = 'stale'
                stale_workers.append(worker_id)
        
        # Don't remove stale workers, just mark them
        # This helps track workers that may have crashed
    
    async def _display_loop(self):
        """Display loop for terminal output."""
        while self.auto_refresh:
            try:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Display based on current mode
                if self.display_mode == "dashboard":
                    self._display_dashboard()
                elif self.display_mode == "tasks":
                    self._display_tasks()
                elif self.display_mode == "errors":
                    self._display_errors()
                elif self.display_mode == "events":
                    self._display_events()
                
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:
                print(f"Display error: {e}")
                await asyncio.sleep(1)
    
    def _display_dashboard(self):
        """Display main dashboard."""
        print("🎯 REAL-TIME WORKER MONITOR - DASHBOARD")
        print("=" * 100)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Workers: {len(self.workers)} | "
              f"Tasks Completed: {sum(w.tasks_completed for w in self.workers.values())}")
        print("-" * 100)
        
        # Worker table
        headers = ["Worker ID", "Status", "Current Task", "Progress", "Completed", "Failed", "Success Rate", "Specialization"]
        rows = []
        
        for worker_id, worker in sorted(self.workers.items()):
            # Calculate task progress
            progress = ""
            if worker.status == 'working' and worker.task_start_time:
                elapsed = time.time() - worker.task_start_time
                progress = f"{elapsed:.1f}s"
            
            # Current task display
            task_display = ""
            if worker.current_task:
                task_title = worker.current_task.get('title', 'Unknown')
                if len(task_title) > 40:
                    task_title = task_title[:37] + "..."
                task_display = task_title
            
            # Status with emoji
            status_emoji = {
                'idle': '😴',
                'working': '🔨',
                'error': '❌',
                'shutdown': '🛑',
                'stale': '⚠️'
            }
            status = f"{status_emoji.get(worker.status, '❓')} {worker.status}"
            
            rows.append([
                worker_id,
                status,
                task_display,
                progress,
                worker.tasks_completed,
                worker.tasks_failed,
                f"{worker.performance_metrics['success_rate']:.1f}%",
                worker.specialization
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Summary statistics
        print("\n📊 SUMMARY STATISTICS")
        print("-" * 50)
        total_completed = sum(w.tasks_completed for w in self.workers.values())
        total_failed = sum(w.tasks_failed for w in self.workers.values())
        total_tasks = total_completed + total_failed
        
        if total_tasks > 0:
            overall_success_rate = (total_completed / total_tasks) * 100
        else:
            overall_success_rate = 0
        
        working_count = sum(1 for w in self.workers.values() if w.status == 'working')
        idle_count = sum(1 for w in self.workers.values() if w.status == 'idle')
        error_count = sum(1 for w in self.workers.values() if w.status == 'error')
        
        print(f"Working: {working_count} | Idle: {idle_count} | Errors: {error_count}")
        print(f"Total Tasks: {total_tasks} | Success Rate: {overall_success_rate:.1f}%")
        print(f"Recent Errors: {len(self.error_log)}")
        
        # Recent activity
        print("\n📜 RECENT ACTIVITY")
        print("-" * 50)
        recent_tasks = sorted(self.task_history, key=lambda x: x['timestamp'], reverse=True)[:5]
        for task in recent_tasks:
            timestamp = task['timestamp'].split('T')[1].split('.')[0]
            task_title = task.get('task', {}).get('title', 'Unknown')[:50]
            status_icon = "✅" if task['status'] == 'completed' else "❌"
            duration = task.get('duration', 0)
            print(f"{timestamp} | {status_icon} {task['worker_id']}: {task_title} ({duration:.1f}s)")
    
    def _display_tasks(self):
        """Display detailed task view."""
        print("📋 REAL-TIME WORKER MONITOR - TASKS")
        print("=" * 100)
        
        # Current tasks
        print("\n🔨 CURRENTLY EXECUTING TASKS")
        print("-" * 100)
        
        headers = ["Worker", "Task Title", "Type", "Priority", "Duration", "Repository"]
        rows = []
        
        for worker_id, worker in self.workers.items():
            if worker.status == 'working' and worker.current_task:
                task = worker.current_task
                duration = time.time() - worker.task_start_time if worker.task_start_time else 0
                
                rows.append([
                    worker_id,
                    task.get('title', 'Unknown')[:50],
                    task.get('type', 'Unknown'),
                    task.get('priority', 'Unknown'),
                    f"{duration:.1f}s",
                    task.get('repository', 'None')[:30]
                ])
        
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        else:
            print("No tasks currently executing")
        
        # Task history
        print("\n📜 RECENT TASK HISTORY")
        print("-" * 100)
        
        headers = ["Time", "Worker", "Task", "Status", "Duration"]
        rows = []
        
        recent_tasks = sorted(self.task_history, key=lambda x: x['timestamp'], reverse=True)[:20]
        for task in recent_tasks:
            timestamp = task['timestamp'].split('T')[1].split('.')[0]
            task_title = task.get('task', {}).get('title', 'Unknown')[:50]
            status = "✅ Completed" if task['status'] == 'completed' else "❌ Failed"
            duration = f"{task.get('duration', 0):.1f}s" if 'duration' in task else "N/A"
            
            rows.append([
                timestamp,
                task['worker_id'],
                task_title,
                status,
                duration
            ])
        
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        else:
            print("No task history available")
    
    def _display_errors(self):
        """Display error view."""
        print("❌ REAL-TIME WORKER MONITOR - ERRORS")
        print("=" * 100)
        
        if not self.error_log:
            print("No errors recorded! 🎉")
            return
        
        # Recent errors
        print("\n🚨 RECENT ERRORS")
        print("-" * 100)
        
        headers = ["Time", "Worker", "Error Type", "Message"]
        rows = []
        
        recent_errors = sorted(self.error_log, key=lambda x: x['time'], reverse=True)[:20]
        for error in recent_errors:
            timestamp = error['time'].split('T')[1].split('.')[0]
            worker_id = error.get('worker_id', 'System')
            error_type = error.get('type', 'Unknown')
            message = error.get('error', 'Unknown error')[:80]
            
            rows.append([
                timestamp,
                worker_id,
                error_type,
                message
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Error summary by worker
        print("\n📊 ERROR SUMMARY BY WORKER")
        print("-" * 50)
        
        worker_errors = defaultdict(int)
        for error in self.error_log:
            worker_id = error.get('worker_id', 'System')
            worker_errors[worker_id] += 1
        
        for worker_id, count in sorted(worker_errors.items(), key=lambda x: x[1], reverse=True):
            print(f"{worker_id}: {count} errors")
    
    def _display_events(self):
        """Display event statistics."""
        print("📊 REAL-TIME WORKER MONITOR - EVENTS")
        print("=" * 100)
        
        print("\n🎯 EVENT COUNTS")
        print("-" * 50)
        
        headers = ["Event Type", "Count"]
        rows = []
        
        for event_type, count in sorted(self.event_counts.items(), key=lambda x: x[1], reverse=True):
            rows.append([event_type, count])
        
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        else:
            print("No events recorded yet")
        
        # Event rate
        print("\n📈 EVENT RATES")
        print("-" * 50)
        
        total_events = sum(self.event_counts.values())
        if total_events > 0:
            print(f"Total Events: {total_events}")
            # In a real implementation, you'd track events over time windows
            # for proper rate calculation
    
    async def run_interactive(self):
        """Run in interactive mode with keyboard controls."""
        try:
            import aioconsole
            
            # Start monitoring in background
            monitor_task = asyncio.create_task(self.start_monitoring())
            
            # Handle keyboard input
            while True:
                key = await aioconsole.ainput()
                
                if key.lower() == 'q':
                    break
                elif key.lower() == 'd':
                    self.display_mode = "dashboard"
                elif key.lower() == 't':
                    self.display_mode = "tasks"
                elif key.lower() == 'e':
                    self.display_mode = "errors"
                elif key.lower() == 'v':
                    self.display_mode = "events"
                elif key.lower() == 'r':
                    self.auto_refresh = not self.auto_refresh
                elif key.lower() == '+':
                    self.refresh_interval = max(0.1, self.refresh_interval - 0.1)
                elif key.lower() == '-':
                    self.refresh_interval = min(5.0, self.refresh_interval + 0.1)
                
            monitor_task.cancel()
            
        except ImportError:
            # Fallback to non-interactive mode
            await self.start_monitoring()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Real-Time Worker Activity Monitor')
    parser.add_argument('--mode', choices=['dashboard', 'tasks', 'errors', 'events'], 
                       default='dashboard', help='Initial display mode')
    parser.add_argument('--refresh', type=float, default=1.0, 
                       help='Refresh interval in seconds')
    parser.add_argument('--no-auto-refresh', action='store_true',
                       help='Disable auto-refresh')
    
    args = parser.parse_args()
    
    monitor = RealTimeWorkerMonitor()
    monitor.display_mode = args.mode
    monitor.refresh_interval = args.refresh
    monitor.auto_refresh = not args.no_auto_refresh
    
    try:
        await monitor.run_interactive()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")


if __name__ == "__main__":
    asyncio.run(main())