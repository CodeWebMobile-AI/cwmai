#!/usr/bin/env python3
"""
Monitor Dynamic Claude Executor Pool

This script monitors a running Dynamic Claude Executor Pool and displays
real-time metrics about its scaling behavior and performance.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.claude_execution_queue import ClaudeExecutionQueue


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_pool_metrics(stats: Dict[str, Any]):
    """Print pool metrics in a formatted way."""
    print("\n" + "="*60)
    print(f"📊 DYNAMIC CLAUDE EXECUTOR POOL METRICS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Basic info
    print(f"\n🔧 Configuration:")
    print(f"  Current Executors: {stats.get('num_executors', 0)}")
    print(f"  Min/Max Range: {stats.get('min_executors', 1)} - {stats.get('max_executors', 10)}")
    
    # Scaling metrics
    scaling = stats.get('scaling_metrics', {})
    print(f"\n📈 Scaling Metrics:")
    print(f"  Current Utilization: {scaling.get('current_utilization', 0):.1%}")
    print(f"  Avg Queue Depth (5min): {scaling.get('avg_queue_depth_5min', 0):.1f} tasks")
    print(f"  Avg Wait Time (5min): {format_duration(scaling.get('avg_wait_time_5min', 0))}")
    print(f"  Avg Utilization (5min): {scaling.get('avg_utilization_5min', 0):.1%}")
    
    # Recent scaling events
    events = scaling.get('recent_scaling_events', [])
    if events:
        print(f"\n🔄 Recent Scaling Events:")
        for event in events[-5:]:  # Show last 5 events
            timestamp = event.get('timestamp', '')
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp
            else:
                time_str = timestamp.strftime('%H:%M:%S') if timestamp else 'Unknown'
            
            event_type = "📈 UP" if event.get('type') == 'scale_up' else "📉 DOWN"
            print(f"  {time_str} {event_type}: {event.get('from_count')}→{event.get('to_count')} "
                  f"({event.get('reason', 'Unknown reason')})")
    
    # Performance stats
    print(f"\n⚡ Performance:")
    print(f"  Total Completed: {stats.get('total_completed', 0)}")
    print(f"  Total Failed: {stats.get('total_failed', 0)}")
    if stats.get('avg_execution_time', 0) > 0:
        print(f"  Avg Execution Time: {format_duration(stats.get('avg_execution_time', 0))}")
    
    # Individual executor status
    executors = stats.get('executors', {})
    if executors:
        print(f"\n👷 Executor Status:")
        for exec_id, exec_stats in sorted(executors.items()):
            status = "🟢 BUSY" if exec_stats.get('current_task') else "⚪ IDLE"
            idle_time = exec_stats.get('idle_time', 0)
            uptime = exec_stats.get('uptime', 0)
            completed = exec_stats.get('tasks_completed', 0)
            
            print(f"  {exec_id}: {status} | Completed: {completed} | "
                  f"Uptime: {format_duration(uptime)} | Idle: {format_duration(idle_time)}")
            
            # Show current task if busy
            current_task = exec_stats.get('current_task')
            if current_task:
                print(f"    └─ Working on: {current_task.get('repository', 'Unknown')} "
                      f"issue #{current_task.get('issue_number', 'Unknown')}")


async def monitor_pool(queue_file: str = "claude_execution_queue.json", interval: int = 10):
    """Monitor the pool by reading queue statistics."""
    print("🔍 Monitoring Dynamic Claude Executor Pool")
    print(f"Queue file: {queue_file}")
    print(f"Update interval: {interval}s")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    queue = ClaudeExecutionQueue(queue_file)
    
    try:
        while True:
            # Get queue stats
            queue_stats = await queue.get_stats()
            
            # Try to read pool stats from a state file if available
            pool_stats_file = "dynamic_claude_pool_stats.json"
            pool_stats = {}
            
            if os.path.exists(pool_stats_file):
                try:
                    with open(pool_stats_file, 'r') as f:
                        pool_stats = json.load(f)
                except:
                    pass
            
            # Display queue info
            print("\n" + "-"*60)
            print(f"📦 Queue Status:")
            print(f"  Pending: {queue_stats.get('pending', 0)}")
            print(f"  Processing: {queue_stats.get('processing', 0)}")
            print(f"  Completed: {queue_stats.get('completed', 0)}")
            
            if queue_stats.get('avg_execution_time', 0) > 0:
                print(f"  Avg Execution Time: {format_duration(queue_stats.get('avg_execution_time', 0))}")
            
            # Show active executors
            executors = queue_stats.get('executors', [])
            if executors:
                print(f"  Active Executors: {', '.join(executors)}")
            
            # Display pool metrics if available
            if pool_stats:
                print_pool_metrics(pool_stats)
            
            await asyncio.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")


async def show_queue_details(queue_file: str = "claude_execution_queue.json"):
    """Show detailed information about the queue."""
    queue = ClaudeExecutionQueue(queue_file)
    
    print("📋 Claude Execution Queue Details\n")
    
    # Get all tasks
    pending = queue.get_pending_tasks()
    processing = queue.get_processing_tasks()
    
    if pending:
        print(f"🔵 Pending Tasks ({len(pending)}):")
        for task in pending[:10]:  # Show first 10
            created = task.get('created_at', 'Unknown')
            if created != 'Unknown':
                try:
                    dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    created = dt.strftime('%H:%M:%S')
                except:
                    pass
            
            print(f"  • {task.get('repository', 'Unknown')} #{task.get('issue_number', '?')} "
                  f"(created: {created})")
        
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
    
    if processing:
        print(f"\n🟡 Processing Tasks ({len(processing)}):")
        for task in processing:
            started = task.get('started_at', 'Unknown')
            if started != 'Unknown':
                try:
                    dt = datetime.fromisoformat(started.replace('Z', '+00:00'))
                    duration = (datetime.now(datetime.timezone.utc) - dt).total_seconds()
                    started = f"{format_duration(duration)} ago"
                except:
                    pass
            
            print(f"  • {task.get('repository', 'Unknown')} #{task.get('issue_number', '?')} "
                  f"by {task.get('executor_id', 'Unknown')} ({started})")
    
    # Summary stats
    stats = await queue.get_stats()
    print(f"\n📊 Summary:")
    print(f"  Total Processed: {stats.get('total_processed', 0)}")
    print(f"  Average Execution Time: {format_duration(stats.get('avg_execution_time', 0))}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Dynamic Claude Executor Pool")
    parser.add_argument('--queue-file', default='claude_execution_queue.json',
                        help='Path to queue file (default: claude_execution_queue.json)')
    parser.add_argument('--interval', type=int, default=10,
                        help='Update interval in seconds (default: 10)')
    parser.add_argument('--details', action='store_true',
                        help='Show queue details and exit')
    
    args = parser.parse_args()
    
    if args.details:
        asyncio.run(show_queue_details(args.queue_file))
    else:
        asyncio.run(monitor_pool(args.queue_file, args.interval))