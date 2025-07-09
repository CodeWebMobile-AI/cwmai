#!/usr/bin/env python3
"""
Monitor the system for 3 minutes to verify the feedback loop fixes are working.
"""

import asyncio
import json
import time
import subprocess
import signal
import sys
from datetime import datetime
from pathlib import Path

# Track statistics
stats = {
    'start_time': None,
    'end_time': None,
    'issues_created': 0,
    'issues_completed': 0,
    'ai_work_generated': 0,
    'duplicate_prevented': 0,
    'claude_tasks_queued': 0,
    'claude_tasks_completed': 0,
    'work_discovery_cycles': 0,
    'mixed_work_cycles': 0,
    'pure_ai_cycles': 0
}

# Process handle
hybrid_process = None


def signal_handler(signum, frame):
    """Handle shutdown signal."""
    print("\n\nShutting down monitoring...")
    if hybrid_process:
        hybrid_process.terminate()
        hybrid_process.wait()
    
    # Print final statistics
    print_final_stats()
    sys.exit(0)


def print_final_stats():
    """Print final statistics."""
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    duration = (stats['end_time'] or time.time()) - stats['start_time']
    print(f"Test Duration: {duration:.1f} seconds")
    print(f"Issues Created: {stats['issues_created']}")
    print(f"Issues Completed (claude-implemented): {stats['issues_completed']}")
    print(f"AI Work Generated: {stats['ai_work_generated']}")
    print(f"Duplicates Prevented: {stats['duplicate_prevented']}")
    print(f"Claude Tasks Queued: {stats['claude_tasks_queued']}")
    print(f"Claude Tasks Completed: {stats['claude_tasks_completed']}")
    print(f"Work Discovery Cycles: {stats['work_discovery_cycles']}")
    print(f"Mixed Work Cycles (issues + AI): {stats['mixed_work_cycles']}")
    print(f"Pure AI Work Cycles: {stats['pure_ai_cycles']}")
    
    # Check success criteria
    print("\n" + "-" * 60)
    print("SUCCESS CRITERIA CHECK:")
    if stats['claude_tasks_completed'] > 0:
        print("✅ Claude is completing tasks")
    else:
        print("❌ No Claude tasks completed")
    
    if stats['issues_completed'] > 0:
        print("✅ Issues are being marked as completed")
    else:
        print("⚠️  No issues marked as completed (may need more time)")
    
    if stats['ai_work_generated'] > 0:
        print("✅ AI work is being generated")
    else:
        print("❌ No AI work generated")
    
    if stats['mixed_work_cycles'] > 0:
        print("✅ Work mixing (60/40) is happening")
    else:
        print("⚠️  No mixed work cycles detected")
    
    if stats['duplicate_prevented'] > 0:
        print("✅ Duplicate prevention is working")
    else:
        print("ℹ️  No duplicates encountered")


async def monitor_logs(process):
    """Monitor the process output for key events."""
    try:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
                
            # Print the line
            print(line.rstrip())
            
            # Parse statistics from logs
            if "Creating local issue for:" in line:
                stats['issues_created'] += 1
            elif "Marked issue #" in line and "as completed" in line:
                stats['issues_completed'] += 1
            elif "Claude task already completed for issue" in line:
                stats['duplicate_prevented'] += 1
            elif "Claude task" in line and "queued" in line:
                stats['claude_tasks_queued'] += 1
            elif "Task" in line and "completed in" in line and "s" in line:
                stats['claude_tasks_completed'] += 1
            elif "Discovering work" in line:
                stats['work_discovery_cycles'] += 1
            elif "Mixing work:" in line:
                stats['mixed_work_cycles'] += 1
                # Also count AI work from the mix
                if "AI-generated tasks" in line:
                    import re
                    match = re.search(r'(\d+) AI-generated tasks', line)
                    if match:
                        stats['ai_work_generated'] += int(match.group(1))
            elif "No unimplemented issues found, will use AI-powered discovery" in line:
                stats['pure_ai_cycles'] += 1
            elif "Discovered" in line and "work items:" in line:
                # Count AI-generated work
                import re
                match = re.search(r'Discovered (\d+) work items', line)
                if match and stats['pure_ai_cycles'] > stats.get('last_pure_ai_cycle', 0):
                    stats['ai_work_generated'] += int(match.group(1))
                    stats['last_pure_ai_cycle'] = stats['pure_ai_cycles']
                    
    except Exception as e:
        print(f"Monitor error: {e}")


async def check_issue_states():
    """Periodically check issue states to verify completion."""
    await asyncio.sleep(30)  # Wait 30 seconds before first check
    
    while True:
        try:
            # Check claude-implemented labels
            from scripts.local_github_simulator import LocalGitHubSimulator
            simulator = LocalGitHubSimulator()
            
            completed_count = 0
            total_count = 0
            
            repos = simulator.list_repositories()
            for repo in repos:
                issues = simulator.get_issues(repo.full_name, state="all")
                for issue in issues:
                    total_count += 1
                    if 'claude-implemented' in issue.labels:
                        completed_count += 1
            
            print(f"\n📊 Issue State Check: {completed_count}/{total_count} issues have claude-implemented label\n")
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"State check error: {e}")
            await asyncio.sleep(60)


async def main():
    """Run the monitoring."""
    global hybrid_process
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("3-MINUTE SYSTEM MONITORING TEST")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("Monitoring for work generation fixes...")
    print("-" * 60)
    
    stats['start_time'] = time.time()
    
    # Start the hybrid orchestrator
    cmd = [
        sys.executable,
        "run_hybrid_continuous.py",
        "--workers", "10"
    ]
    
    print(f"Starting command: {' '.join(cmd)}")
    
    hybrid_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Create monitoring tasks
    monitor_task = asyncio.create_task(monitor_logs(hybrid_process))
    state_check_task = asyncio.create_task(check_issue_states())
    
    # Run for 3 minutes
    try:
        await asyncio.sleep(180)  # 3 minutes
    except KeyboardInterrupt:
        pass
    
    stats['end_time'] = time.time()
    
    # Shutdown
    print("\n\nTime's up! Shutting down...")
    hybrid_process.terminate()
    await asyncio.sleep(2)
    
    if hybrid_process.poll() is None:
        hybrid_process.kill()
    
    # Cancel monitoring tasks
    monitor_task.cancel()
    state_check_task.cancel()
    
    # Print statistics
    print_final_stats()


if __name__ == "__main__":
    asyncio.run(main())