#!/usr/bin/env python3
"""
Monitor the hybrid continuous AI system for 5 minutes to verify fixes.
"""

import asyncio
import subprocess
import time
import signal
import sys
import json
import os
from datetime import datetime, timezone
from collections import defaultdict


class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.process = None
        self.output_lines = []
        self.worker_activity = defaultdict(lambda: {'status': 'idle', 'tasks': []})
        self.repository_tasks = defaultdict(int)
        self.error_count = 0
        self.warning_count = 0
        self.task_completions = 0
        self.worker_status_changes = []
        self.deleted_repo_references = 0
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n🛑 Monitoring interrupted by user")
        if self.process:
            self.process.terminate()
        self.print_summary()
        sys.exit(0)
    
    async def run_system_with_timeout(self, duration_minutes=5):
        """Run the system for specified duration."""
        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print(f"🚀 Starting hybrid continuous AI system monitoring for {duration_minutes} minutes...")
        print("="*80)
        
        # Start the system with proper environment
        env = os.environ.copy()
        env['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
        
        # Export environment variables
        export_process = subprocess.run(
            ['bash', '-c', 'export $(cat .env.local | grep -v "^#" | xargs)'],
            env=env
        )
        
        # Start the actual system
        self.process = subprocess.Popen(
            ['arch', '-arm64', '/usr/bin/python3', 'run_hybrid_continuous.py', '--workers', '10'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Monitor output
        end_time = time.time() + (duration_minutes * 60)
        last_status_check = time.time()
        
        while time.time() < end_time:
            if self.process.poll() is not None:
                print("\n⚠️  Process exited unexpectedly!")
                break
                
            # Read output
            line = self.process.stdout.readline()
            if line:
                self.output_lines.append(line.strip())
                self.analyze_line(line.strip())
                
                # Print important lines
                if any(keyword in line for keyword in ['ERROR', 'WARNING', 'Worker', 'Completed', 'idle', 'working']):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] {line.strip()}")
            
            # Periodic status check every 30 seconds
            if time.time() - last_status_check > 30:
                await self.check_system_status()
                last_status_check = time.time()
            
            await asyncio.sleep(0.1)
        
        # Stop the process
        if self.process:
            self.process.terminate()
            self.process.wait()
        
        print("\n" + "="*80)
        print("✅ Monitoring completed")
        self.print_summary()
    
    def analyze_line(self, line):
        """Analyze each output line for important information."""
        # Track errors and warnings
        if 'ERROR' in line:
            self.error_count += 1
        if 'WARNING' in line:
            self.warning_count += 1
            
        # Track deleted repository references
        deleted_repos = ['subtrack-ai', 'test-repo', 'sub-wise']
        if any(repo in line for repo in deleted_repos):
            self.deleted_repo_references += 1
        
        # Track worker status changes
        if 'Worker' in line and ('idle' in line or 'working' in line):
            # Extract worker ID and status
            if 'worker_' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'worker_' in part:
                        worker_id = part.strip(':')
                        if 'idle' in line:
                            self.worker_activity[worker_id]['status'] = 'idle'
                        elif 'working' in line:
                            self.worker_activity[worker_id]['status'] = 'working'
                        self.worker_status_changes.append((time.time(), worker_id, self.worker_activity[worker_id]['status']))
        
        # Track task completions
        if 'Completed task' in line or 'Task completed' in line:
            self.task_completions += 1
        
        # Track repository activity
        if 'claimed task' in line:
            # Extract repository name
            for repo in ['brand-ai', 'chronoai-time-tracker', 'lingua-ai', 'mindful-ai-platform', 
                        'returnwise-portal', 'invoice-zen', 'brandify-ai', 'scholarsynapse']:
                if repo in line:
                    self.repository_tasks[repo] += 1
    
    async def check_system_status(self):
        """Check current system status from files."""
        print("\n📊 Checking system status...")
        
        # Check Claude execution queue
        queue_file = 'claude_execution_queue.json'
        if os.path.exists(queue_file):
            with open(queue_file, 'r') as f:
                queue_data = json.load(f)
            
            pending = len(queue_data.get('pending', []))
            processing = len(queue_data.get('processing', {}))
            
            # Count by repository
            repo_counts = defaultdict(int)
            for task in queue_data.get('processing', {}).values():
                repo = task.get('repository', 'unknown')
                repo_counts[repo] += 1
            
            print(f"  Claude Queue: {pending} pending, {processing} processing")
            if repo_counts:
                print(f"  Active repos: {dict(repo_counts)}")
        
        # Check orchestrator state
        orch_file = 'local_repos/state/continuous_orchestrator_state.json'
        if os.path.exists(orch_file):
            with open(orch_file, 'r') as f:
                orch_data = json.load(f)
            
            workers = orch_data.get('workers', {})
            idle_count = sum(1 for w in workers.values() if w['status'] == 'idle')
            working_count = len(workers) - idle_count
            
            print(f"  Workers: {working_count} working, {idle_count} idle (of {len(workers)} total)")
    
    def print_summary(self):
        """Print monitoring summary."""
        duration = time.time() - self.start_time
        minutes = duration / 60
        
        print("\n" + "="*80)
        print("📈 MONITORING SUMMARY")
        print("="*80)
        print(f"Duration: {minutes:.1f} minutes")
        print(f"Total output lines: {len(self.output_lines)}")
        print(f"Errors: {self.error_count}")
        print(f"Warnings: {self.warning_count}")
        print(f"Task completions: {self.task_completions}")
        print(f"Deleted repo references: {self.deleted_repo_references}")
        
        # Worker utilization
        print("\n👷 Worker Activity:")
        total_changes = len(self.worker_status_changes)
        if total_changes > 0:
            # Calculate time spent working vs idle
            worker_times = defaultdict(lambda: {'working': 0, 'idle': 0})
            
            for i in range(len(self.worker_status_changes)):
                timestamp, worker_id, status = self.worker_status_changes[i]
                
                # Calculate duration until next change or end
                if i < len(self.worker_status_changes) - 1:
                    next_timestamp = self.worker_status_changes[i + 1][0]
                else:
                    next_timestamp = time.time()
                
                duration = next_timestamp - timestamp
                worker_times[worker_id][status] += duration
            
            # Print worker utilization
            for worker_id in sorted(worker_times.keys()):
                total_time = worker_times[worker_id]['working'] + worker_times[worker_id]['idle']
                if total_time > 0:
                    utilization = (worker_times[worker_id]['working'] / total_time) * 100
                    print(f"  {worker_id}: {utilization:.1f}% utilization")
        
        # Repository activity
        print("\n📁 Repository Task Distribution:")
        if self.repository_tasks:
            for repo, count in sorted(self.repository_tasks.items(), key=lambda x: x[1], reverse=True):
                print(f"  {repo}: {count} tasks")
        else:
            print("  No repository task assignments detected")
        
        # Key findings
        print("\n🔍 Key Findings:")
        
        # Check if workers picked up work
        if self.task_completions > 0:
            print(f"  ✅ Workers completed {self.task_completions} tasks")
        else:
            print("  ⚠️  No task completions detected")
        
        # Check for deleted repo issues
        if self.deleted_repo_references > 0:
            print(f"  ⚠️  Found {self.deleted_repo_references} references to deleted repositories")
        else:
            print("  ✅ No references to deleted repositories found")
        
        # Worker utilization
        if total_changes > 10:  # Enough data to analyze
            avg_utilization = sum(worker_times[w]['working'] / (worker_times[w]['working'] + worker_times[w]['idle']) * 100 
                                 for w in worker_times if worker_times[w]['working'] + worker_times[w]['idle'] > 0) / len(worker_times)
            if avg_utilization > 50:
                print(f"  ✅ Good worker utilization: {avg_utilization:.1f}% average")
            else:
                print(f"  ⚠️  Low worker utilization: {avg_utilization:.1f}% average")
        
        # Repository coverage
        expected_repos = ['brand-ai', 'chronoai-time-tracker', 'lingua-ai', 'mindful-ai-platform', 
                         'returnwise-portal', 'invoice-zen', 'brandify-ai', 'scholarsynapse']
        covered_repos = set(self.repository_tasks.keys())
        uncovered = set(expected_repos) - covered_repos
        
        if uncovered:
            print(f"  ⚠️  Repositories with no task assignments: {', '.join(uncovered)}")
        else:
            print("  ✅ All expected repositories received task assignments")


async def main():
    """Main function."""
    monitor = SystemMonitor()
    await monitor.run_system_with_timeout(duration_minutes=5)


if __name__ == "__main__":
    asyncio.run(main())