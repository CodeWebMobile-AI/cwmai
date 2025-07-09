#!/usr/bin/env python3
"""
Monitor the hybrid continuous system for 6 minutes and analyze performance.
"""

import asyncio
import subprocess
import time
import re
import json
from datetime import datetime
from collections import defaultdict

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'workers': defaultdict(int),
            'tasks_completed': 0,
            'queue_status': [],
            'research_tasks': 0,
            'claude_executions': 0,
            'errors': 0,
            'status_updates': []
        }
        self.patterns = {
            'worker_status': r'Status: \{\'total\': (\d+), \'idle\': (\d+), \'working\': (\d+), \'error\': (\d+)\} workers, (\d+) completed',
            'task_completed': r'Worker (\w+) COMPLETED: (.+) in ([\d.]+)s',
            'research_task': r'(MARKET_RESEARCH|PORTFOLIO_ANALYSIS|PROJECT_CREATION|research)',
            'claude_execution': r'Claude executor',
            'queue_status': r'Queue: (\d+) queued, (\d+) processing',
            'discovering_work': r'Discovering work \(max: (\d+), current load: (\d+)\)',
            'work_discovered': r'Discovered (\d+) work items',
            'error': r'(ERROR|error|Error|Failed|failed)',
        }
        
    def parse_line(self, line):
        """Parse a log line and extract metrics."""
        # Worker status
        match = re.search(self.patterns['worker_status'], line)
        if match:
            total, idle, working, error, completed = map(int, match.groups())
            self.metrics['workers']['total'] = total
            self.metrics['workers']['idle'] = idle
            self.metrics['workers']['working'] = working
            self.metrics['workers']['error'] = error
            self.metrics['tasks_completed'] = completed
            
            idle_percentage = (idle / total * 100) if total > 0 else 0
            self.metrics['status_updates'].append({
                'time': time.time() - self.start_time,
                'idle_percentage': idle_percentage,
                'working': working,
                'completed': completed
            })
        
        # Task completion
        if 'COMPLETED:' in line:
            self.metrics['tasks_completed'] += 1
            if any(pattern in line for pattern in ['MARKET_RESEARCH', 'PORTFOLIO_ANALYSIS', 'PROJECT_CREATION']):
                self.metrics['research_tasks'] += 1
        
        # Claude execution
        if 'Claude executor' in line and 'executing' in line:
            self.metrics['claude_executions'] += 1
        
        # Work discovery
        match = re.search(self.patterns['work_discovered'], line)
        if match:
            discovered = int(match.group(1))
            print(f"[MONITOR] Work discovered: {discovered} items")
        
        # Errors
        if re.search(self.patterns['error'], line) and 'NotOpenSSLWarning' not in line:
            self.metrics['errors'] += 1
    
    def get_summary(self):
        """Get summary of metrics."""
        duration = time.time() - self.start_time
        
        # Calculate averages
        avg_idle = 0
        if self.metrics['status_updates']:
            avg_idle = sum(s['idle_percentage'] for s in self.metrics['status_updates']) / len(self.metrics['status_updates'])
        
        return {
            'duration_seconds': duration,
            'total_workers': self.metrics['workers']['total'],
            'average_idle_percentage': avg_idle,
            'tasks_completed': self.metrics['tasks_completed'],
            'research_tasks_completed': self.metrics['research_tasks'],
            'claude_executions': self.metrics['claude_executions'],
            'errors': self.metrics['errors'],
            'status_update_count': len(self.metrics['status_updates'])
        }

async def monitor_system():
    """Run and monitor the system for 6 minutes."""
    monitor = SystemMonitor()
    
    # Start the system
    cmd = [
        'arch', '-arm64', '/usr/bin/python3',
        'run_hybrid_continuous.py',
        '--workers', '10'
    ]
    
    # Set environment
    env = {}
    with open('.env.local', 'r') as f:
        for line in f:
            if not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                env[key] = value
    
    env['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
    env['DISCOVERY_INTERVAL_SECONDS'] = '30'  # Faster discovery
    
    print("🚀 Starting hybrid continuous system with monitoring...")
    print("=" * 80)
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    
    # Monitor for 6 minutes
    end_time = time.time() + 360  # 6 minutes
    last_summary_time = time.time()
    
    while time.time() < end_time:
        try:
            # Read output with timeout
            line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
            if line:
                line = line.decode('utf-8').strip()
                if line:
                    monitor.parse_line(line)
                    
                    # Show important lines
                    if any(keyword in line for keyword in ['Status:', 'COMPLETED:', 'Discovered', 'ERROR']):
                        elapsed = int(time.time() - monitor.start_time)
                        print(f"[{elapsed}s] {line[:120]}...")
            
            # Print summary every 30 seconds
            if time.time() - last_summary_time >= 30:
                summary = monitor.get_summary()
                print(f"\n[{int(summary['duration_seconds'])}s] Summary:")
                print(f"  Workers: {summary['total_workers']} total, {summary['average_idle_percentage']:.1f}% avg idle")
                print(f"  Completed: {summary['tasks_completed']} tasks ({summary['research_tasks_completed']} research)")
                print(f"  Claude: {summary['claude_executions']} executions")
                print(f"  Errors: {summary['errors']}")
                print()
                last_summary_time = time.time()
                
        except asyncio.TimeoutError:
            continue
    
    # Stop the process
    print("\n⏹️  Stopping system...")
    process.terminate()
    await process.wait()
    
    # Final summary
    final_summary = monitor.get_summary()
    
    print("\n" + "=" * 80)
    print("📊 FINAL 6-MINUTE TEST RESULTS")
    print("=" * 80)
    print(f"Duration: {final_summary['duration_seconds']:.1f} seconds")
    print(f"Workers: {final_summary['total_workers']}")
    print(f"Average Idle: {final_summary['average_idle_percentage']:.1f}%")
    print(f"Tasks Completed: {final_summary['tasks_completed']}")
    print(f"Research Tasks: {final_summary['research_tasks_completed']}")
    print(f"Claude Executions: {final_summary['claude_executions']}")
    print(f"Errors: {final_summary['errors']}")
    
    # Success criteria
    print("\n✅ Success Criteria:")
    criteria = []
    criteria.append(("Average idle < 50%", final_summary['average_idle_percentage'] < 50))
    criteria.append(("Completed >= 10 tasks", final_summary['tasks_completed'] >= 10))
    criteria.append(("Research tasks created", final_summary['research_tasks_completed'] > 0))
    
    all_passed = True
    for name, passed in criteria:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    # Save results
    with open('monitor_6min_results.json', 'w') as f:
        json.dump({
            'summary': final_summary,
            'metrics': dict(monitor.metrics),
            'timestamp': datetime.now().isoformat(),
            'success': all_passed
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: monitor_6min_results.json")
    
    if all_passed:
        print("\n🎉 All tests PASSED!")
    else:
        print("\n⚠️  Some tests FAILED.")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = asyncio.run(monitor_system())
    sys.exit(0 if success else 1)