#!/usr/bin/env python3
"""
Monitor the hybrid continuous system for exactly 10 minutes
"""

import subprocess
import signal
import sys
import time
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import re

class SystemMonitor:
    def __init__(self):
        self.start_time = None
        self.output_lines = []
        self.errors = []
        self.warnings = []
        self.issues = defaultdict(list)
        self.process = None
        
    def signal_handler(self, signum, frame):
        """Handle interruption signals"""
        print("\nReceived interrupt signal, cleaning up...")
        if self.process:
            self.process.terminate()
            time.sleep(2)
            if self.process.poll() is None:
                self.process.kill()
        sys.exit(0)
        
    def run_system_with_timeout(self, duration_minutes=10):
        """Run the system for specified duration"""
        # Set up signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print(f"Starting hybrid continuous system monitoring for {duration_minutes} minutes...")
        print("="*80)
        
        # Command to run
        cmd = [
            '/bin/bash', '-c',
            'export $(cat .env.local | grep -v "^#" | xargs) && '
            'PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai '
            'arch -arm64 /usr/bin/python3 run_hybrid_continuous.py --workers 10'
        ]
        
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(minutes=duration_minutes)
        
        # Start the process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"Process started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Will terminate at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80)
        
        # Monitor output in a separate thread
        output_thread = threading.Thread(target=self.capture_output)
        output_thread.daemon = True
        output_thread.start()
        
        # Wait for the duration
        elapsed = 0
        while elapsed < duration_minutes * 60:
            time.sleep(10)  # Check every 10 seconds
            elapsed = (datetime.now() - self.start_time).total_seconds()
            remaining = (duration_minutes * 60) - elapsed
            if remaining > 0:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Monitoring... {int(remaining)} seconds remaining", end='', flush=True)
            
            # Check if process ended prematurely
            if self.process.poll() is not None:
                print(f"\nProcess ended prematurely with code {self.process.poll()}")
                break
        
        print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] Time's up! Terminating process...")
        
        # Terminate the process
        if self.process.poll() is None:
            self.process.terminate()
            time.sleep(5)  # Give it time to cleanup
            
            if self.process.poll() is None:
                print("Process didn't terminate, killing it...")
                self.process.kill()
        
        # Wait for output thread to finish
        output_thread.join(timeout=5)
        
        print(f"\nMonitoring completed. Total lines captured: {len(self.output_lines)}")
        
    def capture_output(self):
        """Capture and analyze output from the process"""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                    
                line = line.strip()
                if line:
                    self.output_lines.append(line)
                    self.analyze_line(line)
        except Exception as e:
            print(f"Error capturing output: {e}")
    
    def analyze_line(self, line):
        """Analyze each line for issues"""
        line_lower = line.lower()
        
        # Detect errors
        if 'error' in line_lower or 'exception' in line_lower:
            self.errors.append(line)
            
            # Categorize specific errors
            if "'description'" in line:
                self.issues['description_errors'].append(line)
            elif 'keyerror' in line_lower:
                self.issues['key_errors'].append(line)
            elif 'deleted' in line_lower:
                self.issues['deleted_repo_errors'].append(line)
        
        # Detect warnings
        if 'warning' in line_lower:
            self.warnings.append(line)
        
        # Detect references to deleted repos
        deleted_repos = ['subtrack-ai', 'sub-wise', 'test-repo']
        for repo in deleted_repos:
            if repo in line_lower:
                self.issues['deleted_repo_references'].append(f"{repo}: {line}")
        
        # Detect work queue issues
        if 'no work found' in line_lower:
            self.issues['no_work_found'].append(line)
        elif 'at capacity' in line_lower:
            self.issues['capacity_issues'].append(line)
        elif 'idle' in line_lower and 'worker' in line_lower:
            self.issues['idle_workers'].append(line)
    
    def generate_report(self):
        """Generate a comprehensive report"""
        print("\n" + "="*80)
        print("SYSTEM MONITORING REPORT - 10 MINUTE RUN")
        print("="*80)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Output Lines: {len(self.output_lines)}")
        print(f"Total Errors: {len(self.errors)}")
        print(f"Total Warnings: {len(self.warnings)}")
        
        print("\n" + "-"*40)
        print("ISSUES SUMMARY:")
        print("-"*40)
        
        for issue_type, items in self.issues.items():
            if items:
                print(f"\n{issue_type.upper().replace('_', ' ')} ({len(items)} occurrences):")
                # Show first 5 examples
                for i, item in enumerate(items[:5]):
                    print(f"  [{i+1}] {item[:150]}...")
                if len(items) > 5:
                    print(f"  ... and {len(items) - 5} more")
        
        # Extract specific metrics from the output
        print("\n" + "-"*40)
        print("EXTRACTED METRICS:")
        print("-"*40)
        
        # Count worker activities
        worker_stats = defaultdict(int)
        task_completions = 0
        api_calls = 0
        
        for line in self.output_lines:
            if 'worker_' in line:
                match = re.search(r'worker_(\d+)', line)
                if match:
                    worker_id = match.group(1)
                    worker_stats[f'worker_{worker_id}'] += 1
            
            if 'completed task' in line.lower() or 'task completed' in line.lower():
                task_completions += 1
            
            if 'ai request' in line.lower():
                api_calls += 1
        
        print(f"Task Completions: {task_completions}")
        print(f"API Calls: {api_calls}")
        print(f"\nWorker Activity Count:")
        for worker, count in sorted(worker_stats.items()):
            print(f"  {worker}: {count} log entries")
        
        # Final recommendations
        print("\n" + "-"*40)
        print("KEY FINDINGS AND RECOMMENDATIONS:")
        print("-"*40)
        
        if self.issues['deleted_repo_references']:
            print("❌ CRITICAL: System still processing deleted repositories")
            
        if self.issues['description_errors']:
            print("❌ CRITICAL: Dynamic work generator 'description' errors persist")
            
        if self.issues['capacity_issues']:
            capacity_count = len(self.issues['capacity_issues'])
            print(f"⚠️  WARNING: Repository capacity issues ({capacity_count} occurrences)")
            
        if self.issues['idle_workers']:
            idle_count = len(set([line for line in self.issues['idle_workers']]))
            print(f"⚠️  WARNING: Workers frequently idle despite work queue")
        
        print("\n" + "="*80)

def main():
    monitor = SystemMonitor()
    
    try:
        # Run for exactly 10 minutes
        monitor.run_system_with_timeout(duration_minutes=10)
        
        # Generate report
        monitor.generate_report()
        
        # Save detailed output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'system_monitor_10min_{timestamp}.log', 'w') as f:
            f.write('\n'.join(monitor.output_lines))
        
        print(f"\nDetailed log saved to: system_monitor_10min_{timestamp}.log")
        
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        if monitor.process:
            monitor.process.terminate()

if __name__ == "__main__":
    main()