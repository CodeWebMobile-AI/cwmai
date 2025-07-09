#!/usr/bin/env python3
"""
Monitor the hybrid continuous system for 10 minutes and log all issues
"""

import subprocess
import threading
import time
import signal
import sys
import re
from datetime import datetime, timedelta
from collections import defaultdict
import json

class SystemMonitor:
    def __init__(self, duration_minutes=10):
        self.duration_minutes = duration_minutes
        self.start_time = None
        self.end_time = None
        self.process = None
        self.output_lines = []
        self.errors = []
        self.warnings = []
        self.issues = defaultdict(list)
        self.stats = {
            'total_lines': 0,
            'error_count': 0,
            'warning_count': 0,
            'tasks_assigned': 0,
            'tasks_completed': 0,
            'api_calls': 0,
            'worker_issues': defaultdict(int),
            'repository_access': defaultdict(int),
            'repeated_errors': defaultdict(int)
        }
        
    def start_system(self):
        """Start the hybrid continuous system"""
        print(f"Starting system monitoring for {self.duration_minutes} minutes...")
        print("="*60)
        
        # Export environment variables and run the system
        cmd = [
            '/bin/bash', '-c',
            'export $(cat .env.local | grep -v "^#" | xargs) && '
            'PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai '
            'arch -arm64 /usr/bin/python3 run_hybrid_continuous.py --workers 10'
        ]
        
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_output)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for duration
        print(f"Monitoring started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Will stop at {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*60)
        
        time.sleep(self.duration_minutes * 60)
        
        # Stop the system
        print("\nStopping system...")
        self.process.terminate()
        time.sleep(5)  # Give it time to cleanup
        
        if self.process.poll() is None:
            print("Force killing process...")
            self.process.kill()
            
        monitor_thread.join(timeout=5)
        
    def monitor_output(self):
        """Monitor and analyze system output"""
        for line in iter(self.process.stdout.readline, ''):
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
                
            self.output_lines.append(line)
            self.stats['total_lines'] += 1
            
            # Real-time analysis
            self.analyze_line(line)
            
            # Print significant lines
            if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'warning', 'critical']):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
    
    def analyze_line(self, line):
        """Analyze a single line of output"""
        line_lower = line.lower()
        
        # Error detection
        if 'error' in line_lower or 'exception' in line_lower:
            self.errors.append(line)
            self.stats['error_count'] += 1
            
            # Categorize errors
            if 'keyerror' in line_lower:
                self.issues['key_errors'].append(line)
            elif 'attributeerror' in line_lower:
                self.issues['attribute_errors'].append(line)
            elif 'connection' in line_lower:
                self.issues['connection_errors'].append(line)
            elif 'timeout' in line_lower:
                self.issues['timeout_errors'].append(line)
            elif 'description' in line_lower:
                self.issues['description_errors'].append(line)
                
            # Track repeated errors
            error_pattern = re.sub(r'\d+', 'N', line)  # Replace numbers with N
            self.stats['repeated_errors'][error_pattern] += 1
        
        # Warning detection
        if 'warning' in line_lower or 'warn' in line_lower:
            self.warnings.append(line)
            self.stats['warning_count'] += 1
        
        # Task tracking
        if 'assigning task' in line_lower or 'assigned task' in line_lower:
            self.stats['tasks_assigned'] += 1
        elif 'completed task' in line_lower or 'task completed' in line_lower:
            self.stats['tasks_completed'] += 1
        
        # API call tracking
        if 'ai request' in line_lower or 'api call' in line_lower:
            self.stats['api_calls'] += 1
        
        # Worker issues
        worker_match = re.search(r'worker[_-]?(\d+)', line_lower)
        if worker_match and ('error' in line_lower or 'failed' in line_lower):
            worker_id = f"worker_{worker_match.group(1)}"
            self.stats['worker_issues'][worker_id] += 1
        
        # Repository access
        repo_match = re.search(r'CodeWebMobile-AI/([a-zA-Z0-9-]+)', line)
        if repo_match:
            repo_name = repo_match.group(1)
            self.stats['repository_access'][repo_name] += 1
            
            # Check for deleted repos
            if repo_name.lower() in ['subtrack-ai', 'sub-wise', 'test-repo']:
                self.issues['deleted_repo_access'].append(f"{line} [DELETED REPO: {repo_name}]")
        
        # Specific issue patterns
        if 'no work found' in line_lower:
            self.issues['no_work_found'].append(line)
        elif 'duplicate' in line_lower:
            self.issues['duplicate_work'].append(line)
        elif 'skipping' in line_lower:
            self.issues['skipped_items'].append(line)
        elif 'retry' in line_lower or 'retrying' in line_lower:
            self.issues['retries'].append(line)
        elif 'claude' in line_lower and 'timeout' in line_lower:
            self.issues['claude_timeouts'].append(line)
        elif 'rate limit' in line_lower:
            self.issues['rate_limits'].append(line)
    
    def generate_report(self):
        """Generate comprehensive report of findings"""
        report = []
        report.append("="*80)
        report.append("SYSTEM MONITORING REPORT")
        report.append("="*80)
        report.append(f"Duration: {self.duration_minutes} minutes")
        report.append(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Output Lines: {self.stats['total_lines']}")
        report.append("")
        
        # Summary Statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-"*40)
        report.append(f"Total Errors: {self.stats['error_count']}")
        report.append(f"Total Warnings: {self.stats['warning_count']}")
        report.append(f"Tasks Assigned: {self.stats['tasks_assigned']}")
        report.append(f"Tasks Completed: {self.stats['tasks_completed']}")
        report.append(f"API Calls Made: {self.stats['api_calls']}")
        report.append("")
        
        # Worker Issues
        if self.stats['worker_issues']:
            report.append("WORKER ISSUES:")
            report.append("-"*40)
            for worker, count in sorted(self.stats['worker_issues'].items()):
                report.append(f"{worker}: {count} issues")
            report.append("")
        
        # Repository Access
        report.append("REPOSITORY ACCESS FREQUENCY:")
        report.append("-"*40)
        for repo, count in sorted(self.stats['repository_access'].items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"{repo}: {count} times")
        report.append("")
        
        # Issues by Category
        report.append("ISSUES BY CATEGORY:")
        report.append("-"*40)
        
        for category, items in self.issues.items():
            if items:
                report.append(f"\n{category.upper().replace('_', ' ')} ({len(items)} occurrences):")
                # Show first 5 examples
                for item in items[:5]:
                    report.append(f"  - {item[:150]}...")
                if len(items) > 5:
                    report.append(f"  ... and {len(items) - 5} more")
        
        # Most Repeated Errors
        report.append("\nMOST REPEATED ERRORS:")
        report.append("-"*40)
        repeated = sorted(self.stats['repeated_errors'].items(), key=lambda x: x[1], reverse=True)[:10]
        for error_pattern, count in repeated:
            if count > 1:
                report.append(f"{count}x: {error_pattern[:100]}...")
        
        # Critical Issues
        report.append("\nCRITICAL ISSUES AND FLAWS:")
        report.append("-"*40)
        
        critical_issues = []
        
        # Check for deleted repository access
        if self.issues['deleted_repo_access']:
            critical_issues.append("❌ System still attempting to access deleted repositories!")
        
        # Check for high error rate
        if self.stats['total_lines'] > 0:
            error_rate = self.stats['error_count'] / self.stats['total_lines']
            if error_rate > 0.05:  # More than 5% error rate
                critical_issues.append(f"❌ High error rate: {error_rate:.1%}")
        
        # Check for task completion rate
        if self.stats['tasks_assigned'] > 0:
            completion_rate = self.stats['tasks_completed'] / self.stats['tasks_assigned']
            if completion_rate < 0.5:  # Less than 50% completion
                critical_issues.append(f"❌ Low task completion rate: {completion_rate:.1%}")
        
        # Check for specific error patterns
        if self.issues['description_errors']:
            critical_issues.append("❌ Dynamic work generator still having description errors")
        
        if self.issues['timeout_errors']:
            critical_issues.append(f"❌ Timeout errors detected: {len(self.issues['timeout_errors'])} occurrences")
        
        if self.issues['rate_limits']:
            critical_issues.append(f"❌ Rate limiting issues: {len(self.issues['rate_limits'])} occurrences")
        
        if critical_issues:
            for issue in critical_issues:
                report.append(issue)
        else:
            report.append("✅ No critical issues detected")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        report.append("-"*40)
        
        if self.issues['no_work_found']:
            report.append("- Consider adjusting work generation parameters")
        
        if self.issues['duplicate_work']:
            report.append("- Review duplicate detection logic")
        
        if self.stats['worker_issues']:
            report.append("- Investigate worker-specific errors")
        
        if self.issues['claude_timeouts']:
            report.append("- Consider increasing Claude execution timeouts")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_detailed_log(self):
        """Save detailed log for further analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full output
        with open(f'system_monitor_output_{timestamp}.log', 'w') as f:
            f.write("\n".join(self.output_lines))
        
        # Save analysis
        analysis = {
            'duration_minutes': self.duration_minutes,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now().isoformat(),
            'stats': dict(self.stats),
            'issues': dict(self.issues),
            'errors': self.errors[:100],  # First 100 errors
            'warnings': self.warnings[:100]  # First 100 warnings
        }
        
        with open(f'system_monitor_analysis_{timestamp}.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return timestamp

def main():
    monitor = SystemMonitor(duration_minutes=10)
    
    try:
        monitor.start_system()
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        if monitor.process:
            monitor.process.terminate()
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        if monitor.process:
            monitor.process.terminate()
    
    # Generate and display report
    report = monitor.generate_report()
    print("\n" + report)
    
    # Save detailed logs
    timestamp = monitor.save_detailed_log()
    print(f"\nDetailed logs saved:")
    print(f"  - system_monitor_output_{timestamp}.log")
    print(f"  - system_monitor_analysis_{timestamp}.json")

if __name__ == "__main__":
    main()