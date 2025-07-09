#!/usr/bin/env python3
"""
Monitor the running hybrid system and collect statistics.
"""

import time
import re
from datetime import datetime
from collections import defaultdict
import subprocess

def monitor_log():
    """Monitor the hybrid continuous log file."""
    print(f"\n{'='*80}")
    print(f"MONITORING HYBRID CONTINUOUS SYSTEM - Started at {datetime.now()}")
    print(f"{'='*80}\n")
    
    # Track metrics
    start_time = time.time()
    errors = defaultdict(int)
    warnings = defaultdict(int)
    workers_active = 0
    total_completed = 0
    work_types = defaultdict(int)
    error_messages = []
    warning_messages = []
    
    # Follow the log
    process = subprocess.Popen(
        ['tail', '-f', '../hybrid_continuous.log'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    try:
        last_update = time.time()
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            # Track errors
            if 'ERROR' in line:
                if 'Error executing work:' in line:
                    errors['work_execution'] += 1
                elif 'Failed to parse' in line:
                    errors['parse_failures'] += 1
                elif 'Error gathering context' in line:
                    errors['context_gathering'] += 1
                else:
                    errors['other'] += 1
                error_messages.append(line.strip())
            
            # Track warnings
            if 'WARNING' in line:
                if 'not in processing' in line:
                    warnings['queue_warnings'] += 1
                elif 'Failed to parse AI response' in line:
                    warnings['ai_parse_failures'] += 1
                else:
                    warnings['other'] += 1
                warning_messages.append(line.strip())
            
            # Track completions
            if 'completed, ' in line and 'work/hour' in line:
                match = re.search(r'(\d+) completed, ([\d.]+) work/hour', line)
                if match:
                    total_completed = int(match.group(1))
                    work_per_hour = float(match.group(2))
            
            # Track worker status
            if 'Status:' in line:
                match = re.search(r"'total': (\d+), 'idle': (\d+), 'working': (\d+)", line)
                if match:
                    workers_active = int(match.group(3))
            
            # Track work types
            if 'executing:' in line:
                if 'Document' in line:
                    work_types['documentation'] += 1
                elif 'Test' in line or 'test' in line:
                    work_types['testing'] += 1
                elif 'Implement' in line:
                    work_types['implementation'] += 1
                elif 'Optimize' in line:
                    work_types['optimization'] += 1
                elif 'Research' in line:
                    work_types['research'] += 1
                else:
                    work_types['other'] += 1
            
            # Print status update every 30 seconds
            if time.time() - last_update > 30:
                elapsed = time.time() - start_time
                print(f"\n{'='*60}")
                print(f"STATUS UPDATE - {elapsed/60:.1f} minutes elapsed")
                print(f"{'='*60}")
                print(f"Workers active: {workers_active}")
                print(f"Total completed: {total_completed}")
                print(f"\nWork Types:")
                for wtype, count in sorted(work_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {wtype}: {count}")
                print(f"\nErrors: {sum(errors.values())}")
                for etype, count in errors.items():
                    if count > 0:
                        print(f"  {etype}: {count}")
                print(f"\nWarnings: {sum(warnings.values())}")
                for wtype, count in warnings.items():
                    if count > 0:
                        print(f"  {wtype}: {count}")
                print(f"{'='*60}")
                last_update = time.time()
                
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        # Don't terminate the process - let it complete naturally
        # process.terminate() - REMOVED to allow long-running tasks
        print("\nMonitor exiting - hybrid system will continue running")
        print("Use 'ps aux | grep run_hybrid_continuous.py' to check if still running")
        
        # Print final summary
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY - Total runtime: {elapsed/60:.1f} minutes")
        print(f"{'='*80}")
        print(f"\nTotal work completed: {total_completed}")
        print(f"Workers active at end: {workers_active}")
        
        print(f"\nWork Type Distribution:")
        total_work = sum(work_types.values())
        for wtype, count in sorted(work_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count/total_work*100) if total_work > 0 else 0
            print(f"  {wtype}: {count} ({percentage:.1f}%)")
        
        print(f"\nError Summary ({sum(errors.values())} total):")
        for etype, count in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {etype}: {count}")
        
        print(f"\nWarning Summary ({sum(warnings.values())} total):")
        for wtype, count in sorted(warnings.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {wtype}: {count}")
        
        print(f"\nRecent Error Messages (last 5):")
        for msg in error_messages[-5:]:
            print(f"  {msg[:100]}...")
            
        print(f"\nPerformance:")
        if elapsed > 0:
            print(f"  Average work/minute: {total_completed/(elapsed/60):.2f}")
            print(f"  Average work/hour: {total_completed/(elapsed/3600):.2f}")
        
        print(f"{'='*80}\n")

if __name__ == "__main__":
    monitor_log()