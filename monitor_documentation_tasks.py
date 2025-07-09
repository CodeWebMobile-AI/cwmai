#!/usr/bin/env python3
"""Simple monitor to check if documentation tasks are being created."""

import subprocess
import sys
import time
import signal

def monitor_system(duration_minutes=5):
    """Monitor system output for documentation task creation."""
    
    print(f"📊 Monitoring system for {duration_minutes} minutes...")
    print("Looking for documentation task creation...\n")
    
    # Start the system
    cmd = [
        'python3', 'run_hybrid_continuous.py', 
        '--workers', '3'
    ]
    
    env = subprocess.os.environ.copy()
    
    # Export environment variables
    with open('.env.local', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                env[key] = value
    
    env['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
    
    # Start process
    process = subprocess.Popen(
        ['arch', '-arm64'] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    
    doc_tasks_found = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_minutes * 60:
            line = process.stdout.readline()
            if not line:
                break
                
            # Print the line for visibility
            print(line.rstrip())
            
            # Check for documentation tasks
            line_lower = line.lower()
            if any(term in line_lower for term in [
                'documentation_architecture',
                'create architecture.md',
                'create monetization_strategy.md',
                'create readme.md',
                'generate all documentation',
                'documentation task'
            ]):
                doc_tasks_found.append({
                    'time': time.time() - start_time,
                    'line': line.strip()
                })
                print(f"\n⚠️  DOCUMENTATION TASK DETECTED at {int(time.time() - start_time)}s!\n")
    
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")
    
    finally:
        # Terminate the process
        process.terminate()
        process.wait()
    
    # Report results
    print(f"\n\n{'='*60}")
    print(f"MONITORING COMPLETE - Duration: {int(time.time() - start_time)}s")
    print(f"{'='*60}\n")
    
    if doc_tasks_found:
        print(f"❌ FOUND {len(doc_tasks_found)} DOCUMENTATION TASKS:")
        for task in doc_tasks_found:
            print(f"  At {int(task['time'])}s: {task['line'][:100]}...")
        print("\n⚠️  System is still creating documentation tasks!")
    else:
        print("✅ NO DOCUMENTATION TASKS FOUND")
        print("  System correctly avoided creating documentation tasks")
    
    return len(doc_tasks_found) == 0

if __name__ == "__main__":
    success = monitor_system(5)  # Monitor for 5 minutes
    sys.exit(0 if success else 1)