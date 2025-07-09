#!/usr/bin/env python3
"""
Test script to run the system for 3 minutes and check for documentation tasks
"""
import time
import subprocess
import signal
import sys
import os

def signal_handler(signum, frame):
    print("\nTest completed or interrupted")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Export environment variables
env = os.environ.copy()
with open('.env.local', 'r') as f:
    for line in f:
        if not line.startswith('#') and '=' in line:
            key, value = line.strip().split('=', 1)
            env[key] = value

env['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'

print("Starting 3-minute test run...")
print("Monitoring for documentation tasks...")
print("-" * 60)

# Start the process
process = subprocess.Popen(
    ['python3', 'run_hybrid_continuous.py'],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

start_time = time.time()
doc_tasks_found = []
setup_tasks_found = []

try:
    while time.time() - start_time < 180:  # 3 minutes
        line = process.stdout.readline()
        if line:
            print(line, end='')
            
            # Check for documentation tasks
            if any(word in line.lower() for word in ['documentation', 'readme', 'docs', 'api_documentation', 'developer_guide', 'admin_guide']):
                if 'task' in line.lower() or 'creating' in line.lower() or 'generated' in line.lower():
                    doc_tasks_found.append(line.strip())
            
            # Check for initial project setup
            if 'setup' in line.lower() and ('project' in line.lower() or 'initial' in line.lower()):
                setup_tasks_found.append(line.strip())
                
except KeyboardInterrupt:
    pass
finally:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

print("\n" + "=" * 60)
print("TEST RESULTS:")
print("=" * 60)
print(f"Test duration: {time.time() - start_time:.1f} seconds")
print(f"\nDocumentation tasks found: {len(doc_tasks_found)}")
if doc_tasks_found:
    print("Documentation task lines:")
    for task in doc_tasks_found[:5]:  # Show first 5
        print(f"  - {task}")
        
print(f"\nSetup tasks found: {len(setup_tasks_found)}")
if setup_tasks_found:
    print("Setup task lines:")
    for task in setup_tasks_found[:5]:  # Show first 5
        print(f"  - {task}")

if doc_tasks_found:
    print("\n⚠️  WARNING: Documentation tasks are still being created!")
else:
    print("\n✅ SUCCESS: No documentation tasks were created!")