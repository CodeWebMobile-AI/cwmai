#!/usr/bin/env python3
"""
Test if multiple Claude processes can run concurrently.
"""

import subprocess
import threading
import time
from datetime import datetime

def run_claude_test(worker_id, test_dir):
    """Run a simple Claude command."""
    start_time = datetime.now()
    print(f"[Worker {worker_id}] Starting at {start_time.strftime('%H:%M:%S')}")
    
    prompt = f"Worker {worker_id}: Create a simple test file called test_{worker_id}.txt with the content 'Hello from worker {worker_id}'"
    
    try:
        # Run Claude with a simple prompt
        process = subprocess.Popen(
            ['claude', '--dangerously-skip-permissions'],
            cwd=test_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=prompt, timeout=60)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"[Worker {worker_id}] Completed at {end_time.strftime('%H:%M:%S')} (duration: {duration:.1f}s)")
        print(f"[Worker {worker_id}] Return code: {process.returncode}")
        
        if process.returncode != 0:
            print(f"[Worker {worker_id}] Error: {stderr}")
            
    except Exception as e:
        print(f"[Worker {worker_id}] Exception: {e}")

def main():
    """Test concurrent Claude execution."""
    test_dir = "/tmp/claude_concurrency_test"
    
    # Create test directory
    subprocess.run(['mkdir', '-p', test_dir])
    
    print("Testing Claude concurrency with 3 workers...")
    print("=" * 60)
    
    # Create threads for concurrent execution
    threads = []
    for i in range(3):
        thread = threading.Thread(target=run_claude_test, args=(i+1, test_dir))
        threads.append(thread)
    
    # Start all threads at the same time
    start_time = datetime.now()
    print(f"Starting all workers at {start_time.strftime('%H:%M:%S')}")
    
    for thread in threads:
        thread.start()
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print("=" * 60)
    print(f"Total execution time: {total_duration:.1f}s")
    
    # Check results
    print("\nChecking created files:")
    result = subprocess.run(['ls', '-la', test_dir], capture_output=True, text=True)
    print(result.stdout)
    
    # Cleanup
    subprocess.run(['rm', '-rf', test_dir])

if __name__ == "__main__":
    main()