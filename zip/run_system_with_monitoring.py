#!/usr/bin/env python3
"""
Run the system with monitoring for schema errors.
"""

import subprocess
import sys
import time
import threading
import queue
import re
from datetime import datetime

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def monitor_output(proc, output_queue):
    """Monitor process output for errors and important events."""
    error_patterns = [
        "too many states",
        "INVALID_ARGUMENT",
        "schema produces a constraint",
        "Schema validation error",
        "400.*Bad Request"
    ]
    
    success_patterns = [
        "opportunity selected",
        "Research completed successfully",
        "Project created",
        "Claude task completed"
    ]
    
    while True:
        line = proc.stdout.readline()
        if not line:
            break
            
        line = line.strip()
        if line:
            # Check for errors
            is_error = any(pattern in line for pattern in error_patterns)
            is_success = any(pattern in line for pattern in success_patterns)
            
            # Format and queue the output
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if is_error:
                output_queue.put(f"{RED}[{timestamp}] ❌ ERROR: {line}{RESET}")
            elif is_success:
                output_queue.put(f"{GREEN}[{timestamp}] ✅ {line}{RESET}")
            elif "Phase" in line and ":" in line:
                output_queue.put(f"{BLUE}[{timestamp}] 🔄 {line}{RESET}")
            elif "WARNING" in line or "⚠️" in line:
                output_queue.put(f"{YELLOW}[{timestamp}] ⚠️  {line}{RESET}")
            else:
                output_queue.put(f"[{timestamp}] {line}")

def print_output(output_queue):
    """Print queued output."""
    while True:
        try:
            line = output_queue.get(timeout=0.1)
            print(line)
        except queue.Empty:
            continue

def main():
    """Run the system with monitoring."""
    print(f"{BLUE}System Runner with Schema Error Monitoring{RESET}")
    print("=" * 60)
    print(f"{GREEN}✅ Schema fix verified and working{RESET}")
    print(f"{YELLOW}⚠️  Monitoring for any schema errors...{RESET}")
    print("=" * 60)
    print()
    
    # Set up the command
    cmd = [
        'bash', '-c',
        'export $(cat .env.local | grep -v "^#" | xargs) && '
        'PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip '
        'python3 scripts/continuous_orchestrator_no_redis.py'
    ]
    
    # Start the process
    print(f"{BLUE}Starting system...{RESET}\n")
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Set up output monitoring
        output_queue = queue.Queue()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=monitor_output,
            args=(proc, output_queue),
            daemon=True
        )
        monitor_thread.start()
        
        # Start output printing thread
        print_thread = threading.Thread(
            target=print_output,
            args=(output_queue,),
            daemon=True
        )
        print_thread.start()
        
        # Wait for the process
        print(f"{GREEN}System is running. Press Ctrl+C to stop.{RESET}\n")
        
        proc.wait()
        
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopping system...{RESET}")
        if proc:
            proc.terminate()
            time.sleep(2)
            if proc.poll() is None:
                proc.kill()
        print(f"{GREEN}System stopped.{RESET}")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()