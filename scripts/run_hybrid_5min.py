#!/usr/bin/env python3
"""
Run the hybrid continuous system for exactly 5 minutes.
"""

import subprocess
import time
import signal
import sys
import os

def run_for_duration():
    """Run the hybrid system for 5 minutes."""
    # Start time
    start_time = time.time()
    duration_seconds = 5 * 60  # 5 minutes
    
    # Set up environment
    env = os.environ.copy()
    
    # Add env vars from .env.local
    if os.path.exists('../.env.local'):
        with open('../.env.local', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key_value = line.strip().split('=', 1)
                    if len(key_value) == 2:
                        env[key_value[0]] = key_value[1]
    
    env['PYTHONPATH'] = str(Path.cwd())
    env['STORAGE_MODE'] = 'local'
    
    # Start the process
    print(f"Starting hybrid continuous system with 20 workers for 5 minutes...")
    print("="*60)
    
    process = subprocess.Popen(
        ['python3', '../run_hybrid_continuous.py', '--workers', '20'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    try:
        # Monitor output and time
        while True:
            # Check if time is up
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                print(f"\n{'='*60}")
                print(f"5 minutes elapsed - stopping system...")
                print(f"{'='*60}")
                break
            
            # Read output
            line = process.stdout.readline()
            if line:
                print(line.strip())
            
            # Check if process ended
            if process.poll() is not None:
                print(f"\nProcess ended early with code: {process.returncode}")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Stop the process
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            time.sleep(2)
            if process.poll() is None:
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
        
        print(f"\nTotal runtime: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    run_for_duration()