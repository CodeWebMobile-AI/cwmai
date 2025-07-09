#!/usr/bin/env python3
"""
Run the hybrid continuous system for 10 minutes and then exit.
"""

import asyncio
import time
import signal
import sys
import subprocess
import os

async def run_for_duration(duration_seconds=600):
    """Run the system for specified duration."""
    print(f"Starting hybrid continuous system for {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)...")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
    
    # Load .env.local variables
    with open('.env.local', 'r') as f:
        for line in f:
            if not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                env[key] = value
    
    # Start the process
    process = subprocess.Popen(
        ['python3', 'run_hybrid_continuous.py', '--workers', '20'],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    start_time = time.time()
    
    try:
        # Read output in real-time
        while True:
            line = process.stdout.readline()
            if line:
                print(line.rstrip())
            
            # Check if duration exceeded
            if time.time() - start_time > duration_seconds:
                print(f"\n⏰ Time's up! Running for {duration_seconds} seconds completed.")
                break
                
            # Check if process ended
            if process.poll() is not None:
                print("\n❌ Process ended unexpectedly")
                break
                
            await asyncio.sleep(0.1)
    
    finally:
        # Terminate the process
        if process.poll() is None:
            print("\n🛑 Stopping the system...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️ Force killing the process...")
                process.kill()
                process.wait()
        
        print(f"\n✅ Test completed. Total runtime: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    asyncio.run(run_for_duration(600))  # 10 minutes