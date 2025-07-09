#!/usr/bin/env python3
"""
Run improvements for exactly 1 minute then stop
"""

import asyncio
import time
import subprocess
import sys
import os

async def run_for_1_minute():
    """Run improvements for 1 minute."""
    start_time = time.time()
    
    # Start the improvement process
    process = await asyncio.create_subprocess_exec(
        sys.executable, 'run_real_improvements.py',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **dict(line.split('=', 1) for line in open('.env.local').read().strip().split('\n') if '=' in line and not line.startswith('#'))}
    )
    
    # Collect output for 1 minute
    output_lines = []
    try:
        while time.time() - start_time < 60:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                if line:
                    output = line.decode().strip()
                    print(output)
                    output_lines.append(output)
            except asyncio.TimeoutError:
                pass
                
            # Check if process ended
            if process.returncode is not None:
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop the process
        process.terminate()
        await process.wait()
        
    print(f"\n\n=== RAN FOR {time.time() - start_time:.1f} SECONDS ===")
    
    # Analyze output for issues
    issues_found = []
    improvements_created = 0
    
    for line in output_lines:
        if "❌ Error:" in line or "ERROR" in line:
            issues_found.append(line)
        elif "✅ Created:" in line or "✅ SUCCESS:" in line:
            improvements_created += 1
            
    print(f"\nSUMMARY:")
    print(f"- Improvements created: {improvements_created}")
    print(f"- Issues found: {len(issues_found)}")
    
    if issues_found:
        print("\nISSUES TO FIX:")
        for issue in issues_found[:10]:  # Show first 10
            print(f"  - {issue}")

if __name__ == '__main__':
    asyncio.run(run_for_1_minute())