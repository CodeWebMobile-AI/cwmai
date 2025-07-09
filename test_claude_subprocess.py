#!/usr/bin/env python3
"""Test Claude subprocess execution."""

import asyncio
import os

async def test_claude():
    print("Testing Claude subprocess execution...")
    
    # Test 1: Simple version check
    proc = await asyncio.create_subprocess_exec(
        'claude', '--version',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    
    print(f"\nTest 1 - Version check:")
    print(f"Return code: {proc.returncode}")
    print(f"Stdout: {stdout.decode()}")
    print(f"Stderr: {stderr.decode()}")
    
    # Test 2: With timeout
    try:
        proc = await asyncio.create_subprocess_exec(
            'claude', '--version',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        print(f"\nTest 2 - With timeout:")
        print(f"Return code: {proc.returncode}")
        print(f"Stdout: {stdout.decode()}")
        print(f"Stderr: {stderr.decode()}")
    except asyncio.TimeoutError:
        print("Test 2 timed out!")
        proc.kill()
    
    # Test 3: Test with a real prompt (but cancel quickly)
    cmd = ['claude', '--dangerously-skip-permissions']
    prompt = "What is 2+2?"
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Send prompt and wait briefly
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode('utf-8')),
            timeout=5
        )
        print(f"\nTest 3 - Real prompt:")
        print(f"Return code: {proc.returncode}")
        print(f"Stdout length: {len(stdout)}")
        print(f"Stderr: {stderr.decode()}")
    except asyncio.TimeoutError:
        print("\nTest 3 - Timed out (expected for real execution)")
        proc.kill()
        await proc.wait()
    
    # Test 4: Check environment
    print(f"\nEnvironment check:")
    print(f"PATH: {os.environ.get('PATH', 'Not set')}")
    print(f"Which claude: {os.popen('which claude').read().strip()}")

if __name__ == "__main__":
    asyncio.run(test_claude())