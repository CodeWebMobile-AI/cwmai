#!/usr/bin/env python3
"""Test Claude execution in a repository."""

import asyncio
import os

async def test_claude_in_repo():
    repo_path = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/ai-text-summarizer"
    
    if not os.path.exists(repo_path):
        print(f"Repository path not found: {repo_path}")
        return
    
    print(f"Testing Claude in repository: {repo_path}")
    
    # Simple test prompt
    cmd = ['claude', '--dangerously-skip-permissions']
    prompt = "List the files in the current directory."
    
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Working directory: {repo_path}")
    print(f"Prompt: {prompt}")
    
    # Run Claude
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=repo_path
    )
    
    print("\nSending prompt to Claude...")
    
    # Send prompt and wait for response
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode('utf-8')),
            timeout=30  # 30 second timeout for simple task
        )
        
        print(f"\nReturn code: {proc.returncode}")
        print(f"\nStdout ({len(stdout)} bytes):")
        print("-" * 40)
        print(stdout.decode()[:1000])  # First 1000 chars
        if len(stdout) > 1000:
            print(f"... (truncated, total {len(stdout)} bytes)")
        print("-" * 40)
        
        print(f"\nStderr ({len(stderr)} bytes):")
        print("-" * 40)
        print(stderr.decode())
        print("-" * 40)
        
    except asyncio.TimeoutError:
        print("\nExecution timed out after 30 seconds")
        proc.kill()
        await proc.wait()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_claude_in_repo())