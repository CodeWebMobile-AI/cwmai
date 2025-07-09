#!/usr/bin/env python3
"""Test Claude with simple Laravel tasks one by one."""

import subprocess
import os
import time

# Use actual repository
repo_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/ai-text-summarizer"
os.chdir(repo_dir)

# Simple task 1: Just update User model
simple_prompt = """Update the User model at app/Models/User.php to add the HasApiTokens trait:

1. Add: use Laravel\Sanctum\HasApiTokens; at the top with other imports
2. Add HasApiTokens to the class traits

Make this change now."""

print(f"Testing Claude with simple Laravel task")
print(f"Repository: {repo_dir}")
print("-" * 50)

start_time = time.time()

# Run Claude with timeout
process = subprocess.run(
    ["/usr/local/bin/claude", "--dangerously-skip-permissions"],
    input=simple_prompt,
    capture_output=True,
    text=True,
    cwd=repo_dir,
    timeout=30
)

execution_time = time.time() - start_time

print(f"\nClaude completed in {execution_time:.1f} seconds")
print(f"Return code: {process.returncode}")
print(f"\nSTDOUT: {process.stdout}")
print(f"\nSTDERR: {process.stderr}")

# Check if change was made
print("\nChecking User model:")
os.system("grep -n 'HasApiTokens' app/Models/User.php")