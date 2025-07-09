#!/usr/bin/env python3
"""Test Claude with Sanctum setup task."""

import subprocess
import os
import time

repo_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/ai-text-summarizer"
os.chdir(repo_dir)

# Task that requires Sanctum
sanctum_prompt = """Set up Laravel Sanctum for API authentication:

1. Install Sanctum by running: php artisan install:api
2. Update User model to use HasApiTokens trait
3. Create a simple API login controller at app/Http/Controllers/Api/AuthController.php with login method

Make these changes now."""

print(f"Testing Claude with Sanctum setup")
print(f"Repository: {repo_dir}")
print("-" * 50)

start_time = time.time()

# Run Claude with a reasonable timeout
process = subprocess.run(
    ["/usr/local/bin/claude", "--dangerously-skip-permissions"],
    input=sanctum_prompt,
    capture_output=True,
    text=True,
    cwd=repo_dir,
    timeout=60  # 60 second timeout
)

execution_time = time.time() - start_time

print(f"\nClaude completed in {execution_time:.1f} seconds")
print(f"Return code: {process.returncode}")
print(f"\nSTDOUT ({len(process.stdout)} chars):")
print(process.stdout[:1000] + "..." if len(process.stdout) > 1000 else process.stdout)
print(f"\nSTDERR ({len(process.stderr)} chars):")
print(process.stderr[:500] + "..." if len(process.stderr) > 500 else process.stderr)

# Check what was done
print("\nChecking changes:")
print("1. User model:")
os.system("grep -n 'HasApiTokens' app/Models/User.php || echo 'HasApiTokens not found in User model'")

print("\n2. AuthController:")
os.system("ls -la app/Http/Controllers/Api/AuthController.php 2>/dev/null || echo 'AuthController not created'")