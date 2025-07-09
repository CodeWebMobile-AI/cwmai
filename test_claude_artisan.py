#!/usr/bin/env python3
"""Test Claude with artisan command."""

import subprocess
import os
import time

repo_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/ai-text-summarizer"
os.chdir(repo_dir)

# Task that requires running artisan
artisan_prompt = """Create a new Post model with a migration:

1. Run: php artisan make:model Post -m
2. The command will create the model and migration files

Run this command now and tell me what files were created."""

print(f"Testing Claude with artisan command")
print(f"Repository: {repo_dir}")
print("-" * 50)

start_time = time.time()

# Run Claude with longer timeout for command execution
try:
    process = subprocess.run(
        ["/usr/local/bin/claude", "--dangerously-skip-permissions"],
        input=artisan_prompt,
        capture_output=True,
        text=True,
        cwd=repo_dir,
        timeout=60
    )
    execution_time = time.time() - start_time
    
    print(f"\nClaude completed in {execution_time:.1f} seconds")
    print(f"Return code: {process.returncode}")
    print(f"\nSTDOUT: {process.stdout}")
    print(f"\nSTDERR: {process.stderr}")
    
except subprocess.TimeoutExpired:
    print("\nClaude timed out after 60 seconds!")
    print("This suggests Claude is getting stuck when trying to run shell commands.")

# Check if files were created
print("\nChecking for Post model:")
os.system("ls -la app/Models/Post.php 2>/dev/null || echo 'Post model not found'")
print("\nChecking for new migrations:")
os.system("ls -la database/migrations/*posts* 2>/dev/null || echo 'No posts migration found'")