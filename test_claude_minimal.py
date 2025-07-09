#!/usr/bin/env python3
"""Test Claude with minimal task to find the issue."""

import subprocess
import os
import time

# Use actual repository
repo_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/ai-text-summarizer"
os.chdir(repo_dir)

# Minimal prompt - just create one model
minimal_prompt = """Create a Post model in app/Models/Post.php with these fields:
- id (primary key)
- title (string)
- content (text)
- user_id (foreign key)
- timestamps

Make the file now."""

print(f"Testing Claude with minimal prompt")
print(f"Repository: {repo_dir}")
print("-" * 50)

start_time = time.time()

# Run Claude
process = subprocess.run(
    ["/usr/local/bin/claude", "--dangerously-skip-permissions"],
    input=minimal_prompt,
    capture_output=True,
    text=True,
    cwd=repo_dir
)

execution_time = time.time() - start_time

print(f"\nClaude completed in {execution_time:.1f} seconds")
print(f"Return code: {process.returncode}")
print(f"\nSTDOUT: {process.stdout}")
print(f"\nSTDERR: {process.stderr}")

# Check if file was created
if os.path.exists("app/Models/Post.php"):
    print("\nPost.php created successfully!")
    with open("app/Models/Post.php", "r") as f:
        print(f.read()[:500])
else:
    print("\nPost.php was NOT created")