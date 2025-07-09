#!/usr/bin/env python3
"""Test Claude with the actual complex prompt from the system."""

import subprocess
import os
import threading
import time

# Use actual repository
repo_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/ai-text-summarizer"
os.chdir(repo_dir)

# The actual prompt from the system (truncated for initial testing)
actual_prompt = """Please implement the following task from issue #1:

Title: Initial Project Setup
Type: FEATURE

Description:
## Initial Project Setup

Based on the project architecture for **ai-text-summarizer**, complete the following setup tasks:

1. **Configure API authentication**
   - [ ] Set up Sanctum/API authentication middleware as per the auth flow design
   - [ ] Create auth controllers following the project's authentication architecture
   - [ ] Add protected API routes based on the API design specifications
   - [ ] Implement role-based access control if specified in architecture

2. **Create initial data models**
   - [ ] Add core models: Post, Category
   - [ ] Create corresponding migrations following the database schema design
   - [ ] Set up model relationships as defined in the architecture
   - [ ] Add resource controllers with proper API versioning

Requirements:
- Implement the feature as described
- Add appropriate tests
- Update documentation if needed

Please make the necessary changes to implement this task.
You have permission to create and edit files as needed.
If you need to create new files, place them in appropriate directories.
Ensure all changes follow the project's coding standards.

IMPORTANT: Please go ahead and make the actual file changes now.
Do not wait for additional permission - implement the solution directly."""

print(f"Testing Claude with actual system prompt")
print(f"Repository: {repo_dir}")
print(f"Prompt length: {len(actual_prompt)} characters")
print("-" * 50)

# Function to monitor progress
def monitor_progress(start_time):
    while True:
        elapsed = int(time.time() - start_time)
        print(f"Claude still running... ({elapsed}s elapsed)")
        time.sleep(30)

# Start monitoring thread
start_time = time.time()
monitor_thread = threading.Thread(target=monitor_progress, args=(start_time,))
monitor_thread.daemon = True
monitor_thread.start()

# Run Claude without timeout
print("Starting Claude execution (no timeout)...")
process = subprocess.Popen(
    ["/usr/local/bin/claude", "--dangerously-skip-permissions"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=repo_dir
)

# Let it run as long as needed
stdout, stderr = process.communicate(input=actual_prompt)
execution_time = time.time() - start_time

print(f"\nClaude completed in {execution_time:.1f} seconds")
print(f"Return code: {process.returncode}")
print(f"\nSTDOUT ({len(stdout) if stdout else 0} chars):")
if stdout:
    # Print first and last parts of output
    if len(stdout) > 2000:
        print(stdout[:1000])
        print("\n... [output truncated] ...\n")
        print(stdout[-1000:])
    else:
        print(stdout)
        
print(f"\nSTDERR ({len(stderr) if stderr else 0} chars):")
if stderr:
    print(stderr[:1000] + "..." if len(stderr) > 1000 else stderr)

# Check what changed
print("\nChecking changes made...")
print("\n1. New models created:")
os.system("ls -la app/Models/ | grep -E '(Post|Category)' || echo 'No Post/Category models found'")

print("\n2. New migrations created:")
os.system("ls -la database/migrations/ | grep -v '0001_01' | tail -5 || echo 'No new migrations found'")

print("\n3. New controllers created:")
os.system("find app/Http/Controllers -name '*Controller.php' -newer package.json 2>/dev/null | head -10 || echo 'No new controllers found'")

print("\n4. Git status summary:")
os.system("git status --porcelain | wc -l | xargs -I {} echo '{} files changed'")
os.system("git status --porcelain | head -10")