#!/usr/bin/env python3
"""Test Claude in actual repository directory without timeout."""

import subprocess
import os
import threading
import time

# Use actual repository
repo_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/ai-text-summarizer"
os.chdir(repo_dir)

# Simple test prompt to install Zustand
test_prompt = """Please install Zustand for state management by running: npm install zustand

After installation, create a simple example store at resources/js/stores/exampleStore.ts with:
- A count state
- increment and decrement actions

Make the actual changes now."""

print(f"Testing Claude in actual repository: {repo_dir}")
print(f"Test prompt: {test_prompt}")
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
stdout, stderr = process.communicate(input=test_prompt)
execution_time = time.time() - start_time

print(f"\nClaude completed in {execution_time:.1f} seconds")
print(f"Return code: {process.returncode}")
print(f"\nSTDOUT ({len(stdout) if stdout else 0} chars):")
if stdout:
    print(stdout[:1000] + "..." if len(stdout) > 1000 else stdout)
print(f"\nSTDERR ({len(stderr) if stderr else 0} chars):")
if stderr:
    print(stderr[:1000] + "..." if len(stderr) > 1000 else stderr)

# Check package.json for zustand
print("\nChecking if Zustand was installed...")
os.system("grep zustand package.json || echo 'Zustand not found in package.json'")

# Check if store file was created
store_path = "resources/js/stores/exampleStore.ts"
if os.path.exists(store_path):
    print(f"\nStore file created at {store_path}")
    with open(store_path, "r") as f:
        print("Content preview:")
        content = f.read()
        print(content[:500] + "..." if len(content) > 500 else content)
else:
    print(f"\nNo store file found at {store_path}")

# Check git status
print("\nGit status:")
os.system("git status --porcelain | head -10")