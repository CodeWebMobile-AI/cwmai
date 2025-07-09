#!/usr/bin/env python3
"""Test Claude execution to capture actual error."""

import subprocess
import os

# Test with a simple prompt
test_prompt = """Please create a simple test file called hello.txt with the content "Hello from Claude"."""

# Change to a test directory
test_dir = "/tmp/claude_test"
os.makedirs(test_dir, exist_ok=True)
os.chdir(test_dir)

print(f"Testing Claude in directory: {test_dir}")
print(f"Claude executable: /usr/local/bin/claude")
print(f"Test prompt: {test_prompt}")
print("-" * 50)

# Run Claude with the test prompt
process = subprocess.Popen(
    ["/usr/local/bin/claude", "--dangerously-skip-permissions"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=test_dir
)

stdout, stderr = process.communicate(input=test_prompt)

print(f"Return code: {process.returncode}")
print(f"\nSTDOUT ({len(stdout)} chars):")
print(stdout)
print(f"\nSTDERR ({len(stderr)} chars):")
print(stderr)

# Check if file was created
if os.path.exists("hello.txt"):
    with open("hello.txt", "r") as f:
        print(f"\nFile created with content: {f.read()}")
else:
    print("\nNo file was created")

# List directory contents
print(f"\nDirectory contents:")
for item in os.listdir(test_dir):
    print(f"  - {item}")