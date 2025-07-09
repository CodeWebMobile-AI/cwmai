#!/usr/bin/env python3
"""Test Claude with a complex prompt similar to the system's."""

import subprocess
import os
import json

# Create a test Laravel-like directory structure
test_dir = "/tmp/claude_laravel_test"
os.makedirs(test_dir, exist_ok=True)
os.chdir(test_dir)

# Create minimal Laravel-like structure
os.makedirs("app/Models", exist_ok=True)
os.makedirs("database/migrations", exist_ok=True)
os.makedirs("resources/js/components", exist_ok=True)

# Create a package.json
package_json = {
    "name": "test-project",
    "version": "1.0.0",
    "dependencies": {
        "react": "^19.0.0"
    }
}
with open("package.json", "w") as f:
    json.dump(package_json, f, indent=2)

# Complex prompt similar to what the system generates
complex_prompt = """Please implement the following task from issue #1:

Title: Initial Project Setup
Type: FEATURE

Description:
## Initial Project Setup

Based on the project architecture for **ai-text-summarizer**, complete the following setup tasks:

1. **Configure API authentication**
   - [ ] Set up Sanctum/API authentication middleware as per the auth flow design
   - [ ] Create auth controllers following the project's authentication architecture

2. **Create initial data models**
   - [ ] Add core models: Post, Category
   - [ ] Create corresponding migrations following the database schema design

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

print(f"Testing Claude in directory: {test_dir}")
print(f"Prompt length: {len(complex_prompt)} characters")
print("-" * 50)

# Run Claude with the complex prompt
process = subprocess.Popen(
    ["/usr/local/bin/claude", "--dangerously-skip-permissions"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=test_dir
)

stdout, stderr = process.communicate(input=complex_prompt)

print(f"Return code: {process.returncode}")
print(f"\nSTDOUT ({len(stdout)} chars):")
if stdout:
    print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
print(f"\nSTDERR ({len(stderr)} chars):")
if stderr:
    print(stderr[:500] + "..." if len(stderr) > 500 else stderr)

# Check what files were created/modified
print(f"\nChecking for created files...")
for root, dirs, files in os.walk(test_dir):
    level = root.replace(test_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")