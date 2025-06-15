#!/usr/bin/env python3
"""Fix all redis_integration imports to use relative imports."""

import os
import re

# Directory containing the files
dir_path = "/workspaces/cwmai/scripts/redis_integration"

# Pattern to match absolute imports
pattern = re.compile(r'from redis_integration\.(\w+) import')
replacement = r'from .\1 import'

# Files to fix
files = [
    "redis_analytics.py",
    "redis_cache_manager.py", 
    "redis_locks_manager.py",
    "redis_monitoring.py",
    "redis_pubsub_manager.py",
    "redis_state_manager.py",
    "redis_streams_manager.py"
]

for filename in files:
    filepath = os.path.join(dir_path, filename)
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace absolute imports with relative imports
    new_content = pattern.sub(replacement, content)
    
    # Write back if changed
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed imports in {filename}")
    else:
        print(f"No changes needed in {filename}")

print("\nAll imports fixed!")