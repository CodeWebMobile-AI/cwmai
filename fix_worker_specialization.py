#!/usr/bin/env python3
"""Fix to prevent hardcoded repository names from being used as worker specializations."""

import os

# Read the continuous_orchestrator.py file
file_path = "scripts/continuous_orchestrator.py"
with open(file_path, 'r') as f:
    content = f.read()

# Find and show the current _assign_worker_specialization method
import re
pattern = r'(def _assign_worker_specialization.*?(?=\n    def|\n\nclass|\Z))'
match = re.search(pattern, content, re.DOTALL)

if match:
    print("Current _assign_worker_specialization method:")
    print("-" * 80)
    print(match.group(1))
    print("-" * 80)
    
    # Check if there's a fallback to hardcoded repos
    method_content = match.group(1)
    if "ai-creative-studio" in content or "moderncms-with-ai-powered" in content:
        print("\nFOUND HARDCODED REPOSITORIES in the file!")
        
        # Find where they appear
        for line_num, line in enumerate(content.split('\n'), 1):
            if "ai-creative-studio" in line or "moderncms-with-ai-powered" in line:
                print(f"Line {line_num}: {line.strip()}")
else:
    print("Could not find _assign_worker_specialization method")

# Also check for any initialization of projects with these values
print("\n\nChecking for project initialization...")
if "ai-creative-studio" in content:
    print("Found references to ai-creative-studio in the file")
if "moderncms-with-ai-powered" in content:
    print("Found references to moderncms-with-ai-powered in the file")