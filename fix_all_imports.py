#!/usr/bin/env python3
"""Fix all relative imports in the scripts directory."""

import os
import re

def fix_relative_imports(file_path):
    """Fix relative imports in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match relative imports
    pattern = r'^from \.(\w+)'
    
    # Replace relative imports with absolute imports
    new_content = re.sub(pattern, r'from \1', content, flags=re.MULTILINE)
    
    if content != new_content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Fixed imports in {file_path}")
        return True
    return False

def main():
    scripts_dir = 'scripts'
    fixed_count = 0
    
    for filename in os.listdir(scripts_dir):
        if filename.endswith('.py'):
            file_path = os.path.join(scripts_dir, filename)
            if fix_relative_imports(file_path):
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files!")

if __name__ == "__main__":
    main()