#!/usr/bin/env python3
"""
Find and copy all missing dependencies for the zip folder
"""

import os
import shutil
import subprocess
import sys
import re

def find_missing_module(error_output):
    """Extract missing module name from error"""
    # Match both "No module named 'xxx'" and "No module named 'scripts.xxx'"
    match = re.search(r"No module named '([^']+)'", error_output)
    if match:
        module_name = match.group(1)
        # Remove 'scripts.' prefix if present
        if module_name.startswith('scripts.'):
            module_name = module_name[8:]
        return module_name
    return None

def copy_missing_file(module_name, parent_dir="/Users/fortressarigo/PhpstormProjects/cwmai", zip_dir="/Users/fortressarigo/PhpstormProjects/cwmai/zip"):
    """Try to find and copy the missing module"""
    # Try different locations
    possible_paths = [
        os.path.join(parent_dir, f"{module_name}.py"),
        os.path.join(parent_dir, "scripts", f"{module_name}.py"),
        os.path.join(parent_dir, module_name.replace('.', '/') + ".py")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Determine destination
            if "scripts/" in path:
                dest = os.path.join(zip_dir, "scripts", os.path.basename(path))
            else:
                dest = os.path.join(zip_dir, os.path.basename(path))
            
            print(f"Copying {path} to {dest}")
            shutil.copy2(path, dest)
            return True
    
    print(f"Could not find {module_name} in parent directory")
    return False

def test_conversational_assistant():
    """Test the enhanced conversational assistant and return any errors"""
    cmd = [
        "bash", "-c",
        "cd /Users/fortressarigo/PhpstormProjects/cwmai/zip && " +
        "export $(cat .env.local | grep -v '^#' | xargs) && " +
        "PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai/zip " +
        "arch -arm64 /usr/bin/python3 -c 'import scripts.enhanced_conversational_assistant'"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return result.stderr
    return None

def main():
    print("Fixing dependencies for enhanced_conversational_assistant.py...")
    
    max_iterations = 50  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # Test import
        error = test_conversational_assistant()
        
        if error is None:
            print("\n✅ All dependencies resolved! Enhanced conversational assistant can be imported successfully.")
            break
        
        # Find missing module
        missing_module = find_missing_module(error)
        
        if missing_module:
            print(f"Missing module: {missing_module}")
            if not copy_missing_file(missing_module):
                print(f"ERROR: Could not find {missing_module}. You may need to copy it manually.")
                print(f"Full error:\n{error}")
                break
        else:
            print("Could not parse missing module from error:")
            print(error)
            break
    
    if iteration >= max_iterations:
        print("ERROR: Too many iterations, something might be wrong")

if __name__ == "__main__":
    main()