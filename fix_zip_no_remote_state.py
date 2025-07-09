#!/usr/bin/env python3
"""
Fix the zip folder to never load state from remote GitHub repository.
This ensures the zip folder is completely standalone.
"""

import os
import re

def fix_state_manager_no_remote():
    """Modify StateManager to skip remote state loading."""
    
    state_manager_path = "/Users/fortressarigo/PhpstormProjects/cwmai/zip/scripts/state_manager.py"
    
    with open(state_manager_path, 'r') as f:
        content = f.read()
    
    # Find the load_state method and modify it to skip remote loading
    # Replace the remote loading section with a comment
    pattern = r'(# Try loading from remote repository\s*try:.*?except Exception as e:.*?print\(f"Error loading remote state: {e}"\))'
    replacement = '''# Skip remote repository loading in standalone zip folder
        # This ensures complete isolation from parent system
        # try:
        #     state = self._load_from_repo()
        #     ... (remote loading disabled for standalone mode)'''
    
    # Use re.DOTALL to match across multiple lines
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Alternative approach if regex doesn't work perfectly
    if "Try loading from remote repository" in content and "self._load_from_repo()" in content:
        lines = content.split('\n')
        new_lines = []
        skip_mode = False
        indent_level = 0
        
        for i, line in enumerate(lines):
            if "# Try loading from remote repository" in line:
                new_lines.append("        # Skip remote repository loading in standalone zip folder")
                new_lines.append("        # This ensures complete isolation from parent system")
                skip_mode = True
                indent_level = len(line) - len(line.lstrip())
                continue
                
            if skip_mode:
                # Skip until we find the end of the try-except block
                if line.strip() and len(line) - len(line.lstrip()) <= indent_level and not line.strip().startswith("except"):
                    skip_mode = False
                    new_lines.append(line)
                elif "print(f\"Error loading remote state:" in line:
                    skip_mode = False
                    # Skip this line too, then continue normally
                    continue
                else:
                    # Comment out the line
                    if line.strip():
                        new_lines.append("        # " + line)
                    else:
                        new_lines.append(line)
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
    
    with open(state_manager_path, 'w') as f:
        f.write(content)
    
    print("✓ Modified StateManager to disable remote state loading")
    print("  The zip folder will now only use local state files")

def clean_all_state_files():
    """Remove all state files to ensure fresh start."""
    
    zip_dir = "/Users/fortressarigo/PhpstormProjects/cwmai/zip"
    
    # List of state files to remove
    state_files = [
        "system_state.json",
        "task_state.json", 
        "work_queue_backup.json",
        "claude_execution_queue.json",
        "continuous_orchestrator_state.json",
        "task_history.json",
        "production_context.json",
        "dynamic_claude_pool_stats.json",
        "cwmai_state.db",
        "cwmai_state.db-shm",
        "cwmai_state.db-wal",
        "ultimate_orchestrator.db"
    ]
    
    # Also look for any .json files in scripts directory
    scripts_dir = os.path.join(zip_dir, "scripts")
    
    print("\nCleaning all state files...")
    
    for file in state_files:
        file_path = os.path.join(zip_dir, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  ✓ Removed {file}")
    
    # Clean any JSON files in scripts
    if os.path.exists(scripts_dir):
        for file in os.listdir(scripts_dir):
            if file.endswith('.json') or file.endswith('.db'):
                file_path = os.path.join(scripts_dir, file)
                os.remove(file_path)
                print(f"  ✓ Removed scripts/{file}")

def main():
    print("=== Fixing Zip Folder Remote State Loading ===\n")
    
    # First, clean all state files
    clean_all_state_files()
    
    # Then fix the StateManager
    fix_state_manager_no_remote()
    
    print("\n✅ Complete! The zip folder is now fully standalone.")
    print("\nTo test:")
    print("1. cd /Users/fortressarigo/PhpstormProjects/cwmai/zip")
    print("2. ./run_from_zip.sh run_hybrid_continuous.py --workers 2")
    print("\nThe system should now start with 0 projects!")

if __name__ == "__main__":
    main()