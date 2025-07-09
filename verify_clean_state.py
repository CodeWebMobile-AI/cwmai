#!/usr/bin/env python3
"""
Verify that the CWMAI system is in a clean state
"""
import json
import os
from pathlib import Path

def check_json_file(filepath, expected_empty):
    """Check if a JSON file is empty or has expected empty structure"""
    if not os.path.exists(filepath):
        return True, "File does not exist"
    
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        if isinstance(expected_empty, dict):
            # Check if all expected keys are empty
            for key in expected_empty:
                if key in data and data[key]:
                    return False, f"Key '{key}' is not empty: {len(data[key])} items"
        elif isinstance(expected_empty, list):
            if data:
                return False, f"List is not empty: {len(data)} items"
        else:
            if data != expected_empty:
                return False, f"Data does not match expected: {data}"
        
        return True, "Empty as expected"
    except Exception as e:
        return False, f"Error reading file: {e}"

def main():
    print("CWMAI Clean State Verification")
    print("=" * 50)
    
    issues = []
    
    # Check main state files
    print("\n=== Checking main state files ===")
    
    main_files = {
        "system_state.json": {"repositories": {}, "tasks": {}},
        "claude_execution_queue.json": {"queue": [], "in_progress": {}, "completed": {}, "failed": {}},
        "work_queue_backup.json": [],
        "in_progress_tasks.json": {},
        "task_state.json": {"tasks": {}, "completed_tasks": []},
    }
    
    for filename, expected in main_files.items():
        filepath = Path(f"/Users/fortressarigo/PhpstormProjects/cwmai/{filename}")
        is_clean, msg = check_json_file(filepath, expected)
        status = "✓" if is_clean else "✗"
        print(f"{status} {filename}: {msg}")
        if not is_clean:
            issues.append(f"{filename}: {msg}")
    
    # Check local_repos/state files
    print("\n=== Checking local_repos/state files ===")
    
    state_files = {
        "active_tasks.json": {},
        "completed_tasks.json": {},
        "task_hashes.json": {},
        "continuous_orchestrator_state.json": {},
        "project_creation_states.json": {},
        "task_assignment_state.json": {},
        "smart_task_history.json": {},
        "work_discovery_cache.json": {}
    }
    
    state_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state")
    for filename, expected in state_files.items():
        filepath = state_dir / filename
        is_clean, msg = check_json_file(filepath, expected)
        status = "✓" if is_clean else "✗"
        print(f"{status} {filename}: {msg}")
        if not is_clean:
            issues.append(f"state/{filename}: {msg}")
    
    # Check for repositories
    print("\n=== Checking for cloned repositories ===")
    
    repos_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos")
    repo_count = 0
    if repos_dir.exists():
        for item in repos_dir.iterdir():
            if item.is_dir() and item.name not in ["state", "repositories", "templates"]:
                repo_count += 1
                issues.append(f"Repository still exists: {item.name}")
    
    if repo_count == 0:
        print("✓ No repositories found")
    else:
        print(f"✗ Found {repo_count} repositories")
    
    # Summary
    print("\n" + "=" * 50)
    if not issues:
        print("✅ SYSTEM IS CLEAN")
        print("- All state files are empty")
        print("- No repositories exist")
        print("- Ready for fresh start")
    else:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\nTotal issues: {len(issues)}")

if __name__ == "__main__":
    main()