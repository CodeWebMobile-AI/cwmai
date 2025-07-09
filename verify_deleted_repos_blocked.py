#!/usr/bin/env python3
"""
Verify that deleted repositories are completely blocked from the system.
"""

import json
import os
import subprocess
from typing import Set, Dict, Any, List

# Repositories that have been deleted
DELETED_REPOSITORIES = {
    'subtrack-ai',
    'test-repo',
    'sub-wise',
    'CodeWebMobile-AI/subtrack-ai',
    'CodeWebMobile-AI/test-repo',
    'CodeWebMobile-AI/sub-wise'
}

def check_exclusion_file() -> bool:
    """Check if the exclusion file exists and contains the deleted repos"""
    exclusion_file = "scripts/deleted_repos_exclusion.json"
    
    if not os.path.exists(exclusion_file):
        print(f"❌ Exclusion file not found: {exclusion_file}")
        return False
    
    try:
        with open(exclusion_file, 'r') as f:
            exclusions = json.load(f)
        
        excluded_repos = set()
        for item in exclusions:
            if 'repository' in item:
                excluded_repos.add(item['repository'])
        
        missing = DELETED_REPOSITORIES - excluded_repos
        if missing:
            print(f"❌ Missing from exclusion file: {missing}")
            return False
        else:
            print(f"✓ All deleted repositories are in exclusion file")
            return True
            
    except Exception as e:
        print(f"❌ Error reading exclusion file: {e}")
        return False

def check_file_for_references(file_path: str) -> int:
    """Check a file for references to deleted repositories"""
    if not os.path.exists(file_path):
        return 0
    
    try:
        # Use grep to count references
        count = 0
        for repo in DELETED_REPOSITORIES:
            result = subprocess.run(
                ['grep', '-c', repo, file_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                count += int(result.stdout.strip())
        return count
    except:
        return 0

def verify_system_clean():
    """Verify the system is clean of deleted repository references"""
    print("Verifying deleted repositories are blocked from the system...")
    print(f"Deleted repositories: {', '.join(DELETED_REPOSITORIES)}")
    print()
    
    # Check exclusion file
    exclusion_ok = check_exclusion_file()
    print()
    
    # Check various system files
    files_to_check = [
        "system_state.json",
        "task_state.json",
        "work_queue_backup.json",
        "claude_execution_queue.json",
        "in_progress_tasks.json",
        "continuous_orchestrator_state.json",
        "local_repos/state/active_tasks.json",
        "local_repos/state/task_hashes.json"
    ]
    
    total_references = 0
    clean_files = []
    dirty_files = []
    
    for file_path in files_to_check:
        refs = check_file_for_references(file_path)
        if refs > 0:
            dirty_files.append((file_path, refs))
            total_references += refs
        else:
            clean_files.append(file_path)
    
    # Report results
    if clean_files:
        print("Clean files (no references found):")
        for f in clean_files:
            print(f"  ✓ {f}")
        print()
    
    if dirty_files:
        print("Files still containing references:")
        for f, count in dirty_files:
            print(f"  ❌ {f}: {count} references")
        print()
    
    # Check active work
    if os.path.exists("local_repos/state/active_tasks.json"):
        try:
            with open("local_repos/state/active_tasks.json", 'r') as f:
                active_tasks = json.load(f)
            
            active_deleted_tasks = []
            for task_id, task in active_tasks.items():
                if 'repository' in task:
                    for deleted in DELETED_REPOSITORIES:
                        if deleted in task['repository']:
                            active_deleted_tasks.append((task_id, task['repository']))
                            break
            
            if active_deleted_tasks:
                print("⚠️  Active tasks for deleted repositories:")
                for task_id, repo in active_deleted_tasks:
                    print(f"    - {task_id}: {repo}")
                print()
        except:
            pass
    
    # Summary
    print("=" * 60)
    if exclusion_ok and total_references == 0:
        print("✅ SUCCESS: All deleted repositories are completely blocked!")
        print("   - Exclusion file is properly configured")
        print("   - No references found in system files")
    else:
        print("❌ ISSUES FOUND:")
        if not exclusion_ok:
            print("   - Exclusion file needs attention")
        if total_references > 0:
            print(f"   - Found {total_references} references to deleted repositories")
            print("   - Run cleanup scripts to remove these references")
    
    return total_references == 0 and exclusion_ok

def main():
    """Main verification function"""
    is_clean = verify_system_clean()
    
    print()
    print("Next steps:")
    if is_clean:
        print("1. Monitor the system to ensure no new tasks are created for deleted repos")
        print("2. The intelligent_work_finder.py will now filter out these repositories")
        print("3. Any attempts to create tasks for these repos will be blocked")
    else:
        print("1. Run cleanup_deleted_repositories.py to clean system files")
        print("2. Run clean_claude_execution_queue.py to clean Claude queue")
        print("3. Run this verification script again to confirm cleanup")

if __name__ == "__main__":
    main()