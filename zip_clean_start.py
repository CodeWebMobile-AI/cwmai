#!/usr/bin/env python3
"""
Clean start script for the zip folder.
Removes all cached state and ensures only zip folder repositories are discovered.
"""

import os
import json
import shutil
from pathlib import Path


def clean_zip_folder():
    """Clean all cached state files from the zip folder."""
    zip_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip")
    
    if not zip_dir.exists():
        print(f"Error: Zip directory not found at {zip_dir}")
        return False
    
    # Change to zip directory
    original_dir = os.getcwd()
    os.chdir(zip_dir)
    
    print("Cleaning zip folder state...")
    
    # Remove all JSON state files
    state_files = [
        "system_state.json",
        "task_state.json",
        "work_queue_backup.json",
        "production_context.json",
        "claude_execution_queue.json",
        "dynamic_claude_pool_stats.json",
        "continuous_orchestrator_state.json",
        "task_history.json",
        "benchmark_history.json",
        "bug_history.json"
    ]
    
    removed_count = 0
    for file in state_files:
        if Path(file).exists():
            os.remove(file)
            print(f"  ✓ Removed {file}")
            removed_count += 1
    
    # Clean SQLite database if exists
    db_files = ["cwmai_state.db", "cwmai_state.db-shm", "cwmai_state.db-wal", "ultimate_orchestrator.db"]
    for db_file in db_files:
        if Path(db_file).exists():
            os.remove(db_file)
            print(f"  ✓ Removed {db_file}")
            removed_count += 1
    
    # Clean work queue directory if exists
    if Path("scripts/work_queue_backup.json").exists():
        os.remove("scripts/work_queue_backup.json")
        print("  ✓ Removed scripts/work_queue_backup.json")
        removed_count += 1
    
    print(f"\n✓ Cleaned {removed_count} state files")
    
    # Count repositories in zip folder
    repos_path = Path("local_repos/repositories")
    if repos_path.exists():
        repo_count = sum(1 for org_dir in repos_path.iterdir() if org_dir.is_dir()
                        for repo_dir in org_dir.iterdir() if repo_dir.is_dir())
        print(f"\n📁 Zip folder contains {repo_count} repositories")
        
        # List the repositories
        for org_dir in repos_path.iterdir():
            if org_dir.is_dir():
                for repo_dir in org_dir.iterdir():
                    if repo_dir.is_dir():
                        print(f"  - {org_dir.name}/{repo_dir.name}")
    else:
        print("\n⚠️  No local_repos/repositories directory found")
    
    # Return to original directory
    os.chdir(original_dir)
    
    return True


def test_clean_discovery():
    """Test that discovery works correctly after cleaning."""
    print("\n" + "="*60)
    print("Testing clean repository discovery...")
    print("="*60)
    
    # Change to zip directory for test
    original_dir = os.getcwd()
    zip_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip")
    os.chdir(zip_dir)
    
    # Set up environment
    env_file = Path("../.env.local")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
    
    os.environ["PYTHONPATH"] = str(zip_dir)
    
    try:
        # Import and test
        from scripts.state_manager import StateManager
        
        sm = StateManager()
        print("\n1. Testing discover_local_repositories()...")
        repos = sm.discover_local_repositories()
        print(f"   ✓ Discovered {len(repos)} repositories")
        
        print("\n2. Testing load_state_with_repository_discovery()...")
        state = sm.load_state_with_repository_discovery()
        projects = state.get('projects', {})
        print(f"   ✓ State contains {len(projects)} projects")
        
        if len(projects) == len(repos):
            print("\n✅ SUCCESS: State contains only zip folder repositories!")
        else:
            print(f"\n⚠️  WARNING: Mismatch - discovered {len(repos)} but state has {len(projects)}")
            print("   This might be due to existing state. Run this script again to fully clean.")
        
        # List projects in state
        print("\nProjects in state:")
        for proj_name in sorted(projects.keys()):
            print(f"  - {proj_name}")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Return to original directory
        os.chdir(original_dir)


def main():
    print("Zip Folder Clean Start Utility")
    print("="*40)
    
    # Clean the zip folder
    if clean_zip_folder():
        # Test discovery
        test_clean_discovery()
        
        print("\n" + "="*60)
        print("✅ Zip folder is now clean and ready!")
        print("\nTo run the system:")
        print("  ./run_from_zip.sh run_hybrid_continuous.py --workers 10")
    else:
        print("\n❌ Failed to clean zip folder")


if __name__ == "__main__":
    main()