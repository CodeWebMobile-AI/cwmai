#!/usr/bin/env python3
"""
Reset local repositories and state for a fresh start.
Deletes all local repositories and cleans up related state.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime


def backup_state(state_file, backup_dir):
    """Backup a state file before modifying."""
    if os.path.exists(state_file):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{os.path.basename(state_file)}.backup_{timestamp}"
        backup_path = backup_dir / backup_name
        shutil.copy2(state_file, backup_path)
        print(f"  📦 Backed up {os.path.basename(state_file)} to {backup_name}")
        return backup_path
    return None


def main():
    print("🧹 Resetting Local Repositories and State")
    print("=" * 60)
    
    # Define paths
    local_repos_dir = Path("local_repos")
    repositories_dir = local_repos_dir / "repositories"
    state_dir = local_repos_dir / "state"
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    # State files to clean
    state_files = [
        "system_state.json",
        "task_state.json",
        "continuous_orchestrator_state.json",
        "research_cache.json",
        "claude_execution_queue.json",
        "in_progress_tasks.json",
        "work_queue_backup.json"
    ]
    
    # Step 1: Delete all repositories
    print("\n1️⃣ Deleting Local Repositories")
    print("-" * 40)
    
    if repositories_dir.exists():
        repos = list(repositories_dir.rglob("*"))
        repo_count = len([r for r in repos if r.is_dir() and r.parent.name == "CodeWebMobile-AI"])
        
        if repo_count > 0:
            print(f"Found {repo_count} repositories to delete:")
            for repo in repositories_dir.glob("CodeWebMobile-AI/*"):
                if repo.is_dir():
                    print(f"  🗑️  Deleting: {repo.name}")
                    shutil.rmtree(repo)
            
            # Clean up CodeWebMobile-AI directory if empty
            org_dir = repositories_dir / "CodeWebMobile-AI"
            if org_dir.exists() and not any(org_dir.iterdir()):
                org_dir.rmdir()
                print("  🗑️  Removed empty organization directory")
        else:
            print("  ✅ No repositories found")
    else:
        print("  ✅ Repositories directory doesn't exist")
    
    # Step 2: Clean state files
    print("\n2️⃣ Cleaning State Files")
    print("-" * 40)
    
    for state_file in state_files:
        file_path = Path(state_file)
        if file_path.exists():
            # Backup first
            backup_state(file_path, backup_dir)
            
            # Clean based on file type
            if state_file == "system_state.json":
                # Reset system state but keep structure
                with open(file_path, 'r') as f:
                    state = json.load(f)
                
                # Remove all projects
                if 'projects' in state:
                    old_count = len(state['projects'])
                    state['projects'] = {}
                    print(f"  🧹 Removed {old_count} projects from system_state.json")
                
                # Reset worker states
                if 'workers' in state:
                    for worker_id in state['workers']:
                        state['workers'][worker_id]['status'] = 'idle'
                        state['workers'][worker_id]['current_task'] = None
                    print(f"  🔄 Reset {len(state['workers'])} workers to idle")
                
                # Save cleaned state
                with open(file_path, 'w') as f:
                    json.dump(state, f, indent=2)
                    
            elif state_file == "task_state.json":
                # Reset task state
                with open(file_path, 'r') as f:
                    state = json.load(f)
                
                # Remove all tasks related to projects
                if 'tasks' in state:
                    old_tasks = state['tasks']
                    state['tasks'] = {}
                    
                    # Keep only non-project tasks
                    project_types = ['NEW_PROJECT', 'PROJECT_SETUP', 'DOCUMENTATION', 'ARCHITECTURE']
                    kept = 0
                    for task_id, task in old_tasks.items():
                        if task.get('task_type') not in project_types:
                            state['tasks'][task_id] = task
                            kept += 1
                    
                    removed = len(old_tasks) - kept
                    print(f"  🧹 Removed {removed} project-related tasks, kept {kept} others")
                
                with open(file_path, 'w') as f:
                    json.dump(state, f, indent=2)
                    
            elif state_file == "research_cache.json":
                # Clear research cache completely
                os.remove(file_path)
                print(f"  🗑️  Deleted {state_file} (will regenerate)")
                
            elif state_file in ["claude_execution_queue.json", "in_progress_tasks.json"]:
                # Clear these completely
                with open(file_path, 'w') as f:
                    json.dump({"tasks": []}, f, indent=2)
                print(f"  🧹 Cleared {state_file}")
                
            else:
                # For other files, just note they exist
                print(f"  ℹ️  Found {state_file} (not modified)")
    
    # Step 3: Clean local state directory
    print("\n3️⃣ Cleaning Local State Directory")
    print("-" * 40)
    
    if state_dir.exists():
        # Remove all repository state files
        state_files = list(state_dir.glob("*.json"))
        if state_files:
            print(f"Found {len(state_files)} state files:")
            for state_file in state_files:
                print(f"  🗑️  Deleting: {state_file.name}")
                state_file.unlink()
        else:
            print("  ✅ No state files found")
    else:
        print("  ✅ State directory doesn't exist")
    
    # Step 4: Clean any cached data
    print("\n4️⃣ Cleaning Cached Data")
    print("-" * 40)
    
    cache_files = [
        ".python_file_count_cache.json",
        ".codebase_cache",
        "issue_cache"
    ]
    
    for cache in cache_files:
        cache_path = Path(cache)
        if cache_path.exists():
            if cache_path.is_dir():
                shutil.rmtree(cache_path)
                print(f"  🗑️  Deleted directory: {cache}")
            else:
                cache_path.unlink()
                print(f"  🗑️  Deleted file: {cache}")
    
    # Step 5: Create fresh directories
    print("\n5️⃣ Creating Fresh Directory Structure")
    print("-" * 40)
    
    # Ensure directories exist
    directories = [
        local_repos_dir,
        repositories_dir,
        repositories_dir / "CodeWebMobile-AI",
        state_dir,
        local_repos_dir / "templates"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  📁 Ensured directory: {directory}")
    
    # Summary
    print("\n✅ Reset Complete!")
    print("=" * 60)
    print("• All local repositories deleted")
    print("• State files cleaned and backed up")
    print("• Cache files removed")
    print("• Fresh directory structure created")
    print(f"\n📦 Backups saved to: {backup_dir}")
    print("\n🚀 Ready for fresh start with local repositories!")


if __name__ == "__main__":
    main()