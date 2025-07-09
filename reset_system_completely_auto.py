#!/usr/bin/env python3
"""
Complete System Reset Script for CWMAI (Auto Version)
=====================================================
This script performs a complete reset of the CWMAI system without user interaction.
"""

import os
import shutil
import json
import sys
from datetime import datetime
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(message):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{message.center(60)}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_error(message):
    print(f"{RED}✗ {message}{RESET}")

def remove_repositories():
    """Remove all repositories from local_repos directory."""
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos")
    
    if not local_repos_path.exists():
        print_warning("local_repos directory doesn't exist")
        return
    
    print("\nRemoving repositories...")
    
    # List all directories in local_repos
    repos_removed = 0
    for item in local_repos_path.iterdir():
        if item.is_dir() and item.name != 'state':
            try:
                shutil.rmtree(item)
                print_success(f"Removed repository: {item.name}")
                repos_removed += 1
            except Exception as e:
                print_error(f"Failed to remove {item.name}: {e}")
    
    print(f"\nTotal repositories removed: {repos_removed}")

def reset_state_files():
    """Reset all state files to empty."""
    state_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state")
    
    if not state_path.exists():
        print_warning("State directory doesn't exist")
        return
    
    print("\nResetting state files...")
    
    state_files = [
        "active_tasks.json",
        "completed_tasks.json",
        "continuous_orchestrator_state.json",
        "task_hashes.json",
        "task_history.json",
        "task_state.json"
    ]
    
    for file_name in state_files:
        file_path = state_path / file_name
        try:
            with open(file_path, 'w') as f:
                json.dump({}, f, indent=2)
            print_success(f"Reset: {file_name}")
        except Exception as e:
            print_error(f"Failed to reset {file_name}: {e}")

def clear_caches():
    """Clear all cache directories."""
    cache_dirs = [
        "/Users/fortressarigo/PhpstormProjects/cwmai/repository_analysis_cache",
        "/Users/fortressarigo/PhpstormProjects/cwmai/task_monitoring",
        "/Users/fortressarigo/PhpstormProjects/cwmai/research_knowledge",
        "/Users/fortressarigo/PhpstormProjects/cwmai/.codebase_cache"
    ]
    
    print("\nClearing cache directories...")
    
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                cache_path.mkdir(exist_ok=True)
                print_success(f"Cleared: {cache_path.name}")
            except Exception as e:
                print_error(f"Failed to clear {cache_path.name}: {e}")
        else:
            print_warning(f"Cache directory doesn't exist: {cache_path.name}")

def reset_claude_queue():
    """Reset the Claude execution queue."""
    queue_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/claude_execution_queue.json")
    
    print("\nResetting Claude execution queue...")
    
    try:
        with open(queue_path, 'w') as f:
            json.dump({}, f, indent=2)
        print_success("Reset Claude execution queue")
    except Exception as e:
        print_error(f"Failed to reset Claude queue: {e}")

def reset_system_state():
    """Reset the main system_state.json file."""
    system_state_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/system_state.json")
    
    print("\nResetting system state...")
    
    empty_state = {
        "repositories": {},
        "tasks": {},
        "metrics": {
            "total_repositories": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "active_tasks": 0
        },
        "last_updated": datetime.utcnow().isoformat()
    }
    
    try:
        with open(system_state_path, 'w') as f:
            json.dump(empty_state, f, indent=2)
        print_success("Reset system_state.json")
    except Exception as e:
        print_error(f"Failed to reset system state: {e}")

def clear_backup_files():
    """Clear all backup files."""
    backup_patterns = [
        "*_backup_*.json",
        "*.backup",
        "backup_*"
    ]
    
    print("\nClearing backup files...")
    
    base_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai")
    backups_removed = 0
    
    for pattern in backup_patterns:
        for backup_file in base_path.rglob(pattern):
            try:
                backup_file.unlink()
                backups_removed += 1
            except Exception as e:
                print_error(f"Failed to remove {backup_file.name}: {e}")
    
    # Also remove state_backups directory if it exists
    state_backups_path = base_path / "state_backups"
    if state_backups_path.exists():
        try:
            shutil.rmtree(state_backups_path)
            print_success("Removed state_backups directory")
        except Exception as e:
            print_error(f"Failed to remove state_backups: {e}")
    
    print(f"Total backup files removed: {backups_removed}")

def clear_misc_files():
    """Clear miscellaneous state and queue files."""
    misc_files = [
        "work_queue_backup.json",
        "in_progress_tasks.json",
        "coordination_state.json",
        "learning_state.json",
        "pattern_learning_state.json",
        "code_evolution_state.json"
    ]
    
    print("\nClearing miscellaneous files...")
    
    base_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai")
    
    for file_name in misc_files:
        file_path = base_path / file_name
        if file_path.exists():
            try:
                file_path.unlink()
                print_success(f"Removed: {file_name}")
            except Exception as e:
                print_error(f"Failed to remove {file_name}: {e}")

def main():
    """Main function to orchestrate the complete system reset."""
    print_header("COMPLETE SYSTEM RESET - AUTO MODE")
    print(f"{YELLOW}This will completely reset your CWMAI system!{RESET}")
    
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Reset initiated at: {timestamp}")
    
    # Execute reset operations
    operations = [
        ("Removing Repositories", remove_repositories),
        ("Resetting State Files", reset_state_files),
        ("Clearing Caches", clear_caches),
        ("Resetting Claude Queue", reset_claude_queue),
        ("Resetting System State", reset_system_state),
        ("Clearing Backup Files", clear_backup_files),
        ("Clearing Miscellaneous Files", clear_misc_files)
    ]
    
    for operation_name, operation_func in operations:
        print(f"\n{BLUE}[{operation_name}]{RESET}")
        try:
            operation_func()
        except Exception as e:
            print_error(f"Error during {operation_name}: {e}")
            continue
    
    print_header("System Reset Complete")
    print(f"{GREEN}✓ Your CWMAI system has been completely reset!{RESET}")
    print(f"\nYou now have a clean slate with:")
    print("  • 0 repositories")
    print("  • 0 tasks")
    print("  • Empty state files")
    print("  • Cleared caches")
    print("  • Reset system configuration")
    print(f"\n{YELLOW}Note: You may need to restart any running orchestrator processes.{RESET}")

if __name__ == "__main__":
    main()