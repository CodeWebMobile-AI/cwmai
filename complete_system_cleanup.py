#!/usr/bin/env python3
"""
Complete System Cleanup Script
Stops running processes and clears all state files for a fresh start
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(message):
    """Print a colored header"""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{message.center(60)}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")

def stop_running_processes():
    """Stop all cwmai-related processes"""
    print(f"{YELLOW}Stopping running processes...{RESET}")
    
    # Find and kill Python processes running cwmai scripts
    try:
        # Get all Python processes
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        killed_count = 0
        for line in lines:
            if 'python' in line and any(x in line for x in ['cwmai', 'orchestrator', 'worker', 'executor']):
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"  {GREEN}✓{RESET} Stopped process {pid}: {' '.join(parts[10:])[:60]}...")
                        killed_count += 1
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        print(f"  {RED}✗{RESET} Permission denied for process {pid}")
        
        if killed_count > 0:
            print(f"\n{GREEN}Stopped {killed_count} processes{RESET}")
        else:
            print(f"  No running cwmai processes found")
            
    except Exception as e:
        print(f"  {RED}Error stopping processes: {e}{RESET}")

def clear_state_files():
    """Clear all state files"""
    print(f"\n{YELLOW}Clearing state files...{RESET}")
    
    state_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state")
    
    state_files = [
        "active_tasks.json",
        "completed_tasks.json",
        "continuous_orchestrator_state.json",
        "task_hashes.json",
        "task_history.json",
        "task_state.json",
        "in_progress_tasks.json",
        "work_queue_backup.json"
    ]
    
    cleared_count = 0
    for filename in state_files:
        file_path = state_dir / filename
        if file_path.exists():
            with open(file_path, 'w') as f:
                json.dump({}, f, indent=2)
            print(f"  {GREEN}✓{RESET} Cleared {filename}")
            cleared_count += 1
    
    print(f"\n{GREEN}Cleared {cleared_count} state files{RESET}")

def reset_system_state():
    """Reset system_state.json"""
    print(f"\n{YELLOW}Resetting system_state.json...{RESET}")
    
    system_state_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/system_state.json")
    
    system_state = {
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
    
    with open(system_state_path, 'w') as f:
        json.dump(system_state, f, indent=2)
    
    print(f"  {GREEN}✓{RESET} Reset system_state.json")

def clear_claude_queue():
    """Clear Claude execution queue"""
    print(f"\n{YELLOW}Clearing Claude execution queue...{RESET}")
    
    queue_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/claude_execution_queue.json")
    
    if queue_path.exists():
        with open(queue_path, 'w') as f:
            json.dump({"queue": []}, f, indent=2)
        print(f"  {GREEN}✓{RESET} Cleared Claude execution queue")
    else:
        print(f"  Queue file not found")

def clear_caches():
    """Clear all cache directories"""
    print(f"\n{YELLOW}Clearing cache directories...{RESET}")
    
    base_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state")
    cache_dirs = [
        "repository_analysis_cache",
        "task_monitoring",
        "research_knowledge",
        ".codebase_cache"
    ]
    
    cleared_count = 0
    for cache_dir in cache_dirs:
        cache_path = base_path / cache_dir
        if cache_path.exists() and cache_path.is_dir():
            # Clear the directory contents
            for item in cache_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)
            print(f"  {GREEN}✓{RESET} Cleared {cache_dir}")
            cleared_count += 1
    
    print(f"\n{GREEN}Cleared {cleared_count} cache directories{RESET}")

def main():
    """Main cleanup function"""
    print_header("COMPLETE SYSTEM CLEANUP")
    
    print(f"{RED}This will:{RESET}")
    print(f"  • Stop all running cwmai processes")
    print(f"  • Clear all state files")
    print(f"  • Reset system_state.json")
    print(f"  • Clear Claude execution queue")
    print(f"  • Clear all caches")
    
    # Stop running processes
    stop_running_processes()
    
    # Clear state files
    clear_state_files()
    
    # Reset system state
    reset_system_state()
    
    # Clear Claude queue
    clear_claude_queue()
    
    # Clear caches
    clear_caches()
    
    print_header("CLEANUP COMPLETE")
    
    print(f"{GREEN}System has been cleaned and reset!{RESET}\n")
    print(f"To start fresh:")
    print(f"  1. Export environment variables:")
    print(f"     {BLUE}export $(cat .env.local | grep -v '^#' | xargs){RESET}")
    print(f"  2. Start the system:")
    print(f"     {BLUE}python3 run_hybrid_continuous.py --workers 10{RESET}")
    print(f"\nThe system will start with 0 repositories and 0 tasks.")

if __name__ == "__main__":
    main()