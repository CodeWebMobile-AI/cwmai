#!/usr/bin/env python3
"""
Force stop all CWMAI processes and clean up state
"""
import os
import signal
import subprocess
import json
import shutil
import time
from pathlib import Path

def force_kill_processes():
    """Forcefully kill all Python processes related to CWMAI"""
    print("\n=== Forcefully stopping all CWMAI processes ===")
    
    # Get list of all Python processes
    try:
        # Use ps to find all python processes with cwmai in the command
        ps_output = subprocess.check_output(
            ["ps", "aux"], 
            universal_newlines=True
        )
        
        killed_count = 0
        for line in ps_output.split('\n'):
            if 'python' in line.lower() and 'cwmai' in line.lower():
                try:
                    # Extract PID (second column)
                    parts = line.split()
                    if len(parts) > 1:
                        pid = int(parts[1])
                        print(f"Killing process {pid}: {' '.join(parts[10:])[:100]}...")
                        os.kill(pid, signal.SIGKILL)  # Force kill
                        killed_count += 1
                except (ValueError, ProcessLookupError) as e:
                    continue
        
        # Also check for specific script names
        script_patterns = [
            'run_hybrid_continuous.py',
            'continuous_orchestrator',
            'enhanced_conversational_assistant',
            'dynamic_work_generator',
            'claude_executor',
            'worker_status_monitor',
            'task_monitoring',
            'smart_task_system'
        ]
        
        for pattern in script_patterns:
            try:
                # Use pkill to kill by pattern
                subprocess.run(['pkill', '-9', '-f', pattern], capture_output=True)
            except:
                pass
        
        if killed_count > 0:
            print(f"✓ Forcefully killed {killed_count} CWMAI processes")
            # Wait for processes to fully terminate
            time.sleep(2)
        else:
            print("✓ No running CWMAI processes found")
            
    except subprocess.CalledProcessError:
        print("✗ Error checking processes")

def clear_all_state_files():
    """Clear all state files to ensure clean slate"""
    print("\n=== Clearing all state files ===")
    
    # State files in local_repos/state/
    state_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state")
    if state_dir.exists():
        state_files = [
            "active_tasks.json",
            "completed_tasks.json", 
            "task_hashes.json",
            "continuous_orchestrator_state.json",
            "project_creation_states.json",
            "task_assignment_state.json",
            "smart_task_history.json",
            "work_discovery_cache.json"
        ]
        
        for file_name in state_files:
            file_path = state_dir / file_name
            if file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump({}, f, indent=2)
                print(f"✓ Cleared {file_name}")
    
    # Clear main state files
    main_state_files = {
        "system_state.json": {
            "repositories": {},
            "tasks": {},
            "metrics": {
                "total_repositories": 0,
                "total_tasks": 0,
                "completed_tasks": 0,
                "active_tasks": 0
            },
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        },
        "claude_execution_queue.json": {
            "queue": [],
            "in_progress": {},
            "completed": {},
            "failed": {}
        },
        "work_queue_backup.json": [],
        "in_progress_tasks.json": {},
        "evolutionary_config.json": {},
        "coordination_state.json": {},
        "task_state.json": {
            "tasks": {},
            "completed_tasks": [],
            "last_task_id": 0
        }
    }
    
    for file_name, initial_content in main_state_files.items():
        file_path = Path(f"/Users/fortressarigo/PhpstormProjects/cwmai/{file_name}")
        if file_path.exists():
            with open(file_path, 'w') as f:
                json.dump(initial_content, f, indent=2)
            print(f"✓ Reset {file_name}")
    
    # Clear other JSON files that might contain state
    additional_files = [
        "dynamic_claude_pool_stats.json",
        "learning_state.json",
        "pattern_learning_state.json",
        "code_evolution_state.json"
    ]
    
    for file_name in additional_files:
        file_path = Path(f"/Users/fortressarigo/PhpstormProjects/cwmai/{file_name}")
        if file_path.exists():
            with open(file_path, 'w') as f:
                json.dump({}, f, indent=2)
            print(f"✓ Cleared {file_name}")

def clear_cache_directories():
    """Clear all cache directories"""
    print("\n=== Clearing cache directories ===")
    
    cache_dirs = [
        ".codebase_cache",
        "repository_analysis_cache",
        "evolutionary_knowledge",
        "evolutionary_memory",
        "research_knowledge",
        "logs"
    ]
    
    base_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai")
    
    for cache_dir in cache_dirs:
        dir_path = base_path / cache_dir
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                dir_path.mkdir(exist_ok=True)
                print(f"✓ Cleared {cache_dir}/")
            except Exception as e:
                print(f"✗ Error clearing {cache_dir}/: {e}")

def remove_local_repositories():
    """Remove all cloned repositories"""
    print("\n=== Removing local repositories ===")
    
    repos_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos")
    if repos_dir.exists():
        # Keep the state directory but remove all repository directories
        for item in repos_dir.iterdir():
            if item.is_dir() and item.name != "state":
                try:
                    shutil.rmtree(item)
                    print(f"✓ Removed repository: {item.name}")
                except Exception as e:
                    print(f"✗ Error removing {item.name}: {e}")

def clear_sqlite_databases():
    """Clear SQLite database files"""
    print("\n=== Clearing SQLite databases ===")
    
    db_files = [
        "cwmai_state.db",
        "cwmai_state.db-shm",
        "cwmai_state.db-wal",
        "ultimate_orchestrator.db",
        "redis_fallback.db"
    ]
    
    base_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai")
    
    for db_file in db_files:
        file_path = base_path / db_file
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"✓ Removed {db_file}")
            except Exception as e:
                print(f"✗ Error removing {db_file}: {e}")

def main():
    print("CWMAI Force Stop and Complete Cleanup")
    print("=" * 50)
    
    # Step 1: Force kill all processes
    force_kill_processes()
    
    # Step 2: Clear all state files
    clear_all_state_files()
    
    # Step 3: Clear cache directories
    clear_cache_directories()
    
    # Step 4: Remove local repositories
    remove_local_repositories()
    
    # Step 5: Clear SQLite databases
    clear_sqlite_databases()
    
    print("\n=== Final verification ===")
    
    # Check if any processes are still running
    try:
        ps_output = subprocess.check_output(
            ["ps", "aux"], 
            universal_newlines=True
        )
        
        still_running = False
        for line in ps_output.split('\n'):
            if 'python' in line.lower() and 'cwmai' in line.lower():
                print(f"⚠️  Still running: {line[:100]}...")
                still_running = True
        
        if not still_running:
            print("✓ No CWMAI processes are running")
    except:
        pass
    
    # Check state files
    state_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/system_state.json")
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
            repos = len(state.get('repositories', {}))
            tasks = len(state.get('tasks', {}))
            print(f"✓ System state: {repos} repositories, {tasks} tasks")
    
    print("\n✅ Complete cleanup finished!")
    print("The system is now in a clean state with:")
    print("- No running processes")
    print("- No repositories")
    print("- No tasks")
    print("- All caches cleared")
    print("\nYou can now start fresh.")

if __name__ == "__main__":
    main()