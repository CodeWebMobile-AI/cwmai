#!/usr/bin/env python3
"""
Cleanup script to remove contaminated repositories from system state.
Removes .github and other excluded repositories from all state files.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.repository_exclusion import RepositoryExclusion

def cleanup_state_file(file_path: str) -> bool:
    """Clean contamination from a state file."""
    if not os.path.exists(file_path):
        print(f"State file not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        original_projects = state.get('projects', {})
        if not original_projects:
            print(f"No projects found in {file_path}")
            return False
            
        cleaned_projects = RepositoryExclusion.filter_excluded_repos_dict(original_projects)
        
        if len(cleaned_projects) < len(original_projects):
            removed_projects = list(set(original_projects.keys()) - set(cleaned_projects.keys()))
            state['projects'] = cleaned_projects
            state['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            # Add cleanup metadata
            state['cleanup_metadata'] = {
                'cleaned_at': datetime.now(timezone.utc).isoformat(),
                'removed_projects': removed_projects,
                'cleanup_reason': 'Removed excluded repositories from state'
            }
            
            # Reset resource efficiency metric if it exists
            if 'metrics' in state and 'resource_efficiency' in state['metrics']:
                if state['metrics']['resource_efficiency'] == 0.0:
                    state['metrics']['resource_efficiency'] = 0.5  # Set to neutral value
                    print(f"Reset resource_efficiency metric from 0.0 to 0.5")
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2, sort_keys=True)
            
            print(f"✓ Cleaned {file_path} - removed {len(removed_projects)} excluded projects: {removed_projects}")
            return True
        else:
            print(f"No contamination found in {file_path}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False

def cleanup_orchestrator_state(file_path: str) -> bool:
    """Clean worker specializations in orchestrator state."""
    if not os.path.exists(file_path):
        print(f"Orchestrator state not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        changes_made = False
        
        # Clean work queue
        if 'work_queue' in state:
            original_count = len(state['work_queue'])
            cleaned_queue = []
            for w in state['work_queue']:
                repo = w.get('repository', '')
                if repo and not RepositoryExclusion.is_excluded_repo(repo):
                    cleaned_queue.append(w)
                elif not repo:  # Keep non-repository work items
                    cleaned_queue.append(w)
            
            state['work_queue'] = cleaned_queue
            removed_work = original_count - len(cleaned_queue)
            if removed_work > 0:
                print(f"Removed {removed_work} work items for excluded repositories")
                changes_made = True
        
        # Clean worker specializations
        if 'workers' in state:
            for worker_id, worker_data in state['workers'].items():
                if 'specialization' in worker_data:
                    spec = worker_data['specialization']
                    if spec != 'system_tasks' and RepositoryExclusion.is_excluded_repo(spec):
                        old_spec = spec
                        worker_data['specialization'] = 'general'
                        print(f"Reset {worker_id} specialization from '{old_spec}' to 'general'")
                        changes_made = True
        
        if changes_made:
            state['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"✓ Cleaned orchestrator state: {file_path}")
            return True
        else:
            print(f"No contamination found in orchestrator state")
            return False
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error cleaning orchestrator state: {e}")
        return False

def cleanup_task_history(file_path: str) -> bool:
    """Clean task history to remove duplicate optimization tasks."""
    if not os.path.exists(file_path):
        print(f"Task history not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            tasks = json.load(f)
        
        if isinstance(tasks, list):
            # Count optimization tasks
            opt_tasks = [t for t in tasks if 'Optimize system resource efficiency' in t.get('title', '')]
            if len(opt_tasks) > 5:  # Keep only last 5
                # Remove older optimization tasks
                tasks_to_keep = []
                opt_count = 0
                for task in reversed(tasks):
                    if 'Optimize system resource efficiency' in task.get('title', ''):
                        if opt_count < 5:
                            tasks_to_keep.append(task)
                            opt_count += 1
                    else:
                        tasks_to_keep.append(task)
                
                tasks = list(reversed(tasks_to_keep))
                
                with open(file_path, 'w') as f:
                    json.dump(tasks, f, indent=2)
                
                print(f"✓ Cleaned task history - removed {len(opt_tasks) - 5} duplicate optimization tasks")
                return True
        
        return False
        
    except Exception as e:
        print(f"Error cleaning task history: {e}")
        return False

def main():
    """Run cleanup on all state files."""
    print("=" * 60)
    print("CWMAI State Contamination Cleanup Tool")
    print("=" * 60)
    print("\nThis will clean up:")
    print("  • Excluded repositories (.github, cwmai) from state")
    print("  • Invalid worker specializations")
    print("  • Duplicate optimization tasks")
    print("  • Reset stuck resource efficiency metrics")
    print()
    
    # Clean main system state
    print("1. Cleaning system state...")
    cleanup_state_file('system_state.json')
    
    # Clean orchestrator state
    print("\n2. Cleaning orchestrator state...")
    cleanup_orchestrator_state('continuous_orchestrator_state.json')
    
    # Clean task history
    print("\n3. Cleaning task history...")
    cleanup_task_history('task_history.json')
    
    # Clean any other state files
    print("\n4. Cleaning other state files...")
    state_files = ['task_state.json', 'completed_tasks.json']
    for file in state_files:
        if os.path.exists(file):
            cleanup_state_file(file)
    
    print("\n" + "=" * 60)
    print("✨ Cleanup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. The excluded repositories have been removed from all state files")
    print("2. Workers with excluded specializations have been reset to 'general'")
    print("3. Resource efficiency metric has been reset if it was stuck at 0.0")
    print("4. Restart the continuous orchestrator to use the clean state")
    print("\nRun: python run_continuous_ai.py --mode development --workers 2")

if __name__ == "__main__":
    main()