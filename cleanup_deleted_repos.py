#!/usr/bin/env python3
"""
Cleanup script to remove deleted repositories from all system state files.
This script:
1. Removes references to deleted repos from system_state.json
2. Removes work items for deleted repos from continuous_orchestrator_state.json
3. Removes tasks for deleted repos from task_state.json
4. Removes history entries for deleted repos from task_history.json
5. Adds them to a permanent exclusion list
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_system_state(file_path: str = "system_state.json") -> Dict[str, Any]:
    """Load the system state from JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def save_system_state(state: Dict[str, Any], file_path: str = "system_state.json"):
    """Save the system state to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def check_repo_exists(repo_full_name: str, github_token: str = None) -> bool:
    """Check if a GitHub repository exists."""
    try:
        if github_token:
            g = Github(github_token)
            repo = g.get_repo(repo_full_name)
            return True
    except:
        pass
    
    # Fallback to API check without auth
    try:
        response = requests.head(f"https://api.github.com/repos/{repo_full_name}")
        return response.status_code == 200
    except:
        return False

def add_to_permanent_exclusion(repo_names: List[str]):
    """Add repositories to permanent exclusion list."""
    exclusion_file = "scripts/deleted_repos_exclusion.json"
    
    # Load existing exclusions
    exclusions = []
    if os.path.exists(exclusion_file):
        with open(exclusion_file, 'r') as f:
            exclusions = json.load(f)
    
    # Add new exclusions with timestamp
    for repo in repo_names:
        if repo not in [e['repository'] for e in exclusions]:
            exclusions.append({
                'repository': repo,
                'deleted_date': datetime.now(timezone.utc).isoformat(),
                'reason': 'Repository no longer exists on GitHub'
            })
    
    # Save updated exclusions
    os.makedirs(os.path.dirname(exclusion_file), exist_ok=True)
    with open(exclusion_file, 'w') as f:
        json.dump(exclusions, f, indent=2)
    
    print(f"Added {len(repo_names)} repositories to permanent exclusion list")

def cleanup_deleted_repos(github_token: str = None, dry_run: bool = False):
    """Main cleanup function."""
    # Load system state
    state = load_system_state()
    
    if 'projects' not in state:
        print("No projects found in system state")
        return
    
    deleted_repos = []
    repos_to_remove = []
    
    print(f"Checking {len(state['projects'])} repositories...")
    
    for repo_name, repo_info in state['projects'].items():
        full_name = repo_info.get('full_name', repo_name)
        print(f"Checking {full_name}...", end=' ')
        
        if check_repo_exists(full_name, github_token):
            print("✓ Exists")
        else:
            print("✗ Deleted")
            deleted_repos.append(full_name)
            repos_to_remove.append(repo_name)
    
    if not deleted_repos:
        print("\nNo deleted repositories found!")
        return
    
    print(f"\nFound {len(deleted_repos)} deleted repositories:")
    for repo in deleted_repos:
        print(f"  - {repo}")
    
    if dry_run:
        print("\nDRY RUN: No changes made")
        return
    
    # Remove from system state
    for repo_name in repos_to_remove:
        del state['projects'][repo_name]
        print(f"Removed {repo_name} from system state")
    
    # Update last_updated timestamp
    state['last_updated'] = datetime.now(timezone.utc).isoformat()
    
    # Save updated state
    save_system_state(state)
    print(f"\nUpdated system state saved")
    
    # Add to permanent exclusion
    add_to_permanent_exclusion(deleted_repos)

def clean_continuous_orchestrator_state(repos_to_remove: List[str]):
    """Clean up continuous_orchestrator_state.json."""
    filepath = 'continuous_orchestrator_state.json'
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    original_count = {
        'work_queue': len(data.get('work_queue', [])),
        'completed_work': len(data.get('completed_work', []))
    }
    
    # Clean work queue
    if 'work_queue' in data:
        cleaned_queue = []
        for item in data['work_queue']:
            repo = item.get('repository', '')
            if repo not in repos_to_remove:
                cleaned_queue.append(item)
            else:
                logger.info(f"Removing work item for {repo}: {item.get('title', 'Unknown')}")
        data['work_queue'] = cleaned_queue
    
    # Clean completed work
    if 'completed_work' in data:
        cleaned_completed = []
        for item in data['completed_work']:
            repo = item.get('repository', '')
            if repo not in repos_to_remove:
                cleaned_completed.append(item)
            else:
                logger.info(f"Removing completed work for {repo}: {item.get('title', 'Unknown')}")
        data['completed_work'] = cleaned_completed
    
    # Create backup
    backup_path = f"{filepath}.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    os.rename(filepath, backup_path)
    logger.info(f"Created backup: {backup_path}")
    
    # Save cleaned data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Cleaned orchestrator state: removed {original_count['work_queue'] - len(data.get('work_queue', []))} from queue, "
                f"{original_count['completed_work'] - len(data.get('completed_work', []))} from completed")

def clean_task_state(repos_to_remove: List[str]):
    """Clean up task_state.json."""
    filepath = 'task_state.json'
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    original_count = len(data.get('tasks', {}))
    
    # Clean tasks
    if 'tasks' in data:
        cleaned_tasks = {}
        for task_id, task_data in data['tasks'].items():
            repo = task_data.get('repository', '')
            if repo not in repos_to_remove:
                cleaned_tasks[task_id] = task_data
            else:
                logger.info(f"Removing task {task_id} for {repo}: {task_data.get('title', 'Unknown')}")
        data['tasks'] = cleaned_tasks
    
    # Create backup
    backup_path = f"{filepath}.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    os.rename(filepath, backup_path)
    logger.info(f"Created backup: {backup_path}")
    
    # Save cleaned data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Cleaned task state: removed {original_count - len(data.get('tasks', {}))} tasks")

def clean_task_history(repos_to_remove: List[str]):
    """Clean up task_history.json."""
    filepath = 'task_history.json'
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        return
    
    original_count = len(data)
    
    # Filter out entries for deleted repos
    cleaned_history = []
    for entry in data:
        # Check if this entry references a deleted repo
        details = entry.get('details', {})
        title = details.get('title', '')
        
        # Check if title mentions any deleted repo
        mentions_deleted = any(repo in title for repo in repos_to_remove)
        
        if not mentions_deleted:
            cleaned_history.append(entry)
        else:
            logger.info(f"Removing history entry: {title}")
    
    # Create backup
    backup_path = f"{filepath}.backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    os.rename(filepath, backup_path)
    logger.info(f"Created backup: {backup_path}")
    
    # Save cleaned data
    with open(filepath, 'w') as f:
        json.dump(cleaned_history, f, indent=2, default=str)
    
    logger.info(f"Cleaned task history: removed {original_count - len(cleaned_history)} entries")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cleanup deleted repositories from all system state files')
    parser.add_argument('--token', help='GitHub token for authentication', 
                        default=os.environ.get('GITHUB_TOKEN'))
    parser.add_argument('--dry-run', action='store_true', 
                        help='Check repos without making changes')
    parser.add_argument('--specific-repos', nargs='+',
                        help='Specific repositories to remove (e.g., mobile-app-platform mobile-portfolio-app cms-platform-laravel-react)')
    
    args = parser.parse_args()
    
    if args.specific_repos:
        # Clean specific repos from all state files
        repos_to_remove = args.specific_repos
        logger.info(f"Removing specific repositories: {', '.join(repos_to_remove)}")
        
        # Clean system state
        state = load_system_state()
        removed_from_state = []
        
        for repo in repos_to_remove:
            # Try different variations
            variations = [repo, f"CodeWebMobile-AI/{repo}", repo.lower()]
            for var in variations:
                if var in state.get('projects', {}):
                    del state['projects'][var]
                    removed_from_state.append(var)
                    logger.info(f"Removed {var} from system state")
        
        if removed_from_state:
            state['last_updated'] = datetime.now(timezone.utc).isoformat()
            save_system_state(state)
            add_to_permanent_exclusion(removed_from_state)
        
        # Clean other state files
        logger.info("\nCleaning continuous_orchestrator_state.json...")
        clean_continuous_orchestrator_state(repos_to_remove)
        
        logger.info("\nCleaning task_state.json...")
        clean_task_state(repos_to_remove)
        
        logger.info("\nCleaning task_history.json...")
        clean_task_history(repos_to_remove)
        
        logger.info("\nCleanup completed!")
        logger.info("Note: The system should be restarted after cleanup to ensure clean state.")
    else:
        cleanup_deleted_repos(args.token, args.dry_run)

if __name__ == "__main__":
    main()