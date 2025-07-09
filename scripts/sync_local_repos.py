#!/usr/bin/env python3
"""
Synchronize Local Repository State

This script synchronizes the repositories.json state file with the actual
folders in the local repository directory, ensuring accurate repository counts.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone


def sync_repositories():
    """Synchronize repository state with actual folders."""
    # Paths
    base_path = Path("../local_repos")
    repos_path = base_path / "repositories" / "CodeWebMobile-AI"
    state_file = base_path / "state" / "repositories.json"
    
    # Load current state
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
    else:
        state = {
            "repositories": [],
            "total_count": 0,
            "last_updated": None,
            "metadata": {"version": "1.0", "storage_mode": "local"}
        }
    
    # Get actual repository folders
    actual_repos = set()
    if repos_path.exists():
        print(f"Looking in: {repos_path.absolute()}")
        for item in repos_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                actual_repos.add(item.name)
                print(f"  Found: {item.name}")
    
    print(f"\nTotal: {len(actual_repos)} actual repository folders")
    
    # Create a map of existing repos in state
    state_repos = {repo['name']: repo for repo in state['repositories']}
    
    # Find repos in state but not on disk
    missing_on_disk = set(state_repos.keys()) - actual_repos
    if missing_on_disk:
        print(f"\nRepositories in state but not on disk: {missing_on_disk}")
    
    # Find repos on disk but not in state
    missing_in_state = actual_repos - set(state_repos.keys())
    if missing_in_state:
        print(f"\nRepositories on disk but not in state: {missing_in_state}")
    
    # Remove repos that don't exist on disk
    state['repositories'] = [
        repo for repo in state['repositories'] 
        if repo['name'] in actual_repos
    ]
    
    # Add missing repos to state
    for repo_name in missing_in_state:
        repo_path = repos_path / repo_name
        state['repositories'].append({
            "name": repo_name,
            "full_name": f"CodeWebMobile-AI/{repo_name}",
            "description": "",
            "path": str(repo_path.relative_to(Path.cwd())),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "default_branch": "main",
            "private": False,
            "html_url": f"file://{repo_path.absolute()}"
        })
    
    # Update counts
    state['total_count'] = len(state['repositories'])
    state['last_updated'] = datetime.now(timezone.utc).isoformat()
    
    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\nUpdated state file with {state['total_count']} repositories")
    print(f"State file: {state_file}")
    
    # List all repositories
    print("\nActive repositories:")
    for repo in sorted(state['repositories'], key=lambda x: x['name']):
        print(f"  - {repo['name']}")


if __name__ == "__main__":
    sync_repositories()