#!/usr/bin/env python3
"""
Clean deleted GitHub repositories from system state.
"""

import json
import os
import sys
from datetime import datetime, timezone
import requests
from typing import Dict, Any, Set

def check_github_repo_exists(repo_full_name: str, github_token: str = None) -> bool:
    """Check if a GitHub repository exists."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    url = f"https://api.github.com/repos/{repo_full_name}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking repository {repo_full_name}: {e}")
        return False

def clean_system_state(state_file: str = "system_state.json") -> None:
    """Remove deleted repositories from system state."""
    
    # Load GitHub token from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Warning: GITHUB_TOKEN not found in environment. API rate limits may apply.")
    
    # Load current state
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
    except FileNotFoundError:
        print(f"State file {state_file} not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error loading state file: {e}")
        return
    
    # Check if projects exist
    if "projects" not in state:
        print("No projects found in state")
        return
    
    deleted_repos = []
    active_repos = {}
    
    print(f"Checking {len(state['projects'])} repositories...")
    
    for repo_name, repo_data in state["projects"].items():
        if repo_data.get("type") == "github_repository":
            full_name = repo_data.get("full_name", "")
            
            print(f"Checking {full_name}...", end=" ")
            
            if check_github_repo_exists(full_name, github_token):
                print("✓ exists")
                active_repos[repo_name] = repo_data
            else:
                print("✗ deleted")
                deleted_repos.append(repo_name)
        else:
            # Keep non-GitHub projects
            active_repos[repo_name] = repo_data
    
    # Update state with only active repositories
    if deleted_repos:
        print(f"\nRemoving {len(deleted_repos)} deleted repositories:")
        for repo in deleted_repos:
            print(f"  - {repo}")
        
        state["projects"] = active_repos
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # Update repository count
        if "repository_discovery" in state:
            state["repository_discovery"]["repositories_found"] = len(active_repos)
            state["repository_discovery"]["last_cleanup"] = datetime.now(timezone.utc).isoformat()
        
        # Backup original state
        backup_file = f"{state_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\nBacked up original state to {backup_file}")
        
        # Save updated state
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Updated {state_file}")
        
        # Also update the state in scripts directory if it exists
        scripts_state_file = "scripts/system_state.json"
        if os.path.exists(scripts_state_file):
            with open(scripts_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"Updated {scripts_state_file}")
    else:
        print("\nNo deleted repositories found. State is clean.")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean deleted GitHub repositories from system state")
    parser.add_argument("--state-file", default="system_state.json", help="Path to system state file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without making changes")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        # TODO: Implement dry run
    
    clean_system_state(args.state_file)

if __name__ == "__main__":
    main()