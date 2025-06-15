#!/usr/bin/env python3
"""Debug repository discovery to see what's being found."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from state_manager import StateManager
from repository_exclusion import should_process_repo

def debug_discovery():
    """Debug repository discovery."""
    state_manager = StateManager()
    
    print("=== DEBUGGING REPOSITORY DISCOVERY ===\n")
    
    # Discover repositories
    discovered_repos = state_manager.discover_organization_repositories()
    
    print(f"\nTotal repositories discovered: {len(discovered_repos)}")
    
    # List all discovered repositories
    print("\nDiscovered repositories:")
    for repo in discovered_repos:
        name = repo.get('name', 'Unknown')
        full_name = repo.get('full_name', name)
        excluded = not should_process_repo(full_name)
        status = "EXCLUDED" if excluded else "INCLUDED"
        print(f"  - {full_name} [{status}]")
        if name in ['ai-creative-studio', 'moderncms-with-ai-powered-content-recommendations']:
            print(f"    >>> FOUND TARGET REPO: {name}")
            print(f"    Description: {repo.get('description', 'No description')}")
            print(f"    Created: {repo.get('created_at', 'Unknown')}")
            print(f"    Health Score: {repo.get('health_score', 0)}")
    
    # Check if the problematic repos are in the list
    print("\n=== SEARCHING FOR SPECIFIC REPOSITORIES ===")
    target_repos = ['ai-creative-studio', 'moderncms-with-ai-powered-content-recommendations']
    
    for target in target_repos:
        found = False
        for repo in discovered_repos:
            if repo.get('name') == target or repo.get('full_name').endswith(target):
                found = True
                print(f"\n✓ Found '{target}':")
                print(f"  Full name: {repo.get('full_name')}")
                print(f"  URL: {repo.get('url')}")
                print(f"  Created: {repo.get('created_at')}")
                print(f"  Health Score: {repo.get('health_score')}")
                break
        
        if not found:
            print(f"\n✗ '{target}' NOT FOUND in discovery")

if __name__ == "__main__":
    debug_discovery()