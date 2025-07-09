#!/usr/bin/env python3
"""Test that intelligent_work_finder only checks local repositories in LOCAL mode."""

import os
import asyncio
import json
from pathlib import Path

# Set LOCAL mode
os.environ['STORAGE_MODE'] = 'local'

async def test_local_work_finder():
    """Test the intelligent work finder in LOCAL mode."""
    from scripts.intelligent_work_finder import IntelligentWorkFinder
    from scripts.local_github_simulator import LocalGitHubSimulator
    
    # Create work finder
    work_finder = IntelligentWorkFinder()
    
    # Get local repositories
    simulator = LocalGitHubSimulator()
    local_repos = simulator.list_repositories()
    
    print(f"Found {len(local_repos)} local repositories:")
    for repo in local_repos:
        print(f"  - {repo.name} at {repo.path}")
    
    # Discover work
    print("\nDiscovering work...")
    work_items = await work_finder.discover_work(max_items=5)
    
    print(f"\nFound {len(work_items)} work items:")
    for item in work_items:
        print(f"  - [{item.task_type}] {item.title} (repo: {item.repository})")
    
    # Verify all work items are for local repositories
    local_repo_names = {repo.name for repo in local_repos}
    for item in work_items:
        if item.repository and item.repository not in local_repo_names:
            print(f"❌ ERROR: Work item for non-local repository: {item.repository}")
            return False
    
    print("\n✅ All work items are for local repositories only!")
    return True

if __name__ == "__main__":
    # Export env vars for API access
    result = asyncio.run(test_local_work_finder())
    exit(0 if result else 1)