#!/usr/bin/env python3
"""
Startup cleanup script to ensure deleted repositories don't reappear.
Run this before starting the main system.
"""

import os
import sys
import json
from datetime import datetime

def cleanup_on_startup():
    """Clean up deleted repositories from system state on startup."""
    print("🧹 Running startup cleanup...")
    
    # Check if deleted repos exclusion file exists
    exclusion_file = "scripts/deleted_repos_exclusion.json"
    if not os.path.exists(exclusion_file):
        print("No deleted repos exclusion file found, skipping cleanup")
        return
    
    # Load deleted repos
    with open(exclusion_file, 'r') as f:
        deleted_repos = json.load(f)
    
    if not deleted_repos:
        print("No deleted repositories to clean up")
        return
    
    # Load system state
    state_file = "system_state.json"
    if not os.path.exists(state_file):
        print("No system state file found")
        return
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    # Check and remove deleted repos
    removed = []
    if 'projects' in state:
        repos_to_check = [repo['repository'] for repo in deleted_repos if 'repository' in repo]
        
        for repo_name in list(state['projects'].keys()):
            # Check against all variations
            for deleted_repo in repos_to_check:
                if (repo_name == deleted_repo or 
                    repo_name.lower() == deleted_repo.lower() or
                    state['projects'][repo_name].get('full_name') == deleted_repo):
                    
                    print(f"🗑️  Removing deleted repository: {repo_name}")
                    del state['projects'][repo_name]
                    removed.append(repo_name)
                    break
    
    # Always update discovery count if present
    updated = False
    if 'repository_discovery' in state:
        state['repository_discovery']['repositories_found'] = len(state.get('projects', {}))
        state['repository_discovery']['last_discovery'] = datetime.now().isoformat()
        updated = True

    if removed:
        # Update timestamp for removals
        state['last_updated'] = datetime.now().isoformat()
        updated = True

    if updated:
        # Save updated local state (file) before syncing to Redis
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        # Sync updated state to Redis to avoid stale remote state resurrecting deleted repos
        try:
            import asyncio
            from scripts.redis_state_adapter import RedisEnabledStateManager

            rsm = RedisEnabledStateManager(local_path=state_file)
            # initialize Redis adapter and force sync to Redis
            asyncio.run(rsm.initialize_redis())
            asyncio.run(rsm.redis_adapter._sync_to_redis())
            print("🔄 Synced updated state to Redis")
        except Exception as e:
            print(f"⚠️  Could not sync state to Redis: {e}")

    if removed:
        print(f"✅ Cleaned up {len(removed)} deleted repositories")
    else:
        print("✅ No deleted repositories found in system state")

    print("🚀 Startup cleanup complete!")

if __name__ == "__main__":
    cleanup_on_startup()