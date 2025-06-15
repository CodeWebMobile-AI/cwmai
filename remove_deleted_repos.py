#!/usr/bin/env python3
"""
Remove deleted GitHub repositories from system state and add them to exclusion list.
"""

import json
import os
from datetime import datetime, timezone

def main():
    """Remove deleted repositories and update exclusion list."""
    
    # Repositories confirmed to be deleted (404 on GitHub)
    deleted_repos = [
        {
            "repository": "CodeWebMobile-AI/ai-creative-studio",
            "deleted_date": datetime.now(timezone.utc).isoformat(),
            "reason": "Repository not found on GitHub (404)"
        },
        {
            "repository": "CodeWebMobile-AI/moderncms-with-ai-powered-content-recommendations",
            "deleted_date": datetime.now(timezone.utc).isoformat(),
            "reason": "Repository not found on GitHub (404)"
        }
    ]
    
    # 1. Create/update deleted repos exclusion file
    exclusion_file = "scripts/deleted_repos_exclusion.json"
    existing_deletions = []
    
    if os.path.exists(exclusion_file):
        with open(exclusion_file, 'r') as f:
            existing_deletions = json.load(f)
    
    # Add new deletions
    for repo in deleted_repos:
        if not any(r['repository'] == repo['repository'] for r in existing_deletions):
            existing_deletions.append(repo)
    
    # Save exclusion file
    with open(exclusion_file, 'w') as f:
        json.dump(existing_deletions, f, indent=2)
    print(f"✅ Updated exclusion list: {exclusion_file}")
    
    # 2. Remove from system state
    state_file = "system_state.json"
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        removed = []
        if 'projects' in state:
            for repo_name in ['ai-creative-studio', 'moderncms-with-ai-powered-content-recommendations']:
                if repo_name in state['projects']:
                    del state['projects'][repo_name]
                    removed.append(repo_name)
        
        if removed:
            # Update metadata
            state['last_updated'] = datetime.now(timezone.utc).isoformat()
            if 'repository_discovery' in state:
                state['repository_discovery']['repositories_found'] = len(state.get('projects', {}))
                state['repository_discovery']['last_cleanup'] = datetime.now(timezone.utc).isoformat()
            
            # Save state
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"✅ Removed {len(removed)} repositories from system state")
        else:
            print("ℹ️  Repositories not found in system state")
    
    # 3. Also update the state in scripts directory if it exists
    scripts_state_file = "scripts/system_state.json"
    if os.path.exists(scripts_state_file):
        with open(scripts_state_file, 'r') as f:
            state = json.load(f)
        
        removed = []
        if 'projects' in state:
            for repo_name in ['ai-creative-studio', 'moderncms-with-ai-powered-content-recommendations']:
                if repo_name in state['projects']:
                    del state['projects'][repo_name]
                    removed.append(repo_name)
        
        if removed:
            state['last_updated'] = datetime.now(timezone.utc).isoformat()
            with open(scripts_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"✅ Removed {len(removed)} repositories from scripts/system_state.json")
    
    print("\n🧹 Cleanup complete! The deleted repositories have been:")
    print("   - Added to the exclusion list")
    print("   - Removed from system state")
    print("\nThey will not reappear in future repository discoveries.")

if __name__ == "__main__":
    main()