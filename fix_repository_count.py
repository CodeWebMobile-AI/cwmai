#!/usr/bin/env python3
"""
Fix repository count issue by rediscovering repositories.

This script will:
1. Load the current state
2. Discover all repositories in the organization
3. Update the state with the discovered repositories
4. Save the updated state
"""

import os
import sys
import json
from datetime import datetime, timezone

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from state_manager import StateManager


def main():
    """Main function to fix repository count."""
    print("Fixing repository count issue...")
    print("-" * 50)
    
    # Initialize state manager
    state_manager = StateManager()
    
    # Load current state
    print("\n1. Loading current state...")
    current_state = state_manager.load_state()
    current_projects = current_state.get('projects', {})
    print(f"   Current projects count: {len(current_projects)}")
    
    # Show current repository discovery info
    repo_discovery = current_state.get('repository_discovery', {})
    print(f"   Last discovery info: {json.dumps(repo_discovery, indent=2)}")
    
    # Discover repositories
    print("\n2. Discovering repositories...")
    repositories = state_manager.discover_organization_repositories()
    print(f"   Discovered {len(repositories)} repositories")
    
    if repositories:
        # Update state with discovered repositories
        print("\n3. Updating state with discovered repositories...")
        
        # Ensure projects dict exists
        if 'projects' not in current_state:
            current_state['projects'] = {}
        
        # Add each repository to projects
        for repo_data in repositories:
            project_id = repo_data['name']
            current_state['projects'][project_id] = {
                'name': repo_data['name'],
                'full_name': repo_data['full_name'],
                'description': repo_data['description'],
                'url': repo_data['url'],
                'clone_url': repo_data['clone_url'],
                'language': repo_data['language'],
                'health_score': repo_data['health_score'],
                'last_checked': repo_data['discovered_at'],
                'metrics': repo_data['metrics'],
                'recent_activity': repo_data['recent_activity'],
                'topics': repo_data['topics'],
                'default_branch': repo_data['default_branch'],
                'type': 'github_repository',
                'status': 'active',
                'stars': repo_data['metrics']['stars'],
                'open_issues_count': repo_data['metrics']['issues_open'],
                'archived': False,  # We skip archived repos in discovery
                'action_history': [
                    {
                        "action": "repository_discovered",
                        "details": f"Repository {repo_data['name']} discovered and integrated into system",
                        "outcome": "success_discovered",
                        "timestamp": repo_data['discovered_at']
                    }
                ]
            }
            print(f"   ✓ Added repository: {repo_data['name']}")
        
        # Update repository discovery metadata
        current_state['repository_discovery'] = {
            'last_discovery': datetime.now(timezone.utc).isoformat(),
            'repositories_found': len(repositories),
            'discovery_source': 'github_organization',
            'organization': state_manager.organization
        }
        
        # Update last_updated timestamp
        current_state['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Save the updated state
        print("\n4. Saving updated state...")
        state_manager.save_state_locally(current_state)
        print("   ✓ State saved successfully")
        
        # Verify the fix
        print("\n5. Verifying the fix...")
        verification_state = state_manager.load_state()
        verified_projects = verification_state.get('projects', {})
        print(f"   Verified projects count: {len(verified_projects)}")
        
        if len(verified_projects) == len(repositories):
            print("\n✅ SUCCESS: Repository count issue has been fixed!")
            print(f"   The system now correctly shows {len(verified_projects)} repositories.")
        else:
            print("\n⚠️  WARNING: Verification shows a different count than expected")
            print(f"   Expected: {len(repositories)}, Got: {len(verified_projects)}")
    else:
        print("\n❌ ERROR: No repositories were discovered.")
        print("   Please check:")
        print("   - GitHub token is properly set (CLAUDE_PAT or GITHUB_TOKEN)")
        print("   - The organization name is correct")
        print("   - You have access to the organization's repositories")


if __name__ == "__main__":
    main()