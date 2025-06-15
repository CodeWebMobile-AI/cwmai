#!/usr/bin/env python3
"""
Fix the repository discovery issue where repositories are found but not saved to state.

This script will:
1. Run the discovery process
2. Actually save the discovered repositories to the state
3. Verify the fix worked
"""

import os
import sys
import json
from datetime import datetime, timezone

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from state_manager import StateManager
from repository_exclusion import should_process_repo

def main():
    """Main function to fix repository discovery."""
    print("=== FIXING REPOSITORY DISCOVERY ISSUE ===\n")
    
    # Initialize state manager
    state_manager = StateManager()
    
    # Load current state
    print("1. Loading current state...")
    current_state = state_manager.load_state()
    
    # Show current projects
    current_projects = current_state.get('projects', {})
    print(f"   Current projects in state: {len(current_projects)}")
    if current_projects:
        for proj_id in current_projects:
            print(f"   - {proj_id}")
    else:
        print("   ⚠️  No projects currently in state!")
    
    # Run discovery with repository loading
    print("\n2. Running repository discovery with proper state update...")
    state_with_repos = state_manager.load_state_with_repository_discovery()
    
    # Check if repositories were added
    updated_projects = state_with_repos.get('projects', {})
    print(f"\n3. After discovery:")
    print(f"   Projects in state: {len(updated_projects)}")
    
    if updated_projects:
        print("\n   Discovered repositories:")
        for proj_id, proj_data in updated_projects.items():
            name = proj_data.get('name', proj_id)
            status = proj_data.get('status', 'unknown')
            health = proj_data.get('health_score', 0)
            print(f"   - {name} (status: {status}, health: {health:.1f})")
    else:
        print("   ⚠️  Still no projects after discovery!")
        
        # Try manual discovery to debug
        print("\n4. Attempting manual discovery to diagnose issue...")
        discovered = state_manager.discover_organization_repositories()
        print(f"   Manually discovered {len(discovered)} repositories")
        
        if discovered:
            print("\n   Manually discovered repos:")
            for repo in discovered:
                print(f"   - {repo.get('name')} ({repo.get('full_name')})")
                
            # Manually add them to state
            print("\n5. Manually adding discovered repositories to state...")
            if 'projects' not in current_state:
                current_state['projects'] = {}
                
            for repo_data in discovered:
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
                    'action_history': [
                        {
                            "action": "repository_discovered",
                            "details": f"Repository {repo_data['name']} discovered and integrated into system",
                            "outcome": "success_discovered",
                            "timestamp": repo_data['discovered_at']
                        }
                    ]
                }
            
            # Update discovery metadata
            current_state['repository_discovery'] = {
                'last_discovery': datetime.now(timezone.utc).isoformat(),
                'repositories_found': len(discovered),
                'discovery_source': 'github_organization',
                'organization': 'CodeWebMobile-AI'
            }
            
            # Save the state
            state_manager.save_state_locally(current_state)
            print(f"   ✓ Manually saved {len(discovered)} repositories to state")
    
    # Verify the fix
    print("\n6. Verifying the fix...")
    verification_state = state_manager.load_state()
    final_projects = verification_state.get('projects', {})
    
    print(f"\n=== RESULTS ===")
    print(f"Projects in state after fix: {len(final_projects)}")
    
    if final_projects:
        print("\nActive repositories:")
        for proj_id, proj_data in final_projects.items():
            name = proj_data.get('name', proj_id)
            status = proj_data.get('status', 'unknown')
            health = proj_data.get('health_score', 0)
            last_checked = proj_data.get('last_checked', 'never')
            print(f"  ✓ {name}")
            print(f"    Status: {status}, Health: {health:.1f}")
            print(f"    Last checked: {last_checked}")
        
        print(f"\n✅ SUCCESS: {len(final_projects)} repositories are now properly loaded in system state!")
    else:
        print("\n❌ FAILED: No repositories in state. There may be an issue with:")
        print("   - GitHub token/authentication")
        print("   - Organization access permissions")
        print("   - Repository exclusion rules")
        
        # Check exclusion rules
        print("\nChecking exclusion rules...")
        from repository_exclusion import RepositoryExclusion
        excluded = RepositoryExclusion.get_excluded_repos_list()
        print(f"Excluded repositories: {excluded}")

if __name__ == "__main__":
    main()