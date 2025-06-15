#!/usr/bin/env python3
"""Fix repository discovery issue - ensure repos are loaded into system state."""

import asyncio
import os
import json
from datetime import datetime, timezone
from scripts.state_manager import StateManager
from scripts.repository_analyzer import RepositoryAnalyzer
from github import Github


async def fix_repository_discovery():
    """Discover repositories and ensure they're saved to state."""
    print("🔍 Starting repository discovery fix...")
    
    # Initialize components
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ GITHUB_TOKEN not found in environment")
        return False
        
    state_manager = StateManager()
    analyzer = RepositoryAnalyzer(github_token)
    g = Github(github_token)
    
    # Get organization
    org_name = "CodeWebMobile-AI"
    try:
        org = g.get_organization(org_name)
        print(f"✓ Connected to organization: {org_name}")
    except Exception as e:
        print(f"❌ Failed to connect to organization: {e}")
        return False
    
    # Discover repositories
    print("\n📂 Discovering repositories...")
    discovered_repos = {}
    
    for repo in org.get_repos():
        if repo.name in ['cwmai', '.github', 'cwmai.git']:
            print(f"  ⏭️  Skipping excluded repository: {repo.name}")
            continue
            
        print(f"  📦 Found repository: {repo.name}")
        
        # Analyze repository
        try:
            analysis = await analyzer.analyze_repository(repo.full_name)
            
            repo_data = {
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description or '',
                'url': repo.html_url,
                'created_at': repo.created_at.isoformat(),
                'updated_at': repo.updated_at.isoformat(),
                'language': repo.language or 'Unknown',
                'stars': repo.stargazers_count,
                'forks': repo.forks_count,
                'open_issues': repo.open_issues_count,
                'has_issues': repo.has_issues,
                'has_wiki': repo.has_wiki,
                'has_downloads': repo.has_downloads,
                'health_score': analysis.get('health_score', 80),
                'active': True,
                'last_commit': analysis.get('last_commit_date', repo.updated_at.isoformat()),
                'tech_stack': analysis.get('tech_stack', []),
                'discovered_at': datetime.now(timezone.utc).isoformat()
            }
            
            discovered_repos[repo.name] = repo_data
            print(f"    ✓ Analyzed: health_score={repo_data['health_score']}")
            
        except Exception as e:
            print(f"    ⚠️  Analysis failed: {e}")
            # Still add basic info
            discovered_repos[repo.name] = {
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description or '',
                'url': repo.html_url,
                'created_at': repo.created_at.isoformat(),
                'updated_at': repo.updated_at.isoformat(),
                'language': repo.language or 'Unknown',
                'health_score': 75,
                'active': True,
                'discovered_at': datetime.now(timezone.utc).isoformat()
            }
    
    print(f"\n✓ Discovered {len(discovered_repos)} repositories")
    
    # Load current state
    current_state = state_manager.load_state()
    print(f"\n📊 Current state has {len(current_state.get('projects', {}))} projects")
    
    # Update state with discovered repositories
    if 'projects' not in current_state:
        current_state['projects'] = {}
        
    current_state['projects'].update(discovered_repos)
    current_state['last_repository_discovery'] = datetime.now(timezone.utc).isoformat()
    current_state['total_active_projects'] = len(discovered_repos)
    
    # Save updated state
    state_manager.state = current_state
    state_manager.save_state()
    print(f"\n💾 Saved {len(discovered_repos)} repositories to state")
    
    # Verify by reloading
    verified_state = state_manager.load_state()
    verified_projects = verified_state.get('projects', {})
    print(f"\n✅ Verification: State now contains {len(verified_projects)} projects")
    
    if verified_projects:
        print("\n📋 Active projects:")
        for name, project in verified_projects.items():
            print(f"  - {name}: {project.get('description', 'No description')[:60]}...")
    
    return True


if __name__ == "__main__":
    # Load environment
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    # Run the fix
    success = asyncio.run(fix_repository_discovery())
    
    if success:
        print("\n🎉 Repository discovery fix completed successfully!")
        print("The system should now see all active projects.")
    else:
        print("\n❌ Repository discovery fix failed!")