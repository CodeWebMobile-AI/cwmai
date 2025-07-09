#!/usr/bin/env python3
"""
Reconcile repository discovery - add missing repositories to repositories.json
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone


def main():
    # Paths
    repos_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    repos_json_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/state/repositories.json")
    
    # Read current repositories.json
    with open(repos_json_path, 'r') as f:
        repos_data = json.load(f)
    
    # Get currently registered repositories
    registered_repos = {repo['name'] for repo in repos_data['repositories']}
    print(f"Currently registered repositories ({len(registered_repos)}): {sorted(registered_repos)}")
    
    # Scan actual directories
    actual_repos = []
    if repos_dir.exists():
        for item in repos_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                actual_repos.append(item.name)
    
    actual_repos_set = set(actual_repos)
    print(f"\nActual directories found ({len(actual_repos_set)}): {sorted(actual_repos_set)}")
    
    # Find missing repositories
    missing_repos = actual_repos_set - registered_repos
    print(f"\nMissing from repositories.json ({len(missing_repos)}): {sorted(missing_repos)}")
    
    if missing_repos:
        print("\nAdding missing repositories to repositories.json...")
        
        for repo_name in sorted(missing_repos):
            repo_path = repos_dir / repo_name
            
            # Try to read metadata.json if it exists
            metadata_path = repo_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                description = metadata.get('description', '')
            else:
                # Default description
                description = f"AI-powered {repo_name.replace('-', ' ')} application"
            
            # Create repository entry
            repo_entry = {
                "name": repo_name,
                "full_name": f"CodeWebMobile-AI/{repo_name}",
                "description": description,
                "path": str(repo_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "default_branch": "main",
                "private": False,
                "html_url": f"file://{repo_path}",
                "language": None,
                "languages_url": None,
                "open_issues_count": 0,
                "forks_count": 0,
                "stargazers_count": 0
            }
            
            repos_data['repositories'].append(repo_entry)
            print(f"  ✅ Added: {repo_name}")
        
        # Update total count
        repos_data['total_count'] = len(repos_data['repositories'])
        repos_data['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Save updated repositories.json
        with open(repos_json_path, 'w') as f:
            json.dump(repos_data, f, indent=2)
        
        print(f"\n✅ Updated repositories.json with {len(missing_repos)} new repositories")
        print(f"Total repositories now: {repos_data['total_count']}")
    else:
        print("\n✅ No missing repositories found - repositories.json is up to date")
    
    # Check for entries in repositories.json that don't have directories
    registered_but_missing = registered_repos - actual_repos_set
    if registered_but_missing:
        print(f"\n⚠️  Warning: These repositories are in repositories.json but have no directory:")
        for repo in sorted(registered_but_missing):
            print(f"  - {repo}")


if __name__ == "__main__":
    main()