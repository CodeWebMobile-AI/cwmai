#!/usr/bin/env python3
"""
Clone All CodeWebMobile-AI Repositories

This script clones all repositories from the CodeWebMobile-AI organization
to the local file system.
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# Add scripts to path
sys.path.append(str(Path(__file__).parent))


def get_github_repos():
    """Get all repositories from CodeWebMobile-AI organization using GitHub API."""
    try:
        from github import Github
        
        # Get GitHub token
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        if not github_token:
            print("❌ No GitHub token found in environment variables")
            print("   Please set GITHUB_TOKEN or CLAUDE_PAT")
            return []
        
        print("🔑 Using GitHub API to fetch repositories...")
        g = Github(github_token)
        
        # Get organization
        org = g.get_organization("CodeWebMobile-AI")
        
        # Get all repos
        repos = []
        for repo in org.get_repos():
            repos.append({
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description,
                'clone_url': repo.clone_url,
                'ssh_url': repo.ssh_url,
                'created_at': repo.created_at,
                'updated_at': repo.updated_at,
                'default_branch': repo.default_branch,
                'private': repo.private
            })
        
        return repos
        
    except Exception as e:
        print(f"❌ Error fetching repositories: {e}")
        return []


def clone_repository(repo_info, target_dir):
    """Clone a single repository."""
    repo_name = repo_info['name']
    repo_path = target_dir / repo_name
    
    if repo_path.exists():
        print(f"⏭️  Skipping {repo_name} - already exists")
        return True
    
    print(f"📥 Cloning {repo_name}...")
    
    try:
        # Try SSH first, fall back to HTTPS
        clone_url = repo_info['ssh_url']
        
        # Test SSH access
        test_cmd = subprocess.run(
            ['ssh', '-T', 'git@github.com'],
            capture_output=True,
            text=True
        )
        
        # If SSH fails, use HTTPS
        if test_cmd.returncode != 1:  # SSH to GitHub returns 1 on success
            clone_url = repo_info['clone_url']
            print(f"   Using HTTPS clone URL")
        else:
            print(f"   Using SSH clone URL")
        
        # Clone the repository
        subprocess.run(
            ['git', 'clone', clone_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"✅ Successfully cloned {repo_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone {repo_name}: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def main():
    """Main function to clone all repositories."""
    print("🚀 CodeWebMobile-AI Repository Cloner")
    print("=" * 60)
    
    # Determine target directory
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path.home() / "CodeWebMobile-AI-Repos"
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Target directory: {target_dir}")
    
    # Export environment variables
    env_file = Path('.env.local')
    if env_file.exists():
        print("🔧 Loading environment variables from .env.local...")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Get repositories
    repos = get_github_repos()
    
    if not repos:
        print("❌ No repositories found")
        return
    
    print(f"\n📊 Found {len(repos)} repositories")
    print("=" * 60)
    
    # Clone each repository
    success_count = 0
    failed_repos = []
    
    for i, repo in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] Repository: {repo['name']}")
        print(f"   Description: {repo['description']}")
        
        if clone_repository(repo, target_dir):
            success_count += 1
        else:
            failed_repos.append(repo['name'])
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Clone Summary:")
    print(f"   ✅ Successfully cloned: {success_count}")
    print(f"   ❌ Failed: {len(failed_repos)}")
    
    if failed_repos:
        print(f"\n❌ Failed repositories:")
        for repo in failed_repos:
            print(f"   - {repo}")
    
    # Save repository metadata
    metadata_file = target_dir / "repositories_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'organization': 'CodeWebMobile-AI',
            'clone_date': datetime.now().isoformat(),
            'total_repos': len(repos),
            'successfully_cloned': success_count,
            'repositories': repos
        }, f, indent=2, default=str)
    
    print(f"\n💾 Repository metadata saved to: {metadata_file}")
    
    # Create a summary README
    readme_path = target_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# CodeWebMobile-AI Repositories

This directory contains all repositories from the CodeWebMobile-AI organization.

## Summary
- **Organization**: CodeWebMobile-AI
- **Clone Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Repositories**: {len(repos)}
- **Successfully Cloned**: {success_count}

## Repositories

""")
        for repo in repos:
            status = "✅" if repo['name'] not in failed_repos else "❌"
            f.write(f"{status} **{repo['name']}** - {repo['description'] or 'No description'}\n")
            f.write(f"   - Created: {repo['created_at']}\n")
            f.write(f"   - Updated: {repo['updated_at']}\n\n")
    
    print(f"📄 Summary README created: {readme_path}")
    print(f"\n✅ Clone operation completed!")
    print(f"   All repositories are in: {target_dir}")


if __name__ == '__main__':
    main()