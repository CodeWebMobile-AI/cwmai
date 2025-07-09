#!/usr/bin/env python3
"""
Migrate GitHub Repositories to Local Mode

This script clones all repositories from the CodeWebMobile-AI organization
and imports them into the local GitHub simulator.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime
import json

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.local_issue_creator import LocalIssueCreator


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
        try:
            org = g.get_organization("CodeWebMobile-AI")
        except Exception as e:
            print(f"❌ Could not access organization: {e}")
            # Try to get repos directly
            user = g.get_user()
            repos = []
            for repo in user.get_repos():
                if repo.full_name.startswith("CodeWebMobile-AI/"):
                    repos.append({
                        'name': repo.name,
                        'full_name': repo.full_name,
                        'description': repo.description,
                        'clone_url': repo.clone_url,
                        'ssh_url': repo.ssh_url,
                        'created_at': repo.created_at,
                        'updated_at': repo.updated_at,
                        'default_branch': repo.default_branch,
                        'private': repo.private,
                        'open_issues_count': repo.open_issues_count
                    })
            return repos
        
        # Get all repos from org
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
                'private': repo.private,
                'open_issues_count': repo.open_issues_count
            })
        
        return repos
        
    except Exception as e:
        print(f"❌ Error fetching repositories: {e}")
        return []


def import_to_local_simulator(repo_info, repo_path, simulator):
    """Import a cloned repository into the local simulator."""
    try:
        repo_name = repo_info['name']
        
        # Check if already exists in local simulator
        existing_repo = simulator.get_repository(f"CodeWebMobile-AI/{repo_name}")
        if existing_repo:
            print(f"   ⏭️  Already in local simulator")
            return True
        
        # Create repository in simulator
        print(f"   📝 Registering in local simulator...")
        
        # Add to repositories state
        repo_data = {
            "name": repo_name,
            "full_name": f"CodeWebMobile-AI/{repo_name}",
            "description": repo_info['description'] or "",
            "path": str(repo_path),
            "created_at": repo_info['created_at'].isoformat() if hasattr(repo_info['created_at'], 'isoformat') else str(repo_info['created_at']),
            "updated_at": repo_info['updated_at'].isoformat() if hasattr(repo_info['updated_at'], 'isoformat') else str(repo_info['updated_at']),
            "default_branch": repo_info['default_branch'],
            "private": repo_info['private'],
            "html_url": f"file://{repo_path}"
        }
        
        simulator.repos_state["repositories"].append(repo_data)
        simulator.repos_state["total_count"] += 1
        simulator._save_state()
        
        # Create issues directory if it doesn't exist
        issues_dir = repo_path / "issues"
        issues_dir.mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            "name": repo_name,
            "full_name": f"CodeWebMobile-AI/{repo_name}",
            "description": repo_info['description'] or "",
            "created_at": str(repo_info['created_at']),
            "updated_at": str(repo_info['updated_at']),
            "next_issue_number": 1,
            "organization": "CodeWebMobile-AI",
            "imported_from_github": True,
            "import_date": datetime.now().isoformat(),
            "github_open_issues": repo_info.get('open_issues_count', 0)
        }
        
        metadata_path = repo_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Successfully imported to local simulator")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to import: {e}")
        return False


def clone_and_import_repository(repo_info, target_dir, simulator):
    """Clone a repository and import it into the local simulator."""
    repo_name = repo_info['name']
    org_dir = target_dir / "repositories" / "CodeWebMobile-AI"
    org_dir.mkdir(parents=True, exist_ok=True)
    
    repo_path = org_dir / repo_name
    
    # Check if already cloned
    if repo_path.exists():
        print(f"   ✅ Already cloned")
        # Still try to import to simulator
        return import_to_local_simulator(repo_info, repo_path, simulator)
    
    print(f"   📥 Cloning repository...")
    
    try:
        # Clone using HTTPS (more reliable)
        clone_url = repo_info['clone_url']
        
        subprocess.run(
            ['git', 'clone', clone_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"   ✅ Successfully cloned")
        
        # Import to local simulator
        return import_to_local_simulator(repo_info, repo_path, simulator)
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to clone: {e}")
        if e.stderr:
            print(f"      Error: {e.stderr}")
        return False


def migrate_github_issues(repo_info, simulator):
    """Migrate GitHub issues to local format (placeholder for future implementation)."""
    # This could be expanded to actually fetch and migrate issues
    print(f"   ℹ️  GitHub issues: {repo_info.get('open_issues_count', 0)} open issues (migration not implemented)")


def main():
    """Main function to migrate repositories."""
    print("🚀 GitHub to Local Repository Migration")
    print("=" * 60)
    
    # Set up paths
    local_repos_dir = Path("local_repos")
    local_repos_dir.mkdir(exist_ok=True)
    
    # Export environment variables
    env_file = Path('.env.local')
    if env_file.exists():
        print("🔧 Loading environment variables from .env.local...")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    # Initialize local simulator
    print("🏠 Initializing local GitHub simulator...")
    simulator = LocalGitHubSimulator(base_path=str(local_repos_dir))
    
    # Get repositories from GitHub
    repos = get_github_repos()
    
    if not repos:
        print("❌ No repositories found")
        return
    
    print(f"\n📊 Found {len(repos)} repositories on GitHub")
    print("=" * 60)
    
    # Exclude cwmai repository
    repos = [r for r in repos if r['name'] != 'cwmai']
    print(f"📋 Will migrate {len(repos)} repositories (excluding cwmai)")
    
    # Clone and import each repository
    success_count = 0
    failed_repos = []
    
    for i, repo in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] {repo['name']}")
        print(f"   Description: {repo['description'] or 'No description'}")
        
        if clone_and_import_repository(repo, local_repos_dir, simulator):
            success_count += 1
            # Optionally migrate issues
            migrate_github_issues(repo, simulator)
        else:
            failed_repos.append(repo['name'])
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Migration Summary:")
    print(f"   ✅ Successfully migrated: {success_count}")
    print(f"   ❌ Failed: {len(failed_repos)}")
    
    if failed_repos:
        print(f"\n❌ Failed repositories:")
        for repo in failed_repos:
            print(f"   - {repo}")
    
    # Save migration report
    report_file = local_repos_dir / "migration_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            'migration_date': datetime.now().isoformat(),
            'source': 'GitHub',
            'organization': 'CodeWebMobile-AI',
            'total_repos': len(repos),
            'successfully_migrated': success_count,
            'failed_repos': failed_repos,
            'repositories': repos
        }, f, indent=2, default=str)
    
    print(f"\n💾 Migration report saved to: {report_file}")
    
    # Show current local repos
    print(f"\n📂 Current local repositories:")
    local_repos = simulator.list_repositories()
    for repo in local_repos:
        print(f"   - {repo.full_name}")
    
    print(f"\n✅ Migration completed!")
    print(f"   Repositories are now available in local mode")
    print(f"\n💡 To use local mode, run:")
    print(f"   python3 cwmai_local.py switch-local")
    print(f"   python3 cwmai_local.py list-repos")


if __name__ == '__main__':
    main()