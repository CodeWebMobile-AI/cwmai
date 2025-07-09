#!/usr/bin/env python3
"""
Clone and Sync CodeWebMobile-AI Repositories with Issues

This script:
1. Clones all repositories from the CodeWebMobile-AI organization
2. Downloads all issues for each repository
3. Sets up the local repository structure for the CWMAI system
"""

import os
import subprocess
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Repositories to exclude (add names here)
EXCLUDED_REPOS = [
    ".github",  # GitHub organization profile repository
    "cwmai",    # The main CWMAI repository (this is the current project)
]


def load_env_file():
    """Load environment variables from .env.local file."""
    env_file = Path('.env.local')
    if env_file.exists():
        logger.info("Loading environment variables from .env.local...")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    try:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
                    except ValueError:
                        continue


def get_github_repos(org_name: str = "CodeWebMobile-AI") -> List[Dict]:
    """Get all repositories from the organization using GitHub API."""
    try:
        from github import Github, Auth
        
        # Get GitHub token
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        if not github_token:
            logger.error("No GitHub token found in environment variables")
            logger.error("Please set GITHUB_TOKEN or CLAUDE_PAT")
            return []
        
        logger.info(f"🔑 Using GitHub API to fetch repositories from {org_name}...")
        auth = Auth.Token(github_token)
        g = Github(auth=auth)
        
        # Get organization
        org = g.get_organization(org_name)
        
        # Get all repos
        repos = []
        for repo in org.get_repos():
            if repo.name not in EXCLUDED_REPOS:
                repos.append({
                    'name': repo.name,
                    'full_name': repo.full_name,
                    'description': repo.description,
                    'clone_url': repo.clone_url,
                    'ssh_url': repo.ssh_url,
                    'created_at': repo.created_at.isoformat() if repo.created_at else None,
                    'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                    'default_branch': repo.default_branch,
                    'private': repo.private,
                    'archived': repo.archived
                })
            else:
                logger.info(f"⏭️  Excluding {repo.name}")
        
        return repos
        
    except Exception as e:
        logger.error(f"Error fetching repositories: {e}")
        return []


def clone_repository(repo_info: Dict, target_dir: Path) -> bool:
    """Clone a single repository."""
    repo_name = repo_info['name']
    repo_path = target_dir / repo_name
    
    if repo_path.exists():
        logger.info(f"⏭️  Repository {repo_name} already exists, pulling latest...")
        try:
            # Pull latest changes
            subprocess.run(
                ['git', 'pull'],
                cwd=str(repo_path),
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✅ Updated {repo_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Could not pull updates for {repo_name}: {e}")
            return True  # Still return True since repo exists
    
    logger.info(f"📥 Cloning {repo_name}...")
    
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
            logger.info(f"   Using HTTPS clone URL")
        else:
            logger.info(f"   Using SSH clone URL")
        
        # Clone the repository
        subprocess.run(
            ['git', 'clone', clone_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"✅ Successfully cloned {repo_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to clone {repo_name}: {e}")
        if e.stderr:
            logger.error(f"   Error: {e.stderr}")
        return False


def download_repository_issues(org_name: str, repo_name: str, 
                             local_repo_path: Path, github_client) -> int:
    """Download all issues for a single repository."""
    try:
        # Get repository
        repo = github_client.get_repo(f"{org_name}/{repo_name}")
        
        # Create issues directory
        issues_dir = local_repo_path / "issues"
        issues_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file if it doesn't exist
        metadata_file = local_repo_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "repository": repo_name,
                "organization": org_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "next_issue_number": 1
            }
        
        # Download all issues
        issue_count = 0
        highest_issue_number = 0
        
        logger.info(f"   📋 Downloading issues for {repo_name}...")
        
        for issue in repo.get_issues(state='all'):
            issue_data = {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body or "",
                "state": issue.state,
                "created_at": issue.created_at.isoformat() if issue.created_at else None,
                "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
                "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
                "labels": [label.name for label in issue.labels],
                "assignees": [assignee.login for assignee in issue.assignees],
                "milestone": issue.milestone.title if issue.milestone else None,
                "comments": issue.comments,
                "html_url": issue.html_url,
                "user": issue.user.login if issue.user else None
            }
            
            # Save issue to file
            issue_file = issues_dir / f"issue-{issue.number}.json"
            with open(issue_file, 'w') as f:
                json.dump(issue_data, f, indent=2)
            
            issue_count += 1
            highest_issue_number = max(highest_issue_number, issue.number)
            
            # Also download comments if any
            if issue.comments > 0:
                comments = []
                for comment in issue.get_comments():
                    comments.append({
                        "id": comment.id,
                        "body": comment.body,
                        "created_at": comment.created_at.isoformat() if comment.created_at else None,
                        "updated_at": comment.updated_at.isoformat() if comment.updated_at else None,
                        "user": comment.user.login if comment.user else None
                    })
                
                # Save comments
                comments_file = issues_dir / f"issue-{issue.number}-comments.json"
                with open(comments_file, 'w') as f:
                    json.dump(comments, f, indent=2)
        
        # Update metadata with next issue number
        metadata["next_issue_number"] = highest_issue_number + 1
        metadata["last_issue_sync"] = datetime.now(timezone.utc).isoformat()
        metadata["total_issues"] = issue_count
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"   ✓ Downloaded {issue_count} issues")
        return issue_count
        
    except Exception as e:
        logger.error(f"   ✗ Failed to download issues for {repo_name}: {e}")
        return 0


def update_local_state(target_dir: Path, repos: List[Dict], org_name: str):
    """Update the local repository state file."""
    # State is stored in the local_repos directory, not in the org subdirectory
    state_dir = Path("local_repos/state")
    state_dir.mkdir(exist_ok=True, parents=True)
    
    state_file = state_dir / "repositories.json"
    
    # Create repository entries
    local_repos = []
    for repo in repos:
        local_repos.append({
            "name": repo['name'],
            "path": f"local_repos/repositories/CodeWebMobile-AI/{repo['name']}",
            "description": repo['description'] or "",
            "created_at": repo['created_at'],
            "updated_at": repo['updated_at'],
            "has_issues": True,
            "organization": org_name
        })
    
    state = {
        "organization": org_name,
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "repositories": local_repos,
        "total_repositories": len(local_repos)
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"💾 Updated local state file: {state_file}")


def main():
    """Main function to clone repositories and download issues."""
    print("🚀 CodeWebMobile-AI Repository & Issues Sync")
    print("=" * 60)
    
    # Load environment variables
    load_env_file()
    
    # Organization name
    org_name = "CodeWebMobile-AI"
    
    # Determine target directory
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        # Use the correct local repository structure
        target_dir = Path("local_repos/repositories/CodeWebMobile-AI")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Target directory: {target_dir}")
    
    # Get repositories
    repos = get_github_repos(org_name)
    
    if not repos:
        logger.error("❌ No repositories found")
        return 1
    
    logger.info(f"\n📊 Found {len(repos)} repositories (excluding {len(EXCLUDED_REPOS)} excluded)")
    print("=" * 60)
    
    # Initialize GitHub client for issues
    from github import Github, Auth
    github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
    auth = Auth.Token(github_token)
    github_client = Github(auth=auth)
    
    # Clone each repository and download issues
    success_count = 0
    failed_repos = []
    total_issues = 0
    
    for i, repo in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] Processing: {repo['name']}")
        if repo['description']:
            print(f"   Description: {repo['description']}")
        
        # Clone repository
        if clone_repository(repo, target_dir):
            success_count += 1
            
            # Download issues
            repo_path = target_dir / repo['name']
            issue_count = download_repository_issues(org_name, repo['name'], 
                                                   repo_path, github_client)
            total_issues += issue_count
        else:
            failed_repos.append(repo['name'])
    
    # Update local state
    update_local_state(target_dir, repos, org_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Sync Summary:")
    print(f"   ✅ Successfully synced: {success_count}/{len(repos)} repositories")
    print(f"   📋 Total issues downloaded: {total_issues}")
    
    if failed_repos:
        print(f"\n❌ Failed repositories:")
        for repo in failed_repos:
            print(f"   - {repo}")
    
    # Create a summary README
    readme_path = target_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# CodeWebMobile-AI Local Repositories

This directory contains all repositories from the {org_name} organization.

## Summary
- **Organization**: {org_name}
- **Sync Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Repositories**: {len(repos)}
- **Successfully Synced**: {success_count}
- **Total Issues**: {total_issues}

## Repositories

""")
        for repo in repos:
            status = "✅" if repo['name'] not in failed_repos else "❌"
            repo_path = target_dir / repo['name']
            issue_count = 0
            if repo_path.exists():
                issues_dir = repo_path / "issues"
                if issues_dir.exists():
                    issue_count = len(list(issues_dir.glob("issue-*.json")))
            
            f.write(f"{status} **{repo['name']}** - {repo['description'] or 'No description'}\n")
            f.write(f"   - Issues: {issue_count}\n")
            f.write(f"   - Created: {repo['created_at']}\n")
            f.write(f"   - Updated: {repo['updated_at']}\n\n")
    
    logger.info(f"\n📄 Summary README created: {readme_path}")
    logger.info(f"\n✅ Sync operation completed!")
    logger.info(f"   All repositories are in: {target_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())