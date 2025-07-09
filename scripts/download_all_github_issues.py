#!/usr/bin/env python3
"""
Download All GitHub Issues for Local Repositories

This script downloads all issues from GitHub for each repository we're managing
locally and saves them to the local repository structure.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from github import Github, Auth
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_repository_issues(github_client, org_name: str, repo_name: str, 
                             local_repo_path: Path) -> int:
    """Download all issues for a single repository.
    
    Args:
        github_client: Authenticated GitHub client
        org_name: Organization name
        repo_name: Repository name
        local_repo_path: Path to local repository
        
    Returns:
        Number of issues downloaded
    """
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
        
        logger.info(f"Downloading issues for {repo_name}...")
        
        for issue in repo.get_issues(state='all'):
            issue_data = {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body or "",
                "state": issue.state,
                "created_at": issue.created_at.isoformat(),
                "updated_at": issue.updated_at.isoformat(),
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
                        "created_at": comment.created_at.isoformat(),
                        "updated_at": comment.updated_at.isoformat(),
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
        
        logger.info(f"  ✓ Downloaded {issue_count} issues for {repo_name}")
        return issue_count
        
    except Exception as e:
        logger.error(f"  ✗ Failed to download issues for {repo_name}: {e}")
        return 0


def main():
    """Main function to download all issues."""
    # Get GitHub token
    github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
    if not github_token:
        logger.error("No GitHub token found. Set GITHUB_TOKEN or CLAUDE_PAT environment variable.")
        return 1
    
    # Initialize GitHub client
    auth = Auth.Token(github_token)
    github = Github(auth=auth)
    
    # Organization name
    org_name = "CodeWebMobile-AI"
    
    # Load local repository state
    state_file = Path("../local_repos/state/repositories.json")
    if not state_file.exists():
        logger.error(f"Repository state file not found: {state_file}")
        return 1
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    local_repos = state.get("repositories", [])
    logger.info(f"Found {len(local_repos)} local repositories")
    
    # Download issues for each repository
    total_issues = 0
    success_count = 0
    
    for repo_data in local_repos:
        repo_name = repo_data["name"]
        repo_path = Path(repo_data["path"])
        
        # Make path absolute if relative
        if not repo_path.is_absolute():
            repo_path = Path("..") / repo_path
        
        if repo_path.exists():
            issue_count = download_repository_issues(github, org_name, repo_name, repo_path)
            total_issues += issue_count
            if issue_count > 0:
                success_count += 1
        else:
            logger.warning(f"  ! Local path not found for {repo_name}: {repo_path}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info(f"Download Summary:")
    logger.info(f"  Total repositories: {len(local_repos)}")
    logger.info(f"  Successful downloads: {success_count}")
    logger.info(f"  Total issues downloaded: {total_issues}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())