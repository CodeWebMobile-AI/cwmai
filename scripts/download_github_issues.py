#!/usr/bin/env python3
"""
Download GitHub Issues for Local Mode

Downloads all issues from GitHub repositories and saves them locally
so they can be accessed in local mode without GitHub API calls.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
from github import Github, Auth
import time

from scripts.state_manager import StateManager
from scripts.local_github_simulator import LocalGitHubSimulator


class GitHubIssueDownloader:
    """Downloads GitHub issues for local storage."""
    
    def __init__(self):
        """Initialize the issue downloader."""
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        self.local_simulator = LocalGitHubSimulator()
        
        # Get GitHub token
        self.github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        if not self.github_token:
            raise ValueError("No GitHub token found. Set GITHUB_TOKEN or CLAUDE_PAT environment variable.")
        
        # Initialize GitHub client
        auth = Auth.Token(self.github_token)
        self.github = Github(auth=auth)
        
    def download_all_issues(self) -> Dict[str, Any]:
        """Download all issues for all repositories.
        
        Returns:
            Summary of downloaded issues
        """
        print("Starting GitHub issue download for local mode...")
        
        # Load system state to get repositories
        state = self.state_manager.load_state()
        projects = state.get('projects', {})
        
        if not projects:
            print("No projects found in state.")
            return {'total_repos': 0, 'total_issues': 0}
        
        summary = {
            'total_repos': 0,
            'total_issues': 0,
            'repos_processed': [],
            'errors': []
        }
        
        # Get organization
        org_name = "CodeWebMobile-AI"
        try:
            org = self.github.get_organization(org_name)
        except Exception as e:
            self.logger.error(f"Failed to get organization {org_name}: {e}")
            return summary
        
        # Process each repository
        for repo_name, repo_data in projects.items():
            try:
                print(f"\nProcessing repository: {repo_name}")
                
                # Get the GitHub repository
                try:
                    github_repo = org.get_repo(repo_name)
                except Exception as e:
                    print(f"  ⚠️  Repository {repo_name} not found on GitHub: {e}")
                    summary['errors'].append(f"{repo_name}: Not found on GitHub")
                    continue
                
                # Download issues
                issues_data = self._download_repo_issues(github_repo)
                
                # Save issues locally
                saved_count = self._save_issues_locally(repo_name, issues_data)
                
                print(f"  ✓ Downloaded {saved_count} issues for {repo_name}")
                
                summary['total_repos'] += 1
                summary['total_issues'] += saved_count
                summary['repos_processed'].append({
                    'name': repo_name,
                    'issues_count': saved_count
                })
                
                # Small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                error_msg = f"Error processing {repo_name}: {e}"
                self.logger.error(error_msg)
                summary['errors'].append(error_msg)
                print(f"  ❌ {error_msg}")
        
        # Print summary
        print("\n" + "="*50)
        print("DOWNLOAD SUMMARY")
        print("="*50)
        print(f"Total repositories processed: {summary['total_repos']}")
        print(f"Total issues downloaded: {summary['total_issues']}")
        
        if summary['repos_processed']:
            print("\nRepositories:")
            for repo in summary['repos_processed']:
                print(f"  - {repo['name']}: {repo['issues_count']} issues")
        
        if summary['errors']:
            print("\nErrors:")
            for error in summary['errors']:
                print(f"  - {error}")
        
        return summary
    
    def _download_repo_issues(self, github_repo) -> List[Dict[str, Any]]:
        """Download all issues for a repository.
        
        Args:
            github_repo: GitHub repository object
            
        Returns:
            List of issue data dictionaries
        """
        issues_data = []
        
        try:
            # Get all issues (including closed ones)
            issues = github_repo.get_issues(state='all')
            
            for issue in issues:
                # Skip pull requests
                if issue.pull_request:
                    continue
                
                # Extract issue data
                issue_data = {
                    'number': issue.number,
                    'title': issue.title,
                    'body': issue.body or '',
                    'state': issue.state,
                    'created_at': issue.created_at.isoformat() if issue.created_at else None,
                    'updated_at': issue.updated_at.isoformat() if issue.updated_at else None,
                    'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                    'author': issue.user.login if issue.user else 'unknown',
                    'labels': [label.name for label in issue.labels],
                    'assignees': [assignee.login for assignee in issue.assignees],
                    'comments_count': issue.comments,
                    'html_url': issue.html_url,
                    'milestone': issue.milestone.title if issue.milestone else None
                }
                
                # Get comments if any
                if issue.comments > 0:
                    comments = []
                    for comment in issue.get_comments():
                        comments.append({
                            'author': comment.user.login if comment.user else 'unknown',
                            'body': comment.body,
                            'created_at': comment.created_at.isoformat() if comment.created_at else None
                        })
                    issue_data['comments'] = comments
                else:
                    issue_data['comments'] = []
                
                issues_data.append(issue_data)
                
        except Exception as e:
            self.logger.error(f"Error downloading issues: {e}")
        
        return issues_data
    
    def _save_issues_locally(self, repo_name: str, issues_data: List[Dict[str, Any]]) -> int:
        """Save issues to local storage.
        
        Args:
            repo_name: Repository name
            issues_data: List of issue data
            
        Returns:
            Number of issues saved
        """
        if not issues_data:
            return 0
        
        try:
            # Get or create local repository
            org_name = "CodeWebMobile-AI"
            try:
                local_repo = self.local_simulator.get_repository(f"{org_name}/{repo_name}")
            except:
                # Create local repo if it doesn't exist
                local_repo = self.local_simulator.create_repository(
                    name=repo_name,
                    description=f"Local copy of {repo_name}",
                    organization=org_name
                )
            
            # Clear existing issues
            issues_file = Path(local_repo.path) / '.github' / 'issues.json'
            issues_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save all issues
            saved_issues = []
            for issue_data in issues_data:
                # Create issue in local storage
                local_issue = self.local_simulator.create_issue(
                    repo_full_name=f"{org_name}/{repo_name}",
                    title=issue_data['title'],
                    body=issue_data['body'],
                    labels=issue_data['labels']
                )
                
                # Update issue with additional data
                issue_file = Path(local_repo.path) / '.github' / 'issues' / f"{local_issue.number}.json"
                issue_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Merge local issue data with downloaded data
                full_issue_data = {
                    **issue_data,
                    'number': local_issue.number,
                    'downloaded_at': datetime.now(timezone.utc).isoformat(),
                    'local_storage': True
                }
                
                with open(issue_file, 'w') as f:
                    json.dump(full_issue_data, f, indent=2)
                
                saved_issues.append(full_issue_data)
            
            # Also save a summary file
            with open(issues_file, 'w') as f:
                json.dump({
                    'repository': repo_name,
                    'downloaded_at': datetime.now(timezone.utc).isoformat(),
                    'total_issues': len(saved_issues),
                    'issues': saved_issues
                }, f, indent=2)
            
            return len(saved_issues)
            
        except Exception as e:
            self.logger.error(f"Error saving issues locally: {e}")
            return 0


def main():
    """Main function to download all GitHub issues."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        downloader = GitHubIssueDownloader()
        summary = downloader.download_all_issues()
        
        # Save summary to file
        summary_file = "github_issues_download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                **summary,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())