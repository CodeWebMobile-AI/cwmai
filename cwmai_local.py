#!/usr/bin/env python3
"""
CWMAI Local Mode CLI

Command-line interface for local storage mode operations.
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.storage_factory import StorageFactory, StorageMode
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.local_issue_creator import LocalIssueCreator


def switch_to_local():
    """Switch to local storage mode."""
    print("🏠 Switching to local storage mode...")
    os.environ['STORAGE_MODE'] = 'local'
    
    # Update .env.local
    env_file = Path('.env.local')
    if env_file.exists():
        lines = env_file.read_text().splitlines()
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('STORAGE_MODE='):
                lines[i] = 'STORAGE_MODE=local'
                updated = True
                break
        
        if not updated:
            lines.append('STORAGE_MODE=local')
        
        env_file.write_text('\n'.join(lines) + '\n')
    
    print("✅ Now using local storage (no GitHub API calls)")


def switch_to_github():
    """Switch to GitHub storage mode."""
    print("🌐 Switching to GitHub storage mode...")
    os.environ['STORAGE_MODE'] = 'github'
    
    # Update .env.local
    env_file = Path('.env.local')
    if env_file.exists():
        lines = env_file.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith('STORAGE_MODE='):
                lines[i] = 'STORAGE_MODE=github'
                break
        
        env_file.write_text('\n'.join(lines) + '\n')
    
    print("✅ Now using GitHub API")


def list_local_repos():
    """List all local repositories."""
    sim = LocalGitHubSimulator()
    repos = sim.list_repositories()
    
    print(f"\n📂 Found {len(repos)} local repositories:")
    print("=" * 60)
    
    for repo in repos:
        print(f"\n📁 {repo.full_name}")
        print(f"   Description: {repo.description}")
        print(f"   Path: {repo.path}")
        
        # Get issue counts
        open_issues = sim.get_issues(repo.full_name, state='open')
        closed_issues = sim.get_issues(repo.full_name, state='closed')
        
        print(f"   Issues: {len(open_issues)} open, {len(closed_issues)} closed")
        
        # Show recent open issues
        if open_issues:
            print("   Recent open issues:")
            for issue in open_issues[:3]:
                print(f"     - #{issue.number}: {issue.title}")


def show_local_issues(repo_name: str = None):
    """Show issues from local repositories."""
    sim = LocalGitHubSimulator()
    
    if repo_name:
        # Show issues for specific repo
        full_name = repo_name if '/' in repo_name else f"CodeWebMobile-AI/{repo_name}"
        issues = sim.get_issues(full_name, state='all')
        
        print(f"\n📋 Issues for {full_name}:")
        print("=" * 60)
        
        for issue in issues:
            status = "✅" if issue.state == "closed" else "🔴"
            print(f"{status} #{issue.number}: {issue.title}")
            print(f"   Labels: {', '.join(issue.labels)}")
            print(f"   Created: {issue.created_at.strftime('%Y-%m-%d %H:%M')}")
            if issue.state == "closed" and issue.closed_at:
                print(f"   Closed: {issue.closed_at.strftime('%Y-%m-%d %H:%M')}")
            print()
    else:
        # Show all open issues
        all_issues = []
        for repo in sim.list_repositories():
            issues = sim.get_issues(repo.full_name, state='open')
            for issue in issues:
                all_issues.append((repo, issue))
        
        print(f"\n📋 All open issues ({len(all_issues)} total):")
        print("=" * 60)
        
        for repo, issue in all_issues:
            print(f"🔴 {repo.name}/#{issue.number}: {issue.title}")
            print(f"   Created: {issue.created_at.strftime('%Y-%m-%d %H:%M')}")
            print()


async def create_local_issue(repo_name: str, title: str, description: str = ""):
    """Create a new issue in a local repository."""
    sim = LocalGitHubSimulator()
    
    full_name = repo_name if '/' in repo_name else f"CodeWebMobile-AI/{repo_name}"
    
    # Check if repo exists
    repo = sim.get_repository(full_name)
    if not repo:
        print(f"❌ Repository {full_name} does not exist")
        return
    
    # Create issue
    issue = sim.create_issue(
        repo_full_name=full_name,
        title=title,
        body=description or f"Issue created via CLI at {datetime.now()}",
        labels=["manual", "priority:medium"]
    )
    
    print(f"✅ Created issue #{issue.number} in {full_name}")
    print(f"   Title: {issue.title}")


async def process_next_task():
    """Process the next available task."""
    factory = StorageFactory()
    if not factory.is_local_mode():
        print("❌ Not in local mode. Run: cwmai_local.py switch-local")
        return
    
    sim = LocalGitHubSimulator()
    
    # Find next open issue
    for repo in sim.list_repositories():
        issues = sim.get_issues(repo.full_name, state='open')
        if issues:
            issue = issues[0]
            print(f"\n📋 Processing task:")
            print(f"   Issue: #{issue.number} - {issue.title}")
            print(f"   Repository: {repo.full_name}")
            print(f"   Created: {issue.created_at}")
            
            # Simulate processing
            print("\n⚙️  Processing... (simulation)")
            await asyncio.sleep(2)
            
            # Mark as closed
            sim.update_issue(repo.full_name, issue.number, state='closed')
            print("✅ Task completed and issue closed")
            return
    
    print("✅ No open tasks found")


def show_status():
    """Show current storage mode and statistics."""
    factory = StorageFactory()
    sim = LocalGitHubSimulator() if factory.is_local_mode() else None
    
    print("\n📊 CWMAI Status")
    print("=" * 60)
    print(f"Storage Mode: {'🏠 Local' if factory.is_local_mode() else '🌐 GitHub'}")
    
    if sim:
        repos = sim.list_repositories()
        total_issues = 0
        open_issues = 0
        
        for repo in repos:
            all_issues = sim.get_issues(repo.full_name, state='all')
            total_issues += len(all_issues)
            open_issues += len([i for i in all_issues if i.state == 'open'])
        
        print(f"\nLocal Statistics:")
        print(f"  Repositories: {len(repos)}")
        print(f"  Total Issues: {total_issues}")
        print(f"  Open Issues: {open_issues}")
        print(f"  Closed Issues: {total_issues - open_issues}")
    else:
        print("\nGitHub mode active - use GitHub CLI or web interface for statistics")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='CWMAI Local Mode CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Switch commands
    subparsers.add_parser('switch-local', help='Switch to local storage mode')
    subparsers.add_parser('switch-github', help='Switch to GitHub storage mode')
    
    # Repository commands
    subparsers.add_parser('list-repos', help='List local repositories')
    
    # Issue commands
    issues_parser = subparsers.add_parser('list-issues', help='List issues')
    issues_parser.add_argument('--repo', help='Filter by repository name')
    
    create_parser = subparsers.add_parser('create-issue', help='Create a new issue')
    create_parser.add_argument('repo', help='Repository name')
    create_parser.add_argument('title', help='Issue title')
    create_parser.add_argument('--description', help='Issue description', default='')
    
    # Task processing
    subparsers.add_parser('process-task', help='Process next available task')
    
    # Status
    subparsers.add_parser('status', help='Show current status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'switch-local':
        switch_to_local()
    elif args.command == 'switch-github':
        switch_to_github()
    elif args.command == 'list-repos':
        list_local_repos()
    elif args.command == 'list-issues':
        show_local_issues(args.repo)
    elif args.command == 'create-issue':
        asyncio.run(create_local_issue(args.repo, args.title, args.description))
    elif args.command == 'process-task':
        asyncio.run(process_next_task())
    elif args.command == 'status':
        show_status()


if __name__ == '__main__':
    main()