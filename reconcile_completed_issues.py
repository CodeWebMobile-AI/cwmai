#!/usr/bin/env python3
"""
Reconcile Completed Issues

Scans all local repositories for issues with 'claude-implemented' label
and updates the completed_tasks.json and task_state.json files accordingly.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.state_manager import StateManager


async def reconcile_all_completed_issues():
    """Scan all repositories and reconcile completed issues."""
    print("🔄 Reconciling Completed Issues with State Files")
    print("=" * 60)
    
    # Initialize components
    local_sim = LocalGitHubSimulator()
    state_manager = StateManager()
    
    # Get all repositories
    repos = local_sim.list_repositories()
    print(f"\n📂 Found {len(repos)} repositories to check")
    
    # Track statistics
    total_issues = 0
    completed_issues = 0
    newly_reconciled = 0
    already_recorded = 0
    
    # Load current state
    task_state = state_manager.load_task_state()
    if 'completed_tasks' not in task_state:
        task_state['completed_tasks'] = []
    
    # Get existing completed task IDs
    existing_ids = {task.get('issue_id') for task in task_state['completed_tasks'] if task.get('issue_id')}
    
    for repo in repos:
        print(f"\n🔍 Checking {repo.full_name}...")
        
        # Get all issues for the repository
        issues = local_sim.get_issues(repo.full_name, state="all")
        total_issues += len(issues)
        
        for issue in issues:
            # Check if issue has 'claude-implemented' label
            if 'claude-implemented' in issue.labels:
                completed_issues += 1
                issue_id = f"{repo.full_name}#{issue.number}"
                
                # Check if already recorded
                if issue_id in existing_ids:
                    already_recorded += 1
                    print(f"   ✓ Issue #{issue.number} already recorded")
                else:
                    # Add to completed tasks
                    newly_reconciled += 1
                    print(f"   📝 Reconciling issue #{issue.number}: {issue.title}")
                    
                    completed_task = {
                        'id': f"reconciled_{repo.full_name}_{issue.number}_{int(datetime.now().timestamp())}",
                        'task_type': 'RECONCILED',
                        'title': issue.title,
                        'repository': repo.full_name,
                        'completed_at': issue.updated_at.isoformat() if issue.updated_at else datetime.now(timezone.utc).isoformat(),
                        'issue_id': issue_id,
                        'issue_number': issue.number,
                        'reconciled': True,
                        'reconciliation_reason': 'Bulk reconciliation - found claude-implemented label',
                        'labels': issue.labels
                    }
                    
                    task_state['completed_tasks'].append(completed_task)
    
    # Save updated state
    if newly_reconciled > 0:
        print(f"\n💾 Saving {newly_reconciled} newly reconciled tasks...")
        state_manager.save_task_state(task_state)
        
        # Also update completed_tasks.json
        completed_tasks = state_manager.load_completed_tasks()
        # Add only the newly reconciled tasks
        new_tasks = task_state['completed_tasks'][-newly_reconciled:]
        completed_tasks.extend(new_tasks)
        state_manager.save_completed_tasks(completed_tasks)
        
        print("✅ State files updated successfully!")
    else:
        print("\n✅ All completed issues were already recorded in state files")
    
    # Print summary
    print("\n📊 Reconciliation Summary:")
    print(f"   Total issues scanned: {total_issues}")
    print(f"   Completed issues found: {completed_issues}")
    print(f"   Already recorded: {already_recorded}")
    print(f"   Newly reconciled: {newly_reconciled}")
    
    # Check for issues that might be completed but not labeled
    print("\n🔍 Checking for potentially completed issues without labels...")
    unlabeled_closed = 0
    
    for repo in repos:
        issues = local_sim.get_issues(repo.full_name, state="closed")
        for issue in issues:
            if 'claude-implemented' not in issue.labels and issue.state == 'closed':
                unlabeled_closed += 1
                print(f"   ⚠️  {repo.full_name} Issue #{issue.number}: {issue.title} (closed but not labeled)")
    
    if unlabeled_closed > 0:
        print(f"\n⚠️  Found {unlabeled_closed} closed issues without 'claude-implemented' label")
        print("   These might be completed but not properly labeled")
    else:
        print("\n✅ All closed issues are properly labeled")


async def show_current_state():
    """Show current state of completed tasks."""
    print("\n\n📋 Current Completed Tasks State")
    print("=" * 60)
    
    state_manager = StateManager()
    task_state = state_manager.load_task_state()
    
    completed_tasks = task_state.get('completed_tasks', [])
    print(f"\nTotal completed tasks in state: {len(completed_tasks)}")
    
    # Group by repository
    by_repo = {}
    for task in completed_tasks:
        repo = task.get('repository', 'unknown')
        if repo not in by_repo:
            by_repo[repo] = []
        by_repo[repo].append(task)
    
    for repo, tasks in by_repo.items():
        print(f"\n📂 {repo}: {len(tasks)} completed tasks")
        for task in tasks[-5:]:  # Show last 5
            reconciled = " [RECONCILED]" if task.get('reconciled') else ""
            print(f"   - {task.get('title', 'Unknown')}{reconciled}")
        if len(tasks) > 5:
            print(f"   ... and {len(tasks) - 5} more")


async def main():
    """Run reconciliation."""
    try:
        # Set local mode
        os.environ['STORAGE_MODE'] = 'LOCAL'
        
        # Run reconciliation
        await reconcile_all_completed_issues()
        
        # Show current state
        await show_current_state()
        
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())