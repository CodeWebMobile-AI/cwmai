"""
GitHub Issue Queue (No Redis Version)

Simple in-memory implementation of GitHub issue queue for the no-redis orchestrator.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import asyncio

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.github_issue_creator import GitHubIssueCreator
from scripts.claude_execution_queue import ClaudeExecutionQueue


class GitHubIssueQueueNoRedis:
    """Simple GitHub issue queue without Redis dependency."""
    
    def __init__(self, ai_brain=None):
        """Initialize GitHub issue queue.
        
        Args:
            ai_brain: AI brain instance (for potential future use)
        """
        self.logger = logging.getLogger(__name__)
        self.ai_brain = ai_brain
        self.github_creator = GitHubIssueCreator()
        
        # Track processed issue IDs to avoid duplicates
        self.processed_issues = set()
        self._lock = asyncio.Lock()
        
        # Initialize Claude execution queue to check for completed tasks
        self.claude_queue = ClaudeExecutionQueue()
        
    async def get_work_items(self) -> List[WorkItem]:
        """Get work items from GitHub issues.
        
        Returns:
            List of work items from open GitHub issues
        """
        try:
            # Get open issues from GitHub
            issues = await self.github_creator.get_open_issues()
            
            work_items = []
            async with self._lock:
                for issue in issues:
                    issue_id = f"github_issue_{issue['number']}"
                    repository = issue.get('repository', {}).get('full_name', '')
                    issue_number = issue['number']
                    
                    # Skip if already processed
                    if issue_id in self.processed_issues:
                        continue
                    
                    # Check if Claude has already completed this issue
                    if repository and self.claude_queue.has_completed_task_for_issue(repository, issue_number):
                        self.logger.info(f"Skipping issue #{issue_number} in {repository} - already completed by Claude")
                        self.processed_issues.add(issue_id)
                        continue
                    
                    # Check if Claude is currently processing this issue
                    if repository and self.claude_queue.has_pending_task_for_issue(repository, issue_number):
                        self.logger.info(f"Skipping issue #{issue_number} in {repository} - Claude task pending")
                        self.processed_issues.add(issue_id)
                        continue
                    
                    # Mark as processed
                    self.processed_issues.add(issue_id)
                    
                    # Create work item
                    work_item = WorkItem(
                        id=issue_id,
                        task_type="github_issue",
                        title=f"GitHub Issue #{issue['number']}: {issue['title']}",
                        description=issue.get('body', ''),
                        repository=issue.get('repository', {}).get('full_name'),
                        priority=self._determine_priority(issue),
                        metadata={
                            'issue_number': issue['number'],
                            'issue_url': issue['html_url'],
                            'created_at': issue['created_at'],
                            'labels': [label['name'] for label in issue.get('labels', [])]
                        }
                    )
                    
                    work_items.append(work_item)
            
            # Clean up old processed issues periodically
            if len(self.processed_issues) > 1000:
                # Keep only the most recent 500
                self.processed_issues = set(list(self.processed_issues)[-500:])
            
            return work_items
            
        except Exception as e:
            self.logger.error(f"Error fetching GitHub issues: {e}")
            return []
    
    def _determine_priority(self, issue: Dict[str, Any]) -> TaskPriority:
        """Determine priority based on issue labels and age.
        
        Args:
            issue: GitHub issue data
            
        Returns:
            TaskPriority enum value
        """
        labels = [label['name'].lower() for label in issue.get('labels', [])]
        
        # Check for priority labels
        if 'critical' in labels or 'urgent' in labels:
            return TaskPriority.CRITICAL
        elif 'high-priority' in labels or 'important' in labels:
            return TaskPriority.HIGH
        elif 'low-priority' in labels:
            return TaskPriority.LOW
        
        # Check age of issue
        created_at = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
        age_days = (datetime.now(timezone.utc) - created_at).days
        
        if age_days > 7:
            return TaskPriority.HIGH
        elif age_days > 3:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW


# Alias for compatibility
GitHubIssueQueue = GitHubIssueQueueNoRedis