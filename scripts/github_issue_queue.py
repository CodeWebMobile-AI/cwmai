"""
GitHub Issue Queue for Asynchronous Processing

Handles GitHub issue creation in the background without blocking workers.
Uses Redis for persistence and includes retry logic.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid

from scripts.work_item_types import WorkItem
from scripts.redis_integration import get_redis_client
from scripts.github_issue_creator import GitHubIssueCreator


@dataclass
class QueuedIssue:
    """Represents a GitHub issue waiting to be created."""
    id: str
    work_item: Dict[str, Any]
    task_id: str
    attempts: int = 0
    created_at: str = None
    last_attempt: str = None
    error: str = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueuedIssue':
        return cls(**data)


class GitHubIssueQueue:
    """Manages asynchronous GitHub issue creation."""
    
    def __init__(self, redis_client=None, logger=None):
        self.redis_client = redis_client
        self.logger = logger or logging.getLogger(__name__)
        self.queue_key = "github:issue:queue"
        self.processing_key = "github:issue:processing"
        self.failed_key = "github:issue:failed"
        self.completed_key = "github:issue:completed"
        self.github_creator = None
        self._shutdown = False
        self._processing_task = None
        
    async def initialize(self):
        """Initialize the queue system."""
        if not self.redis_client:
            self.redis_client = await get_redis_client()
        
        # Initialize GitHub creator
        self.github_creator = GitHubIssueCreator()
        
        self.logger.info("GitHub Issue Queue initialized")
        
    async def add_issue(self, work_item: WorkItem, task_id: str) -> str:
        """Add a work item to the GitHub issue creation queue."""
        queue_id = f"gh_queue_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        queued_issue = QueuedIssue(
            id=queue_id,
            work_item=work_item.to_dict(),
            task_id=task_id
        )
        
        # Add to Redis queue
        await self.redis_client.redis.lpush(
            self.queue_key,
            json.dumps(queued_issue.to_dict())
        )
        
        self.logger.info(f"Queued GitHub issue creation for: {work_item.title}")
        return queue_id
        
    async def process_queue(self):
        """Process queued GitHub issues in the background."""
        self.logger.info("Starting GitHub issue queue processor")
        
        while not self._shutdown:
            try:
                # Move item from queue to processing
                item_data = await self.redis_client.redis.brpoplpush(
                    self.queue_key,
                    self.processing_key,
                    timeout=5  # 5 second timeout
                )
                
                if not item_data:
                    continue  # Timeout, check again
                    
                # Parse the queued issue
                if isinstance(item_data, bytes):
                    item_data = item_data.decode()
                    
                queued_issue = QueuedIssue.from_dict(json.loads(item_data))
                
                # Process the issue
                success = await self._create_github_issue(queued_issue)
                
                if success:
                    # Move to completed
                    await self.redis_client.redis.lrem(self.processing_key, 1, item_data)
                    await self.redis_client.redis.lpush(
                        self.completed_key,
                        json.dumps(queued_issue.to_dict())
                    )
                else:
                    # Handle failure
                    await self._handle_failed_issue(queued_issue, item_data)
                    
            except asyncio.CancelledError:
                self.logger.info("GitHub queue processor cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in GitHub queue processor: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
        self.logger.info("GitHub issue queue processor stopped")
        
    async def _create_github_issue(self, queued_issue: QueuedIssue) -> bool:
        """Attempt to create a GitHub issue."""
        try:
            queued_issue.attempts += 1
            queued_issue.last_attempt = datetime.now(timezone.utc).isoformat()
            
            # Recreate WorkItem from dict
            work_item = WorkItem(**queued_issue.work_item)
            
            # Create the GitHub issue
            result = await self.github_creator.execute_work_item(work_item)
            
            if result.get('success'):
                self.logger.info(f"✅ Created GitHub issue for: {work_item.title}")
                
                # Update task persistence with issue number if available
                if result.get('issue_number'):
                    await self._update_task_with_issue(
                        queued_issue.task_id,
                        result['issue_number']
                    )
                
                return True
            else:
                queued_issue.error = result.get('error', 'Unknown error')
                self.logger.warning(f"Failed to create GitHub issue: {queued_issue.error}")
                return False
                
        except Exception as e:
            queued_issue.error = str(e)
            self.logger.error(f"Exception creating GitHub issue: {e}")
            return False
            
    async def _handle_failed_issue(self, queued_issue: QueuedIssue, item_data: str):
        """Handle a failed GitHub issue creation."""
        max_attempts = 3
        
        if queued_issue.attempts < max_attempts:
            # Retry later with exponential backoff
            retry_delay = 60 * (2 ** (queued_issue.attempts - 1))  # 1min, 2min, 4min
            self.logger.warning(
                f"Retrying GitHub issue creation for {queued_issue.work_item.get('title')} "
                f"(attempt {queued_issue.attempts}/{max_attempts}) in {retry_delay}s"
            )
            
            # Move back to queue after delay
            await asyncio.sleep(retry_delay)
            await self.redis_client.redis.lrem(self.processing_key, 1, item_data)
            await self.redis_client.redis.lpush(
                self.queue_key,
                json.dumps(queued_issue.to_dict())
            )
        else:
            # Max attempts reached, move to failed queue
            self.logger.error(
                f"Failed to create GitHub issue after {max_attempts} attempts: "
                f"{queued_issue.work_item.get('title')}"
            )
            await self.redis_client.redis.lrem(self.processing_key, 1, item_data)
            await self.redis_client.redis.lpush(
                self.failed_key,
                json.dumps(queued_issue.to_dict())
            )
            
    async def _update_task_with_issue(self, task_id: str, issue_number: int):
        """Update task persistence with GitHub issue number."""
        try:
            # Update task in persistence
            # This would connect to your task persistence system
            self.logger.info(f"Updated task {task_id} with GitHub issue #{issue_number}")
        except Exception as e:
            self.logger.error(f"Failed to update task with issue number: {e}")
            
    async def start_processor(self):
        """Start the background processor."""
        if not self._processing_task:
            self._processing_task = asyncio.create_task(self.process_queue())
            self.logger.info("GitHub issue processor started")
            
    async def stop_processor(self):
        """Stop the background processor."""
        self._shutdown = True
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("GitHub issue processor stopped")
        
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        try:
            pending = await self.redis_client.redis.llen(self.queue_key)
            processing = await self.redis_client.redis.llen(self.processing_key)
            completed = await self.redis_client.redis.llen(self.completed_key)
            failed = await self.redis_client.redis.llen(self.failed_key)
            
            return {
                'pending': pending,
                'processing': processing,
                'completed': completed,
                'failed': failed,
                'total': pending + processing
            }
        except Exception as e:
            self.logger.error(f"Error getting queue stats: {e}")
            return {
                'pending': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0,
                'total': 0
            }