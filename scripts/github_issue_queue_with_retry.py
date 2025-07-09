"""
GitHub Issue Queue with Retry Support (No Redis Version)

Enhanced version of GitHub issue queue that integrates with the retry manager
for automatic retries when GitHub is unavailable.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid

from scripts.work_item_types import WorkItem
from scripts.github_issue_creator import GitHubIssueCreator
from scripts.github_retry_manager import GitHubRetryManager


class GitHubIssueQueueWithRetry:
    """GitHub issue queue with automatic retry support."""
    
    def __init__(self, logger=None):
        """Initialize GitHub issue queue with retry support."""
        self.logger = logger or logging.getLogger(__name__)
        self.github_creator = GitHubIssueCreator()
        
        # Initialize retry manager
        self.retry_manager = GitHubRetryManager(
            github_creator=self.github_creator,
            task_manager=None  # Will be set if needed
        )
        
        # In-memory queue for immediate processing
        self.queue = asyncio.Queue()
        self.processing = False
        self.processor_task = None
        
        # File-based persistence for queue state
        self.queue_file = Path("github_issue_queue_state.json")
        self.load_queue_state()
        
    def load_queue_state(self):
        """Load any persisted queue items."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                    # Restore queued items
                    for item in data.get('pending', []):
                        self.queue.put_nowait(item)
                    self.logger.info(f"Loaded {len(data.get('pending', []))} pending items from queue state")
            except Exception as e:
                self.logger.error(f"Failed to load queue state: {e}")
    
    def save_queue_state(self):
        """Save current queue state to file."""
        try:
            # Get all items from queue without removing them
            items = []
            temp_items = []
            
            # Drain queue temporarily
            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    items.append(item)
                    temp_items.append(item)
                except asyncio.QueueEmpty:
                    break
            
            # Restore items to queue
            for item in temp_items:
                self.queue.put_nowait(item)
            
            # Save to file
            data = {
                'pending': items,
                'saved_at': datetime.now(timezone.utc).isoformat()
            }
            with open(self.queue_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save queue state: {e}")
    
    async def add_issue(self, work_item: WorkItem, task_id: str) -> str:
        """Add a work item to the GitHub issue creation queue."""
        queue_id = f"gh_queue_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        queue_item = {
            'id': queue_id,
            'work_item': work_item.to_dict() if hasattr(work_item, 'to_dict') else work_item.__dict__,
            'task_id': task_id,
            'queued_at': datetime.now(timezone.utc).isoformat()
        }
        
        await self.queue.put(queue_item)
        self.save_queue_state()
        
        print(f"📋 Queued GitHub issue creation for: {work_item.title}")
        return queue_id
    
    async def start_processor(self):
        """Start the background processor for the queue."""
        if self.processing:
            return
            
        self.processing = True
        
        # Start retry manager
        await self.retry_manager.start()
        
        # Start queue processor
        self.processor_task = asyncio.create_task(self._process_queue())
        print("✅ GitHub issue queue processor started with retry support")
    
    async def stop_processor(self):
        """Stop the queue processor."""
        self.processing = False
        
        # Stop retry manager
        await self.retry_manager.stop()
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Save final state
        self.save_queue_state()
        print("GitHub issue queue processor stopped")
    
    async def _process_queue(self):
        """Process queued GitHub issues with retry support."""
        while self.processing:
            try:
                # Wait for an item with timeout
                try:
                    queue_item = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check GitHub health before attempting
                github_status = await self.retry_manager.check_github_health()
                
                if github_status.value == "available":
                    # Try to create the issue
                    success = await self._create_github_issue(queue_item)
                    
                    if success:
                        print(f"✅ Successfully created GitHub issue from queue")
                    else:
                        # Failed but GitHub is available - might be a different issue
                        await self._handle_failed_issue(queue_item, "Creation failed")
                else:
                    # GitHub is not available - add to retry queue
                    self.retry_manager._log_github_failure(
                        task={
                            'title': queue_item['work_item'].get('title', 'Unknown'),
                            'repository': queue_item['work_item'].get('repository', 'N/A'),
                            'task_type': queue_item['work_item'].get('task_type', 'Unknown'),
                            'priority': queue_item['work_item'].get('priority', 'Medium'),
                            'failed_at': datetime.now(timezone.utc).isoformat(),
                            'retry_count': 0
                        },
                        error=f"GitHub service status: {github_status.value}"
                    )
                    
                    # Add to retry manager
                    work_item_data = queue_item['work_item']
                    work_item = WorkItem(**work_item_data) if isinstance(work_item_data, dict) else work_item_data
                    await self.retry_manager.add_failed_task(
                        work_item,
                        f"GitHub unavailable: {github_status.value}"
                    )
                
                # Update queue state
                self.save_queue_state()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing queue item: {e}")
                await asyncio.sleep(5)
    
    async def _create_github_issue(self, queue_item: Dict[str, Any]) -> bool:
        """Attempt to create a GitHub issue."""
        try:
            # Recreate WorkItem from dict
            work_item_data = queue_item['work_item']
            if isinstance(work_item_data, dict):
                work_item = WorkItem(**work_item_data)
            else:
                work_item = work_item_data
            
            # Check if GitHub integration is available
            if not self.github_creator.can_create_issues():
                raise RuntimeError("GitHub integration not configured")
            
            # Create the issue
            result = await self.github_creator.execute_work_item(work_item)
            
            return result.get('success', False)
            
        except Exception as e:
            self.logger.error(f"Failed to create GitHub issue: {e}")
            # Add to retry queue
            await self.retry_manager.add_failed_task(work_item, str(e))
            return False
    
    async def _handle_failed_issue(self, queue_item: Dict[str, Any], error: str):
        """Handle a failed issue creation."""
        work_item_data = queue_item['work_item']
        if isinstance(work_item_data, dict):
            work_item = WorkItem(**work_item_data)
        else:
            work_item = work_item_data
            
        # Add to retry manager
        await self.retry_manager.add_failed_task(work_item, error)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        retry_status = self.retry_manager.get_queue_status()
        
        return {
            'queue_size': self.queue.qsize(),
            'processing': self.processing,
            'retry_manager': retry_status
        }