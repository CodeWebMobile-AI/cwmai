"""
GitHub Retry Manager - Automatically retries failed GitHub operations when service becomes available.

This module provides:
1. Automatic retry queue for failed GitHub issues
2. Health checking to detect when GitHub is available
3. Detailed error logging and user notifications
4. Persistent retry queue that survives restarts
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
import os
from enum import Enum

from scripts.logger import get_logger
from scripts.storage_mode import StorageMode
from scripts.storage_factory import StorageFactory


class GitHubStatus(Enum):
    """GitHub service status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    AUTH_FAILED = "auth_failed"
    UNKNOWN = "unknown"


@dataclass
class FailedGitHubTask:
    """Represents a task that failed to create GitHub issue"""
    task_id: str
    title: str
    description: str
    repository: str
    task_type: str
    priority: str
    failed_at: str
    retry_count: int = 0
    last_error: str = ""
    work_item_data: Dict[str, Any] = None


class GitHubRetryManager:
    """Manages automatic retries for failed GitHub operations"""
    
    def __init__(self, github_creator=None, task_manager=None, storage_mode: str = None):
        self.logger = get_logger(self.__class__.__name__)
        self.github_creator = github_creator
        self.task_manager = task_manager
        
        # Determine storage mode
        self.storage_mode = storage_mode or os.getenv('STORAGE_MODE', StorageMode.GITHUB)
        self.storage_factory = StorageFactory(self.storage_mode)
        
        # In local mode, retry manager is not needed
        if self.storage_factory.is_local_mode():
            self.logger.info("GitHubRetryManager bypassed in LOCAL mode")
            self.is_local_mode = True
            self.is_running = False
            return
        else:
            self.is_local_mode = False
        
        # Persistent storage
        self.retry_queue_file = Path("github_retry_queue.json")
        self.retry_queue: List[FailedGitHubTask] = []
        self.load_retry_queue()
        
        # Health check configuration
        self.health_check_interval = 60  # Check every minute
        self.retry_interval = 300  # Retry failed issues every 5 minutes
        self.max_retry_attempts = 10
        
        # Status tracking
        self.last_status = GitHubStatus.UNKNOWN
        self.last_health_check = None
        self.consecutive_failures = 0
        
        # Running state
        self.is_running = False
        self.health_check_task = None
        self.retry_task = None
        
    def load_retry_queue(self):
        """Load retry queue from persistent storage"""
        if self.retry_queue_file.exists():
            try:
                with open(self.retry_queue_file, 'r') as f:
                    data = json.load(f)
                    self.retry_queue = [
                        FailedGitHubTask(**item) for item in data.get('queue', [])
                    ]
                    self.logger.info(f"Loaded {len(self.retry_queue)} tasks from retry queue")
            except Exception as e:
                self.logger.error(f"Failed to load retry queue: {e}")
                self.retry_queue = []
    
    def save_retry_queue(self):
        """Save retry queue to persistent storage"""
        try:
            data = {
                'queue': [asdict(task) for task in self.retry_queue],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            with open(self.retry_queue_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save retry queue: {e}")
    
    async def add_failed_task(self, work_item: Any, error: str) -> str:
        """Add a failed task to the retry queue"""
        # In local mode, no retry queue needed
        if self.is_local_mode:
            return work_item.id
        
        failed_task = FailedGitHubTask(
            task_id=work_item.id,
            title=work_item.title,
            description=work_item.description,
            repository=work_item.repository or "N/A",
            task_type=work_item.task_type,
            priority=str(work_item.priority),
            failed_at=datetime.now(timezone.utc).isoformat(),
            last_error=str(error),
            work_item_data=work_item.to_dict() if hasattr(work_item, 'to_dict') else None
        )
        
        self.retry_queue.append(failed_task)
        self.save_retry_queue()
        
        # Log detailed error
        self._log_github_failure(failed_task, error)
        
        return failed_task.task_id
    
    def _log_github_failure(self, task: FailedGitHubTask, error: str):
        """Log detailed error information when GitHub fails"""
        print("\n" + "="*80)
        print("❌ GITHUB ISSUE CREATION FAILED - FALLING BACK TO LOCAL TASK MANAGER")
        print("="*80)
        print(f"📋 Task: {task.title}")
        print(f"📁 Repository: {task.repository}")
        print(f"🏷️  Type: {task.task_type}")
        print(f"🎯 Priority: {task.priority}")
        print(f"🕐 Failed at: {task.failed_at}")
        print(f"🔄 Retry count: {task.retry_count}")
        print(f"\n❗ Error: {error}")
        
        # Check specific error types
        if "401" in str(error) or "unauthorized" in str(error).lower():
            print("\n🔐 AUTHENTICATION ERROR:")
            print("   - Check if GITHUB_TOKEN or CLAUDE_PAT is set correctly")
            print("   - Verify token has 'repo' and 'issues' permissions")
            print(f"   - Current token exists: {bool(os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT'))}")
        elif "404" in str(error):
            print("\n🔍 REPOSITORY NOT FOUND:")
            print(f"   - Check if repository '{os.getenv('GITHUB_REPOSITORY')}' exists")
            print("   - Verify you have access to the repository")
        elif "403" in str(error) or "rate limit" in str(error).lower():
            print("\n⏱️  RATE LIMIT ERROR:")
            print("   - GitHub API rate limit exceeded")
            print("   - Will retry automatically when limit resets")
        else:
            print("\n🌐 NETWORK/OTHER ERROR:")
            print("   - Check internet connectivity")
            print("   - GitHub service might be down")
        
        print("\n✅ Task saved locally and queued for retry")
        print(f"🔄 Will retry automatically when GitHub becomes available")
        print(f"📊 Tasks in retry queue: {len(self.retry_queue)}")
        print("="*80 + "\n")
    
    async def check_github_health(self) -> GitHubStatus:
        """Check if GitHub API is accessible"""
        # In local mode, always available
        if self.is_local_mode:
            return GitHubStatus.AVAILABLE
        
        try:
            # Check authentication
            token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
            if not token:
                return GitHubStatus.AUTH_FAILED
            
            # Check repository configuration
            repo = os.getenv('GITHUB_REPOSITORY')
            if not repo:
                self.logger.error("GITHUB_REPOSITORY not configured")
                return GitHubStatus.UNAVAILABLE
            
            # Test API endpoint
            headers = {
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            async with aiohttp.ClientSession() as session:
                # Check rate limit first
                async with session.get(
                    'https://api.github.com/rate_limit',
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        remaining = data['rate']['remaining']
                        if remaining == 0:
                            reset_time = data['rate']['reset']
                            self.logger.warning(f"GitHub rate limit exceeded. Resets at {datetime.fromtimestamp(reset_time)}")
                            return GitHubStatus.RATE_LIMITED
                    elif response.status == 401:
                        return GitHubStatus.AUTH_FAILED
                
                # Check repository access
                async with session.get(
                    f'https://api.github.com/repos/{repo}',
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return GitHubStatus.AVAILABLE
                    elif response.status == 404:
                        self.logger.error(f"Repository {repo} not found or not accessible")
                        return GitHubStatus.UNAVAILABLE
                    else:
                        return GitHubStatus.UNAVAILABLE
                        
        except asyncio.TimeoutError:
            self.logger.error("GitHub health check timed out")
            return GitHubStatus.UNAVAILABLE
        except Exception as e:
            self.logger.error(f"GitHub health check failed: {e}")
            return GitHubStatus.UNKNOWN
    
    async def _health_check_loop(self):
        """Continuously monitor GitHub availability"""
        while self.is_running:
            try:
                new_status = await self.check_github_health()
                self.last_health_check = datetime.now(timezone.utc)
                
                # Status change detection
                if new_status != self.last_status:
                    self._log_status_change(self.last_status, new_status)
                    
                    # If GitHub is now available and we have queued tasks, trigger retry
                    if new_status == GitHubStatus.AVAILABLE and self.retry_queue:
                        self.logger.info("GitHub is now available - triggering retry of queued tasks")
                        asyncio.create_task(self._retry_failed_tasks())
                
                self.last_status = new_status
                
                # Adjust check interval based on status
                if new_status == GitHubStatus.AVAILABLE:
                    await asyncio.sleep(self.health_check_interval)
                else:
                    # Check more frequently when unavailable
                    await asyncio.sleep(30)
                    
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    def _log_status_change(self, old_status: GitHubStatus, new_status: GitHubStatus):
        """Log status changes with detailed information"""
        print("\n" + "="*60)
        print("🔄 GITHUB STATUS CHANGE DETECTED")
        print("="*60)
        print(f"Previous: {old_status.value}")
        print(f"Current: {new_status.value}")
        print(f"Time: {datetime.now(timezone.utc).isoformat()}")
        
        if new_status == GitHubStatus.AVAILABLE:
            print("\n✅ GitHub is now AVAILABLE!")
            if self.retry_queue:
                print(f"📋 {len(self.retry_queue)} tasks queued for retry")
        elif new_status == GitHubStatus.RATE_LIMITED:
            print("\n⏱️  GitHub API rate limit exceeded")
            print("Will retry when limit resets")
        elif new_status == GitHubStatus.AUTH_FAILED:
            print("\n🔐 GitHub authentication failed")
            print("Please check GITHUB_TOKEN/CLAUDE_PAT configuration")
        else:
            print(f"\n❌ GitHub is {new_status.value}")
            print("Tasks will be queued for retry when available")
        
        print("="*60 + "\n")
    
    async def _retry_failed_tasks(self):
        """Retry creating GitHub issues for failed tasks"""
        if not self.retry_queue or not self.github_creator:
            return
        
        print(f"\n🔄 Starting retry of {len(self.retry_queue)} failed GitHub tasks...")
        
        succeeded = []
        failed = []
        
        for task in self.retry_queue[:]:  # Copy list to allow modification
            try:
                # Check if we should still retry
                if task.retry_count >= self.max_retry_attempts:
                    self.logger.warning(f"Task {task.task_id} exceeded max retries ({self.max_retry_attempts})")
                    self.retry_queue.remove(task)
                    continue
                
                print(f"\n📋 Retrying: {task.title}")
                
                # Recreate work item from saved data
                if task.work_item_data:
                    # Use the GitHub creator to create the issue
                    result = await self.github_creator.create_issue_from_data(task.work_item_data)
                    
                    if result.get('success'):
                        print(f"✅ Successfully created GitHub issue #{result.get('issue_number')}")
                        succeeded.append(task)
                        self.retry_queue.remove(task)
                    else:
                        raise Exception(result.get('error', 'Unknown error'))
                else:
                    self.logger.error(f"No work item data for task {task.task_id}")
                    self.retry_queue.remove(task)
                    
            except Exception as e:
                task.retry_count += 1
                task.last_error = str(e)
                failed.append(task)
                print(f"❌ Retry failed: {e}")
        
        # Save updated queue
        self.save_retry_queue()
        
        # Summary
        print(f"\n📊 Retry Summary:")
        print(f"   ✅ Succeeded: {len(succeeded)}")
        print(f"   ❌ Failed: {len(failed)}")
        print(f"   📋 Still queued: {len(self.retry_queue)}")
    
    async def start(self):
        """Start the retry manager"""
        # In local mode, don't start
        if self.is_local_mode:
            self.logger.info("GitHubRetryManager not started in LOCAL mode")
            return
        
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting GitHub Retry Manager")
        
        # Start health check loop
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Start periodic retry loop
        self.retry_task = asyncio.create_task(self._retry_loop())
        
        # Do initial health check
        initial_status = await self.check_github_health()
        self._log_status_change(GitHubStatus.UNKNOWN, initial_status)
        self.last_status = initial_status
    
    async def _retry_loop(self):
        """Periodically retry failed tasks"""
        while self.is_running:
            try:
                await asyncio.sleep(self.retry_interval)
                
                if self.last_status == GitHubStatus.AVAILABLE and self.retry_queue:
                    await self._retry_failed_tasks()
                    
            except Exception as e:
                self.logger.error(f"Error in retry loop: {e}")
    
    async def stop(self):
        """Stop the retry manager"""
        self.is_running = False
        
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.retry_task:
            self.retry_task.cancel()
        
        self.save_retry_queue()
        self.logger.info("GitHub Retry Manager stopped")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current status of the retry queue"""
        # In local mode, return empty status
        if self.is_local_mode:
            return {
                'github_status': GitHubStatus.AVAILABLE.value,
                'storage_mode': 'local',
                'queued_tasks': 0,
                'queue_details': []
            }
        
        return {
            'github_status': self.last_status.value,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'queued_tasks': len(self.retry_queue),
            'queue_details': [
                {
                    'title': task.title,
                    'repository': task.repository,
                    'retry_count': task.retry_count,
                    'failed_at': task.failed_at
                }
                for task in self.retry_queue
            ]
        }