"""
Claude Executor

Processes Claude execution tasks from the queue in parallel.
Multiple executors can run simultaneously to enable parallel Claude execution.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from scripts.claude_execution_queue import ClaudeExecutionQueue, ClaudeTask


class ClaudeExecutor:
    """Executes Claude tasks from the queue."""
    
    def __init__(self, executor_id: str, queue: ClaudeExecutionQueue, 
                 claude_executable: str = "claude", max_retries: int = 2):
        """Initialize the executor.
        
        Args:
            executor_id: Unique ID for this executor
            queue: The Claude execution queue
            claude_executable: Path to Claude executable
            max_retries: Maximum number of retries for failed tasks
        """
        self.executor_id = executor_id
        self.queue = queue
        self.claude_executable = claude_executable
        self.max_retries = max_retries
        self.logger = logging.getLogger(f"{__name__}.{executor_id}")
        self.shutdown_requested = False
        self.current_task: Optional[ClaudeTask] = None
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0
        }
    
    async def run(self):
        """Main executor loop."""
        self.logger.info(f"Claude executor {self.executor_id} started")
        
        while not self.shutdown_requested:
            try:
                # Get next task from queue
                task = await self.queue.get_next_task(self.executor_id)
                
                if not task:
                    # No work available, wait a bit
                    await asyncio.sleep(5)
                    continue
                
                # Set current task BEFORE processing
                self.current_task = task
                
                try:
                    # Execute the task
                    result = await self._execute_task(task)
                    
                    if result['success']:
                        await self.queue.complete_task(task.id, result)
                        self.stats['tasks_completed'] += 1
                    else:
                        # Check if we should retry
                        retry_count = result.get('retry_count', 0)
                        if retry_count < self.max_retries:
                            result['retry_count'] = retry_count + 1
                            await self.queue.fail_task(task.id, result.get('error', 'Unknown error'), retry=True)
                        else:
                            await self.queue.fail_task(task.id, result.get('error', 'Max retries exceeded'), retry=False)
                            self.stats['tasks_failed'] += 1
                
                finally:
                    # ALWAYS clear current task, even if an exception occurred
                    self.current_task = None
                    self.logger.debug(f"Cleared current_task for executor {self.executor_id}")
                
            except Exception as e:
                self.logger.error(f"Executor error: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Ensure current_task is cleared even on unexpected errors
                self.current_task = None
                await asyncio.sleep(5)
        
        self.logger.info(f"Claude executor {self.executor_id} stopped")
    async def _execute_task(self, task: ClaudeTask) -> Dict[str, Any]:
        """Execute a single Claude task.
        
        Args:
            task: The task to execute
            
        Returns:
            Execution result
        """
        start_time = time.time()
        self.logger.info(f"Executing task {task.id} for {task.repository} issue #{task.issue_number}")
        
        try:
            # Check if Claude executable exists
            proc = await asyncio.create_subprocess_exec(
                self.claude_executable, '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                if proc.returncode != 0:
                    stderr_text = stderr.decode() if stderr else ''
                    stdout_text = stdout.decode() if stdout else ''
                    error_details = stderr_text or stdout_text or "Unknown error"
                    return {
                        'success': False,
                        'error': f"Claude executable not working: {error_details.strip()}"
                    }
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    'success': False,
                    'error': "Claude executable check timed out"
                }
            
            # Prepare the command
            cmd = [
                self.claude_executable,
                "--dangerously-skip-permissions"
            ]
            
            # Get repository path
            repo_path = self._get_repo_path(task.repository)
            if not repo_path or not os.path.exists(repo_path):
                return {
                    'success': False,
                    'error': f"Repository path not found: {repo_path}"
                }
            
            # Log execution start
            self.logger.info(f"🤖 Starting Claude execution for {task.repository} issue #{task.issue_number}")
            self.logger.info(f"Working directory: {repo_path}")
            
            # Create subprocess with periodic logging (NO TIMEOUT)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path
            )
            
            # Send prompt to stdin
            prompt_bytes = task.prompt.encode('utf-8')
            
            # Create task for process communication (NO TIMEOUT)
            comm_task = asyncio.create_task(process.communicate(input=prompt_bytes))
            
            # Log progress periodically
            start_time = time.time()
            last_log_time = start_time
            log_interval = 30  # Log every 30 seconds
            
            while not comm_task.done():
                await asyncio.sleep(1)
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Log progress
                if current_time - last_log_time >= log_interval:
                    self.logger.info(f"⏳ Claude still executing for {task.repository} "
                                   f"issue #{task.issue_number} ({elapsed:.0f}s elapsed)")
                    last_log_time = current_time
                
                # Info on long executions (not a warning, it's expected)
                if elapsed > 300 and int(elapsed) % 60 == 0:  # Every minute after 5 minutes
                    self.logger.info(f"⏳ Extended Claude execution for {task.repository} "
                                   f"issue #{task.issue_number}: {elapsed/60:.1f} minutes (this is normal for complex tasks)")
            
            # Get result
            stdout, stderr = await comm_task
            returncode = process.returncode
            
            execution_time = time.time() - start_time
            self.stats['total_execution_time'] += execution_time
            
            # Log completion
            if returncode == 0:
                self.logger.info(f"✅ Claude execution completed for {task.repository} "
                               f"issue #{task.issue_number} in {execution_time:.1f}s")
                
                # Check for file changes
                changes = await self._check_file_changes(repo_path)
                
                return {
                    'success': True,
                    'execution_time': execution_time,
                    'files_changed': changes.get('files_changed', 0),
                    'files_added': changes.get('files_added', 0),
                    'files_deleted': changes.get('files_deleted', 0),
                    'stdout': stdout.decode('utf-8') if stdout else '',
                    'returncode': returncode
                }
            else:
                stdout_text = stdout.decode('utf-8') if stdout else ''
                stderr_text = stderr.decode('utf-8') if stderr else ''
                
                # Check for specific error conditions
                error_msg = f"Claude returned non-zero exit code: {returncode}"
                if "Credit balance is too low" in stdout_text:
                    error_msg = "Claude API credit balance is too low. Please add credits at https://console.anthropic.com/settings/billing"
                    self.logger.error(f"❌ Claude execution failed due to insufficient credits for {task.repository} issue #{task.issue_number}")
                elif "Permission denied" in stderr_text or "Permission denied" in stdout_text:
                    error_msg = "Claude permission denied. Check file permissions in the repository."
                elif "not found" in stderr_text.lower() or "not found" in stdout_text.lower():
                    error_msg = f"Claude could not find required files or commands. Check repository structure."
                else:
                    self.logger.error(f"❌ Claude execution failed for {task.repository} "
                                    f"issue #{task.issue_number} with code {returncode}")
                
                # Log the actual output for debugging
                if stdout_text:
                    self.logger.debug(f"Claude stdout: {stdout_text[:500]}")
                if stderr_text:
                    self.logger.debug(f"Claude stderr: {stderr_text[:500]}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'stderr': stderr_text,
                    'stdout': stdout_text,
                    'execution_time': execution_time,
                    'returncode': returncode
                }
                
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self.logger.error(f"⏱️ Claude execution timed out for {task.repository} "
                            f"issue #{task.issue_number} after {execution_time:.1f}s")
            return {
                'success': False,
                'error': 'Execution timeout',
                'execution_time': execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error executing Claude task: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _get_repo_path(self, repository: str) -> Optional[str]:
        """Get the local path for a repository.
        
        Args:
            repository: Repository name
            
        Returns:
            Repository path or None
        """
        # Check local_repos directory
        local_path = os.path.join("local_repos", "repositories", repository)
        if os.path.exists(local_path):
            return local_path
        
        # Check if it's a full organization/repo name
        if '/' in repository:
            org, repo = repository.split('/', 1)
            local_path = os.path.join("local_repos", "repositories", org, repo)
            if os.path.exists(local_path):
                return local_path
        
        return None
    
    async def _check_file_changes(self, repo_path: str) -> Dict[str, int]:
        """Check for file changes in the repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary with change counts
        """
        try:
            # Run git status to check for changes
            process = await asyncio.create_subprocess_exec(
                'git', 'status', '--porcelain',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path
            )
            
            stdout, _ = await process.communicate()
            
            if process.returncode == 0 and stdout:
                lines = stdout.decode('utf-8').strip().split('\n')
                
                files_changed = 0
                files_added = 0
                files_deleted = 0
                
                for line in lines:
                    if line.strip():
                        status = line[:2]
                        if 'A' in status or '?' in status:
                            files_added += 1
                        elif 'D' in status:
                            files_deleted += 1
                        elif 'M' in status:
                            files_changed += 1
                
                return {
                    'files_changed': files_changed,
                    'files_added': files_added,
                    'files_deleted': files_deleted,
                    'total_changes': len(lines)
                }
            
        except Exception as e:
            self.logger.debug(f"Error checking file changes: {e}")
        
        return {
            'files_changed': 0,
            'files_added': 0,
            'files_deleted': 0,
            'total_changes': 0
        }
    
    async def shutdown(self):
        """Shutdown the executor gracefully."""
        self.logger.info(f"Shutting down executor {self.executor_id}")
        self.shutdown_requested = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics.
        
        Returns:
            Executor statistics
        """
        stats = self.stats.copy()
        stats['executor_id'] = self.executor_id
        stats['current_task'] = self.current_task.to_dict() if self.current_task else None
        
        if self.stats['tasks_completed'] > 0:
            stats['avg_execution_time'] = self.stats['total_execution_time'] / self.stats['tasks_completed']
        else:
            stats['avg_execution_time'] = 0
        
        return stats


class ClaudeExecutorPool:
    """Manages a pool of Claude executors."""
    
    def __init__(self, num_executors: int = 3, queue: Optional[ClaudeExecutionQueue] = None):
        """Initialize the executor pool.
        
        Args:
            num_executors: Number of parallel executors
            queue: Shared execution queue (created if not provided)
        """
        self.num_executors = num_executors
        self.queue = queue or ClaudeExecutionQueue()
        self.executors: Dict[str, ClaudeExecutor] = {}
        self.executor_tasks: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start all executors."""
        self.logger.info(f"Starting Claude executor pool with {self.num_executors} executors")
        
        for i in range(self.num_executors):
            executor_id = f"claude_executor_{i:02d}"
            executor = ClaudeExecutor(executor_id, self.queue)
            self.executors[executor_id] = executor
            
            # Start executor task
            self.executor_tasks[executor_id] = asyncio.create_task(executor.run())
        
        self.logger.info(f"Started {len(self.executors)} Claude executors")
    
    async def stop(self):
        """Stop all executors."""
        self.logger.info("Stopping Claude executor pool")
        
        # Request shutdown for all executors
        for executor in self.executors.values():
            await executor.shutdown()
        
        # Wait for all executor tasks to complete
        if self.executor_tasks:
            await asyncio.gather(*self.executor_tasks.values(), return_exceptions=True)
        
        self.logger.info("Claude executor pool stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics.
        
        Returns:
            Pool statistics
        """
        stats = {
            'num_executors': self.num_executors,
            'executors': {}
        }
        
        for executor_id, executor in self.executors.items():
            stats['executors'][executor_id] = executor.get_stats()
        
        # Aggregate stats
        total_completed = sum(e.stats['tasks_completed'] for e in self.executors.values())
        total_failed = sum(e.stats['tasks_failed'] for e in self.executors.values())
        total_time = sum(e.stats['total_execution_time'] for e in self.executors.values())
        
        stats['total_completed'] = total_completed
        stats['total_failed'] = total_failed
        
        if total_completed > 0:
            stats['avg_execution_time'] = total_time / total_completed
        else:
            stats['avg_execution_time'] = 0
        
        return stats