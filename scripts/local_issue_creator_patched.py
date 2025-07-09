"""
Local Issue Creator

Creates issues locally instead of using GitHub API.
Drop-in replacement for GitHubIssueCreator when in local mode.
Now with Claude Code CLI integration for autonomous implementation.
"""

import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import os
import json
import subprocess
import shutil

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.task_manager import TaskManager, TaskType, TaskPriority as LegacyPriority
from scripts.enhanced_smart_task_system import EnhancedSmartTaskSystem, get_task_type_for_string
from scripts.task_types import SmartTaskType
from scripts.monitoring_integration import MonitoringIntegration
from scripts.task_monitoring_system import get_monitoring_system


class LocalIssueCreator:
    """Creates issues locally instead of using GitHub API."""
    
    def __init__(self, enable_claude_execution=True):
        """Initialize the local issue creator.
        
        Args:
            enable_claude_execution: If True, will use Claude Code CLI to implement issues
        """
        self.logger = logging.getLogger(__name__)
        self.local_simulator = LocalGitHubSimulator()
        
        # Don't initialize task manager to avoid recursion
        # self.task_manager = TaskManager(repository=None)
        
        # Initialize enhanced task system
        self.task_system = EnhancedSmartTaskSystem()
        
        # No rate limiting needed for local operations
        self.min_issue_interval = 0
        self.last_issue_created = 0
        
        # Claude Code CLI integration
        self.enable_claude_execution = enable_claude_execution
        self.claude_executable = self._find_claude_executable()
        
        # Priority mapping
        self.priority_map = {
            TaskPriority.CRITICAL: LegacyPriority.CRITICAL,
            TaskPriority.HIGH: LegacyPriority.HIGH,
            TaskPriority.MEDIUM: LegacyPriority.MEDIUM,
            TaskPriority.LOW: LegacyPriority.LOW,
            TaskPriority.BACKGROUND: LegacyPriority.LOW
        }
        
        # Initialize monitoring
        self.monitoring = MonitoringIntegration()
        self.monitor = get_monitoring_system()
        
        # Task type mapping (map to SmartTaskType values)
        self.task_type_map = {
            'TESTING': SmartTaskType.TESTING_UNIT_TESTS,
            'FEATURE': SmartTaskType.FEATURE_BUSINESS_LOGIC,
            'BUG_FIX': SmartTaskType.MAINTENANCE_BUG_FIX,
            'DOCUMENTATION': SmartTaskType.TESTING_INTEGRATION_TESTS,
            'RESEARCH': SmartTaskType.RESEARCH_TECHNOLOGY,
            'SYSTEM_IMPROVEMENT': SmartTaskType.OPTIMIZATION_PERFORMANCE,
            'MAINTENANCE': SmartTaskType.MAINTENANCE_REFACTORING,
            'NEW_PROJECT': SmartTaskType.NEW_PROJECT,
            'INTEGRATION': SmartTaskType.INTEGRATION_API,
            'REPOSITORY_HEALTH': SmartTaskType.REPOSITORY_HEALTH,
            'API_ENDPOINT': SmartTaskType.FEATURE_API_ENDPOINT,
            'UI_COMPONENT': SmartTaskType.FEATURE_UI_COMPONENT,
            'DEPLOYMENT': SmartTaskType.INFRASTRUCTURE_DEPLOYMENT,
            'SECURITY': SmartTaskType.MAINTENANCE_SECURITY_PATCH,
            'CACHING': SmartTaskType.OPTIMIZATION_CACHING,
            'E2E_TESTS': SmartTaskType.TESTING_E2E_TESTS,
            'API_TESTS': SmartTaskType.TESTING_API_TESTS
        }
        
    async def execute_work_item(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a work item by creating a local issue with monitoring.
        
        Args:
            work_item: The work item to execute
            
        Returns:
            Execution result with success status and details
        """
        print(f"Creating local issue for: {work_item.title}")
        
        # Start monitoring
        task_id = work_item.id
        task_type = work_item.task_type
        repository = work_item.repository or "unknown"
        
        # Get estimated duration from metadata or predict
        estimated_duration = work_item.metadata.get('estimated_duration', 
            self.monitoring.predict_task_duration(task_type, repository))
        
        # Start task monitoring
        metrics = self.monitor.start_task_monitoring(
            task_id=task_id,
            task_type=task_type,
            repository=repository,
            estimated_duration=estimated_duration
        )
        
        # Record lifecycle event
        self.monitoring.track_lifecycle_event(task_id, "started", {
            "executor": "LocalIssueCreator",
            "claude_enabled": self.enable_claude_execution
        })
        
        # Ensure repository exists first (create if needed)
        # Check if repository already contains organization prefix
        if "/" in work_item.repository:
            repo_full_name = work_item.repository
        else:
            repo_full_name = f"CodeWebMobile-AI/{work_item.repository}"
        
        try:
            repo = self.local_simulator.get_repository(repo_full_name)
            self.logger.debug(f"Repository exists: {repo_full_name}")
        except Exception as e:
            # Repository doesn't exist, create it
            print(f"Repository {repo_full_name} doesn't exist, creating it...")
            self.logger.info(f"Repository {repo_full_name} doesn't exist, creating it...")
            try:
                # Extract repo name and org from full name
                if "/" in work_item.repository:
                    org, repo_name = work_item.repository.split("/", 1)
                else:
                    org = "CodeWebMobile-AI"
                    repo_name = work_item.repository
                
                self.local_simulator.create_repository(
                    name=repo_name,
                    description=f"Repository for {repo_name}",
                    organization=org
                )
                print(f"✅ Created repository: {repo_full_name}")
                self.logger.info(f"Created repository: {repo_full_name}")
            except Exception as create_error:
                self.logger.error(f"Failed to create repository: {create_error}")
                print(f"❌ Failed to create repository: {create_error}")
                # Complete monitoring with failure
                self.monitor.complete_task_monitoring(task_id, success=False, error=str(create_error))
                self.monitoring.track_lifecycle_event(task_id, "failed", {
                    "error": str(create_error),
                    "stage": "repository_creation"
                })
                
                return {
                    'success': False,
                    'error': f'Failed to create repository: {create_error}',
                    'repository': work_item.repository,
                    'value_created': 0
                }
        
        try:
            # Check if issue already exists (idempotency check)
            print("Checking for existing issues...")
            existing_issue = await self._check_existing_issue(work_item)
            print(f"Existing issue check result: {existing_issue}")
            if existing_issue:
                print(f"ℹ️ Issue already exists: #{existing_issue} for: {work_item.title}")
                
                # Check if issue is already implemented
                existing_issue_data = await self._get_issue_details(repo_full_name, existing_issue)
                if existing_issue_data and 'claude-implemented' in existing_issue_data.get('labels', []):
                    print(f"✅ Issue #{existing_issue} already implemented")
                    return {
                        'success': True,
                        'task_id': work_item.id,
                        'issue_number': existing_issue,
                        'repository': work_item.repository,
                        'value_created': 0,
                        'execution_time': time.time(),
                        'existing_issue': True,
                        'already_implemented': True
                    }
                
                # Issue exists but not implemented - execute with Claude if enabled
                if self.enable_claude_execution and self.claude_executable:
                    print(f"🤖 Issue exists but not implemented. Executing with Claude Code CLI...")
                    
                    # Get issue object for Claude execution
                    issue_obj = type('Issue', (), {
                        'number': existing_issue,
                        'title': work_item.title,
                        'body': work_item.description
                    })()
                    
                    claude_result = await self._execute_with_claude(work_item, repo_full_name, issue_obj)
                    
                    if claude_result['success']:
                        print(f"✅ Claude successfully implemented existing issue #{existing_issue}!")
                        # Update issue to mark as implemented
                        self.local_simulator.update_issue(
                            repo_full_name, 
                            existing_issue, 
                            labels=existing_issue_data.get('labels', []) + ['claude-implemented']
                        )
                        return {
                            'success': True,
                            'task_id': work_item.id,
                            'issue_number': existing_issue,
                            'repository': work_item.repository,
                            'value_created': self._calculate_value_created(work_item),
                            'execution_time': time.time(),
                            'existing_issue': True,
                            'claude_execution': claude_result
                        }
                    else:
                        print(f"❌ Claude execution failed for issue #{existing_issue}")
                        return {
                            'success': False,
                            'task_id': work_item.id,
                            'issue_number': existing_issue,
                            'repository': work_item.repository,
                            'error': f"Claude execution failed: {claude_result.get('error')}",
                            'execution_time': time.time(),
                            'existing_issue': True,
                            'claude_execution': claude_result
                        }
                else:
                    print(f"⏭️ Skipping issue #{existing_issue} - already exists and Claude execution disabled")
                    return {
                        'success': False,
                        'task_id': work_item.id,
                        'issue_number': existing_issue,
                        'repository': work_item.repository,
                        'error': 'Issue exists but not implemented (Claude execution disabled)',
                        'execution_time': time.time(),
                        'existing_issue': True,
                        'skipped': True
                    }
            
            # Convert work item to issue format
            issue_data = self._convert_work_item_to_issue(work_item)
            
            # Create local issue
            issue = self.local_simulator.create_issue(
                repo_full_name=repo_full_name,
                title=issue_data['title'],
                body=issue_data['body'],
                labels=issue_data['labels']
            )
            
            if issue:
                print(f"✅ Created local issue #{issue.number} for: {work_item.title}")
                
                # Save task mapping
                await self._save_task_mapping(work_item, issue.number)
                
                # NEW: Execute with Claude Code CLI if enabled
                if self.enable_claude_execution and self.claude_executable:
                    print(f"🤖 Executing with Claude Code CLI...")
                    claude_result = await self._execute_with_claude(work_item, repo_full_name, issue)
                    
                    if claude_result['success']:
                        print(f"✅ Claude successfully implemented the issue!")
                        # Update issue to mark as implemented
                        self.local_simulator.update_issue(
                            repo_full_name, 
                            issue.number, 
                            labels=issue_data['labels'] + ['claude-implemented']
                        )
                        
                        # Task is only complete if Claude succeeded
                        return {
                            'success': True,
                            'task_id': work_item.id,
                            'issue_number': issue.number,
                            'repository': work_item.repository,
                            'value_created': self._calculate_value_created(work_item),
                            'execution_time': time.time(),
                            'storage_mode': 'local',
                            'claude_execution': claude_result
                        }
                    else:
                        print(f"❌ Claude execution failed: {claude_result.get('error', 'Unknown error')}")
                        # Task failed because Claude couldn't implement it
                        return {
                            'success': False,
                            'task_id': work_item.id,
                            'issue_number': issue.number,
                            'repository': work_item.repository,
                            'error': f"Issue created but Claude execution failed: {claude_result.get('error')}",
                            'execution_time': time.time(),
                            'storage_mode': 'local',
                            'claude_execution': claude_result
                        }
                else:
                    # Claude execution disabled - task is not complete, just issue created
                    print(f"⚠️ Issue created but Claude execution disabled - task not complete")
                    return {
                        'success': False,
                        'task_id': work_item.id,
                        'issue_number': issue.number,
                        'repository': work_item.repository,
                        'error': 'Issue created but not implemented (Claude execution disabled)',
                        'value_created': 0,
                        'execution_time': time.time(),
                        'storage_mode': 'local',
                        'issue_only': True
                    }
            else:
                self.logger.warning(f"❌ Failed to create local issue for: {work_item.title}")
                return {
                    'success': False,
                    'error': 'Local issue creation failed',
                    'task_id': work_item.id,
                    'repository': work_item.repository,
                    'value_created': 0
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error executing work item {work_item.title}: {e}")
            return {
                'success': False,
                'error': str(e),
                'repository': work_item.repository,
                'value_created': 0
            }
    
    def _convert_work_item_to_issue(self, work_item: WorkItem) -> Dict[str, Any]:
        """Convert a work item to issue format.
        
        Args:
            work_item: Work item to convert
            
        Returns:
            Issue data dictionary
        """
        # Map priority
        legacy_priority = self.priority_map.get(work_item.priority, LegacyPriority.MEDIUM)
        
        # Create labels
        labels = [
            f"priority:{legacy_priority.value}",
            f"type:{work_item.task_type}",
            "ai-managed",
            "local-mode"
        ]
        
        # Create enhanced description
        description = self._create_enhanced_description(work_item)
        
        return {
            'title': work_item.title,
            'body': description,
            'labels': labels
        }
    
    def _create_enhanced_description(self, work_item: WorkItem, smart_task_type: Optional[SmartTaskType] = None) -> str:
        """Create an enhanced description for the issue.
        
        Args:
            work_item: Work item to create description for
            smart_task_type: The SmartTaskType if identified
            
        Returns:
            Enhanced description with context and details
        """
        # Get SmartTaskType if not provided
        if not smart_task_type:
            if work_item.task_type in self.task_type_map:
                smart_task_type = self.task_type_map[work_item.task_type]
            else:
                smart_task_type = get_task_type_for_string(work_item.task_type)
        
        description_parts = [
            f"**Task Type**: {work_item.task_type}",
            f"**Priority**: {work_item.priority.name if hasattr(work_item.priority, 'name') else str(work_item.priority)}",
            f"**Estimated AI Cycles**: {work_item.estimated_cycles}",
            "",
            "## Description",
            work_item.description,
            ""
        ]
        
        # Add SmartTaskType metadata if available
        if smart_task_type:
            metadata = self.task_system.get_task_metadata(smart_task_type.value)
            if metadata:
                description_parts.extend([
                    "## Task Intelligence",
                    f"- **Category**: {metadata.category}",
                    f"- **Complexity**: {metadata.complexity}",
                    f"- **Typical Duration**: {metadata.typical_duration_hours} hours",
                    f"- **Required Skills**: {', '.join(metadata.required_skills) if metadata.required_skills else 'None specified'}",
                    f"- **Risk Level**: {metadata.risk_level}",
                    ""
                ])
        
        # Add repository context if available
        if work_item.repository:
            description_parts.extend([
                "## Repository Context",
                f"This task is specific to the `{work_item.repository}` repository.",
                ""
            ])
        
        # Add metadata if available
        if work_item.metadata:
            description_parts.extend([
                "## Additional Context",
                ""
            ])
            
            for key, value in work_item.metadata.items():
                description_parts.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            
            description_parts.append("")
        
        # Add dependencies if any
        if work_item.dependencies:
            description_parts.extend([
                "## Dependencies",
                "This task depends on the completion of:",
                ""
            ])
            
            for dep in work_item.dependencies:
                description_parts.append(f"- {dep}")
            
            description_parts.append("")
        
        # Add automation footer
        description_parts.extend([
            "---",
            f"🤖 **Automatically generated by Continuous AI System (Local Mode)**",
            f"📅 **Created**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"🆔 **Work Item ID**: {work_item.id}",
            "",
            "_This issue was created locally by the 24/7 AI system._"
        ])
        
        return "\n".join(description_parts)
    
    def _calculate_value_created(self, work_item: WorkItem) -> float:
        """Calculate the value created by executing this work item.
        
        Args:
            work_item: The executed work item
            
        Returns:
            Value score (0.0 to 2.0)
        """
        base_values = {
            'FEATURE': 1.5,
            'BUG_FIX': 1.2,
            'TESTING': 1.0,
            'DOCUMENTATION': 0.8,
            'SYSTEM_IMPROVEMENT': 1.3,
            'NEW_PROJECT': 2.0,
            'RESEARCH': 0.7,
            'MAINTENANCE': 0.6,
            'INTEGRATION': 1.4,
            'REPOSITORY_HEALTH': 0.9
        }
        
        base_value = base_values.get(work_item.task_type, 0.5)
        
        # Priority multiplier
        priority_multipliers = {
            TaskPriority.CRITICAL: 1.5,
            TaskPriority.HIGH: 1.2,
            TaskPriority.MEDIUM: 1.0,
            TaskPriority.LOW: 0.8,
            TaskPriority.BACKGROUND: 0.6
        }
        
        priority_multiplier = priority_multipliers.get(work_item.priority, 1.0)
        
        # Complexity multiplier (based on estimated cycles)
        complexity_multiplier = min(1.0 + (work_item.estimated_cycles * 0.1), 1.5)
        
        final_value = base_value * priority_multiplier * complexity_multiplier
        
        return round(final_value, 2)
    
    async def _check_existing_issue(self, work_item: WorkItem) -> Optional[int]:
        """Check if an issue already exists for this work item.
        
        Args:
            work_item: Work item to check
            
        Returns:
            Issue number if exists, None otherwise
        """
        try:
            # Check local issues
            # Check if repository already contains organization prefix
            if "/" in work_item.repository:
                repo_full_name = work_item.repository
            else:
                repo_full_name = f"CodeWebMobile-AI/{work_item.repository}"
            print(f"Checking issues for repository: {repo_full_name}")
            issues = self.local_simulator.get_issues(
                repo_full_name=repo_full_name,
                state="all"
            )
            
            # Check for exact title match
            for issue in issues:
                if issue.title.lower() == work_item.title.lower():
                    print(f"Found exact title match: Issue #{issue.number}")
                    return issue.number
                
                # Check for work item ID in body
                if issue.body and f"Work Item ID**: {work_item.id}" in issue.body:
                    print(f"Found work item ID match: Issue #{issue.number}")
                    return issue.number
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error checking for existing issue: {e}")
            return None
    
    async def _save_task_mapping(self, work_item: WorkItem, issue_number: int):
        """Save mapping between work item and issue number.
        
        Args:
            work_item: The work item
            issue_number: The created issue number
        """
        # Update active tasks file
        active_tasks_file = self.local_simulator.state_path / "active_tasks.json"
        
        try:
            with open(active_tasks_file, 'r') as f:
                tasks_data = json.load(f)
        except:
            tasks_data = {"tasks": [], "total_count": 0, "last_updated": None}
        
        # Add task mapping
        task_entry = {
            "work_item_id": work_item.id,
            "issue_number": issue_number,
            "repository": work_item.repository,
            "title": work_item.title,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "assigned"
        }
        
        tasks_data["tasks"].append(task_entry)
        tasks_data["total_count"] += 1
        tasks_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # Save updated data
        with open(active_tasks_file, 'w') as f:
            json.dump(tasks_data, f, indent=2)
    
    def can_create_issues(self) -> bool:
        """Check if we can create local issues.
        
        Returns:
            True (always available for local mode)
        """
        return True
    
    async def get_open_issues(self) -> List[Dict[str, Any]]:
        """Get open issues from local repositories.
        
        Returns:
            List of open issues with metadata
        """
        all_issues = []
        
        try:
            # Get all repositories
            repos = self.local_simulator.list_repositories()
            
            for repo in repos:
                # Get open issues for each repository
                issues = self.local_simulator.get_issues(
                    repo_full_name=repo.full_name,
                    state="open"
                )
                
                for issue in issues:
                    all_issues.append({
                        'number': issue.number,
                        'title': issue.title,
                        'body': issue.body,
                        'html_url': issue.html_url,
                        'created_at': issue.created_at.isoformat(),
                        'labels': [{'name': label} for label in issue.labels],
                        'repository': {
                            'full_name': repo.full_name
                        },
                        'storage_mode': 'local'
                    })
            
            # Sort by creation date (newest first)
            all_issues.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Limit to recent issues (last 50)
            return all_issues[:50]
            
        except Exception as e:
            self.logger.error(f"Error in get_open_issues: {e}")
            return []
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limit status (always unlimited for local).
        
        Returns:
            Rate limit status information
        """
        return {
            'can_create_issue_now': True,
            'seconds_until_next_allowed': 0,
            'last_issue_created': 0,
            'min_interval': 0,
            'storage_mode': 'local',
            'rate_limit': 'unlimited'
        }
    
    def _find_claude_executable(self) -> Optional[str]:
        """Find the Claude Code CLI executable.
        
        Returns:
            Path to claude executable or None if not found
        """
        # Check common locations
        claude_paths = [
            shutil.which("claude"),  # System PATH
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            os.path.expanduser("~/.local/bin/claude"),
            "./claude"  # Local directory
        ]
        
        for path in claude_paths:
            if path and os.path.exists(path) and os.access(path, os.X_OK):
                self.logger.info(f"Found Claude Code CLI at: {path}")
                return path
        
        self.logger.warning("Claude Code CLI not found. Autonomous execution disabled.")
        return None
    
    async def _execute_with_claude(self, work_item: WorkItem, repo_full_name: str, issue: Any) -> Dict[str, Any]:
        """Execute the issue implementation using Claude Code CLI.
        
        Args:
            work_item: The work item to implement
            repo_full_name: Full repository name
            issue: The created issue object
            
        Returns:
            Execution result dictionary
        """
        try:
            # Get repository path
            org, repo_name = repo_full_name.split("/")
            repo_path = self.local_simulator.repos_path / org / repo_name
            
            if not repo_path.exists():
                return {
                    'success': False,
                    'error': f'Repository path not found: {repo_path}'
                }
            
            # Prepare Claude prompt based on issue type and description
            prompt = self._prepare_claude_prompt(work_item, issue)
            
            # Execute Claude Code CLI
            self.logger.info(f"Executing Claude for issue #{issue.number} in {repo_path}")
            self.logger.info(f"Claude executable: {self.claude_executable}")
            self.logger.info(f"Working directory: {repo_path}")
            self.logger.info(f"Prompt length: {len(prompt)} characters")
            
            # Log first 500 chars of prompt for debugging
            self.logger.debug(f"Prompt preview: {prompt[:500]}...")
            
            start_time = time.time()
            
            # Track Claude execution with monitoring
            with self.monitoring.track_execution_step(work_item.id, "claude_execution"):
                # Claude Code needs the prompt via stdin, not as argument
                self.logger.info(f"Running Claude Code with prompt via stdin")
            
            # Use Popen to interact with Claude with auto-permissions
            process = subprocess.Popen(
                [self.claude_executable, '--dangerously-skip-permissions'],
                cwd=str(repo_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered for real-time output
            )
            
            # Stream output in real-time
            import select
            import fcntl
            
            # Make stdout non-blocking
            fl = fcntl.fcntl(process.stdout, fcntl.F_GETFL)
            fcntl.fcntl(process.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            
            # Collect output while streaming
            stdout_lines = []
            stderr_lines = []
            
            def read_output():
                try:
                    while True:
                        reads = [process.stdout, process.stderr]
                        ret = select.select(reads, [], [], 0.1)
                        
                        for fd in ret[0]:
                            if fd == process.stdout:
                                line = fd.readline()
                                if line:
                                    stdout_lines.append(line)
                                    self.logger.info(f"[CLAUDE] {line.rstrip()}")
                            elif fd == process.stderr:
                                line = fd.readline()
                                if line:
                                    stderr_lines.append(line)
                                    self.logger.warning(f"[CLAUDE ERR] {line.rstrip()}")
                        
                        if process.poll() is not None:
                            break
                except Exception as e:
                    self.logger.debug(f"Error reading output: {e}")
            
            # Send prompt and close stdin to signal we're done
            # NO TIMEOUT - let Claude run as long as it needs
            self.logger.info(f"Claude execution started, waiting for completion...")
            
            # Log progress every 30 seconds
            import threading
            stop_progress = threading.Event()
            
            def log_progress():
                elapsed = 0
                last_output_time = time.time()
                while not stop_progress.is_set():
                    # Check if Claude is producing output
                    if process.poll() is None:  # Still running
                        current_time = time.time()
                        if current_time - last_output_time > 30:
                            elapsed = int(current_time - start_time)
                            self.logger.info(f"[CLAUDE STATUS] Still working... ({elapsed}s elapsed)")
                            last_output_time = current_time
                    stop_progress.wait(5)  # Check more frequently
            
            progress_thread = threading.Thread(target=log_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Send prompt and start reading output
            process.stdin.write(prompt)
            process.stdin.close()
            
            # Read output in real-time
            output_thread = threading.Thread(target=read_output)
            output_thread.daemon = True
            output_thread.start()
            
            # Wait for process to complete
            process.wait()
            stop_progress.set()
            output_thread.join(timeout=1)
            
            # Collect final output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            
            result = subprocess.CompletedProcess(
                args=[self.claude_executable],
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
            execution_time = time.time() - start_time
            
            # Track Claude execution with monitoring
            self.monitoring.track_claude_execution(
                work_item.id, 
                success=(result.returncode == 0),
                duration=execution_time
            )
            
            self.logger.info(f"Claude execution completed in {execution_time:.2f}s with return code: {result.returncode}")
            
            # Log warning if execution took too long
            if execution_time > 300:  # 5 minutes
                self.logger.warning(f"Claude execution took {execution_time:.2f}s (>{execution_time/60:.1f} minutes) for issue #{issue.number}")
            
            if result.returncode == 0:
                self.logger.info(f"Claude execution successful in {execution_time:.2f}s")
                
                # Check if any files were changed
                git_status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True
                )
                
                files_changed = len(git_status.stdout.strip().split('\n')) if git_status.stdout.strip() else 0
                
                self.logger.info(f"Git status after Claude execution:")
                self.logger.info(f"Files changed: {files_changed}")
                if git_status.stdout.strip():
                    self.logger.info(f"Changed files:\n{git_status.stdout}")
                
                # Also log Claude's output for debugging
                self.logger.info(f"Claude output length: {len(result.stdout)} characters")
                if result.stdout:
                    self.logger.debug(f"Claude output preview: {result.stdout[:500]}...")
                
                return {
                    'success': True,
                    'execution_time': execution_time,
                    'files_changed': files_changed,
                    'output': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout  # Last 1000 chars
                }
            else:
                self.logger.error(f"Claude execution failed with return code: {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                self.logger.error(f"STDOUT: {result.stdout}")
                # Log more context for debugging
                self.logger.error(f"Working directory: {repo_path}")
                self.logger.error(f"Prompt length: {len(prompt)} characters")
                self.logger.error(f"Execution time: {execution_time:.2f}s")
                
                return {
                    'success': False,
                    'error': result.stderr or result.stdout or "Claude execution failed",
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'execution_time': execution_time
                }
                
        except Exception as e:
            import traceback
            self.logger.error(f"Error executing Claude: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            self.logger.error(f"Working directory: {repo_path}")
            self.logger.error(f"Claude executable: {self.claude_executable}")
            
            return {
                'success': False,
                'error': str(e),
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
    
    async def _get_issue_details(self, repo_full_name: str, issue_number: int) -> Optional[Dict[str, Any]]:
        """Get details of an existing issue.
        
        Args:
            repo_full_name: Full repository name
            issue_number: Issue number
            
        Returns:
            Issue details dict or None
        """
        try:
            issues = self.local_simulator.get_issues(repo_full_name, state="all")
            for issue in issues:
                if issue.number == issue_number:
                    return {
                        'number': issue.number,
                        'title': issue.title,
                        'body': issue.body,
                        'labels': issue.labels,
                        'state': issue.state,
                        'created_at': issue.created_at
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error getting issue details: {e}")
            return None
    
    def _prepare_claude_prompt(self, work_item: WorkItem, issue: Any) -> str:
        """Prepare a prompt for Claude based on the work item.
        
        Args:
            work_item: The work item
            issue: The issue object
            
        Returns:
            Prompt string for Claude
        """
        # Extract key information
        task_type = work_item.task_type
        title = work_item.title
        description = work_item.description
        
        # Build a focused prompt
        prompt_parts = [
            f"Please implement the following task from issue #{issue.number}:",
            f"",
            f"Title: {title}",
            f"Type: {task_type}",
            f"",
            f"Description:",
            description,
            f"",
            f"Requirements:",
        ]
        
        # Add specific instructions based on task type
        if "TEST" in task_type:
            prompt_parts.extend([
                "- Write comprehensive tests",
                "- Ensure all tests pass",
                "- Follow existing test patterns in the codebase"
            ])
        elif "FEATURE" in task_type:
            prompt_parts.extend([
                "- Implement the feature as described",
                "- Add appropriate tests",
                "- Update documentation if needed"
            ])
        elif "BUG_FIX" in task_type:
            prompt_parts.extend([
                "- Identify and fix the bug",
                "- Add regression tests",
                "- Ensure no existing tests break"
            ])
        elif "OPTIMIZATION" in task_type:
            prompt_parts.extend([
                "- Optimize the code for better performance",
                "- Maintain functionality",
                "- Add benchmarks if appropriate"
            ])
        elif "DOCUMENTATION" in task_type:
            prompt_parts.extend([
                "- Update or create documentation",
                "- Ensure accuracy and clarity",
                "- Follow project documentation standards"
            ])
        else:
            prompt_parts.extend([
                "- Complete the task as described",
                "- Follow best practices",
                "- Ensure code quality"
            ])
        
        prompt_parts.extend([
            "",
            "Please make the necessary changes to implement this task.",
            "You have permission to create and edit files as needed.",
            "If you need to create new files, place them in appropriate directories.",
            "Ensure all changes follow the project's coding standards.",
            "",
            "IMPORTANT: Please go ahead and make the actual file changes now.",
            "Do not wait for additional permission - implement the solution directly."
        ])
        
        return "\n".join(prompt_parts)