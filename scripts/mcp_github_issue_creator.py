"""
MCP-Enabled GitHub Issue Creator for Continuous AI System

Uses MCP (Model Context Protocol) for GitHub operations instead of direct API calls.
This provides better reliability, rate limiting, and standardized error handling.
"""

import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import os

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.mcp_integration import MCPIntegrationHub
from scripts.task_manager import TaskManager, TaskType, TaskPriority as LegacyPriority


class MCPGitHubIssueCreator:
    """Creates real GitHub issues using MCP integration."""
    
    def __init__(self, mcp_hub: Optional[MCPIntegrationHub] = None):
        """Initialize the MCP-enabled GitHub issue creator.
        
        Args:
            mcp_hub: Optional pre-initialized MCP integration hub
        """
        self.logger = logging.getLogger(__name__)
        self.mcp_hub = mcp_hub
        self._initialized = False
        
        # Rate limiting for GitHub API
        self.last_issue_created = 0
        self.min_issue_interval = 10  # Minimum 10 seconds between issues
        
        # Priority mapping
        self.priority_map = {
            TaskPriority.CRITICAL: LegacyPriority.CRITICAL,
            TaskPriority.HIGH: LegacyPriority.HIGH,
            TaskPriority.MEDIUM: LegacyPriority.MEDIUM,
            TaskPriority.LOW: LegacyPriority.LOW,
            TaskPriority.BACKGROUND: LegacyPriority.LOW
        }
        
        # Task type mapping
        self.task_type_map = {
            'TESTING': TaskType.TESTING,
            'FEATURE': TaskType.FEATURE,
            'BUG_FIX': TaskType.BUG_FIX,
            'DOCUMENTATION': TaskType.DOCUMENTATION,
            'RESEARCH': TaskType.DOCUMENTATION,
            'SYSTEM_IMPROVEMENT': TaskType.PERFORMANCE,
            'MAINTENANCE': TaskType.REFACTOR,
            'NEW_PROJECT': TaskType.NEW_PROJECT,
            'INTEGRATION': TaskType.FEATURE,
            'REPOSITORY_HEALTH': TaskType.REFACTOR
        }
        
        # Labels for different task types
        self.label_map = {
            'TESTING': ['testing', 'automated-test'],
            'FEATURE': ['enhancement', 'feature'],
            'BUG_FIX': ['bug', 'fix'],
            'DOCUMENTATION': ['documentation', 'docs'],
            'RESEARCH': ['research', 'investigation'],
            'SYSTEM_IMPROVEMENT': ['performance', 'optimization'],
            'MAINTENANCE': ['maintenance', 'refactor'],
            'NEW_PROJECT': ['new-project', 'greenfield'],
            'INTEGRATION': ['integration', 'third-party'],
            'REPOSITORY_HEALTH': ['repo-health', 'maintenance']
        }
    
    async def _ensure_initialized(self):
        """Ensure MCP is initialized."""
        if not self._initialized:
            if not self.mcp_hub:
                self.mcp_hub = MCPIntegrationHub()
            
            # Initialize only GitHub MCP
            await self.mcp_hub.initialize(servers=['github'])
            self._initialized = True
    
    async def execute_work_item(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a work item by creating a real GitHub issue using MCP.
        
        Args:
            work_item: The work item to execute
            
        Returns:
            Execution result with success status and details
        """
        await self._ensure_initialized()
        
        self.logger.info(f"Creating GitHub issue via MCP for: {work_item.title}")
        
        # Rate limiting
        await self._enforce_rate_limit()
        
        try:
            # Check if issue already exists
            existing_issue = await self._check_existing_issue_mcp(work_item)
            if existing_issue:
                self.logger.info(f"✅ Issue already exists: #{existing_issue['number']} for: {work_item.title}")
                return {
                    'success': True,
                    'task_id': work_item.id,
                    'issue_number': existing_issue['number'],
                    'repository': work_item.repository,
                    'value_created': 0,
                    'execution_time': time.time(),
                    'existing_issue': True
                }
            
            # Determine target repository
            target_repo = self._determine_target_repository(work_item)
            if not target_repo:
                return {
                    'success': False,
                    'error': 'Could not determine target repository',
                    'task_id': work_item.id,
                    'value_created': 0
                }
            
            # Create enhanced description
            description = self._create_enhanced_description(work_item)
            
            # Get labels for the task type
            labels = self.label_map.get(work_item.task_type, ['enhancement'])
            
            # Add priority label
            if work_item.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                labels.append(f"priority-{work_item.priority.name.lower()}")
            
            # Create issue via MCP
            result = await self.mcp_hub.github.create_issue(
                repo=target_repo,
                title=work_item.title,
                body=description,
                labels=labels
            )
            
            if result and 'number' in result:
                self.logger.info(f"✅ Created GitHub issue #{result['number']} for: {work_item.title}")
                return {
                    'success': True,
                    'task_id': work_item.id,
                    'issue_number': result['number'],
                    'issue_url': result.get('html_url'),
                    'repository': target_repo,
                    'value_created': self._calculate_value_created(work_item),
                    'execution_time': time.time()
                }
            else:
                self.logger.warning(f"❌ Failed to create GitHub issue for: {work_item.title}")
                return {
                    'success': False,
                    'error': 'GitHub issue creation failed',
                    'task_id': work_item.id,
                    'repository': target_repo,
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
    
    def _determine_target_repository(self, work_item: WorkItem) -> Optional[str]:
        """Determine the target repository for the issue.
        
        Args:
            work_item: Work item to process
            
        Returns:
            Full repository name (owner/repo) or None
        """
        if work_item.repository:
            # Use the work item's repository
            if '/' in work_item.repository:
                # Full repo name provided
                return work_item.repository
            else:
                # Just repo name, assume same organization as main repo
                main_repo = os.getenv('GITHUB_REPOSITORY', '')
                if main_repo and '/' in main_repo:
                    org_name = main_repo.split('/')[0]
                    return f"{org_name}/{work_item.repository}"
                else:
                    # Fallback to CodeWebMobile-AI org
                    return f"CodeWebMobile-AI/{work_item.repository}"
        else:
            # No repository specified, use main repo
            return os.getenv('GITHUB_REPOSITORY', '')
    
    async def _check_existing_issue_mcp(self, work_item: WorkItem) -> Optional[Dict]:
        """Check if an issue already exists using MCP.
        
        Args:
            work_item: Work item to check
            
        Returns:
            Issue data if exists, None otherwise
        """
        try:
            target_repo = self._determine_target_repository(work_item)
            if not target_repo:
                return None
            
            # Search for existing issues
            existing_issues = await self.mcp_hub.github.list_issues(
                repo=target_repo,
                state='open'
            )
            
            # Check for exact title match or high similarity
            for issue in existing_issues:
                # Exact title match
                if issue['title'].lower() == work_item.title.lower():
                    self.logger.info(f"Found exact title match: Issue #{issue['number']}")
                    return issue
                
                # Check for work item ID in body
                if issue.get('body') and f"Work Item ID**: {work_item.id}" in issue['body']:
                    self.logger.info(f"Found work item ID match: Issue #{issue['number']}")
                    return issue
                
                # Check similarity
                similarity = self._calculate_title_similarity(issue['title'], work_item.title)
                threshold = 0.85
                
                if similarity > threshold:
                    self.logger.info(
                        f"Found similar task: Issue #{issue['number']} "
                        f"(similarity: {similarity:.2%})"
                    )
                    return issue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error checking for existing issue: {e}")
            return None
    
    def _create_enhanced_description(self, work_item: WorkItem) -> str:
        """Create an enhanced description for the GitHub issue.
        
        Args:
            work_item: Work item to create description for
            
        Returns:
            Enhanced description with context and details
        """
        description_parts = [
            f"**Task Type**: {work_item.task_type}",
            f"**Priority**: {work_item.priority.name if hasattr(work_item.priority, 'name') else str(work_item.priority)}",
            f"**Estimated AI Cycles**: {work_item.estimated_cycles}",
            "",
            "## Description",
            work_item.description,
            ""
        ]
        
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
        
        # Add @claude mention for AI implementation
        description_parts.extend([
            "## Implementation",
            "@claude Please implement this task following the description above.",
            ""
        ])
        
        # Add automation footer
        description_parts.extend([
            "---",
            f"🤖 **Automatically generated by Continuous AI System**",
            f"📅 **Created**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"🆔 **Work Item ID**: {work_item.id}",
            f"🔧 **Created via MCP Integration**",
            "",
            "_This issue was intelligently discovered and created by the 24/7 AI system using Model Context Protocol._"
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
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles using Jaccard similarity."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting for GitHub API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_issue_created
        
        if time_since_last < self.min_issue_interval:
            wait_time = self.min_issue_interval - time_since_last
            self.logger.debug(f"Rate limiting: waiting {wait_time:.1f}s before creating issue")
            await asyncio.sleep(wait_time)
        
        self.last_issue_created = time.time()
    
    def can_create_issues(self) -> bool:
        """Check if we can create GitHub issues.
        
        Returns:
            True if GitHub issue creation is possible
        """
        # Check for GitHub token in environment
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        github_repo = os.getenv('GITHUB_REPOSITORY')
        
        return bool(github_token and github_repo)
    
    async def list_recent_issues(self, repo: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """List recent issues in a repository using MCP.
        
        Args:
            repo: Repository name (uses default if not provided)
            limit: Maximum number of issues to return
            
        Returns:
            List of issue data
        """
        await self._ensure_initialized()
        
        if not repo:
            repo = os.getenv('GITHUB_REPOSITORY', '')
        
        issues = await self.mcp_hub.github.list_issues(repo=repo, state='open')
        return issues[:limit]
    
    async def search_repositories(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for repositories using MCP.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of repository data
        """
        await self._ensure_initialized()
        
        return await self.mcp_hub.github.search_repositories(query=query, limit=limit)
    
    async def close(self):
        """Close MCP connections."""
        if self._initialized and self.mcp_hub:
            await self.mcp_hub.close()
            self._initialized = False