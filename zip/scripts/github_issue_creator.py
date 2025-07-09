"""
GitHub Issue Creator for Continuous AI System

Creates real GitHub issues instead of fake tasks, integrating with the existing
task management system for proper issue creation and tracking.
"""

import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import os

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.task_manager import TaskManager, TaskType, TaskPriority as LegacyPriority


class GitHubIssueCreator:
    """Creates real GitHub issues from work items."""
    
    def __init__(self):
        """Initialize the GitHub issue creator."""
        self.logger = logging.getLogger(__name__)
        # Initialize without default repository
        self.task_manager = TaskManager(repository=None)
        
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
        
        # Task type mapping (map to available TaskType values)
        self.task_type_map = {
            'TESTING': TaskType.TESTING,  # Map testing to testing
            'FEATURE': TaskType.FEATURE,
            'BUG_FIX': TaskType.BUG_FIX,
            'DOCUMENTATION': TaskType.DOCUMENTATION,
            'RESEARCH': TaskType.DOCUMENTATION,  # Research becomes documentation
            'SYSTEM_IMPROVEMENT': TaskType.PERFORMANCE,  # System improvement becomes performance
            'MAINTENANCE': TaskType.REFACTOR,  # Maintenance becomes refactor
            'NEW_PROJECT': TaskType.NEW_PROJECT,
            'INTEGRATION': TaskType.FEATURE,  # Integration becomes feature
            'REPOSITORY_HEALTH': TaskType.REFACTOR  # Repository health becomes refactor
        }
    
    async def execute_work_item(self, work_item: WorkItem) -> Dict[str, Any]:
        """Execute a work item by creating a real GitHub issue.
        
        Args:
            work_item: The work item to execute
            
        Returns:
            Execution result with success status and details
        """
        self.logger.info(f"Creating GitHub issue for: {work_item.title}")
        
        # Rate limiting
        await self._enforce_rate_limit()
        
        try:
            # Check if issue already exists (idempotency check)
            existing_issue = await self._check_existing_issue(work_item)
            if existing_issue == -1:
                # Task exists locally but no GitHub issue yet
                self.logger.info(f"⏭️ Task already exists locally (no GitHub issue yet): {work_item.title}")
                return {
                    'success': True,
                    'task_id': work_item.id,
                    'repository': work_item.repository,
                    'value_created': 0,
                    'duplicate': True,
                    'duplicate_type': 'local_task',
                    'execution_time': time.time()
                }
            elif existing_issue:
                self.logger.info(f"✅ Issue already exists: #{existing_issue} for: {work_item.title}")
                return {
                    'success': True,
                    'task_id': work_item.id,
                    'issue_number': existing_issue,
                    'repository': work_item.repository,
                    'value_created': 0,  # No new value since issue already exists
                    'execution_time': time.time(),
                    'existing_issue': True
                }
            
            # Convert work item to task format
            task = self._convert_work_item_to_task(work_item)
            
            # Create GitHub issue
            issue_number = self.task_manager.create_github_issue(task)
            
            if issue_number:
                self.logger.info(f"✅ Created GitHub issue #{issue_number} for: {work_item.title}")
                return {
                    'success': True,
                    'task_id': task['id'],
                    'issue_number': issue_number,
                    'repository': work_item.repository,
                    'value_created': self._calculate_value_created(work_item),
                    'execution_time': time.time()
                }
            else:
                self.logger.warning(f"❌ Failed to create GitHub issue for: {work_item.title}")
                return {
                    'success': False,
                    'error': 'GitHub issue creation failed',
                    'task_id': task['id'],
                    'repository': work_item.repository,
                    'value_created': 0
                }
                
        except ValueError as e:
            # Handle duplicate task exception
            if "Duplicate task already exists" in str(e):
                self.logger.warning(f"⏭️ Skipping duplicate task: {work_item.title}")
                return {
                    'success': True,  # Consider it success since task exists
                    'task_id': work_item.id,
                    'repository': work_item.repository,
                    'value_created': 0,
                    'error': 'Duplicate task',
                    'duplicate': True
                }
            else:
                raise
                
        except Exception as e:
            self.logger.error(f"❌ Error executing work item {work_item.title}: {e}")
            return {
                'success': False,
                'error': str(e),
                'repository': work_item.repository,
                'value_created': 0
            }
    
    def _convert_work_item_to_task(self, work_item: WorkItem) -> Dict[str, Any]:
        """Convert a work item to task manager format.
        
        Args:
            work_item: Work item to convert
            
        Returns:
            Task dictionary compatible with TaskManager
        """
        # Map task type
        legacy_task_type = self.task_type_map.get(work_item.task_type, TaskType.FEATURE)
        
        # Map priority
        legacy_priority = self.priority_map.get(work_item.priority, LegacyPriority.MEDIUM)
        
        # Create enhanced description
        description = self._create_enhanced_description(work_item)
        
        # Create task using task manager
        task = self.task_manager.create_task(
            task_type=legacy_task_type,
            title=work_item.title,
            description=description,
            priority=legacy_priority,
            repository=work_item.repository
        )
        
        return task
    
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
        
        # Add automation footer
        description_parts.extend([
            "---",
            f"🤖 **Automatically generated by Continuous AI System**",
            f"📅 **Created**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"🆔 **Work Item ID**: {work_item.id}",
            "",
            "_This issue was intelligently discovered and created by the 24/7 AI system._"
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
        """Check if we can create GitHub issues (have proper credentials).
        
        Returns:
            True if GitHub issue creation is possible
        """
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        github_repo = os.getenv('GITHUB_REPOSITORY')
        
        self.logger.debug(f"🔍 Checking GitHub integration:")
        self.logger.debug(f"   - GITHUB_TOKEN exists: {bool(os.getenv('GITHUB_TOKEN'))}")
        self.logger.debug(f"   - CLAUDE_PAT exists: {bool(os.getenv('CLAUDE_PAT'))}")
        self.logger.debug(f"   - Combined token exists: {bool(github_token)}")
        self.logger.debug(f"   - GITHUB_REPOSITORY: {github_repo}")
        self.logger.debug(f"   - Result: {bool(github_token and github_repo)}")
        
        return bool(github_token and github_repo)
    
    async def _check_existing_issue(self, work_item: WorkItem) -> Optional[int]:
        """Check if an issue already exists for this work item.
        
        Args:
            work_item: Work item to check
            
        Returns:
            Issue number if exists, None otherwise
        """
        try:
            # First check local task state to avoid duplicates
            from scripts.task_manager import TaskManager, TaskType
            # Create task manager for specific repository
            task_manager = TaskManager(repository=repo_full_name)
            
            # Convert work item type to TaskType
            try:
                task_type = TaskType(work_item.task_type.lower())
                existing_task = task_manager._find_duplicate_task(
                    task_type,
                    work_item.title,
                    work_item.description,
                    work_item.repository
                )
                
                if existing_task:
                    self.logger.info(f"Found existing task in local state: {existing_task.get('id')}")
                    # Return the GitHub issue number if it exists
                    issue_number = existing_task.get('github_issue_number')
                    if issue_number:
                        return issue_number
                    # If no GitHub issue yet, prevent creation of duplicate
                    self.logger.warning(f"Task exists locally but no GitHub issue yet: {existing_task.get('id')}")
                    return -1  # Special value to indicate task exists but no issue
            except (ValueError, AttributeError) as e:
                self.logger.debug(f"Could not check local task state: {e}")
                # Continue with GitHub check
            
            # Then check GitHub issues
            # Get GitHub token
            github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
            if not github_token:
                return None
            
            # Import GitHub library
            from github import Github
            g = Github(github_token)
            
            # Determine the target repository
            target_repo_name = None
            if work_item.repository:
                # Use the work item's repository
                if '/' in work_item.repository:
                    # Full repo name provided
                    target_repo_name = work_item.repository
                else:
                    # Just repo name, assume same organization as main repo
                    main_repo = os.getenv('GITHUB_REPOSITORY', '')
                    if main_repo and '/' in main_repo:
                        org_name = main_repo.split('/')[0]
                        target_repo_name = f"{org_name}/{work_item.repository}"
                    else:
                        # Fallback to CodeWebMobile-AI org
                        target_repo_name = f"CodeWebMobile-AI/{work_item.repository}"
            else:
                # No repository specified, use main repo
                target_repo_name = os.getenv('GITHUB_REPOSITORY', '')
            
            if not target_repo_name:
                self.logger.warning("No target repository determined for duplicate check")
                return None
            
            # Get the repository
            try:
                repo = g.get_repo(target_repo_name)
                self.logger.debug(f"Checking for duplicates in repository: {target_repo_name}")
            except Exception as e:
                # Check if it's a 404 error (repository doesn't exist)
                if "404" in str(e):
                    self.logger.debug(f"Repository {target_repo_name} does not exist, skipping duplicate check")
                else:
                    self.logger.warning(f"Failed to access repository {target_repo_name}: {e}")
                return None
            
            # Use GitHub search API for more efficient duplicate detection
            self.logger.debug(f"Searching for duplicates of: {work_item.title}")
            
            # Extract key terms from title for search
            title_terms = self._extract_search_terms(work_item.title)
            
            # Build search queries
            search_queries = [
                f'repo:{target_repo_name} is:issue state:open "{work_item.title}"',  # Exact title
                f'repo:{target_repo_name} is:issue state:open {title_terms}',  # Key terms
                f'repo:{target_repo_name} is:issue "Work Item ID** {work_item.id}"'  # Work item ID
            ]
            
            # Also check recently closed issues (last 30 days)
            from datetime import datetime, timedelta
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            search_queries.append(
                f'repo:{target_repo_name} is:issue state:closed closed:>{thirty_days_ago} "{work_item.title}"'
            )
            
            checked_issues = set()  # Track checked issue numbers to avoid duplicates
            
            # Define similarity thresholds for different task types
            similarity_thresholds = {
                'DOCUMENTATION': 0.85,
                'BUG_FIX': 0.90,
                'FEATURE': 0.88,
                'TESTING': 0.87,
                'NEW_PROJECT': 0.95,  # Higher threshold for new projects
                'RESEARCH': 0.85
            }
            
            for query in search_queries:
                try:
                    self.logger.debug(f"Searching with query: {query}")
                    search_results = g.search_issues(query=query)
                    
                    for issue in search_results:
                        if issue.number in checked_issues:
                            continue
                        checked_issues.add(issue.number)
                        
                        # Check for exact title match
                        if issue.title.lower() == work_item.title.lower():
                            state_info = f" (closed {issue.closed_at})" if issue.state == 'closed' else ""
                            self.logger.info(f"Found exact title match: Issue #{issue.number}{state_info}")
                            return issue.number
                        
                        # Check for work item ID in body
                        if issue.body and f"Work Item ID**: {work_item.id}" in issue.body:
                            self.logger.info(f"Found work item ID match: Issue #{issue.number}")
                            return issue.number
                        
                        # Enhanced similarity check with multiple algorithms
                        similarity = self._calculate_enhanced_similarity(issue.title, work_item.title)
                        
                        threshold = similarity_thresholds.get(work_item.task_type, 0.88)
                        
                        if similarity > threshold:
                            state_info = f" (closed {issue.closed_at})" if issue.state == 'closed' else ""
                            self.logger.info(
                                f"Found similar {work_item.task_type} task: Issue #{issue.number} "
                                f"(similarity: {similarity:.2%}){state_info}"
                            )
                            return issue.number
                        
                        # Stop after checking 50 issues per query
                        if len(checked_issues) > 50:
                            break
                            
                except Exception as e:
                    self.logger.warning(f"Error searching with query '{query}': {e}")
                    continue
            
            # Fallback to traditional iteration if search fails
            if len(checked_issues) == 0:
                self.logger.debug("Search API failed, falling back to iteration method")
                issues = repo.get_issues(state='open', sort='created', direction='desc')
                
                count = 0
                for issue in issues:
                    if count >= 100:  # Reduced from 200 for better performance
                        break
                    count += 1
                    
                    # Check for exact title match
                    if issue.title.lower() == work_item.title.lower():
                        self.logger.info(f"Found exact title match: Issue #{issue.number}")
                        return issue.number
                    
                    # Check for high similarity
                    similarity = self._calculate_enhanced_similarity(issue.title, work_item.title)
                    threshold = similarity_thresholds.get(work_item.task_type, 0.88)
                    
                    if similarity > threshold:
                        self.logger.info(f"Found similar task: Issue #{issue.number} (similarity: {similarity:.2%})")
                        return issue.number
            
            self.logger.debug(f"No duplicates found after checking {len(checked_issues)} unique issues")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error checking for existing issue: {e}")
            return None
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles using Jaccard similarity."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_enhanced_similarity(self, title1: str, title2: str) -> float:
        """Calculate enhanced similarity using multiple algorithms.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Combined similarity score (0.0 to 1.0)
        """
        # Normalize titles
        title1_lower = title1.lower().strip()
        title2_lower = title2.lower().strip()
        
        # Exact match
        if title1_lower == title2_lower:
            return 1.0
        
        # Calculate Jaccard similarity
        jaccard_sim = self._calculate_title_similarity(title1, title2)
        
        # Calculate Levenshtein distance-based similarity
        lev_sim = self._levenshtein_similarity(title1_lower, title2_lower)
        
        # Calculate token overlap with common words removed
        token_sim = self._token_overlap_similarity(title1_lower, title2_lower)
        
        # Check for substring containment
        substring_bonus = 0.0
        if title1_lower in title2_lower or title2_lower in title1_lower:
            substring_bonus = 0.2
        
        # Weighted combination
        combined_similarity = (
            jaccard_sim * 0.3 +
            lev_sim * 0.3 +
            token_sim * 0.4 +
            substring_bonus
        )
        
        return min(1.0, combined_similarity)
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein distance-based similarity.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple Levenshtein distance implementation
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Create distance matrix
        dist = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dist[i][0] = i
        for j in range(len2 + 1):
            dist[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    dist[i][j] = dist[i-1][j-1]
                else:
                    dist[i][j] = min(
                        dist[i-1][j] + 1,    # deletion
                        dist[i][j-1] + 1,    # insertion
                        dist[i-1][j-1] + 1   # substitution
                    )
        
        # Convert distance to similarity
        max_len = max(len1, len2)
        return 1.0 - (dist[len1][len2] / max_len)
    
    def _token_overlap_similarity(self, title1: str, title2: str) -> float:
        """Calculate token overlap similarity with stop words removed.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Common stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be'
        }
        
        # Extract meaningful tokens
        tokens1 = {word for word in title1.split() if word not in stop_words and len(word) > 2}
        tokens2 = {word for word in title2.split() if word not in stop_words and len(word) > 2}
        
        if not tokens1 or not tokens2:
            # Fall back to simple word comparison if no meaningful tokens
            return self._calculate_title_similarity(title1, title2)
        
        # Calculate overlap
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_search_terms(self, title: str) -> str:
        """Extract key search terms from title for GitHub search.
        
        Args:
            title: Issue title
            
        Returns:
            Space-separated search terms
        """
        # Remove common words and symbols
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'add', 'create', 'implement', 'fix', 'update', 'improve', 'enhance'
        }
        
        # Extract words
        words = title.lower().split()
        
        # Filter out stop words and short words
        key_terms = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Take the most important terms (max 5)
        return ' '.join(key_terms[:5])
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status.
        
        Returns:
            Rate limit status information
        """
        current_time = time.time()
        time_since_last = current_time - self.last_issue_created
        can_create_now = time_since_last >= self.min_issue_interval
        
        return {
            'can_create_issue_now': can_create_now,
            'seconds_until_next_allowed': max(0, self.min_issue_interval - time_since_last),
            'last_issue_created': self.last_issue_created,
            'min_interval': self.min_issue_interval
        }