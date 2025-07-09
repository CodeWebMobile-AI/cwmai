"""
Simple Work Generator

A lightweight work generator that creates work items without heavy AI analysis.
Used as a fallback when the intelligent work finder takes too long.
"""

import random
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging

from scripts.work_item_types import WorkItem, TaskPriority
from scripts.task_types import SmartTaskType, TaskCategory
from scripts.ai_task_content_generator import AITaskContentGenerator


class SimpleWorkGenerator:
    """Simple work generator for quick work item creation."""
    
    def __init__(self, ai_brain=None):
        """Initialize the simple work generator.
        
        Args:
            ai_brain: Optional AI brain for content generation
        """
        self.logger = logging.getLogger(__name__)
        self.ai_brain = ai_brain
        self.content_generator = AITaskContentGenerator(ai_brain) if ai_brain else None
        
        # Task type distribution weights
        self.task_weights = {
            TaskCategory.DEVELOPMENT: 0.35,
            TaskCategory.TESTING: 0.20,
            TaskCategory.DOCUMENTATION: 0.15,
            TaskCategory.OPTIMIZATION: 0.10,
            TaskCategory.SECURITY: 0.10,
            TaskCategory.MAINTENANCE: 0.05,
            TaskCategory.INTEGRATION: 0.05
        }
        
    def generate_work_items(self, 
                          projects: Dict[str, Any], 
                          max_items: int = 5,
                          excluded_repos: List[str] = None) -> List[WorkItem]:
        """Generate work items for active projects.
        
        Args:
            projects: Dictionary of project data
            max_items: Maximum number of work items to generate
            excluded_repos: List of repositories to exclude
            
        Returns:
            List of generated work items
        """
        excluded_repos = excluded_repos or []
        work_items = []
        
        # Filter active projects
        active_projects = {
            name: data for name, data in projects.items()
            if data.get('status') == 'active' and name not in excluded_repos
        }
        
        if not active_projects:
            self.logger.warning("No active projects found for work generation")
            return []
        
        # Generate work items
        for _ in range(max_items):
            # Select random project
            project_name = random.choice(list(active_projects.keys()))
            project_data = active_projects[project_name]
            
            # Select task category based on weights
            task_category = self._select_task_category()
            
            # Generate work item
            work_item = self._create_work_item(project_name, project_data, task_category)
            if work_item:
                work_items.append(work_item)
        
        return work_items
    
    def _select_task_category(self) -> TaskCategory:
        """Select a task category based on weights.
        
        Returns:
            Selected task category
        """
        categories = list(self.task_weights.keys())
        weights = list(self.task_weights.values())
        return random.choices(categories, weights=weights)[0]
    
    def _create_work_item(self, 
                         project_name: str, 
                         project_data: Dict[str, Any],
                         task_category: TaskCategory) -> Optional[WorkItem]:
        """Create a single work item.
        
        Args:
            project_name: Name of the project
            project_data: Project data dictionary
            task_category: Task category to create
            
        Returns:
            WorkItem or None if creation failed
        """
        try:
            # Map category to task type
            category_to_types = {
                TaskCategory.DEVELOPMENT: ["FEATURE_BUSINESS_LOGIC", "FEATURE_USER_INTERFACE", "FEATURE_API", "FEATURE_CRUD_OPERATIONS"],
                TaskCategory.TESTING: ["TESTING_UNIT_TESTS", "TESTING_INTEGRATION_TESTS", "TESTING_E2E_TESTS"],
                TaskCategory.DOCUMENTATION: ["TESTING_UNIT_TESTS", "TESTING_INTEGRATION_TESTS", "TESTING_E2E_TESTS"],
                TaskCategory.OPTIMIZATION: ["OPTIMIZATION_PERFORMANCE", "OPTIMIZATION_DATABASE", "OPTIMIZATION_CODE"],
                TaskCategory.SECURITY: ["SECURITY_AUDIT", "SECURITY_AUTHENTICATION", "SECURITY_VULNERABILITY"],
                TaskCategory.MAINTENANCE: ["BUG_FIX_CRITICAL", "BUG_FIX_FUNCTIONAL", "REFACTOR_CODE"],
                TaskCategory.INTEGRATION: ["INTEGRATION_API", "INTEGRATION_THIRD_PARTY", "INTEGRATION_WEBHOOKS"]
            }
            
            # Select specific task type
            task_types = category_to_types.get(task_category, ["FEATURE_BUSINESS_LOGIC"])
            task_type = random.choice(task_types)
            
            # Generate content
            if self.content_generator:
                # Use AI to generate content
                repo_context = {
                    'basic_info': project_data,
                    'technical_stack': project_data.get('tech_stack', {}),
                    'recent_activity': {'recent_commits': []},
                    'issues_analysis': {'recent_issues': [], 'bug_issues': []}
                }
                
                # For now, skip AI generation in simple generator to avoid async issues
                title, description = self._generate_simple_content(project_name, task_category)
            else:
                # Fallback content generation
                title, description = self._generate_simple_content(project_name, task_category)
            
            # Determine priority
            priority = self._determine_priority(task_category)
            
            # Create work item
            work_item = WorkItem(
                id=str(uuid.uuid4()),
                task_type=task_type,
                title=title,
                description=description,
                repository=project_name,
                priority=priority,
                metadata={
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'source': 'simple_work_generator',
                    'category': task_category.value,
                    'estimated_hours': self._estimate_hours(task_category)
                }
            )
            
            return work_item
            
        except Exception as e:
            self.logger.error(f"Error creating work item: {e}")
            return None
    
    def _generate_simple_content(self, project_name: str, task_category: TaskCategory) -> tuple:
        """Generate simple content without AI.
        
        Args:
            project_name: Name of the project
            task_category: Task category
            
        Returns:
            Tuple of (title, description)
        """
        templates = {
            TaskCategory.DEVELOPMENT: (
                f"Add new feature to {project_name}",
                f"Implement a new feature that enhances the functionality of {project_name}. "
                "This should improve user experience and add value to the application."
            ),
            TaskCategory.MAINTENANCE: (
                f"Fix bug in {project_name}",
                f"Investigate and fix a bug that affects the stability of {project_name}. "
                "Ensure the fix is thoroughly tested."
            ),
            TaskCategory.TESTING: (
                f"Add tests for {project_name}",
                f"Improve test coverage for {project_name} by adding unit and integration tests. "
                "Focus on critical functionality."
            ),
            TaskCategory.DOCUMENTATION: (
                f"Update documentation for {project_name}",
                f"Update and improve the documentation for {project_name}. "
                "Ensure all features are properly documented."
            ),
            TaskCategory.OPTIMIZATION: (
                f"Optimize performance in {project_name}",
                f"Identify and optimize performance bottlenecks in {project_name}. "
                "Focus on improving response times and resource usage."
            ),
            TaskCategory.SECURITY: (
                f"Security audit for {project_name}",
                f"Conduct a security audit of {project_name} and fix any vulnerabilities found. "
                "Follow security best practices."
            ),
            TaskCategory.INTEGRATION: (
                f"Add integration for {project_name}",
                f"Implement integration with external services or APIs for {project_name}. "
                "Ensure proper error handling and testing."
            )
        }
        
        return templates.get(task_category, ("Generic task", "Complete this task"))
    
    def _determine_priority(self, task_category: TaskCategory) -> TaskPriority:
        """Determine priority based on task category.
        
        Args:
            task_category: Task category
            
        Returns:
            Task priority
        """
        priority_map = {
            TaskCategory.SECURITY: TaskPriority.CRITICAL,
            TaskCategory.MAINTENANCE: TaskPriority.HIGH,
            TaskCategory.DEVELOPMENT: TaskPriority.MEDIUM,
            TaskCategory.OPTIMIZATION: TaskPriority.MEDIUM,
            TaskCategory.TESTING: TaskPriority.MEDIUM,
            TaskCategory.INTEGRATION: TaskPriority.MEDIUM,
            TaskCategory.DOCUMENTATION: TaskPriority.LOW
        }
        
        return priority_map.get(task_category, TaskPriority.MEDIUM)
    
    def _estimate_hours(self, task_category: TaskCategory) -> int:
        """Estimate hours for task category.
        
        Args:
            task_category: Task category
            
        Returns:
            Estimated hours
        """
        estimates = {
            TaskCategory.DEVELOPMENT: 8,
            TaskCategory.MAINTENANCE: 4,
            TaskCategory.TESTING: 6,
            TaskCategory.DOCUMENTATION: 3,
            TaskCategory.OPTIMIZATION: 6,
            TaskCategory.SECURITY: 8,
            TaskCategory.INTEGRATION: 6
        }
        
        return estimates.get(task_category, 4)