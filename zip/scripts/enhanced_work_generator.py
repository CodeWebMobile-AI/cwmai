"""Enhanced Work Generator for Continuous Operation

Ensures the system never runs out of work by proactively generating tasks
based on current system state, repository needs, and intelligent analysis.
"""

import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
import uuid

from scripts.work_item_types import WorkItem, TaskPriority, WorkOpportunity
from repository_exclusion import RepositoryExclusion

# Try to import AI content generator
try:
    from scripts.ai_task_content_generator import AITaskContentGenerator
    AI_CONTENT_GENERATOR_AVAILABLE = True
except ImportError:
    AI_CONTENT_GENERATOR_AVAILABLE = False


class EnhancedWorkGenerator:
    """Generates work proactively to maintain queue thresholds."""
    
    def __init__(self, ai_brain=None, system_state=None, logger=None):
        self.ai_brain = ai_brain
        self.system_state = system_state or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize AI content generator if available
        self.ai_content_generator = None
        if AI_CONTENT_GENERATOR_AVAILABLE and self.ai_brain:
            try:
                self.ai_content_generator = AITaskContentGenerator(self.ai_brain)
                self.logger.info("✓ AI content generator initialized for enhanced work generation")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AI content generator: {e}")
        
        # Work generation templates (fallback only)
        self.work_templates = {
            "DOCUMENTATION": [
                "Update {repo} README with latest features",
                "Create API documentation for {repo}",
                "Write installation guide for {repo}",
                "Document configuration options in {repo}",
                "Create troubleshooting guide for {repo}",
                "Write developer guidelines for {repo}",
                "Update changelog for {repo}",
                "Create architecture documentation for {repo}"
            ],
            "TESTING": [
                "Add unit tests for {repo} core modules",
                "Create integration tests for {repo}",
                "Write end-to-end tests for {repo}",
                "Add performance benchmarks for {repo}",
                "Create test fixtures for {repo}",
                "Improve test coverage in {repo}",
                "Add edge case tests for {repo}",
                "Create automated test suite for {repo}"
            ],
            "OPTIMIZATION": [
                "Optimize database queries in {repo}",
                "Improve caching strategy for {repo}",
                "Reduce memory usage in {repo}",
                "Optimize API response times in {repo}",
                "Improve build performance for {repo}",
                "Optimize resource usage in {repo}",
                "Reduce code complexity in {repo}",
                "Improve startup time for {repo}"
            ],
            "FEATURE": [
                "Add user authentication to {repo}",
                "Implement data export functionality in {repo}",
                "Create dashboard for {repo}",
                "Add notification system to {repo}",
                "Implement search functionality in {repo}",
                "Add API versioning to {repo}",
                "Create plugin system for {repo}",
                "Add multi-language support to {repo}"
            ],
            "MAINTENANCE": [
                "Update dependencies in {repo}",
                "Clean up deprecated code in {repo}",
                "Fix linting issues in {repo}",
                "Organize project structure in {repo}",
                "Update CI/CD configuration for {repo}",
                "Review and close stale issues in {repo}",
                "Clean up unused imports in {repo}",
                "Update security policies for {repo}"
            ],
            "RESEARCH": [
                "Research best practices for {topic}",
                "Analyze performance bottlenecks in {repo}",
                "Investigate new technologies for {repo}",
                "Research user needs for {repo}",
                "Analyze competitor solutions to {topic}",
                "Research security vulnerabilities in {repo}",
                "Investigate scalability options for {repo}",
                "Research integration possibilities for {repo}"
            ],
            "NEW_PROJECT": [
                "Create AI-powered chatbot application",
                "Develop mobile task management platform",
                "Build real-time analytics dashboard",
                "Create e-commerce marketplace solution",
                "Develop social networking application",
                "Build educational learning platform",
                "Create health tracking application",
                "Develop financial management system",
                "Build IoT device management platform",
                "Create content management system"
            ]
        }
        
        # Track generated work to ensure variety
        self.generated_work_history: Set[str] = set()
        self.last_generation_time = datetime.now(timezone.utc)
        
    async def generate_work_batch(self, 
                                  target_count: int = 10,
                                  min_priority: TaskPriority = TaskPriority.LOW,
                                  focus_areas: Optional[List[str]] = None) -> List[WorkItem]:
        """Generate a batch of diverse work items.
        
        Args:
            target_count: Number of work items to generate
            min_priority: Minimum priority for generated work
            focus_areas: Optional list of task types to focus on
            
        Returns:
            List of generated work items
        """
        # Validate parameters
        if not isinstance(target_count, int):
            self.logger.error(f"target_count must be an integer, got {type(target_count)}")
            target_count = 10
        
        if not isinstance(min_priority, TaskPriority):
            self.logger.error(f"min_priority must be TaskPriority enum, got {type(min_priority)}")
            min_priority = TaskPriority.LOW
            
        if focus_areas is not None and not isinstance(focus_areas, list):
            self.logger.error(f"focus_areas must be a list or None, got {type(focus_areas)}")
            focus_areas = None
        
        work_items = []
        
        # Get available repositories (check both 'projects' and 'repositories')
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_repos = list({**projects, **repositories}.keys())
        repos = RepositoryExclusion.filter_excluded_repos(all_repos)
        
        # Check if there are any active projects
        from scripts.repository_exclusion import is_excluded_repo
        active_projects = {
            name: data for name, data in {**projects, **repositories}.items()
            if not is_excluded_repo(name)
        }
        
        if not active_projects:
            # No active projects - ONLY generate NEW_PROJECT tasks
            self.logger.warning("⚠️ No active projects exist - ONLY generating NEW_PROJECT tasks")
            repos = [None]
            task_types = ["NEW_PROJECT"]  # Force only NEW_PROJECT
        else:
            # Determine task types to generate
            if focus_areas:
                task_types = focus_areas
            else:
                task_types = list(self.work_templates.keys())
            
        # Check if we're in development mode and remove SYSTEM_IMPROVEMENT
        is_development = os.getenv('NODE_ENV', 'production').lower() == 'development'
        if is_development and 'SYSTEM_IMPROVEMENT' in task_types:
            self.logger.info("⚠️ Removing SYSTEM_IMPROVEMENT from task types in development mode")
            task_types = [t for t in task_types if t != 'SYSTEM_IMPROVEMENT']
        
        # Generate diverse work
        attempts = 0
        max_attempts = target_count * 3  # Allow retries for uniqueness
        
        while len(work_items) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Select random task type and repository
            task_type = random.choice(task_types)
            repo = random.choice(repos) if repos else None
            
            # Generate work item
            work_item = await self._generate_single_work_item(task_type, repo, min_priority)
            
            if work_item and work_item.title not in self.generated_work_history:
                work_items.append(work_item)
                self.generated_work_history.add(work_item.title)
        
        # Clean up old history (keep last 100 items)
        if len(self.generated_work_history) > 100:
            self.generated_work_history = set(list(self.generated_work_history)[-100:])
        
        self.logger.info(f"Generated {len(work_items)} work items")
        return work_items
    
    async def _generate_single_work_item(self,
                                         task_type: str,
                                         repository: Optional[str],
                                         min_priority: TaskPriority) -> Optional[WorkItem]:
        """Generate a single work item, preferring AI content over templates."""
        try:
            # Try AI content generation first
            if self.ai_content_generator and repository:
                try:
                    # Get repository context
                    repo_data = self.system_state.get('projects', {}).get(repository, {})
                    repo_context = {
                        'basic_info': {
                            'description': repo_data.get('description', ''),
                            'language': repo_data.get('language', 'Unknown'),
                            'open_issues_count': repo_data.get('metrics', {}).get('issues_open', 0)
                        },
                        'technical_stack': repo_data.get('topics', []),
                        'recent_activity': repo_data.get('recent_activity', {}),
                        'issues_analysis': {},
                        'performance_metrics': {},
                        'security_analysis': {},
                        'test_coverage': {},
                        'documentation_status': {}
                    }
                    
                    # Generate AI content based on task type
                    result = None
                    if task_type == "FEATURE":
                        result = await self.ai_content_generator.generate_feature_content(repository, repo_context)
                    elif task_type == "DOCUMENTATION":
                        result = await self.ai_content_generator.generate_documentation_content(repository, repo_context)
                    elif task_type == "TESTING":
                        result = await self.ai_content_generator.generate_testing_content(repository, repo_context)
                    elif task_type == "SECURITY":
                        result = await self.ai_content_generator.generate_security_content(repository, repo_context)
                    elif task_type == "OPTIMIZATION":
                        result = await self.ai_content_generator.generate_optimization_content(repository, repo_context)
                    elif task_type == "BUG_FIX":
                        result = await self.ai_content_generator.generate_bug_fix_content(repository, repo_context)
                    elif task_type == "SYSTEM_IMPROVEMENT":
                        # For system improvements, use optimization content generator as it's similar in nature
                        result = await self.ai_content_generator.generate_optimization_content(repository, repo_context)
                    else:
                        # Fall back to template for unsupported types
                        self.logger.debug(f"AI generation not available for {task_type}, falling back to template")
                        raise ValueError(f"AI generation not implemented for {task_type}")
                    
                    # Validate result
                    if not result or not isinstance(result, (tuple, list)):
                        raise ValueError(f"Invalid result from AI content generator: {type(result)}")
                    if len(result) < 2:
                        raise ValueError(f"Invalid result from AI content generator: insufficient items {len(result)}")
                    
                    title, description = result[0], result[1]
                    
                    # Adjust title for system improvements
                    if task_type == "SYSTEM_IMPROVEMENT" and not title.lower().startswith("improve") and not title.lower().startswith("enhance"):
                        title = f"Improve {title}"
                    
                    # Determine priority
                    if task_type in ["BUG_FIX", "SECURITY"]:
                        priority = TaskPriority.HIGH
                    elif task_type in ["FEATURE", "OPTIMIZATION"]:
                        priority = TaskPriority.MEDIUM
                    else:
                        priority = TaskPriority.LOW
                    
                    # Ensure minimum priority
                    if priority.value > min_priority.value:
                        priority = min_priority
                    
                    # Create work item with AI-generated content
                    metadata = {
                        'generated': True,
                        'generation_time': datetime.now(timezone.utc).isoformat(),
                        'generator': 'enhanced_work_generator_ai',
                        'ai_generated': True
                    }
                    
                    # For NEW_PROJECT tasks, ensure we'll have proper metadata
                    # The project creator will generate venture analysis and architecture
                    if task_type == "NEW_PROJECT":
                        metadata['needs_venture_analysis'] = True
                        metadata['needs_architecture'] = True
                        self.logger.info(f"📊 NEW_PROJECT task will generate venture analysis and architecture during creation")
                    
                    work_item = WorkItem(
                        id=f"gen_{uuid.uuid4().hex[:8]}",
                        task_type=task_type,
                        title=title,
                        description=description,
                        priority=priority,
                        repository=repository,
                        estimated_cycles=random.randint(2, 5),
                        metadata=metadata
                    )
                    
                    self.logger.debug(f"Generated AI-powered work item: {title}")
                    return work_item
                    
                except Exception as e:
                    self.logger.debug(f"AI generation failed, falling back to template: {e}")
                    # Fall through to template generation
            
            # Template-based generation (fallback)
            templates = self.work_templates.get(task_type, [])
            if not templates:
                return None
            
            # Select template
            template = random.choice(templates)
            
            # Format title
            if task_type == "NEW_PROJECT":
                # NEW_PROJECT tasks don't need repository formatting
                title = template
            elif repository:
                title = template.format(repo=repository, topic=f"{repository} features")
            else:
                # System-level task
                title = template.replace("{repo}", "the system").replace("{topic}", "system architecture")
            
            # Determine priority
            if task_type in ["BUG_FIX", "SECURITY", "NEW_PROJECT"]:
                priority = TaskPriority.HIGH
            elif task_type in ["FEATURE", "OPTIMIZATION"]:
                priority = TaskPriority.MEDIUM
            else:
                priority = TaskPriority.LOW
            
            # Ensure minimum priority
            if priority.value > min_priority.value:
                priority = min_priority
            
            # Create work item
            metadata = {
                'generated': True,
                'generation_time': datetime.now(timezone.utc).isoformat(),
                'generator': 'enhanced_work_generator'
            }
            
            # For NEW_PROJECT tasks, ensure we'll have proper metadata
            if task_type == "NEW_PROJECT":
                metadata['needs_venture_analysis'] = True
                metadata['needs_architecture'] = True
                self.logger.info(f"📊 NEW_PROJECT task (template) will generate venture analysis and architecture during creation")
            
            work_item = WorkItem(
                id=f"gen_{uuid.uuid4().hex[:8]}",
                task_type=task_type,
                title=title,
                description=f"Auto-generated task: {title}",
                priority=priority,
                repository=repository,
                estimated_cycles=random.randint(1, 5),
                metadata=metadata
            )
            
            return work_item
            
        except Exception as e:
            self.logger.error(f"Error generating work item: {e}")
            return None
    
    async def generate_emergency_work(self, count: int = 5) -> List[WorkItem]:
        """Generate high-priority work when queue is critically low."""
        self.logger.warning(f"Generating {count} emergency work items")
        
        # Focus on high-impact areas (excluding SYSTEM_IMPROVEMENT)
        focus_areas = ["OPTIMIZATION", "BUG_FIX", "SECURITY", "FEATURE"]
        
        return await self.generate_work_batch(
            target_count=count,
            min_priority=TaskPriority.HIGH,
            focus_areas=focus_areas
        )
    
    async def generate_maintenance_work(self, count: int = 3) -> List[WorkItem]:
        """Generate routine maintenance work."""
        focus_areas = ["MAINTENANCE", "DOCUMENTATION", "TESTING"]
        
        return await self.generate_work_batch(
            target_count=count,
            min_priority=TaskPriority.LOW,
            focus_areas=focus_areas
        )
    
    async def generate_research_work(self, count: int = 2) -> List[WorkItem]:
        """Generate research and investigation work."""
        focus_areas = ["RESEARCH", "DOCUMENTATION"]
        
        return await self.generate_work_batch(
            target_count=count,
            min_priority=TaskPriority.MEDIUM,
            focus_areas=focus_areas
        )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about work generation."""
        return {
            'total_generated': len(self.generated_work_history),
            'last_generation': self.last_generation_time.isoformat(),
            'unique_titles': len(self.generated_work_history),
            'available_templates': sum(len(templates) for templates in self.work_templates.values())
        }