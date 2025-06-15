"""
Intelligent Work Finder for Continuous 24/7 AI Operation

Continuously discovers new work opportunities to ensure the AI system
never runs out of productive tasks to perform.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import uuid
import json

from scripts.work_item_types import WorkItem, TaskPriority, WorkOpportunity
from scripts.ai_brain import IntelligentAIBrain
from scripts.repository_exclusion import RepositoryExclusion
from scripts.task_persistence import TaskPersistence
from scripts.market_research_engine import MarketResearchEngine, ProjectOpportunity
from scripts.portfolio_intelligence import PortfolioIntelligence
from scripts.project_lifecycle_analyzer import ProjectLifecycleAnalyzer
from scripts.task_types import (
    SmartTaskType, TaskTypeSelector, get_task_type_for_string,
    ArchitectureType, TaskCategory
)


class IntelligentWorkFinder:
    """Intelligent work discovery system for continuous operation."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, system_state: Dict[str, Any]):
        """Initialize the work finder.
        
        Args:
            ai_brain: AI brain for intelligent analysis
            system_state: Current system state
        """
        self.ai_brain = ai_brain
        self.system_state = system_state
        self.logger = logging.getLogger(__name__)
        
        # Work discovery history
        self.discovered_work: List[WorkOpportunity] = []
        self.work_sources_checked: Dict[str, datetime] = {}
        self.last_portfolio_analysis: Optional[datetime] = None
        
        # Task persistence for deduplication
        self.task_persistence = TaskPersistence("work_finder_completed_tasks.json")
        
        # Initialize intelligent components
        self.market_research = MarketResearchEngine(ai_brain)
        self.portfolio_intelligence = PortfolioIntelligence(ai_brain)
        self.lifecycle_analyzer = ProjectLifecycleAnalyzer(ai_brain)
        
        # Configuration
        self.discovery_interval = 30  # Check for new work every 30 seconds
        self.portfolio_analysis_interval = 300  # Deep analysis every 5 minutes
        self.max_work_per_discovery = 5
        
        # Track completed tasks per repository
        self.completed_tasks_by_repo: Dict[str, Set[str]] = {}
        
    async def discover_work(self, max_items: int = 5, current_workload: int = 0) -> List[WorkItem]:
        """Discover new work opportunities.
        
        Args:
            max_items: Maximum number of work items to return
            current_workload: Number of currently active workers
            
        Returns:
            List of discovered work items
        """
        self.logger.info(f"Discovering work (max: {max_items}, current load: {current_workload})")
        
        # Check if there are any active projects in the system
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        from scripts.repository_exclusion import is_excluded_repo
        active_projects = {
            name: data for name, data in all_projects.items()
            if not is_excluded_repo(name)
        }
        
        # If no active projects exist, only generate NEW_PROJECT tasks
        if not active_projects:
            self.logger.warning("⚠️ No active projects found. Only generating NEW_PROJECT tasks.")
            new_project_opportunities = []
            
            # Generate dynamic project ideas based on market research
            project_ideas = await self._generate_dynamic_project_ideas()
            
            # Select projects based on max_items
            for i in range(min(max_items, len(project_ideas))):
                project = project_ideas[i]
                new_project_opportunities.append(WorkOpportunity(
                    source="portfolio_gap",
                    type="NEW_PROJECT",
                    priority=TaskPriority.HIGH,
                    title=project['title'],
                    description=project['description'],
                    repository=None,
                    estimated_cycles=5,
                    metadata={
                        'project_type': project['type'],
                        'starter_kit': 'laravel/react-starter-kit',
                        'detailed_description': project['description']
                    }
                ))
            
            # Convert to work items
            work_items = []
            for opp in new_project_opportunities:
                work_item = opp.to_work_item()
                if not self.task_persistence.is_duplicate_task(work_item):
                    work_items.append(work_item)
            
            self.logger.info(f"📋 Generated {len(work_items)} NEW_PROJECT tasks (no existing projects)")
            return work_items
        
        opportunities = []
        
        try:
            # 1. Repository-based work discovery
            repo_work = await self._discover_repository_work()
            opportunities.extend(repo_work)
        except Exception as e:
            self.logger.error(f"Error in repository work discovery: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        try:
            # 2. Portfolio gap analysis
            gap_work = await self._discover_portfolio_gaps()
            opportunities.extend(gap_work)
        except Exception as e:
            self.logger.error(f"Error in portfolio gap discovery: {e}")
        
        # DISABLED: System improvement opportunities
        # try:
        #     # 3. System improvement opportunities
        #     system_work = await self._discover_system_improvements()
        #     opportunities.extend(system_work)
        # except Exception as e:
        #     self.logger.error(f"Error in system improvement discovery: {e}")
        
        try:
            # 4. Research opportunities
            research_work = await self._discover_research_opportunities()
            opportunities.extend(research_work)
        except Exception as e:
            self.logger.error(f"Error in research opportunity discovery: {e}")
        
        try:
            # 5. Maintenance tasks
            maintenance_work = await self._discover_maintenance_tasks()
            opportunities.extend(maintenance_work)
        except Exception as e:
            self.logger.error(f"Error in maintenance task discovery: {e}")
        
        try:
            # 6. Cross-repository integration opportunities
            integration_work = await self._discover_integration_opportunities()
            opportunities.extend(integration_work)
        except Exception as e:
            self.logger.error(f"Error in integration opportunity discovery: {e}")
        
        # Filter and prioritize
        filtered_work = self._filter_and_prioritize(opportunities, max_items)
        
        # Convert to work items and filter duplicates
        work_items = []
        for opp in filtered_work:
            work_item = opp.to_work_item()
            
            # Check for duplicates using task persistence
            if not self.task_persistence.is_duplicate_task(work_item):
                work_items.append(work_item)
            else:
                self.logger.debug(f"🔄 Filtered duplicate work: {work_item.title}")
        
        # If we have no new work due to duplicates, try to generate more diverse work
        if not work_items and len(filtered_work) > 0:
            self.logger.info("🔄 All discovered work was duplicates - generating more diverse opportunities")
            try:
                # Generate some alternative work types
                alternative_work = await self._generate_alternative_work()
                for opp in alternative_work:
                    work_item = opp.to_work_item()
                    if not self.task_persistence.is_duplicate_task(work_item):
                        work_items.append(work_item)
                        if len(work_items) >= max_items:
                            break
            except Exception as e:
                self.logger.warning(f"Error generating alternative work: {e}")
        
        # Log discovery results
        if work_items:
            self.logger.info(f"Discovered {len(work_items)} work items:")
            for item in work_items:
                self.logger.info(f"  - {item.task_type}: {item.title} (priority: {item.priority.name})")
        
        # Track discovered work
        self.discovered_work.extend(filtered_work)
        
        # Log discovery summary
        if work_items:
            task_types = {}
            for item in work_items:
                task_types[item.task_type] = task_types.get(item.task_type, 0) + 1
            
            type_summary = ', '.join(f"{count} {task_type}" for task_type, count in task_types.items())
            self.logger.info(f"📋 Work discovery summary: {type_summary}")
        
        return work_items
    
    async def _discover_repository_work(self) -> List[WorkOpportunity]:
        """Discover work opportunities from repositories using lifecycle analysis."""
        opportunities = []
        
        # Check both 'projects' and 'repositories' keys for backward compatibility
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        filtered_projects = RepositoryExclusion.filter_excluded_repos_dict(all_projects)
        
        # Double-check that repositories still exist (defensive programming)
        # This prevents work generation for repos that might have been deleted
        # after system state was loaded but before work generation
        valid_projects = {}
        for repo_name, repo_data in filtered_projects.items():
            # Basic validation - ensure the repository has essential data
            if repo_data.get('name') or repo_data.get('full_name'):
                valid_projects[repo_name] = repo_data
            else:
                self.logger.warning(f"Skipping invalid repository entry: {repo_name}")
        
        self.logger.debug(f"Repository discovery: {len(all_projects)} total, {len(filtered_projects)} after exclusions, {len(valid_projects)} valid")
        
        for repo_name, repo_data in valid_projects.items():
            # Perform lifecycle analysis
            lifecycle_analysis = await self._analyze_repository_lifecycle(repo_name, repo_data)
            current_stage = lifecycle_analysis.get('current_stage', 'unknown')
            appropriate_task_types = lifecycle_analysis.get('appropriate_task_types', [])
            
            # Get architecture type
            architecture = self._determine_architecture_type(repo_data)
            
            # Get completed tasks for this repo
            completed_tasks = self.completed_tasks_by_repo.get(repo_name, set())
            
            # Get smart task types for this repository
            smart_tasks = TaskTypeSelector.get_appropriate_task_types(
                architecture=architecture,
                lifecycle_stage=current_stage,
                current_needs=repo_data.get('specific_needs', []),
                completed_tasks=completed_tasks
            )
            
            # Generate opportunities based on smart task types
            for task_type in smart_tasks[:3]:  # Limit to top 3 per repo
                opportunity = self._create_smart_opportunity(
                    task_type=task_type,
                    repo_name=repo_name,
                    repo_data=repo_data,
                    lifecycle_stage=current_stage
                )
                if opportunity:
                    opportunities.append(opportunity)
            
            # Fallback to basic analysis if no smart tasks generated
            if not smart_tasks:
                # Check repository health and activity
                health_score = repo_data.get('health_score', 100)
                recent_activity = repo_data.get('recent_activity', {})
            
                # Low health repositories need attention
                if health_score < 85:
                    opportunities.append(WorkOpportunity(
                        source=f"repository:{repo_name}",
                        type=SmartTaskType.REPOSITORY_HEALTH.value,
                        priority=TaskPriority.HIGH,
                        title=f"Improve {repo_name} health score",
                        description=f"Repository health is {health_score}%. Identify and fix issues.",
                        repository=repo_name,
                        estimated_cycles=3,
                        metadata={
                            'current_health': health_score,
                            'lifecycle_stage': current_stage,
                            'architecture': architecture.value if architecture else 'unknown'
                        }
                    ))
            
                # Recent commits suggest various improvement opportunities
                recent_commits = recent_activity.get('recent_commits', 0)
                if recent_commits > 0:
                    # Rotate through different types of improvements
                    improvement_types = [
                        {
                            'type': 'TESTING',
                            'title': f"Add comprehensive tests for {repo_name}",
                            'description': f"Recent {recent_commits} commits need test coverage to ensure reliability.",
                            'priority': TaskPriority.HIGH
                        },
                        {
                            'type': 'DOCUMENTATION',
                            'title': f"Update documentation for {repo_name} changes",
                            'description': f"Document recent features and changes in {repo_name} for better maintainability.",
                            'priority': TaskPriority.MEDIUM
                        },
                        {
                            'type': 'FEATURE',
                            'title': f"Enhance {repo_name} with new functionality",
                            'description': f"Build upon recent changes in {repo_name} to add valuable new features.",
                            'priority': TaskPriority.MEDIUM
                        },
                        {
                            'type': 'PERFORMANCE',
                            'title': f"Optimize {repo_name} performance and structure",
                            'description': f"Review and optimize {repo_name} codebase for better performance and maintainability.",
                            'priority': TaskPriority.LOW
                        }
                    ]
                
                    # Select improvement type based on repository characteristics
                    selected_improvement = improvement_types[0]  # Default to testing
                
                    if recent_commits > 10:  # Very active repo
                        selected_improvement = improvement_types[1]  # Documentation
                    elif 'typescript' in str(repo_data.get('topics', [])).lower():
                        selected_improvement = improvement_types[2]  # Feature
                    elif recent_commits > 5:
                        selected_improvement = improvement_types[3]  # Optimization
                
                    # Check if similar task already exists before creating opportunity
                    from scripts.task_manager import TaskManager, TaskType
                    # Check for duplicate without default repository
                    task_manager = TaskManager(repository=repo_name)
                
                    try:
                        # Check for duplicate in local task state
                        task_type = TaskType(selected_improvement['type'].lower())
                        existing_task = task_manager._find_duplicate_task(
                            task_type,
                            selected_improvement['title'],
                            selected_improvement['description'],
                            repo_name
                        )
                    
                        if not existing_task:
                            opportunities.append(WorkOpportunity(
                                source=f"repository:{repo_name}",
                                type=selected_improvement['type'],
                                priority=selected_improvement['priority'],
                                title=selected_improvement['title'],
                                description=selected_improvement['description'],
                                repository=repo_name,
                                estimated_cycles=2,
                                metadata={'recent_commits': recent_commits}
                            ))
                    except Exception as e:
                        self.logger.debug(f"Could not check for duplicate task: {e}")
        
        return opportunities


    async def _analyze_repository_lifecycle(self, repo_name: str, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository lifecycle stage."""
        try:
            # Prepare repository analysis data
            repo_analysis = {
                'repository': repo_name,
                'basic_info': repo_data,
                'health_metrics': {
                    'health_score': repo_data.get('health_score', 100),
                    'recent_commits': repo_data.get('recent_activity', {}).get('recent_commits', 0)
                },
                'code_analysis': repo_data.get('code_analysis', {}),
                'issues_analysis': repo_data.get('issues_analysis', {}),
                'recent_activity': repo_data.get('recent_activity', {}),
                'architecture': repo_data.get('architecture', {})
            }
            
            # Perform lifecycle analysis
            lifecycle_result = await self.lifecycle_analyzer.analyze_project_stage(repo_analysis)
            return lifecycle_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing lifecycle for {repo_name}: {e}")
            return {
                'current_stage': 'unknown',
                'appropriate_task_types': ['FEATURE', 'TESTING', 'DOCUMENTATION']
            }
    
    def _determine_architecture_type(self, repo_data: Dict[str, Any]) -> Optional[ArchitectureType]:
        """Determine repository architecture type."""
        # All our projects use Laravel React starter kit
        return ArchitectureType.LARAVEL_REACT
    
    def _create_smart_opportunity(self, task_type: SmartTaskType, repo_name: str,
                                repo_data: Dict[str, Any], lifecycle_stage: str) -> Optional[WorkOpportunity]:
        """Create a smart work opportunity based on task type."""
        metadata = TaskTypeSelector.get_task_metadata(task_type)
        
        # Generate appropriate title and description
        title = self._generate_task_title(task_type, repo_name, repo_data)
        description = self._generate_task_description(task_type, repo_name, repo_data, lifecycle_stage)
        
        # Calculate priority based on lifecycle stage
        priority = self._calculate_smart_priority(task_type, lifecycle_stage, metadata)
        
        # Estimate cycles based on repository complexity
        repo_complexity = repo_data.get('complexity_score', 1.0)
        estimated_cycles = TaskTypeSelector.estimate_task_duration(task_type, repo_complexity)
        
        return WorkOpportunity(
            source=f"repository:{repo_name}",
            type=task_type.value,
            priority=priority,
            title=title,
            description=description,
            repository=repo_name,
            estimated_cycles=estimated_cycles,
            metadata={
                'task_category': metadata.category.value,
                'complexity': metadata.complexity.value,
                'lifecycle_stage': lifecycle_stage,
                'required_skills': list(metadata.required_skills),
                'success_criteria': metadata.success_criteria
            }
        )
    
    def _generate_task_title(self, task_type: SmartTaskType, repo_name: str, 
                           repo_data: Dict[str, Any]) -> str:
        """Generate context-aware task title."""
        base_titles = {
            SmartTaskType.SETUP_LARAVEL_API: f"Set up Laravel API structure for {repo_name}",
            SmartTaskType.SETUP_SANCTUM_AUTH: f"Implement Sanctum authentication in {repo_name}",
            SmartTaskType.FEATURE_API_ENDPOINT: f"Add new API endpoint to {repo_name}",
            SmartTaskType.TESTING_UNIT_TESTS: f"Add unit tests for {repo_name}",
            SmartTaskType.DOCUMENTATION_API: f"Document API endpoints in {repo_name}",
            SmartTaskType.OPTIMIZATION_PERFORMANCE: f"Optimize {repo_name} performance",
        }
        
        return base_titles.get(task_type, f"Implement {task_type.value} for {repo_name}")
    
    def _generate_task_description(self, task_type: SmartTaskType, repo_name: str,
                                 repo_data: Dict[str, Any], lifecycle_stage: str) -> str:
        """Generate detailed task description based on context."""
        metadata = TaskTypeSelector.get_task_metadata(task_type)
        
        description = f"Repository {repo_name} is in {lifecycle_stage} stage. "
        
        if task_type == SmartTaskType.SETUP_LARAVEL_API:
            description += "Set up the Laravel API structure with proper routing, controllers, and middleware."
        elif task_type == SmartTaskType.FEATURE_API_ENDPOINT:
            description += "Implement a new API endpoint with proper validation, error handling, and documentation."
        else:
            description += f"Complete {task_type.value} to advance the project."
            
        # Add success criteria
        if metadata.success_criteria:
            criteria_text = ", ".join(f"{k}: {v}" for k, v in metadata.success_criteria.items())
            description += f" Success criteria: {criteria_text}."
            
        return description
    
    def _calculate_smart_priority(self, task_type: SmartTaskType, lifecycle_stage: str,
                                metadata: Any) -> TaskPriority:
        """Calculate priority based on task type and lifecycle stage."""
        # Stage-based priority adjustments
        stage_priority_boost = {
            'inception': {SmartTaskType.SETUP_PROJECT_STRUCTURE, SmartTaskType.SETUP_DATABASE_SCHEMA},
            'early_development': {SmartTaskType.FEATURE_CRUD_OPERATIONS, SmartTaskType.TESTING_UNIT_TESTS},
            'active_development': {SmartTaskType.FEATURE_API_ENDPOINT, SmartTaskType.TESTING_INTEGRATION_TESTS},
            'growth': {SmartTaskType.OPTIMIZATION_PERFORMANCE, SmartTaskType.INFRASTRUCTURE_SCALING},
            'mature': {SmartTaskType.MAINTENANCE_SECURITY_PATCH, SmartTaskType.OPTIMIZATION_CACHING}
        }
        
        # Check if this task type should be prioritized for current stage
        if lifecycle_stage in stage_priority_boost:
            if task_type in stage_priority_boost[lifecycle_stage]:
                return TaskPriority.HIGH
                
        # Use metadata priority modifier
        if metadata.priority_modifier > 1.2:
            return TaskPriority.HIGH
        elif metadata.priority_modifier > 1.0:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW
            
    async def _discover_portfolio_gaps(self) -> List[WorkOpportunity]:
        """Discover gaps in the project portfolio using intelligent analysis."""
        opportunities = []
        
        # Check both 'projects' and 'repositories' keys for backward compatibility
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        filtered_projects = RepositoryExclusion.filter_excluded_repos_dict(all_projects)
        
        try:
            # Perform deep portfolio analysis
            portfolio_insights = await self.portfolio_intelligence.analyze_portfolio(
                filtered_projects,
                force_refresh=False
            )
            
            # Generate market-driven project opportunities
            project_opportunities = await self.market_research.generate_project_opportunities(
                filtered_projects,
                max_opportunities=3
            )
            
            # Convert ProjectOpportunity to WorkOpportunity
            for proj_opp in project_opportunities:
                # Create unique metadata for the opportunity
                metadata = {
                    'problem_statement': proj_opp.problem_statement,
                    'solution_approach': proj_opp.solution_approach,
                    'unique_value_proposition': proj_opp.unique_value_proposition,
                    'target_market': proj_opp.target_market,
                    'tech_stack': proj_opp.tech_stack,
                    'market_demand': proj_opp.market_demand,
                    'innovation_score': proj_opp.innovation_score,
                    'complexity': proj_opp.estimated_complexity
                }
                
                # Determine priority based on market demand and innovation
                if proj_opp.market_demand > 0.8:
                    priority = TaskPriority.HIGH
                elif proj_opp.market_demand > 0.6:
                    priority = TaskPriority.MEDIUM
                else:
                    priority = TaskPriority.LOW
                
                # Estimate cycles based on complexity
                cycles_map = {'low': 3, 'medium': 5, 'high': 8}
                complexity = proj_opp.estimated_complexity if proj_opp.estimated_complexity else 'medium'
                estimated_cycles = cycles_map.get(complexity, 5)
                
                opportunities.append(WorkOpportunity(
                    source="market_research",
                    type="NEW_PROJECT",
                    priority=priority,
                    title=proj_opp.title,
                    description=proj_opp.description,
                    repository=None,
                    estimated_cycles=estimated_cycles,
                    metadata=metadata
                ))
            
            # Add strategic gap-based opportunities
            for gap in portfolio_insights.strategic_gaps[:2]:
                if gap.get('urgency') == 'high':
                    opportunities.append(WorkOpportunity(
                        source="portfolio_intelligence",
                        type="STRATEGIC_INITIATIVE",
                        priority=TaskPriority.HIGH,
                        title=f"Address {gap.get('gap_type', 'strategic')} gap",
                        description=gap.get('description', 'Strategic portfolio gap identified'),
                        repository=None,
                        estimated_cycles=4,
                        metadata={
                            'gap_details': gap,
                            'recommended_action': gap.get('recommended_action', '')
                        }
                    ))
            
            self.logger.info(f"📊 Portfolio analysis complete: {len(opportunities)} opportunities found")
            
        except Exception as e:
            self.logger.error(f"Error in intelligent portfolio analysis: {e}")
            # Fallback to basic analysis if intelligent analysis fails
            return await self._basic_portfolio_gap_analysis(filtered_projects)
        
        return opportunities
    
    async def _basic_portfolio_gap_analysis(self, filtered_projects: Dict[str, Any]) -> List[WorkOpportunity]:
        """Fallback basic portfolio gap analysis."""
        opportunities = []
        
        # Use _generate_dynamic_project_ideas for basic gap filling
        project_ideas = await self._generate_dynamic_project_ideas()
        
        for i, project in enumerate(project_ideas[:2]):
            opportunities.append(WorkOpportunity(
                source="portfolio_gap",
                type="NEW_PROJECT",
                priority=TaskPriority.MEDIUM,
                title=project.get('title', f'Project {i+1}'),
                description=project.get('description', 'New project opportunity'),
                repository=None,
                estimated_cycles=5,
                metadata={
                    'project_type': project.get('type', 'general'),
                    'tech_stack': project.get('tech_stack', [])
                }
            ))
        
        return opportunities
    
    async def _discover_system_improvements(self) -> List[WorkOpportunity]:
        """Discover system improvement opportunities."""
        opportunities = []
        
        # Check if we're in development mode and skip if so
        import os
        is_development = os.getenv('NODE_ENV', 'production').lower() == 'development'
        if is_development:
            self.logger.info("⚠️ Skipping system improvement discovery in development mode")
            return opportunities
        
        # Check system performance metrics
        performance = self.system_state.get('system_performance', {})
        failed_actions = performance.get('failed_actions', 0)
        total_cycles = performance.get('total_cycles', 1)
        
        if failed_actions > 0:
            failure_rate = failed_actions / max(total_cycles, 1)
            if failure_rate > 0.1:  # More than 10% failure rate
                opportunities.append(WorkOpportunity(
                    source="system_monitoring",
                    type="SYSTEM_IMPROVEMENT",
                    priority=TaskPriority.CRITICAL,
                    title="Fix system reliability issues",
                    description=f"System has {failure_rate:.1%} failure rate. Investigate and fix.",
                    estimated_cycles=3,
                    metadata={'failure_rate': failure_rate, 'failed_actions': failed_actions}
                ))
        
        # Check for outdated dependencies or configurations - add variety
        import random
        maintenance_tasks = [
            {
                'title': "Update system dependencies and configurations",
                'description': "Regular maintenance to keep system components up to date.",
                'improvement_type': 'maintenance'
            },
            {
                'title': "Clean up unused code and dead imports",
                'description': "Remove technical debt by cleaning up unused code paths.",
                'improvement_type': 'cleanup'
            },
            {
                'title': "Optimize Docker container sizes",
                'description': "Reduce container bloat and improve deployment efficiency.",
                'improvement_type': 'containerization'
            },
            {
                'title': "Review and update security policies",
                'description': "Ensure security best practices are being followed.",
                'improvement_type': 'security'
            },
            {
                'title': "Implement automated backup strategy",
                'description': "Set up automated backups for critical system data.",
                'improvement_type': 'backup'
            }
        ]
        
        selected_maintenance = random.choice(maintenance_tasks)
        
        opportunities.append(WorkOpportunity(
            source="system_monitoring",
            type="SYSTEM_IMPROVEMENT", 
            priority=TaskPriority.LOW,
            title=selected_maintenance['title'],
            description=selected_maintenance['description'],
            estimated_cycles=2,
            metadata={'improvement_type': selected_maintenance['improvement_type']}
        ))
        
        # Performance optimization
        learning_metrics = performance.get('learning_metrics', {})
        efficiency = learning_metrics.get('resource_efficiency', 1.0)
        
        if efficiency < 0.8:  # Less than 80% efficiency
            # Add variety to system improvement tasks
            import random
            
            improvement_options = [
                {
                    'title': "Optimize system resource efficiency",
                    'description': f"System efficiency is {efficiency:.1%}. Analyze and optimize resource usage patterns.",
                    'improvement_type': 'resource_optimization'
                },
                {
                    'title': "Implement performance monitoring dashboard",
                    'description': f"Current efficiency: {efficiency:.1%}. Build real-time monitoring for better insights.",
                    'improvement_type': 'monitoring'
                },
                {
                    'title': "Refactor task scheduling algorithm",
                    'description': f"Efficiency at {efficiency:.1%}. Improve task distribution and scheduling.",
                    'improvement_type': 'scheduling'
                },
                {
                    'title': "Optimize database query performance",
                    'description': f"System running at {efficiency:.1%} efficiency. Tune database operations.",
                    'improvement_type': 'database'
                },
                {
                    'title': "Implement intelligent caching strategy",
                    'description': f"Boost efficiency from {efficiency:.1%} with smarter caching.",
                    'improvement_type': 'caching'
                }
            ]
            
            selected = random.choice(improvement_options)
            
            opportunities.append(WorkOpportunity(
                source="system_monitoring",
                type="SYSTEM_IMPROVEMENT",
                priority=TaskPriority.HIGH,
                title=selected['title'],
                description=selected['description'],
                estimated_cycles=4,
                metadata={
                    'efficiency': efficiency, 
                    'improvement_type': selected['improvement_type']
                }
            ))
        
        return opportunities
    
    async def _discover_research_opportunities(self) -> List[WorkOpportunity]:
        """Discover research opportunities."""
        opportunities = []
        
        # Technology trend research
        opportunities.append(WorkOpportunity(
            source="research_scheduler",
            type="RESEARCH",
            priority=TaskPriority.BACKGROUND,
            title="Research emerging AI development trends",
            description="Stay current with latest AI development tools and frameworks.",
            estimated_cycles=2,
            metadata={'research_area': 'ai_trends', 'frequency': 'weekly'}
        ))
        
        # Competitive analysis
        opportunities.append(WorkOpportunity(
            source="research_scheduler",
            type="RESEARCH", 
            priority=TaskPriority.LOW,
            title="Analyze competitor project structures",
            description="Research how similar AI development teams structure their projects.",
            estimated_cycles=3,
            metadata={'research_area': 'competitive_analysis', 'frequency': 'monthly'}
        ))
        
        # Best practices research
        opportunities.append(WorkOpportunity(
            source="research_scheduler",
            type="RESEARCH",
            priority=TaskPriority.MEDIUM,
            title="Research software architecture best practices",
            description="Study latest patterns for scalable AI-driven applications.",
            estimated_cycles=2,
            metadata={'research_area': 'architecture', 'frequency': 'bi-weekly'}
        ))
        
        return opportunities
    
    async def _discover_maintenance_tasks(self) -> List[WorkOpportunity]:
        """Discover maintenance tasks."""
        opportunities = []
        
        # Check when last backup was created
        last_backup_check = self.work_sources_checked.get('backup_system', datetime.min.replace(tzinfo=timezone.utc))
        if (datetime.now(timezone.utc) - last_backup_check).seconds > 3600:  # Check hourly
            opportunities.append(WorkOpportunity(
                source="maintenance_scheduler",
                type="MAINTENANCE",
                priority=TaskPriority.LOW,
                title="Verify system backup integrity",
                description="Check that system state backups are being created properly.",
                estimated_cycles=1,
                metadata={'maintenance_type': 'backup_verification'}
            ))
            self.work_sources_checked['backup_system'] = datetime.now(timezone.utc)
        
        # Log file cleanup
        opportunities.append(WorkOpportunity(
            source="maintenance_scheduler",
            type="MAINTENANCE",
            priority=TaskPriority.BACKGROUND,
            title="Clean up old log files",
            description="Archive or remove old log files to free up space.",
            estimated_cycles=1,
            metadata={'maintenance_type': 'log_cleanup'}
        ))
        
        # Performance monitoring
        opportunities.append(WorkOpportunity(
            source="maintenance_scheduler", 
            type="MONITORING",
            priority=TaskPriority.LOW,
            title="Generate performance metrics report",
            description="Analyze system performance and generate insights.",
            estimated_cycles=1,
            metadata={'maintenance_type': 'performance_report'}
        ))
        
        return opportunities
    
    async def _discover_integration_opportunities(self) -> List[WorkOpportunity]:
        """Discover cross-repository integration opportunities."""
        opportunities = []
        
        # Check both 'projects' and 'repositories' keys for backward compatibility
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        all_projects = {**projects, **repositories}
        
        # Filter out excluded repositories
        filtered_projects = RepositoryExclusion.filter_excluded_repos_dict(all_projects)
        project_list = list(filtered_projects.keys())
        
        if len(project_list) >= 2:
            # API integration opportunities
            cms_projects = [name for name, data in filtered_projects.items() 
                          if 'cms' in (data.get('description') or '').lower()]
            ai_projects = [name for name, data in filtered_projects.items()
                         if 'ai' in (data.get('description') or '').lower()]
            
            if cms_projects and ai_projects:
                opportunities.append(WorkOpportunity(
                    source="integration_analysis",
                    type="INTEGRATION",
                    priority=TaskPriority.MEDIUM,
                    title=f"Integrate {cms_projects[0]} with {ai_projects[0]}",
                    description="Create API connections between CMS and AI systems.",
                    estimated_cycles=5,
                    metadata={
                        'integration_type': 'api',
                        'source_repo': cms_projects[0],
                        'target_repo': ai_projects[0]
                    }
                ))
        
        # Shared component library
        if len(project_list) >= 3:
            opportunities.append(WorkOpportunity(
                source="integration_analysis",
                type="NEW_PROJECT",
                priority=TaskPriority.MEDIUM,
                title="Create shared component library",
                description="Build reusable components for all projects.",
                estimated_cycles=4,
                metadata={'integration_type': 'shared_library'}
            ))
        
        return opportunities
    
    async def _generate_alternative_work(self) -> List[WorkOpportunity]:
        """Generate alternative work when all regular work is duplicates."""
        opportunities = []
        
        # Generate time-based maintenance tasks
        current_hour = datetime.now(timezone.utc).hour
        
        if current_hour % 6 == 0:  # Every 6 hours
            opportunities.append(WorkOpportunity(
                source="alternative_scheduler",
                type="MAINTENANCE",
                priority=TaskPriority.LOW,
                title="Perform system health check",
                description="Comprehensive health check of the AI system components.",
                estimated_cycles=1,
                metadata={'maintenance_type': 'health_check', 'scheduled': True}
            ))
        
        if current_hour % 8 == 0:  # Every 8 hours  
            opportunities.append(WorkOpportunity(
                source="alternative_scheduler",
                type="RESEARCH",
                priority=TaskPriority.BACKGROUND,
                title="Research AI development best practices",
                description="Study latest trends in AI development and system optimization.",
                estimated_cycles=2,
                metadata={'research_area': 'best_practices', 'scheduled': True}
            ))
        
        # Generate code quality improvement tasks
        opportunities.append(WorkOpportunity(
            source="alternative_scheduler",
            type="SYSTEM_IMPROVEMENT",
            priority=TaskPriority.MEDIUM,
            title="Analyze code quality metrics",
            description="Review codebase for potential improvements and optimizations.",
            estimated_cycles=3,
            metadata={'improvement_type': 'code_quality', 'scheduled': True}
        ))
        
        # Generate random exploration task
        import random
        exploration_topics = [
            "container orchestration patterns",
            "microservices architecture",
            "database optimization techniques",
            "cloud deployment strategies",
            "API design patterns",
            "security vulnerability scanning",
            "performance monitoring tools",
            "automated testing strategies"
        ]
        
        topic = random.choice(exploration_topics)
        opportunities.append(WorkOpportunity(
            source="alternative_scheduler",
            type="RESEARCH",
            priority=TaskPriority.LOW,
            title=f"Explore {topic}",
            description=f"Research and document findings about {topic} for future implementation.",
            estimated_cycles=2,
            metadata={'research_area': 'exploration', 'topic': topic, 'scheduled': True}
        ))
        
        return opportunities
    
    def _filter_and_prioritize(self, opportunities: List[WorkOpportunity], max_items: int) -> List[WorkOpportunity]:
        """Filter and prioritize work opportunities."""
        # Remove duplicates based on title
        seen_titles = set()
        unique_opportunities = []
        
        for opp in opportunities:
            if opp.title not in seen_titles:
                unique_opportunities.append(opp)
                seen_titles.add(opp.title)
        
        # Sort by priority (lower value = higher priority)
        unique_opportunities.sort(key=lambda x: (x.priority.value, x.estimated_cycles))
        
        # Return top items
        return unique_opportunities[:max_items]
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get work discovery statistics."""
        total_discovered = len(self.discovered_work)
        by_source = {}
        by_type = {}
        by_priority = {}
        
        for work in self.discovered_work:
            # Count by source
            source = work.source.split(':')[0]  # Get base source name
            by_source[source] = by_source.get(source, 0) + 1
            
            # Count by type
            by_type[work.type] = by_type.get(work.type, 0) + 1
            
            # Count by priority
            priority_name = work.priority.name
            by_priority[priority_name] = by_priority.get(priority_name, 0) + 1
        
        return {
            'total_discovered': total_discovered,
            'by_source': by_source,
            'by_type': by_type, 
            'by_priority': by_priority,
            'last_discovery': datetime.now(timezone.utc).isoformat()
        }
    
    async def _generate_dynamic_project_ideas(self) -> List[Dict[str, Any]]:
        """Generate project ideas dynamically based on market research.
        
        Returns:
            List of project ideas based on real market needs
        """
        try:
            # Use AI to research and generate project ideas
            prompt = """
            Generate 3-5 UNIQUE and SPECIFIC project ideas based on real market needs.
            
            Requirements:
            1. Each project must solve a SPECIFIC, REAL-WORLD problem
            2. Must have clear target audience and monetization strategy
            3. Should use modern web technologies (NOT limited to any specific framework)
            4. AVOID generic terms like "platform", "system", "solution"
            5. Project names should be creative and memorable (like "Spotify", "Notion", "Slack")
            6. Description must explain the EXACT problem and HOW it's solved
            
            Examples of GOOD project ideas:
            - "TaskBuddy" - AI that learns your work patterns and automatically schedules deep work blocks
            - "PriceHawk" - Monitors competitor pricing in real-time for e-commerce sellers
            - "CodeReview.ai" - Automated code review bot that learns from your team's style guide
            
            For each project provide:
            - type: Specific category (e.g., "AI productivity tool", "e-commerce analytics", "developer tooling")
            - title: Creative, memorable project name (NOT generic like "Mobile App Platform")
            - description: Clear explanation of the SPECIFIC problem and solution (2-3 sentences)
            - tech_stack: Recommended technologies (can be ANY modern stack, not just Laravel/React)
            
            Return as JSON array.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                ideas_text = response.get('result', '')
                # Parse JSON from response
                import re
                json_match = re.search(r'\[.*\]', ideas_text, re.DOTALL)
                if json_match:
                    ideas = json.loads(json_match.group())
                    return ideas
            
            # Fallback: Return minimal set if AI fails
            return [
                {
                    'type': 'solution',
                    'title': 'Market Research Project',
                    'description': 'A project based on current market research and real needs'
                }
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to generate dynamic project ideas: {e}")
            # Return single fallback project
            return [
                {
                    'type': 'solution',
                    'title': 'Dynamic Solution Project',
                    'description': 'A project that solves real-world problems based on research'
                }
            ]