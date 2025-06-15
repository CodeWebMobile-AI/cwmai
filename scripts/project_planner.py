"""
Project Planner

Creates intelligent project roadmaps and milestone-based task sequences
based on project type, stage, and goals.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import uuid

# Import dependencies
try:
    from project_lifecycle_analyzer import ProjectStage
    LIFECYCLE_AVAILABLE = True
except ImportError:
    LIFECYCLE_AVAILABLE = False
    # Define fallback
    class ProjectStage(Enum):
        INCEPTION = "inception"
        EARLY_DEVELOPMENT = "early_development"
        ACTIVE_DEVELOPMENT = "active_development"
        GROWTH = "growth"
        MATURE = "mature"
        MAINTENANCE = "maintenance"


@dataclass
class Milestone:
    """Represents a project milestone."""
    id: str
    name: str
    description: str
    target_date: Optional[datetime]
    dependencies: List[str]  # IDs of prerequisite milestones
    tasks: List[str]  # Task IDs
    success_criteria: Dict[str, Any]
    stage: ProjectStage
    priority: str  # critical, high, medium, low
    status: str = "planned"  # planned, in_progress, completed, blocked


@dataclass
class ProjectRoadmap:
    """Complete project roadmap with milestones and phases."""
    project_name: str
    project_type: str
    current_stage: ProjectStage
    target_stage: ProjectStage
    milestones: List[Milestone]
    phases: List[Dict[str, Any]]
    estimated_duration_days: int
    key_risks: List[str]
    success_metrics: Dict[str, Any]
    generated_at: datetime


@dataclass
class TaskSequence:
    """Sequence of tasks for achieving a specific goal."""
    sequence_id: str
    goal: str
    tasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]  # task_id -> [dependency_ids]
    parallel_groups: List[List[str]]  # Groups of tasks that can run in parallel
    estimated_duration_days: int
    critical_path: List[str]  # Task IDs on critical path


class ProjectPlanner:
    """Plans projects intelligently based on type, stage, and goals."""
    
    def __init__(self, ai_brain=None, lifecycle_analyzer=None):
        """Initialize project planner.
        
        Args:
            ai_brain: AI brain for intelligent planning
            lifecycle_analyzer: Project lifecycle analyzer
        """
        self.ai_brain = ai_brain
        self.lifecycle_analyzer = lifecycle_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Project templates by type
        self.project_templates = self._load_project_templates()
        
        # Milestone patterns
        self.milestone_patterns = self._load_milestone_patterns()
        
    def _load_project_templates(self) -> Dict[str, Any]:
        """Load project templates for different types."""
        return {
            "web_application": {
                "phases": [
                    {"name": "Foundation", "duration_days": 14, "focus": "setup"},
                    {"name": "Core Development", "duration_days": 30, "focus": "features"},
                    {"name": "Enhancement", "duration_days": 21, "focus": "ux"},
                    {"name": "Optimization", "duration_days": 14, "focus": "performance"},
                    {"name": "Launch Preparation", "duration_days": 7, "focus": "deployment"}
                ],
                "key_milestones": [
                    "Development Environment Ready",
                    "Authentication System Complete",
                    "Core Features Implemented",
                    "Testing Suite Complete",
                    "Production Deployment"
                ]
            },
            "api_service": {
                "phases": [
                    {"name": "Design", "duration_days": 7, "focus": "architecture"},
                    {"name": "Implementation", "duration_days": 21, "focus": "endpoints"},
                    {"name": "Integration", "duration_days": 14, "focus": "clients"},
                    {"name": "Hardening", "duration_days": 14, "focus": "security"}
                ],
                "key_milestones": [
                    "API Design Complete",
                    "Core Endpoints Working",
                    "Authentication Implemented",
                    "Documentation Complete",
                    "Load Testing Passed"
                ]
            },
            "mobile_backend": {
                "phases": [
                    {"name": "Infrastructure", "duration_days": 7, "focus": "setup"},
                    {"name": "API Development", "duration_days": 21, "focus": "endpoints"},
                    {"name": "Real-time Features", "duration_days": 14, "focus": "websockets"},
                    {"name": "Scaling", "duration_days": 14, "focus": "performance"}
                ],
                "key_milestones": [
                    "Backend Infrastructure Ready",
                    "User Management Complete",
                    "Push Notifications Working",
                    "Real-time Sync Implemented",
                    "Scalability Verified"
                ]
            }
        }
    
    def _load_milestone_patterns(self) -> Dict[ProjectStage, List[str]]:
        """Load milestone patterns for each project stage."""
        return {
            ProjectStage.INCEPTION: [
                "Repository Setup Complete",
                "Development Environment Ready",
                "Architecture Defined",
                "First Commit"
            ],
            ProjectStage.EARLY_DEVELOPMENT: [
                "Core Models Implemented",
                "Basic UI Complete",
                "Authentication Working",
                "First Feature Complete"
            ],
            ProjectStage.ACTIVE_DEVELOPMENT: [
                "MVP Feature Set Complete",
                "Test Coverage 60%",
                "CI/CD Pipeline Active",
                "Beta Release Ready"
            ],
            ProjectStage.GROWTH: [
                "Performance Optimized",
                "Monitoring Implemented",
                "Security Hardened",
                "Scale Testing Complete"
            ],
            ProjectStage.MATURE: [
                "API Stable",
                "Documentation Complete",
                "Full Test Coverage",
                "Production Stable"
            ]
        }
    
    async def create_project_roadmap(self, project_analysis: Dict[str, Any]) -> ProjectRoadmap:
        """Create a comprehensive project roadmap.
        
        Args:
            project_analysis: Repository analysis including lifecycle stage
            
        Returns:
            Complete project roadmap
        """
        self.logger.info(f"Creating roadmap for {project_analysis.get('repository', 'project')}")
        
        # Extract project info
        project_name = project_analysis.get('repository', 'Unknown Project')
        project_type = self._determine_project_type(project_analysis)
        
        # Get lifecycle info
        lifecycle = project_analysis.get('lifecycle_analysis', {})
        current_stage = self._parse_stage(lifecycle.get('current_stage', 'inception'))
        target_stage = self._determine_target_stage(current_stage, project_analysis)
        
        # Generate milestones
        milestones = await self._generate_milestones(
            project_type, current_stage, target_stage, project_analysis
        )
        
        # Create phases
        phases = self._create_project_phases(project_type, current_stage, target_stage)
        
        # Calculate timeline
        estimated_duration = self._calculate_project_duration(milestones, phases)
        
        # Identify risks
        key_risks = self._identify_project_risks(project_analysis, current_stage)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(project_type, target_stage)
        
        return ProjectRoadmap(
            project_name=project_name,
            project_type=project_type,
            current_stage=current_stage,
            target_stage=target_stage,
            milestones=milestones,
            phases=phases,
            estimated_duration_days=estimated_duration,
            key_risks=key_risks,
            success_metrics=success_metrics,
            generated_at=datetime.now(timezone.utc)
        )
    
    async def create_task_sequence(self, goal: str, context: Dict[str, Any]) -> TaskSequence:
        """Create a task sequence for achieving a specific goal.
        
        Args:
            goal: The goal to achieve
            context: Context including project analysis
            
        Returns:
            Task sequence with dependencies
        """
        self.logger.info(f"Creating task sequence for goal: {goal}")
        
        # Generate tasks for the goal
        tasks = await self._generate_goal_tasks(goal, context)
        
        # Determine dependencies
        dependencies = self._analyze_task_dependencies(tasks)
        
        # Identify parallel execution opportunities
        parallel_groups = self._identify_parallel_groups(tasks, dependencies)
        
        # Calculate critical path
        critical_path = self._calculate_critical_path(tasks, dependencies)
        
        # Estimate duration
        duration = self._estimate_sequence_duration(tasks, dependencies, parallel_groups)
        
        return TaskSequence(
            sequence_id=f"seq_{uuid.uuid4().hex[:8]}",
            goal=goal,
            tasks=tasks,
            dependencies=dependencies,
            parallel_groups=parallel_groups,
            estimated_duration_days=duration,
            critical_path=critical_path
        )
    
    def _determine_project_type(self, analysis: Dict[str, Any]) -> str:
        """Determine project type from analysis."""
        # Check language and tech stack
        language = analysis.get('basic_info', {}).get('language', '').lower()
        tech_stack = analysis.get('technical_stack', {})
        topics = analysis.get('basic_info', {}).get('topics', [])
        
        # Check for specific indicators
        if 'api' in str(topics).lower() or 'rest' in str(topics).lower():
            return "api_service"
        elif 'mobile' in str(topics).lower() or 'ios' in str(topics).lower() or 'android' in str(topics).lower():
            return "mobile_backend"
        elif 'laravel' in str(tech_stack).lower() or 'react' in str(tech_stack).lower():
            return "web_application"
        elif 'dashboard' in str(topics).lower():
            return "web_application"
        else:
            return "web_application"  # Default
    
    def _parse_stage(self, stage_str: str) -> ProjectStage:
        """Parse stage string to enum."""
        try:
            return ProjectStage(stage_str)
        except:
            return ProjectStage.INCEPTION
    
    def _determine_target_stage(self, current_stage: ProjectStage, 
                              analysis: Dict[str, Any]) -> ProjectStage:
        """Determine appropriate target stage."""
        # Typical progression
        stage_progression = {
            ProjectStage.INCEPTION: ProjectStage.EARLY_DEVELOPMENT,
            ProjectStage.EARLY_DEVELOPMENT: ProjectStage.ACTIVE_DEVELOPMENT,
            ProjectStage.ACTIVE_DEVELOPMENT: ProjectStage.GROWTH,
            ProjectStage.GROWTH: ProjectStage.MATURE,
            ProjectStage.MATURE: ProjectStage.MATURE,
            ProjectStage.MAINTENANCE: ProjectStage.MAINTENANCE
        }
        
        # Default to next stage
        target = stage_progression.get(current_stage, ProjectStage.ACTIVE_DEVELOPMENT)
        
        # Adjust based on project needs
        if analysis.get('issues_analysis', {}).get('bug_count', 0) > 10:
            # Too many bugs, don't advance too far
            if target == ProjectStage.GROWTH:
                target = ProjectStage.ACTIVE_DEVELOPMENT
        
        return target
    
    async def _generate_milestones(self, project_type: str, current_stage: ProjectStage,
                                 target_stage: ProjectStage, 
                                 analysis: Dict[str, Any]) -> List[Milestone]:
        """Generate project milestones."""
        milestones = []
        
        # Get template milestones
        template = self.project_templates.get(project_type, {})
        template_milestones = template.get('key_milestones', [])
        
        # Get stage-specific milestones
        current_patterns = self.milestone_patterns.get(current_stage, [])
        target_patterns = self.milestone_patterns.get(target_stage, [])
        
        # Combine and deduplicate
        all_milestone_names = list(set(template_milestones + current_patterns + target_patterns))
        
        # Generate milestone objects
        for i, name in enumerate(all_milestone_names):
            milestone = await self._create_milestone(
                name, i, project_type, current_stage, analysis
            )
            milestones.append(milestone)
        
        # Sort by priority and dependencies
        milestones = self._sort_milestones(milestones)
        
        return milestones
    
    async def _create_milestone(self, name: str, index: int, project_type: str,
                              stage: ProjectStage, analysis: Dict[str, Any]) -> Milestone:
        """Create a single milestone."""
        milestone_id = f"m_{uuid.uuid4().hex[:8]}"
        
        # Generate detailed description
        description = await self._generate_milestone_description(name, project_type, analysis)
        
        # Estimate target date (rough estimate)
        days_offset = (index + 1) * 14  # Every 2 weeks
        target_date = datetime.now(timezone.utc) + timedelta(days=days_offset)
        
        # Determine dependencies
        dependencies = self._determine_milestone_dependencies(name, index)
        
        # Define success criteria
        success_criteria = self._define_milestone_criteria(name, project_type)
        
        # Set priority
        priority = self._determine_milestone_priority(name, stage)
        
        return Milestone(
            id=milestone_id,
            name=name,
            description=description,
            target_date=target_date,
            dependencies=dependencies,
            tasks=[],  # Will be populated later
            success_criteria=success_criteria,
            stage=stage,
            priority=priority
        )
    
    async def _generate_milestone_description(self, name: str, project_type: str,
                                            analysis: Dict[str, Any]) -> str:
        """Generate detailed milestone description."""
        if not self.ai_brain:
            return f"Complete {name} for {project_type} project"
        
        prompt = f"""
        Generate a detailed description for this project milestone:
        
        Milestone: {name}
        Project Type: {project_type}
        Current Issues: {analysis.get('issues_analysis', {}).get('total_open', 0)}
        
        Provide a 2-3 sentence description of what this milestone entails
        and why it's important for the project.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return response.get('content', f"Complete {name}")
    
    def _determine_milestone_dependencies(self, name: str, index: int) -> List[str]:
        """Determine milestone dependencies."""
        # Simple dependency logic - could be enhanced
        dependencies = []
        
        # Setup milestones have no dependencies
        if any(word in name.lower() for word in ['setup', 'repository', 'environment']):
            return []
        
        # Later milestones depend on earlier ones (simplified)
        if index > 0:
            dependencies.append(f"m_{index-1}")  # Depends on previous milestone
        
        return dependencies
    
    def _define_milestone_criteria(self, name: str, project_type: str) -> Dict[str, Any]:
        """Define success criteria for milestone."""
        criteria = {
            "completed": False,
            "deliverables": []
        }
        
        # Add specific criteria based on milestone name
        if "test" in name.lower():
            criteria["test_coverage"] = 0.6
            criteria["tests_passing"] = True
        elif "deployment" in name.lower():
            criteria["deployed"] = True
            criteria["health_check_passing"] = True
        elif "documentation" in name.lower():
            criteria["docs_complete"] = True
            criteria["api_docs_generated"] = True
        
        return criteria
    
    def _determine_milestone_priority(self, name: str, stage: ProjectStage) -> str:
        """Determine milestone priority."""
        # Critical milestones
        if any(word in name.lower() for word in ['setup', 'security', 'authentication']):
            return "critical"
        
        # High priority for early stages
        if stage in [ProjectStage.INCEPTION, ProjectStage.EARLY_DEVELOPMENT]:
            if any(word in name.lower() for word in ['core', 'basic', 'environment']):
                return "high"
        
        # Medium priority for most features
        if "feature" in name.lower():
            return "medium"
        
        return "medium"
    
    def _sort_milestones(self, milestones: List[Milestone]) -> List[Milestone]:
        """Sort milestones by dependencies and priority."""
        # Simple topological sort - could be enhanced
        sorted_milestones = []
        remaining = milestones.copy()
        
        while remaining:
            # Find milestones with no pending dependencies
            ready = [m for m in remaining if not any(
                dep in [r.id for r in remaining] for dep in m.dependencies
            )]
            
            if not ready:
                # Circular dependency or error - just add remaining
                sorted_milestones.extend(remaining)
                break
            
            # Sort ready milestones by priority
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            ready.sort(key=lambda m: priority_order.get(m.priority, 2))
            
            # Add first ready milestone
            milestone = ready[0]
            sorted_milestones.append(milestone)
            remaining.remove(milestone)
        
        return sorted_milestones
    
    def _create_project_phases(self, project_type: str, current_stage: ProjectStage,
                             target_stage: ProjectStage) -> List[Dict[str, Any]]:
        """Create project phases."""
        template = self.project_templates.get(project_type, {})
        phases = template.get('phases', [])
        
        # Filter phases based on current and target stage
        if current_stage == ProjectStage.INCEPTION:
            return phases  # All phases
        elif current_stage == ProjectStage.EARLY_DEVELOPMENT:
            return phases[1:]  # Skip foundation
        elif current_stage == ProjectStage.ACTIVE_DEVELOPMENT:
            return phases[2:]  # Skip foundation and core dev
        else:
            # For later stages, focus on optimization and maintenance
            return [
                {"name": "Optimization", "duration_days": 21, "focus": "performance"},
                {"name": "Maintenance", "duration_days": 30, "focus": "stability"}
            ]
    
    def _calculate_project_duration(self, milestones: List[Milestone],
                                  phases: List[Dict[str, Any]]) -> int:
        """Calculate total project duration."""
        # Use phase durations as base
        phase_duration = sum(p.get('duration_days', 0) for p in phases)
        
        # Or use milestone dates
        if milestones:
            last_milestone = max(milestones, key=lambda m: m.target_date or datetime.now(timezone.utc))
            milestone_duration = (last_milestone.target_date - datetime.now(timezone.utc)).days
            return max(phase_duration, milestone_duration)
        
        return phase_duration
    
    def _identify_project_risks(self, analysis: Dict[str, Any],
                              current_stage: ProjectStage) -> List[str]:
        """Identify key project risks."""
        risks = []
        
        # Stage-specific risks
        if current_stage == ProjectStage.INCEPTION:
            risks.append("Unclear requirements may delay development")
        elif current_stage == ProjectStage.EARLY_DEVELOPMENT:
            risks.append("Technical debt accumulation without proper architecture")
        
        # Analysis-based risks
        if analysis.get('health_metrics', {}).get('days_since_update', 0) > 30:
            risks.append("Low activity may indicate stalled development")
        
        if analysis.get('issues_analysis', {}).get('bug_count', 0) > 10:
            risks.append("High bug count may impact new feature development")
        
        if not analysis.get('code_analysis', {}).get('documentation'):
            risks.append("Lack of documentation may slow onboarding and maintenance")
        
        return risks[:5]  # Top 5 risks
    
    def _define_success_metrics(self, project_type: str,
                              target_stage: ProjectStage) -> Dict[str, Any]:
        """Define success metrics for the project."""
        metrics = {
            "stage_advancement": f"Reach {target_stage.value} stage",
            "timeline_adherence": "Complete within estimated duration"
        }
        
        # Type-specific metrics
        if project_type == "web_application":
            metrics["user_satisfaction"] = "Achieve 4+ star rating"
            metrics["performance"] = "Page load under 3 seconds"
        elif project_type == "api_service":
            metrics["reliability"] = "99.9% uptime"
            metrics["response_time"] = "Average response under 200ms"
        
        # Stage-specific metrics
        if target_stage == ProjectStage.MATURE:
            metrics["test_coverage"] = "Achieve 80% test coverage"
            metrics["documentation"] = "Complete API documentation"
        
        return metrics
    
    async def _generate_goal_tasks(self, goal: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks for achieving a specific goal."""
        if not self.ai_brain:
            return self._generate_default_tasks(goal)
        
        prompt = f"""
        Generate a sequence of tasks to achieve this goal:
        
        Goal: {goal}
        Project Type: {context.get('project_type', 'web application')}
        Current Stage: {context.get('current_stage', 'unknown')}
        
        Generate 5-10 specific tasks that would achieve this goal.
        Each task should have:
        - id: Unique identifier
        - title: Clear task title
        - description: What needs to be done
        - estimated_hours: Realistic estimate
        - dependencies: List of task IDs this depends on
        - can_parallel: Whether this can run in parallel with others
        
        Format as JSON array.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        tasks = self._parse_json_array_response(response)
        
        # Ensure all tasks have required fields
        for i, task in enumerate(tasks):
            task['id'] = task.get('id', f"task_{i}")
            task['estimated_hours'] = task.get('estimated_hours', 8)
            task['dependencies'] = task.get('dependencies', [])
            task['can_parallel'] = task.get('can_parallel', False)
        
        return tasks
    
    def _generate_default_tasks(self, goal: str) -> List[Dict[str, Any]]:
        """Generate default tasks for a goal."""
        # Simple default task generation
        return [
            {
                "id": "task_1",
                "title": f"Plan approach for {goal}",
                "description": "Research and plan implementation approach",
                "estimated_hours": 4,
                "dependencies": [],
                "can_parallel": False
            },
            {
                "id": "task_2",
                "title": f"Implement {goal}",
                "description": "Core implementation work",
                "estimated_hours": 16,
                "dependencies": ["task_1"],
                "can_parallel": False
            },
            {
                "id": "task_3",
                "title": f"Test {goal}",
                "description": "Write and run tests",
                "estimated_hours": 8,
                "dependencies": ["task_2"],
                "can_parallel": True
            },
            {
                "id": "task_4",
                "title": f"Document {goal}",
                "description": "Create documentation",
                "estimated_hours": 4,
                "dependencies": ["task_2"],
                "can_parallel": True
            }
        ]
    
    def _analyze_task_dependencies(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze and validate task dependencies."""
        dependencies = {}
        
        for task in tasks:
            task_id = task['id']
            task_deps = task.get('dependencies', [])
            
            # Validate dependencies exist
            valid_deps = [dep for dep in task_deps if any(t['id'] == dep for t in tasks)]
            dependencies[task_id] = valid_deps
        
        return dependencies
    
    def _identify_parallel_groups(self, tasks: List[Dict[str, Any]],
                                dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Identify groups of tasks that can run in parallel."""
        parallel_groups = []
        processed = set()
        
        for task in tasks:
            if task['id'] in processed:
                continue
            
            # Find tasks that can run in parallel with this one
            group = [task['id']]
            
            if task.get('can_parallel', False):
                for other in tasks:
                    if (other['id'] != task['id'] and 
                        other['id'] not in processed and
                        other.get('can_parallel', False) and
                        dependencies.get(other['id'], []) == dependencies.get(task['id'], [])):
                        group.append(other['id'])
                        processed.add(other['id'])
            
            processed.add(task['id'])
            if len(group) > 1:
                parallel_groups.append(group)
        
        return parallel_groups
    
    def _calculate_critical_path(self, tasks: List[Dict[str, Any]],
                               dependencies: Dict[str, List[str]]) -> List[str]:
        """Calculate critical path through tasks."""
        # Simple critical path - find longest dependency chain
        task_map = {t['id']: t for t in tasks}
        
        def get_path_length(task_id: str, memo=None) -> Tuple[int, List[str]]:
            if memo is None:
                memo = {}
            
            if task_id in memo:
                return memo[task_id]
            
            task = task_map.get(task_id)
            if not task:
                return 0, []
            
            task_hours = task.get('estimated_hours', 0)
            
            if not dependencies.get(task_id):
                memo[task_id] = (task_hours, [task_id])
                return task_hours, [task_id]
            
            max_length = 0
            max_path = []
            
            for dep in dependencies[task_id]:
                dep_length, dep_path = get_path_length(dep, memo)
                if dep_length > max_length:
                    max_length = dep_length
                    max_path = dep_path
            
            total_length = max_length + task_hours
            total_path = max_path + [task_id]
            
            memo[task_id] = (total_length, total_path)
            return total_length, total_path
        
        # Find task with longest path
        max_length = 0
        critical_path = []
        
        for task in tasks:
            length, path = get_path_length(task['id'])
            if length > max_length:
                max_length = length
                critical_path = path
        
        return critical_path
    
    def _estimate_sequence_duration(self, tasks: List[Dict[str, Any]],
                                  dependencies: Dict[str, List[str]],
                                  parallel_groups: List[List[str]]) -> int:
        """Estimate duration for task sequence."""
        # Calculate based on critical path
        critical_path = self._calculate_critical_path(tasks, dependencies)
        task_map = {t['id']: t for t in tasks}
        
        total_hours = sum(
            task_map[task_id].get('estimated_hours', 0)
            for task_id in critical_path
        )
        
        # Convert hours to days (assuming 8 hour work days)
        return int(total_hours / 8) + 1
    
    def _parse_json_array_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse JSON array from AI response."""
        content = response.get('content', '')
        
        try:
            import re
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON array response: {e}")
            
        return []