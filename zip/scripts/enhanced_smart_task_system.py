"""
Enhanced Smart Task Type System

This module integrates all components of the intelligent task ecosystem:
- Plugin-based task type discovery
- YAML configuration with validation
- Dependency injection
- DAG-based dependency modeling
- Context-aware prioritization
- Project management intelligence
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import logging

from scripts.task_type_plugin_base import DynamicTaskTypeRegistry, TaskTypePlugin
from scripts.task_type_config import YAMLTaskTypeLoader, TaskTypeConfigModel
from scripts.task_type_container import DIContainer, get_container
from scripts.task_dependency_graph import TaskDependencyGraph, TaskNode, TaskStatus
from scripts.project_context_model import ProjectContext, ProjectStage
from scripts.task_prioritization_strategies import (
    PrioritizationStrategyFactory,
    PrioritizedTask
)
from scripts.task_types import SmartTaskType, TaskTypeMetadata, get_task_type_for_string

logger = logging.getLogger(__name__)


class EnhancedSmartTaskSystem:
    """
    The main interface for the enhanced smart task type system.
    
    This system provides:
    - Dynamic task type management with plugins
    - Intelligent task selection based on context
    - Dependency-aware execution planning
    - Advanced project management analytics
    """
    
    def __init__(
        self,
        plugin_dir: Optional[Path] = None,
        config_dir: Optional[Path] = None,
        container: Optional[DIContainer] = None
    ):
        # Directories
        self.plugin_dir = plugin_dir or Path(__file__).parent / "task_type_plugins"
        self.config_dir = config_dir or Path(__file__).parent / "task_type_configs"
        
        # Core components
        self.container = container or get_container()
        self.registry = DynamicTaskTypeRegistry(self.plugin_dir)
        self.yaml_loader = YAMLTaskTypeLoader(self.config_dir)
        self.dependency_graph = TaskDependencyGraph()
        
        # Strategy for prioritization
        self.prioritization_strategy = "adaptive"  # Default to adaptive
        
        # Cache for performance
        self._task_metadata_cache: Dict[str, TaskTypeConfigModel] = {}
        
        # Initialize the system
        self._initialize()
    
    def _initialize(self):
        """Initialize the task system components."""
        # Register core services with DI container
        self.container.register("task_registry", instance=self.registry)
        self.container.register("yaml_loader", instance=self.yaml_loader)
        self.container.register("dependency_graph", instance=self.dependency_graph)
        
        # Load plugins
        self.registry.discover_plugins()
        
        # Load YAML configurations
        self._load_yaml_configs()
        
        # Migrate existing SmartTaskType definitions
        self._migrate_existing_task_types()
    
    def _load_yaml_configs(self):
        """Load task type definitions from YAML files."""
        configs = self.yaml_loader.load_all_configs()
        
        for plugin_name, config in configs.items():
            for task_type_config in config.task_types:
                self._task_metadata_cache[task_type_config.name] = task_type_config
    
    def _migrate_existing_task_types(self):
        """Migrate existing SmartTaskType enum to the new system."""
        # Import the existing registry
        from scripts.task_types import TASK_TYPE_REGISTRY
        
        for smart_type, metadata in TASK_TYPE_REGISTRY.items():
            # Convert to new config model
            config = TaskTypeConfigModel(
                name=smart_type.value,
                category=metadata.category.value,
                complexity=metadata.complexity.name.lower(),
                typical_duration_hours=metadata.typical_duration_hours,
                required_skills=list(metadata.required_skills),
                applicable_stages=list(metadata.applicable_stages),
                applicable_architectures=[
                    arch.value for arch in metadata.applicable_architectures
                ],
                prerequisites=metadata.prerequisites,
                success_criteria=metadata.success_criteria,
                priority_modifier=metadata.priority_modifier
            )
            
            self._task_metadata_cache[smart_type.value] = config
    
    def get_task_metadata(self, task_type: str) -> Optional[TaskTypeConfigModel]:
        """Get metadata for a task type."""
        # Check cache first
        if task_type in self._task_metadata_cache:
            return self._task_metadata_cache[task_type]
        
        # Check plugin registry
        plugin_metadata = self.registry.get_task_metadata(task_type)
        if plugin_metadata:
            # Convert to config model
            return self._convert_metadata_to_config(task_type, plugin_metadata)
        
        # Try to map from string
        smart_type = get_task_type_for_string(task_type)
        if smart_type and smart_type.value != task_type:
            # Only recurse if we got a different value
            return self.get_task_metadata(smart_type.value)
        
        return None
    
    def _convert_metadata_to_config(
        self, 
        task_type: str, 
        metadata: TaskTypeMetadata
    ) -> TaskTypeConfigModel:
        """Convert old TaskTypeMetadata to new ConfigModel."""
        return TaskTypeConfigModel(
            name=task_type,
            category=metadata.category.value,
            complexity=metadata.complexity.name.lower(),
            typical_duration_hours=metadata.typical_duration_hours,
            required_skills=list(metadata.required_skills),
            applicable_stages=list(metadata.applicable_stages),
            applicable_architectures=[
                arch.value for arch in metadata.applicable_architectures
            ],
            prerequisites=metadata.prerequisites,
            success_criteria=metadata.success_criteria,
            priority_modifier=metadata.priority_modifier
        )
    
    def select_tasks_for_context(
        self,
        context: ProjectContext,
        max_tasks: int = 10,
        strategy: Optional[str] = None
    ) -> List[PrioritizedTask]:
        """Select appropriate tasks based on project context."""
        # Get all available task types
        all_tasks = self._get_applicable_tasks(context)
        
        # Create prioritization strategy
        strategy_name = strategy or self.prioritization_strategy
        prioritizer = PrioritizationStrategyFactory.create_strategy(strategy_name)
        
        # Prioritize tasks
        prioritized = prioritizer.prioritize_tasks(all_tasks, context)
        
        # Filter out completed tasks
        available_tasks = [
            task for task in prioritized
            if task.task_type not in context.completed_task_types
        ]
        
        # Check prerequisites
        ready_tasks = []
        for task in available_tasks:
            if self._check_prerequisites(task, context):
                ready_tasks.append(task)
        
        # Return top N tasks
        return ready_tasks[:max_tasks]
    
    def _get_applicable_tasks(
        self, 
        context: ProjectContext
    ) -> Dict[str, TaskTypeConfigModel]:
        """Get all tasks applicable to the current context."""
        applicable = {}
        
        for task_type, metadata in self._task_metadata_cache.items():
            # Check stage applicability
            if context.stage.value not in metadata.applicable_stages:
                continue
            
            # Check architecture applicability
            if metadata.applicable_architectures:
                arch_value = context.technical_context.architecture_type.value
                if arch_value not in metadata.applicable_architectures:
                    continue
            
            applicable[task_type] = metadata
        
        return applicable
    
    def _check_prerequisites(
        self, 
        task: PrioritizedTask, 
        context: ProjectContext
    ) -> bool:
        """Check if all prerequisites for a task are met."""
        for prereq in task.metadata.prerequisites:
            if prereq not in context.completed_task_types:
                return False
        return True
    
    def create_execution_plan(
        self,
        selected_tasks: List[str],
        context: ProjectContext
    ) -> Tuple[List[List[str]], float]:
        """Create an execution plan with dependency analysis.
        
        Returns:
            Tuple of (execution_groups, estimated_duration)
            where execution_groups are tasks that can run in parallel
        """
        # Clear existing graph
        self.dependency_graph = TaskDependencyGraph()
        
        # Add all tasks to the graph
        for task_type in selected_tasks:
            metadata = self.get_task_metadata(task_type)
            if metadata:
                node = TaskNode(
                    task_id=task_type,
                    task_type=task_type,
                    metadata=metadata
                )
                self.dependency_graph.add_task(node)
        
        # Add dependencies
        for task_type in selected_tasks:
            metadata = self.get_task_metadata(task_type)
            if metadata:
                # Add prerequisite dependencies
                for prereq in metadata.prerequisites:
                    if prereq in selected_tasks:
                        self.dependency_graph.add_dependency(prereq, task_type)
                
                # Add explicit dependencies
                for dep in metadata.dependencies:
                    if dep in selected_tasks:
                        self.dependency_graph.add_dependency(task_type, dep)
        
        # Get parallel execution groups
        parallel_groups = self.dependency_graph.get_parallel_groups()
        
        # Calculate critical path
        critical_path, duration = self.dependency_graph.calculate_critical_path()
        
        return parallel_groups, duration
    
    def analyze_project_schedule(
        self,
        selected_tasks: List[str],
        context: ProjectContext,
        target_deadline: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive project schedule analysis."""
        # Create execution plan
        parallel_groups, expected_duration = self.create_execution_plan(
            selected_tasks, context
        )
        
        # Calculate PERT probability if deadline provided
        completion_probability = None
        if target_deadline:
            days_until_deadline = (target_deadline - datetime.now()).days
            target_hours = days_until_deadline * 8  # Assuming 8-hour workdays
            completion_probability = self.dependency_graph.calculate_pert_probability(
                target_hours
            )
        
        # Get critical path
        critical_path, _ = self.dependency_graph.calculate_critical_path()
        
        # Analyze resource requirements
        resource_requirements = self._analyze_resource_requirements(
            selected_tasks, context
        )
        
        return {
            "execution_groups": parallel_groups,
            "expected_duration_hours": expected_duration,
            "critical_path": critical_path,
            "completion_probability": completion_probability,
            "resource_requirements": resource_requirements,
            "gantt_data": self.dependency_graph.export_to_gantt_data()
        }
    
    def _analyze_resource_requirements(
        self,
        selected_tasks: List[str],
        context: ProjectContext
    ) -> Dict[str, Any]:
        """Analyze resource requirements for selected tasks."""
        total_hours = 0
        required_skills = set()
        skill_hours = {}
        
        for task_type in selected_tasks:
            metadata = self.get_task_metadata(task_type)
            if metadata:
                total_hours += metadata.typical_duration_hours
                required_skills.update(metadata.required_skills)
                
                for skill in metadata.required_skills:
                    if skill not in skill_hours:
                        skill_hours[skill] = 0
                    skill_hours[skill] += metadata.typical_duration_hours
        
        # Check team capacity
        available_hours = sum(
            member.available_hours_per_week - member.current_workload
            for member in context.team_members
        )
        
        return {
            "total_hours_required": total_hours,
            "required_skills": list(required_skills),
            "skill_hours": skill_hours,
            "team_capacity_hours": available_hours,
            "capacity_utilization": total_hours / available_hours if available_hours > 0 else float('inf')
        }
    
    def recommend_task_assignments(
        self,
        selected_tasks: List[str],
        context: ProjectContext
    ) -> Dict[str, str]:
        """Recommend which team member should handle each task."""
        assignments = {}
        
        # Sort tasks by priority
        task_priorities = []
        for task_type in selected_tasks:
            metadata = self.get_task_metadata(task_type)
            if metadata:
                task_priorities.append((task_type, metadata.priority_modifier))
        
        task_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Assign tasks to team members
        for task_type, _ in task_priorities:
            metadata = self.get_task_metadata(task_type)
            if not metadata:
                continue
            
            # Find best team member
            best_member = None
            best_score = 0
            
            for member in context.team_members:
                # Check if member has required skills
                skill_match = len(
                    metadata.required_skills.intersection(member.skills)
                ) / len(metadata.required_skills) if metadata.required_skills else 1.0
                
                # Check availability
                if member.availability_ratio <= 0:
                    continue
                
                # Calculate assignment score
                score = (
                    skill_match * 
                    member.availability_ratio * 
                    member.task_completion_rate
                )
                
                # Boost if task type is preferred
                if task_type in member.preferred_task_types:
                    score *= 1.5
                
                if score > best_score:
                    best_score = score
                    best_member = member
            
            if best_member:
                assignments[task_type] = best_member.id
                # Update member workload
                best_member.current_workload += metadata.typical_duration_hours
        
        return assignments
    
    def update_task_status(
        self,
        task_type: str,
        status: TaskStatus,
        context: ProjectContext
    ):
        """Update the status of a task and record in context."""
        self.dependency_graph.update_task_status(task_type, status)
        
        if status == TaskStatus.COMPLETED:
            context.completed_task_types.add(task_type)
            if task_type in context.active_tasks:
                context.active_tasks.remove(task_type)
        elif status == TaskStatus.IN_PROGRESS:
            if task_type not in context.active_tasks:
                context.active_tasks.append(task_type)
        elif status == TaskStatus.FAILED:
            context.failed_task_types.add(task_type)
            if task_type in context.active_tasks:
                context.active_tasks.remove(task_type)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get metrics about the task system."""
        return {
            "total_task_types": len(self._task_metadata_cache),
            "plugin_count": len(self.registry._plugins),
            "active_tasks": len([
                task for task in self.dependency_graph._task_nodes.values()
                if task.status == TaskStatus.IN_PROGRESS
            ]),
            "completed_tasks": len([
                task for task in self.dependency_graph._task_nodes.values()
                if task.status == TaskStatus.COMPLETED
            ]),
            "available_strategies": PrioritizationStrategyFactory.get_available_strategies()
        }


# Convenience functions for backward compatibility
def create_enhanced_task_system() -> EnhancedSmartTaskSystem:
    """Create and initialize the enhanced smart task system."""
    return EnhancedSmartTaskSystem()


def get_task_recommendations(
    context: ProjectContext,
    max_tasks: int = 5
) -> List[Dict[str, Any]]:
    """Get task recommendations for the current context."""
    system = create_enhanced_task_system()
    
    # Get prioritized tasks
    tasks = system.select_tasks_for_context(context, max_tasks)
    
    # Convert to simple format
    recommendations = []
    for task in tasks:
        recommendations.append({
            "task_type": task.task_type,
            "priority_score": task.priority_score,
            "complexity": task.metadata.complexity,
            "duration_hours": task.metadata.typical_duration_hours,
            "required_skills": task.metadata.required_skills,
            "priority_factors": task.priority_factors
        })
    
    return recommendations