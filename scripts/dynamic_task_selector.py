"""
Dynamic Task Type Selector

This module provides a drop-in replacement for TaskTypeSelector that uses
the dynamic task loading system while maintaining the same interface.
"""

from typing import List, Optional, Set, Dict
from scripts.task_types import ArchitectureType, TaskTypeMetadata, TaskComplexity, TaskCategory
from scripts.dynamic_task_loader import get_dynamic_task_loader


class DynamicTaskTypeSelector:
    """Dynamic task type selection based on context."""
    
    @staticmethod
    def get_appropriate_task_types(
        architecture: Optional[ArchitectureType],
        lifecycle_stage: str,
        current_needs: List[str],
        completed_tasks: Set[str]
    ) -> List[str]:
        """Get task types appropriate for the current context.
        
        Args:
            architecture: Project architecture type
            lifecycle_stage: Current lifecycle stage
            current_needs: Identified needs from analysis
            completed_tasks: Set of already completed task types
            
        Returns:
            List of appropriate task type names (not enum values)
        """
        loader = get_dynamic_task_loader()
        return loader.get_appropriate_task_types(
            architecture, lifecycle_stage, current_needs, completed_tasks
        )
    
    @staticmethod
    def get_task_metadata(task_name: str) -> Optional[TaskTypeMetadata]:
        """Get metadata for a task type.
        
        Args:
            task_name: Name of the task type (string, not enum)
            
        Returns:
            TaskTypeMetadata or default if not found
        """
        loader = get_dynamic_task_loader()
        metadata = loader.get_task_metadata(task_name)
        
        if metadata:
            return metadata
        
        # Return default metadata if not found
        return TaskTypeMetadata(
            category=TaskCategory.DEVELOPMENT,
            complexity=TaskComplexity.MODERATE,
            typical_duration_hours=4
        )
    
    @staticmethod
    def estimate_task_duration(
        task_name: str,
        repository_complexity: float = 1.0
    ) -> int:
        """Estimate task duration in cycles.
        
        Args:
            task_name: The task type name
            repository_complexity: Complexity multiplier for the repository
            
        Returns:
            Estimated cycles (where 1 cycle ≈ 1 hour)
        """
        metadata = DynamicTaskTypeSelector.get_task_metadata(task_name)
        base_hours = metadata.typical_duration_hours
        
        # Adjust for repository complexity
        adjusted_hours = base_hours * repository_complexity
        
        # Convert to cycles (rounded up)
        return max(1, int(adjusted_hours + 0.5))
    
    @staticmethod
    def get_all_task_names() -> Set[str]:
        """Get all available task type names.
        
        Returns:
            Set of all task type names
        """
        loader = get_dynamic_task_loader()
        return loader.get_all_task_names()


def get_task_type_for_string(task_string: str) -> Optional[str]:
    """Convert a string task type to a valid task name.
    
    Args:
        task_string: String representation of task type
        
    Returns:
        Valid task name or None if not found
    """
    loader = get_dynamic_task_loader()
    all_tasks = loader.get_all_task_names()
    
    # Try direct match
    if task_string.lower() in all_tasks:
        return task_string.lower()
    
    # Try mapping common strings
    string_mappings = {
        "feature": "feature_business_logic",
        "testing": "testing_unit_tests",
        "optimization": "optimization_performance",
        "maintenance": "maintenance_bug_fix",
        "infrastructure": "infrastructure_deployment",
        "integration": "integration_api",
        "research": "research_technology",
        "repository_health": "repository_health",
        "new_project": "new_project",
    }
    
    task_lower = task_string.lower()
    for key, value in string_mappings.items():
        if key in task_lower and value in all_tasks:
            return value
    
    return None