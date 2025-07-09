"""
Dynamic Task Type Loader

This module bridges the gap between the static SmartTaskType enum
and the dynamic plugin system, allowing seamless migration.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import yaml
import logging

from scripts.task_types import (
    TaskTypeMetadata, TaskCategory, TaskComplexity, 
    ArchitectureType, SmartTaskType
)
from scripts.task_type_config import TaskTypePluginConfigModel


class DynamicTaskLoader:
    """Loads task types from YAML configuration files."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the dynamic task loader.
        
        Args:
            config_dir: Directory containing task type YAML files
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = config_dir or Path(__file__).parent / "task_type_configs"
        self._task_registry: Dict[str, TaskTypeMetadata] = {}
        self._task_names: Set[str] = set()
        self._loaded = False
        
    def load_task_types(self) -> Dict[str, TaskTypeMetadata]:
        """Load all task types from YAML configuration files.
        
        Returns:
            Dictionary mapping task type names to their metadata
        """
        if self._loaded:
            return self._task_registry
            
        self._task_registry.clear()
        self._task_names.clear()
        
        # Load all YAML files in the config directory
        if self.config_dir.exists():
            for yaml_file in self.config_dir.glob("*.yaml"):
                try:
                    self._load_yaml_file(yaml_file)
                except Exception as e:
                    self.logger.error(f"Error loading {yaml_file}: {e}")
        
        self._loaded = True
        self.logger.info(f"Loaded {len(self._task_registry)} task types from {self.config_dir}")
        return self._task_registry
    
    def _load_yaml_file(self, yaml_path: Path) -> None:
        """Load task types from a single YAML file.
        
        Args:
            yaml_path: Path to the YAML file
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate using Pydantic model
        try:
            config = TaskTypePluginConfigModel(**data)
        except Exception as e:
            self.logger.error(f"Validation error in {yaml_path}: {e}")
            return
        
        # Convert to TaskTypeMetadata objects
        for task_config in config.task_types:
            # Map string values to enums
            try:
                category = TaskCategory(task_config.category)
            except ValueError:
                self.logger.warning(f"Invalid category '{task_config.category}' for {task_config.name}")
                continue
            
            # Map complexity strings to enum
            complexity_map = {
                "trivial": TaskComplexity.TRIVIAL,
                "simple": TaskComplexity.SIMPLE,
                "moderate": TaskComplexity.MODERATE,
                "complex": TaskComplexity.COMPLEX,
                "epic": TaskComplexity.EPIC
            }
            
            complexity = complexity_map.get(task_config.complexity, TaskComplexity.MODERATE)
            
            # Convert architecture strings to enum (only LARAVEL_REACT for now)
            architectures = set()
            for arch in task_config.applicable_architectures:
                if arch == "laravel_react":
                    architectures.add(ArchitectureType.LARAVEL_REACT)
            
            # Create TaskTypeMetadata
            metadata = TaskTypeMetadata(
                category=category,
                complexity=complexity,
                typical_duration_hours=task_config.typical_duration_hours,
                required_skills=set(task_config.required_skills),
                applicable_stages=set(task_config.applicable_stages),
                applicable_architectures=architectures,
                prerequisites=task_config.prerequisites,
                success_criteria=task_config.success_criteria,
                priority_modifier=task_config.priority_modifier
            )
            
            self._task_registry[task_config.name] = metadata
            self._task_names.add(task_config.name)
    
    def get_task_metadata(self, task_name: str) -> Optional[TaskTypeMetadata]:
        """Get metadata for a specific task type.
        
        Args:
            task_name: Name of the task type
            
        Returns:
            TaskTypeMetadata or None if not found
        """
        if not self._loaded:
            self.load_task_types()
        return self._task_registry.get(task_name)
    
    def get_all_task_names(self) -> Set[str]:
        """Get all available task type names.
        
        Returns:
            Set of task type names
        """
        if not self._loaded:
            self.load_task_types()
        return self._task_names.copy()
    
    def get_appropriate_task_types(
        self,
        architecture: Optional[ArchitectureType],
        lifecycle_stage: str,
        current_needs: List[str],
        completed_tasks: Set[str]
    ) -> List[str]:
        """Get task types appropriate for the current context.
        
        This mimics TaskTypeSelector.get_appropriate_task_types but returns
        task names instead of enum values.
        
        Args:
            architecture: Project architecture type
            lifecycle_stage: Current lifecycle stage
            current_needs: Identified needs from analysis
            completed_tasks: Set of already completed task types
            
        Returns:
            List of appropriate task type names
        """
        if not self._loaded:
            self.load_task_types()
            
        appropriate_types = []
        
        for task_name, metadata in self._task_registry.items():
            # Check if task is applicable to current stage
            if lifecycle_stage not in metadata.applicable_stages:
                continue
                
            # Check if task is applicable to architecture
            if architecture and metadata.applicable_architectures:
                if architecture not in metadata.applicable_architectures:
                    continue
            
            # Check prerequisites
            if metadata.prerequisites:
                if not all(prereq in completed_tasks for prereq in metadata.prerequisites):
                    continue
            
            # Check if already completed
            if task_name in completed_tasks:
                continue
                
            appropriate_types.append(task_name)
        
        # Sort by priority and complexity
        appropriate_types.sort(
            key=lambda t: (
                -self._task_registry[t].priority_modifier,
                self._task_registry[t].complexity.value
            )
        )
        
        return appropriate_types


# Global instance for easy access
_global_loader = DynamicTaskLoader()


def get_dynamic_task_loader() -> DynamicTaskLoader:
    """Get the global dynamic task loader instance.
    
    Returns:
        DynamicTaskLoader instance
    """
    return _global_loader


def reload_task_types() -> None:
    """Force reload of task types from configuration files."""
    _global_loader._loaded = False
    _global_loader.load_task_types()