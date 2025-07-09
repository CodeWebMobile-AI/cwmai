"""
Plugin-based Task Type System Base Classes

This module provides the foundation for a dynamic, extensible task type system
that supports runtime plugin discovery and registration.
"""

from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Type
from enum import Enum
import importlib
import pkgutil
import inspect
from pathlib import Path

from scripts.task_types import (
    TaskCategory, TaskComplexity, ArchitectureType,
    TaskTypeMetadata, SmartTaskType
)


class TaskTypePlugin(ABC):
    """Base class for all task type plugins."""
    
    @abstractmethod
    def get_task_types(self) -> Dict[str, TaskTypeMetadata]:
        """Return a dictionary of task type names to their metadata."""
        pass
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """Return plugin information (name, version, author, etc.)."""
        pass
    
    def validate_prerequisites(self, completed_tasks: Set[str]) -> bool:
        """Validate that all prerequisites for this plugin's tasks are met."""
        for task_name, metadata in self.get_task_types().items():
            if metadata.prerequisites:
                if not all(prereq in completed_tasks for prereq in metadata.prerequisites):
                    return False
        return True


class TaskTypeRegistryMeta(ABCMeta):
    """Metaclass for automatic plugin registration."""
    
    _registry: Dict[str, Type[TaskTypePlugin]] = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Register plugin if it's a concrete implementation
        if not inspect.isabstract(cls) and issubclass(cls, TaskTypePlugin):
            plugin_info = cls().get_plugin_info()
            plugin_name = plugin_info.get('name', name)
            mcs._registry[plugin_name] = cls
            
        return cls
    
    @classmethod
    def get_registry(mcs) -> Dict[str, Type[TaskTypePlugin]]:
        """Get the plugin registry."""
        return mcs._registry.copy()
    
    @classmethod
    def clear_registry(mcs):
        """Clear the plugin registry (useful for testing)."""
        mcs._registry.clear()


class DynamicTaskTypeRegistry:
    """Dynamic registry that discovers and manages task type plugins."""
    
    def __init__(self, plugin_dir: Optional[Path] = None):
        self.plugin_dir = plugin_dir or Path(__file__).parent / "task_type_plugins"
        self._plugins: Dict[str, TaskTypePlugin] = {}
        self._task_metadata: Dict[str, TaskTypeMetadata] = {}
        self._task_to_plugin: Dict[str, str] = {}
        
    def discover_plugins(self):
        """Discover and load all task type plugins."""
        # Clear existing plugins
        self._plugins.clear()
        self._task_metadata.clear()
        self._task_to_plugin.clear()
        
        # Ensure plugin directory exists
        self.plugin_dir.mkdir(exist_ok=True)
        
        # Import all Python modules in the plugin directory
        if self.plugin_dir.exists():
            for module_info in pkgutil.iter_modules([str(self.plugin_dir)]):
                if not module_info.ispkg:
                    module_name = f"scripts.task_type_plugins.{module_info.name}"
                    try:
                        importlib.import_module(module_name)
                    except Exception as e:
                        print(f"Error loading plugin {module_name}: {e}")
        
        # Get all registered plugins from the metaclass
        for plugin_name, plugin_class in TaskTypeRegistryMeta.get_registry().items():
            try:
                plugin_instance = plugin_class()
                self._plugins[plugin_name] = plugin_instance
                
                # Register all task types from this plugin
                for task_name, metadata in plugin_instance.get_task_types().items():
                    self._task_metadata[task_name] = metadata
                    self._task_to_plugin[task_name] = plugin_name
                    
            except Exception as e:
                print(f"Error instantiating plugin {plugin_name}: {e}")
    
    def get_task_metadata(self, task_name: str) -> Optional[TaskTypeMetadata]:
        """Get metadata for a specific task type."""
        return self._task_metadata.get(task_name)
    
    def get_all_task_types(self) -> Dict[str, TaskTypeMetadata]:
        """Get all registered task types and their metadata."""
        return self._task_metadata.copy()
    
    def get_tasks_by_category(self, category: TaskCategory) -> Dict[str, TaskTypeMetadata]:
        """Get all tasks of a specific category."""
        return {
            name: metadata
            for name, metadata in self._task_metadata.items()
            if metadata.category == category
        }
    
    def get_tasks_by_stage(self, stage: str) -> Dict[str, TaskTypeMetadata]:
        """Get all tasks applicable to a specific lifecycle stage."""
        return {
            name: metadata
            for name, metadata in self._task_metadata.items()
            if stage in metadata.applicable_stages
        }
    
    def get_tasks_by_architecture(
        self, architecture: ArchitectureType
    ) -> Dict[str, TaskTypeMetadata]:
        """Get all tasks applicable to a specific architecture."""
        return {
            name: metadata
            for name, metadata in self._task_metadata.items()
            if not metadata.applicable_architectures or 
               architecture in metadata.applicable_architectures
        }
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific plugin."""
        plugin = self._plugins.get(plugin_name)
        return plugin.get_plugin_info() if plugin else None
    
    def reload_plugins(self):
        """Reload all plugins (useful for development)."""
        # Clear the metaclass registry
        TaskTypeRegistryMeta.clear_registry()
        
        # Clear module cache for plugin modules
        import sys
        modules_to_remove = [
            mod for mod in sys.modules
            if mod.startswith('scripts.task_type_plugins.')
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Rediscover plugins
        self.discover_plugins()


# Example usage of creating a plugin
class ExampleTaskTypePlugin(TaskTypePlugin):
    """Example plugin showing how to create custom task types."""
    
    def get_task_types(self) -> Dict[str, TaskTypeMetadata]:
        return {
            "example_custom_task": TaskTypeMetadata(
                category=TaskCategory.DEVELOPMENT,
                complexity=TaskComplexity.MODERATE,
                typical_duration_hours=4,
                required_skills={"custom_skill"},
                applicable_stages={"all"},
                success_criteria={"example": True}
            )
        }
    
    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "name": "example_plugin",
            "version": "1.0.0",
            "author": "System",
            "description": "Example task type plugin"
        }