"""
Dependency Injection Container for Task Type System

This module implements a lightweight DI container to decouple components
and make the system more testable and maintainable.
"""

from typing import Dict, Any, Type, Callable, Optional, TypeVar, Generic, List
from abc import ABC, abstractmethod
import inspect
from functools import wraps

T = TypeVar('T')


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not found in the container."""
    pass


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected."""
    pass


class DIContainer:
    """Dependency Injection Container for managing services and dependencies."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._resolving: set = set()
    
    def register(
        self,
        service_name: str,
        factory: Optional[Callable] = None,
        instance: Optional[Any] = None,
        singleton: bool = True
    ):
        """Register a service with the container.
        
        Args:
            service_name: Name of the service
            factory: Factory function to create the service
            instance: Pre-created instance
            singleton: Whether to create only one instance
        """
        if instance is not None:
            self._services[service_name] = instance
        elif factory is not None:
            self._factories[service_name] = factory
            if singleton:
                self._singletons[service_name] = None
        else:
            raise ValueError("Either factory or instance must be provided")
    
    def register_class(self, service_name: str, cls: Type, singleton: bool = True):
        """Register a class as a service.
        
        The container will automatically inject dependencies based on
        the class constructor parameters.
        """
        def factory():
            return self._create_instance(cls)
        
        self.register(service_name, factory=factory, singleton=singleton)
    
    def get(self, service_name: str) -> Any:
        """Get a service from the container."""
        if service_name in self._resolving:
            raise CircularDependencyError(f"Circular dependency detected for {service_name}")
        
        # Check if we have a direct instance
        if service_name in self._services:
            return self._services[service_name]
        
        # Check if we have a singleton instance
        if service_name in self._singletons and self._singletons[service_name] is not None:
            return self._singletons[service_name]
        
        # Check if we have a factory
        if service_name in self._factories:
            self._resolving.add(service_name)
            try:
                instance = self._factories[service_name]()
                if service_name in self._singletons:
                    self._singletons[service_name] = instance
                return instance
            finally:
                self._resolving.remove(service_name)
        
        raise ServiceNotFoundError(f"Service '{service_name}' not found")
    
    def _create_instance(self, cls: Type) -> Any:
        """Create an instance of a class with dependency injection."""
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Try to resolve by parameter name
            if param_name in self._services or param_name in self._factories:
                kwargs[param_name] = self.get(param_name)
            # Try to resolve by type annotation
            elif param.annotation != inspect.Parameter.empty:
                type_name = param.annotation.__name__
                if type_name in self._services or type_name in self._factories:
                    kwargs[param_name] = self.get(type_name)
        
        return cls(**kwargs)
    
    def clear(self):
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._resolving.clear()


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global DI container."""
    return _container


def inject(func: Callable) -> Callable:
    """Decorator to inject dependencies into a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        container = get_container()
        
        # Inject missing parameters
        for param_name, param in sig.parameters.items():
            if param_name not in kwargs:
                try:
                    kwargs[param_name] = container.get(param_name)
                except ServiceNotFoundError:
                    # If service not found, check type annotation
                    if param.annotation != inspect.Parameter.empty:
                        type_name = param.annotation.__name__
                        try:
                            kwargs[param_name] = container.get(type_name)
                        except ServiceNotFoundError:
                            pass
        
        return func(*args, **kwargs)
    
    return wrapper


class Injectable(ABC):
    """Base class for services that should be injectable."""
    
    @classmethod
    def register(cls, container: Optional[DIContainer] = None, **kwargs):
        """Register this class with the DI container."""
        if container is None:
            container = get_container()
        
        service_name = kwargs.get('name', cls.__name__)
        singleton = kwargs.get('singleton', True)
        
        container.register_class(service_name, cls, singleton=singleton)


# Example services for the task type system
class TaskTypeRegistry(Injectable):
    """Registry service for managing task types."""
    
    def __init__(self, config_loader: Optional['ConfigLoader'] = None):
        self.config_loader = config_loader
        self._task_types = {}
    
    def register_task_type(self, name: str, metadata: Dict[str, Any]):
        """Register a new task type."""
        self._task_types[name] = metadata
    
    def get_task_type(self, name: str) -> Optional[Dict[str, Any]]:
        """Get task type metadata."""
        return self._task_types.get(name)


class ConfigLoader(Injectable):
    """Service for loading configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "./configs"
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """Load configuration by name."""
        # Implementation would load from YAML/JSON
        return {}


class TaskSelector(Injectable):
    """Service for selecting appropriate tasks."""
    
    def __init__(self, registry: TaskTypeRegistry):
        self.registry = registry
    
    def select_tasks(self, context: Dict[str, Any]) -> List[str]:
        """Select tasks based on context."""
        # Implementation would use registry to select tasks
        return []


# Example usage
def setup_task_type_container():
    """Set up the DI container for the task type system."""
    container = get_container()
    
    # Register services
    ConfigLoader.register(container)
    TaskTypeRegistry.register(container)
    TaskSelector.register(container)
    
    # Services with dependencies will be automatically wired
    return container