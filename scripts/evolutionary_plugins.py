"""
Evolutionary Plugin System

Provides extensibility through plugins for custom improvement generators,
fitness metrics, selection strategies, and more.
"""

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type
from abc import ABC, abstractmethod
import os
import json


class EvolutionaryPlugin(ABC):
    """Base class for all evolutionary plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
        
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
        
    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """Plugin type: generator, fitness, selection, mutation, etc."""
        pass
        
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute plugin functionality."""
        pass
        

class ImprovementGeneratorPlugin(EvolutionaryPlugin):
    """Plugin for custom improvement generation."""
    
    @property
    def plugin_type(self) -> str:
        return "generator"
        
    @abstractmethod
    async def generate_improvement(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an improvement."""
        pass
        

class FitnessMetricPlugin(EvolutionaryPlugin):
    """Plugin for custom fitness metrics."""
    
    @property
    def plugin_type(self) -> str:
        return "fitness"
        
    @abstractmethod
    async def calculate_fitness(self, improvement: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate fitness score."""
        pass
        

class SelectionStrategyPlugin(EvolutionaryPlugin):
    """Plugin for custom selection strategies."""
    
    @property
    def plugin_type(self) -> str:
        return "selection"
        
    @abstractmethod
    async def select_parents(self, population: List[Dict], num_parents: int) -> List[Dict]:
        """Select parents for next generation."""
        pass
        

class MutationOperatorPlugin(EvolutionaryPlugin):
    """Plugin for custom mutation operators."""
    
    @property
    def plugin_type(self) -> str:
        return "mutation"
        
    @abstractmethod
    async def mutate(self, improvement: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
        """Mutate an improvement."""
        pass
        

class EvolutionaryPluginManager:
    """Manages loading and execution of evolutionary plugins."""
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.logger = logging.getLogger(__name__)
        self.plugin_dirs = plugin_dirs or ["./evolutionary_plugins"]
        
        # Plugin registry
        self.plugins: Dict[str, Dict[str, EvolutionaryPlugin]] = {
            "generator": {},
            "fitness": {},
            "selection": {},
            "mutation": {},
            "validator": {},
            "analyzer": {}
        }
        
        # Plugin metadata
        self.plugin_metadata = {}
        
        # Load plugins
        self.discover_and_load_plugins()
        
    def discover_and_load_plugins(self):
        """Discover and load all plugins from plugin directories."""
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                plugin_path.mkdir(parents=True, exist_ok=True)
                self._create_example_plugins(plugin_path)
                
            # Scan for Python files
            for py_file in plugin_path.glob("*.py"):
                if py_file.stem.startswith("_"):
                    continue
                    
                try:
                    self._load_plugin_from_file(py_file)
                except Exception as e:
                    self.logger.error(f"Failed to load plugin from {py_file}: {e}")
                    
    def _load_plugin_from_file(self, file_path: Path):
        """Load a plugin from a Python file."""
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if not spec or not spec.loader:
            return
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin classes
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, EvolutionaryPlugin) and 
                obj != EvolutionaryPlugin and
                not inspect.isabstract(obj)):
                
                try:
                    # Instantiate plugin
                    plugin_instance = obj()
                    plugin_type = plugin_instance.plugin_type
                    plugin_name = plugin_instance.name
                    
                    # Register plugin
                    if plugin_type in self.plugins:
                        self.plugins[plugin_type][plugin_name] = plugin_instance
                        
                        # Store metadata
                        self.plugin_metadata[plugin_name] = {
                            "type": plugin_type,
                            "version": plugin_instance.version,
                            "file": str(file_path),
                            "class": name
                        }
                        
                        self.logger.info(f"Loaded plugin: {plugin_name} ({plugin_type})")
                        
                except Exception as e:
                    self.logger.error(f"Failed to instantiate plugin {name}: {e}")
                    
    def get_plugin(self, plugin_type: str, plugin_name: str) -> Optional[EvolutionaryPlugin]:
        """Get a specific plugin."""
        return self.plugins.get(plugin_type, {}).get(plugin_name)
        
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """List available plugins."""
        if plugin_type:
            return list(self.plugins.get(plugin_type, {}).keys())
        else:
            all_plugins = []
            for ptype, plugins in self.plugins.items():
                all_plugins.extend([f"{ptype}:{name}" for name in plugins.keys()])
            return all_plugins
            
    async def execute_plugin(self, plugin_type: str, plugin_name: str, 
                           context: Dict[str, Any]) -> Any:
        """Execute a specific plugin."""
        plugin = self.get_plugin(plugin_type, plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_type}:{plugin_name}")
            
        return await plugin.execute(context)
        
    async def execute_plugin_chain(self, plugin_type: str, context: Dict[str, Any]) -> List[Any]:
        """Execute all plugins of a type and collect results."""
        results = []
        
        for plugin_name, plugin in self.plugins.get(plugin_type, {}).items():
            try:
                result = await plugin.execute(context)
                results.append({
                    "plugin": plugin_name,
                    "result": result
                })
            except Exception as e:
                self.logger.error(f"Plugin {plugin_name} failed: {e}")
                
        return results
        
    def _create_example_plugins(self, plugin_dir: Path):
        """Create example plugins to demonstrate the system."""
        
        # Example improvement generator plugin
        generator_code = '''"""
Example Improvement Generator Plugin

Generates improvements focused on caching and performance.
"""

from scripts.evolutionary_plugins import ImprovementGeneratorPlugin
from typing import Dict, Any
import random


class CacheOptimizationGenerator(ImprovementGeneratorPlugin):
    """Generates cache optimization improvements."""
    
    @property
    def name(self) -> str:
        return "cache_optimizer"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    async def generate_improvement(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a cache optimization improvement."""
        cache_strategies = [
            "lru_cache", "ttl_cache", "distributed_cache", 
            "query_cache", "result_memoization"
        ]
        
        strategy = random.choice(cache_strategies)
        
        return {
            "name": f"{strategy}_implementation",
            "type": "optimization",
            "description": f"Implement {strategy} for improved performance",
            "approach": f"Use {strategy} pattern to reduce redundant computations",
            "metrics": ["cache_hit_rate", "response_time", "memory_usage"],
            "integration_points": ["http_client", "database_layer"],
            "code": self._generate_cache_code(strategy)
        }
        
    def _generate_cache_code(self, strategy: str) -> str:
        """Generate example cache code."""
        if strategy == "lru_cache":
            return """
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(param):
    # Expensive computation here
    result = perform_computation(param)
    return result
"""
        else:
            return f"# {strategy} implementation"
            
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute plugin."""
        return await self.generate_improvement(context)
'''
        
        # Example fitness metric plugin
        fitness_code = '''"""
Example Fitness Metric Plugin

Calculates fitness based on code quality metrics.
"""

from scripts.evolutionary_plugins import FitnessMetricPlugin
from typing import Dict, Any
import ast


class CodeQualityFitness(FitnessMetricPlugin):
    """Calculates fitness based on code quality."""
    
    @property
    def name(self) -> str:
        return "code_quality"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    async def calculate_fitness(self, improvement: Dict[str, Any], 
                               context: Dict[str, Any]) -> float:
        """Calculate fitness based on code quality metrics."""
        code = improvement.get("code", "")
        
        # Simple quality metrics
        score = 0.5  # Base score
        
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Has docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if ast.get_docstring(node):
                        score += 0.1
                        
            # Has type hints
            if ":" in code and "->" in code:
                score += 0.1
                
            # Has error handling
            if "try:" in code:
                score += 0.1
                
            # Reasonable length
            lines = code.split("\\n")
            if 10 < len(lines) < 100:
                score += 0.1
                
        except:
            score = 0.3  # Syntax error
            
        return min(1.0, score)
        
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute plugin."""
        improvement = context.get("improvement", {})
        return await self.calculate_fitness(improvement, context)
'''
        
        # Write example plugins
        (plugin_dir / "cache_optimization_generator.py").write_text(generator_code)
        (plugin_dir / "code_quality_fitness.py").write_text(fitness_code)
        
        # Create plugin manifest
        manifest = {
            "plugins": [
                {
                    "name": "cache_optimizer",
                    "type": "generator",
                    "file": "cache_optimization_generator.py",
                    "description": "Generates cache optimization improvements"
                },
                {
                    "name": "code_quality",
                    "type": "fitness",
                    "file": "code_quality_fitness.py",
                    "description": "Evaluates code quality metrics"
                }
            ],
            "version": "1.0.0"
        }
        
        with open(plugin_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
            
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about all loaded plugins."""
        info = {
            "loaded_plugins": {},
            "plugin_counts": {}
        }
        
        for plugin_type, plugins in self.plugins.items():
            info["plugin_counts"][plugin_type] = len(plugins)
            info["loaded_plugins"][plugin_type] = list(plugins.keys())
            
        return info