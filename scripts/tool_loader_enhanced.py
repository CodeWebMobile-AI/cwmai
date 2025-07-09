"""
Enhanced Tool Loader with Category Support

This module provides enhanced tool loading capabilities that support
organized tool categories and improved discovery.
"""

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Tuple
import asyncio
import functools

class EnhancedToolLoader:
    """Enhanced tool loader with category support."""
    
    def __init__(self, tool_system):
        """Initialize the enhanced tool loader.
        
        Args:
            tool_system: Reference to the main ToolCallingSystem
        """
        self.tool_system = tool_system
        self.logger = logging.getLogger(__name__)
        self.tool_categories: Dict[str, List[str]] = {}
        self.category_descriptions = {
            'core': 'Consolidated core functionality tools',
            'monetization': 'Business and monetization analysis tools',
            'repository': 'Repository and development control tools'
        }
        
    def load_all_tools(self) -> Dict[str, int]:
        """Load all tools from the custom tools directory.
        
        Returns:
            Dictionary with category names and tool counts
        """
        custom_tools_dir = Path("scripts/custom_tools")
        loaded_counts = {}
        
        # First, load tools from root directory (backward compatibility)
        root_tools = self._load_tools_from_directory(custom_tools_dir, category="general")
        if root_tools > 0:
            loaded_counts["general"] = root_tools
            
        # Then load from category subdirectories
        for subdir in custom_tools_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('_'):
                count = self._load_tools_from_directory(subdir, category=subdir.name)
                if count > 0:
                    loaded_counts[subdir.name] = count
                    
        # Log summary
        total_tools = sum(loaded_counts.values())
        self.logger.info(f"Loaded {total_tools} tools across {len(loaded_counts)} categories")
        
        return loaded_counts
        
    def _load_tools_from_directory(self, directory: Path, category: str) -> int:
        """Load tools from a specific directory.
        
        Args:
            directory: Path to the directory containing tools
            category: Category name for these tools
            
        Returns:
            Number of tools successfully loaded
        """
        if not directory.exists():
            return 0
            
        loaded_count = 0
        self.tool_categories[category] = []
        
        for tool_file in directory.glob("*.py"):
            if tool_file.name.startswith('_'):
                continue
                
            if self._load_single_tool(tool_file, category):
                loaded_count += 1
                self.tool_categories[category].append(tool_file.stem)
                
        return loaded_count
        
    def _load_single_tool(self, tool_file: Path, category: str) -> bool:
        """Load a single tool file.
        
        Args:
            tool_file: Path to the tool file
            category: Category for this tool
            
        Returns:
            True if successfully loaded, False otherwise
        """
        tool_name = tool_file.stem
        
        try:
            # Dynamic import
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            if not spec or not spec.loader:
                return False
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Try multiple function resolution strategies
            func = None
            
            # Strategy 1: Look for main() function
            if hasattr(module, 'main'):
                func = getattr(module, 'main')
                
            # Strategy 2: Look for function matching filename
            elif hasattr(module, tool_name):
                func = getattr(module, tool_name)
                
            # Strategy 3: Look for execute() function
            elif hasattr(module, 'execute'):
                func = getattr(module, 'execute')
                
            # Strategy 4: Look for class with execute method
            else:
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if hasattr(obj, 'execute'):
                        # Create instance and use execute method
                        instance = obj()
                        func = instance.execute
                        break
                        
            if func and callable(func):
                # Handle self parameter if needed
                func = self._wrap_function_if_needed(func)
                
                # Extract metadata
                description = self._extract_description(module, func, tool_name)
                parameters = self._extract_parameters(module, func)
                examples = getattr(module, '__examples__', [])
                
                # Register the tool with category
                self._register_tool_with_category(
                    name=tool_name,
                    func=func,
                    description=description,
                    parameters=parameters,
                    examples=examples,
                    category=category
                )
                
                return True
            else:
                self.logger.warning(f"No callable function found in {tool_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading tool {tool_file}: {str(e)}")
            return False
            
    def _wrap_function_if_needed(self, func: Callable) -> Callable:
        """Wrap function if it expects 'self' parameter.
        
        Args:
            func: Function to potentially wrap
            
        Returns:
            Wrapped or original function
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # If function expects 'self' as first parameter, wrap it
        if params and params[0] == 'self':
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def wrapped_func(**kwargs):
                    return await func(self.tool_system, **kwargs)
            else:
                @functools.wraps(func)
                def wrapped_func(**kwargs):
                    return func(self.tool_system, **kwargs)
            return wrapped_func
        
        return func
        
    def _extract_description(self, module, func: Callable, tool_name: str) -> str:
        """Extract description from module or function.
        
        Args:
            module: The module object
            func: The function object
            tool_name: Name of the tool
            
        Returns:
            Description string
        """
        # Try module-level description first
        if hasattr(module, '__description__'):
            return module.__description__
            
        # Then try docstring from top of file
        if module.__doc__:
            # Extract first paragraph
            lines = module.__doc__.strip().split('\n\n')[0].split('\n')
            description = ' '.join(line.strip() for line in lines)
            if description:
                return description
                
        # Try function docstring
        if func.__doc__:
            lines = func.__doc__.strip().split('\n\n')[0].split('\n')
            description = ' '.join(line.strip() for line in lines)
            if description:
                return description
                
        # Default description
        return f"Custom tool: {tool_name}"
        
    def _extract_parameters(self, module, func: Callable) -> Dict[str, Any]:
        """Extract parameters from module or function signature.
        
        Args:
            module: The module object
            func: The function object
            
        Returns:
            Parameters dictionary
        """
        # Try module-level parameters first
        if hasattr(module, '__parameters__'):
            return module.__parameters__
            
        # Extract from function signature
        sig = inspect.signature(func)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'cls']:
                continue
                
            param_info = {
                'type': 'string',  # Default type
                'required': param.default == param.empty
            }
            
            # Try to infer type from annotation
            if param.annotation != param.empty:
                if param.annotation == int:
                    param_info['type'] = 'integer'
                elif param.annotation == float:
                    param_info['type'] = 'number'
                elif param.annotation == bool:
                    param_info['type'] = 'boolean'
                elif param.annotation == list:
                    param_info['type'] = 'array'
                elif param.annotation == dict:
                    param_info['type'] = 'object'
                    
            # Add default value if exists
            if param.default != param.empty:
                param_info['default'] = param.default
                
            parameters[param_name] = param_info
            
        return parameters
        
    def _register_tool_with_category(self, name: str, func: Callable, 
                                   description: str, parameters: Dict[str, Any],
                                   examples: List[str], category: str):
        """Register a tool with category information.
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            parameters: Tool parameters
            examples: Usage examples
            category: Tool category
        """
        # Use the tool system's register method
        self.tool_system.register_tool(
            name=name,
            func=func,
            description=description,
            parameters=parameters,
            examples=examples
        )
        
        # Add category information
        if name in self.tool_system.tools:
            self.tool_system.tools[name].category = category
            
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tools in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of tool names in that category
        """
        return self.tool_categories.get(category, [])
        
    def get_all_categories(self) -> Dict[str, str]:
        """Get all available categories with descriptions.
        
        Returns:
            Dictionary of category names to descriptions
        """
        result = {}
        for category in self.tool_categories:
            result[category] = self.category_descriptions.get(
                category, 
                f"Tools in {category} category"
            )
        return result
        
    def find_tools_for_task(self, task_description: str) -> List[Tuple[str, str, float]]:
        """Find best tools for a given task description.
        
        Args:
            task_description: Description of the task
            
        Returns:
            List of (tool_name, category, relevance_score) tuples
        """
        results = []
        task_lower = task_description.lower()
        
        # Keywords for each category
        category_keywords = {
            'monetization': ['business', 'monetize', 'revenue', 'market', 'opportunity', 'profit'],
            'repository': ['repo', 'git', 'code', 'project', 'build', 'develop', 'database'],
            'core': ['analyze', 'count', 'monitor', 'create', 'file', 'system']
        }
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                tools = self.get_tools_by_category(category)
                for tool in tools:
                    # Additional scoring based on tool name
                    tool_score = score
                    if any(word in tool.lower() for word in task_lower.split()):
                        tool_score += 2
                        
                    results.append((tool, category, tool_score))
                    
        # Sort by relevance score
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:10]  # Return top 10 matches