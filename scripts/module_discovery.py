"""
Module Discovery System

Intelligently discovers available modules and suggests alternatives
for non-existent imports.
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

from scripts.logger import get_logger


class ModuleDiscovery:
    """Discovers available modules and suggests alternatives."""
    
    def __init__(self):
        self.logger = get_logger('ModuleDiscovery')
        self.scripts_dir = Path("scripts")
        self.module_cache = None
        self._discovered_modules = None
        self._class_to_module = None
        self._function_to_module = None
        
    def discover_all_modules(self) -> Dict[str, Dict[str, any]]:
        """Discover all available modules and their contents."""
        if self._discovered_modules is not None:
            return self._discovered_modules
            
        self._discovered_modules = {}
        self._class_to_module = {}
        self._function_to_module = {}
        
        for py_file in self.scripts_dir.glob("*.py"):
            if py_file.name.startswith('_'):
                continue
                
            module_name = py_file.stem
            module_info = self._analyze_module(py_file)
            
            if module_info:
                self._discovered_modules[module_name] = module_info
                
                # Build reverse mappings
                for class_name in module_info.get('classes', []):
                    self._class_to_module[class_name] = module_name
                    
                for func_name in module_info.get('functions', []):
                    self._function_to_module[func_name] = module_name
                    
        print(f"Discovered {len(self._discovered_modules)} modules")
        return self._discovered_modules
        
    def _analyze_module(self, file_path: Path) -> Optional[Dict[str, any]]:
        """Analyze a Python module to extract its contents."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            module_info = {
                'path': str(file_path),
                'classes': [],
                'functions': [],
                'imports': [],
                'description': None
            }
            
            # Extract module docstring
            if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
                module_info['description'] = tree.body[0].value.s.strip()
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    module_info['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):
                        module_info['functions'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_info['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_info['imports'].append(node.module)
                        
            return module_info
            
        except Exception as e:
            self.logger.debug(f"Could not analyze {file_path}: {e}")
            return None
            
    def find_module_for_class(self, class_name: str) -> Optional[str]:
        """Find which module contains a specific class."""
        if self._class_to_module is None:
            self.discover_all_modules()
            
        return self._class_to_module.get(class_name)
        
    def find_module_for_function(self, function_name: str) -> Optional[str]:
        """Find which module contains a specific function."""
        if self._function_to_module is None:
            self.discover_all_modules()
            
        return self._function_to_module.get(function_name)
        
    def suggest_module_for_import(self, missing_module: str) -> Optional[str]:
        """Suggest an alternative module for a missing import."""
        if self._discovered_modules is None:
            self.discover_all_modules()
            
        # Direct module mappings
        module_mappings = {
            # GitHub related
            'github_api': 'github_issue_creator',
            'github_manager': 'github_issue_creator',
            'github_client': 'mcp_integration',
            
            # Error handling
            'error_reporter': 'error_handler',
            'error_manager': 'error_handler',
            'exception_handler': 'error_handler',
            
            # Configuration
            'config_manager': 'mcp_config',
            'configuration': 'mcp_config',
            'settings_manager': 'mcp_config',
            
            # Task/Tool execution
            'tool_executor': 'tool_calling_system',
            'task_executor': 'task_manager',
            'workflow_manager': 'workflow_orchestrator',
            
            # AI related
            'ai_executor': 'ai_brain',
            'ai_manager': 'ai_brain',
            'ai_client': 'http_ai_client',
            
            # State management
            'state_storage': 'state_manager',
            'state_handler': 'state_manager',
            'persistence_manager': 'state_manager',
        }
        
        # Check direct mapping
        if missing_module in module_mappings:
            suggested = module_mappings[missing_module]
            if suggested in self._discovered_modules:
                return suggested
                
        # Try fuzzy matching
        missing_parts = set(missing_module.lower().split('_'))
        best_match = None
        best_score = 0
        
        for module_name in self._discovered_modules:
            module_parts = set(module_name.lower().split('_'))
            common_parts = missing_parts.intersection(module_parts)
            score = len(common_parts) / max(len(missing_parts), len(module_parts))
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = module_name
                
        return best_match
        
    def suggest_class_import(self, class_name: str) -> Optional[Tuple[str, str]]:
        """Suggest import statement for a class."""
        module = self.find_module_for_class(class_name)
        if module:
            return (f"from scripts.{module} import {class_name}", module)
        return None
        
    def suggest_function_import(self, function_name: str) -> Optional[Tuple[str, str]]:
        """Suggest import statement for a function."""
        module = self.find_module_for_function(function_name)
        if module:
            return (f"from scripts.{module} import {function_name}", module)
        return None
        
    def get_module_classes(self, module_name: str) -> List[str]:
        """Get all classes in a module."""
        if self._discovered_modules is None:
            self.discover_all_modules()
            
        module_info = self._discovered_modules.get(module_name, {})
        return module_info.get('classes', [])
        
    def get_module_functions(self, module_name: str) -> List[str]:
        """Get all functions in a module."""
        if self._discovered_modules is None:
            self.discover_all_modules()
            
        module_info = self._discovered_modules.get(module_name, {})
        return module_info.get('functions', [])


# Singleton instance
_module_discovery = None

def get_module_discovery() -> ModuleDiscovery:
    """Get the singleton ModuleDiscovery instance."""
    global _module_discovery
    if _module_discovery is None:
        _module_discovery = ModuleDiscovery()
    return _module_discovery