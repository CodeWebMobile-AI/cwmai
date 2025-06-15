#!/usr/bin/env python3
"""
Enhanced Tool Validation System
Validates and tests tools before they're added to the system
"""

import ast
import asyncio
import inspect
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone


class ToolTestResult:
    """Result of tool testing."""
    def __init__(self):
        self.syntax_valid = False
        self.imports_valid = False
        self.signature_valid = False
        self.execution_success = False
        self.has_error_handling = False
        self.returns_dict = False
        self.performance_ms = 0
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.test_output: Any = None
        
    @property
    def is_valid(self) -> bool:
        """Check if tool passes all critical tests."""
        return (self.syntax_valid and 
                self.imports_valid and 
                self.signature_valid and 
                self.execution_success and
                self.returns_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.is_valid,
            "syntax_valid": self.syntax_valid,
            "imports_valid": self.imports_valid,
            "signature_valid": self.signature_valid,
            "execution_success": self.execution_success,
            "has_error_handling": self.has_error_handling,
            "returns_dict": self.returns_dict,
            "performance_ms": self.performance_ms,
            "issues": self.issues,
            "warnings": self.warnings,
            "test_output": str(self.test_output) if self.test_output else None
        }


class EnhancedToolValidator:
    """Enhanced validation system for auto-generated tools."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.required_imports = {
            'StateManager': 'from scripts.state_manager import StateManager',
            'TaskManager': 'from scripts.task_manager import TaskManager',
        }
        
    async def validate_tool(self, tool_file: Path, tool_name: str) -> ToolTestResult:
        """Comprehensive tool validation."""
        result = ToolTestResult()
        
        if not tool_file.exists():
            result.issues.append(f"Tool file not found: {tool_file}")
            return result
        
        code = tool_file.read_text()
        
        # Step 1: Validate syntax
        self._validate_syntax(code, result)
        if not result.syntax_valid:
            return result
        
        # Step 2: Check code structure
        self._check_code_structure(code, tool_name, result)
        
        # Step 3: Validate imports
        self._validate_imports(code, result)
        
        # Step 4: Test execution
        await self._test_execution(tool_file, tool_name, result)
        
        # Step 5: Performance test
        if result.execution_success:
            await self._test_performance(tool_file, tool_name, result)
        
        return result
    
    def _validate_syntax(self, code: str, result: ToolTestResult):
        """Validate Python syntax."""
        try:
            ast.parse(code)
            result.syntax_valid = True
        except SyntaxError as e:
            result.syntax_valid = False
            result.issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
    
    def _check_code_structure(self, code: str, tool_name: str, result: ToolTestResult):
        """Check for structural issues."""
        lines = code.split('\n')
        
        # Check for problematic patterns
        has_main_func = False
        has_tool_func = False
        has_self_param = False
        
        for i, line in enumerate(lines, 1):
            # Check for main() function
            if 'def main(' in line or 'async def main(' in line:
                result.warnings.append(f"Line {i}: Contains main() function")
            
            # Check for __main__ block
            if "if __name__ == '__main__':" in line:
                result.warnings.append(f"Line {i}: Contains __main__ block")
            
            # Check for tool function
            if f'def {tool_name}(' in line or f'async def {tool_name}(' in line:
                has_tool_func = True
                # Check for self parameter
                if '(self' in line:
                    has_self_param = True
                    result.issues.append(f"Line {i}: Tool function has 'self' parameter")
            
            # Check for error handling
            if 'try:' in line:
                result.has_error_handling = True
        
        if not has_tool_func:
            result.issues.append(f"Missing tool function: {tool_name}")
        
        result.signature_valid = has_tool_func and not has_self_param
        
        # Check for required metadata
        if '__description__' not in code:
            result.warnings.append("Missing __description__ metadata")
        if '__parameters__' not in code:
            result.warnings.append("Missing __parameters__ metadata")
    
    def _validate_imports(self, code: str, result: ToolTestResult):
        """Validate imports are correct."""
        try:
            tree = ast.parse(code)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            # Check if StateManager/TaskManager are used but not imported
            if 'StateManager()' in code and not any('StateManager' in imp for imp in imports):
                result.issues.append("Uses StateManager but doesn't import it")
                result.imports_valid = False
            elif 'TaskManager()' in code and not any('TaskManager' in imp for imp in imports):
                result.issues.append("Uses TaskManager but doesn't import it")
                result.imports_valid = False
            else:
                result.imports_valid = True
                
        except Exception as e:
            result.imports_valid = False
            result.issues.append(f"Import validation error: {str(e)}")
    
    async def _test_execution(self, tool_file: Path, tool_name: str, result: ToolTestResult):
        """Test tool execution."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the tool function
            if not hasattr(module, tool_name):
                result.issues.append(f"Module missing function: {tool_name}")
                return
            
            func = getattr(module, tool_name)
            
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                output = await func()
            else:
                output = func()
            
            result.test_output = output
            result.execution_success = True
            
            # Check output type
            if isinstance(output, dict):
                result.returns_dict = True
            else:
                result.issues.append(f"Returns {type(output).__name__}, expected dict")
                result.returns_dict = False
                
        except Exception as e:
            result.execution_success = False
            result.issues.append(f"Execution error: {str(e)}")
    
    async def _test_performance(self, tool_file: Path, tool_name: str, result: ToolTestResult):
        """Test tool performance."""
        try:
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            func = getattr(module, tool_name)
            
            # Measure execution time
            start_time = asyncio.get_event_loop().time()
            
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
                
            end_time = asyncio.get_event_loop().time()
            result.performance_ms = (end_time - start_time) * 1000
            
            # Warn if too slow
            if result.performance_ms > 5000:
                result.warnings.append(f"Slow execution: {result.performance_ms:.0f}ms")
                
        except Exception as e:
            result.warnings.append(f"Performance test failed: {str(e)}")


class SafeToolLoader:
    """Safely loads and registers validated tools."""
    
    def __init__(self, tool_system):
        self.tool_system = tool_system
        self.validator = EnhancedToolValidator()
        self.logger = logging.getLogger(__name__)
        
    async def load_and_validate_tool(self, tool_file: Path, tool_name: str) -> Dict[str, Any]:
        """Load a tool only if it passes validation."""
        self.logger.info(f"Validating tool: {tool_name}")
        
        # Validate the tool
        test_result = await self.validator.validate_tool(tool_file, tool_name)
        
        if not test_result.is_valid:
            self.logger.error(f"Tool validation failed: {tool_name}")
            return {
                "success": False,
                "tool_name": tool_name,
                "validation_result": test_result.to_dict(),
                "message": f"Tool validation failed with {len(test_result.issues)} issues"
            }
        
        # Load the tool into the system
        try:
            spec = importlib.util.spec_from_file_location(tool_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            func = getattr(module, tool_name)
            
            # Register the tool
            self.tool_system.register_tool(
                name=tool_name,
                func=func,
                description=getattr(module, '__description__', f"Auto-generated tool: {tool_name}"),
                parameters=getattr(module, '__parameters__', {}),
                examples=getattr(module, '__examples__', [])
            )
            
            # Mark as AI-created
            self.tool_system.tools[tool_name].created_by_ai = True
            
            self.logger.info(f"Successfully loaded validated tool: {tool_name}")
            
            return {
                "success": True,
                "tool_name": tool_name,
                "validation_result": test_result.to_dict(),
                "message": "Tool validated and loaded successfully",
                "performance_ms": test_result.performance_ms
            }
            
        except Exception as e:
            self.logger.error(f"Error loading validated tool: {e}")
            return {
                "success": False,
                "tool_name": tool_name,
                "error": str(e),
                "message": "Tool passed validation but failed to load"
            }


# Example usage in tool_calling_system.py
async def create_and_validate_tool(tool_system, name: str, description: str, requirements: str) -> Dict[str, Any]:
    """Create a tool and validate it before adding to the system."""
    # First create the tool
    creation_result = await tool_system._create_new_tool(name, description, requirements)
    
    if not creation_result.get('success'):
        return creation_result
    
    # Then validate it
    tool_file = Path(creation_result.get('file'))
    loader = SafeToolLoader(tool_system)
    
    validation_result = await loader.load_and_validate_tool(tool_file, name)
    
    if not validation_result.get('success'):
        # Remove the invalid tool file
        tool_file.unlink()
        return validation_result
    
    return validation_result