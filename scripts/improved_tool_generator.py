#!/usr/bin/env python3
"""
Improved Tool Generator with Enhanced Context and Validation
Provides comprehensive context to AI for generating working tools
"""

import json
import asyncio
import ast
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from scripts.http_ai_client import HTTPAIClient
from scripts.enhanced_tool_validation import EnhancedToolValidator
from scripts.tool_generation_templates import ToolGenerationTemplates


class ImprovedToolGenerator:
    """Generate high-quality tools with comprehensive context."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.validator = EnhancedToolValidator()
        self.templates = ToolGenerationTemplates()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load working tool examples
        self.tool_examples = self._load_working_examples()
        
    def _load_working_examples(self) -> List[Dict[str, str]]:
        """Load examples of working tools for context."""
        examples = []
        
        # Example 1: Simple counter tool
        examples.append({
            "name": "count_files",
            "code": '''"""
AI-Generated Tool: count_files
Description: Count files in a directory
Generated: 2025-06-15
Requirements: Count all files in specified directory
"""

import os
from pathlib import Path
from typing import Dict, Any

__description__ = "Count files in a directory"
__parameters__ = {
    "directory": {
        "type": "string",
        "description": "Directory path to count files in",
        "required": False,
        "default": "."
    }
}
__examples__ = [
    {"description": "Count files in current directory", "code": "await count_files()"},
    {"description": "Count files in scripts", "code": "await count_files(directory='scripts')"}
]


async def count_files(**kwargs) -> Dict[str, Any]:
    """Count all files in specified directory."""
    directory = kwargs.get('directory', '.')
    
    try:
        path = Path(directory)
        if not path.exists():
            return {"error": f"Directory not found: {directory}"}
        
        if not path.is_dir():
            return {"error": f"Not a directory: {directory}"}
        
        file_count = 0
        dir_count = 0
        
        for item in path.iterdir():
            if item.is_file():
                file_count += 1
            elif item.is_dir():
                dir_count += 1
        
        return {
            "directory": str(path.absolute()),
            "file_count": file_count,
            "dir_count": dir_count,
            "total_items": file_count + dir_count,
            "summary": f"Found {file_count} files and {dir_count} directories"
        }
        
    except Exception as e:
        return {"error": f"Error counting files: {str(e)}"}
'''
        })
        
        # Example 2: Tool with state management
        examples.append({
            "name": "get_system_info",
            "code": '''"""
AI-Generated Tool: get_system_info
Description: Get system information and status
Generated: 2025-06-15
Requirements: Retrieve system status from state
"""

import platform
from datetime import datetime
from typing import Dict, Any

from scripts.state_manager import StateManager

__description__ = "Get system information and status"
__parameters_ = {}
__examples__ = [
    {"description": "Get system info", "code": "await get_system_info()"}
]


async def get_system_info(**kwargs) -> Dict[str, Any]:
    """Get system information and status."""
    try:
        # Initialize state manager
        state_manager = StateManager()
        state = state_manager.load_state()
        
        # Get system info
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Get state info
        projects = state.get('projects', {})
        tasks = state.get('active_tasks', [])
        
        return {
            "system": system_info,
            "projects_count": len(projects),
            "active_tasks": len(tasks),
            "healthy": state.get('healthy', False),
            "summary": f"System running {system_info['platform']} with {len(projects)} projects"
        }
        
    except Exception as e:
        return {"error": f"Error getting system info: {str(e)}"}
'''
        })
        
        return examples
    
    def _get_common_imports(self) -> str:
        """Get commonly needed imports based on tool requirements."""
        # Now delegated to ToolGenerationTemplates for dynamic discovery
        return self.templates.get_import_context()

    def _get_validation_rules(self) -> str:
        """Get validation rules for generated tools."""
        return """
VALIDATION RULES:
1. Always validate input parameters
2. Return errors as {"error": "message"} instead of raising exceptions
3. Include type hints for all parameters and return values
4. Use try-except blocks for error handling
5. Return dictionaries with descriptive keys
6. Include a 'summary' key for human-readable results
7. Test edge cases (empty inputs, missing files, etc.)
"""

    async def generate_tool(self, name: str, description: str, requirements: str) -> Dict[str, Any]:
        """Generate a tool with comprehensive context."""
        
        # Build enhanced prompt with examples and context
        prompt = f"""You are an expert Python developer creating a tool for the CWMAI system.

TASK: Create a Python module for a tool named '{name}'.

DESCRIPTION: {description}

REQUIREMENTS: {requirements}

WORKING EXAMPLES FOR REFERENCE:
{json.dumps(self.tool_examples, indent=2)}

{self._get_validation_rules()}

{self.templates.get_import_context()}

MODULE DISCOVERY SUMMARY:
{self.templates.get_available_modules_summary()}

CRITICAL REQUIREMENTS:
1. Module must have proper docstring
2. Define __description__, __parameters__, __examples__ at module level
3. Main function must be async and named '{name}'
4. Function takes **kwargs and returns Dict[str, Any]
5. Include comprehensive error handling
6. Validate all inputs
7. Return meaningful error messages
8. Include 'summary' in successful responses

PARAMETER DEFINITION FORMAT:
__parameters__ = {{
    "param_name": {{
        "type": "string|integer|boolean|list|dict",
        "description": "Clear description",
        "required": true|false,
        "default": "default_value"  # if not required
    }}
}}

AVOID THESE COMMON ERRORS:
- Missing imports (check all used modules)
- Syntax errors in string literals
- Undefined variables
- Missing error handling
- Returning non-dict values
- Using 'self' parameter (tools are functions, not methods)

Generate ONLY the Python code for the module. No explanations.
"""

        # Generate with prefill for better quality
        response = await self.ai_client.generate_enhanced_response(
            prompt,
            prefill="```python\n#!/usr/bin/env python3\n"
        )
        
        if not response.get('content'):
            return {"success": False, "error": "No response from AI"}
        
        # Extract and clean code
        code = self._extract_code(response['content'])
        
        # Validate the generated code
        validation_result = await self._validate_generated_code(name, code)
        
        if not validation_result['valid']:
            # Try to fix common issues
            code = self._auto_fix_common_issues(code, validation_result['issues'])
            
            # Re-validate
            validation_result = await self._validate_generated_code(name, code)
        
        return {
            "success": validation_result['valid'],
            "code": code,
            "validation": validation_result,
            "name": name
        }
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from AI response."""
        # Remove markdown code blocks
        if '```python' in response:
            code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            code = response.split('```')[1].split('```')[0]
        else:
            code = response
        
        return code.strip()
    
    async def _validate_generated_code(self, name: str, code: str) -> Dict[str, Any]:
        """Validate generated code."""
        issues = []
        
        try:
            # Check syntax
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            return {"valid": False, "issues": issues}
        
        # Check for required elements
        if f"async def {name}" not in code:
            issues.append(f"Missing async function named {name}")
        
        if "__description__" not in code:
            issues.append("Missing __description__ variable")
        
        if "__parameters__" not in code:
            issues.append("Missing __parameters__ variable")
        
        if "__examples__" not in code:
            issues.append("Missing __examples__ variable")
        
        # Check for common import issues
        if "StateManager" in code and "from scripts.state_manager import StateManager" not in code:
            issues.append("Using StateManager without importing it")
        
        # Check for error handling
        if "try:" not in code:
            issues.append("No try-except blocks for error handling")
        
        if 'return {"error"' not in code and 'return None' not in code:
            issues.append("No error return patterns found")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def _auto_fix_common_issues(self, code: str, issues: List[str]) -> str:
        """Attempt to fix common issues in generated code."""
        fixed_code = code
        
        # Fix missing imports
        for issue in issues:
            if "StateManager without importing" in issue:
                import_line = "from scripts.state_manager import StateManager\n"
                if import_line not in fixed_code:
                    # Add after other imports
                    lines = fixed_code.split('\n')
                    import_index = 0
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            import_index = i + 1
                    lines.insert(import_index, import_line.strip())
                    fixed_code = '\n'.join(lines)
        
        return fixed_code


async def demonstrate_improved_generation():
    """Demonstrate the improved tool generation."""
    generator = ImprovedToolGenerator()
    
    # Test generating a tool
    result = await generator.generate_tool(
        name="find_large_files",
        description="Find all files larger than a specified size",
        requirements="Search for files exceeding a size threshold in bytes"
    )
    
    if result['success']:
        print("✓ Tool generated successfully!")
        print("\nGenerated code:")
        print(result['code'])
    else:
        print("✗ Tool generation failed:")
        print(result['validation']['issues'])


if __name__ == "__main__":
    asyncio.run(demonstrate_improved_generation())