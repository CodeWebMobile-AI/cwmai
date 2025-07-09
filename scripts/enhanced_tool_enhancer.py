#!/usr/bin/env python3
"""
Enhanced Tool Enhancer
======================

Reimplementation of _enhance_tool that actually updates tool code files
with proper context and intelligence.
"""

import ast
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from scripts.http_ai_client import HTTPAIClient
from scripts.code_analyzer import CodeAnalyzer
from scripts.intelligent_tool_generator import IntelligentToolGenerator


class EnhancedToolEnhancer:
    """Enhanced tool enhancement with actual code modification capabilities."""
    
    def __init__(self):
        self.ai_client = HTTPAIClient()
        self.code_analyzer = CodeAnalyzer()
        self.tool_generator = IntelligentToolGenerator()
        
    async def enhance_tool(self, tool_name: str, 
                          enhancement_request: str,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance a tool based on request with full context.
        
        Args:
            tool_name: Name of the tool to enhance
            enhancement_request: What enhancement is needed
            context: Additional context (like parameter mismatches, usage patterns)
            
        Returns:
            Dict with enhancement results
        """
        # Find the tool file
        tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
        if not tool_path.exists():
            return {
                "success": False,
                "error": f"Tool {tool_name} not found at {tool_path}"
            }
        
        # Read current tool code
        current_code = tool_path.read_text()
        
        # Analyze current tool structure
        analysis = await self._analyze_tool_structure(current_code, tool_name)
        
        # Gather enhancement context
        enhancement_context = await self._gather_enhancement_context(
            tool_name, enhancement_request, analysis, context
        )
        
        # Generate enhanced code
        enhanced_code = await self._generate_enhanced_code(
            current_code, enhancement_request, enhancement_context
        )
        
        if not enhanced_code:
            return {
                "success": False,
                "error": "Failed to generate enhanced code"
            }
        
        # Validate the enhanced code
        validation = await self._validate_enhanced_code(enhanced_code, current_code)
        
        if validation["valid"]:
            # Backup and update
            backup_path = self._backup_tool(tool_path, current_code)
            tool_path.write_text(enhanced_code)
            
            return {
                "success": True,
                "backup": str(backup_path),
                "changes": validation["changes"],
                "message": f"Tool {tool_name} enhanced successfully"
            }
        else:
            return {
                "success": False,
                "error": "Enhanced code validation failed",
                "issues": validation["issues"]
            }
    
    async def _analyze_tool_structure(self, code: str, tool_name: str) -> Dict[str, Any]:
        """Analyze the current structure of a tool."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"error": "Could not parse tool code"}
        
        structure = {
            "imports": [],
            "functions": [],
            "classes": [],
            "main_function": None,
            "has_parameters_metadata": False,
            "has_description": False,
            "async_functions": []
        }
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    structure["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    structure["imports"].append(f"{module}.{alias.name}")
        
        # Find functions and classes
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "has_docstring": ast.get_docstring(node) is not None
                }
                structure["functions"].append(func_info)
                
                if isinstance(node, ast.AsyncFunctionDef):
                    structure["async_functions"].append(node.name)
                
                # Likely the main function if not private
                if not node.name.startswith('_') and node.name == tool_name:
                    structure["main_function"] = func_info
            
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append({
                    "name": node.name,
                    "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                })
        
        # Check for metadata
        if "__parameters__" in code:
            structure["has_parameters_metadata"] = True
        if "__description__" in code:
            structure["has_description"] = True
        
        return structure
    
    async def _gather_enhancement_context(self, tool_name: str,
                                        enhancement_request: str,
                                        current_analysis: Dict[str, Any],
                                        additional_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Gather comprehensive context for the enhancement."""
        
        context = {
            "tool_name": tool_name,
            "current_structure": current_analysis,
            "enhancement_request": enhancement_request,
            "similar_tools": [],
            "usage_patterns": [],
            "best_practices": []
        }
        
        # Add any additional context provided
        if additional_context:
            context.update(additional_context)
        
        # Find similar tools for pattern reference
        similar_tools = await self._find_similar_enhanced_tools(tool_name)
        context["similar_tools"] = similar_tools
        
        # Get usage patterns if available
        usage_patterns = await self._get_tool_usage_patterns(tool_name)
        context["usage_patterns"] = usage_patterns
        
        # Determine best practices for this enhancement
        if "parameter" in enhancement_request.lower():
            context["best_practices"].append("Add parameters with sensible defaults")
            context["best_practices"].append("Update docstring with parameter documentation")
            context["best_practices"].append("Validate parameter values")
            context["best_practices"].append("Maintain backward compatibility")
        
        return context
    
    async def _generate_enhanced_code(self, current_code: str,
                                    enhancement_request: str,
                                    context: Dict[str, Any]) -> Optional[str]:
        """Generate the enhanced tool code."""
        
        # Build comprehensive prompt
        prompt = f"""Enhance this tool based on the request. You must return complete, working Python code.

Current Tool Code:
```python
{current_code}
```

Enhancement Request: {enhancement_request}

Context:
- Tool Name: {context['tool_name']}
- Current Structure: {json.dumps(context['current_structure'], indent=2)}

Best Practices to Follow:
{chr(10).join(f"- {bp}" for bp in context.get('best_practices', []))}

Requirements:
1. Return the COMPLETE enhanced tool code (not just changes)
2. Preserve all existing functionality
3. Add clear comments for new functionality
4. Update docstrings appropriately
5. Update __parameters__ metadata if adding parameters
6. Ensure proper error handling
7. Follow the existing code style
8. Make sure all imports are included

{self._get_enhancement_specific_instructions(enhancement_request, context)}

Return only the Python code, no explanations."""

        response = await self.ai_client.generate_enhanced_response(prompt)
        enhanced_code = response.get("content", "")
        
        # Extract code if wrapped in markdown
        if "```python" in enhanced_code:
            match = re.search(r"```python\n(.*?)```", enhanced_code, re.DOTALL)
            if match:
                enhanced_code = match.group(1)
        
        return enhanced_code.strip()
    
    def _get_enhancement_specific_instructions(self, request: str, context: Dict[str, Any]) -> str:
        """Get specific instructions based on enhancement type."""
        
        instructions = []
        
        # Parameter addition
        if "parameter" in request.lower() or context.get("missing_params"):
            params = context.get("missing_params", {})
            instructions.append(f"""
For adding parameters:
- Add new parameters to the function signature with appropriate defaults
- Update the function logic to use these parameters
- Add parameter validation where needed
- Update __parameters__ dictionary if it exists
- Document each parameter in the docstring

New parameters to add: {json.dumps(params, indent=2)}
""")
        
        # Performance enhancement
        if "performance" in request.lower() or "optimize" in request.lower():
            instructions.append("""
For performance optimization:
- Add caching where appropriate
- Use async operations if beneficial
- Optimize loops and data structures
- Add batch processing if applicable
""")
        
        # Feature addition
        if "feature" in request.lower() or "capability" in request.lower():
            instructions.append("""
For new features:
- Implement the feature as a separate function if complex
- Integrate smoothly with existing functionality
- Add appropriate error handling
- Consider edge cases
""")
        
        return "\n".join(instructions)
    
    async def _validate_enhanced_code(self, enhanced_code: str, 
                                    original_code: str) -> Dict[str, Any]:
        """Validate the enhanced code."""
        
        validation = {
            "valid": True,
            "issues": [],
            "changes": []
        }
        
        # 1. Syntax check
        try:
            ast.parse(enhanced_code)
        except SyntaxError as e:
            validation["valid"] = False
            validation["issues"].append(f"Syntax error: {e}")
            return validation
        
        # 2. Compare structures
        original_structure = await self._analyze_tool_structure(original_code, "")
        enhanced_structure = await self._analyze_tool_structure(enhanced_code, "")
        
        # Check that main function still exists
        if original_structure.get("main_function") and not enhanced_structure.get("main_function"):
            validation["valid"] = False
            validation["issues"].append("Main function was removed")
        
        # 3. Identify changes
        # New functions
        original_funcs = {f["name"] for f in original_structure["functions"]}
        enhanced_funcs = {f["name"] for f in enhanced_structure["functions"]}
        new_funcs = enhanced_funcs - original_funcs
        if new_funcs:
            validation["changes"].append(f"Added functions: {list(new_funcs)}")
        
        # New parameters
        if original_structure.get("main_function") and enhanced_structure.get("main_function"):
            original_args = set(original_structure["main_function"]["args"])
            enhanced_args = set(enhanced_structure["main_function"]["args"])
            new_args = enhanced_args - original_args
            if new_args:
                validation["changes"].append(f"Added parameters: {list(new_args)}")
        
        # 4. Check for required patterns
        if "__description__" not in enhanced_code:
            validation["issues"].append("Missing __description__ metadata")
        
        # 5. Check imports are preserved
        for imp in original_structure["imports"]:
            if imp not in enhanced_structure["imports"]:
                validation["issues"].append(f"Lost import: {imp}")
        
        return validation
    
    async def _find_similar_enhanced_tools(self, tool_name: str) -> List[Dict[str, Any]]:
        """Find tools that have been similarly enhanced."""
        similar = []
        
        # Check enhancement history
        history_path = Path("tool_enhancement_history.json")
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
                for enhancement in history.get("enhancements", []):
                    if enhancement.get("type") == "parameter_addition":
                        similar.append({
                            "tool": enhancement["tool"],
                            "enhancement": enhancement["description"],
                            "date": enhancement["date"]
                        })
            except:
                pass
        
        return similar[:5]  # Return top 5
    
    async def _get_tool_usage_patterns(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get usage patterns for the tool."""
        patterns = []
        
        # Check tool usage logs if available
        usage_log_path = Path(f"logs/tool_usage/{tool_name}.log")
        if usage_log_path.exists():
            # Parse usage patterns
            # This is a placeholder - actual implementation would parse logs
            pass
        
        return patterns
    
    def _backup_tool(self, tool_path: Path, current_code: str) -> Path:
        """Backup the current tool before modification."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path("scripts/custom_tools/backups")
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / f"{tool_path.stem}_backup_{timestamp}.py"
        backup_path.write_text(current_code)
        
        # Also save enhancement record
        record = {
            "tool": tool_path.stem,
            "timestamp": datetime.now().isoformat(),
            "backup": str(backup_path)
        }
        
        history_path = Path("tool_enhancement_history.json")
        if history_path.exists():
            history = json.loads(history_path.read_text())
        else:
            history = {"enhancements": []}
        
        history["enhancements"].append(record)
        history_path.write_text(json.dumps(history, indent=2))
        
        return backup_path