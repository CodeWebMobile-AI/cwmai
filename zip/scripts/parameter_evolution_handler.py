#!/usr/bin/env python3
"""
Parameter Evolution Handler
===========================

Detects when tools are called with non-existent parameters and intelligently
evolves the tools to support these parameters when they make sense.
"""

import ast
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from scripts.http_ai_client import HTTPAIClient
from scripts.code_analyzer import CodeAnalyzer
from scripts.tool_calling_system import ToolCallingSystem


class ParameterEvolutionHandler:
    """Handles the evolution of tool parameters based on usage attempts."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_client = HTTPAIClient()
        self.code_analyzer = CodeAnalyzer()
        self.evolution_history = self._load_evolution_history()
        
    def _load_evolution_history(self) -> Dict[str, Any]:
        """Load history of parameter evolutions."""
        history_path = Path("parameter_evolution_history.json")
        if history_path.exists():
            try:
                return json.loads(history_path.read_text())
            except:
                return {"evolutions": [], "patterns": {}}
        return {"evolutions": [], "patterns": {}}
    
    def _save_evolution_history(self):
        """Save evolution history."""
        history_path = Path("parameter_evolution_history.json")
        history_path.write_text(json.dumps(self.evolution_history, indent=2))
    
    async def handle_parameter_mismatch(self, tool_name: str, 
                                      attempted_params: Dict[str, Any],
                                      actual_params: List[str],
                                      error_message: str) -> Dict[str, Any]:
        """
        Handle when a tool is called with parameters it doesn't support.
        
        Args:
            tool_name: Name of the tool that was called
            attempted_params: Parameters that were attempted
            actual_params: Parameters the tool actually accepts
            error_message: The error message from the failed call
            
        Returns:
            Dict with action taken and result
        """
        self.logger.info(f"Parameter mismatch detected for {tool_name}")
        self.logger.info(f"Attempted: {attempted_params}")
        self.logger.info(f"Actual: {actual_params}")
        
        # Find which parameters don't exist
        missing_params = {k: v for k, v in attempted_params.items() 
                         if k not in actual_params}
        
        if not missing_params:
            return {
                "action": "none",
                "reason": "No missing parameters detected",
                "suggestion": f"Use parameters: {actual_params}"
            }
        
        # Analyze if the missing parameters make sense
        analysis = await self._analyze_parameter_intent(
            tool_name, missing_params, actual_params
        )
        
        if analysis["makes_sense"]:
            # Attempt to evolve the tool
            evolution_result = await self._evolve_tool(
                tool_name, missing_params, analysis
            )
            
            if evolution_result["success"]:
                # Record the evolution
                self._record_evolution(tool_name, missing_params, analysis)
                
                # Reload the tool
                tool_system = ToolCallingSystem()
                tool_system._load_tools()  # Reload all tools
                
                return {
                    "action": "evolved",
                    "evolved_params": list(missing_params.keys()),
                    "reason": analysis["reason"],
                    "retry": True,
                    "message": f"Tool {tool_name} has been enhanced with parameters: {list(missing_params.keys())}"
                }
            else:
                return {
                    "action": "evolution_failed",
                    "reason": evolution_result["error"],
                    "suggestion": f"Valid parameters are: {actual_params}",
                    "attempted_params": missing_params
                }
        else:
            # Parameters don't make sense, provide guidance
            return {
                "action": "invalid_params",
                "reason": analysis["reason"],
                "suggestion": f"Valid parameters for {tool_name}: {actual_params}",
                "alternatives": analysis.get("alternatives", [])
            }
    
    async def _analyze_parameter_intent(self, tool_name: str,
                                      missing_params: Dict[str, Any],
                                      existing_params: List[str]) -> Dict[str, Any]:
        """Analyze why these parameters were requested."""
        
        # Check if we've seen similar patterns before
        pattern_key = f"{tool_name}:{','.join(sorted(missing_params.keys()))}"
        if pattern_key in self.evolution_history["patterns"]:
            cached = self.evolution_history["patterns"][pattern_key]
            if cached["occurrences"] > 3:  # If requested multiple times, likely valid
                return {
                    "makes_sense": True,
                    "reason": cached["reason"],
                    "confidence": 0.9
                }
        
        # Use AI to analyze the intent
        prompt = f"""Analyze if these parameters make sense for the tool '{tool_name}':

Tool: {tool_name}
Current parameters: {existing_params}
Requested parameters: {json.dumps(missing_params, indent=2)}

Consider:
1. Does the parameter name suggest a logical enhancement?
2. Is this a common pattern in similar tools?
3. Would adding this parameter improve the tool's functionality?
4. Are there existing parameters that already serve this purpose?

Respond with JSON:
{{
    "makes_sense": true/false,
    "reason": "explanation",
    "suggested_implementation": "how to implement if makes sense",
    "alternatives": ["existing params that might work"],
    "confidence": 0.0-1.0
}}"""

        response = await self.ai_client.generate_enhanced_response(prompt)
        try:
            analysis = json.loads(response.get("content", "{}"))
            
            # Record this pattern
            if pattern_key not in self.evolution_history["patterns"]:
                self.evolution_history["patterns"][pattern_key] = {
                    "occurrences": 0,
                    "first_seen": datetime.now().isoformat(),
                    "reason": analysis.get("reason", "")
                }
            self.evolution_history["patterns"][pattern_key]["occurrences"] += 1
            self._save_evolution_history()
            
            return analysis
        except:
            return {
                "makes_sense": False,
                "reason": "Could not analyze parameter intent",
                "confidence": 0.0
            }
    
    async def _evolve_tool(self, tool_name: str,
                          missing_params: Dict[str, Any],
                          analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve the tool to support new parameters."""
        
        # Find the tool file
        tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
        if not tool_path.exists():
            return {
                "success": False,
                "error": f"Tool file not found: {tool_path}"
            }
        
        # Read current tool code
        current_code = tool_path.read_text()
        
        # Parse the AST
        try:
            tree = ast.parse(current_code)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Could not parse tool code: {e}"
            }
        
        # Find the main function and its parameters
        tool_function = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip helper functions
                if not node.name.startswith('_') and node.name != '__init__':
                    tool_function = node
                    break
        
        if not tool_function:
            return {
                "success": False,
                "error": "Could not find tool function"
            }
        
        # Update the function with new parameters
        updater = ToolCodeUpdater()
        updated_code = await updater.add_parameters_to_tool(
            current_code,
            tool_function.name,
            missing_params,
            analysis.get("suggested_implementation", "")
        )
        
        if updated_code:
            # Backup current version
            backup_path = tool_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            backup_path.write_text(current_code)
            
            # Write updated code
            tool_path.write_text(updated_code)
            
            return {
                "success": True,
                "backup": str(backup_path),
                "updated_params": list(missing_params.keys())
            }
        else:
            return {
                "success": False,
                "error": "Failed to update tool code"
            }
    
    def _record_evolution(self, tool_name: str, 
                         params: Dict[str, Any],
                         analysis: Dict[str, Any]):
        """Record this evolution for learning."""
        evolution = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "added_params": list(params.keys()),
            "reason": analysis.get("reason", ""),
            "confidence": analysis.get("confidence", 0.0)
        }
        
        self.evolution_history["evolutions"].append(evolution)
        self._save_evolution_history()


class ToolCodeUpdater:
    """Updates tool code to add new parameters."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_client = HTTPAIClient()
    
    async def add_parameters_to_tool(self, current_code: str,
                                   function_name: str,
                                   new_params: Dict[str, Any],
                                   implementation_hint: str) -> Optional[str]:
        """Add new parameters to a tool function."""
        
        # Create the enhancement prompt
        prompt = f"""Update this tool function to support new parameters.

Current code:
```python
{current_code}
```

Function to update: {function_name}
New parameters to add: {json.dumps(new_params, indent=2)}
Implementation hint: {implementation_hint}

Requirements:
1. Add the new parameters to the function signature (with default values)
2. Update the function body to use these parameters appropriately
3. Update any docstring to document the new parameters
4. Update __parameters__ if it exists
5. Ensure backward compatibility (existing calls should still work)
6. Add proper validation for the new parameters

Return the complete updated code, not just the changes.
Ensure the code is syntactically correct and follows the existing style."""

        response = await self.ai_client.generate_enhanced_response(prompt)
        updated_code = response.get("content", "")
        
        # Extract code from response
        if "```python" in updated_code:
            code_match = re.search(r"```python\n(.*?)```", updated_code, re.DOTALL)
            if code_match:
                updated_code = code_match.group(1)
        
        # Validate the updated code
        try:
            ast.parse(updated_code)
            return updated_code
        except SyntaxError as e:
            self.logger.error(f"Generated invalid code: {e}")
            return None


class ParameterIntentAnalyzer:
    """Analyzes the intent behind parameter usage attempts."""
    
    def __init__(self):
        self.common_patterns = {
            "filter": ["search", "match", "find", "select"],
            "limit": ["max", "count", "top", "first"],
            "sort": ["order", "orderby", "sortby"],
            "format": ["output", "style", "template"],
            "verbose": ["detailed", "debug", "full"],
            "include": ["with", "add", "show"],
            "exclude": ["without", "hide", "skip"]
        }
    
    def analyze_parameter_name(self, param_name: str) -> Dict[str, Any]:
        """Analyze what a parameter name suggests about its intent."""
        
        param_lower = param_name.lower()
        
        # Check against common patterns
        for pattern, variations in self.common_patterns.items():
            if param_lower == pattern or param_lower in variations:
                return {
                    "pattern": pattern,
                    "intent": self._get_pattern_intent(pattern),
                    "common": True
                }
        
        # Analyze the name structure
        if param_lower.startswith("is_") or param_lower.startswith("has_"):
            return {
                "pattern": "boolean_flag",
                "intent": "Toggle a feature or check a condition",
                "common": True
            }
        
        if param_lower.endswith("_id") or param_lower.endswith("_name"):
            return {
                "pattern": "identifier",
                "intent": "Specify a particular item or resource",
                "common": True
            }
        
        return {
            "pattern": "custom",
            "intent": "Domain-specific parameter",
            "common": False
        }
    
    def _get_pattern_intent(self, pattern: str) -> str:
        """Get the intent description for a pattern."""
        intents = {
            "filter": "Narrow down results based on criteria",
            "limit": "Control the number of results returned",
            "sort": "Order results by a specific attribute",
            "format": "Control output presentation",
            "verbose": "Control level of detail in output",
            "include": "Add additional information to results",
            "exclude": "Remove specific information from results"
        }
        return intents.get(pattern, "Unknown intent")