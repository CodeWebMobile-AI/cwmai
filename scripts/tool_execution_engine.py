"""
Tool Execution Engine
====================

A robust, non-recursive tool execution engine that prevents infinite loops
and handles errors gracefully.
"""

import re
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field


class ExecutionState(Enum):
    """States for the execution engine."""
    INITIAL = "initial"
    EXECUTING = "executing"
    ERROR = "error"
    COMPLETE = "complete"
    WAITING_FOR_AI = "waiting_for_ai"


@dataclass
class ExecutionContext:
    """Context for a tool execution session."""
    original_request: str
    state: ExecutionState = ExecutionState.INITIAL
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    current_result: Any = None
    max_attempts: int = 5
    

class ToolExecutionEngine:
    """Non-recursive tool execution engine with proper error handling."""
    
    def __init__(self, ai_brain, tool_system, discovery_flow):
        self.ai_brain = ai_brain
        self.tool_system = tool_system
        self.discovery_flow = discovery_flow
        
    async def execute_request(self, user_request: str, initial_action: str) -> str:
        """Execute a user request starting with an initial action."""
        context = ExecutionContext(original_request=user_request)
        
        # Main execution loop - no recursion!
        while context.state != ExecutionState.COMPLETE and context.error_count < context.max_attempts:
            if context.state == ExecutionState.INITIAL:
                # Execute the initial action
                result = await self._execute_single_action(initial_action, context)
                if result:
                    context.current_result = result
                    context.state = ExecutionState.EXECUTING
                else:
                    context.state = ExecutionState.ERROR
                    
            elif context.state == ExecutionState.EXECUTING:
                # Determine next action
                next_action = await self._determine_next_action(context)
                if next_action:
                    result = await self._execute_single_action(next_action, context)
                    if result:
                        context.current_result = result
                    else:
                        context.error_count += 1
                else:
                    # No more actions, we're done
                    context.state = ExecutionState.COMPLETE
                    
            elif context.state == ExecutionState.ERROR:
                # Try to recover from error
                recovery_action = await self._get_error_recovery_action(context)
                if recovery_action:
                    result = await self._execute_single_action(recovery_action, context)
                    if result:
                        context.current_result = result
                        context.state = ExecutionState.EXECUTING
                        context.error_count = 0  # Reset error count on successful recovery
                    else:
                        context.error_count += 1
                else:
                    context.state = ExecutionState.COMPLETE
        
        # Return final result
        return self._format_final_result(context)
    
    async def _execute_single_action(self, action: str, context: ExecutionContext) -> Optional[Any]:
        """Execute a single action and update context."""
        # Parse the action
        if not action.startswith('EXECUTE_TOOL:'):
            return None
            
        tool_match = re.search(r'EXECUTE_TOOL:\s*(.+?)(?:\n|$)', action)
        if not tool_match:
            return None
            
        tool_call = tool_match.group(1).strip()
        
        # Check for invalid patterns
        if '[next tool call]' in tool_call or tool_call == 'None':
            context.execution_history.append({
                'action': action,
                'error': 'Invalid tool call pattern',
                'result': None
            })
            return None
        
        # Extract tool name
        tool_name_match = re.match(r'(\w+)\s*\(', tool_call)
        tool_name = tool_name_match.group(1) if tool_name_match else None
        
        # Execute the tool
        try:
            result = await self._execute_tool_call(tool_call)
            
            # Record successful execution
            context.execution_history.append({
                'action': action,
                'tool_name': tool_name,
                'tool_call': tool_call,
                'result': result,
                'success': True
            })
            
            return result
            
        except Exception as e:
            # Record failed execution
            context.execution_history.append({
                'action': action,
                'tool_name': tool_name,
                'tool_call': tool_call,
                'error': str(e),
                'success': False
            })
            context.state = ExecutionState.ERROR
            return None
    
    async def _execute_tool_call(self, tool_call: str) -> Any:
        """Execute a tool call and return the result."""
        # Parse tool name and parameters
        match = re.match(r'(\w+)\((.*)\)', tool_call)
        if not match:
            raise ValueError(f"Invalid tool call format: {tool_call}")
        
        tool_name = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters
        params = {}
        if params_str:
            # Simple parameter parsing
            param_pattern = r'(\w+)\s*=\s*[\'"]([^\'"]*)[\'""]'
            for key, value in re.findall(param_pattern, params_str):
                params[key] = value
        
        # Call the tool
        result = await self.tool_system.call_tool(tool_name, **params)
        
        if result.get('success'):
            return result.get('result')
        else:
            raise Exception(result.get('error', 'Unknown error'))
    
    async def _determine_next_action(self, context: ExecutionContext) -> Optional[str]:
        """Determine the next action based on context."""
        # First check if we have a flow-based next action
        last_execution = context.execution_history[-1] if context.execution_history else None
        
        if last_execution and last_execution.get('tool_name'):
            # Use discovery flow to determine next step
            next_tool_call = self.discovery_flow.determine_next_tool_call(
                last_execution['tool_name'],
                context.current_result,
                context.original_request
            )
            
            if next_tool_call:
                return next_tool_call
        
        # Ask AI for next step
        prompt = f"""Based on the execution so far, determine the next step.

Original request: {context.original_request}

Last result: {json.dumps(context.current_result) if isinstance(context.current_result, (dict, list)) else str(context.current_result)}

Options:
1. If the request is complete, respond: COMPLETE
2. If another tool is needed, respond: EXECUTE_TOOL: tool_name(params)

Respond with ONLY one of the above options."""

        try:
            response = await self.ai_brain.execute_capability('intelligent_analysis', {
                'prompt': prompt,
                'max_length': 200,
                'model': 'gemini'
            })
            
            ai_response = response.get('result', '').strip()
            
            if 'COMPLETE' in ai_response:
                return None  # Signal completion
            elif 'EXECUTE_TOOL:' in ai_response:
                return ai_response
            else:
                return None  # Can't determine next action, complete
                
        except Exception:
            return None  # Error getting AI response, complete
    
    async def _get_error_recovery_action(self, context: ExecutionContext) -> Optional[str]:
        """Get an action to recover from an error."""
        last_error = None
        for entry in reversed(context.execution_history):
            if not entry.get('success', True):
                last_error = entry
                break
        
        if not last_error:
            return None
        
        # Check if we've already tried to create this tool
        failed_tool = last_error.get('tool_name')
        failed_tools = [e.get('tool_name') for e in context.execution_history if not e.get('success', True)]
        
        # If we've failed with this tool multiple times, don't retry
        if failed_tools.count(failed_tool) >= 2:
            return None
        
        # For other errors, ask AI for alternative
        prompt = f"""The previous action failed. Suggest an alternative approach.

Original request: {context.original_request}

Failed action: {last_error.get('tool_call')}
Error: {last_error.get('error')}

Suggest ONE alternative action in the format:
EXECUTE_TOOL: tool_name(params)

Or if no alternative exists, respond: COMPLETE"""

        try:
            response = await self.ai_brain.execute_capability('intelligent_analysis', {
                'prompt': prompt,
                'max_length': 200,
                'model': 'gemini'
            })
            
            ai_response = response.get('result', '').strip()
            
            if 'COMPLETE' in ai_response:
                return None
            elif 'EXECUTE_TOOL:' in ai_response:
                return ai_response
            else:
                return None
                
        except Exception:
            return None
    
    def _format_final_result(self, context: ExecutionContext) -> str:
        """Format the final result for the user."""
        if context.error_count >= context.max_attempts:
            return "I encountered too many errors while trying to complete your request. The system may need maintenance."
        
        if context.current_result is None:
            return "I couldn't complete your request due to errors."
        
        # Format based on result type
        if isinstance(context.current_result, dict):
            if 'error' in context.current_result:
                return f"I encountered an error: {context.current_result['error']}"
            else:
                return json.dumps(context.current_result, indent=2)
        elif isinstance(context.current_result, list):
            return json.dumps(context.current_result, indent=2)
        else:
            return str(context.current_result)