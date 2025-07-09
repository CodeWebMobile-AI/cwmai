"""
Focused Assistant
=================

A minimal, focused approach to AI assistance that maintains context
only for the current task.
"""

import json
import re
from typing import Dict, Any, Optional, List
from scripts.tool_calling_system import ToolCallingSystem
from scripts.ai_brain import IntelligentAIBrain


class FocusedAssistant:
    """An AI assistant that maintains focus on the current task."""
    
    def __init__(self):
        self.tool_system = ToolCallingSystem()
        self.ai_brain = IntelligentAIBrain()
        self.current_task_context = {
            'user_request': None,
            'last_tool': None,
            'last_result': None,
            'steps_taken': []
        }
    
    async def handle_request(self, user_input: str) -> str:
        """Handle a user request with focused context."""
        # Reset context for new request
        self.current_task_context = {
            'user_request': user_input,
            'last_tool': None,
            'last_result': None,
            'steps_taken': []
        }
        
        # Check if user is asking for something that needs tool discovery
        if self._needs_tool_discovery(user_input):
            return await self._execute_tool_discovery_flow(user_input)
        else:
            # Regular tool execution
            return await self._execute_single_step(user_input)
    
    def _needs_tool_discovery(self, user_input: str) -> bool:
        """Check if request needs tool discovery."""
        discovery_keywords = [
            'create a tool', 'check if', 'analyze', 'find', 
            'verify', 'ensure', 'validate', 'examine'
        ]
        return any(keyword in user_input.lower() for keyword in discovery_keywords)
    
    async def _execute_tool_discovery_flow(self, user_input: str) -> str:
        """Execute the tool discovery flow."""
        # Step 1: Search for related code
        keywords = self._extract_keywords(user_input)
        
        prompt = f"""User wants to: {user_input}

Step 1: Search for related code.
EXECUTE_TOOL: grep_files(pattern='{keywords[0]}', file_pattern='*.py')"""
        
        # Get AI response
        response = await self.ai_brain.generate_enhanced_response(prompt)
        tool_call = self._extract_tool_call(response)
        
        if tool_call:
            # Execute grep
            grep_result = await self._execute_tool(tool_call)
            self.current_task_context['last_result'] = grep_result
            
            if grep_result.get('success') and grep_result.get('results'):
                # Step 2: Read the most relevant file
                first_file = grep_result['results'][0]['file']
                
                prompt = f"""Found {len(grep_result['results'])} files containing '{keywords[0]}'.
Most relevant file: {first_file}

Step 2: Read this file to understand the pattern.
EXECUTE_TOOL: read_file(file_path='{first_file}')"""
                
                response = await self.ai_brain.generate_enhanced_response(prompt)
                tool_call = self._extract_tool_call(response)
                
                if tool_call:
                    # Execute read
                    read_result = await self._execute_tool(tool_call)
                    
                    # Step 3: Create the tool
                    tool_name = self._generate_tool_name(user_input)
                    
                    prompt = f"""Based on the code patterns found, create a tool.

User request: {user_input}
Tool name: {tool_name}

Step 3: Create the tool.
EXECUTE_TOOL: create_tool_from_search(tool_name='{tool_name}', description='{user_input}')"""
                    
                    response = await self.ai_brain.generate_enhanced_response(prompt)
                    tool_call = self._extract_tool_call(response)
                    
                    if tool_call:
                        result = await self._execute_tool(tool_call)
                        return f"Created tool '{tool_name}' successfully. The tool is now available for use."
        
        return "Could not complete tool discovery flow."
    
    async def _execute_single_step(self, user_input: str) -> str:
        """Execute a single step based on user input."""
        prompt = f"""User request: {user_input}

Available tools: {', '.join(self.tool_system.get_available_tools()[:20])}

Respond with ONLY a tool execution command:
EXECUTE_TOOL: tool_name(param1='value1', param2='value2')"""
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        tool_call = self._extract_tool_call(response)
        
        if tool_call:
            result = await self._execute_tool(tool_call)
            return json.dumps(result, indent=2)
        
        return "Could not determine appropriate action."
    
    def _extract_tool_call(self, response: str) -> Optional[str]:
        """Extract tool call from AI response."""
        match = re.search(r'EXECUTE_TOOL:\s*(.+?)(?:\n|$)', str(response))
        if match:
            return match.group(1).strip()
        return None
    
    async def _execute_tool(self, tool_call: str) -> Dict[str, Any]:
        """Execute a tool call."""
        match = re.match(r'(\w+)\((.*)\)', tool_call)
        if not match:
            return {"success": False, "error": "Invalid tool call format"}
        
        tool_name = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters
        params = {}
        if params_str:
            param_pattern = r'(\w+)=[\'"]([^\'"]*)[\'"]'
            for match in re.finditer(param_pattern, params_str):
                params[match.group(1)] = match.group(2)
        
        # Execute tool
        return await self.tool_system.call_tool(tool_name, **params)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from user input."""
        # Simple keyword extraction
        words = text.lower().split()
        keywords = []
        
        # Look for important words
        important_words = ['architecture', 'quality', 'check', 'verify', 'analyze', 'ensure']
        for word in words:
            if len(word) > 4 and word not in ['create', 'would', 'should', 'could']:
                keywords.append(word)
        
        return keywords[:3] if keywords else ['code', 'pattern', 'tool']
    
    def _generate_tool_name(self, user_input: str) -> str:
        """Generate a tool name from user input."""
        # Extract key words
        words = re.findall(r'\w+', user_input.lower())
        
        # Filter out common words
        skip_words = {'a', 'an', 'the', 'to', 'for', 'that', 'this', 'with', 'from'}
        key_words = [w for w in words if w not in skip_words and len(w) > 2]
        
        # Take first 3-4 key words
        tool_name = '_'.join(key_words[:4])
        
        return tool_name or 'custom_tool'