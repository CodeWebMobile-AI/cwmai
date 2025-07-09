#!/usr/bin/env python3
"""
Enhanced Conversational AI Assistant with Claude Code-like behavior
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

from scripts.ai_brain import IntelligentAIBrain
from scripts.conversational_ai_logger import log_system_message

class ConversationalAIAssistantV2:
    """Claude Code-like assistant that continues searching until it finds what the user needs"""
    
    def __init__(self):
        self.ai_brain = IntelligentAIBrain()
        self.conversation_history = []
        self.max_iterations = 10  # Prevent infinite loops
        
    async def process_query(self, user_input: str) -> str:
        """Process user query with iterative tool execution"""
        
        # Get available tools
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        available_tools = tool_system.get_tools_for_ai()
        tool_names = [t['name'] for t in available_tools]
        
        # Build initial context
        context = {
            'original_request': user_input,
            'iteration': 0,
            'findings': [],
            'tools_used': [],
            'satisfied': False
        }
        
        # Add conversation history
        history = self._format_recent_history()
        
        # Initial response
        current_response = await self._get_ai_decision(user_input, context, available_tools, history)
        
        # Iterative execution loop - keep going until satisfied or max iterations
        while context['iteration'] < self.max_iterations and not context['satisfied']:
            context['iteration'] += 1
            
            # Extract and execute tool calls from the response
            tool_calls = self._extract_tool_calls(current_response, tool_names)
            
            if tool_calls:
                # Execute all tool calls
                for tool_name, params in tool_calls:
                    try:
                        result = await self._execute_tool(tool_name, params)
                        context['findings'].append({
                            'tool': tool_name,
                            'params': params,
                            'result': result
                        })
                        context['tools_used'].append(tool_name)
                    except Exception as e:
                        context['findings'].append({
                            'tool': tool_name,
                            'error': str(e)
                        })
                
                # Ask AI if it needs to continue searching
                current_response = await self._check_if_satisfied(user_input, context, available_tools)
            else:
                # No more tools to call, we're done
                context['satisfied'] = True
        
        # Generate final response
        final_response = await self._generate_final_response(user_input, context)
        
        # Save to history
        self.conversation_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_input': user_input,
            'response': final_response,
            'context': context
        })
        
        return final_response
    
    async def _get_ai_decision(self, user_input: str, context: Dict, tools: List, history: str) -> str:
        """Get AI's decision on what to do next"""
        
        prompt = f"""
{history}

Current request: {user_input}

You are a helpful AI assistant. Your goal is to find and provide exactly what the user is asking for.
Don't stop at surface-level information - dig deeper to get the complete answer.

What we know so far:
- Iteration: {context['iteration']}
- Tools used: {', '.join(context['tools_used']) if context['tools_used'] else 'None yet'}
- Findings: {json.dumps(context['findings'], indent=2) if context['findings'] else 'None yet'}

Available tools (first 30):
{self._format_tools(tools[:30])}

What would you like to do next? If you need more information to answer the user's question fully, 
call the appropriate tools. Think step by step about what information is still missing.
"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return response.get('result', '')
    
    async def _check_if_satisfied(self, user_input: str, context: Dict, tools: List) -> str:
        """Check if we have enough information to answer the user"""
        
        prompt = f"""
Original user request: {user_input}

Information gathered so far:
{json.dumps(context['findings'], indent=2)}

Do you have enough information to fully answer the user's request? 
- If YES: Set satisfied=true and provide a comprehensive answer
- If NO: Explain what's still missing and call more tools to find it

Think carefully: Does the user want just a summary, or do they want specific details?
For example, if they asked for "the 13 tasks", they want to see the actual list of 13 tasks, 
not just know that 13 tasks exist.

Available tools: {', '.join([t['name'] for t in tools[:20]])}

Continue searching if needed. Be thorough.
"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        result = response.get('result', '')
        
        # Check if AI indicates satisfaction
        if 'satisfied=true' in result.lower() or 'i have enough information' in result.lower():
            context['satisfied'] = True
        
        return result
    
    async def _generate_final_response(self, user_input: str, context: Dict) -> str:
        """Generate the final response based on all findings"""
        
        prompt = f"""
The user asked: {user_input}

After searching through {context['iteration']} iterations using these tools: {', '.join(context['tools_used'])}

Here's everything I found:
{json.dumps(context['findings'], indent=2)}

Please provide a natural, conversational response that directly answers the user's question.
Focus on what they specifically asked for. Be concise but complete.
"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return response.get('result', 'I completed my search but couldn\'t find the specific information requested.')
    
    def _extract_tool_calls(self, response: str, valid_tools: List[str]) -> List[Tuple[str, Dict]]:
        """Extract tool calls from AI response"""
        tool_calls = []
        
        # Pattern 1: tool_name() or tool_name(params)
        pattern = r'\b(\w+)\s*\((.*?)\)'
        
        for match in re.finditer(pattern, response):
            tool_name = match.group(1)
            params_str = match.group(2).strip()
            
            if tool_name in valid_tools:
                # Parse parameters
                params = {}
                if params_str:
                    try:
                        # Try to parse as dict
                        if params_str.startswith('{'):
                            params = json.loads(params_str)
                        else:
                            # Parse key=value pairs
                            for param in params_str.split(','):
                                if '=' in param:
                                    key, value = param.split('=', 1)
                                    key = key.strip().strip('"\'')
                                    value = value.strip().strip('"\'')
                                    params[key] = value
                    except:
                        pass
                
                tool_calls.append((tool_name, params))
        
        return tool_calls
    
    async def _execute_tool(self, tool_name: str, params: Dict) -> Any:
        """Execute a tool and return results"""
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        
        result = await tool_system.call_tool(tool_name, **params)
        return result.get('result', result)
    
    def _format_tools(self, tools: List[Dict]) -> str:
        """Format tools for display"""
        lines = []
        for tool in tools:
            lines.append(f"- {tool['name']}: {tool.get('description', 'No description')}")
        return '\n'.join(lines)
    
    def _format_recent_history(self) -> str:
        """Format recent conversation history"""
        if not self.conversation_history:
            return ""
        
        recent = self.conversation_history[-3:]
        lines = ["Recent conversation:"]
        
        for turn in recent:
            lines.append(f"User: {turn['user_input']}")
            lines.append(f"Assistant: {turn['response'][:200]}...")
            lines.append("")
        
        return '\n'.join(lines)


# Test the new approach
async def test_v2():
    assistant = ConversationalAIAssistantV2()
    
    # This should trigger multiple tool calls
    response = await assistant.process_query("List all 13 tasks mentioned in the system status")
    print("Response:", response)


if __name__ == "__main__":
    asyncio.run(test_v2())