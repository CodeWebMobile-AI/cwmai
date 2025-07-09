#!/usr/bin/env python3
"""
Claude Code-like Conversational AI Assistant
Continues searching until it finds what the user actually wants
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

from scripts.ai_brain import IntelligentAIBrain
from scripts.conversational_ai_logger import log_system_message

class ConversationalAIAssistantV3:
    """Assistant that behaves like Claude Code - keeps searching until it finds the answer"""
    
    def __init__(self):
        self.ai_brain = IntelligentAIBrain()
        self.conversation_history = []
        self.max_iterations = 10
        
    async def process_query(self, user_input: str) -> str:
        """Process query with Claude Code-like persistence"""
        
        # Get available tools
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        self.tool_system = tool_system
        available_tools = tool_system.get_tools_for_ai()
        
        # Build context
        context = {
            'user_input': user_input,
            'tools_available': [t['name'] for t in available_tools],
            'execution_history': [],
            'current_understanding': '',
            'iteration': 0
        }
        
        # Keep searching until we have a complete answer
        while context['iteration'] < self.max_iterations:
            context['iteration'] += 1
            
            # Get AI's next action
            next_action = await self._get_next_action(context, available_tools)
            
            # Execute the action
            if 'FINAL_ANSWER:' in next_action:
                # AI is satisfied with the answer
                return next_action.replace('FINAL_ANSWER:', '').strip()
            
            # Extract and execute tools
            tool_executions = await self._execute_tools_from_response(next_action, context['tools_available'])
            
            if tool_executions:
                context['execution_history'].extend(tool_executions)
                context['current_understanding'] = await self._summarize_findings(context)
            else:
                # No tools executed, but AI didn't give final answer
                # Force it to try harder
                context['current_understanding'] = "No progress made. Need to try different tools."
        
        # Max iterations reached
        return await self._generate_best_answer(context)
    
    async def _get_next_action(self, context: Dict, available_tools: List) -> str:
        """Get AI's next action - either tool calls or final answer"""
        
        # Build execution history summary
        history_summary = ""
        if context['execution_history']:
            history_summary = "\nPrevious tool executions:\n"
            for exec_data in context['execution_history'][-5:]:  # Last 5 executions
                history_summary += f"- {exec_data['tool']}: "
                if 'error' in exec_data:
                    history_summary += f"ERROR: {exec_data['error']}\n"
                else:
                    result_str = json.dumps(exec_data['result'], indent=2) if isinstance(exec_data['result'], dict) else str(exec_data['result'])
                    history_summary += f"{result_str[:200]}...\n" if len(result_str) > 200 else f"{result_str}\n"
        
        prompt = f"""User request: {context['user_input']}

Current iteration: {context['iteration']}/{self.max_iterations}
{history_summary}

Current understanding: {context['current_understanding']}

Available tools (use these to find information):
{self._format_tools(available_tools[:40])}

INSTRUCTIONS:
1. If you have found COMPLETE information to answer the user's request, respond with:
   FINAL_ANSWER: [your complete answer here]

2. If you need more information, call the appropriate tools. For example:
   - For system status: get_system_status()
   - For task lists: list_tasks() or get_tasks()
   - For searching files: grep_files(pattern="task") or read_file(file_path="/path/to/file")
   
3. IMPORTANT: If the user asks for "13 tasks", they want to see the actual list of 13 tasks, not just know that 13 exist.
   Keep searching until you find the actual tasks!

4. Be persistent! If one tool doesn't give you what you need, try another. Think creatively about where the information might be stored.

What's your next action?"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return response.get('result', '')
    
    async def _execute_tools_from_response(self, response: str, valid_tools: List[str]) -> List[Dict]:
        """Extract and execute all tool calls from response"""
        executions = []
        
        # Multiple patterns to catch tool calls
        patterns = [
            r'\b(\w+)\s*\((.*?)\)',  # function_name(params)
            r'`(\w+)\((.*?)\)`',      # `function_name(params)`
            r'^(\w+)$',               # Just tool name on its own line
        ]
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            for pattern in patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        tool_name = match[0]
                        params_str = match[1] if len(match) > 1 else ''
                    else:
                        tool_name = match
                        params_str = ''
                    
                    if tool_name in valid_tools:
                        # Execute the tool
                        try:
                            params = self._parse_params(params_str)
                            result = await self.tool_system.call_tool(tool_name, **params)
                            
                            executions.append({
                                'tool': tool_name,
                                'params': params,
                                'result': result.get('result', result)
                            })
                        except Exception as e:
                            executions.append({
                                'tool': tool_name,
                                'params': params_str,
                                'error': str(e)
                            })
        
        return executions
    
    def _parse_params(self, params_str: str) -> Dict:
        """Parse parameter string into dict"""
        if not params_str:
            return {}
        
        # Try JSON first
        try:
            if params_str.startswith('{'):
                return json.loads(params_str)
        except:
            pass
        
        # Try key=value format
        params = {}
        if '=' in params_str:
            for part in params_str.split(','):
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    
                    # Try to parse value as JSON
                    try:
                        params[key] = json.loads(value)
                    except:
                        params[key] = value
        
        return params
    
    async def _summarize_findings(self, context: Dict) -> str:
        """Summarize what we've learned so far"""
        if not context['execution_history']:
            return "No information gathered yet"
        
        prompt = f"""Based on these tool executions:
{json.dumps(context['execution_history'], indent=2)}

Summarize what we've learned about: {context['user_input']}

Be specific about what we found and what's still missing."""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return response.get('result', 'Unable to summarize findings')
    
    async def _generate_best_answer(self, context: Dict) -> str:
        """Generate best possible answer with available information"""
        prompt = f"""The user asked: {context['user_input']}

After {context['iteration']} iterations, here's what I found:
{json.dumps(context['execution_history'], indent=2)}

Current understanding: {context['current_understanding']}

Please provide the best possible answer with the information available. If some information is missing, acknowledge that."""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return response.get('result', 'I searched extensively but could not find the complete information requested.')
    
    def _format_tools(self, tools: List[Dict]) -> str:
        """Format tools for display"""
        lines = []
        for tool in tools:
            desc = tool.get('description', 'No description')[:60]
            lines.append(f"- {tool['name']}: {desc}...")
        return '\n'.join(lines)


# Test function
async def test_v3():
    assistant = ConversationalAIAssistantV3()
    
    queries = [
        "What's the system status?",
        "List all 13 tasks mentioned in the system status",
        "Show me the actual 13 tasks that were completed"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        response = await assistant.process_query(query)
        print(f"Response: {response}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_v3())