#!/usr/bin/env python3
"""
Conversational AI Assistant V4
Smarter context-aware tool selection without hardcoding
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

from scripts.ai_brain import IntelligentAIBrain
from scripts.conversational_ai_logger import log_system_message

class ConversationalAIAssistantV4:
    """V4: Smarter about checking internal resources before going external"""
    
    def __init__(self):
        self.ai_brain = IntelligentAIBrain()
        self.conversation_history = []
        self.max_iterations = 10
        
    async def process_query(self, user_input: str) -> str:
        """Process query with intelligent tool selection"""
        
        # Get available tools
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        self.tool_system = tool_system
        available_tools = tool_system.get_tools_for_ai()
        
        # First, analyze what the user might be asking about
        query_analysis = await self._analyze_query_intent(user_input, available_tools)
        
        # Build context with analysis
        context = {
            'user_input': user_input,
            'query_analysis': query_analysis,
            'tools_available': [t['name'] for t in available_tools],
            'execution_history': [],
            'current_understanding': '',
            'iteration': 0
        }
        
        # Execute search strategy based on analysis
        while context['iteration'] < self.max_iterations:
            context['iteration'] += 1
            
            # Get AI's next action with query analysis context
            next_action = await self._get_next_action_with_context(context, available_tools)
            
            # Check for final answer
            if 'FINAL_ANSWER:' in next_action:
                return next_action.replace('FINAL_ANSWER:', '').strip()
            
            # Execute tools
            tool_executions = await self._execute_tools_from_response(next_action, context['tools_available'])
            
            if tool_executions:
                context['execution_history'].extend(tool_executions)
                context['current_understanding'] = await self._summarize_findings(context)
            else:
                # No tools executed, update understanding
                context['current_understanding'] = "No new information found in this iteration."
        
        # Max iterations reached
        return await self._generate_best_answer(context)
    
    async def _analyze_query_intent(self, user_input: str, available_tools: List[Dict]) -> Dict[str, Any]:
        """Analyze what the user is likely asking about"""
        
        # Group tools by category
        tool_categories = self._categorize_tools(available_tools)
        
        prompt = f"""Analyze this user query: "{user_input}"

Think about:
1. Are they asking about something that could exist in their local system?
   - A repository/project name (e.g., "hydra-health-ai", "my-project")
   - A file or directory
   - A task or issue
   - System status or metrics
   
2. Are they asking for external/general information?
   - General concepts or technologies
   - Best practices or tutorials
   - External APIs or services

3. What specific words or patterns suggest internal vs external?

Available tool categories:
{self._format_tool_categories(tool_categories)}

Provide analysis as:
QUERY_TYPE: internal/external/ambiguous
CONFIDENCE: high/medium/low
KEY_TERMS: [specific terms that might be project/file names]
REASONING: [why you think this]
SEARCH_STRATEGY: [what to check first, second, etc.]
"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return self._parse_query_analysis(response.get('result', ''))
    
    async def _get_next_action_with_context(self, context: Dict, available_tools: List) -> str:
        """Get next action considering query analysis"""
        
        # Build execution history
        history_summary = self._build_history_summary(context)
        
        # Build search guidance based on query analysis
        search_guidance = self._build_search_guidance(context)
        
        prompt = f"""User request: {context['user_input']}

Query Analysis:
{json.dumps(context['query_analysis'], indent=2)}

{search_guidance}

Current iteration: {context['iteration']}/{self.max_iterations}
{history_summary}

Current understanding: {context['current_understanding']}

Available tools:
{self._format_tools(available_tools[:40])}

INSTRUCTIONS:
1. Follow the search strategy suggested in the query analysis
2. If query seems internal but nothing found locally, THEN try external search
3. Be thorough - check multiple sources if needed
4. When you have a complete answer, respond with: FINAL_ANSWER: [your answer]

What's your next action?"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return response.get('result', '')
    
    def _categorize_tools(self, tools: List[Dict]) -> Dict[str, List[str]]:
        """Categorize tools by their likely purpose"""
        categories = {
            'repository': [],
            'file_system': [],
            'search_internal': [],
            'system_info': [],
            'external_search': [],
            'task_management': [],
            'other': []
        }
        
        for tool in tools:
            name = tool['name'].lower()
            desc = tool.get('description', '').lower()
            
            if any(word in name + desc for word in ['repository', 'repo', 'analyze_repository', 'get_repositories']):
                categories['repository'].append(tool['name'])
            elif any(word in name + desc for word in ['file', 'read', 'list_files', 'directory']):
                categories['file_system'].append(tool['name'])
            elif any(word in name + desc for word in ['grep', 'search_code', 'find']) and 'web' not in name:
                categories['search_internal'].append(tool['name'])
            elif any(word in name + desc for word in ['system', 'status', 'worker', 'check_system']):
                categories['system_info'].append(tool['name'])
            elif any(word in name + desc for word in ['research', 'web', 'brave', 'external']):
                categories['external_search'].append(tool['name'])
            elif any(word in name + desc for word in ['task', 'issue', 'get_tasks']):
                categories['task_management'].append(tool['name'])
            else:
                categories['other'].append(tool['name'])
        
        return categories
    
    def _format_tool_categories(self, categories: Dict[str, List[str]]) -> str:
        """Format categorized tools"""
        lines = []
        for cat, tools in categories.items():
            if tools:
                lines.append(f"{cat.replace('_', ' ').title()}: {', '.join(tools[:5])}")
        return '\n'.join(lines)
    
    def _parse_query_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse query analysis from AI"""
        result = {
            'query_type': 'ambiguous',
            'confidence': 'low',
            'key_terms': [],
            'reasoning': '',
            'search_strategy': []
        }
        
        # Extract query type
        if 'QUERY_TYPE:' in analysis:
            match = re.search(r'QUERY_TYPE:\s*(\w+)', analysis, re.IGNORECASE)
            if match:
                result['query_type'] = match.group(1).lower()
        
        # Extract confidence
        if 'CONFIDENCE:' in analysis:
            match = re.search(r'CONFIDENCE:\s*(\w+)', analysis, re.IGNORECASE)
            if match:
                result['confidence'] = match.group(1).lower()
        
        # Extract key terms
        if 'KEY_TERMS:' in analysis:
            match = re.search(r'KEY_TERMS:\s*\[(.*?)\]', analysis, re.IGNORECASE | re.DOTALL)
            if match:
                terms = match.group(1).split(',')
                result['key_terms'] = [t.strip().strip('"\'') for t in terms if t.strip()]
        
        # Extract reasoning
        if 'REASONING:' in analysis:
            match = re.search(r'REASONING:\s*(.+?)(?:\n[A-Z]+:|$)', analysis, re.IGNORECASE | re.DOTALL)
            if match:
                result['reasoning'] = match.group(1).strip()
        
        # Extract search strategy
        if 'SEARCH_STRATEGY:' in analysis:
            match = re.search(r'SEARCH_STRATEGY:\s*(.+?)(?:\n[A-Z]+:|$)', analysis, re.IGNORECASE | re.DOTALL)
            if match:
                strategy = match.group(1).strip()
                # Try to extract ordered steps
                steps = re.findall(r'(?:\d+\.|\-)\s*(.+?)(?=\d+\.|\-|$)', strategy, re.DOTALL)
                if steps:
                    result['search_strategy'] = [s.strip() for s in steps]
                else:
                    result['search_strategy'] = [strategy]
        
        return result
    
    def _build_search_guidance(self, context: Dict) -> str:
        """Build search guidance based on query analysis"""
        analysis = context['query_analysis']
        
        if analysis['query_type'] == 'internal' and analysis['confidence'] in ['high', 'medium']:
            return f"""SEARCH GUIDANCE: The query appears to be asking about something INTERNAL to the system.
Key terms that might be project/file names: {', '.join(analysis['key_terms'])}

Suggested approach:
1. First check repositories (get_repositories, analyze_repository)
2. Then check files/directories (list_files, read_file)
3. Then check tasks/issues (list_tasks, get_repository_issues)
4. Only if nothing found internally, consider external search"""
        
        elif analysis['query_type'] == 'external':
            return f"""SEARCH GUIDANCE: The query appears to be asking about EXTERNAL/general information.
However, still check if any key terms match internal resources first: {', '.join(analysis['key_terms'])}"""
        
        else:
            return f"""SEARCH GUIDANCE: The query is ambiguous - could be internal or external.
Key terms to check: {', '.join(analysis['key_terms'])}

Recommended approach:
1. Quick check for internal resources matching key terms
2. If found, explore those thoroughly
3. If not found, proceed with external search"""
    
    def _build_history_summary(self, context: Dict) -> str:
        """Build execution history summary"""
        if not context['execution_history']:
            return ""
        
        summary = "\nPrevious tool executions:\n"
        for exec_data in context['execution_history'][-5:]:
            summary += f"- {exec_data['tool']}: "
            if 'error' in exec_data:
                summary += f"ERROR: {exec_data['error']}\n"
            else:
                result_str = json.dumps(exec_data['result'], indent=2) if isinstance(exec_data['result'], dict) else str(exec_data['result'])
                summary += f"{result_str[:200]}...\n" if len(result_str) > 200 else f"{result_str}\n"
        
        return summary
    
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
async def test_v4():
    assistant = ConversationalAIAssistantV4()
    
    queries = [
        "tell me about hydra-health-ai",
        "what is machine learning?",
        "show me the system status",
        "list files in the scripts directory"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        response = await assistant.process_query(query)
        print(f"Response: {response}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_v4())