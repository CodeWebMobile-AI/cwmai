#!/usr/bin/env python3
"""
Conversational AI Assistant V5
Smarter repository and resource discovery with fuzzy matching
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher

from scripts.ai_brain import IntelligentAIBrain
from scripts.conversational_ai_logger import log_system_message

class ConversationalAIAssistantV5:
    """V5: Pre-fetches resources and uses fuzzy matching to find the right ones"""
    
    def __init__(self):
        self.ai_brain = IntelligentAIBrain()
        self.conversation_history = []
        self.max_iterations = 10
        self._cached_resources = {}
        
    async def process_query(self, user_input: str) -> str:
        """Process query with intelligent resource discovery"""
        
        # Get available tools
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        self.tool_system = tool_system
        available_tools = tool_system.get_tools_for_ai()
        
        # Pre-fetch available resources
        resources = await self._prefetch_resources()
        
        # Analyze query with resource context
        query_analysis = await self._analyze_query_with_resources(user_input, available_tools, resources)
        
        # Build context
        context = {
            'user_input': user_input,
            'query_analysis': query_analysis,
            'available_resources': resources,
            'tools_available': [t['name'] for t in available_tools],
            'execution_history': [],
            'current_understanding': '',
            'iteration': 0
        }
        
        # Execute search strategy
        while context['iteration'] < self.max_iterations:
            context['iteration'] += 1
            
            # Get next action
            next_action = await self._get_next_action_with_resources(context, available_tools)
            
            # Check for final answer
            if 'FINAL_ANSWER:' in next_action:
                return next_action.replace('FINAL_ANSWER:', '').strip()
            
            # Execute tools with parameter correction
            tool_executions = await self._execute_tools_with_correction(next_action, context)
            
            if tool_executions:
                context['execution_history'].extend(tool_executions)
                context['current_understanding'] = await self._summarize_findings(context)
            else:
                context['current_understanding'] = "No new information found in this iteration."
        
        # Max iterations reached
        return await self._generate_best_answer(context)
    
    async def _prefetch_resources(self) -> Dict[str, List[Dict]]:
        """Pre-fetch available resources to provide context"""
        resources = {
            'repositories': [],
            'recent_tasks': []
        }
        
        try:
            # Get repositories
            result = await self.tool_system.call_tool('get_repositories')
            if result.get('success') and result.get('result'):
                repos = result['result'].get('repositories', [])
                resources['repositories'] = repos
                self._cached_resources['repositories'] = repos
        except:
            # Try cached if available
            resources['repositories'] = self._cached_resources.get('repositories', [])
        
        return resources
    
    async def _analyze_query_with_resources(self, user_input: str, available_tools: List[Dict], resources: Dict) -> Dict[str, Any]:
        """Analyze query with knowledge of available resources"""
        
        # Get repository names and descriptions
        repo_info = []
        for repo in resources.get('repositories', []):
            name = repo.get('name', '')
            desc = repo.get('description', '')
            repo_info.append(f"- {name}: {desc[:60]}...")
        
        repo_context = "\n".join(repo_info) if repo_info else "No repositories found"
        
        prompt = f"""Analyze this user query: "{user_input}"

Available repositories in the system:
{repo_context}

Consider:
1. Does the query mention any of these repositories (exact or partial match)?
2. Could the query be asking about a concept that matches a repository description?
3. Is this asking about internal resources or external information?

Analyze and respond with:
QUERY_TYPE: internal/external/ambiguous
CONFIDENCE: high/medium/low
MATCHED_RESOURCES: [list any repositories or resources that might match]
KEY_TERMS: [important terms from the query]
REASONING: [why you made this classification]
SEARCH_STRATEGY: [ordered steps to find the information]
"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        analysis = self._parse_resource_analysis(response.get('result', ''))
        
        # Add fuzzy matching results
        if analysis['key_terms']:
            analysis['fuzzy_matches'] = self._find_fuzzy_matches(
                analysis['key_terms'], 
                resources.get('repositories', [])
            )
        
        return analysis
    
    def _find_fuzzy_matches(self, terms: List[str], repositories: List[Dict]) -> List[Dict]:
        """Find repositories that fuzzy match the given terms"""
        matches = []
        
        for term in terms:
            term_lower = term.lower()
            
            for repo in repositories:
                repo_name = repo.get('name', '').lower()
                repo_desc = repo.get('description', '').lower()
                
                # Calculate similarity scores
                name_score = SequenceMatcher(None, term_lower, repo_name).ratio()
                
                # Check if term is part of repo name or vice versa
                contains_score = 1.0 if term_lower in repo_name or repo_name in term_lower else 0
                
                # Check description
                desc_score = 0.5 if term_lower in repo_desc else 0
                
                # Combined score
                total_score = max(name_score, contains_score, desc_score)
                
                if total_score > 0.6:  # Threshold for matches
                    matches.append({
                        'repository': repo.get('name'),
                        'score': total_score,
                        'match_type': 'exact' if total_score == 1.0 else 'fuzzy'
                    })
        
        # Sort by score and remove duplicates
        seen = set()
        unique_matches = []
        for match in sorted(matches, key=lambda x: x['score'], reverse=True):
            if match['repository'] not in seen:
                seen.add(match['repository'])
                unique_matches.append(match)
        
        return unique_matches
    
    def _parse_resource_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse analysis with resource matching"""
        result = {
            'query_type': 'ambiguous',
            'confidence': 'low',
            'matched_resources': [],
            'key_terms': [],
            'reasoning': '',
            'search_strategy': [],
            'fuzzy_matches': []
        }
        
        # Extract matched resources
        if 'MATCHED_RESOURCES:' in analysis:
            match = re.search(r'MATCHED_RESOURCES:\s*\[(.*?)\]', analysis, re.IGNORECASE | re.DOTALL)
            if match:
                resources = match.group(1).split(',')
                result['matched_resources'] = [r.strip().strip('"\'') for r in resources if r.strip()]
        
        # Extract other fields (similar to V4)
        if 'QUERY_TYPE:' in analysis:
            match = re.search(r'QUERY_TYPE:\s*(\w+)', analysis, re.IGNORECASE)
            if match:
                result['query_type'] = match.group(1).lower()
        
        if 'CONFIDENCE:' in analysis:
            match = re.search(r'CONFIDENCE:\s*(\w+)', analysis, re.IGNORECASE)
            if match:
                result['confidence'] = match.group(1).lower()
        
        if 'KEY_TERMS:' in analysis:
            match = re.search(r'KEY_TERMS:\s*\[(.*?)\]', analysis, re.IGNORECASE | re.DOTALL)
            if match:
                terms = match.group(1).split(',')
                result['key_terms'] = [t.strip().strip('"\'') for t in terms if t.strip()]
        
        if 'REASONING:' in analysis:
            match = re.search(r'REASONING:\s*(.+?)(?:\n[A-Z]+:|$)', analysis, re.IGNORECASE | re.DOTALL)
            if match:
                result['reasoning'] = match.group(1).strip()
        
        if 'SEARCH_STRATEGY:' in analysis:
            match = re.search(r'SEARCH_STRATEGY:\s*(.+?)(?:\n[A-Z]+:|$)', analysis, re.IGNORECASE | re.DOTALL)
            if match:
                strategy = match.group(1).strip()
                steps = re.findall(r'(?:\d+\.|\-)\s*(.+?)(?=\d+\.|\-|$)', strategy, re.DOTALL)
                if steps:
                    result['search_strategy'] = [s.strip() for s in steps]
                else:
                    result['search_strategy'] = [strategy]
        
        return result
    
    async def _get_next_action_with_resources(self, context: Dict, available_tools: List) -> str:
        """Get next action with resource awareness"""
        
        # Build execution history
        history_summary = self._build_history_summary(context)
        
        # Build resource hints
        resource_hints = self._build_resource_hints(context)
        
        prompt = f"""User request: {context['user_input']}

Query Analysis:
{json.dumps(context['query_analysis'], indent=2)}

{resource_hints}

Current iteration: {context['iteration']}/{self.max_iterations}
{history_summary}

Current understanding: {context['current_understanding']}

Available tools:
{self._format_tools(available_tools[:40])}

INSTRUCTIONS:
1. If the query matches a repository, use the EXACT repository name from the matched resources
2. If unsure which repository, check fuzzy matches and pick the best one
3. Use analyze_repository(repo="exact-repo-name") with the correct name
4. When you have a complete answer, respond with: FINAL_ANSWER: [your answer]

What's your next action?"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return response.get('result', '')
    
    def _build_resource_hints(self, context: Dict) -> str:
        """Build hints about available resources"""
        hints = []
        
        analysis = context['query_analysis']
        
        # Add matched resources
        if analysis.get('matched_resources'):
            hints.append(f"Directly matched resources: {', '.join(analysis['matched_resources'])}")
        
        # Add fuzzy matches
        if analysis.get('fuzzy_matches'):
            fuzzy_list = []
            for match in analysis['fuzzy_matches'][:3]:  # Top 3
                fuzzy_list.append(f"{match['repository']} (score: {match['score']:.2f})")
            hints.append(f"Possible matches: {', '.join(fuzzy_list)}")
        
        # Add available repositories if needed
        if context.get('available_resources', {}).get('repositories'):
            repo_names = [r['name'] for r in context['available_resources']['repositories']]
            hints.append(f"All available repositories: {', '.join(repo_names[:10])}")
        
        return "RESOURCE HINTS:\n" + "\n".join(hints) if hints else ""
    
    async def _execute_tools_with_correction(self, response: str, context: Dict) -> List[Dict]:
        """Execute tools with parameter correction for repository names"""
        executions = []
        valid_tools = context['tools_available']
        
        # Pattern to match tool calls
        patterns = [
            r'\b(\w+)\s*\((.*?)\)',  # function_name(params)
            r'`(\w+)\((.*?)\)`',      # `function_name(params)`
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
                        # Parse and correct parameters
                        params = self._parse_and_correct_params(tool_name, params_str, context)
                        
                        try:
                            result = await self.tool_system.call_tool(tool_name, **params)
                            
                            executions.append({
                                'tool': tool_name,
                                'params': params,
                                'result': result.get('result', result)
                            })
                        except Exception as e:
                            executions.append({
                                'tool': tool_name,
                                'params': params,
                                'error': str(e)
                            })
        
        return executions
    
    def _parse_and_correct_params(self, tool_name: str, params_str: str, context: Dict) -> Dict:
        """Parse parameters and correct repository names if needed"""
        params = {}
        
        if not params_str:
            return params
        
        # Try to parse parameters
        try:
            # Handle different parameter formats
            if params_str.startswith('{'):
                params = json.loads(params_str)
            elif '=' in params_str:
                # Parse key=value format
                for part in params_str.split(','):
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip().strip('"\'')
                        value = value.strip().strip('"\'')
                        params[key] = value
            else:
                # Single parameter - for analyze_repository it should be 'repo'
                if tool_name == 'analyze_repository':
                    params = {'repo': params_str.strip('"\'')}
                else:
                    params = {'name': params_str.strip('"\''), 'repo': params_str.strip('"\'')}
        except:
            if tool_name == 'analyze_repository':
                params = {'repo': params_str.strip('"\'')}
            else:
                params = {'name': params_str.strip('"\''), 'repo': params_str.strip('"\'')}
        
        # Correct repository names for analyze_repository tool
        if tool_name == 'analyze_repository' and context.get('available_resources'):
            repos = context['available_resources'].get('repositories', [])
            repo_names = {r['name'].lower(): r['name'] for r in repos}
            
            # For analyze_repository, ensure we use 'repo' parameter
            if 'name' in params and 'repo' not in params:
                params['repo'] = params['name']
                del params['name']
            
            # Check the repo parameter
            if 'repo' in params and params['repo']:
                param_lower = params['repo'].lower()
                
                # Exact match
                if param_lower in repo_names:
                    params['repo'] = repo_names[param_lower]
                else:
                    # Fuzzy match
                    best_match = None
                    best_score = 0
                    
                    for repo_lower, repo_actual in repo_names.items():
                        score = SequenceMatcher(None, param_lower, repo_lower).ratio()
                        if score > best_score:
                            best_score = score
                            best_match = repo_actual
                    
                    if best_match and best_score > 0.6:
                        params['repo'] = best_match
        
        return params
    
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

Please provide the best possible answer with the information available."""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'gemini'
        })
        
        return response.get('result', 'I searched but could not find complete information.')
    
    def _format_tools(self, tools: List[Dict]) -> str:
        """Format tools for display"""
        lines = []
        for tool in tools:
            desc = tool.get('description', 'No description')[:60]
            lines.append(f"- {tool['name']}: {desc}...")
        return '\n'.join(lines)


# Test function
async def test_v5():
    assistant = ConversationalAIAssistantV5()
    
    queries = [
        "tell me about hydra-health-ai",
        "what's the hydration project?",
        "show me info about brand guardian",
        "what is machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        response = await assistant.process_query(query)
        print(f"Response: {response}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_v5())