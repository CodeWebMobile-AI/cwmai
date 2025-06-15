"""
Natural Language Interface for CWMAI

This module provides natural language processing capabilities to interpret
user commands and map them to CWMAI system operations.
"""

import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from scripts.ai_brain import IntelligentAIBrain
from scripts.mcp_integration import MCPIntegrationHub
from scripts.task_manager import TaskManager
from scripts.state_manager import StateManager
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.http_ai_client import HTTPAIClient


class NaturalLanguageInterface:
    """Natural language command parser and executor for CWMAI."""
    
    def __init__(self):
        """Initialize the natural language interface."""
        self.ai_brain = IntelligentAIBrain(enable_round_robin=True)
        self.mcp_hub = MCPIntegrationHub()
        self.state_manager = StateManager()
        self.http_client = HTTPAIClient()
        
        # Command patterns and their handlers
        self.command_patterns = {
            'create_issue': [
                r'create (?:an )?issue (?:for |in )?(?P<repo>\S+)(?: about| for)? (?P<topic>.*)',
                r'(?:make|add) (?:a )?(?:new )?issue (?:for |in )?(?P<repo>\S+)(?: about| for)? (?P<topic>.*)',
                r'(?:open|file) (?:an? )?issue (?:for |in )?(?P<repo>\S+)(?: about| for)? (?P<topic>.*)'
            ],
            'search_repos': [
                r'search (?:for )?repositor(?:ies|y) (?:for |with |containing )?(?P<query>.*)',
                r'find (?:me )?(?:some )?repos? (?:for |with |about |containing )?(?P<query>.*)',
                r'look for (?:some )?repositor(?:ies|y) (?:for |with |about )?(?P<query>.*)'
            ],
            'create_architecture': [
                r'create (?:an )?architecture (?:for |of )?(?P<project>.*)',
                r'(?:design|generate) (?:an? )?(?:system )?architecture (?:for |of )?(?P<project>.*)',
                r'(?:make|build) (?:me )?(?:an? )?architecture (?:for |of )?(?P<project>.*)'
            ],
            'show_status': [
                r'(?:show|display|get)(?: me)?(?: the)? (?:system )?status',
                r'(?:what\'s|what is)(?: the)? (?:system )?status\??',
                r'status(?: report)?',
                r'how (?:is|are) (?:things|we) doing\??'
            ],
            'list_tasks': [
                r'(?:list|show|display)(?: me)?(?: all)?(?: the)? (?:active |current |pending )?tasks?',
                r'(?:what|which) tasks? (?:are|is) (?:active|pending|current)',
                r'(?:show|get)(?: me)? (?:task|todo) list'
            ],
            'analyze_performance': [
                r'analyze (?:system )?performance',
                r'(?:show|display|get)(?: me)? performance (?:metrics|stats|statistics)',
                r'how (?:is|are) (?:we|the system) performing\??',
                r'performance (?:report|analysis)'
            ],
            'create_task': [
                r'create (?:a )?(?:new )?task (?:for |to )?(?P<description>.*)',
                r'(?:add|make) (?:a )?(?:new )?task (?:for |to )?(?P<description>.*)',
                r'(?:generate|schedule) (?:a )?task (?:for |to )?(?P<description>.*)'
            ],
            'help': [
                r'help(?:\s+(?P<topic>\w+))?',
                r'(?:what|which) (?:commands?|can I|do you) (?:support|understand|know)',
                r'(?:show|list)(?: me)? (?:available )?commands?'
            ]
        }
        
        # Context for maintaining conversation state
        self.context = {
            'last_command': None,
            'last_repo': None,
            'last_query': None,
            'conversation_history': []
        }
        
    async def initialize(self):
        """Initialize MCP connections and load system state."""
        try:
            await self.mcp_hub.initialize()
            self.state_manager.load_state()
            return True
        except Exception as e:
            print(f"Warning: Could not initialize MCP hub: {e}")
            print("Continuing with limited functionality...")
            return False
            
    async def close(self):
        """Close connections and save state."""
        await self.mcp_hub.close()
        
    def parse_command(self, user_input: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
        """
        Parse user input to identify command and extract parameters.
        
        Args:
            user_input: The natural language command from the user
            
        Returns:
            Tuple of (command_type, parameters) or (None, None) if not recognized
        """
        user_input = user_input.strip().lower()
        
        # Add to conversation history
        self.context['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'input': user_input
        })
        
        # Try to match against known patterns
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, user_input, re.IGNORECASE)
                if match:
                    params = match.groupdict()
                    self.context['last_command'] = command_type
                    return command_type, params
                    
        return None, None
        
    async def interpret_with_ai(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Use AI to interpret ambiguous commands.
        
        Args:
            user_input: The natural language command
            
        Returns:
            Tuple of (command_type, parameters)
        """
        prompt = f"""
        Interpret this user command for the CWMAI system and extract the intent and parameters.
        
        User command: "{user_input}"
        
        Available commands:
        - create_issue: Create a GitHub issue (needs: repo, topic)
        - search_repos: Search for repositories (needs: query)
        - create_architecture: Generate system architecture (needs: project)
        - show_status: Display system status
        - list_tasks: Show active tasks
        - analyze_performance: Show performance metrics
        - create_task: Create a new task (needs: description)
        - help: Show help information
        
        Context:
        - Last command: {self.context.get('last_command', 'none')}
        - Last repository: {self.context.get('last_repo', 'none')}
        
        Return a JSON object with:
        {{
            "command": "command_type",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }},
            "confidence": 0.0-1.0,
            "interpretation": "Natural language explanation"
        }}
        """
        
        try:
            response = await self.ai_brain.execute_capability('command_interpretation', {
                'prompt': prompt
            })
            
            if response['status'] == 'success':
                result = self.ai_brain.extract_json_from_response(response['result'])
                if result:
                    return result['command'], result['parameters']
        except Exception as e:
            print(f"AI interpretation failed: {e}")
            
        return 'unknown', {}
        
    async def execute_command(self, command_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the identified command.
        
        Args:
            command_type: The type of command to execute
            params: Parameters for the command
            
        Returns:
            Result dictionary with status and response
        """
        try:
            if command_type == 'create_issue':
                return await self._create_issue(params)
            elif command_type == 'search_repos':
                return await self._search_repositories(params)
            elif command_type == 'create_architecture':
                return await self._create_architecture(params)
            elif command_type == 'show_status':
                return await self._show_status()
            elif command_type == 'list_tasks':
                return await self._list_tasks()
            elif command_type == 'analyze_performance':
                return await self._analyze_performance()
            elif command_type == 'create_task':
                return await self._create_task(params)
            elif command_type == 'help':
                return await self._show_help(params.get('topic'))
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown command: {command_type}'
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error executing command: {str(e)}'
            }
            
    async def _create_issue(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Create a GitHub issue."""
        repo = params.get('repo', self.context.get('last_repo'))
        topic = params.get('topic', '')
        
        if not repo:
            return {
                'status': 'error',
                'message': 'Please specify a repository name'
            }
            
        if not topic:
            return {
                'status': 'error',
                'message': 'Please specify what the issue is about'
            }
            
        # Update context
        self.context['last_repo'] = repo
        
        # Generate detailed issue content using AI
        prompt = f"""
        Create a detailed GitHub issue for the following:
        Repository: {repo}
        Topic: {topic}
        
        Generate:
        1. A clear, concise title
        2. Detailed description with context
        3. Acceptance criteria
        4. Technical requirements
        5. Suggested implementation approach
        
        Format as JSON with: title, body, labels (array)
        """
        
        response = await self.ai_brain.execute_capability('issue_generation', {
            'prompt': prompt
        })
        
        if response['status'] == 'success':
            issue_data = self.ai_brain.extract_json_from_response(response['result'])
            
            if issue_data and self.mcp_hub.github:
                # Create the issue via MCP
                result = await self.mcp_hub.github.create_issue(
                    repo=repo,
                    title=issue_data.get('title', topic),
                    body=issue_data.get('body', topic),
                    labels=issue_data.get('labels', ['enhancement'])
                )
                
                if result:
                    return {
                        'status': 'success',
                        'message': f"Issue created successfully!",
                        'data': result
                    }
                    
        return {
            'status': 'error',
            'message': 'Failed to create issue. GitHub integration may not be available.'
        }
        
    async def _search_repositories(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Search for repositories."""
        query = params.get('query', '')
        
        if not query:
            return {
                'status': 'error',
                'message': 'Please specify what to search for'
            }
            
        self.context['last_query'] = query
        
        if self.mcp_hub.github:
            repos = await self.mcp_hub.github.search_repositories(query, limit=10)
            
            if repos:
                return {
                    'status': 'success',
                    'message': f"Found {len(repos)} repositories",
                    'data': repos
                }
        
        # Fallback to local repository analysis
        analyzer = RepositoryAnalyzer()
        local_repos = analyzer.analyze_repositories(query)
        
        return {
            'status': 'success',
            'message': f"Found {len(local_repos)} local repositories",
            'data': local_repos
        }
        
    async def _create_architecture(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Generate system architecture."""
        project = params.get('project', '')
        
        if not project:
            return {
                'status': 'error',
                'message': 'Please specify the project name or description'
            }
            
        # Use AI to generate architecture
        prompt = f"""
        Create a comprehensive system architecture for: {project}
        
        Include:
        1. High-level architecture overview
        2. Component breakdown
        3. Technology stack recommendations
        4. Data flow diagrams (describe textually)
        5. Deployment architecture
        6. Security considerations
        7. Scalability approach
        
        Format as structured markdown.
        """
        
        response = await self.ai_brain.execute_capability('architecture_generation', {
            'prompt': prompt
        })
        
        if response['status'] == 'success':
            return {
                'status': 'success',
                'message': 'Architecture generated successfully',
                'data': {
                    'project': project,
                    'architecture': response['result']
                }
            }
            
        return {
            'status': 'error',
            'message': 'Failed to generate architecture'
        }
        
    async def _show_status(self) -> Dict[str, Any]:
        """Show system status."""
        state = self.state_manager.get_state()
        
        # Calculate statistics
        total_tasks = len(state.get('tasks', []))
        active_tasks = len([t for t in state.get('tasks', []) if t.get('status') == 'active'])
        completed_tasks = len([t for t in state.get('tasks', []) if t.get('status') == 'completed'])
        
        # Get AI provider status
        providers = []
        if self.ai_brain.http_ai_client.providers_available.get('anthropic', False):
            providers.append('Anthropic')
        if self.ai_brain.http_ai_client.providers_available.get('openai', False):
            providers.append('OpenAI')
        if self.ai_brain.http_ai_client.providers_available.get('gemini', False):
            providers.append('Gemini')
        if self.ai_brain.http_ai_client.providers_available.get('deepseek', False):
            providers.append('DeepSeek')
            
        return {
            'status': 'success',
            'message': 'System status retrieved',
            'data': {
                'total_tasks': total_tasks,
                'active_tasks': active_tasks,
                'completed_tasks': completed_tasks,
                'ai_providers': providers,
                'repositories': len(state.get('repositories', [])),
                'last_update': state.get('last_update', 'Unknown')
            }
        }
        
    async def _list_tasks(self) -> Dict[str, Any]:
        """List active tasks."""
        state = self.state_manager.get_state()
        tasks = [t for t in state.get('tasks', []) if t.get('status') in ['active', 'pending']]
        
        return {
            'status': 'success',
            'message': f'Found {len(tasks)} active/pending tasks',
            'data': tasks[:10]  # Limit to 10 most recent
        }
        
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance."""
        state = self.state_manager.get_state()
        performance = state.get('performance_metrics', {})
        
        # Calculate additional metrics
        tasks = state.get('tasks', [])
        if tasks:
            completion_rate = len([t for t in tasks if t.get('status') == 'completed']) / len(tasks)
        else:
            completion_rate = 0
            
        return {
            'status': 'success',
            'message': 'Performance analysis complete',
            'data': {
                'completion_rate': f"{completion_rate * 100:.1f}%",
                'total_tasks': len(tasks),
                'avg_completion_time': performance.get('avg_completion_time', 'N/A'),
                'success_rate': performance.get('success_rate', 'N/A'),
                'last_24h_tasks': performance.get('last_24h_tasks', 0)
            }
        }
        
    async def _create_task(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Create a new task."""
        description = params.get('description', '')
        
        if not description:
            return {
                'status': 'error',
                'message': 'Please provide a task description'
            }
            
        # Use TaskManager to create task
        task_manager = TaskManager()
        task = task_manager.create_task(
            title=description[:100],  # First 100 chars as title
            description=description,
            task_type='feature',  # Default type
            priority='medium'
        )
        
        return {
            'status': 'success',
            'message': 'Task created successfully',
            'data': task
        }
        
    async def _show_help(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Show help information."""
        if topic:
            help_topics = {
                'issue': "Create GitHub issues with: 'create issue for [repo] about [topic]'",
                'search': "Search repositories with: 'search repositories for [query]'",
                'architecture': "Generate architecture with: 'create architecture for [project]'",
                'status': "Check system status with: 'show status' or 'status'",
                'tasks': "List tasks with: 'list tasks' or 'show active tasks'",
                'performance': "Analyze performance with: 'analyze performance'",
                'task': "Create tasks with: 'create task [description]'"
            }
            
            help_text = help_topics.get(topic, f"No help available for '{topic}'")
            
            return {
                'status': 'success',
                'message': f'Help for {topic}',
                'data': {'help': help_text}
            }
        
        # General help
        commands = [
            "• create issue for [repo] about [topic] - Create a GitHub issue",
            "• search repositories for [query] - Search for repositories",
            "• create architecture for [project] - Generate system architecture",
            "• show status - Display system status",
            "• list tasks - Show active tasks",
            "• analyze performance - Show performance metrics",
            "• create task [description] - Create a new task",
            "• help [topic] - Show help for a specific topic"
        ]
        
        return {
            'status': 'success',
            'message': 'Available commands',
            'data': {'commands': commands}
        }
        
    async def process_natural_language(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point for processing natural language commands.
        
        Args:
            user_input: The natural language input from the user
            
        Returns:
            Result dictionary with status, message, and optional data
        """
        # First try pattern matching
        command_type, params = self.parse_command(user_input)
        
        # If no match, use AI interpretation
        if not command_type:
            command_type, params = await self.interpret_with_ai(user_input)
            
        # Execute the command
        result = await self.execute_command(command_type, params)
        
        # Update conversation history with result
        self.context['conversation_history'][-1]['result'] = result
        
        return result