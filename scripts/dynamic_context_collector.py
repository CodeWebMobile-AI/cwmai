"""
Dynamic Context Collector for Conversational AI

This module dynamically gathers relevant context based on user queries,
providing the AI with comprehensive system information without hardcoding.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from pathlib import Path

from scripts.state_manager import StateManager
from scripts.task_manager import TaskManager
from scripts.mcp_integration import MCPIntegrationHub
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.http_ai_client import HTTPAIClient


class DynamicContextCollector:
    """Collects dynamic context for AI queries based on query analysis."""
    
    def __init__(self):
        """Initialize the context collector."""
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        self.task_manager = TaskManager()
        self.mcp_hub = MCPIntegrationHub()
        self.repo_analyzer = RepositoryAnalyzer()
        self.ai_client = HTTPAIClient()
        
        # Define context categories and their collection methods
        self.context_collectors = {
            'system_capabilities': self._collect_system_capabilities,
            'system_status': self._collect_system_status,
            'repositories': self._collect_repository_info,
            'tasks': self._collect_task_info,
            'continuous_ai': self._collect_continuous_ai_status,
            'available_commands': self._collect_available_commands,
            'recent_activity': self._collect_recent_activity
        }
        
    async def analyze_query_needs(self, query: str) -> Set[str]:
        """Analyze what context is needed for a given query.
        
        Args:
            query: User's input query
            
        Returns:
            Set of context types needed
        """
        query_lower = query.lower()
        needed_context = set()
        
        # Always include basic capabilities
        needed_context.add('system_capabilities')
        needed_context.add('available_commands')
        
        # Analyze query for specific needs
        context_mappings = {
            'repositories': ['repo', 'repository', 'project', 'code'],
            'system_status': ['status', 'running', 'active', 'system', 'health'],
            'tasks': ['task', 'todo', 'work', 'job', 'issue'],
            'continuous_ai': ['continuous', 'ai system', 'workers', 'orchestrator'],
            'recent_activity': ['recent', 'latest', 'history', 'what happened']
        }
        
        for context_type, keywords in context_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                needed_context.add(context_type)
        
        # Use AI to detect additional context needs
        if len(needed_context) < 3:  # If we haven't identified much context
            ai_analysis = await self._ai_analyze_context_needs(query)
            needed_context.update(ai_analysis)
        
        return needed_context
    
    async def _ai_analyze_context_needs(self, query: str) -> Set[str]:
        """Use AI to analyze what context might be needed."""
        prompt = f"""
        Analyze this user query and determine what system context is needed to answer it properly.
        
        Query: "{query}"
        
        Available context types:
        - system_capabilities: What the system can do
        - system_status: Current running status and health
        - repositories: Repository and project information  
        - tasks: Task queue and active work
        - continuous_ai: Autonomous AI system status
        - recent_activity: Recent system actions
        
        Return a JSON array of needed context types.
        Example: ["system_status", "tasks"]
        """
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt, prefill='[')
            if response.get('content'):
                content = response['content']
                if not content.strip().startswith('['):
                    content = '[' + content
                context_types = json.loads(content)
                return set(context_types) & set(self.context_collectors.keys())
        except:
            pass
        
        return set()
    
    async def gather_context_for_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Gather all relevant context for a given query.
        
        Args:
            query: User's input query
            conversation_history: Recent conversation turns
            
        Returns:
            Dictionary containing all gathered context
        """
        # Determine what context is needed
        needed_context = await self.analyze_query_needs(query)
        self.logger.info(f"Query needs context: {needed_context}")
        
        # Gather context in parallel
        context_tasks = {}
        for context_type in needed_context:
            if context_type in self.context_collectors:
                context_tasks[context_type] = self.context_collectors[context_type]()
        
        # Execute all collectors in parallel
        results = {}
        if context_tasks:
            gathered = await asyncio.gather(*context_tasks.values(), return_exceptions=True)
            for context_type, result in zip(context_tasks.keys(), gathered):
                if isinstance(result, Exception):
                    self.logger.error(f"Error collecting {context_type}: {result}")
                    results[context_type] = None
                else:
                    results[context_type] = result
        
        # Build final context
        context = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query': query,
            'conversation_history': conversation_history or [],
            'gathered_context': results,
            'context_summary': self._summarize_context(results)
        }
        
        return context
    
    def _summarize_context(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the gathered context."""
        summary = {
            'has_repositories': bool(results.get('repositories', {}).get('repositories')),
            'repository_count': len(results.get('repositories', {}).get('repositories', [])),
            'has_active_tasks': bool(results.get('tasks', {}).get('active_tasks')),
            'task_count': len(results.get('tasks', {}).get('active_tasks', [])),
            'continuous_ai_running': results.get('continuous_ai', {}).get('running', False),
            'system_healthy': results.get('system_status', {}).get('healthy', True)
        }
        return summary
    
    async def _collect_system_capabilities(self) -> Dict[str, Any]:
        """Collect information about system capabilities."""
        return {
            'core_features': [
                'GitHub issue creation and management',
                'Repository search and analysis',
                'System architecture generation',
                'Task creation and tracking',
                'Performance monitoring',
                'Continuous AI orchestration',
                'Multi-repository coordination',
                'Smart system resets'
            ],
            'integrations': {
                'github': await self._check_github_integration(),
                'mcp': await self._check_mcp_status(),
                'redis': self._check_redis_availability()
            },
            'ai_providers': self._get_ai_provider_status()
        }
    
    async def _collect_system_status(self) -> Dict[str, Any]:
        """Collect current system status."""
        try:
            # Load current state
            self.state_manager.load_state()
            state = self.state_manager.state or {}
            
            # Check continuous AI
            continuous_ai = await self._collect_continuous_ai_status()
            
            return {
                'healthy': True,
                'uptime': self._calculate_uptime(state),
                'performance_metrics': state.get('system_performance', {}),
                'continuous_ai_active': continuous_ai.get('running', False),
                'last_activity': state.get('last_updated', 'Unknown')
            }
        except Exception as e:
            self.logger.error(f"Error collecting system status: {e}")
            return {'healthy': False, 'error': str(e)}
    
    async def _collect_repository_info(self) -> Dict[str, Any]:
        """Collect repository information."""
        try:
            # Load state to get tracked repositories
            self.state_manager.load_state()
            state = self.state_manager.state or {}
            
            repositories = []
            repo_count = 0
            
            # Get repositories from state
            for repo_name, repo_data in state.get('projects', {}).items():
                repo_count += 1
                repositories.append({
                    'name': repo_name,
                    'description': repo_data.get('description', 'No description'),
                    'language': repo_data.get('language', 'Unknown'),
                    'stars': repo_data.get('stars', 0),
                    'issues': repo_data.get('open_issues_count', 0),
                    'last_updated': repo_data.get('updated_at', 'Unknown')
                })
            
            return {
                'repositories': repositories[:10],  # Limit to 10 for context size
                'total_count': repo_count,
                'has_more': repo_count > 10
            }
        except Exception as e:
            self.logger.error(f"Error collecting repository info: {e}")
            return {'repositories': [], 'error': str(e)}
    
    async def _collect_task_info(self) -> Dict[str, Any]:
        """Collect task queue information."""
        try:
            # Get tasks from task manager
            task_queue = self.task_manager.get_task_queue()
            active_tasks = [task for task in task_queue if task.get('status') == 'active']
            pending_tasks = [task for task in task_queue if task.get('status') == 'pending']
            
            return {
                'active_tasks': active_tasks[:5],  # Limit for context
                'pending_tasks': pending_tasks[:5],
                'total_active': len(active_tasks),
                'total_pending': len(pending_tasks),
                'queue_health': 'healthy' if len(active_tasks) < 10 else 'busy'
            }
        except Exception as e:
            self.logger.error(f"Error collecting task info: {e}")
            return {'active_tasks': [], 'error': str(e)}
    
    async def _collect_continuous_ai_status(self) -> Dict[str, Any]:
        """Collect continuous AI system status."""
        try:
            # Check if continuous orchestrator state exists
            state_file = Path("continuous_orchestrator_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    orchestrator_state = json.load(f)
                    
                return {
                    'running': True,
                    'last_updated': orchestrator_state.get('last_updated', 'Unknown'),
                    'metrics': orchestrator_state.get('metrics', {}),
                    'active_workers': orchestrator_state.get('metrics', {}).get('active_workers', 0),
                    'tasks_completed': orchestrator_state.get('metrics', {}).get('tasks_completed', 0)
                }
            else:
                return {'running': False}
        except Exception as e:
            self.logger.error(f"Error checking continuous AI: {e}")
            return {'running': False, 'error': str(e)}
    
    async def _collect_available_commands(self) -> Dict[str, Any]:
        """Collect available commands and their descriptions."""
        return {
            'commands': {
                'create_issue': {
                    'description': 'Create a GitHub issue',
                    'usage': 'create issue for [repo] about [topic]',
                    'examples': ['create issue for myapp about fixing login bug']
                },
                'search_repos': {
                    'description': 'Search for repositories',
                    'usage': 'search repositories for [query]',
                    'examples': ['search repositories for React']
                },
                'show_status': {
                    'description': 'Show system status',
                    'usage': 'show status',
                    'examples': ['show status', 'what is the system status?']
                },
                'list_tasks': {
                    'description': 'List active tasks',
                    'usage': 'list tasks',
                    'examples': ['list tasks', 'show active tasks']
                },
                'start_continuous_ai': {
                    'description': 'Start the continuous AI system',
                    'usage': 'start continuous AI',
                    'examples': ['start the continuous AI system']
                },
                'reset_system': {
                    'description': 'Reset the system',
                    'usage': 'reset system',
                    'examples': ['reset the system', 'clear all logs']
                }
            }
        }
    
    async def _collect_recent_activity(self) -> Dict[str, Any]:
        """Collect recent system activity."""
        try:
            # Get from conversation history if available
            recent_actions = []
            
            # Check task history
            task_history_file = Path("task_history.json")
            if task_history_file.exists():
                with open(task_history_file, 'r') as f:
                    history = json.load(f)
                    recent_actions = history.get('recent_tasks', [])[:5]
            
            return {
                'recent_actions': recent_actions,
                'last_action_time': recent_actions[0].get('timestamp') if recent_actions else None
            }
        except Exception as e:
            self.logger.error(f"Error collecting recent activity: {e}")
            return {'recent_actions': []}
    
    # Helper methods
    async def _check_github_integration(self) -> Dict[str, Any]:
        """Check GitHub integration status."""
        try:
            await self.mcp_hub.initialize()
            return {
                'available': bool(self.mcp_hub.github),
                'authenticated': True if self.mcp_hub.github else False
            }
        except:
            return {'available': False, 'authenticated': False}
    
    async def _check_mcp_status(self) -> Dict[str, Any]:
        """Check MCP integration status."""
        return {
            'available': hasattr(self.mcp_hub, 'is_initialized') and self.mcp_hub.is_initialized,
            'servers': len(self.mcp_hub.clients) if hasattr(self.mcp_hub, 'clients') else 0
        }
    
    def _check_redis_availability(self) -> Dict[str, Any]:
        """Check Redis availability."""
        try:
            import redis
            return {'available': True, 'type': 'redis'}
        except ImportError:
            return {'available': False, 'type': 'none'}
    
    def _get_ai_provider_status(self) -> Dict[str, Any]:
        """Get AI provider status."""
        providers = self.ai_client.providers_available
        return {
            'available_providers': [p for p, available in providers.items() if available],
            'primary_provider': 'anthropic' if providers.get('anthropic') else 'auto',
            'total_providers': sum(providers.values())
        }
    
    def _calculate_uptime(self, state: Dict[str, Any]) -> str:
        """Calculate system uptime from state."""
        try:
            if 'created_at' in state:
                created = datetime.fromisoformat(state['created_at'].replace('Z', '+00:00'))
                uptime = datetime.now(timezone.utc) - created
                days = uptime.days
                hours = uptime.seconds // 3600
                return f"{days}d {hours}h"
        except:
            pass
        return "Unknown"
    
    def format_context_for_ai(self, context: Dict[str, Any]) -> str:
        """Format gathered context into a string suitable for AI prompt."""
        formatted_parts = []
        gathered = context.get('gathered_context', {})
        
        # System capabilities
        if 'system_capabilities' in gathered and gathered['system_capabilities']:
            caps = gathered['system_capabilities']
            formatted_parts.append("SYSTEM CAPABILITIES:")
            if 'core_features' in caps:
                formatted_parts.append(f"- Core features: {', '.join(caps['core_features'][:5])}")
            if 'ai_providers' in caps and caps['ai_providers']:
                providers = caps['ai_providers'].get('available_providers', [])
                formatted_parts.append(f"- AI providers: {', '.join(providers) if providers else 'None available'}")
        
        # System status
        if 'system_status' in gathered and gathered['system_status']:
            status = gathered['system_status']
            formatted_parts.append("\nSYSTEM STATUS:")
            formatted_parts.append(f"- Healthy: {status.get('healthy', 'Unknown')}")
            formatted_parts.append(f"- Continuous AI: {'Running' if status.get('continuous_ai_active') else 'Not running'}")
        
        # Repositories
        if 'repositories' in gathered and gathered['repositories']:
            repos = gathered['repositories']
            formatted_parts.append(f"\nREPOSITORIES: {repos.get('total_count', 0)} total")
            for repo in repos.get('repositories', [])[:3]:
                desc = repo.get('description', 'No description')
                if desc and len(desc) > 50:
                    desc = desc[:50] + "..."
                formatted_parts.append(f"- {repo['name']}: {desc}")
        
        # Tasks
        if 'tasks' in gathered and gathered['tasks']:
            tasks = gathered['tasks']
            formatted_parts.append(f"\nTASKS: {tasks.get('total_active', 0)} active, {tasks.get('total_pending', 0)} pending")
        
        # Available commands
        if 'available_commands' in gathered and gathered['available_commands']:
            commands = gathered['available_commands'].get('commands', {})
            if commands:
                formatted_parts.append("\nAVAILABLE COMMANDS:")
                for cmd, info in list(commands.items())[:5]:
                    formatted_parts.append(f"- {cmd}: {info['description']}")
        
        # If no context was gathered, provide minimal info
        if not formatted_parts:
            formatted_parts.append("LIMITED CONTEXT AVAILABLE")
            formatted_parts.append("- System is operational")
            formatted_parts.append("- Basic commands available")
        
        return "\n".join(formatted_parts)